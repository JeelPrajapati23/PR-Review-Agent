"""optional LLM-as-a-Judge pass over already-generated reviews.

evaluate_results.py's keyword-match scoring only measures recall of a fixed
vocabulary list, and can't detect two things that actually matter for
judging review quality: a correct finding phrased in different words than
meta.json's issue_keywords, and false positives -- issues the panel invented
that were never actually in the code. This script fixes both by having a
second LLM (Gemini, not Groq) semantically grade each stored review against
its fixture's ground truth.

Deliberately decoupled from run_reviews.py and from app/config.py:
  * It never imports app.agent or instantiates app.config.Settings, so it
    does not require GROQ_API_KEY/GITHUB_API_TOKEN/GITHUB_WEBHOOK_SECRET to
    be set -- only GEMINI_API_KEY, resolved from the environment or .env
    directly (see _resolve_gemini_config below).
  * It only reads evaluation/results/<id>.json files that run_reviews.py
    already wrote; it never calls the Groq-backed review panel itself, so
    running this script spends Gemini tokens, never Groq tokens.

Like run_reviews.py, each fixture's verdict is written to disk immediately
(evaluation/results/<id>.judge.json, atomic write) as soon as it's graded,
and an already-graded fixture is skipped on the next run (unless --force),
so a Gemini-side rate limit or a Ctrl-C partway through loses nothing.

Usage:
    python evaluation/judge_results.py              # resume: grades whatever's left
    python evaluation/judge_results.py --force       # re-grades everything
    python evaluation/judge_results.py --only sec-01-sql-injection
"""
import argparse
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field

DATASET_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = DATASET_ROOT / "manifest.json"

_DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"

# F-beta weighting for the headline score: beta=2 weights recall 4x as
# heavily as precision, i.e. missing a real vulnerability is penalized much
# harder than an extra false positive -- the right bias for a security/
# correctness review panel, where a false negative is the costlier mistake.
_F_BETA = 2.0

# Gemini's free tier enforces a per-minute request cap; spacing requests at
# least this many seconds apart keeps this script under ~10 req/min without
# needing to parse rate-limit headers or guess at a backoff schedule.
_GEMINI_MIN_REQUEST_INTERVAL_SECONDS = 6.0

# 5xx ("model currently experiencing high demand") errors are transient
# server-side capacity issues, not a reason to give up on a fixture --
# observed hitting the majority of calls in a single run. A few retries with
# linear backoff clears most of them without needing a whole extra pass of
# this script.
_GEMINI_SERVER_ERROR_MAX_RETRIES = 3
_GEMINI_SERVER_ERROR_BACKOFF_SECONDS = 8.0


class JudgeVerdict(BaseModel):
    """Structured grade for one fixture's review, against its ground truth."""

    true_positive_caught: bool = Field(
        description="True only if the review substantively identifies the specific ground-truth "
        "issue described -- a correct diagnosis, not merely a comment that happens to touch the "
        "same file. A correct diagnosis phrased differently than the reference keywords still "
        "counts: judge substance, not vocabulary."
    )
    section_correct: bool = Field(
        description="True if the caught finding was routed into the review section appropriate for "
        "its category (security issues under 'Security & Correctness Issues'; algorithmic/logic "
        "issues under 'Performance Observations', 'Edge Cases', or 'Architectural Suggestions' as "
        "fits). False if not caught at all, or if caught but filed under a clearly wrong section."
    )
    severity_matched: bool = Field(
        description="True if the review's treatment of the issue (when caught) is proportionate to "
        "its stated ground-truth severity -- e.g. a 'high' severity issue reads as a real, "
        "actionable concern rather than a footnote. False if not caught, or if severity is "
        "clearly understated relative to the ground truth."
    )
    false_positive_count: int = Field(
        ge=0,
        description="Count of OTHER findings in the review, unrelated to the ground-truth issue, "
        "that are incorrect, fabricated, or describe problems that do not actually exist in the "
        "code. 0 if the review raises no such spurious findings.",
    )
    reasoning: str = Field(description="1-3 sentence justification for the scores above, citing specifics from the review text.")


_JUDGE_SYSTEM_INSTRUCTION = """You are an impartial grading judge for an AI code-review panel's output.

You will be given two things: (1) ground truth describing one specific issue that was deliberately \
planted in a small PR fixture -- its file, its category, its severity, and a precise description of \
what's wrong -- and (2) the full synthesized review text an AI review panel produced for that same PR.

Grade the review AGAINST THE GROUND TRUTH ONLY. Do not reward the review for being well-written or \
thorough in general -- reward it only for correctly identifying and explaining the SPECIFIC planted \
issue, and penalize it for inventing problems that are not actually present.

Respond only with the structured fields requested. No extra commentary outside those fields."""


def _build_user_prompt(entry: dict, meta: dict, review_text: str) -> str:
    expected = meta["expected_findings"]
    return (
        "GROUND TRUTH\n"
        f"Category: {entry['category']}\n"
        f"Target specialist: {entry['target_specialist']}\n"
        f"Target file: {expected['must_flag_file']}\n"
        f"Expected severity: {expected['severity']}\n"
        f"Injected issue: {meta['injected_issue']}\n"
        f"Reference keywords (context only, not required verbatim): {', '.join(expected['issue_keywords'])}\n\n"
        "REVIEW UNDER EVALUATION\n"
        f"{review_text}"
    )


def _resolve_gemini_config() -> tuple[str | None, str]:
    """Resolve (api_key, model) from the environment, falling back to a
    direct .env read -- same source-order convention pydantic-settings uses
    (real env vars win, .env is the fallback) without pulling in
    app.config.Settings, which would otherwise force this Gemini-only
    script to also provide unrelated GROQ_/GITHUB_ secrets it never uses.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("GEMINI_MODEL") or _DEFAULT_GEMINI_MODEL

    env_path = DATASET_ROOT.parent / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not value or value == "changeme":
                continue
            if key == "GEMINI_API_KEY" and not api_key:
                api_key = value
            elif key == "GEMINI_MODEL" and not os.environ.get("GEMINI_MODEL"):
                model = value

    return api_key, model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Re-grade fixtures that already have a stored verdict")
    parser.add_argument("--only", default=None, help="Only grade the fixture with this id")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory (relative to evaluation/) to read reviews from and write judge verdicts/summary into",
    )
    return parser.parse_args()


def _write_json_atomically(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _grade_fixture(client: genai.Client, model: str, entry: dict, meta: dict, review_text: str) -> JudgeVerdict:
    response = client.models.generate_content(
        model=model,
        contents=_build_user_prompt(entry, meta, review_text),
        config=types.GenerateContentConfig(
            system_instruction=_JUDGE_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=JudgeVerdict,
            temperature=0.0,
        ),
    )
    if response.parsed is not None:
        return response.parsed
    # Fallback if the SDK couldn't auto-parse for any reason -- validate the
    # raw JSON text ourselves rather than silently losing the verdict.
    return JudgeVerdict.model_validate_json(response.text)


def _grade_fixture_with_retries(
    client: genai.Client, model: str, entry: dict, meta: dict, review_text: str, fixture_id: str, progress: str
) -> JudgeVerdict:
    for attempt in range(_GEMINI_SERVER_ERROR_MAX_RETRIES + 1):
        try:
            return _grade_fixture(client, model, entry, meta, review_text)
        except genai_errors.ServerError as exc:
            if attempt >= _GEMINI_SERVER_ERROR_MAX_RETRIES:
                raise
            backoff = _GEMINI_SERVER_ERROR_BACKOFF_SECONDS * (attempt + 1)
            print(
                f"{progress} {fixture_id}: transient Gemini {exc.code} ('high demand') -- "
                f"retrying in {backoff:.0f}s (attempt {attempt + 1}/{_GEMINI_SERVER_ERROR_MAX_RETRIES})"
            )
            time.sleep(backoff)
    raise AssertionError("unreachable")


def _compute_metrics(tp: int, fn: int, fp: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta_sq = _F_BETA**2
    denom = beta_sq * precision + recall
    f_beta = (1 + beta_sq) * precision * recall / denom if denom > 0 else 0.0
    return precision, recall, f_beta


def _render_dashboard(rows: list[dict]) -> str:
    categories = sorted({row["category"] for row in rows})
    lines = []
    width = 88
    lines.append("=" * width)
    lines.append("LLM-AS-JUDGE EVALUATION DASHBOARD".center(width))
    lines.append(f"(F{_F_BETA:g} score -- recall weighted {_F_BETA**2:g}x over precision)".center(width))
    lines.append("=" * width)
    header = f"{'Category':<14}{'Graded':>8}{'TP':>5}{'FN':>5}{'FP':>5}{'Precision':>11}{'Recall':>9}{'F2':>8}{'Section%':>10}{'Sev%':>7}"
    lines.append(header)
    lines.append("-" * width)

    overall_tp = overall_fn = overall_fp = 0
    overall_section = overall_severity = overall_n = 0

    for category in categories:
        cat_rows = [r for r in rows if r["category"] == category and r["verdict"] is not None]
        total_in_category = sum(1 for r in rows if r["category"] == category)
        tp = sum(1 for r in cat_rows if r["verdict"]["true_positive_caught"])
        fn = sum(1 for r in cat_rows if not r["verdict"]["true_positive_caught"])
        fp = sum(r["verdict"]["false_positive_count"] for r in cat_rows)
        section_ok = sum(1 for r in cat_rows if r["verdict"]["section_correct"])
        severity_ok = sum(1 for r in cat_rows if r["verdict"]["severity_matched"])
        n = len(cat_rows)

        precision, recall, f_beta = _compute_metrics(tp, fn, fp)
        section_pct = (section_ok / n * 100) if n else 0.0
        severity_pct = (severity_ok / n * 100) if n else 0.0

        lines.append(
            f"{category:<14}{f'{n}/{total_in_category}':>8}{tp:>5}{fn:>5}{fp:>5}"
            f"{precision:>11.3f}{recall:>9.3f}{f_beta:>8.3f}{section_pct:>9.1f}%{severity_pct:>6.1f}%"
        )

        overall_tp += tp
        overall_fn += fn
        overall_fp += fp
        overall_section += section_ok
        overall_severity += severity_ok
        overall_n += n

    lines.append("-" * width)
    total_fixtures = len(rows)
    precision, recall, f_beta = _compute_metrics(overall_tp, overall_fn, overall_fp)
    section_pct = (overall_section / overall_n * 100) if overall_n else 0.0
    severity_pct = (overall_severity / overall_n * 100) if overall_n else 0.0
    lines.append(
        f"{'OVERALL':<14}{f'{overall_n}/{total_fixtures}':>8}{overall_tp:>5}{overall_fn:>5}{overall_fp:>5}"
        f"{precision:>11.3f}{recall:>9.3f}{f_beta:>8.3f}{section_pct:>9.1f}%{severity_pct:>6.1f}%"
    )
    lines.append("=" * width)

    ungraded = [r["id"] for r in rows if r["verdict"] is None]
    if ungraded:
        lines.append(f"Excluded from metrics (not reviewed/completed or not yet graded): {', '.join(ungraded)}")

    return "\n".join(lines)


def _render_markdown(rows: list[dict]) -> str:
    lines = ["# LLM-as-judge evaluation results\n", f"F{_F_BETA:g} score weights recall {_F_BETA**2:g}x over precision.\n"]
    lines.append("| id | category | caught | section_correct | severity_matched | false_positives |")
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        verdict = row["verdict"]
        if verdict is None:
            lines.append(f"| {row['id']} | {row['category']} | - | - | - | - |")
            continue
        lines.append(
            f"| {row['id']} | {row['category']} | {verdict['true_positive_caught']} | "
            f"{verdict['section_correct']} | {verdict['severity_matched']} | {verdict['false_positive_count']} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    results_dir = DATASET_ROOT / args.results_dir
    judge_summary_json_path = results_dir / "_judge_summary.json"
    judge_summary_md_path = results_dir / "_judge_summary.md"

    api_key, model = _resolve_gemini_config()
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY is not set (checked the environment and .env). This script grades "
            "already-generated reviews with Gemini and needs its own key -- add GEMINI_API_KEY to "
            ".env (see .env.example) or export it, then re-run."
        )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    if args.only:
        entries = [e for e in entries if e["id"] == args.only]
        if not entries:
            raise SystemExit(f"No manifest entry with id '{args.only}'")

    client = genai.Client(api_key=api_key)
    total = len(entries)
    graded_this_run = 0
    last_request_at: float | None = None

    for index, entry in enumerate(entries, start=1):
        fixture_id = entry["id"]
        review_path = results_dir / f"{fixture_id}.json"
        judge_path = results_dir / f"{fixture_id}.judge.json"
        fixture_dir = DATASET_ROOT / "golden_dataset" / entry["path"]
        meta = json.loads((fixture_dir / "meta.json").read_text(encoding="utf-8"))

        if not review_path.exists():
            print(f"[{index}/{total}] {fixture_id}: no stored review yet, skipping (run run_reviews.py first)")
            continue

        review_record = json.loads(review_path.read_text(encoding="utf-8"))
        agent_result = review_record.get("agent_result", {})
        if agent_result.get("status") != "completed":
            print(f"[{index}/{total}] {fixture_id}: review status='{agent_result.get('status')}', not gradable, skipping")
            continue

        if judge_path.exists() and not args.force:
            existing_verdict = json.loads(judge_path.read_text(encoding="utf-8")).get("verdict")
            if existing_verdict is None:
                # A prior attempt failed (rate limit, auth error, transient
                # API error, etc.) and never actually produced a verdict --
                # treating this file as "already graded" would silently skip
                # it forever. Fall through and retry it for real.
                print(f"[{index}/{total}] {fixture_id}: previous grading attempt failed, retrying")
            else:
                print(f"[{index}/{total}] {fixture_id}: already graded, skipping (use --force to redo)")
                continue

        if last_request_at is not None:
            wait_for = _GEMINI_MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - last_request_at)
            if wait_for > 0:
                print(f"[{index}/{total}] {fixture_id}: throttling {wait_for:.1f}s to respect Gemini's per-minute limit")
                time.sleep(wait_for)

        print(f"[{index}/{total}] {fixture_id}: grading with {model}...")
        started_iso = datetime.now(timezone.utc).isoformat()
        started_at = time.monotonic()
        last_request_at = started_at
        try:
            verdict = _grade_fixture_with_retries(
                client, model, entry, meta, agent_result["summary"], fixture_id, f"[{index}/{total}]"
            )
        except genai_errors.APIError as exc:
            if exc.code == 429:
                print(f"[{index}/{total}] {fixture_id}: Gemini rate limit hit -- stopping run, nothing lost")
                break
            if exc.code in (400, 401, 403):
                # An auth/permission failure (bad or revoked GEMINI_API_KEY,
                # no access to this model, etc.) is not fixture-specific --
                # every remaining fixture would fail identically, so grinding
                # through the rest just burns the throttle delay for nothing.
                # Don't even write a result for this fixture: an error record
                # here would (before the resume-skip fix above) or could
                # again in the future look like "already attempted."
                print(
                    f"[{index}/{total}] {fixture_id}: Gemini auth/permission error "
                    f"({exc.code}: {exc.message}) -- stopping run, fix GEMINI_API_KEY and retry"
                )
                break
            print(f"[{index}/{total}] {fixture_id}: Gemini API error ({exc.code}: {exc.message}) -- storing error, continuing")
            _write_json_atomically(
                judge_path,
                {
                    "id": fixture_id,
                    "category": entry["category"],
                    "judged_at_utc": started_iso,
                    "wall_clock_seconds": round(time.monotonic() - started_at, 1),
                    "judge_model": model,
                    "error": f"APIError {exc.code}: {exc.message}",
                    "verdict": None,
                },
            )
            continue
        except Exception as exc:
            print(f"[{index}/{total}] {fixture_id}: grading failed ({type(exc).__name__}: {exc}) -- storing error, continuing")
            _write_json_atomically(
                judge_path,
                {
                    "id": fixture_id,
                    "category": entry["category"],
                    "judged_at_utc": started_iso,
                    "wall_clock_seconds": round(time.monotonic() - started_at, 1),
                    "judge_model": model,
                    "error": f"{type(exc).__name__}: {exc}",
                    "verdict": None,
                },
            )
            continue

        record = {
            "id": fixture_id,
            "category": entry["category"],
            "judged_at_utc": started_iso,
            "wall_clock_seconds": round(time.monotonic() - started_at, 1),
            "judge_model": model,
            "error": None,
            "verdict": verdict.model_dump(),
        }
        _write_json_atomically(judge_path, record)
        graded_this_run += 1
        print(
            f"[{index}/{total}] {fixture_id}: caught={verdict.true_positive_caught} "
            f"section_correct={verdict.section_correct} severity_matched={verdict.severity_matched} "
            f"false_positives={verdict.false_positive_count} -- stored"
        )

    rows = []
    for entry in entries:
        judge_path = results_dir / f"{entry['id']}.judge.json"
        verdict = None
        if judge_path.exists():
            record = json.loads(judge_path.read_text(encoding="utf-8"))
            verdict = record.get("verdict")
        rows.append({"id": entry["id"], "category": entry["category"], "verdict": verdict})

    dashboard = _render_dashboard(rows)
    print(f"\n{dashboard}\n")
    print(f"{graded_this_run} fixture(s) graded this run.")

    results_dir.mkdir(parents=True, exist_ok=True)
    judge_summary_json_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    judge_summary_md_path.write_text(_render_markdown(rows), encoding="utf-8")
    print(f"Written to {judge_summary_json_path} and {judge_summary_md_path}")


if __name__ == "__main__":
    main()
