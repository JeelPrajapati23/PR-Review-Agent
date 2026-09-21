"""Run the golden dataset (see manifest.json) through the real review panel,
one PR fixture at a time, writing each result to disk as soon as it completes.

Groq's free-tier daily token budget (see app/telemetry.py) is far too tight
to review all 15 fixtures in one sitting. This script is built around that
constraint rather than around finishing in one run:

  * Before each fixture, it checks app.telemetry.check_budget_ok() -- the
    exact same gate app/tasks.py uses before a real review. If the budget is
    exhausted, it stops immediately instead of burning a Groq call that would
    likely just fail, and prints how many fixtures are left.
  * Each fixture's result is written to evaluation/results/<id>.json right
    after that fixture finishes (atomic write via a temp file + os.replace),
    not batched at the end -- so a crash, Ctrl-C, or a hard Groq rate-limit
    error after fixture 9 of 15 does not lose fixtures 1-9's results.
  * Already-reviewed fixtures (a result file already exists) are skipped on
    the next run, so simply re-running this script after the daily quota
    resets picks up where it left off. Use --force to re-review a fixture
    that already has a stored result.

Usage:
    python evaluation/run_reviews.py              # resume: reviews whatever's left
    python evaluation/run_reviews.py --force       # re-reviews everything
    python evaluation/run_reviews.py --only sec-01-sql-injection

Requires a reachable Redis (checkpointer + telemetry) and a real
GROQ_API_KEY / GITHUB_API_TOKEN / GITHUB_WEBHOOK_SECRET (via .env, same as
running the app for real) -- this makes genuine Groq calls and spends real
tokens, exactly like tests/test_integration.py's single fixture.

Optional key rotation: set GROQ_API_KEYS (comma-separated, in .env or the
environment) to more than one Groq account's key -- Groq's real per-org
daily quota is per-*org*, not per-key, so these need to be
separate accounts to actually buy more headroom, not just extra keys on the
same one. When check_budget_ok() reports exhausted, or a real Groq
RateLimitError/APIConnectionError/APITimeoutError is hit mid-review, this
script resets today's Redis usage counters (see
app.telemetry.reset_daily_usage) and rotates to the next key automatically,
retrying the same fixture rather than stopping the whole run -- only once
every key has been tried does it fall back to the original stop-and-resume
behavior. GROQ_API_KEY (singular) still works exactly as before if
GROQ_API_KEYS is left unset.

IMPORTANT: this rotation resets the *shared* Redis usage counters that
app/tasks.py's live webhook pipeline also reads via check_budget_ok. Only
run this against a Redis genuinely dedicated to local/eval use -- never
point REDIS_URL at the same Redis a production deployment uses, or this
will reset its real daily budget out from under it.
"""
import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Run as `python evaluation/run_reviews.py`, sys.path[0] is evaluation/ (the
# script's own directory), not the repo root -- unlike simulate_pr.py, which
# gets `from app...` for free by living at the repo root. Insert the repo
# root explicitly so app.agent/app.config/app.telemetry resolve regardless
# of where this script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("GITHUB_WEBHOOK_SECRET", "eval-secret")

import redis
from groq import APIConnectionError, APITimeoutError, RateLimitError

from app.agent import run_pr_review_agent
from app.config import get_settings
from app.telemetry import _get_sync_redis_client, check_budget_ok, reset_daily_usage

DATASET_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = DATASET_ROOT / "manifest.json"

# Fake repo identity used for every fixture -- same pattern as
# tests/test_integration.py. GitHub notification calls (commit status, PR
# review) will fail against this nonexistent repo/PR; app/agent.py's
# _notify/_post_review already catch GitHubNotifyError and log-and-continue,
# so this is harmless noise, not a failure.
_FAKE_REPO_FULL_NAME = "octocat/pr-review-eval-fixture"
_FAKE_REPO_CLONE_URL = "https://github.com/octocat/pr-review-eval-fixture.git"


class _KeyRotator:
    """Cycles forward through a fixed list of API keys within one process run.

    Not persisted across separate invocations -- each fresh run starts back
    at the first key. That's deliberate: whether key N was exhausted in a
    prior run says nothing about whether it's exhausted now (rate limits
    refill on their own schedule), so remembering "last used index" across
    runs would as often skip a key that's fine again as it would save time.
    """

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("_KeyRotator needs at least one key")
        self._keys = keys
        self._index = 0

    @property
    def current(self) -> str:
        return self._keys[self._index]

    @property
    def current_number(self) -> int:
        """1-indexed, for human-readable log lines."""
        return self._index + 1

    @property
    def total(self) -> int:
        return len(self._keys)

    def advance(self) -> bool:
        """Move to the next key. Returns False (state unchanged) if this was
        already the last one -- the caller decides what "nothing left to
        rotate to" means for its own run.
        """
        if self._index + 1 >= len(self._keys):
            return False
        self._index += 1
        return True


def _read_env_value(env_path: Path, var_name: str) -> str | None:
    """Manually read var_name out of a .env file. Needed because
    GROQ_API_KEYS isn't a declared app.config.Settings field, so
    pydantic-settings' own .env parsing never sees it -- same reasoning
    evaluation/judge_results.py's _resolve_gemini_config already documents
    for GEMINI_API_KEY.
    """
    if not env_path.is_file():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key == var_name and value and value != "changeme":
            return value
    return None


def _parse_key_list(raw: str | None) -> list[str]:
    """Split a comma-separated key list, stripping whitespace and dropping
    empties (a trailing comma shouldn't produce a bogus empty-string "key").
    """
    if not raw:
        return []
    return [key.strip() for key in raw.split(",") if key.strip()]


def _resolve_groq_key_list() -> list[str]:
    """Resolve the rotation list from GROQ_API_KEYS (comma-separated), env
    var first then .env -- same source-order convention as everywhere else
    in this codebase. Empty if GROQ_API_KEYS isn't set anywhere; the caller
    falls back to the single required GROQ_API_KEY in that case, so a setup
    with no rotation configured behaves exactly as before.

    Deliberately a separate var from GROQ_API_KEY (singular): that one is a
    required app.config.Settings field the live production pipeline also
    reads, so it must always hold exactly one valid key, never a joined
    list.
    """
    raw = os.environ.get("GROQ_API_KEYS")
    if not raw:
        raw = _read_env_value(DATASET_ROOT.parent / ".env", "GROQ_API_KEYS")
    return _parse_key_list(raw)


def _activate_groq_key(key: str) -> None:
    """Point every subsequent get_settings() call -- including the ones
    app.agent makes internally on each specialist/synthesizer call -- at a
    different Groq key.

    get_settings is @lru_cache'd for the process lifetime, so switching keys
    mid-process requires clearing that cache after changing
    the env var; otherwise every later call keeps returning the Settings
    object built from whichever key was active the first time get_settings()
    was ever called.
    """
    os.environ["GROQ_API_KEY"] = key
    get_settings.cache_clear()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Re-review fixtures that already have a stored result")
    parser.add_argument("--only", default=None, help="Only run the fixture with this id")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory (relative to evaluation/) to write results into -- e.g. 'results_v2' to run a "
        "fresh pass without touching the existing 'results' directory",
    )
    return parser.parse_args()


def _redis_reachable(url: str) -> bool:
    try:
        client = redis.Redis.from_url(url, socket_connect_timeout=2)
        client.ping()
        client.close()
        return True
    except Exception:
        return False


def _write_result_atomically(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _build_pr_metadata(entry: dict, meta: dict, pr_number: int, run_tag: str, attempt: int = 1) -> dict:
    # The fake sha folds in run_tag (derived from --results-dir) rather than
    # just fixture_id -- app/agent.py checkpoints panel/specialist state in
    # Redis keyed off this sha, and that checkpointer has no TTL. A sha fixed
    # to fixture_id alone would make every future re-run of this script
    # resume and keep growing the same Redis thread as every prior run
    # instead of starting fresh, silently ballooning prompt tokens each time.
    # Tying it to --results-dir keeps retries of a genuinely deferred fixture
    # within the same run resuming correctly, while a different
    # --results-dir gets its own thread lineage automatically.
    #
    # attempt only changes the sha when > 1 -- i.e. only for a same-run key
    # rotation retry (see main()). Without an attempt suffix, every rotation
    # retry for the same fixture within one run would resume the same
    # checkpoint thread as the attempt before it, growing the conversation
    # (and thus the odds of hitting the same TPM ceiling again) with each
    # retry instead of resetting. attempt=1 (the default) reproduces the
    # exact prior sha so cross-invocation deferred-retry resume is unaffected.
    fixture_id = entry["id"]
    sha = f"eval-{run_tag}-{fixture_id}" if attempt == 1 else f"eval-{run_tag}-{fixture_id}-attempt{attempt}"
    return {
        "action": "opened",
        "repository": {
            "id": 1,
            "name": "pr-review-eval-fixture",
            "full_name": _FAKE_REPO_FULL_NAME,
            "clone_url": _FAKE_REPO_CLONE_URL,
        },
        "pull_request": {
            "number": pr_number,
            "title": meta["pr_title"],
            "draft": False,
            "head": {"ref": f"eval/{fixture_id}", "sha": sha},
            "modified_files": meta.get("modified_files", []),
            "added_files": meta.get("added_files", []),
        },
    }


def _today_usage_tokens() -> int:
    today = datetime.now(timezone.utc).date().isoformat()
    client = _get_sync_redis_client()
    prompt_tokens, completion_tokens = client.mget(
        f"usage:groq:prompt_tokens:{today}", f"usage:groq:completion_tokens:{today}"
    )
    return int(prompt_tokens or 0) + int(completion_tokens or 0)


def _is_completed(results_dir: Path, fixture_id: str) -> bool:
    path = results_dir / f"{fixture_id}.json"
    if not path.exists():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("agent_result", {}).get("status") == "completed"


def main() -> None:
    args = _parse_args()

    groq_keys = _resolve_groq_key_list()
    if groq_keys and not os.environ.get("GROQ_API_KEY"):
        # Settings.groq_api_key has no default, so get_settings() below would
        # otherwise raise before we ever get a chance to activate a rotation
        # key -- seed it with the first one so construction always succeeds,
        # whether or not GROQ_API_KEY (singular) was also set.
        os.environ["GROQ_API_KEY"] = groq_keys[0]

    settings = get_settings()
    rotator = _KeyRotator(groq_keys or [settings.groq_api_key])
    _activate_groq_key(rotator.current)
    if rotator.total > 1:
        print(f"Groq key rotation active: {rotator.total} keys available (starting on key 1)")

    results_dir = DATASET_ROOT / args.results_dir
    run_tag = re.sub(r"[^a-zA-Z0-9_-]", "-", args.results_dir)

    if not _redis_reachable(settings.redis_url):
        raise SystemExit(f"Redis not reachable at {settings.redis_url} -- start it before running this script.")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    if args.only:
        entries = [e for e in entries if e["id"] == args.only]
        if not entries:
            raise SystemExit(f"No manifest entry with id '{args.only}'")

    total = len(entries)
    reviewed_this_run = 0
    skipped = 0

    for index, entry in enumerate(entries, start=1):
        fixture_id = entry["id"]
        result_path = results_dir / f"{fixture_id}.json"
        fixture_dir = DATASET_ROOT / "golden_dataset" / entry["path"]
        meta = json.loads((fixture_dir / "meta.json").read_text(encoding="utf-8"))

        if result_path.exists() and not args.force:
            existing_status = json.loads(result_path.read_text(encoding="utf-8")).get("agent_result", {}).get("status")
            if existing_status == "deferred":
                # A prior run stopped here on a real Groq rate-limit/quota
                # error -- this fixture was never actually reviewed, so
                # treating its result file as "already done" would silently
                # skip it forever. Fall through and retry it for real.
                print(f"[{index}/{total}] {fixture_id}: previously deferred, retrying")
            else:
                print(f"[{index}/{total}] {fixture_id}: already reviewed, skipping (use --force to redo)")
                skipped += 1
                continue

        stop_run = False
        attempt = 1
        while True:
            if not check_budget_ok(settings.groq_model):
                used = _today_usage_tokens()
                if rotator.advance():
                    attempt += 1
                    print(
                        f"[{index}/{total}] {fixture_id}: budget exhausted on key {rotator.current_number - 1}/"
                        f"{rotator.total} ({used} tokens used) -- rotating to key {rotator.current_number}/"
                        f"{rotator.total} (attempt {attempt}, fresh checkpoint thread) and resetting today's "
                        f"usage counter"
                    )
                    reset_daily_usage()
                    _activate_groq_key(rotator.current)
                    continue
                remaining_ids = [e["id"] for e in entries[index - 1 :] if not _is_completed(results_dir, e["id"])]
                extra = f" across all {rotator.total} rotated keys" if rotator.total > 1 else ""
                print(
                    f"\nStopping: today's Groq token budget is exhausted{extra} ({used} tokens used). "
                    f"{len(remaining_ids)} fixture(s) left ({', '.join(remaining_ids)}). "
                    f"Re-run this script after the daily quota resets to continue -- already-stored "
                    f"results in {results_dir} are untouched."
                )
                stop_run = True
                break

            key_suffix = f" (key {rotator.current_number}/{rotator.total}, attempt {attempt})" if rotator.total > 1 else ""
            print(f"[{index}/{total}] {fixture_id}: reviewing{key_suffix}...")
            pr_number = 900_000 + index
            pr_metadata = _build_pr_metadata(entry, meta, pr_number, run_tag, attempt=attempt)

            started_at = time.monotonic()
            started_iso = datetime.now(timezone.utc).isoformat()
            try:
                agent_result = asyncio.run(run_pr_review_agent(pr_metadata, fixture_dir))
            except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
                # A real Groq rate-limit/connection failure mid-review -- try
                # the next rotation key (if any) on the same fixture before
                # giving up on the whole run.
                if rotator.advance():
                    attempt += 1
                    print(
                        f"[{index}/{total}] {fixture_id}: {type(exc).__name__} on key {rotator.current_number - 1}/"
                        f"{rotator.total} -- rotating to key {rotator.current_number}/{rotator.total} "
                        f"(attempt {attempt}, fresh checkpoint thread) and retrying"
                    )
                    reset_daily_usage()
                    _activate_groq_key(rotator.current)
                    continue
                record = {
                    "id": fixture_id,
                    "category": entry["category"],
                    "target_specialist": entry["target_specialist"],
                    "expected_findings": meta["expected_findings"],
                    "reviewed_at_utc": started_iso,
                    "wall_clock_seconds": round(time.monotonic() - started_at, 1),
                    "agent_result": {"status": "deferred", "error": f"{type(exc).__name__}: {exc}"},
                }
                _write_result_atomically(result_path, record)
                print(
                    f"[{index}/{total}] {fixture_id}: deferred ({type(exc).__name__}) -- all {rotator.total} "
                    f"key(s) exhausted, stored, stopping run"
                )
                stop_run = True
                break
            except Exception as exc:
                # Anything else is fixture-specific (bad fixture setup, a
                # genuine bug), not quota exhaustion -- store it and move on
                # so one bad fixture doesn't stall the others. Not a rotation
                # case: a different key wouldn't fix a bug in the fixture.
                record = {
                    "id": fixture_id,
                    "category": entry["category"],
                    "target_specialist": entry["target_specialist"],
                    "expected_findings": meta["expected_findings"],
                    "reviewed_at_utc": started_iso,
                    "wall_clock_seconds": round(time.monotonic() - started_at, 1),
                    "agent_result": {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
                }
                _write_result_atomically(result_path, record)
                print(f"[{index}/{total}] {fixture_id}: ERROR ({type(exc).__name__}: {exc}) -- stored, continuing")
                reviewed_this_run += 1
                break

            record = {
                "id": fixture_id,
                "category": entry["category"],
                "target_specialist": entry["target_specialist"],
                "expected_findings": meta["expected_findings"],
                "reviewed_at_utc": started_iso,
                "wall_clock_seconds": round(time.monotonic() - started_at, 1),
                "agent_result": agent_result,
            }
            _write_result_atomically(result_path, record)
            reviewed_this_run += 1
            print(f"[{index}/{total}] {fixture_id}: {agent_result.get('status')} -- stored")
            break

        if stop_run:
            break

    done = sum(1 for e in entries if (results_dir / f"{e['id']}.json").exists())
    print(
        f"\n{reviewed_this_run} fixture(s) reviewed this run, {skipped} already had results, "
        f"{done}/{total} total stored in {results_dir}."
    )
    if done < total:
        print("Run this script again (it will resume automatically) once ready to continue.")
    else:
        print("All fixtures reviewed. Run evaluation/evaluate_results.py to score them.")


if __name__ == "__main__":
    main()
