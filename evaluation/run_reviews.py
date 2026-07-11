"""Run the golden dataset (see manifest.json) through the real review panel,
one PR fixture at a time, writing each result to disk as soon as it completes.

Groq's free-tier daily token budget (see CLAUDE.md / app/telemetry.py) is far
too tight to review all 15 fixtures in one sitting. This script is built
around that constraint rather than around finishing in one run:

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

Requires a reachable Redis (checkpointer + telemetry, see CLAUDE.md's
agent-redis container) and a real GROQ_API_KEY / GITHUB_API_TOKEN /
GITHUB_WEBHOOK_SECRET (via .env, same as running the app for real) --
this makes genuine Groq calls and spends real tokens, exactly like
tests/test_integration.py's single fixture.
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
from app.telemetry import _get_sync_redis_client, check_budget_ok

DATASET_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = DATASET_ROOT / "manifest.json"

# Fake repo identity used for every fixture -- same pattern as
# tests/test_integration.py. GitHub notification calls (commit status, PR
# review) will fail against this nonexistent repo/PR; app/agent.py's
# _notify/_post_review already catch GitHubNotifyError and log-and-continue,
# so this is harmless noise, not a failure.
_FAKE_REPO_FULL_NAME = "octocat/pr-review-eval-fixture"
_FAKE_REPO_CLONE_URL = "https://github.com/octocat/pr-review-eval-fixture.git"


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


def _build_pr_metadata(entry: dict, meta: dict, pr_number: int, run_tag: str) -> dict:
    # The fake sha folds in run_tag (derived from --results-dir) rather than
    # just fixture_id -- app/agent.py checkpoints panel/specialist state in
    # Redis keyed off this sha (see _thread_id_for in CLAUDE.md), and that
    # checkpointer has no TTL. A sha fixed to fixture_id alone would make
    # every future re-run of this script -- another day, another
    # --results-dir, a regression check after a prompt change -- resume and
    # keep growing the *same* Redis thread as every prior run instead of
    # starting fresh, silently ballooning prompt tokens each time (confirmed
    # directly: one golden-dataset fixture accumulated 47 checkpoints across
    # repeated runs, driving a single review from the expected ~10-12k
    # tokens to 50k+). Tying it to --results-dir keeps retries of a
    # genuinely deferred fixture within the *same* run resuming correctly
    # (the whole point of the deferred-retry feature below), while a
    # different --results-dir -- a deliberately fresh pass -- gets its own
    # thread lineage automatically, with no manual Redis cleanup required.
    fixture_id = entry["id"]
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
            "head": {"ref": f"eval/{fixture_id}", "sha": f"eval-{run_tag}-{fixture_id}"},
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
    settings = get_settings()
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

        if not check_budget_ok(settings.groq_model):
            used = _today_usage_tokens()
            remaining_ids = [e["id"] for e in entries[index - 1 :] if not _is_completed(results_dir, e["id"])]
            print(
                f"\nStopping: today's Groq token budget is exhausted ({used} tokens used). "
                f"{len(remaining_ids)} fixture(s) left ({', '.join(remaining_ids)}). "
                f"Re-run this script after the daily quota resets to continue -- already-stored "
                f"results in {results_dir} are untouched."
            )
            break

        print(f"[{index}/{total}] {fixture_id}: reviewing...")
        pr_number = 900_000 + index
        pr_metadata = _build_pr_metadata(entry, meta, pr_number, run_tag)

        started_at = time.monotonic()
        started_iso = datetime.now(timezone.utc).isoformat()
        try:
            agent_result = asyncio.run(run_pr_review_agent(pr_metadata, fixture_dir))
        except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
            # A real Groq rate-limit/connection failure mid-review -- almost
            # certainly the daily quota, not a fluke worth retrying inside
            # this loop. Store what we know and stop the whole run rather
            # than immediately hammering Groq again for the next fixture.
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
            print(f"[{index}/{total}] {fixture_id}: deferred ({type(exc).__name__}) -- stored, stopping run")
            break
        except Exception as exc:
            # Anything else is fixture-specific (bad fixture setup, a genuine
            # bug), not quota exhaustion -- store it and move on so one bad
            # fixture doesn't stall the other 14.
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
            continue

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
