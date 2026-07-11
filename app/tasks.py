import asyncio
import logging

from celery import Task
from groq import APIConnectionError, APITimeoutError, RateLimitError

from app.agent import run_pr_review_agent
from app.config import get_settings
from app.git_ops import GitCheckoutError, ensure_repo_checkout
from app.github_client import GitHubNotifyError, list_changed_files, set_commit_status
from app.telemetry import check_budget_ok
from app.worker import celery_app

logger = logging.getLogger(__name__)

# Only genuinely transient Groq API failures are worth a backoff retry. Any
# other exception (a local bug -- KeyError on a malformed event_data, an
# uncaught AttributeError, etc.) should fail the task on turn 1 so it shows
# up in tracking immediately instead of being masked behind a 3x exponential
# backoff loop that can't fix it anyway.
_TRANSIENT_GROQ_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError)

# Upper bound on a single review panel run. Celery's own task_time_limit/
# task_soft_time_limit are *not* enforced by the solo pool (`-P solo`,
# required on Windows per CLAUDE.md -- confirmed via solo.TaskPool's own
# _get_info reporting an empty 'timeouts' tuple), so a genuine hang inside
# run_pr_review_agent (a stuck MCP subprocess, a wedged Groq call) would
# otherwise never raise, never hit on_failure, and leave the PR's commit
# status stuck on 'pending' forever -- the exact failure mode
# _CommitStatusOnFailureTask exists to prevent. Enforcing the bound here with
# asyncio.wait_for works regardless of pool type. 10 minutes is generous
# headroom over a normal review (observed ~8-12k tokens across three
# sequential Groq calls, typically well under a minute) while still bounding
# the worst case. A TimeoutError here isn't one of _TRANSIENT_GROQ_ERRORS, so
# it fails the task immediately rather than retrying a hang three more times.
_AGENT_TIMEOUT_SECONDS = 10 * 60


def _notify_status(repository: str, sha: str, state: str, description: str, pr_number: int) -> None:
    try:
        set_commit_status(repository, sha, state, description)
    except GitHubNotifyError:
        logger.exception("Failed to set '%s' commit status for %s#%s", state, repository, pr_number)


class _CommitStatusOnFailureTask(Task):
    """Guarantees a PR's commit status never gets stuck on 'pending'.

    Celery invokes on_failure exactly once, only for a task's genuinely
    terminal failure -- a non-retryable exception, or a retryable one (see
    _TRANSIENT_GROQ_ERRORS) that has exhausted every autoretry_for attempt.
    It is never called for an individual retry attempt itself: those raise
    an internal Retry control-flow exception instead (see
    celery/app/autoretry.py's add_autoretry_behaviour, which wraps this
    task's run() and calls task.retry() -- a different path entirely from
    an unhandled failure reaching the tracer). That's exactly the signal
    needed here: post the final failure status once real failure is
    certain, without flickering the commit status pending/failure/pending
    across intermediate retries the way notifying on every exception inside
    the task body would.
    """

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        event_data = args[0] if args else kwargs.get("event_data")
        try:
            repository = event_data["repository"]["full_name"]
            sha = event_data["pull_request"]["head"]["sha"]
            pr_number = event_data["pull_request"]["number"]
        except (KeyError, TypeError, IndexError):
            logger.error(
                "process_pr_review_task (task_id=%s) failed with no usable event_data to notify GitHub: %s",
                task_id,
                exc,
            )
            return

        logger.error(
            "process_pr_review_task permanently failed for %s#%s after all retries: %s",
            repository,
            pr_number,
            exc,
        )
        _notify_status(
            repository,
            sha,
            "failure",
            "Review aborted: execution failure or provider exhaustion",
            pr_number,
        )


@celery_app.task(
    name="process_pr_review_task",
    base=_CommitStatusOnFailureTask,
    autoretry_for=_TRANSIENT_GROQ_ERRORS,
    max_retries=3,
    retry_backoff=True,
    retry_jitter=True,
)
def process_pr_review_task(event_data: dict) -> dict:
    repository = event_data["repository"]["full_name"]
    pull_request = event_data["pull_request"]
    pr_number = pull_request["number"]
    branch = pull_request["head"]["ref"]
    sha = pull_request["head"]["sha"]
    is_draft = pull_request["draft"]
    action = event_data["action"]
    clone_url = event_data["repository"]["clone_url"]

    # Draft PRs aren't ready for review yet; skip before any Groq/checkout
    # work happens rather than spending budget on code the author hasn't
    # marked ready. No commit status is posted here (mirrors how main.py
    # silently skips other non-actionable webhook events) -- nothing has
    # been marked "pending" yet at this point in the flow.
    if is_draft:
        logger.info("Skipping draft PR %s#%s", repository, pr_number)
        return {
            "repository": repository,
            "pr_number": pr_number,
            "branch": branch,
            "draft": is_draft,
            "status": "skipped",
            "summary": "Skipping: Draft Pull Request",
        }

    # Token budget guard: check before any checkout/agent work starts, using
    # the daily totals app/telemetry.py's record_usage accumulates. Posts a
    # 'success' commit status rather than 'failure' -- GitHub's classic
    # Commit Status API has no true "neutral" state, and running out of
    # shared daily Groq quota is an ops constraint, not a defect in this PR,
    # so it shouldn't block a merge gated on this check the way a real
    # review failure should.
    settings = get_settings()
    if not check_budget_ok(settings.groq_model):
        logger.warning("Token Budget Exhausted: deferring review for %s#%s", repository, pr_number)
        _notify_status(
            repository,
            sha,
            "success",
            "Review deferred: daily Groq token budget reached",
            pr_number,
        )
        return {
            "repository": repository,
            "pr_number": pr_number,
            "branch": branch,
            "draft": is_draft,
            "status": "skipped",
            "summary": "Token budget exhausted for today; review deferred.",
        }

    # A genuine GitHub webhook payload (as opposed to simulate_pr.py's, which
    # populates these client-side) never carries a file list, so fetch it
    # here whenever it's missing -- otherwise target_files stays empty and,
    # combined with the agent no longer being allowed to list the whole repo,
    # it would have nothing to review.
    if not pull_request.get("modified_files") and not pull_request.get("added_files"):
        try:
            modified_files, added_files = list_changed_files(repository, pr_number)
            pull_request["modified_files"] = modified_files
            pull_request["added_files"] = added_files
        except GitHubNotifyError:
            logger.exception(
                "Failed to fetch changed files for %s#%s; proceeding with an empty target_files list",
                repository,
                pr_number,
            )

    target_files = pull_request.get("modified_files", []) + pull_request.get("added_files", [])

    logger.info(
        "Processing PR review task: repo=%s pr=#%s branch=%s draft=%s action=%s target_files=%s",
        repository,
        pr_number,
        branch,
        is_draft,
        action,
        target_files,
    )

    _notify_status(repository, sha, "pending", "PR review agent is analyzing this PR", pr_number)

    try:
        repo_path = ensure_repo_checkout(clone_url, repository, branch)
    except GitCheckoutError as exc:
        logger.exception("Git checkout failed for %s#%s", repository, pr_number)
        _notify_status(repository, sha, "failure", "Could not check out the PR branch for review", pr_number)
        return {
            "repository": repository,
            "pr_number": pr_number,
            "branch": branch,
            "draft": is_draft,
            "status": "error",
            "summary": f"Could not prepare repository checkout: {exc}",
        }

    result = asyncio.run(
        asyncio.wait_for(run_pr_review_agent(event_data, repo_path), timeout=_AGENT_TIMEOUT_SECONDS)
    )

    logger.info("Agent review result for %s#%s: %s", repository, pr_number, result)

    logger.info("Finished PR review task for %s#%s", repository, pr_number)

    return {
        "repository": repository,
        "pr_number": pr_number,
        "branch": branch,
        "draft": is_draft,
        "status": result.get("status", "completed"),
        "summary": result.get("summary"),
    }
