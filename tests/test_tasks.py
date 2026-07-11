import os
from unittest.mock import patch

os.environ.setdefault("GITHUB_WEBHOOK_SECRET", "test-secret")
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("GITHUB_API_TOKEN", "test-token")

from app.tasks import _CommitStatusOnFailureTask, process_pr_review_task


def _event_data(draft: bool = False, action: str = "opened") -> dict:
    return {
        "action": action,
        "pull_request": {
            "number": 42,
            "title": "Add feature",
            "draft": draft,
            "head": {"ref": "feature/x", "sha": "abc123"},
            "modified_files": ["app/main.py"],
            "added_files": [],
        },
        "repository": {
            "id": 1,
            "name": "pr-review-agent",
            "full_name": "octocat/pr-review-agent",
            "clone_url": "https://github.com/octocat/pr-review-agent.git",
        },
    }


@patch("app.tasks.check_budget_ok")
@patch("app.tasks.ensure_repo_checkout")
def test_draft_pr_short_circuits_before_budget_check_or_checkout(mock_checkout, mock_budget_ok):
    result = process_pr_review_task(_event_data(draft=True))

    assert result == {
        "repository": "octocat/pr-review-agent",
        "pr_number": 42,
        "branch": "feature/x",
        "draft": True,
        "status": "skipped",
        "summary": "Skipping: Draft Pull Request",
    }
    mock_budget_ok.assert_not_called()
    mock_checkout.assert_not_called()


@patch("app.tasks.set_commit_status")
@patch("app.tasks.ensure_repo_checkout")
@patch("app.tasks.check_budget_ok", return_value=False)
def test_budget_exhausted_defers_review_without_checkout(mock_budget_ok, mock_checkout, mock_set_status):
    result = process_pr_review_task(_event_data(draft=False))

    assert result["status"] == "skipped"
    assert result["summary"] == "Token budget exhausted for today; review deferred."
    mock_budget_ok.assert_called_once()
    mock_checkout.assert_not_called()
    mock_set_status.assert_called_once_with(
        "octocat/pr-review-agent",
        "abc123",
        "success",
        "Review deferred: daily Groq token budget reached",
    )


@patch("app.tasks.run_pr_review_agent")
@patch("app.tasks.set_commit_status")
@patch("app.tasks.ensure_repo_checkout")
@patch("app.tasks.check_budget_ok", return_value=True)
def test_normal_flow_proceeds_when_not_draft_and_budget_ok(mock_budget_ok, mock_checkout, mock_set_status, mock_agent):
    mock_checkout.return_value = "/tmp/fake-repo"
    mock_agent.return_value = {"status": "completed", "summary": "All good."}

    result = process_pr_review_task(_event_data(draft=False))

    mock_budget_ok.assert_called_once()
    mock_checkout.assert_called_once()
    mock_agent.assert_called_once()
    assert result["status"] == "completed"
    assert result["summary"] == "All good."


# --- on_failure: never leave a PR stuck on a 'pending' commit status ---


@patch("app.tasks.set_commit_status")
def test_on_failure_posts_failure_status_from_event_data(mock_set_status):
    # Exercises the hook's own logic directly (bypassing Celery's tracer),
    # as Celery would call it: args is the positional-args tuple the task
    # was invoked with.
    task = _CommitStatusOnFailureTask()

    task.on_failure(
        RuntimeError("provider exhausted"),
        "task-id-123",
        (_event_data(draft=False),),
        {},
        einfo=None,
    )

    mock_set_status.assert_called_once_with(
        "octocat/pr-review-agent",
        "abc123",
        "failure",
        "Review aborted: execution failure or provider exhaustion",
    )


@patch("app.tasks.set_commit_status")
def test_on_failure_logs_and_does_not_raise_on_malformed_event_data(mock_set_status):
    task = _CommitStatusOnFailureTask()

    # Must not raise even if event_data itself is what's broken -- there's
    # no commit to notify in that case, only a clean log-and-return.
    task.on_failure(RuntimeError("boom"), "task-id-456", ({"not": "a valid payload"},), {}, einfo=None)

    mock_set_status.assert_not_called()


@patch("app.tasks.set_commit_status")
@patch("app.tasks.check_budget_ok", side_effect=RuntimeError("unexpected crash"))
def test_on_failure_fires_via_real_celery_apply_for_non_retryable_exception(mock_budget_ok, mock_set_status):
    # End-to-end wiring check through Celery's actual tracer (.apply()),
    # not just the hook's own logic in isolation: a non-retryable exception
    # (RuntimeError isn't in autoretry_for) should reach on_failure on the
    # very first attempt, with no retries in between.
    result = process_pr_review_task.apply(args=[_event_data(draft=False)], throw=False)

    assert result.failed()
    mock_set_status.assert_called_once_with(
        "octocat/pr-review-agent",
        "abc123",
        "failure",
        "Review aborted: execution failure or provider exhaustion",
    )
