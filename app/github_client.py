from github import Github, GithubException

from app.config import get_settings

STATUS_CONTEXT = "pr-review-agent"

# GitHub rejects commit status descriptions longer than 140 characters.
_MAX_STATUS_DESCRIPTION = 140


class GitHubNotifyError(RuntimeError):
    """Raised when the GitHub API rejects a status update or review comment."""


def _repo(full_name: str):
    return Github(get_settings().github_api_token).get_repo(full_name)


def set_commit_status(full_name: str, sha: str, state: str, description: str) -> None:
    """Set a commit status check on sha. state is one of GitHub's commit status
    states: 'pending', 'success', 'failure', 'error'.
    """
    try:
        commit = _repo(full_name).get_commit(sha)
        commit.create_status(
            state=state,
            description=description[:_MAX_STATUS_DESCRIPTION],
            context=STATUS_CONTEXT,
        )
    except GithubException as exc:
        raise GitHubNotifyError(f"failed to set '{state}' status on {full_name}@{sha}: {exc}") from exc


def post_review(
    full_name: str,
    pr_number: int,
    sha: str,
    body: str,
    comments: list[dict] | None = None,
) -> None:
    """Submit body as a formal PR review (not just an issue comment).

    When comments is given, each dict must have GitHub's inline review-comment
    shape ('path', 'line', 'side', 'body') -- typically 'body' wraps a patch in
    a ```suggestion``` block so the reviewer can apply it from the PR UI with
    one click. GitHub requires line comments to be anchored to a specific
    commit_id, so sha is resolved to a Commit and passed as 'commit' whenever
    comments is non-empty.
    """
    try:
        pr = _repo(full_name).get_pull(pr_number)
        if comments:
            commit = _repo(full_name).get_commit(sha)
            pr.create_review(commit=commit, body=body, event="COMMENT", comments=comments)
        else:
            pr.create_review(body=body, event="COMMENT")
    except GithubException as exc:
        raise GitHubNotifyError(f"failed to post review on {full_name}#{pr_number}: {exc}") from exc


def list_changed_files(full_name: str, pr_number: int) -> tuple[list[str], list[str]]:
    """Return (modified_files, added_files) for pr_number via the GitHub API.

    A genuine GitHub pull_request webhook payload never carries a file list
    (unlike simulate_pr.py's payload, which populates it client-side) --
    GitHub only exposes changed files via this separate "list PR files" API.
    """
    try:
        pr = _repo(full_name).get_pull(pr_number)
        modified_files = []
        added_files = []
        for f in pr.get_files():
            if f.status == "added":
                added_files.append(f.filename)
            else:
                modified_files.append(f.filename)
        return modified_files, added_files
    except GithubException as exc:
        raise GitHubNotifyError(f"failed to list changed files for {full_name}#{pr_number}: {exc}") from exc
