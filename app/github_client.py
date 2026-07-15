import base64
import re
from functools import lru_cache

from github import Auth, Github, GithubException, GithubIntegration

from app.config import get_settings

STATUS_CONTEXT = "pr-review-agent"

# GitHub rejects commit status descriptions longer than 140 characters.
_MAX_STATUS_DESCRIPTION = 140

_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


class GitHubNotifyError(RuntimeError):
    """Raised when the GitHub App's installation token can't be resolved, or
    the GitHub API rejects a status update or review comment."""


def _app_auth() -> Auth.AppAuth:
    settings = get_settings()
    private_key = base64.b64decode(settings.github_app_private_key_b64).decode()
    return Auth.AppAuth(settings.github_app_id, private_key)


@lru_cache
def _installation_auth(full_name: str) -> Auth.AppInstallationAuth:
    """Self-refreshing per-repo installation auth, cached for the process
    lifetime (same lru_cache pattern as config.get_settings()) so repeated
    API calls against the same repo within one review don't each re-resolve
    the installation -- AppInstallationAuth itself handles token refresh
    transparently on every request.
    """
    owner, repo = full_name.split("/", 1)
    app_auth = _app_auth()
    installation = GithubIntegration(auth=app_auth).get_repo_installation(owner, repo)
    return app_auth.get_installation_auth(installation.id)


def get_repo(full_name: str):
    return Github(auth=_installation_auth(full_name)).get_repo(full_name)


def get_installation_token(full_name: str) -> str:
    """A raw installation access token string for full_name, for callers that
    can't use a PyGithub Github(auth=...) client -- namely app/git_ops.py,
    which needs the token as text for git's own HTTP Authorization header.
    Resolved fresh on every call (checkouts happen once per review, so the
    extra API round-trip is negligible) rather than cached, since a raw
    string -- unlike AppInstallationAuth -- doesn't self-refresh.
    """
    owner, repo = full_name.split("/", 1)
    app_auth = _app_auth()
    integration = GithubIntegration(auth=app_auth)
    installation = integration.get_repo_installation(owner, repo)
    return integration.get_access_token(installation.id).token


def set_commit_status(full_name: str, sha: str, state: str, description: str) -> None:
    """Set a commit status check on sha. state is one of GitHub's commit status
    states: 'pending', 'success', 'failure', 'error'.
    """
    try:
        commit = get_repo(full_name).get_commit(sha)
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
        pr = get_repo(full_name).get_pull(pr_number)
        if comments:
            commit = get_repo(full_name).get_commit(sha)
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
        pr = get_repo(full_name).get_pull(pr_number)
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


def _commentable_lines_from_patch(patch: str) -> set[int]:
    """Line numbers (in the file's new/RIGHT-side content) that GitHub will
    accept an inline review comment on for this file -- i.e. actually part of
    the diff (added or context lines within a hunk), not just present
    somewhere in the file. Removed lines don't exist in the new file and are
    skipped without advancing the new-line counter.
    """
    lines: set[int] = set()
    new_line = None
    for line in patch.splitlines():
        match = _HUNK_HEADER_RE.match(line)
        if match:
            new_line = int(match.group(1))
            continue
        if new_line is None or line.startswith("\\"):
            continue
        if line.startswith("-"):
            continue
        lines.add(new_line)
        new_line += 1
    return lines


def get_diff_commentable_lines(full_name: str, pr_number: int) -> dict[str, set[int]]:
    """Map each changed file's repo-relative path to the set of line numbers
    a review comment can actually be anchored to for this PR -- GitHub's
    create_review rejects the *entire* review (not just the offending
    comment) if any comment's (path, line) isn't part of the diff, so callers
    building inline suggestions must filter against this before posting.
    """
    try:
        pr = get_repo(full_name).get_pull(pr_number)
        return {f.filename: _commentable_lines_from_patch(f.patch) for f in pr.get_files() if f.patch}
    except GithubException as exc:
        raise GitHubNotifyError(f"failed to fetch diff for {full_name}#{pr_number}: {exc}") from exc
