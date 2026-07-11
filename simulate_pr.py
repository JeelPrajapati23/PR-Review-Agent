"""Fire a signed mock 'pull_request' webhook at the local FastAPI server to
exercise the full LangGraph self-healing review loop end-to-end against a
real checked-out repo (see GIT_WORKSPACE_ROOT in .env).

Looks up the real PR's branch, head sha, and clone_url via the GitHub API
(keyed off --pr-number) rather than fabricating them, since the commit-status
and review-comment calls only succeed against a PR/commit that actually
exists.

Usage:
    uvicorn app.main:app --port 8000          # in one terminal
    celery -A app.worker.celery_app worker --loglevel=info -P solo   # in another
    python simulate_pr.py --pr-number 3       # in a third

Examples:
    python simulate_pr.py --pr-number 3
    python simulate_pr.py --pr-number 3 --full-name JeelPrajapati23/test-repo
"""
import argparse
import hashlib
import hmac
import json

import httpx

from app.config import get_settings
from app.github_client import get_repo, list_changed_files

WEBHOOK_URL = "http://localhost:8000/webhook/github"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pr-number", type=int, required=True, help="Real, existing PR number to review")
    parser.add_argument(
        "--full-name",
        default="JeelPrajapati23/test-repo",
        help="owner/repo, used both for the GitHub API lookup and the git checkout path under GIT_WORKSPACE_ROOT",
    )
    parser.add_argument("--title", default=None, help="Override the PR title instead of using the real one")
    parser.add_argument(
        "--clone-url",
        default=None,
        help="git remote URL to clone/fetch from (defaults to the repo's real clone_url)",
    )
    return parser.parse_args()


def _build_payload(args: argparse.Namespace) -> dict:
    repo = get_repo(args.full_name)
    pr = repo.get_pull(args.pr_number)

    modified_files, added_files = list_changed_files(args.full_name, args.pr_number)

    return {
        "action": "opened",
        "pull_request": {
            "number": pr.number,
            "title": args.title or pr.title,
            "draft": pr.draft,
            "modified_files": modified_files,
            "added_files": added_files,
            "head": {"ref": pr.head.ref, "sha": pr.head.sha},
        },
        "repository": {
            "id": repo.id,
            "name": repo.name,
            "full_name": args.full_name,
            "clone_url": args.clone_url or repo.clone_url,
        },
    }


def _sign(secret: str, payload: bytes) -> str:
    digest = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def main() -> None:
    args = _parse_args()
    payload = _build_payload(args)

    secret = get_settings().github_webhook_secret
    body = json.dumps(payload).encode()

    response = httpx.post(
        WEBHOOK_URL,
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": _sign(secret, body),
        },
        timeout=10.0,
    )

    pull_request = payload["pull_request"]
    print(
        f"Simulating PR: #{pull_request['number']} title={pull_request['title']!r} "
        f"branch={pull_request['head']['ref']!r} sha={pull_request['head']['sha']}\n"
        f"modified_files={pull_request['modified_files']} added_files={pull_request['added_files']}"
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")


if __name__ == "__main__":
    main()
