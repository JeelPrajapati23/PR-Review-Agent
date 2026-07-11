import hashlib
import hmac
import json
import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("GITHUB_WEBHOOK_SECRET", "test-secret")
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("GITHUB_APP_ID", "123456")
os.environ.setdefault("GITHUB_APP_PRIVATE_KEY_B64", "dGVzdC1rZXk=")

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _mock_redis_lock():
    # Every test hits the dedup-lock check in app.main's webhook handler;
    # stub it out so tests never need a live Redis, and default to "lock
    # acquired" so existing .delay()-was-called assertions keep working.
    with patch("app.main._get_redis_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.set.return_value = True
        mock_get_client.return_value = mock_client
        yield mock_client


def _sign(payload: bytes) -> str:
    secret = get_settings().github_webhook_secret
    digest = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _pull_request_payload(action: str = "opened") -> dict:
    return {
        "action": action,
        "pull_request": {
            "number": 42,
            "title": "Add new feature",
            "draft": False,
            "head": {"ref": "feature/awesome-change", "sha": "abc123"},
        },
        "repository": {
            "id": 123456,
            "name": "pr-review-agent",
            "full_name": "octocat/pr-review-agent",
            "clone_url": "https://github.com/octocat/pr-review-agent.git",
        },
    }


@patch("app.main.process_pr_review_task.delay")
def test_webhook_accepts_valid_pull_request_event(mock_delay):
    body = json.dumps(_pull_request_payload()).encode()

    response = client.post(
        "/webhook/github",
        content=body,
        headers={
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": _sign(body),
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 202
    mock_delay.assert_called_once()


@patch("app.main.process_pr_review_task.delay")
def test_webhook_ignores_non_pull_request_event(mock_delay):
    body = json.dumps({"zen": "Keep it simple."}).encode()

    response = client.post(
        "/webhook/github",
        content=body,
        headers={
            "X-GitHub-Event": "ping",
            "X-Hub-Signature-256": _sign(body),
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 200
    mock_delay.assert_not_called()


@patch("app.main.process_pr_review_task.delay")
def test_webhook_skips_non_actionable_pull_request_action(mock_delay):
    body = json.dumps(_pull_request_payload(action="review_requested")).encode()

    response = client.post(
        "/webhook/github",
        content=body,
        headers={
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": _sign(body),
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 200
    mock_delay.assert_not_called()


@patch("app.main.process_pr_review_task.delay")
def test_webhook_accepts_synchronize_action(mock_delay):
    body = json.dumps(_pull_request_payload(action="synchronize")).encode()

    response = client.post(
        "/webhook/github",
        content=body,
        headers={
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": _sign(body),
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 202
    mock_delay.assert_called_once()


@patch("app.main.process_pr_review_task.delay")
def test_webhook_rejects_invalid_signature(mock_delay):
    body = json.dumps(_pull_request_payload()).encode()

    response = client.post(
        "/webhook/github",
        content=body,
        headers={
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": "sha256=deadbeef",
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 401
    mock_delay.assert_not_called()


@patch("app.main.process_pr_review_task.delay")
def test_webhook_skips_duplicate_redelivery_when_lock_held(mock_delay, _mock_redis_lock):
    # Simulates Redis SET NX failing because a review for this exact head sha
    # is already in flight (a redelivered webhook for the same commit).
    _mock_redis_lock.set.return_value = None
    body = json.dumps(_pull_request_payload()).encode()

    response = client.post(
        "/webhook/github",
        content=body,
        headers={
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": _sign(body),
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 200
    mock_delay.assert_not_called()


@patch("app.main.process_pr_review_task.delay")
def test_webhook_locks_on_repo_pr_and_sha(mock_delay, _mock_redis_lock):
    body = json.dumps(_pull_request_payload()).encode()

    client.post(
        "/webhook/github",
        content=body,
        headers={
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": _sign(body),
            "Content-Type": "application/json",
        },
    )

    _mock_redis_lock.set.assert_called_once_with(
        "lock:review:octocat/pr-review-agent:42:abc123", "1", nx=True, ex=15 * 60
    )
