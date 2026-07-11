"""Live plumbing-correctness test for the review pipeline.

Every other test file in this suite mocks ChatGroq entirely -- which means
nothing in the default `pytest` run ever confirms the actual Groq prompt
format, tool schemas, or structured-output contract still works. A model
swap, a langchain_groq version bump, or a prompt edit could silently break
the live integration and the rest of the suite would stay green.

This file makes one real call chain through the real pipeline: real Groq
calls (both specialists + the Synthesizer), real MCP server subprocesses
(code_server.py, tester_server.py spawned exactly as app/agent.py spawns
them in production), and a real Redis (checkpointer + telemetry). It is
therefore slow, costs real Groq tokens, and needs live infra -- so it's
marked `integration` and excluded from the default run via pytest.ini's
addopts. Run explicitly with:

    pytest tests/test_integration.py -m integration -v
"""

import asyncio
import os
from datetime import datetime, timezone

import pytest
import redis

os.environ.setdefault("GITHUB_WEBHOOK_SECRET", "test-secret")
os.environ.setdefault("GITHUB_APP_ID", "123456")
os.environ.setdefault("GITHUB_APP_PRIVATE_KEY_B64", "dGVzdC1rZXk=")
# Deliberately not setting GROQ_API_KEY here.

from app.agent import run_pr_review_agent  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.telemetry import _get_sync_redis_client  # noqa: E402

pytestmark = pytest.mark.integration


def _has_real_groq_key() -> bool:
    # Goes through get_settings() -- the same lru_cache'd path
    # run_pr_review_agent itself uses -- rather than raw os.environ, because
    # Settings() also reads GROQ_API_KEY from .env (pydantic-settings'
    # source order is env vars > .env > field defaults, confirmed via
    # BaseSettings.settings_customise_sources). A real key that only lives
    # in .env, with nothing exported in the shell, would show up here but
    # not in os.environ directly. This only resolves correctly if this file
    # is run in isolation (as documented above) -- if collected alongside
    # sibling test modules that run os.environ.setdefault("GROQ_API_KEY",
    # "test-key") before this check, that fake value wins (real env vars
    # outrank .env) and this correctly reports no real key available.
    key = get_settings().groq_api_key
    return bool(key) and key != "test-key"


def _redis_reachable(url: str) -> bool:
    try:
        client = redis.Redis.from_url(url, socket_connect_timeout=2)
        client.ping()
        client.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_real_groq_key(), reason="No real GROQ_API_KEY in the environment")
def test_full_review_pipeline_against_real_groq_and_mcp_servers(tmp_path):
    settings = get_settings()
    if not _redis_reachable(settings.redis_url):
        pytest.skip(f"Redis not reachable at {settings.redis_url}")

    # Minimal fixture with one unambiguous, real issue -- kept tiny to
    # minimize token spend against Groq's tight daily budget (see
    # app/telemetry.py's check_budget_ok / CLAUDE.md's TPM/TPD notes).
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "vulnerable.py").write_text(
        "import sqlite3\n"
        "\n"
        "def get_user(conn, username):\n"
        "    query = \"SELECT * FROM users WHERE username = '\" + username + \"'\"\n"
        "    return conn.execute(query).fetchone()\n"
    )

    unique_sha = f"integration-{int(datetime.now(timezone.utc).timestamp())}"
    pr_metadata = {
        "action": "opened",
        "repository": {
            "id": 1,
            "name": "pr-review-agent-fixture",
            "full_name": "octocat/pr-review-agent-fixture",
            "clone_url": "https://github.com/octocat/pr-review-agent-fixture.git",
        },
        "pull_request": {
            "number": 999999,
            "title": "Integration test: SQL injection fixture",
            "draft": False,
            "head": {"ref": "integration-test", "sha": unique_sha},
            "modified_files": ["app/vulnerable.py"],
            "added_files": [],
        },
    }

    today = datetime.now(timezone.utc).date().isoformat()
    usage_client = _get_sync_redis_client()
    prompt_tokens_before = int(usage_client.get(f"usage:groq:prompt_tokens:{today}") or 0)

    result = asyncio.run(run_pr_review_agent(pr_metadata, tmp_path))

    # The contract that matters: a real run through the real graph produces
    # a structured, grounded review. status can only be "completed" if
    # neither specialist's circuit breaker tripped, the Synthesizer produced
    # a parseable ReviewOutput, and _is_grounded() found a real fetched file
    # basename in the final text -- i.e. this asserts the whole chain, not
    # just that *a* Groq call succeeded.
    assert result["status"] == "completed", f"Pipeline did not complete cleanly: {result}"
    assert "vulnerable.py" in result["summary"].lower()

    # Confirms record_usage's real-world contract too: this is the only
    # place in the suite that ever sees Groq's *actual* usage_metadata shape
    # rather than a hand-built fake one.
    prompt_tokens_after = int(usage_client.get(f"usage:groq:prompt_tokens:{today}") or 0)
    assert prompt_tokens_after > prompt_tokens_before, "record_usage recorded no prompt tokens from the live review"
