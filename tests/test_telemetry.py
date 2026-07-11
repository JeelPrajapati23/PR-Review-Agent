import asyncio
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from app.agent import _sum_usage_metadata
from app.telemetry import _METRICS_TTL_SECONDS, calculate_cost, check_budget_ok, record_usage

# Regression guard for a real bug caught via a live-Redis smoke test: a
# module-level cached redis.asyncio client tied to one asyncio.run()'s event
# loop broke on a second, later asyncio.run() call with "Event loop is
# closed". record_usage now opens a fresh client per call instead, so these
# fakes model that -- __aenter__/__aexit__ on the client, not just the
# pipeline -- to make sure a regression back to caching would show up here.


def test_calculate_cost_known_model():
    cost = calculate_cost("llama-3.3-70b-versatile", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost == pytest.approx(0.59 + 0.79)


def test_calculate_cost_scales_linearly_with_tokens():
    cost = calculate_cost("llama-3.3-70b-versatile", prompt_tokens=500_000, completion_tokens=0)
    assert cost == pytest.approx(0.295)


def test_calculate_cost_unknown_model_returns_zero():
    assert calculate_cost("some-other-model", prompt_tokens=1000, completion_tokens=1000) == 0.0


class _FakePipeline:
    """Stands in for redis.asyncio's Pipeline: records every queued command
    instead of talking to a real Redis server, so tests can assert on exact
    key names/values without any live infra.
    """

    def __init__(self):
        self.commands: list[tuple] = []

    def incrby(self, key, amount):
        self.commands.append(("incrby", key, amount))
        return self

    def incrbyfloat(self, key, amount):
        self.commands.append(("incrbyfloat", key, amount))
        return self

    def expire(self, key, ttl):
        self.commands.append(("expire", key, ttl))
        return self

    async def execute(self):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeRedisClient:
    def __init__(self):
        self.pipeline_obj = _FakePipeline()

    def pipeline(self, transaction=True):
        return self.pipeline_obj

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_record_usage_increments_expected_redis_keys():
    fake_client = _FakeRedisClient()
    today = datetime.now(timezone.utc).date().isoformat()

    with patch("app.telemetry._new_redis_client", return_value=fake_client):
        asyncio.run(record_usage("llama-3.3-70b-versatile", prompt_tokens=1000, completion_tokens=500))

    commands = fake_client.pipeline_obj.commands
    expected_cost = calculate_cost("llama-3.3-70b-versatile", 1000, 500)

    assert ("incrby", f"usage:groq:prompt_tokens:{today}", 1000) in commands
    assert ("incrby", f"usage:groq:completion_tokens:{today}", 500) in commands
    assert ("incrbyfloat", f"usage:groq:cost:{today}", expected_cost) in commands
    assert ("expire", f"usage:groq:prompt_tokens:{today}", _METRICS_TTL_SECONDS) in commands
    assert ("expire", f"usage:groq:completion_tokens:{today}", _METRICS_TTL_SECONDS) in commands
    assert ("expire", f"usage:groq:cost:{today}", _METRICS_TTL_SECONDS) in commands


def test_record_usage_skips_redis_when_no_tokens_used():
    fake_client = _FakeRedisClient()

    with patch("app.telemetry._new_redis_client", return_value=fake_client):
        asyncio.run(record_usage("llama-3.3-70b-versatile", prompt_tokens=0, completion_tokens=0))

    assert fake_client.pipeline_obj.commands == []


def test_record_usage_swallows_redis_errors_instead_of_raising():
    class _BrokenClient:
        async def __aenter__(self):
            raise ConnectionError("redis is down")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    with patch("app.telemetry._new_redis_client", return_value=_BrokenClient()):
        # A telemetry failure must never fail the review it's instrumenting.
        asyncio.run(record_usage("llama-3.3-70b-versatile", prompt_tokens=10, completion_tokens=5))


def test_record_usage_opens_a_fresh_client_across_separate_event_loops():
    # Regression test for the exact bug the live smoke test caught: two
    # reviews running in the same worker process each call asyncio.run(...)
    # independently, so a client cached from the first run's event loop must
    # never be reused by the second.
    seen_clients = []

    def make_client():
        client = _FakeRedisClient()
        seen_clients.append(client)
        return client

    with patch("app.telemetry._new_redis_client", side_effect=make_client):
        asyncio.run(record_usage("llama-3.3-70b-versatile", prompt_tokens=1000, completion_tokens=500))
        asyncio.run(record_usage("llama-3.3-70b-versatile", prompt_tokens=100, completion_tokens=50))

    assert len(seen_clients) == 2
    assert seen_clients[0] is not seen_clients[1]
    prompt_key = seen_clients[0].pipeline_obj.commands[0][1]
    assert seen_clients[0].pipeline_obj.commands[0] == ("incrby", prompt_key, 1000)
    assert seen_clients[1].pipeline_obj.commands[0] == ("incrby", prompt_key, 100)


def test_sum_usage_metadata_sums_across_all_ai_messages():
    messages = [
        AIMessage(content="turn 1", usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}),
        AIMessage(content="turn 2", usage_metadata={"input_tokens": 150, "output_tokens": 30, "total_tokens": 180}),
    ]

    assert _sum_usage_metadata(messages) == (250, 50)


def test_sum_usage_metadata_ignores_messages_without_usage_metadata():
    messages = [AIMessage(content="no usage data")]

    assert _sum_usage_metadata(messages) == (0, 0)


class _FakeSyncRedisClient:
    """Stands in for check_budget_ok's synchronous redis.Redis client."""

    def __init__(self, values: dict[str, str]):
        self._values = values

    def mget(self, *keys):
        return [self._values.get(key) for key in keys]


def test_check_budget_ok_true_when_under_limit():
    today = datetime.now(timezone.utc).date().isoformat()
    fake_client = _FakeSyncRedisClient(
        {f"usage:groq:prompt_tokens:{today}": "80000", f"usage:groq:completion_tokens:{today}": "5000"}
    )

    with patch("app.telemetry._get_sync_redis_client", return_value=fake_client):
        assert check_budget_ok("llama-3.3-70b-versatile", safe_limit=90_000) is True


def test_check_budget_ok_false_when_over_limit():
    today = datetime.now(timezone.utc).date().isoformat()
    fake_client = _FakeSyncRedisClient(
        {f"usage:groq:prompt_tokens:{today}": "80000", f"usage:groq:completion_tokens:{today}": "15000"}
    )

    with patch("app.telemetry._get_sync_redis_client", return_value=fake_client):
        assert check_budget_ok("llama-3.3-70b-versatile", safe_limit=90_000) is False


def test_check_budget_ok_true_when_no_usage_recorded_yet():
    fake_client = _FakeSyncRedisClient({})

    with patch("app.telemetry._get_sync_redis_client", return_value=fake_client):
        assert check_budget_ok("llama-3.3-70b-versatile") is True


def test_check_budget_ok_fails_open_on_redis_error():
    class _BrokenClient:
        def mget(self, *keys):
            raise ConnectionError("redis is down")

    with patch("app.telemetry._get_sync_redis_client", return_value=_BrokenClient()):
        assert check_budget_ok("llama-3.3-70b-versatile") is True
