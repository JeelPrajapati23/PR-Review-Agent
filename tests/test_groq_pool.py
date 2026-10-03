import asyncio
import os

os.environ.setdefault("GITHUB_WEBHOOK_SECRET", "test-secret")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("GITHUB_APP_ID", "12345")
os.environ.setdefault("GITHUB_APP_PRIVATE_KEY_B64", "test-private-key-b64")

import httpx
import pytest
from groq import APIStatusError, RateLimitError
from langchain_core.messages import HumanMessage

from app import groq_pool
from app.groq_pool import (
    MIN_OUTPUT_TOKENS,
    GroqKeyPool,
    GroqPoolExhausted,
    RotatingChatGroq,
    fit_request,
    is_daily_limit,
    parse_retry_after,
)


@pytest.fixture(autouse=True)
def _fresh_pool_state(monkeypatch):
    groq_pool._KEY_STATES.clear()
    groq_pool._POOLS.clear()
    monkeypatch.setattr(groq_pool, "_chars_per_token", 3.0)
    yield
    groq_pool._KEY_STATES.clear()
    groq_pool._POOLS.clear()


class _Clock:
    """Stands in for time.monotonic; asyncio.sleep advances it instantly."""

    def __init__(self):
        self.now = 1000.0
        self.slept: list[float] = []

    def monotonic(self):
        return self.now

    async def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(groq_pool.time, "monotonic", fake.monotonic)
    monkeypatch.setattr(groq_pool, "_sleep", fake.sleep)
    return fake


def _status_error(cls, status: int, message: str, headers: dict | None = None):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status, request=request, headers=headers or {})
    if cls is APIStatusError:
        return APIStatusError(message, response=response, body=None)
    return cls(message, response=response, body=None)


# ---------------------------------------------------------------------------
# Rate-limit parsing
# ---------------------------------------------------------------------------


def test_parse_retry_after_reads_groq_message_durations():
    assert parse_retry_after(Exception("Please try again in 1m2.5s.")) == pytest.approx(62.5)
    assert parse_retry_after(Exception("Please try again in 640ms.")) == pytest.approx(0.64)
    assert parse_retry_after(Exception("Please try again in 7.2s")) == pytest.approx(7.2)
    assert parse_retry_after(Exception("no hint here")) is None


def test_parse_retry_after_prefers_header():
    exc = _status_error(RateLimitError, 429, "Please try again in 50s", headers={"retry-after": "3"})
    assert parse_retry_after(exc) == 3.0


def test_is_daily_limit_distinguishes_tpd_from_tpm():
    assert is_daily_limit(Exception("Rate limit reached on tokens per day (TPD): Limit 200000"))
    assert not is_daily_limit(Exception("Rate limit reached on tokens per minute (TPM): Limit 8000"))


# ---------------------------------------------------------------------------
# Fitting a request under the per-minute budget
# ---------------------------------------------------------------------------


def test_fit_request_leaves_small_requests_alone():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    fitted, max_output, estimate = fit_request(messages, 0, 4000, 7200)
    assert fitted == messages
    assert max_output == 4000
    assert estimate <= 7200


def test_fit_request_lowers_max_tokens_before_touching_context():
    messages = [{"role": "user", "content": "x" * 12000}]  # ~4k tokens
    fitted, max_output, estimate = fit_request(messages, 0, 4000, 7200)
    assert fitted[0]["content"] == messages[0]["content"]
    assert MIN_OUTPUT_TOKENS <= max_output < 4000
    assert estimate <= 7200


def test_fit_request_truncates_older_tool_results_first_and_never_the_task():
    task = "review this diff " * 200
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": task},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "OLD" * 4000},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "2"}]},
        {"role": "tool", "tool_call_id": "2", "content": "NEW" * 1000},
    ]
    fitted, max_output, estimate = fit_request(messages, 0, 4000, 7200)
    assert fitted[1]["content"] == task
    assert len(fitted[3]["content"]) < 1000
    assert "omitted to fit" in fitted[3]["content"]
    assert fitted[5]["content"] == messages[5]["content"]
    assert estimate <= 7200
    # Caller's messages untouched.
    assert messages[3]["content"] == "OLD" * 4000


def test_fit_request_trims_latest_tool_result_only_when_still_over():
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "L" * 30000},  # ~10k tokens alone
    ]
    fitted, max_output, estimate = fit_request(messages, 0, 4000, 7200)
    assert "omitted to fit" in fitted[2]["content"]
    assert max_output >= MIN_OUTPUT_TOKENS
    assert estimate <= 7200


# ---------------------------------------------------------------------------
# Key pool: rotation, pacing, cooldowns
# ---------------------------------------------------------------------------


def test_pool_rotates_round_robin_across_keys(clock):
    pool = GroqKeyPool("t", ["k1", "k2", "k3"], tpm_limit=8000, max_wait_seconds=60)

    async def run():
        return [(await pool.acquire(100))[0].key for _ in range(4)]

    assert asyncio.run(run()) == ["k1", "k2", "k3", "k1"]
    assert clock.slept == []


def test_pool_waits_for_window_when_every_key_is_saturated(clock):
    pool = GroqKeyPool("t", ["k1", "k2"], tpm_limit=8000, max_wait_seconds=120)

    async def run():
        await pool.acquire(7000)
        await pool.acquire(7000)
        return (await pool.acquire(7000))[0].key

    assert asyncio.run(run()) == "k1"
    # Waited exactly until k1's first reservation aged out of its minute.
    assert sum(clock.slept) == pytest.approx(60.0)


def test_pool_skips_cooling_key_and_raises_past_max_wait(clock):
    pool = GroqKeyPool("t", ["k1", "k2"], tpm_limit=8000, max_wait_seconds=30)
    daily = _status_error(RateLimitError, 429, "tokens per day (TPD): Please try again in 10m")

    async def run():
        state, entry = await pool.acquire(100)
        pool.on_rate_limited(state, entry, daily)
        state, entry = await pool.acquire(100)
        pool.on_rate_limited(state, entry, daily)
        await pool.acquire(100)

    with pytest.raises(GroqPoolExhausted):
        asyncio.run(run())


# ---------------------------------------------------------------------------
# RotatingChatGroq end to end (fake Groq client per key)
# ---------------------------------------------------------------------------


_OK_RESPONSE = {
    "choices": [{"message": {"role": "assistant", "content": "looks good"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
}


class _FakeCompletions:
    def __init__(self, key, script, calls):
        self.key = key
        self.script = script
        self.calls = calls

    async def create(self, **params):
        self.calls.append((self.key, params["max_tokens"]))
        queue = self.script.get(self.key) or []
        outcome = queue.pop(0) if queue else _OK_RESPONSE
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _model_with_fakes(pool, script, calls):
    model = RotatingChatGroq.from_pool(pool, model="openai/gpt-oss-120b", temperature=0.1, max_tokens=4000)
    model._clients = {key: _FakeCompletions(key, script, calls) for key in pool.keys}
    return model


def test_rotating_model_rotates_to_next_key_on_429(clock):
    pool = GroqKeyPool("t", ["k1", "k2"], tpm_limit=8000, max_wait_seconds=60)
    calls = []
    script = {"k1": [_status_error(RateLimitError, 429, "tokens per minute (TPM). Please try again in 12s")]}
    model = _model_with_fakes(pool, script, calls)

    result = asyncio.run(model.ainvoke([HumanMessage("review")]))

    assert result.content == "looks good"
    assert [key for key, _ in calls] == ["k1", "k2"]
    assert groq_pool._KEY_STATES["k1"].cooldown_until == pytest.approx(clock.now + 12)
    # Reservation settled to real usage (150), the failed one zeroed.
    assert sum(t for _, t in groq_pool._KEY_STATES["k2"].usage) == 150
    assert sum(t for _, t in groq_pool._KEY_STATES["k1"].usage) == 0


def test_rotating_model_waits_out_cooldown_when_single_key(clock):
    pool = GroqKeyPool("t", ["k1"], tpm_limit=8000, max_wait_seconds=60)
    calls = []
    script = {"k1": [_status_error(RateLimitError, 429, "Please try again in 5s")]}
    model = _model_with_fakes(pool, script, calls)

    result = asyncio.run(model.ainvoke([HumanMessage("review")]))
    assert result.content == "looks good"
    assert len(calls) == 2
    assert sum(clock.slept) >= 5


def test_rotating_model_shrinks_request_on_413(clock):
    pool = GroqKeyPool("t", ["k1"], tpm_limit=8000, max_wait_seconds=60)
    calls = []
    too_large = _status_error(APIStatusError, 413, "Request too large on tokens per minute (TPM)")
    script = {"k1": [too_large]}
    model = _model_with_fakes(pool, script, calls)

    asyncio.run(model.ainvoke([HumanMessage("x" * 15000)]))
    assert len(calls) == 2
    assert calls[1][1] < calls[0][1]


def test_rotating_model_is_async_only():
    pool = GroqKeyPool("t", ["k1"], tpm_limit=8000, max_wait_seconds=60)
    model = RotatingChatGroq.from_pool(pool, model="openai/gpt-oss-120b")
    with pytest.raises(NotImplementedError):
        model.invoke([HumanMessage("hi")])
