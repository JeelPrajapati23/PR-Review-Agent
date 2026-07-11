import logging
from datetime import datetime, timezone
from functools import lru_cache

import redis
import redis.asyncio as aioredis

from app.config import get_settings

logger = logging.getLogger(__name__)

# USD cost per 1,000,000 tokens. Sourced from Groq's published pricing at
# https://groq.com/pricing as of writing -- re-verify against that page
# before trusting cost totals for anything billing-sensitive, since provider
# pricing changes without notice and isn't queryable via the Groq API itself.
_MODEL_PRICING_PER_MILLION_TOKENS: dict[str, dict[str, float]] = {
    "llama-3.3-70b-versatile": {"prompt": 0.59, "completion": 0.79},
}

# Metrics keys expire after this long so old daily telemetry self-cleans
# instead of accumulating in Redis forever.
_METRICS_TTL_SECONDS = 7 * 24 * 60 * 60


@lru_cache
def _get_sync_redis_client() -> redis.Redis:
    # Safe to cache, unlike _new_redis_client below: a plain redis.Redis
    # connection isn't bound to an asyncio event loop, so it doesn't hit the
    # "Event loop is closed" failure that ruled out caching the async client.
    # Used only by check_budget_ok, which runs synchronously at the very top
    # of process_pr_review_task before any asyncio.run(...) call exists.
    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


def _new_redis_client() -> aioredis.Redis:
    # Deliberately not cached at module level: app/tasks.py runs each review
    # via its own fresh asyncio.run(...) call (a new event loop every time),
    # and a redis.asyncio client's connection is bound to whichever event
    # loop was running when it first connected -- reusing a cached client
    # across a later, different event loop fails with "Event loop is
    # closed" (reproduced directly: the second of two sequential
    # asyncio.run(record_usage(...)) calls against a real Redis silently
    # dropped its update after this exact failure). A fresh client per call,
    # closed via `async with` below, costs one extra connection setup but
    # sidesteps the problem entirely.
    return aioredis.Redis.from_url(get_settings().redis_url, decode_responses=True)


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """USD cost of a single model call, given its exact token counts."""
    pricing = _MODEL_PRICING_PER_MILLION_TOKENS.get(model)
    if pricing is None:
        logger.warning("No pricing entry for model '%s'; recording $0.00 cost for this call", model)
        return 0.0
    return (prompt_tokens * pricing["prompt"] + completion_tokens * pricing["completion"]) / 1_000_000


async def record_usage(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Atomically accumulate today's token/cost totals in Redis, keyed by UTC date.

    Best-effort: this is observability, not correctness-critical, so a Redis
    failure here is logged and swallowed rather than allowed to fail the
    review it's instrumenting (same fail-open philosophy as the GitHub
    notification helpers in app/github_client.py).
    """
    if prompt_tokens == 0 and completion_tokens == 0:
        return

    cost = calculate_cost(model, prompt_tokens, completion_tokens)
    today = datetime.now(timezone.utc).date().isoformat()
    prompt_key = f"usage:groq:prompt_tokens:{today}"
    completion_key = f"usage:groq:completion_tokens:{today}"
    cost_key = f"usage:groq:cost:{today}"

    try:
        async with _new_redis_client() as client:
            async with client.pipeline(transaction=True) as pipe:
                pipe.incrby(prompt_key, prompt_tokens)
                pipe.expire(prompt_key, _METRICS_TTL_SECONDS)
                pipe.incrby(completion_key, completion_tokens)
                pipe.expire(completion_key, _METRICS_TTL_SECONDS)
                pipe.incrbyfloat(cost_key, cost)
                pipe.expire(cost_key, _METRICS_TTL_SECONDS)
                await pipe.execute()
    except Exception:
        logger.exception("Failed to record Groq usage telemetry to Redis (model=%s)", model)


def check_budget_ok(model: str, safe_limit: int = 90_000) -> bool:
    """True if today's total Groq token usage is still under safe_limit.

    Note: the usage keys aren't dimensioned by model (see record_usage) --
    every Groq call in this system currently draws from one shared daily
    budget regardless of which specialist made it, so `model` is accepted
    here for API symmetry with calculate_cost/record_usage and for the log
    line, not used to select a different Redis key.

    Fails open: if Redis is unreachable, this returns True (budget assumed
    OK) rather than blocking every review over a telemetry outage -- the
    same fail-open philosophy record_usage applies in the other direction.
    """
    today = datetime.now(timezone.utc).date().isoformat()
    prompt_key = f"usage:groq:prompt_tokens:{today}"
    completion_key = f"usage:groq:completion_tokens:{today}"

    try:
        prompt_tokens, completion_tokens = _get_sync_redis_client().mget(prompt_key, completion_key)
    except Exception:
        logger.exception("Failed to check Groq token budget for model '%s'; assuming budget is OK", model)
        return True

    total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)
    if total_tokens >= safe_limit:
        logger.warning(
            "Token budget exhausted for model '%s': %s tokens used today (safe_limit=%s)",
            model,
            total_tokens,
            safe_limit,
        )
        return False
    return True
