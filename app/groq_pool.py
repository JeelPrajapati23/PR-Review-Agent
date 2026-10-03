"""Groq key pools: key rotation, per-key TPM pacing, 429 cooldowns, and
fitting each request under the per-minute token limit.

Every Groq call the panel makes goes through RotatingChatGroq._agenerate --
specialist ReAct turns, their structured-output pass, and the Synthesizer --
so this is the single place that decides which key a call uses, waits when
every key in its pool is saturated, and shrinks a request that wouldn't fit
in one minute's budget at all.

Accounting is in-process (module-level), which matches the deployment: one
Celery worker with --concurrency=1. Groq's own 429s remain the source of
truth; the local window just avoids provoking most of them.
"""

import asyncio
import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import groq
from groq import APIConnectionError, APIStatusError, InternalServerError, RateLimitError
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_groq import ChatGroq
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 60.0
# Requests are packed to this fraction of the TPM limit: the token estimate
# is approximate, and Groq rejects (413) anything over the limit outright.
_BUDGET_SAFETY = 0.9
# Never shrink max_tokens below this to make a request fit -- gpt-oss spends
# part of its output budget on reasoning, so too little truncates the answer.
MIN_OUTPUT_TOKENS = 1500
_KEEP_TRUNCATED_CHARS = 400
_TRUNCATION_NOTE = (
    "\n...[{omitted} chars omitted to fit Groq's per-minute token limit. If you still need this "
    "content, call fetch_file_contents again with start_line/end_line for just the range you need.]\n"
)
# Fallback cooldowns when a 429 carries no parseable retry time.
_DEFAULT_TPM_COOLDOWN = 20.0
_DEFAULT_DAILY_COOLDOWN = 600.0

# Indirection so tests can fast-forward waits without patching asyncio globally.
_sleep = asyncio.sleep

_RETRY_IN_RE = re.compile(r"try again in ((?:\d+(?:\.\d+)?(?:h|ms|m|s))+)", re.IGNORECASE)
_DURATION_PART_RE = re.compile(r"(\d+(?:\.\d+)?)(h|ms|m|s)")
_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


class GroqPoolExhausted(Exception):
    """Every key in a pool stayed rate-limited past the configured max wait."""


def mask_key(key: str) -> str:
    return f"...{key[-4:]}" if len(key) > 4 else "..."


def parse_retry_after(exc: Exception) -> float | None:
    """Seconds Groq asked us to wait, from the retry-after header or the
    "Please try again in 1m2.5s" text in the error message."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        header = headers.get("retry-after")
        if header is not None:
            return float(header)
    except (TypeError, ValueError):
        pass
    match = _RETRY_IN_RE.search(str(exc))
    if not match:
        return None
    return sum(float(value) * _UNIT_SECONDS[unit] for value, unit in _DURATION_PART_RE.findall(match.group(1)))


def is_daily_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "per day" in text or "(tpd)" in text or "(rpd)" in text


@dataclass
class _KeyState:
    key: str
    # [timestamp, tokens] entries; lists so a reservation can be corrected
    # in place once the real usage is known.
    usage: deque = field(default_factory=deque)
    cooldown_until: float = 0.0

    def _prune(self, now: float) -> None:
        while self.usage and now - self.usage[0][0] >= _WINDOW_SECONDS:
            self.usage.popleft()

    def used(self, now: float) -> int:
        self._prune(now)
        return sum(tokens for _, tokens in self.usage)

    def ready_at(self, need: int, limit: int, now: float) -> float:
        """Earliest time this key could take a `need`-token request."""
        start = max(now, self.cooldown_until)
        used = self.used(now)
        for timestamp, tokens in self.usage:
            if used + need <= limit:
                break
            used -= tokens
            start = max(start, timestamp + _WINDOW_SECONDS)
        return start


# Shared across every pool and call in this process, so a key's window
# carries over between back-to-back reviews (and between pools, if a key
# were ever listed in two of them).
_KEY_STATES: dict[str, _KeyState] = {}
_POOLS: dict[tuple, "GroqKeyPool"] = {}


class GroqKeyPool:
    def __init__(self, name: str, keys: list[str], tpm_limit: int, max_wait_seconds: float):
        if not keys:
            raise ValueError(f"Groq key pool '{name}' has no keys")
        self.name = name
        self.tpm_limit = tpm_limit
        self.max_wait_seconds = max_wait_seconds
        self._states = [_KEY_STATES.setdefault(key, _KeyState(key)) for key in keys]
        self._cursor = 0

    @property
    def keys(self) -> list[str]:
        return [state.key for state in self._states]

    async def acquire(self, tokens: int) -> tuple[_KeyState, list]:
        """Reserve `tokens` on the next key (round-robin) with room in its
        rolling minute, sleeping until one frees up if none has room."""
        tokens = min(tokens, self.tpm_limit)
        deadline = time.monotonic() + self.max_wait_seconds
        while True:
            now = time.monotonic()
            count = len(self._states)
            for offset in range(count):
                index = (self._cursor + offset) % count
                state = self._states[index]
                if state.cooldown_until <= now and state.used(now) + tokens <= self.tpm_limit:
                    entry = [now, tokens]
                    state.usage.append(entry)
                    self._cursor = (index + 1) % count
                    return state, entry
            wake = min(state.ready_at(tokens, self.tpm_limit, now) for state in self._states)
            if wake > deadline:
                raise GroqPoolExhausted(
                    f"All {count} key(s) in Groq pool '{self.name}' are rate-limited for another "
                    f"{wake - now:.0f}s, past the {self.max_wait_seconds:.0f}s max wait"
                )
            delay = max(wake - now, 0.25)
            logger.info(
                "Groq pool '%s': all keys saturated for a %s-token request; waiting %.1fs",
                self.name, tokens, delay,
            )
            await _sleep(delay)

    def settle(self, entry: list, actual_tokens: int) -> None:
        entry[1] = actual_tokens

    def on_rate_limited(self, state: _KeyState, entry: list, exc: RateLimitError) -> None:
        daily = is_daily_limit(exc)
        cooldown = parse_retry_after(exc) or (_DEFAULT_DAILY_COOLDOWN if daily else _DEFAULT_TPM_COOLDOWN)
        state.cooldown_until = max(state.cooldown_until, time.monotonic() + cooldown)
        entry[1] = 0
        logger.warning(
            "Groq pool '%s': key %s hit a %s rate limit; cooling down %.1fs and rotating",
            self.name, mask_key(state.key), "daily" if daily else "per-minute", cooldown,
        )


def get_pool(name: str, keys: list[str], tpm_limit: int, max_wait_seconds: float) -> GroqKeyPool:
    """Pools are cached so the round-robin cursor persists across calls."""
    cache_key = (name, tuple(keys), tpm_limit, max_wait_seconds)
    pool = _POOLS.get(cache_key)
    if pool is None:
        pool = _POOLS[cache_key] = GroqKeyPool(name, keys, tpm_limit, max_wait_seconds)
    return pool


# ---------------------------------------------------------------------------
# Token estimation and request fitting
# ---------------------------------------------------------------------------

# Characters per token, calibrated from each response's real prompt_tokens.
# Starts conservative (code tokenizes densely) so early estimates overshoot.
_chars_per_token = 3.0


def estimate_tokens(chars: int) -> int:
    return int(chars / _chars_per_token) + 1


def _calibrate(chars: int, prompt_tokens: int) -> None:
    global _chars_per_token
    if prompt_tokens <= 0 or chars <= 0:
        return
    observed = chars / prompt_tokens
    # Lean slightly below the observation so estimates stay on the safe side.
    _chars_per_token = min(max(0.7 * _chars_per_token + 0.3 * observed * 0.95, 2.2), 4.5)


def _content_chars(message: dict) -> int:
    return len(json.dumps(message, default=str))


def _truncate(text: str, keep_chars: int) -> str:
    if len(text) <= keep_chars + len(_TRUNCATION_NOTE):
        return text
    head = keep_chars * 2 // 3
    tail = keep_chars - head
    omitted = len(text) - head - tail
    return text[:head] + _TRUNCATION_NOTE.format(omitted=omitted) + (text[-tail:] if tail else "")


def fit_request(
    message_dicts: list[dict], extra_chars: int, desired_output: int, budget: int
) -> tuple[list[dict], int, int]:
    """Shrink a request until input + max_tokens fits in `budget` tokens.

    Groq counts a request's max_tokens against the TPM limit up front, so
    the cheap lever comes first: lower max_tokens (down to
    MIN_OUTPUT_TOKENS). Only then is context trimmed -- older tool results
    first, then the most recent ones, then earlier assistant turns -- never
    the system prompt or the task message. Returns (messages, max_tokens,
    estimated total tokens). Never mutates the caller's messages.
    """
    messages = [dict(message) for message in message_dicts]

    def input_tokens() -> int:
        return estimate_tokens(sum(_content_chars(m) for m in messages) + extra_chars)

    est_input = input_tokens()
    if est_input + desired_output <= budget:
        return messages, desired_output, est_input + desired_output
    if est_input + MIN_OUTPUT_TOKENS <= budget:
        max_output = budget - est_input
        return messages, max_output, est_input + max_output

    last_non_tool = max(
        (i for i, m in enumerate(messages) if m.get("role") != "tool"), default=-1
    )
    older_tools = [i for i, m in enumerate(messages) if m.get("role") == "tool" and i < last_non_tool]
    latest_tools = [i for i, m in enumerate(messages) if m.get("role") == "tool" and i > last_non_tool]
    older_assistant = [
        i for i, m in enumerate(messages)
        if m.get("role") == "assistant" and i < last_non_tool and isinstance(m.get("content"), str)
    ]

    def over_by() -> int:
        return input_tokens() + MIN_OUTPUT_TOKENS - budget

    for index in older_tools + older_assistant:
        if over_by() <= 0:
            break
        content = messages[index].get("content")
        if isinstance(content, str):
            messages[index]["content"] = _truncate(content, _KEEP_TRUNCATED_CHARS)

    # Latest tool results are what the model is about to reason over, so
    # they only lose what's needed, split evenly across them.
    if latest_tools and over_by() > 0:
        excess_chars = int(over_by() * _chars_per_token) + 1
        # Plus room for the truncation note each trimmed message gains.
        per_message = excess_chars // len(latest_tools) + len(_TRUNCATION_NOTE) + 32
        for index in latest_tools:
            content = messages[index].get("content")
            if isinstance(content, str):
                keep = max(len(content) - per_message, _KEEP_TRUNCATED_CHARS)
                messages[index]["content"] = _truncate(content, keep)

    est_input = input_tokens()
    max_output = max(MIN_OUTPUT_TOKENS, min(desired_output, budget - est_input))
    if est_input + max_output > budget:
        logger.warning(
            "Groq request still ~%s tokens over budget %s after trimming context",
            est_input + max_output - budget, budget,
        )
    return messages, max_output, est_input + max_output


# ---------------------------------------------------------------------------
# The chat model
# ---------------------------------------------------------------------------


class RotatingChatGroq(ChatGroq):
    """ChatGroq that spreads calls across a key pool.

    Overrides only _agenerate, the one method every async call path reaches
    (bind_tools and with_structured_output both wrap this same instance), so
    create_react_agent and the Synthesizer use it unchanged. Async-only:
    streaming is disabled and the sync path is refused rather than silently
    bypassing rotation on the constructor's first key.
    """

    _pool: GroqKeyPool = PrivateAttr()
    _clients: dict = PrivateAttr(default_factory=dict)

    @classmethod
    def from_pool(cls, pool: GroqKeyPool, **kwargs: Any) -> "RotatingChatGroq":
        # max_retries=0: the SDK's own retry would sleep and retry a 429 on
        # the same key instead of rotating to another one.
        model = cls(api_key=pool.keys[0], max_retries=0, disable_streaming=True, **kwargs)
        model._pool = pool
        return model

    def _client_for(self, key: str):
        # Per-instance (i.e. per review) clients, never module-level: an
        # httpx async client is bound to the event loop it first ran on, and
        # each review runs in its own asyncio.run().
        client = self._clients.get(key)
        if client is None:
            client = self._clients[key] = groq.AsyncGroq(
                api_key=key,
                base_url=self.groq_api_base,
                timeout=self.request_timeout,
                max_retries=0,
            ).chat.completions
        return client

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError("RotatingChatGroq is async-only; use ainvoke")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        pool = self._pool
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        desired_output = params.get("max_tokens") or 4000
        extra_chars = len(json.dumps(params.get("tools") or [], default=str))

        shrink = 1.0
        transient_failures = 0
        max_attempts = max(12, 4 * len(pool.keys))
        for _ in range(max_attempts):
            budget = int(pool.tpm_limit * _BUDGET_SAFETY * shrink)
            fitted, max_output, estimate = fit_request(message_dicts, extra_chars, desired_output, budget)
            state, entry = await pool.acquire(estimate)
            try:
                response = await self._client_for(state.key).create(
                    messages=fitted, **{**params, "max_tokens": max_output}
                )
            except RateLimitError as exc:
                pool.on_rate_limited(state, entry, exc)
                continue
            except APIStatusError as exc:
                entry[1] = 0
                if exc.status_code != 413:
                    raise
                # Request too large for the TPM limit: our estimate was low.
                shrink *= 0.8
                logger.warning("Groq pool '%s': request too large; retrying at %.0f%% budget", pool.name, shrink * 100)
                continue
            except (APIConnectionError, InternalServerError):
                entry[1] = 0
                transient_failures += 1
                if transient_failures > 2:
                    raise
                await _sleep(2 * transient_failures)
                continue

            result = self._create_chat_result(response, params)
            usage = (result.llm_output or {}).get("token_usage") or {}
            prompt_tokens = usage.get("prompt_tokens") or 0
            pool.settle(entry, prompt_tokens + (usage.get("completion_tokens") or 0) or estimate)
            _calibrate(sum(_content_chars(m) for m in fitted) + extra_chars, prompt_tokens)
            return result

        raise GroqPoolExhausted(f"Groq pool '{pool.name}' gave up after {max_attempts} attempts")
