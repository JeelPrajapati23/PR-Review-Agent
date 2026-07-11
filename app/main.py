import hashlib
import hmac
import json
import logging
from functools import lru_cache

import redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.config import get_settings
from app.schemas import PullRequestEvent
from app.tasks import process_pr_review_task

logger = logging.getLogger(__name__)

# docs_url/redoc_url/openapi_url disabled: this app has exactly one route (a
# webhook receiver), so there's nothing for interactive API docs to usefully
# document, and no reason to expose even a low-risk extra surface on an
# internet-facing service.
app = FastAPI(title="PR Review Agent", docs_url=None, redoc_url=None, openapi_url=None)

# Actions that represent an actual code change worth reviewing. Everything
# else (review_requested, labeled, closed, etc.) still arrives as a
# pull_request event but must not enqueue a duplicate Celery review.
_PROCESSABLE_ACTIONS = {"opened", "synchronize"}

# TTL for the dedup lock below. Bounds how long a crashed/stuck review can
# block a legitimate re-review of the same commit, while still covering the
# burst of near-simultaneous redeliveries GitHub sends when this endpoint is
# briefly slow to respond.
_DEDUP_LOCK_TTL_SECONDS = 15 * 60

# Real GitHub PR webhook payloads are well under 1MB even for large PRs (the
# body carries metadata only, never diffs/file contents). This endpoint is
# unauthenticated until _verify_signature runs, and that check needs the raw
# body bytes -- so without a cap, any client can force unbounded memory use
# per request just by POSTing a large body, before the signature is ever
# checked. 5MB gives generous headroom over real payloads while bounding the
# worst case.
_MAX_WEBHOOK_BODY_BYTES = 5 * 1024 * 1024


def _content_length_exceeds_limit(request: Request) -> bool:
    content_length = request.headers.get("content-length")
    if content_length is None:
        return False
    try:
        return int(content_length) > _MAX_WEBHOOK_BODY_BYTES
    except ValueError:
        return False


@lru_cache
def _get_redis_client() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


def _verify_signature(secret: str, payload: bytes, signature_header: str | None) -> bool:
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    received = signature_header.removeprefix("sha256=")
    return hmac.compare_digest(expected, received)


@app.post("/webhook/github")
async def github_webhook(request: Request) -> JSONResponse:
    settings = get_settings()

    # Fast path: reject an oversized request before buffering any of it.
    if _content_length_exceeds_limit(request):
        raise HTTPException(status_code=413, detail="Payload too large")

    raw_payload = await request.body()
    # Belt-and-suspenders: a missing/absent/lying Content-Length header (e.g.
    # chunked transfer-encoding) would skip the check above entirely, so also
    # enforce the cap on what was actually read.
    if len(raw_payload) > _MAX_WEBHOOK_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Payload too large")

    signature_header = request.headers.get("X-Hub-Signature-256")
    if not _verify_signature(settings.github_webhook_secret, raw_payload, signature_header):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    github_event = request.headers.get("X-GitHub-Event")
    if github_event != "pull_request":
        return JSONResponse(status_code=200, content={"message": f"Ignored event: {github_event}"})

    try:
        payload = json.loads(raw_payload)
        event = PullRequestEvent.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid payload: {exc}") from exc

    if event.action not in _PROCESSABLE_ACTIONS:
        logger.info("Skipping pull_request event with action=%s", event.action)
        return JSONResponse(status_code=200, content={"message": f"Ignored action: {event.action}"})

    # Dedup lock: GitHub redelivers a webhook whenever this endpoint is slow
    # or replies non-2xx, and each redelivery would otherwise queue a second
    # full review -- burning another review's worth of the tight Groq token
    # budget -- for a commit snapshot already being analyzed.
    lock_key = f"lock:review:{event.repository.full_name}:{event.pull_request.number}:{event.pull_request.head.sha}"
    lock_acquired = _get_redis_client().set(lock_key, "1", nx=True, ex=_DEDUP_LOCK_TTL_SECONDS)
    if not lock_acquired:
        logger.info("Duplicate webhook redelivery for %s; a review is already in flight", lock_key)
        return JSONResponse(status_code=200, content={"message": "Duplicate webhook redelivery ignored"})

    process_pr_review_task.delay(event.model_dump())

    return JSONResponse(status_code=202, content={"message": "PR review task queued"})
