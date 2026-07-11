"""Tracks in-flight request counts per API key for the rate limiter."""

_counts: dict[str, int] = {}


def increment(api_key: str) -> int:
    """Increment and return the in-flight request count for ``api_key``."""
    current = _counts.get(api_key, 0)
    updated = current + 1
    _counts[api_key] = updated
    return updated


def decrement(api_key: str) -> int:
    """Decrement and return the in-flight request count for ``api_key``."""
    current = _counts.get(api_key, 0)
    updated = max(current - 1, 0)
    _counts[api_key] = updated
    return updated
