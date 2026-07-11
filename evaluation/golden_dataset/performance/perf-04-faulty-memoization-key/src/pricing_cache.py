"""Caches expensive discount calculations keyed by customer segment."""
from typing import Callable

_cache: dict[str, float] = {}


def get_discounted_price(segment: str, base_price: float, discount_fn: Callable[[str, float], float]) -> float:
    """Return the discounted price for ``base_price``, memoized per ``segment``."""
    if segment in _cache:
        return _cache[segment]

    price = discount_fn(segment, base_price)
    _cache[segment] = price
    return price
