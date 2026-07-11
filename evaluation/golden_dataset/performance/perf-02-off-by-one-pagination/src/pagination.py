"""Slices a result set into pages for the API list endpoints."""
from typing import Sequence


def get_page(items: Sequence, page_number: int, page_size: int) -> list:
    """Return the ``page_number``-th page (1-indexed) of ``items``."""
    start = (page_number - 1) * page_size
    end = start + page_size - 1
    return list(items[start:end])
