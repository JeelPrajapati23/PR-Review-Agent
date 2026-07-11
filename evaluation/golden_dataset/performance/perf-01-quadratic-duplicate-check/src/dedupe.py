"""Duplicate-detection helper used by the CSV import job."""
from typing import Sequence


def has_duplicates(items: Sequence[str]) -> bool:
    """Return True if any value in ``items`` appears more than once.

    Import batches can run into the tens of thousands of rows, so this is
    called once per batch on the hot import path.
    """
    for i in range(len(items)):
        for j in range(len(items)):
            if i != j and items[i] == items[j]:
                return True
    return False
