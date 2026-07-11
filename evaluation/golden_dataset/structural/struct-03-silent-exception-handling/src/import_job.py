"""Parses uploaded CSV rows into normalized order records."""
import logging

logger = logging.getLogger(__name__)


def parse_row(raw_row: dict) -> dict | None:
    try:
        return {
            "order_id": int(raw_row["order_id"]),
            "amount": float(raw_row["amount"]),
            "customer": raw_row["customer"].strip(),
        }
    except:
        pass


def parse_rows(raw_rows: list[dict]) -> list[dict]:
    results = []
    for row in raw_rows:
        parsed = parse_row(row)
        if parsed:
            results.append(parsed)
        else:
            print("skipped a bad row")
    return results
