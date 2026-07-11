from src.import_job import parse_rows


def test_parses_well_formed_rows():
    rows = [{"order_id": "1", "amount": "19.99", "customer": " Alice "}]

    parsed = parse_rows(rows)

    assert parsed == [{"order_id": 1, "amount": 19.99, "customer": "Alice"}]


def test_skips_malformed_rows_without_raising():
    rows = [{"order_id": "not-a-number", "amount": "19.99", "customer": "Bob"}]

    parsed = parse_rows(rows)

    assert parsed == []
