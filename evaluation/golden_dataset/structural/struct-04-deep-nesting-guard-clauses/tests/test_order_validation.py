from src.order_validation import validate_order


def test_valid_order_returns_none():
    order = {
        "customer_id": "cust-1",
        "items": [{"sku": "abc"}],
        "shipping_address": "123 Main St",
    }

    assert validate_order(order) is None


def test_missing_customer_id_is_rejected():
    order = {"items": [{"sku": "abc"}], "shipping_address": "123 Main St"}

    assert validate_order(order) == "Missing customer id"


def test_empty_order_is_rejected():
    assert validate_order(None) == "Order is empty"
