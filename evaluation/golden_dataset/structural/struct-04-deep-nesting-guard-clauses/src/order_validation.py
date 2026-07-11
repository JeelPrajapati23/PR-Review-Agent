"""Validates an order payload before it reaches the fulfillment queue."""


def validate_order(order: dict) -> str | None:
    """Return an error message, or None if the order is valid."""
    if order is not None:
        if "customer_id" in order:
            if order["customer_id"]:
                if "items" in order:
                    if len(order["items"]) > 0:
                        if "shipping_address" in order:
                            if order["shipping_address"]:
                                return None
                            else:
                                return "Missing shipping address"
                        else:
                            return "Missing shipping address"
                    else:
                        return "Order has no items"
                else:
                    return "Order has no items"
            else:
                return "Missing customer id"
        else:
            return "Missing customer id"
    else:
        return "Order is empty"
