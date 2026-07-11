from src.pricing_cache import get_discounted_price


def _flat_ten_percent_off(segment: str, base_price: float) -> float:
    return base_price * 0.9


def test_applies_the_discount_function_for_a_new_segment():
    price = get_discounted_price("gold", 100.0, _flat_ten_percent_off)

    assert price == 90.0


def test_reuses_the_cached_result_for_the_same_segment():
    first = get_discounted_price("silver", 50.0, _flat_ten_percent_off)
    second = get_discounted_price("silver", 50.0, _flat_ten_percent_off)

    assert first == second == 45.0
