from src.checkout_pricing import calc


def test_applies_standard_pricing_for_a_small_order():
    total = calc(100, 5, 2)

    assert total == round(100 * 0.925 * 1.13 + 5, 2)


def test_applies_the_bulk_discount_over_the_threshold():
    total = calc(100, 5, 50)

    assert total == round(100 * 0.925 * 1.13 - 5 + 5, 2)
