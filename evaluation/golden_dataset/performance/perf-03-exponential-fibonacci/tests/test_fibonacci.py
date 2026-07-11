from src.fibonacci import fibonacci


def test_base_cases():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1


def test_small_values():
    assert fibonacci(5) == 5
    assert fibonacci(10) == 55
