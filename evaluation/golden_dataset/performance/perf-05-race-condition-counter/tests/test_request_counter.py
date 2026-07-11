from src.request_counter import decrement, increment


def test_increment_returns_the_new_count():
    assert increment("key-a") == 1
    assert increment("key-a") == 2


def test_decrement_never_goes_below_zero():
    assert decrement("key-b") == 0


def test_decrement_reduces_an_existing_count():
    increment("key-c")
    increment("key-c")

    assert decrement("key-c") == 1
