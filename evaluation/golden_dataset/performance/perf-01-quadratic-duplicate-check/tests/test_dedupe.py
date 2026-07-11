from src.dedupe import has_duplicates


def test_detects_a_duplicate_value():
    assert has_duplicates(["a", "b", "a"]) is True


def test_returns_false_for_all_unique_values():
    assert has_duplicates(["a", "b", "c"]) is False


def test_empty_list_has_no_duplicates():
    assert has_duplicates([]) is False
