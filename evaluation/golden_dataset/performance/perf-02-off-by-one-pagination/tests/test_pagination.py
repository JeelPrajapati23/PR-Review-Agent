from src.pagination import get_page


def test_first_page_starts_at_the_first_item():
    items = list(range(1, 11))  # [1..10]

    page = get_page(items, page_number=1, page_size=5)

    assert page[0] == 1
    assert len(page) <= 5


def test_second_page_starts_after_the_first_page_size_items():
    items = list(range(1, 11))

    page = get_page(items, page_number=2, page_size=5)

    assert page[0] == 6


def test_page_beyond_the_data_is_empty():
    items = list(range(1, 6))

    page = get_page(items, page_number=10, page_size=5)

    assert page == []
