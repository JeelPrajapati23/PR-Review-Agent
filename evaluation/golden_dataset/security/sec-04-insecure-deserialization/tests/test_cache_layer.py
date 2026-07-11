from src.cache_layer import load_cached_object, save_cached_object


def test_round_trips_a_dict_through_the_cache(tmp_path):
    payload = {"report_id": 42, "rows": [1, 2, 3]}

    save_cached_object(tmp_path, "report-42", payload)
    loaded = load_cached_object(tmp_path, "report-42")

    assert loaded == payload


def test_returns_none_for_missing_key(tmp_path):
    assert load_cached_object(tmp_path, "does-not-exist") is None
