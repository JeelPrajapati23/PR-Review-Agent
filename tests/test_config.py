import os

os.environ.setdefault("GITHUB_WEBHOOK_SECRET", "test-secret")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("GITHUB_APP_ID", "12345")
os.environ.setdefault("GITHUB_APP_PRIVATE_KEY_B64", "test-private-key-b64")

from app.config import Settings


def _settings(**overrides) -> Settings:
    base = {
        "github_webhook_secret": "s",
        "groq_api_key": "main-key",
        "github_app_id": "1",
        "github_app_private_key_b64": "k",
        "groq_api_keys": None,
        "groq_api_key_security": None,
        "groq_api_key_performance": None,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def test_unset_pools_fall_back_to_main_key_and_stay_sequential():
    settings = _settings()
    for role in ("security", "performance", "synthesizer"):
        assert settings.groq_key_pool(role) == ["main-key"]
    assert settings.specialists_run_in_parallel is False


def test_comma_separated_pools_are_split_trimmed_and_deduplicated():
    settings = _settings(
        groq_api_keys="s1, s2,s3,,s1",
        groq_api_key_security="a1,a2,a3",
        groq_api_key_performance=" b1 , b2,b3 ",
    )
    assert settings.groq_key_pool("synthesizer") == ["s1", "s2", "s3"]
    assert settings.groq_key_pool("security") == ["a1", "a2", "a3"]
    assert settings.groq_key_pool("performance") == ["b1", "b2", "b3"]
    assert settings.specialists_run_in_parallel is True


def test_overlapping_specialist_pools_stay_sequential():
    settings = _settings(groq_api_key_security="a1,shared", groq_api_key_performance="shared,b1")
    assert settings.specialists_run_in_parallel is False


def test_one_specialist_pool_set_runs_parallel_against_main_key():
    settings = _settings(groq_api_key_performance="b1,b2")
    assert settings.groq_key_pool("security") == ["main-key"]
    assert settings.specialists_run_in_parallel is True
