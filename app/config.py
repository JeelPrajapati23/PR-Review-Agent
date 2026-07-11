from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    github_webhook_secret: str
    git_workspace_root: str = "app/workspace"
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    # GitHub App auth (replaces the old static PAT): app_id + private key are
    # exchanged for a short-lived, per-installation access token at call time
    # (see app/github_client.py) rather than used directly against the API.
    # The private key is stored base64-encoded since its raw PEM form (real
    # newlines) doesn't survive being passed through .env/shell/CLI secrets
    # cleanly.
    github_app_id: str
    github_app_private_key_b64: str


@lru_cache
def get_settings() -> Settings:
    return Settings()
