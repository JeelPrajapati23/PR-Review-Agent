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
    github_api_token: str


@lru_cache
def get_settings() -> Settings:
    return Settings()
