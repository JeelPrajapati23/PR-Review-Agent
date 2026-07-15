import os
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

    # LangSmith tracing. On by default -- still requires langsmith_api_key to
    # actually activate (see get_settings() below), so a deployment with no
    # key configured is unaffected; this flag exists mainly as an explicit
    # kill switch if tracing should be disabled despite a key being present.
    langsmith_tracing: bool = True
    langsmith_api_key: str | None = None
    langsmith_project: str = "pr-review-agent"
    langsmith_endpoint: str | None = None


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    _apply_langsmith_env(settings)
    return settings


def _apply_langsmith_env(settings: Settings) -> None:
    # LangChain/LangSmith's tracer reads LANGCHAIN_* directly from
    # os.environ at each run's start, not from this Settings object --
    # pydantic-settings only parses .env into its own fields, it never
    # populates os.environ itself. Bridging here (get_settings() is the one
    # call every entry point -- app.main/app.worker/app.tasks -- makes
    # before touching LangGraph/ChatGroq) guarantees the env vars are set
    # before the first traced call, without needing every call site to know
    # about LangSmith.
    if not (settings.langsmith_tracing and settings.langsmith_api_key):
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
