# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

# git: app/git_ops.py shells out to the `git` binary directly for every PR
# checkout (clone/fetch/checkout/reset/clean) -- there is no Python git
# library involved, so this is a hard runtime dependency, not a build-only one.
# ca-certificates: needed for git/uv/httpx to make outbound HTTPS calls
# (GitHub, Groq, PyPI) without TLS verification failures.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ghcr.io/astral-sh/uv ships the uv/uvx binaries pre-built; copying them
# straight out of that image is faster and more reproducible than installing
# uv via pip inside this image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Dedicated non-root user. Two independent reasons this matters more than it
# would for a typical service:
#   1. celery_worker executes a PR's own pytest suite via
#      mcp_servers/tester_server.py -- untrusted code by design (env vars are
#      already stripped for that subprocess; see tester_server.py). Running
#      the whole container as non-root from the start means that subprocess
#      never has more privilege than "nobody" would grant it anyway, without
#      depending on tester_server.py's own runtime setuid dance actually
#      succeeding.
#   2. It lets docker-compose.yml's `cap_drop: [ALL]` on celery_worker apply
#      cleanly. A process that never needs to change UID at runtime doesn't
#      need CAP_SETUID/CAP_SETGID, so dropping every Linux capability doesn't
#      break anything -- whereas a root (UID 0) process stripped of those two
#      specific capabilities would fail every time tester_server.py's own
#      subprocess.run(..., user="nobody") tries to drop privileges, silently
#      turning every PR review's test-suite validation into a permission
#      error. Starting unprivileged sidesteps that conflict entirely.
RUN groupadd --system appuser \
    && useradd --system --create-home --home-dir /home/appuser --gid appuser --shell /usr/sbin/nologin appuser

# Dependencies first (separate layer from app code) so an app-only change
# doesn't invalidate the dependency-install cache layer on rebuild.
COPY requirements.lock ./
RUN uv pip install --system --no-cache -r requirements.lock

# Only the two packages actually needed at runtime (per CLAUDE.md: app/ and
# mcp_servers/ are the runtime pieces; evaluation/ and tests/ are dev-only)
# -- keeps the image smaller and out of the untrusted-code execution path.
COPY app ./app
COPY mcp_servers ./mcp_servers

# git_workspace_root (GIT_WORKSPACE_ROOT env var, see docker-compose.yml) --
# created and owned by appuser up front so a PR checkout, running as
# appuser, can write to it without any root step in the runtime path.
RUN mkdir -p /app/workspace && chown -R appuser:appuser /app

# Single-token launcher scripts, used only by the Azure Container Apps
# deployment: Azure's `--command`/`--args` CLI flags can't reliably pass
# tokens starting with `-` (e.g. `--host`), so each Container App points
# `--command` at one of these instead of overriding command/args directly.
# docker-compose.yml is unaffected -- it still sets its own `command:`.
COPY docker/entrypoint-web.sh docker/entrypoint-worker.sh ./
RUN chmod +x entrypoint-web.sh entrypoint-worker.sh

USER appuser

EXPOSE 8000

# No default CMD: fastapi_web and celery_worker share this one image and
# supply their own `command:` in docker-compose.yml (or `--command` in
# Azure Container Apps).
