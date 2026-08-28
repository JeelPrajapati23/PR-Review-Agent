#!/bin/sh
# ${PORT:-8000}: Render injects its own PORT for web services and expects
# the container to listen on it (see render.yaml's PORT env var); falls
# back to 8000 for docker-compose/local use, where nothing sets PORT.
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
