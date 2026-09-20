#!/bin/bash
# Redeploy: pull the latest main and rebuild/restart every container. Run
# manually on the VM, or remotely via SSH from
# .github/workflows/deploy-oracle.yml on every push to main.
set -euo pipefail
cd "$(dirname "$0")/.."

git pull --ff-only origin main
docker compose up -d --build
docker image prune -f
