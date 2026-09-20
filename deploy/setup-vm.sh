#!/bin/bash
# One-time provisioning for a fresh Oracle Cloud "Always Free" Ubuntu VM.
# Run this once over SSH, after the instance is up, its public IP is
# reachable, and the OCI console's Security List/NSG already allows inbound
# TCP 22/80/443 (that side is console-only -- this script can't touch it).
set -euo pipefail

REPO_URL="https://github.com/JeelPrajapati23/PR-Review-Agent.git"
APP_DIR="$HOME/pr-review-agent"

# Docker Engine + the Compose plugin, per Docker's official apt repo.
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"

# Oracle's stock Ubuntu image ships iptables rules that DROP everything but
# SSH by default, independently of the OCI console's own Security List --
# both layers must allow a port, not just one. Inserted at the very top of
# INPUT so they take effect regardless of the stock rule ordering below them.
sudo iptables -I INPUT 1 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 1 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save

git clone "$REPO_URL" "$APP_DIR"
cd "$APP_DIR"
cp .env.example .env

cat <<EOF

Next steps:
  1. Edit $APP_DIR/.env with real secrets (GITHUB_WEBHOOK_SECRET, GROQ_API_KEY,
     GITHUB_APP_ID, GITHUB_APP_PRIVATE_KEY_B64, DOMAIN, etc).
  2. Log out and back in (or run 'newgrp docker') so your shell picks up the
     docker group membership just added.
  3. Run: cd $APP_DIR && ./deploy/deploy.sh
EOF
