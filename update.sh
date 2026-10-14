#!/usr/bin/env bash
# Обновить Васю до свежей версии с GitHub:  bash /opt/vasya/update.sh
set -euo pipefail
cd /opt/vasya
git config --global --add safe.directory /opt/vasya 2>/dev/null || true
git pull --ff-only
venv/bin/pip install -q -r requirements.txt
chown -R vasya:vasya /opt/vasya
chmod 600 .env
systemctl restart vasya
sleep 3
systemctl --no-pager -n 8 status vasya | tail -n 8
echo "Обновлено. Логи: journalctl -u vasya -f"
