#!/usr/bin/env bash
# Показать ссылку для телефона (и QR-код — наведи камеру телефона на экран компа)
APP=/opt/vasya
TOKEN=$(grep '^VASYA_TOKEN=' $APP/.env | cut -d= -f2-)
IP=$(curl -fs4 --max-time 10 https://api.ipify.org || hostname -I | awk '{print $1}')
URL="https://${IP//./-}.sslip.io/?k=$TOKEN"
command -v qrencode >/dev/null || apt-get install -y -qq qrencode >/dev/null 2>&1 || true
echo
command -v qrencode >/dev/null && qrencode -t ansiutf8 -m 2 "$URL"
echo
echo "Ссылка:  $URL"
echo "Ключ:    $TOKEN"
echo
