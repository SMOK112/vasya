#!/usr/bin/env bash
# Gemini не пускает с IP сервера -> пускаем ТОЛЬКО запросы к Gemini через Cloudflare WARP.
# WARP ставится в режиме локального прокси: остальной трафик сервера (SSH, бот маркета) не трогается.
# Запуск от root:  bash /opt/vasya/warp.sh
set -euo pipefail
APP=/opt/vasya
PROXY_PORT=40000

say() { echo -e "\n\033[1;33m=== $* ===\033[0m"; }
die() { echo -e "\n\033[1;31m!!! $*\033[0m"; exit 1; }
[ "$(id -u)" = 0 ] || die "Запусти от root"

if ! command -v warp-cli >/dev/null; then
  say "Ставлю Cloudflare WARP"
  export DEBIAN_FRONTEND=noninteractive
  apt-get install -y -qq curl gnupg lsb-release >/dev/null
  curl -fsSL https://pkg.cloudflareclient.com/pubkey.gpg | gpg --yes --dearmor -o /usr/share/keyrings/cloudflare-warp-archive-keyring.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloudflare-warp-archive-keyring.gpg] https://pkg.cloudflareclient.com/ $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/cloudflare-client.list
  apt-get update -qq
  apt-get install -y -qq cloudflare-warp >/dev/null
fi

systemctl enable -q --now warp-svc
sleep 3
W="warp-cli --accept-tos"

say "Настраиваю WARP как прокси (сервер в туннель НЕ уходит)"
$W disconnect >/dev/null 2>&1 || true
if ! $W registration show >/dev/null 2>&1; then
  $W disconnect >/dev/null 2>&1 || true
  $W registration new
fi
$W disconnect >/dev/null 2>&1 || true
$W mode proxy
$W proxy port $PROXY_PORT
$W connect
sleep 5

say "Проверяю WARP"
TRACE=$(curl -s --max-time 15 -x socks5h://127.0.0.1:$PROXY_PORT https://www.cloudflare.com/cdn-cgi/trace || true)
echo "$TRACE" | grep -E '^(ip|loc|warp)=' || die "WARP не отвечает. Скинь вывод: warp-cli status"

say "Проверяю Gemini через WARP"
KEY=$(grep '^GEMINI_API_KEY=' $APP/.env | cut -d= -f2-)
CODE=$(curl -s -o /tmp/gemini_test.json -w '%{http_code}' --max-time 30 -x socks5h://127.0.0.1:$PROXY_PORT \
  -H "x-goog-api-key: $KEY" -H 'Content-Type: application/json' \
  -d '{"contents":[{"parts":[{"text":"Ответь одним словом: ок"}]}]}' \
  "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent" || true)
echo "Ответ Gemini: HTTP $CODE"
if [ "$CODE" != "200" ]; then
  head -c 400 /tmp/gemini_test.json; echo
  rm -f /tmp/gemini_test.json
  die "Через WARP Gemini тоже не пустил. Скинь этот текст в чат — есть план Б."
fi
rm -f /tmp/gemini_test.json

say "Включаю прокси для Васи"
sed -i '/^GEMINI_PROXY=/d' $APP/.env
echo "GEMINI_PROXY=socks5://127.0.0.1:$PROXY_PORT" >> $APP/.env
chown vasya:vasya $APP/.env && chmod 600 $APP/.env
$APP/venv/bin/pip install -q "httpx[socks]"
systemctl restart vasya
sleep 8
journalctl -u vasya -n 6 --no-pager -o cat
echo -e "\n\033[1;32mГотово. Если выше «Gemini на связи» — открывай Васю на телефоне.\033[0m"
