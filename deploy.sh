#!/usr/bin/env bash
# Установка Васи на сервер (Ubuntu/Debian), запускать от root:  bash deploy.sh
# Можно запускать повторно — ничего не сломает, просто обновит.
set -euo pipefail

APP=/opt/vasya
PORT=8080
cd "$(dirname "$0")"
SRC="$(pwd)"

say() { echo -e "\n\033[1;33m=== $* ===\033[0m"; }
die() { echo -e "\n\033[1;31m!!! $*\033[0m"; exit 1; }

[ "$(id -u)" = 0 ] || die "Запусти от root"
command -v apt-get >/dev/null || die "Нужен Ubuntu или Debian"

say "Ставлю системные пакеты"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip curl unzip git openssl ca-certificates >/dev/null

python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
  || die "Нужен Python 3.10+, а тут $(python3 -V). Скинь это в чат."

# ---------- код в /opt/vasya
if [ "$SRC" != "$APP" ]; then
  say "Копирую код в $APP"
  mkdir -p "$APP"
  cp -r "$SRC"/. "$APP"/
fi
cd "$APP"

id vasya >/dev/null 2>&1 || useradd --system --home "$APP" --shell /usr/sbin/nologin vasya

# ---------- подкачка: памяти мало, чтобы Вася и бот маркета не дрались за неё
if [ -z "$(swapon --show --noheadings)" ] && [ "$(df --output=avail -BG / | tail -1 | tr -dc 0-9)" -ge 3 ]; then
  say "Делаю подкачку 1 ГБ (памяти на сервере мало)"
  fallocate -l 1G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# ---------- Python-окружение
say "Ставлю библиотеки (пару минут)"
[ -d venv ] || python3 -m venv venv
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q -r requirements.txt

# ---------- модель для имени
if [ ! -d models/vosk-model-small-ru-0.22 ]; then
  say "Качаю модель для имени «Вась» (45 МБ)"
  curl -fL --retry 3 -o /tmp/vosk-ru.zip https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip
  unzip -q -o /tmp/vosk-ru.zip -d models && rm -f /tmp/vosk-ru.zip
fi

# ---------- адрес: IP сервера -> 1-2-3-4.sslip.io (домен покупать не надо)
IP=$(curl -fs4 --max-time 10 https://api.ipify.org || hostname -I | awk '{print $1}')
DOMAIN="${IP//./-}.sslip.io"

# ---------- секреты
if [ ! -f .env ]; then
  say "Нужен ключ Gemini"
  echo "Вставь ключ из https://aistudio.google.com/apikey (его не будет видно при вставке) и нажми Enter:"
  read -rs KEY; echo
  [ -n "$KEY" ] || die "Ключ пустой"
  TOKEN=$(openssl rand -hex 12)
  umask 077
  printf 'GEMINI_API_KEY=%s\nVASYA_TOKEN=%s\nVASYA_PORT=%s\nVASYA_HOST=127.0.0.1\n' "$KEY" "$TOKEN" "$PORT" > .env
  umask 022
fi
TOKEN=$(grep '^VASYA_TOKEN=' .env | cut -d= -f2-)
chown -R vasya:vasya "$APP"
chmod 600 .env

# ---------- автозапуск
say "Настраиваю автозапуск"
cat > /etc/systemd/system/vasya.service <<EOF
[Unit]
Description=Vasya voice assistant
After=network-online.target
Wants=network-online.target

[Service]
User=vasya
WorkingDirectory=$APP
EnvironmentFile=$APP/.env
ExecStart=$APP/venv/bin/python -u server.py
Restart=always
RestartSec=3
MemoryMax=700M

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable -q vasya
systemctl restart vasya

# ---------- HTTPS: телефон даёт микрофон только по https
say "Настраиваю HTTPS для $DOMAIN"
BUSY=$(ss -ltnpH '( sport = :80 or sport = :443 )' 2>/dev/null || true)
if echo "$BUSY" | grep -q nginx; then
  echo "Порты 80/443 держит nginx — подключаю Васю через него"
  apt-get install -y -qq certbot python3-certbot-nginx >/dev/null
  cat > /etc/nginx/sites-available/vasya <<EOF
server {
    listen 80;
    server_name $DOMAIN;
    client_max_body_size 20m;
    location / {
        proxy_pass http://127.0.0.1:$PORT;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Forwarded-For \$remote_addr;
        proxy_read_timeout 3600s;
    }
}
EOF
  ln -sf /etc/nginx/sites-available/vasya /etc/nginx/sites-enabled/vasya
  nginx -t && systemctl reload nginx
  certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --register-unsafely-without-email --redirect
elif [ -n "$(echo "$BUSY" | grep -v caddy)" ]; then
  echo "$BUSY"
  die "Порты 80/443 заняты не nginx и не caddy. Скинь этот текст в чат — разберёмся."
else
  if ! command -v caddy >/dev/null; then
    apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https gnupg >/dev/null
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' > /etc/apt/sources.list.d/caddy-stable.list
    apt-get update -qq && apt-get install -y -qq caddy >/dev/null
  fi
  CF=/etc/caddy/Caddyfile
  # стандартный файл-заглушку из пакета заменяем, свой не трогаем
  if [ ! -f "$CF" ] || grep -q 'root \* /usr/share/caddy' "$CF"; then : > "$CF"; fi
  if ! grep -q "$DOMAIN" "$CF"; then
    cat >> "$CF" <<EOF

$DOMAIN {
    encode gzip
    reverse_proxy 127.0.0.1:$PORT
}
EOF
  fi
  caddy validate --config "$CF" --adapter caddyfile >/dev/null
  systemctl enable -q caddy
  systemctl reload caddy 2>/dev/null || systemctl restart caddy
fi

if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then
  ufw allow 80/tcp >/dev/null; ufw allow 443/tcp >/dev/null
fi

sleep 4
say "Проверка"
systemctl --no-pager -n 12 status vasya | tail -n 12 || true

echo
echo -e "\033[1;32mГотово! Открой на телефоне в Chrome (можно через мобильный интернет):\033[0m"
echo
echo "    https://$DOMAIN/?k=$TOKEN"
echo
echo "Ссылку с ключом никому не давай. Телефон запомнит ключ, потом хватит https://$DOMAIN"
echo "QR-код и ссылка ещё раз:  bash $APP/link.sh"
echo "Логи Васи:   journalctl -u vasya -f"
echo "Обновить:    bash $APP/update.sh"
