#!/usr/bin/env bash
# Подключить Polza.ai (оплата в рублях, без лимитов Gemini):  bash /opt/vasya/polza.sh
set -euo pipefail
APP=/opt/vasya
[ "$(id -u)" = 0 ] || { echo "Запусти от root"; exit 1; }

echo "Вставь API-ключ Polza.ai (polza.ai → Консоль → API-ключи). Его не будет видно при вставке. Потом Enter:"
read -rs KEY; echo
[ -n "$KEY" ] || { echo "Ключ пустой"; exit 1; }

CODE=$(curl -s -o /tmp/polza_test.json -w '%{http_code}' --max-time 20 \
  -H "Authorization: Bearer $KEY" "https://polza.ai/api/v1/models?type=chat" || true)
if [ "$CODE" != "200" ]; then
  echo "Polza ответила HTTP $CODE:"; head -c 300 /tmp/polza_test.json; echo
  rm -f /tmp/polza_test.json
  echo "Ключ не подошёл или Polza недоступна — проверь ключ и попробуй ещё раз."; exit 1
fi
rm -f /tmp/polza_test.json
echo "Ключ рабочий."

sed -i '/^POLZA_API_KEY=/d' $APP/.env
echo "POLZA_API_KEY=$KEY" >> $APP/.env
chown vasya:vasya $APP/.env && chmod 600 $APP/.env
systemctl restart vasya
sleep 8
journalctl -u vasya -n 8 --no-pager -o cat
echo -e "\n\033[1;32mГотово: Вася думает через Polza.ai. Обнови страницу на телефоне.\033[0m"
echo "Вернуть Gemini: удали строку POLZA_API_KEY из $APP/.env и выполни systemctl restart vasya"
