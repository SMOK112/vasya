# Вася — голосовой ИИ-кореш для очков

Сервер, который слушает телефон (а потом очки), отвечает голосом и смотрит камерой.

```
телефон (Chrome) ──звук/фото──▶ сервер: Vosk (имя «Вась») + Silero VAD (фразы)
                                         │
                                         ▼
                                  Gemini: распознаёт речь и отвечает
                                         │
телефон ◀──────────голос (mp3)────── edge-tts
```

Ничего тяжёлого (без torch и Whisper) — живёт на сервере с 1 ГБ памяти рядом с другими ботами.

## Установка (Ubuntu/Debian, от root)

```bash
git clone https://github.com/SMOK112/vasya.git /opt/vasya && bash /opt/vasya/deploy.sh
```

Скрипт спросит ключ Gemini (https://aistudio.google.com/apikey), всё поставит, включит автозапуск
и HTTPS на адресе `IP-сервера.sslip.io`, а в конце напечатает ссылку для телефона с ключом доступа.

Если в логах «User location is not supported» — Google не любит IP сервера. Тогда:

```bash
bash /opt/vasya/warp.sh       # запросы к Gemini пойдут через Cloudflare WARP, остальной сервер не трогается
```

Мозг через Polza.ai (рубли, без лимитов бесплатного Gemini, ВПН не нужен):

```bash
bash /opt/vasya/polza.sh      # спросит API-ключ Polza, проверит и включит
```

## Как пользоваться

- Ссылку и QR-код для телефона показать ещё раз: `bash /opt/vasya/link.sh`
- Скажи «Вась» — «дзынь» — говори вопрос. Или одной фразой: «Вась, глянь что тут написано».
- Кнопка «Держи и говори» — без имени. «Глянь» — молча показать камеру. «Заткнись» — замолчать.
- Голосом: «приличный режим», «можно материться», «новый разговор».

## Обслуживание

```bash
journalctl -u vasya -f        # логи
bash /opt/vasya/update.sh     # обновить с GitHub
nano /opt/vasya/config.py     # голос, модели, паузы — потом systemctl restart vasya
nano /opt/vasya/.env          # ключ Gemini и ключ страницы
```

Ключи лежат только в `/opt/vasya/.env` на сервере, в репозиторий не попадают.

`models/silero_vad.onnx` — модель [Silero VAD](https://github.com/snakers4/silero-vad) (MIT).
