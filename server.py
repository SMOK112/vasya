# -*- coding: utf-8 -*-
"""
Вася на сервере. Телефон (а потом очки) шлёт звук сюда по интернету, сервер:
  — ловит имя «Вась» (Vosk, локально, почти без нагрузки);
  — режет речь на фразы (Silero VAD, ONNX — без torch);
  — отправляет фразу, обращённую к Васе, прямо в Gemini: он и распознаёт, и отвечает;
  — озвучивает ответ (edge-tts) и шлёт голос обратно.
Ничего тяжёлого: работает на сервере с 1 ГБ памяти.

Запуск: python server.py   (ключи — в файле .env, см. deploy.sh)
"""
import asyncio
import base64
import io
import json
import os
import random
import re
import secrets
import time
import wave

import numpy as np
from aiohttp import web, WSMsgType

import config

BASE = os.path.dirname(os.path.abspath(__file__))
SR = 16000          # частота для распознавания
BLOCK = 512         # 32 мс — шаг детектора речи
BLOCK_S = BLOCK / SR


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_env():
    p = os.path.join(BASE, ".env")
    if os.path.isfile(p) and os.access(p, os.R_OK):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
API_KEY = os.environ.get("GEMINI_API_KEY", "")
TOKEN = os.environ.get("VASYA_TOKEN", "")
PORT = int(os.environ.get("VASYA_PORT", "8080"))
HOST = os.environ.get("VASYA_HOST", "127.0.0.1")

if not API_KEY:
    raise SystemExit("Нет GEMINI_API_KEY в .env — запусти deploy.sh или впиши ключ руками.")
if not TOKEN:
    log("!!! VASYA_TOKEN пустой — страница открыта всем, кто знает адрес. Так лучше не делать.")

# ====================================================================== Gemini
from google import genai
from google.genai import types

PROXY = os.environ.get("GEMINI_PROXY", "")   # напр. socks5://127.0.0.1:40000 (Cloudflare WARP, см. warp.sh)
if PROXY:
    import httpx
    client = genai.Client(api_key=API_KEY, http_options=types.HttpOptions(
        httpx_async_client=httpx.AsyncClient(proxy=PROXY, timeout=httpx.Timeout(60.0, connect=15.0))))
    log("Gemini — через прокси", PROXY)
else:
    client = genai.Client(api_key=API_KEY)

with open(os.path.join(BASE, "persona.txt"), encoding="utf-8") as f:
    PERSONA = f.read()

FORMAT = """

КАК ТЕБЕ ПРИХОДЯТ СООБЩЕНИЯ И КАК ОТВЕЧАТЬ (строго):
Хозяин говорит голосом — тебе приходит аудио, иногда вместе с фото с камеры (это то, что он сейчас видит).
Первая строка ответа — «Я: » и дословно, что сказал хозяин. Имя Вася в начале можно не писать.
Со следующей строки — твой ответ, он будет озвучен.
Если в аудио нет речи (тишина, шум, музыка) — напиши только «Я: -» и больше ничего.
Если хозяин сказал только твоё имя и больше ничего — напиши «Я: Вась» и на следующей строке «[ЖДУ]».
Если чтобы ответить, нужно посмотреть камерой (просит глянуть, прочитать, решить то, что перед ним), а фото нет —
напиши «Я: ...» и на следующей строке только «[ФОТО]».
"""

SAFETY = [types.SafetySetting(category=c, threshold="BLOCK_NONE") for c in (
    "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]


class State:
    def __init__(self):
        self.ws = None               # подключённый телефон (один)
        self.polite = False
        self.search = config.WEB_SEARCH
        self.history = []            # прошлые реплики — только текстом, без аудио и фото
        self.last = 0.0
        self.gen = 0                 # номер ответа; «заткнись» — увеличиваем, старое выкидываем
        self.q = asyncio.Queue()     # фразы на озвучку: (gen, text)
        self.tts_busy = False
        self.play_until = 0.0        # примерно когда телефон договорит
        self.snap_id = 0
        self.snap_fut = None
        self.greeted = 0.0


V = State()


def gen_config():
    mode = ("\n\nСЕЙЧАС ВКЛЮЧЁН ПРИЛИЧНЫЙ РЕЖИМ: никакого мата, общайся вежливо, но с юмором."
            if V.polite else "\n\nСЕЙЧАС МОЖНО И НУЖНО МАТЕРИТЬСЯ — сочно, в каждом ответе.")
    tools = [types.Tool(google_search=types.GoogleSearch())] if V.search else None
    return types.GenerateContentConfig(system_instruction=PERSONA + mode + FORMAT,
                                       temperature=1.0, safety_settings=SAFETY, tools=tools)


def explain_error(e):
    """(текст на экран, фраза голосом)"""
    low = f"{type(e).__name__}: {e}".lower()
    if any(k in low for k in ("api key", "api_key", "unauthenticated", "access_token_type", "401")):
        return ("Ключ Gemini не подошёл. Создай новый в AI Studio и поменяй GEMINI_API_KEY в /opt/vasya/.env, "
                "потом: systemctl restart vasya", "Ключ от мозга не подходит, глянь экран.")
    if "location" in low:
        return ("Google считает, что сервер в неподдерживаемой стране. Выполни на сервере: bash /opt/vasya/warp.sh",
                "Гугл не пускает с этого сервера, глянь экран.")
    if "429" in low or "resource_exhausted" in low or "quota" in low:
        return ("Кончился бесплатный лимит запросов, подожди минуту.", "Бля, лимит кончился. Подожди минутку.")
    if "503" in low or "unavailable" in low or "overloaded" in low or "500" in low:
        return ("Gemini перегружен, попробуй ещё раз через пару секунд.", "Гугл тупит, повтори через секунду.")
    return ("Ошибка Gemini — скинь этот текст в чат.", "Хуйня какая-то с мозгом, глянь экран.")


COOLDOWN = {}          # модель -> до какого времени не трогать (кончился лимит)


def models_now():
    """Модели по порядку, но те, у которых кончился лимит, — в конец."""
    now = time.time()
    ok = [m for m in config.GEMINI_MODELS if COOLDOWN.get(m, 0) <= now]
    return ok + [m for m in config.GEMINI_MODELS if m not in ok]


def retryable(e):
    low = str(e).lower()
    return any(k in low for k in ("503", "unavailable", "overloaded", "429", "resource_exhausted",
                                  "500", "internal", "timeout", "timed out", "deadline"))


# ====================================================================== связь с телефоном
async def send(obj) -> bool:
    ws = V.ws
    if ws is None or ws.closed:
        return False
    try:
        if isinstance(obj, (bytes, bytearray)):
            await ws.send_bytes(bytes(obj))
        else:
            await ws.send_str(json.dumps(obj, ensure_ascii=False))
        return True
    except Exception as e:
        log("Телефон не принял:", e)
        return False


def fire(obj):
    asyncio.get_running_loop().create_task(send(obj))


def ui(who, text):
    log(f"{who}:", text)
    fire({"type": "log", "who": who, "text": text})


def status(text):
    fire({"type": "status", "text": text})


# ====================================================================== голос (edge-tts)
import edge_tts


def clean_for_tts(text):
    text = re.sub(r"\[(ЖДУ|ФОТО)\]", "", text)
    text = re.sub(r"[*_#`>\[\]]", "", text)
    text = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", text)
    text = text.replace("—", ", ").replace("…", ".")
    return re.sub(r"\s+", " ", text).strip()


async def synth(text) -> bytes:
    comm = edge_tts.Communicate(text, config.VOICE, rate=config.TTS_RATE, pitch=config.TTS_PITCH)
    buf = bytearray()
    async for ch in comm.stream():
        if ch.get("type") == "audio":
            buf += ch["data"]
    return bytes(buf)


def is_speaking():
    return V.tts_busy or not V.q.empty() or time.time() < V.play_until


async def tts_worker():
    while True:
        gen, text = await V.q.get()
        V.tts_busy = True
        try:
            text = clean_for_tts(text)
            if gen != V.gen or not re.search(r"[а-яА-Яa-zA-Z0-9]", text):
                continue
            mp3 = None
            try:
                mp3 = await asyncio.wait_for(synth(text), 15)
            except Exception as e:
                log("Озвучка на сервере не вышла, пусть говорит телефон:", type(e).__name__, e)
            if gen != V.gen:
                continue
            now = time.time()
            if mp3:
                if await send(mp3):      # mp3 48 кбит/с — по размеру знаем длительность
                    V.play_until = max(now, V.play_until) + len(mp3) * 8 / 48000 + 0.4
            elif await send({"type": "speak", "text": text}):
                V.play_until = max(now, V.play_until) + len(text) / 13 + 0.6
        finally:
            V.tts_busy = False


def say(text, gen=None):
    V.q.put_nowait((V.gen if gen is None else gen, text))


def stop_talking(tell_phone=True):
    V.gen += 1
    while not V.q.empty():
        V.q.get_nowait()
    V.play_until = 0.0
    if tell_phone:
        fire({"type": "stop"})


# ====================================================================== уши: детектор речи и имя
import onnxruntime


class SileroVAD:
    """Silero VAD v5 через onnxruntime (как их OnnxWrapper, только без torch)."""

    def __init__(self, path):
        o = onnxruntime.SessionOptions()
        o.inter_op_num_threads = 1
        o.intra_op_num_threads = 1
        self.s = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=o)
        self.sr = np.array(SR, dtype=np.int64)
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.ctx = np.zeros((1, 64), dtype=np.float32)

    def __call__(self, block):
        x = np.concatenate([self.ctx, block.reshape(1, -1).astype(np.float32)], axis=1)
        out, self.state = self.s.run(None, {"input": x, "state": self.state, "sr": self.sr})
        self.ctx = x[:, -64:]
        return float(out[0][0])


VAD_PATH = os.path.join(BASE, "models", "silero_vad.onnx")

try:
    import vosk
    vosk.SetLogLevel(-1)
    VOSK_PATH = os.path.join(BASE, "models", config.VOSK_MODEL)
    wake_model = vosk.Model(VOSK_PATH) if os.path.isdir(VOSK_PATH) else None
except Exception as e:
    log("Vosk не загрузился:", e)
    wake_model = None
if wake_model is None:
    log("!!! Нет модели Vosk — имя будет проверять Gemini (дороже). Запусти deploy.sh ещё раз.")

NAMES = {n.lower().replace("ё", "е") for n in config.WAKE_NAMES}
DECOYS = ["вас", "ваш", "ваша", "наш", "нас", "раз", "класс", "часть", "вес", "весь", "лось", "гусь", "[unk]"]


class WakeSpotter:
    """Постоянно слушает только имя — как «Алиса»."""

    def __init__(self):
        words = sorted(NAMES) + [w for w in DECOYS if w not in NAMES]
        self.rec = vosk.KaldiRecognizer(wake_model, SR, json.dumps(words, ensure_ascii=False))
        self.hit = False

    def __call__(self, block):
        pcm = (np.clip(block, -1, 1) * 32767).astype("<i2").tobytes()
        if self.rec.AcceptWaveform(pcm):
            text = json.loads(self.rec.Result()).get("text", "")
            fired = not self.hit and any(w in NAMES for w in text.split())
            self.hit = False
            return fired
        part = json.loads(self.rec.PartialResult()).get("partial", "")
        if not self.hit and any(w in NAMES for w in part.split()):
            self.hit = True
            return True
        return False

    def reset(self):
        self.rec.Reset()
        self.hit = False


def rough_text(audio):
    """Грубая расшифровка Vosk — только чтобы заранее понять «глянь/прочитай» и «одно имя»."""
    if wake_model is None:
        return ""
    rec = vosk.KaldiRecognizer(wake_model, SR)
    rec.AcceptWaveform((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return json.loads(rec.FinalResult()).get("text", "")


class StreamResampler:
    """Пересчёт частоты потоком без щелчков на стыках кусков."""

    def __init__(self, src, dst=SR, taps=97):
        self.ratio = src / dst
        fc = 0.45 * dst / src
        n = np.arange(taps) - (taps - 1) / 2
        h = np.sinc(2 * fc * n) * np.hamming(taps)
        self.h = (h / h.sum()).astype(np.float32)
        self.hist = np.zeros(taps - 1, dtype=np.float32)
        self.buf = np.zeros(0, dtype=np.float32)
        self.pos = 0.0

    def __call__(self, x):
        full = np.concatenate([self.hist, x.astype(np.float32)])
        y = np.convolve(full, self.h, mode="valid")
        self.hist = full[-(len(self.h) - 1):]
        self.buf = np.concatenate([self.buf, y])
        last = len(self.buf) - 1
        if last < self.pos:
            return np.zeros(0, dtype=np.float32)
        n_out = int((last - self.pos) // self.ratio) + 1
        idx = self.pos + np.arange(n_out) * self.ratio
        out = np.interp(idx, np.arange(len(self.buf)), self.buf).astype(np.float32)
        self.pos = idx[-1] + self.ratio
        drop = min(int(self.pos), len(self.buf))
        self.buf = self.buf[drop:]
        self.pos -= drop
        return out


def wake_ping():
    log("   >>> услышал имя")
    fire({"type": "wake"})
    status("Слушаю! Говори")


class Listener:
    PREROLL = 12                       # ~0.4 с до начала речи — чтобы не съесть «Ва-»

    def __init__(self, rate):
        self.rate = int(rate)
        self.resample = StreamResampler(self.rate) if self.rate != SR else None
        self.vad = SileroVAD(VAD_PATH)
        self.spotter = WakeSpotter() if wake_model is not None else None
        self.leftover = np.zeros(0, dtype=np.float32)
        self.pre, self.phrase = [], []
        self.in_speech = False
        self.silence = 0
        self.voiced = 0
        self.wake_in_phrase = False
        self.awake_until = 0.0
        self.t0 = 0.0
        self.ptt = False
        self.ptt_buf = []
        self.was_speaking = False

    def reset_ears(self):
        self.vad.reset()
        if self.spotter is not None:
            self.spotter.reset()
        self.pre, self.phrase, self.in_speech = [], [], False

    def set_ptt(self, on):
        if on:
            stop_talking()
            self.reset_ears()
            self.ptt, self.ptt_buf = True, []
            status("Говори, слушаю...")
            return
        if not self.ptt:
            return
        self.ptt = False
        audio = np.concatenate(self.ptt_buf) if self.ptt_buf else np.zeros(0, np.float32)
        self.ptt_buf = []
        if len(audio) < SR * 0.3:
            status("Слушаю")
            return
        spawn(handle_phrase(audio, ptt=True))

    def feed(self, raw):
        if self.resample is not None:
            raw = self.resample(raw)
        if self.ptt:
            self.ptt_buf.append(raw)
            return
        chunk = np.concatenate([self.leftover, raw])
        while len(chunk) >= BLOCK:
            block, chunk = chunk[:BLOCK], chunk[BLOCK:]
            if is_speaking():          # пока говорит сам — не слушаем, чтобы не слышать себя
                if not self.was_speaking:
                    self.reset_ears()
                self.was_speaking = True
                continue
            if self.was_speaking:
                self.was_speaking = False
                status("Слушаю")
            if self.spotter is not None and self.spotter(block):
                if self.in_speech:
                    self.wake_in_phrase = True
                else:
                    self.awake_until = time.time() + config.AWAKE_S
                wake_ping()
            p = self.vad(block)
            if not self.in_speech:
                self.pre = (self.pre + [block])[-self.PREROLL:]
                if p >= config.VAD_THRESHOLD:
                    self.in_speech, self.phrase = True, list(self.pre)
                    self.silence, self.voiced, self.t0 = 0, 1, time.time()
                    self.wake_in_phrase = False
                continue
            self.phrase.append(block)
            if p >= config.VAD_THRESHOLD:
                self.voiced += 1
                self.silence = 0
            elif p < config.VAD_THRESHOLD - 0.15:
                self.silence += 1
            if self.silence * BLOCK_S * 1000 >= config.PAUSE_MS or len(self.phrase) * BLOCK_S > 25:
                self.end_phrase()
        self.leftover = chunk

    def end_phrase(self):
        self.in_speech = False
        audio = np.concatenate(self.phrase)
        self.phrase, self.pre = [], []
        if self.voiced * BLOCK_S < 0.25:        # щелчок, кашель
            return
        direct = self.wake_in_phrase or self.t0 <= self.awake_until
        if direct:
            self.awake_until = 0.0
            spawn(handle_phrase(audio, wake_hit=self.wake_in_phrase, listener=self))
        elif self.spotter is None:
            spawn(handle_phrase(audio, need_name=True))   # без Vosk имя проверяет сам Gemini


_tasks = set()


def spawn(coro):
    t = asyncio.get_running_loop().create_task(coro)
    _tasks.add(t)
    t.add_done_callback(_tasks.discard)


# ====================================================================== мозг
LOOK = re.compile(r"(глян|гляд|посмотр|смотри|прочита|прочти|читай|написан|нарисован|"
                  r"что (тут|там|здесь)|что это за|реши|видишь|на фото|сфоткай|фоткай|камер)")
SENT_END = re.compile(r"(.+?[.!?…]+)(\s|$)", re.S)


def to_wav(audio) -> bytes:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    return b.getvalue()


async def take_photo(timeout=10.0):
    if V.ws is None:
        return None
    V.snap_id += 1
    V.snap_fut = asyncio.get_running_loop().create_future()
    status("Смотрю...")
    if not await send({"type": "snap", "id": V.snap_id}):
        return None
    try:
        return await asyncio.wait_for(V.snap_fut, timeout)
    except asyncio.TimeoutError:
        return None


def handle_command(t) -> bool:
    low = t.lower().strip(" .!?")
    low = re.sub(r"^(вас\w*)[ ,]*", "", low)
    if "приличный режим" in low:
        V.polite = True
        say("Окей, веду себя прилично.")
        return True
    if "можно материться" in low or "неприличный режим" in low:
        V.polite = False
        say("О, наконец-то, ёпта. Погнали.")
        return True
    if low in ("стоп", "заткнись", "хватит", "замолчи"):
        stop_talking()
        status("Слушаю")
        return True
    if "новый разговор" in low or "забудь всё" in low or "забудь все" in low:
        V.history = []
        say("Всё, чистый лист.")
        return True
    return False


def only_names(text):
    w = text.lower().replace("ё", "е").split()
    return bool(w) and all(x in NAMES for x in w)


async def handle_phrase(audio, ptt=False, wake_hit=False, need_name=False, listener=None):
    stop_talking()                       # новый вопрос — старый ответ больше не нужен
    g = V.gen
    status("Думаю...")
    rough = await asyncio.get_running_loop().run_in_executor(None, rough_text, audio)
    if rough:
        log("   (Vosk расслышал:", rough + ")")
    dur = len(audio) / SR
    if wake_hit and not ptt and (only_names(rough) or (not rough and dur < 1.0)):
        if listener is not None:          # сказал только «Вась» — ждём вопрос
            listener.awake_until = time.time() + config.AWAKE_S
        status("Слушаю! Говори")
        return
    photo = await take_photo() if LOOK.search(rough) else None
    if V.gen != g:
        return
    note = "Голосовое от хозяина."
    if photo:
        note += " Фото с камеры — то, что он сейчас видит."
    note += (" Отвечай, только если он обратился к тебе по имени Вася, иначе «Я: -»."
             if need_name else " Он обращается к тебе.")
    await ask([types.Part.from_bytes(data=to_wav(audio), mime_type="audio/wav")], note, photo, g, listener)


async def handle_look(photo):
    stop_talking()
    g = V.gen
    ui("Ты", "[фото] Глянь, чё тут")
    await ask([], "Хозяин показывает тебе камеру: глянь, чё тут. Если это задание — реши и скажи ответ.",
              photo, g, None, heard_known="[фото] Глянь, чё тут")


async def ask(parts, note, photo, g, listener, heard_known=None, photo_retry=True):
    if time.time() - V.last > config.NEW_SESSION_AFTER_S:
        V.history = []                    # после долгой паузы — новый разговор, дешевле
    V.last = time.time()
    user_parts = list(parts)
    if photo:
        user_parts.append(types.Part.from_bytes(data=photo, mime_type="image/jpeg"))
    user_parts.append(types.Part(text=note))
    contents = V.history + [types.Content(role="user", parts=user_parts)]

    models = models_now()
    heard, answer, said = heard_known, [], False
    i = 0
    while i < len(models):
        model = models[i]
        buf, first_done = "", False
        try:
            stream = await client.aio.models.generate_content_stream(
                model=model, contents=contents, config=gen_config())
            async for ch in stream:
                if V.gen != g:
                    return
                buf += ch.text or ""
                if not first_done:
                    if "\n" not in buf:
                        continue
                    first, rest = buf.split("\n", 1)
                    first_done = True
                    m = re.match(r"^\W*я\s*:\s*(.*)$", first.strip(), re.I)
                    if m and heard_known is not None:
                        buf = rest               # что сказал — уже знаем, строку выкидываем
                    elif m:
                        heard, buf = m.group(1).strip(), rest
                        if heard in ("-", "—", ""):
                            status("Слушаю")
                            return
                        ui("Ты", heard)
                        if handle_command(heard):
                            return
                    elif heard is None:
                        heard = ""
                if "[ЖДУ]" in buf:
                    if listener is not None:
                        listener.awake_until = time.time() + config.AWAKE_S
                    status("Слушаю! Говори")
                    return
                if "[ФОТО]" in buf:
                    break
                while True:
                    m = SENT_END.match(buf)
                    if not m:
                        break
                    sentence, buf = m.group(1).strip(), buf[m.end():]
                    if sentence:
                        ui("Вася", sentence)
                        say(sentence, g)
                        answer.append(sentence)
                        said = True
            # поток закончился
            if not first_done:                # ответ одной строкой
                m = re.match(r"^\W*я\s*:\s*(.*)$", buf.strip(), re.I | re.S)
                if m and heard_known is not None:
                    buf = ""
                elif m:
                    heard, buf = m.group(1).strip(), ""
                    if heard in ("-", "—", ""):
                        status("Слушаю")
                        return
                    ui("Ты", heard)
                    if handle_command(heard):
                        return
            if "[ФОТО]" in buf:
                if photo is None and photo_retry:
                    p = await take_photo()
                    if p is not None and V.gen == g:
                        return await ask(parts, note + " Фото с камеры — то, что он сейчас видит.", p, g,
                                         listener, heard_known=heard or "", photo_retry=False)
                buf = buf.replace("[ФОТО]", "")
            if "[ЖДУ]" in buf:
                status("Слушаю! Говори")
                return
            if buf.strip() and V.gen == g:
                ui("Вася", buf.strip())
                say(buf.strip(), g)
                answer.append(buf.strip())
            break
        except Exception as e:
            log(f"Ошибка Gemini ({model}):", type(e).__name__, str(e)[:300])
            if V.gen != g:
                return
            low = str(e).lower()
            if V.search and not said and any(k in low for k in ("search", "tool", "ground")):
                log("   Выключаю поиск и пробую снова")
                V.search = False
                continue
            if "429" in low or "resource_exhausted" in low or "quota" in low:
                COOLDOWN[model] = time.time() + 30 * 60      # лимит кончился — полчаса не трогаем
            if not said and retryable(e) and i + 1 < len(models):
                log("   Пробую модель", models[i + 1])
                i += 1
                continue
            human, spoken = explain_error(e)
            ui("Ошибка", human + "\n\n" + str(e)[:300])
            say(spoken)
            return

    if answer and V.gen == g:
        text_user = (heard or "(голосовое)") + (" [показал фото]" if photo else "")
        V.history += [types.Content(role="user", parts=[types.Part(text=text_user)]),
                      types.Content(role="model", parts=[types.Part(text=" ".join(answer))])]
        V.history = V.history[-config.HISTORY_TURNS * 2:]


# ====================================================================== веб-сервер
async def index(request):
    return web.FileResponse(os.path.join(BASE, "static", "index.html"),
                            headers={"Cache-Control": "no-store"})


async def check(request):
    ok = not TOKEN or secrets.compare_digest(request.query.get("k", ""), TOKEN)
    if not ok:
        log("Неверный ключ в ссылке:", request.headers.get("X-Forwarded-For", request.remote))
    return web.json_response({"ok": ok}, status=200 if ok else 403)


async def ws_handler(request):
    if TOKEN and not secrets.compare_digest(request.query.get("k", ""), TOKEN):
        log("Чужой без ключа:", request.headers.get("X-Forwarded-For", request.remote))
        return web.Response(status=403, text="нужен ключ")
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=0)
    await ws.prepare(request)
    old, V.ws = V.ws, ws
    if old is not None and not old.closed:
        await old.close()
    log("Телефон подключился:", request.headers.get("X-Forwarded-For", request.remote))
    listener = None
    try:
        async for msg in ws:
          try:
            if msg.type == WSMsgType.BINARY:
                if listener is not None:
                    listener.feed(np.frombuffer(msg.data, dtype="<i2").astype(np.float32) / 32768.0)
            elif msg.type == WSMsgType.TEXT:
                m = json.loads(msg.data)
                t = m.get("type")
                if t == "hello":
                    log("   привет от телефона:", m)
                    listener = Listener(m.get("rate", 48000))
                    log(f"Микрофон телефона: {listener.rate} Гц")
                    status("Слушаю. Скажи «Вась» или держи кнопку")
                    if time.time() - V.greeted > 120:
                        V.greeted = time.time()
                        say("Здорово! Вася на связи." if V.polite else "Здорово, братан! Вася на связи, хули.")
                elif t == "ptt" and listener is not None:
                    listener.set_ptt(bool(m.get("on")))
                elif t == "stop":
                    stop_talking(tell_phone=False)
                    status("Слушаю")
                elif t == "idle":
                    if not V.tts_busy and V.q.empty():
                        V.play_until = min(V.play_until, time.time() + 0.2)
                elif t == "photo":
                    data = m.get("data")
                    data = base64.b64decode(data.split(",", 1)[-1]) if data else None
                    log(f"Фото: {len(data) // 1024} КБ" if data else f"Фото не вышло: {m.get('error')}")
                    if m.get("ask"):
                        if data:
                            spawn(handle_look(data))
                    elif m.get("id") == V.snap_id and V.snap_fut is not None and not V.snap_fut.done():
                        V.snap_fut.set_result(data)
          except Exception:
            import traceback
            log("!!! Ошибка при обработке данных телефона:\n" + traceback.format_exc())
    finally:
        if V.ws is ws:
            V.ws = None
        log("Телефон отключился")
    return ws


async def on_startup(app):
    app["tts"] = asyncio.get_running_loop().create_task(tts_worker())
    log("Проверяю Gemini...")
    try:
        for model in config.GEMINI_MODELS:
            try:
                r = await client.aio.models.generate_content(model=model, contents="Ответь одним словом: ок")
                log(f"Gemini на связи ({model}):", (r.text or "").strip()[:30])
                return
            except Exception as e:
                low = str(e).lower()
                if "429" in low or "quota" in low or "resource_exhausted" in low:
                    COOLDOWN[model] = time.time() + 30 * 60
                    log(f"   {model}: кончился бесплатный лимит, беру следующую")
                    continue
                raise
        log("!!! У всех моделей кончился бесплатный лимит — подожди или поменяй GEMINI_MODELS в config.py")
    except Exception as e:
        log("!!! Gemini не отвечает:", type(e).__name__, str(e)[:300])
        log("!!!", explain_error(e)[0])


def main():
    app = web.Application(client_max_size=20 * 1024 * 1024)
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/check", check)
    app.on_startup.append(on_startup)
    log(f"Вася слушает на {HOST}:{PORT}")
    web.run_app(app, host=HOST, port=PORT, access_log=None, print=None)


if __name__ == "__main__":
    main()
