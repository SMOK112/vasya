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

POLZA_KEY = os.environ.get("POLZA_API_KEY", "")
BRAIN = "polza" if POLZA_KEY else "gemini"     # есть ключ Polza — думаем через Polza.ai, иначе через Gemini
if not API_KEY and not POLZA_KEY:
    raise SystemExit("Нет ни POLZA_API_KEY, ни GEMINI_API_KEY в .env — запусти polza.sh или deploy.sh.")
if not TOKEN:
    log("!!! VASYA_TOKEN пустой — страница открыта всем, кто знает адрес. Так лучше не делать.")

# ====================================================================== Gemini
from google import genai
from google.genai import types

PROXY = os.environ.get("GEMINI_PROXY", "")   # напр. socks5://127.0.0.1:40000 (Cloudflare WARP, см. warp.sh)
if not API_KEY:
    client = None
elif PROXY:
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
        self.phist = []              # история для Polza (текстом)
        self.listener = None


V = State()


def gen_config():
    mode = ("\n\nСЕЙЧАС ВКЛЮЧЁН ПРИЛИЧНЫЙ РЕЖИМ: никакого мата, общайся вежливо, но с юмором."
            if V.polite else "\n\nРЕЖИМ БЕЗ ЦЕНЗУРЫ: матерись жёстко и много, в каждом предложении, как в стиле выше.")
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
        if "perday" in low or "per_day" in low or "daily" in low:
            return ("У всех моделей кончился дневной бесплатный лимит Gemini. Сбросится через сутки "
                    "(в полдень по Москве примерно).", "Всё, братан, на сегодня бесплатный мозг кончился, нахуй.")
        return ("Слишком часто — бесплатный лимит в минуту. Подожди минуту.", "Бля, не гони, дай минуту отдышаться.")
    if "503" in low or "unavailable" in low or "overloaded" in low or "500" in low:
        return ("Gemini перегружен, попробуй ещё раз через пару секунд.", "Гугл тупит, повтори через секунду.")
    return ("Ошибка Gemini — скинь этот текст в чат.", "Хуйня какая-то с мозгом, глянь экран.")


COOLDOWN = {}          # модель -> до какого времени не трогать (кончился лимит)


MODELS = list(config.GEMINI_MODELS)   # дополняется при старте тем, что реально есть у ключа
BAD_WORDS = ("tts", "image", "audio", "live", "embed", "robotics", "computer", "veo", "imagen", "learnlm", "gemma")


async def discover_models():
    """Спросить у Google, какие модели доступны ключу, и добавить подходящие «flash» в очередь."""
    found = []
    try:
        async for m in await client.aio.models.list():
            name = (m.name or "").split("/")[-1]
            acts = m.supported_actions or []
            if "generateContent" in acts and "flash" in name and not any(b in name for b in BAD_WORDS):
                found.append(name)
    except Exception as e:
        log("   Список моделей не получил:", type(e).__name__, str(e)[:120])
        return

    def ver(n):
        m = re.search(r"(\d+(?:\.\d+)?)", n)
        return float(m.group(1)) if m else 0.0

    # свежие версии вперёд, «lite» после полной, превью — в конце
    found.sort(key=lambda n: ("preview" in n or "exp" in n, "lite" in n, -ver(n), n))
    for n in found:
        if n not in MODELS:
            MODELS.append(n)
    log("   Модели по очереди:", ", ".join(MODELS))


def models_now():
    """Модели по порядку, но те, у которых кончился лимит или которых нет, — в конец."""
    now = time.time()
    ok = [m for m in MODELS if COOLDOWN.get(m, 0) <= now]
    return ok + [m for m in MODELS if m not in ok]


def skip_model(model, e):
    """Пометить модель как временно/насовсем бесполезную. True — можно пробовать следующую."""
    low = str(e).lower()
    if "429" in low or "resource_exhausted" in low or "quota" in low:
        COOLDOWN[model] = time.time() + 30 * 60          # лимит кончился — полчаса не трогаем
        return True
    if "404" in low or "not_found" in low or "no longer available" in low:
        COOLDOWN[model] = time.time() + 24 * 3600        # модели больше нет — сутки не трогаем
        return True
    if "400" in low and "invalid_argument" in low:
        COOLDOWN[model] = time.time() + 3600             # модель не умеет то, что надо (звук/фото)
        return True
    return retryable(e)


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
    text = MARKS.sub("", text).strip()
    if not text:
        return
    log(f"{who}:", text)
    fire({"type": "log", "who": who, "text": text})


def status(text):
    fire({"type": "status", "text": text})


# ====================================================================== голос (edge-tts)
import edge_tts


MARKS = re.compile(r"\[(ЖДУ|ФОТО|РЕШАЮ|ПЕРЕСНЯТЬ)\]")


def clean_for_tts(text):
    text = MARKS.sub("", text)
    text = re.sub(r"[*_#`>\[\]]", "", text)
    text = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", text)
    text = text.replace("—", ", ").replace("…", ".")
    return re.sub(r"\s+", " ", text).strip()


async def synth(text) -> bytes:
    comm = edge_tts.Communicate(text, config.VOICE, rate=config.TTS_RATE, pitch=config.TTS_PITCH,
                                volume=getattr(config, "TTS_VOLUME", "+0%"))
    buf = bytearray()
    async for ch in comm.stream():
        if ch.get("type") == "audio":
            buf += ch["data"]
    return bytes(buf)


def is_speaking():
    return V.tts_busy or not V.q.empty() or time.time() < V.play_until


async def synth_safe(text):
    """Озвучить фразу; None — если сервер Microsoft не ответил (тогда скажет сам телефон)."""
    try:
        t = time.time()
        mp3 = await asyncio.wait_for(synth(text), 15)
        log(f"   ⏱ озвучка {time.time() - t:.1f} с")
        return mp3
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log("Озвучка на сервере не вышла, пусть говорит телефон:", type(e).__name__, e)
        return None


async def tts_worker():
    """Фразы озвучиваются сразу, параллельно (say запускает синтез), а отправляются строго по порядку."""
    while True:
        gen, text, task = await V.q.get()
        V.tts_busy = True
        try:
            if gen != V.gen or task is None:
                continue
            mp3 = await task
            if gen != V.gen:
                continue
            now = time.time()
            if mp3:
                if await send(mp3):      # mp3 48 кбит/с — по размеру знаем длительность
                    V.play_until = max(now, V.play_until) + len(mp3) * 8 / 48000 + 0.4
            elif await send({"type": "speak", "text": text}):
                V.play_until = max(now, V.play_until) + len(text) / 13 + 0.6
        except asyncio.CancelledError:
            pass
        finally:
            V.tts_busy = False


def say(text, gen=None):
    text = clean_for_tts(text)
    task = None
    if re.search(r"[а-яА-Яa-zA-Z0-9]", text):
        task = asyncio.get_running_loop().create_task(synth_safe(text))   # синтез стартует сразу
    V.q.put_nowait((V.gen if gen is None else gen, text, task))


def stop_talking(tell_phone=True):
    V.gen += 1
    while not V.q.empty():
        _, _, task = V.q.get_nowait()
        if task is not None:
            task.cancel()
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
MYNAMES = {n.lower().replace("ё", "е") for n in getattr(config, "NAME_ALIASES", [])}
DECOYS = ["вас", "ваш", "ваша", "наш", "нас", "раз", "класс", "часть", "вес", "весь", "лось", "гусь", "[unk]"]


class WakeSpotter:
    """Постоянно слушает имя Васи и имя хозяина. Возвращает 'vasya', 'name' или None."""

    def __init__(self):
        words = sorted(NAMES | MYNAMES) + [w for w in DECOYS if w not in NAMES and w not in MYNAMES]
        self.rec = vosk.KaldiRecognizer(wake_model, SR, json.dumps(words, ensure_ascii=False))
        self.hit = False

    def _label(self, words):
        if any(w in NAMES for w in words):
            return "vasya"
        if any(w in MYNAMES for w in words):
            return "name"
        return None

    def __call__(self, block):
        pcm = (np.clip(block, -1, 1) * 32767).astype("<i2").tobytes()
        if self.rec.AcceptWaveform(pcm):
            lab = None if self.hit else self._label(json.loads(self.rec.Result()).get("text", "").split())
            self.hit = False
            return lab
        part = json.loads(self.rec.PartialResult()).get("partial", "").split()
        lab = self._label(part)
        if lab and not self.hit:
            self.hit = True
            return lab
        return None

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
        self.ctx = []                  # последние секунды звука вокруг (в памяти, перезатираются)
        self.ctx_len = 0
        self.quiet_until = 0.0         # окно после имени хозяина: слушаю разговор, сам молчу

    def context_audio(self):
        return np.concatenate(self.ctx) if self.ctx else None

    def _remember(self, block):
        self.ctx.append(block)
        self.ctx_len += len(block)
        cap = int(config.CONTEXT_SECONDS * SR)
        while self.ctx_len > cap and len(self.ctx) > 1:
            self.ctx_len -= len(self.ctx.pop(0))

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
            self._remember(block)
            if self.spotter is not None:
                lab = self.spotter(block)
                if lab == "vasya":
                    if self.in_speech:
                        self.wake_in_phrase = True
                    else:
                        self.awake_until = time.time() + config.AWAKE_S
                    wake_ping()
                elif lab == "name" and not self.in_speech:
                    self.quiet_until = time.time() + config.HELP_WINDOW_S   # обращаются ко мне — слушаю, молчу
                    status("Слушаю разговор")
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
        quiet = self.t0 <= self.quiet_until
        if direct:
            self.awake_until = 0.0
            spawn(handle_phrase(audio, wake_hit=self.wake_in_phrase, listener=self))
        elif quiet:
            spawn(handle_phrase(audio, quiet=True, listener=self))   # окно разговора: отвечаю только на просьбу
        elif self.spotter is None:
            spawn(handle_phrase(audio, need_name=True))


_tasks = set()


def spawn(coro):
    t = asyncio.get_running_loop().create_task(coro)
    _tasks.add(t)
    t.add_done_callback(_tasks.discard)


# ====================================================================== мозг
LOOK = re.compile(r"(глян|гляд|посмотр|смотри|помог|подскаж|прочита|прочти|читай|написан|нарисован|"
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
        V.phist = []
        say("Всё, чистый лист.")
        return True
    return False


def only_names(text):
    w = text.lower().replace("ё", "е").split()
    return bool(w) and all(x in NAMES for x in w)


HELP_REPLY = re.compile(r"(что (мне )?(ответить|сказать|отвечать)|помоги ответить|как ответить|"
                        r"подскажи что|что ему (ответить|сказать)|выручай|что говорить)")


async def handle_phrase(audio, ptt=False, wake_hit=False, need_name=False, quiet=False, listener=None):
    stop_talking()                       # новый вопрос — старый ответ больше не нужен
    g = V.gen
    status("Думаю...")
    dur = len(audio) / SR
    T0 = time.time()
    # грубая расшифровка Vosk: Gemini-режиму — всегда; в Polza — для «только Вась», просьбы о помощи и тихого окна
    rough = ""
    if BRAIN != "polza" or quiet or (wake_hit and not ptt and dur < 1.6):
        rough = await asyncio.get_running_loop().run_in_executor(None, rough_text, audio)
        if rough:
            log("   (Vosk расслышал:", rough + ")")
    if wake_hit and not ptt and (only_names(rough) or (not rough and dur < 1.0)):
        if listener is not None:          # сказал только «Вась» — ждём вопрос
            listener.awake_until = time.time() + config.AWAKE_S
        status("Слушаю! Говори")
        return
    # просьба подсказать ответ в разговоре — берём последние секунды вокруг и предлагаем реплику
    if (quiet or wake_hit) and BRAIN == "polza" and HELP_REPLY.search(rough.lower()):
        return await reply_help(g, listener)
    if quiet:                             # тихое окно: обращались к хозяину, но помощи не просил — молчу
        if listener is not None:
            listener.quiet_until = time.time() + config.HELP_WINDOW_S
        status("Слушаю разговор")
        return
    if BRAIN == "polza":
        return await phrase_polza(audio, g, ptt, need_name, listener, T0)
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
    if BRAIN == "polza":
        return await ask_polza("Глянь, чё тут. Если это задание — реши и скажи ответ.", photo, g, None, framing=2)
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
            if skip_model(model, e) and not said and i + 1 < len(models):
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



# ====================================================================== мозг через Polza.ai
from polza import Polza, PolzaError, human_error as polza_human, retryable as polza_retryable

PZ = Polza(POLZA_KEY) if POLZA_KEY else None
CHAT = {"models": list(config.CHAT_MODELS), "bg": config.BG_MODEL, "solve": list(config.SOLVE_MODELS)}
SEEN = []                     # (время, что видно) — заметки фонового режима
BG = {"busy": False}
BG_PROMPT = ("Ты — глаза голосового помощника. Посмотри на кадр с камеры очков и ответь ОДНОЙ строкой строго так:\n"
             "ЗАДАНИЕ: да | <что видно, до пятнадцати слов>\n"
             "или\n"
             "ЗАДАНИЕ: нет | <что видно, до пятнадцати слов>\n"
             "«да» — только если в кадре учебное задание, задача, пример, тест или вопрос (в тетради, на доске, в учебнике, "
             "на экране). Что видно — где хозяин, что перед ним, есть ли рядом люди и чем заняты, без догадок, кто они. "
             "Если кадр тёмный или смазанный — напиши только «-».")
OFFER = {"until": 0.0, "next": 0.0}      # Вася предложил решить задание и ждёт «да/нет»
OFFER_LINES = ["Опа, вижу задание. Решаем или сам будешь мучиться?",
               "Слышь, там задание. Порешать за тебя?",
               "Вижу задачку. Решить, или ты сегодня сам умный?"]
OFFER_NOTE = (" Ты только что заметил камерой задание и спросил хозяина, решать ли его. "
              "Если он согласен (да, давай, го, ага, реши, конечно и тому подобное) — ответь только «[РЕШАЮ]». "
              "Если отказывается — коротко и смешно ответь.")
FRAMING_RULE = (" Если задание на фото видно не целиком, обрезано, размыто или слишком мелко, чтобы прочитать, — не решай, "
                "а одной-двумя фразами скажи, как сдвинуть камеру (ближе, дальше, левее, правее, выше, ниже), "
                "и в самом конце поставь «[ПЕРЕСНЯТЬ]». Если всё видно — реши: сначала ответ, потом коротко как.")


def offer_pending():
    return time.time() < OFFER["until"]


async def wait_quiet(extra=0.0):
    """Подождать, пока Вася договорит (и ещё extra секунд)."""
    for _ in range(300):
        if not is_speaking():
            break
        await asyncio.sleep(0.2)
    if extra:
        await asyncio.sleep(extra)


async def solve_task(g, listener):
    OFFER["until"] = 0.0
    photo = await take_photo()
    if photo is None or V.gen != g:
        if photo is None:
            say("Камера не дала кадр, попробуй ещё раз.")
        return
    await ask_polza("Реши задание, которое видно на фото.", photo, g, listener, framing=3)
POLZA_FORMAT = """

КАК ТЕБЕ ПРИХОДЯТ СООБЩЕНИЯ:
Тебе приходит расшифровка речи хозяина (распознавалка может ошибаться в словах — догадывайся по смыслу),
иногда вместе с фото с камеры — это то, что он сейчас видит. Отвечай сразу по делу, не пересказывай вопрос.
Если чтобы ответить, нужно посмотреть камерой (просит глянуть, прочитать, решить то, что перед ним), а фото нет —
ответь только «[ФОТО]» и больше ничего.
"""
HALLUCINATION = re.compile(r"^\W*(большое спасибо|спасибо( за (внимание|просмотр))?|продолжение следует|"
                           r"субтитры.*|редактор субтитров.*|подписывайтесь.*|всем пока|до свидания)\W*$", re.I)
FIRST_CLAUSE = re.compile(r"(.{18,}?,)(\s)", re.S)   # первую фразу можно начать озвучивать с запятой — быстрее
AUDIO_FORMAT = """

КАК ТЕБЕ ПРИХОДЯТ СООБЩЕНИЯ И КАК ОТВЕЧАТЬ (строго):
Хозяин говорит голосом — тебе приходит аудио.
Первая строка ответа — «Я: » и дословно, что сказал хозяин. Со следующей строки — твой ответ, он будет озвучен.
Если в аудио нет речи (тишина, шум) или речь явно не к тебе — напиши только «Я: -».
Если хозяин сказал только твоё имя — напиши «Я: Вась» и на следующей строке «[ЖДУ]».
Если чтобы ответить, нужно посмотреть камерой (глянуть, прочитать, решить то, что перед ним) —
напиши «Я: ...» и на следующей строке только «[ФОТО]».
"""
NAME_PREFIX = re.compile(r"^\W*(в[ао]с[яьеюи]\w*|васян\w*|васек|васёк)[\s,!.?—-]*", re.I)


async def pick_polza_models():
    """Берём модели из config.CHAT_MODELS, которые реально есть и понимают картинки; иначе — самую дешёвую с картинками."""
    try:
        allm = await PZ.models()
    except Exception as e:
        log("   Список моделей Polza не получил:", type(e).__name__, str(e)[:150])
        return
    have = {mid: (img, price) for mid, img, price in allm}
    chosen = [m for m in config.CHAT_MODELS if m in have and have[m][0]]
    if not chosen:
        cands = sorted([(price, mid) for mid, img, price in allm if img and ("flash" in mid or "mini" in mid)])
        cands = cands or sorted([(price, mid) for mid, img, price in allm if img])
        chosen = [mid for _, mid in cands[:3]]
    if chosen:
        CHAT["models"] = chosen
    CHAT["solve"] = [m for m in config.SOLVE_MODELS if m in have and have[m][0]] or list(CHAT["models"])
    log("Polza: задания с фото решает", ", ".join(CHAT["solve"][:2]))
    if not CHAT["bg"] or CHAT["bg"] not in have or not have[CHAT["bg"]][0]:
        cheap = sorted([(price, mid) for mid, img, price in allm
                        if img and any(k in mid for k in ("flash", "mini", "lite", "nano", "luna"))])
        cheap = cheap or sorted([(price, mid) for mid, img, price in allm if img])
        CHAT["bg"] = cheap[0][1] if cheap else CHAT["models"][0]
    log(f"Polza: фон смотрит {CHAT['bg']} ({have.get(CHAT['bg'], (0, 0))[1]:g} ₽/1М)")
    if config.POLZA_DIRECT_AUDIO and PZ.audio_in.get(CHAT["models"][0]):
        log(f"Polza: {CHAT['models'][0]} слышит звук сам — отдельное распознавание не нужно (быстрее)")
    log("Polza: модели по очереди:", ", ".join(f"{m} ({have.get(m, (0, 0))[1]:g} ₽/1М)" for m in CHAT["models"]))


async def phrase_polza(audio, g, ptt, need_name, listener, T0=None):
    T0 = T0 or time.time()
    wav = to_wav(audio)
    model0 = CHAT["models"][0]
    if config.POLZA_DIRECT_AUDIO and PZ.audio_in.get(model0):
        done = await ask_polza_audio(wav, g, ptt, need_name, listener, T0)
        if done:
            return
        PZ.audio_in[model0] = False               # модель не приняла звук — дальше через распознаватель
        log(f"   {model0} не принял звук — перехожу на отдельное распознавание")
    text, err, ok = "", None, False
    for model in config.STT_MODELS:
        try:
            text, cost = await PZ.transcribe(wav, model, prompt="Разговор с Васей.")
            log(f"   ⏱ распознал за {time.time() - T0:.1f} с ({model})")
            ok = True
            break
        except Exception as e:
            err = e
            log(f"Распознавание ({model}) не вышло:", type(e).__name__, str(e)[:200])
            if getattr(e, "status", 0) in (401, 402, 403):
                break
    if not ok:
        human, spoken = polza_human(err)
        ui("Ошибка", human + "\n\n" + str(err)[:300])
        say(spoken)
        return
    if V.gen != g:
        return
    if not text or HALLUCINATION.match(text) or text.strip(" .") == "Разговор с Васей":
        status("Слушаю")
        return
    q = NAME_PREFIX.sub("", text).strip()
    if need_name and q == text.strip():
        log("   (не мне, молчу):", text)
        status("Слушаю")
        return
    ui("Ты", text)
    if handle_command(text):
        return
    if not q.strip(" .,!?") and not ptt:            # сказал только имя — ждём вопрос
        if listener is not None:
            listener.awake_until = time.time() + config.AWAKE_S
        status("Слушаю! Говори")
        return
    if offer_pending():
        await ask_polza(text, None, g, listener, T0=T0, offered=True)
        return
    photo = await take_photo() if LOOK.search(text.lower()) else None
    if V.gen != g:
        return
    await ask_polza(text, photo, g, listener, T0=T0, framing=2 if photo else 0)


async def ask_polza_audio(wav, g, ptt, need_name, listener, T0):
    """Один запрос: модель сама слышит звук и сразу отвечает. False — модель звук не приняла."""
    if time.time() - V.last > config.NEW_SESSION_AFTER_S:
        V.phist = []
    V.last = time.time()
    mode = ("\n\nСЕЙЧАС ВКЛЮЧЁН ПРИЛИЧНЫЙ РЕЖИМ: никакого мата, общайся вежливо, но с юмором."
            if V.polite else "\n\nРЕЖИМ БЕЗ ЦЕНЗУРЫ: общайся в стиле, описанном выше.")
    seen = [(t, x) for t, x in SEEN if time.time() - t < config.BG_MEMORY_MIN * 60][-6:]
    if seen:
        mode += "\n\nЧТО ТЫ НЕДАВНО ВИДЕЛ КАМЕРОЙ (используй, только если к месту):\n" + \
                "\n".join(time.strftime("[%H:%M] ", time.localtime(t)) + x for t, x in seen)
    note = ("Отвечай, только если он обратился к тебе по имени Вася, иначе «Я: -»." if need_name
            else "Он обращается к тебе.")
    offered = offer_pending()
    if offered:
        note += OFFER_NOTE
    content = [{"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}},
               {"type": "text", "text": "Голосовое от хозяина. " + note}]
    messages = [{"role": "system", "content": PERSONA + mode + AUDIO_FORMAT}] + V.phist + \
               [{"role": "user", "content": content}]
    model = CHAT["models"][0]
    buf, heard, first_line, first, answer = "", None, False, True, []
    try:
        async for piece in PZ.chat_stream(model, messages, max_tokens=config.ANSWER_MAX_TOKENS):
            if V.gen != g:
                return True
            buf += piece
            if not first_line:
                if "\n" not in buf:
                    continue
                line, buf = buf.split("\n", 1)
                first_line = True
                m = re.match(r"^\W*я\s*:\s*(.*)$", line.strip(), re.I)
                heard = m.group(1).strip() if m else ""
                if not m:
                    buf = line + "\n" + buf
                log(f"   ⏱ услышал за {time.time() - T0:.1f} с ({model})")
                if heard in ("-", "—"):
                    status("Слушаю")
                    return True
                if heard:
                    ui("Ты", heard)
                    if handle_command(heard):
                        return True
            if "[ЖДУ]" in buf:
                if listener is not None and not ptt:
                    listener.awake_until = time.time() + config.AWAKE_S
                status("Слушаю! Говори")
                return True
            if "[ФОТО]" in buf or "[РЕШАЮ]" in buf:
                break
            while True:
                m = SENT_END.match(buf) or (FIRST_CLAUSE.match(buf) if first else None)
                if not m:
                    break
                sentence, buf = m.group(1).strip(), buf[m.end():]
                if sentence:
                    if first:
                        log(f"   ⏱ первая фраза ответа через {time.time() - T0:.1f} с ({model}, звук напрямую)")
                        first = False
                    ui("Вася", sentence)
                    say(sentence, g)
                    answer.append(sentence)
    except PolzaError as e:
        if e.status == 400 and not answer:
            log("   звук напрямую не прошёл:", str(e)[:200])
            return False
        log(f"Ошибка Polza ({model}):", str(e)[:300])
        human, spoken = polza_human(e)
        ui("Ошибка", human + "\n\n" + str(e)[:300])
        say(spoken)
        return True
    except Exception as e:
        log(f"Ошибка Polza ({model}):", type(e).__name__, str(e)[:300])
        if answer:
            return True
        return False
    if not first_line:                                  # ответ одной строкой
        m = re.match(r"^\W*я\s*:\s*(.*)$", buf.strip(), re.I | re.S)
        if m:
            heard, buf = m.group(1).strip(), ""
            if heard in ("-", "—", ""):
                status("Слушаю")
                return True
            ui("Ты", heard)
            if handle_command(heard):
                return True
    if "[РЕШАЮ]" in buf and offered:
        await solve_task(g, listener)
        return True
    if offered:
        OFFER["until"] = 0.0
    if "[ФОТО]" in buf:
        photo = await take_photo()
        if V.gen == g:
            await ask_polza(heard or "Глянь, что тут.", photo, g, listener, photo_retry=False, T0=T0, framing=2)
        return True
    if buf.strip() and V.gen == g:
        ui("Вася", buf.strip())
        say(buf.strip(), g)
        answer.append(buf.strip())
    if answer and V.gen == g:
        V.phist += [{"role": "user", "content": heard or "(голосовое)"},
                    {"role": "assistant", "content": " ".join(answer)}]
        V.phist = V.phist[-config.HISTORY_TURNS * 2:]
        log(f"   Polza: потрачено с запуска ≈ {PZ.spent:.2f} ₽")
    return True


async def ask_polza(text, photo, g, listener, photo_retry=True, T0=None, framing=0, offered=False):
    T0 = T0 or time.time()
    first = True
    prompt_text = text + (FRAMING_RULE if photo and framing > 1 else
                          " Это уже поправленный кадр — реши по тому, что видно." if photo and framing == 1 else "")
    if offered:
        prompt_text += OFFER_NOTE
    if time.time() - V.last > config.NEW_SESSION_AFTER_S:
        V.phist = []
    V.last = time.time()
    mode = ("\n\nСЕЙЧАС ВКЛЮЧЁН ПРИЛИЧНЫЙ РЕЖИМ: никакого мата, общайся вежливо, но с юмором."
            if V.polite else "\n\nРЕЖИМ БЕЗ ЦЕНЗУРЫ: общайся в стиле, описанном выше.")
    content = [{"type": "text", "text": prompt_text}]
    if photo:
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(photo).decode(), "detail": "high"}})
    seen = [(t, x) for t, x in SEEN if time.time() - t < config.BG_MEMORY_MIN * 60][-6:]
    if seen:
        mode += "\n\nЧТО ТЫ НЕДАВНО ВИДЕЛ КАМЕРОЙ (используй, только если к месту):\n" + \
                "\n".join(time.strftime("[%H:%M] ", time.localtime(t)) + x for t, x in seen)
    messages = [{"role": "system", "content": PERSONA + mode + POLZA_FORMAT}] + V.phist + \
               [{"role": "user", "content": content}]
    status("Думаю...")
    models = list(CHAT["solve"] if photo else CHAT["models"])   # с фото — модель поумнее, болтать — подешевле
    answer, said, i = [], False, 0
    while i < len(models):
        buf = ""
        try:
            async for piece in PZ.chat_stream(models[i], messages, max_tokens=config.ANSWER_MAX_TOKENS,
                                              web_search=config.POLZA_WEB_SEARCH):
                if V.gen != g:
                    return
                buf += piece
                if "[ФОТО]" in buf or "[РЕШАЮ]" in buf:
                    break
                while True:
                    m = SENT_END.match(buf) or (FIRST_CLAUSE.match(buf) if first else None)
                    if not m:
                        break
                    sentence, buf = m.group(1).strip(), buf[m.end():]
                    if sentence:
                        if first:
                            log(f"   ⏱ первая фраза ответа через {time.time() - T0:.1f} с ({models[i]})")
                            first = False
                        ui("Вася", sentence)
                        say(sentence, g)
                        answer.append(sentence)
                        said = True
            break
        except Exception as e:
            log(f"Ошибка Polza ({models[i]}):", type(e).__name__, str(e)[:300])
            if V.gen != g:
                return
            if not said and polza_retryable(e) and i + 1 < len(models):
                i += 1
                log("   Пробую модель", models[i])
                continue
            human, spoken = polza_human(e)
            ui("Ошибка", human + "\n\n" + str(e)[:300])
            say(spoken)
            return
    if offered:
        OFFER["until"] = 0.0
        if "[РЕШАЮ]" in buf:
            return await solve_task(g, listener)
    if "[ФОТО]" in buf:
        if photo is None and photo_retry:
            p = await take_photo()
            if p is not None and V.gen == g:
                return await ask_polza(text, p, g, listener, photo_retry=False, framing=2)
        buf = buf.replace("[ФОТО]", "")
    if buf.strip() and V.gen == g:
        ui("Вася", buf.strip())
        say(buf.strip(), g)
        answer.append(buf.strip())
    full = " ".join(answer) + " " + buf
    if "[ПЕРЕСНЯТЬ]" in full and photo is not None and framing > 1 and V.gen == g:
        # Вася сказал, как поправить камеру — ждём, пока договорит и ты поправишь, и снимаем снова
        status("Поправь камеру...")
        await wait_quiet(2.5)
        if V.gen != g:
            return
        p = await take_photo()
        if p is not None and V.gen == g:
            return await ask_polza(text, p, g, listener, framing=framing - 1)
        return
    if answer and V.gen == g:
        V.phist += [{"role": "user", "content": text + (" [показал фото]" if photo else "")},
                    {"role": "assistant", "content": MARKS.sub("", " ".join(answer))}]
        V.phist = V.phist[-config.HISTORY_TURNS * 2:]
        log(f"   Polza: потрачено с запуска ≈ {PZ.spent:.2f} ₽")

async def reply_help(g, listener):
    """Хозяину трудно в разговоре — берём последние секунды вокруг и подсказываем, что ответить."""
    audio = listener.context_audio() if listener is not None else None
    if audio is None or len(audio) < SR * 1.5:
        say("Я ещё мало что расслышал, дай поговорить чуток.")
        return
    status("Думаю, что сказать...")
    wav = to_wav(audio[-int(config.CONTEXT_SECONDS * SR):])
    note = ("Хозяину трудно подбирать слова в живом разговоре, и он попросил тебя помочь. "
            "Это запись последних секунд разговора вокруг него. Пойми, что ему только что сказали или о чём спросили, "
            "и предложи, ЧТО ЕМУ ОТВЕТИТЬ: одну-две короткие фразы прямо от его лица, которые он может просто повторить. "
            "Без вступлений и без мата, говори сразу реплику. Если непонятно, кому и что отвечать — скажи это одной фразой.")
    content = [{"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}},
               {"type": "text", "text": note}]
    messages = [{"role": "system", "content": "Ты — незаметный суфлёр в ухе у хозяина. Коротко, по делу, спокойно."},
                {"role": "user", "content": content}]
    model = CHAT["models"][0]
    buf, first = "", True
    try:
        async for piece in PZ.chat_stream(model, messages, max_tokens=config.ANSWER_MAX_TOKENS):
            if V.gen != g:
                return
            buf += piece
            while True:
                m = SENT_END.match(buf) or (FIRST_CLAUSE.match(buf) if first else None)
                if not m:
                    break
                sentence, buf = m.group(1).strip(), buf[m.end():]
                if sentence:
                    first = False
                    ui("Суфлёр", sentence)
                    say(sentence, g)
        if buf.strip() and V.gen == g:
            ui("Суфлёр", buf.strip())
            say(buf.strip(), g)
    except Exception as e:
        log("Суфлёр: ошибка:", type(e).__name__, str(e)[:200])
        human, spoken = polza_human(e)
        ui("Ошибка", human)
        say(spoken)
    if listener is not None:
        listener.quiet_until = time.time() + config.HELP_WINDOW_S


async def handle_bg(data_url):
    """Фоновый кадр: одна строчка «что вокруг» в память. Сам Вася при этом молчит."""
    if PZ is None or BG["busy"] or not data_url:
        return
    BG["busy"] = True
    try:
        msgs = [{"role": "system", "content": BG_PROMPT},
                {"role": "user", "content": [{"type": "text", "text": "Кадр с камеры очков."},
                                             {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}}]}]
        out = ""
        async for piece in PZ.chat_stream(CHAT["bg"], msgs, max_tokens=60):
            out += piece
        out = out.strip().strip("«»\"")
        if not out or out in ("-", "—"):
            return
        m = re.match(r"^\W*задание\W*(да|нет)\W*\|?\s*(.*)$", out, re.I | re.S)
        task = bool(m and m.group(1).lower() == "да")
        desc = (m.group(2).strip() if m else out) or out
        SEEN.append((time.time(), desc + (" [задание]" if task else "")))
        del SEEN[:-40]
        log("   👁", ("[задание] " if task else "") + desc)
        now = time.time()
        if task and now > OFFER["next"] and not is_speaking() and now - V.last > 8 and V.ws is not None:
            OFFER["until"] = now + 15
            OFFER["next"] = now + 240                 # не доставать предложениями чаще, чем раз в 4 минуты
            if V.listener is not None:
                V.listener.awake_until = now + 15     # ответ «да/нет» — без имени
            line = random.choice(OFFER_LINES)
            ui("Вася", line)
            say(line)
    except Exception as e:
        log("Фон: не вышло:", type(e).__name__, str(e)[:150])
    finally:
        BG["busy"] = False


# ====================================================================== веб-сервер
async def index(request):
    return web.FileResponse(os.path.join(BASE, "static", "index.html"),
                            headers={"Cache-Control": "no-store"})


def key_ok(k):
    return not TOKEN or secrets.compare_digest(k.strip().lower(), TOKEN.strip().lower())


async def check(request):
    k = request.query.get("k", "")
    ok = key_ok(k)
    if not ok:
        who = request.headers.get("X-Forwarded-For", request.remote)
        log("Открыли без ключа:" if not k.strip() else f"Неверный ключ ({len(k)} симв.):", who)
    return web.json_response({"ok": ok}, status=200 if ok else 403)


async def ws_handler(request):
    if not key_ok(request.query.get("k", "")):
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
                    V.listener = listener
                    log(f"Микрофон телефона: {listener.rate} Гц")
                    status("Слушаю. Скажи «Вась» или держи кнопку")
                    await ws.send_str(json.dumps({"type": "cfg", "bg_interval": config.BG_INTERVAL_S,
                                                  "bg_ok": BRAIN == "polza"}))
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
                elif t == "bg":
                    if BRAIN == "polza" and not is_speaking():
                        spawn(handle_bg(m.get("data")))
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
    if BRAIN == "polza":
        log("Мозг: Polza.ai")
        await pick_polza_models()
        return
    # проверяем только список моделей — он бесплатный и не тратит дневной лимит запросов
    await discover_models()
    log("Gemini доступен, моделей в очереди:", len(MODELS))


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
