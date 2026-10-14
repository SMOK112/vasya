# -*- coding: utf-8 -*-
"""Polza.ai — российский агрегатор нейросетей (OpenAI-совместимый API, оплата в рублях)."""
import base64
import json

import os

import aiohttp

URL = os.environ.get("POLZA_URL", "https://polza.ai/api/v1")


class PolzaError(Exception):
    def __init__(self, status, body):
        super().__init__(f"{status} {str(body)[:400]}")
        self.status = status


class Polza:
    def __init__(self, key):
        self.key = key
        self._s = None
        self.spent = 0.0          # рублей с момента запуска

    async def session(self):
        if self._s is None or self._s.closed:
            self._s = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=90, sock_connect=15, sock_read=60),
                headers={"Authorization": f"Bearer {self.key}"})
        return self._s

    def _cost(self, usage):
        try:
            c = float((usage or {}).get("cost_rub") or (usage or {}).get("cost") or 0)
        except (TypeError, ValueError):
            c = 0.0
        self.spent += c
        return c

    async def models(self):
        """Список чат-моделей: [(id, принимает_картинки, цена_вход_за_1М)]"""
        s = await self.session()
        async with s.get(URL + "/models", params={"type": "chat"}) as r:
            if r.status != 200:
                raise PolzaError(r.status, await r.text())
            data = await r.json()
        out = []
        for m in data.get("data", []):
            arch = m.get("architecture") or {}
            mods = arch.get("input_modalities") or []
            price = ((m.get("top_provider") or {}).get("pricing") or {}).get("prompt_per_million")
            try:
                price = float(price)
            except (TypeError, ValueError):
                price = 1e9
            out.append((m.get("id", ""), "image" in mods, price))
        return out

    async def transcribe(self, wav: bytes, model: str, prompt: str = ""):
        s = await self.session()
        body = {"model": model, "language": "ru",
                "file": "data:audio/wav;base64," + base64.b64encode(wav).decode()}
        if prompt:
            body["prompt"] = prompt
        async with s.post(URL + "/audio/transcriptions", json=body) as r:
            txt = await r.text()
            if r.status != 200:
                raise PolzaError(r.status, txt)
        d = json.loads(txt)
        return (d.get("text") or "").strip(), self._cost(d.get("usage"))

    async def chat_stream(self, model, messages, max_tokens=400, web_search=False):
        """Отдаёт текст ответа кусочками по мере генерации."""
        s = await self.session()
        body = {"model": model, "messages": messages, "stream": True,
                "max_tokens": max_tokens, "temperature": 1.0}
        if web_search:
            body["web_search_options"] = {}
        async with s.post(URL + "/chat/completions", json=body) as r:
            if r.status != 200:
                raise PolzaError(r.status, await r.text())
            async for raw in r.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                except ValueError:
                    continue
                if j.get("error"):
                    raise PolzaError(500, json.dumps(j["error"], ensure_ascii=False))
                for ch in j.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content")
                    if isinstance(piece, str) and piece:
                        yield piece
                if j.get("usage"):
                    self._cost(j["usage"])


def human_error(e):
    """(текст на экран, фраза голосом) — нейтрально, чтобы было понятно, что делать."""
    st = getattr(e, "status", 0)
    low = str(e).lower()
    if st == 402 or "insufficient" in low:
        return ("На Polza.ai кончились деньги — пополни баланс на polza.ai.", "Деньги на мозг кончились, пополни баланс.")
    if st in (401, 403):
        return ("Ключ Polza не подошёл. Выполни на сервере: bash /opt/vasya/polza.sh", "Ключ от мозга не подходит, глянь экран.")
    if st == 429:
        return ("Polza: слишком много запросов, подожди пару секунд.", "Секунду, повтори чуть позже.")
    return ("Ошибка Polza — скинь этот текст в чат.", "Что-то с мозгом, глянь экран.")


def retryable(e):
    st = getattr(e, "status", 0)
    return st in (408, 429, 500, 502, 503, 504) or isinstance(e, (aiohttp.ClientError, TimeoutError))
