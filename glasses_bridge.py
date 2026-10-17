"""Opt-in ESP32-S3 camera bridge for Vasya.

The ESP32 initiates authenticated HTTPS requests. The server never needs to reach
the device's private 192.168.x.x address. Photos are kept in RAM until consumed.
"""
import asyncio
import os
import secrets
import time

from aiohttp import web

MAX_JPEG_BYTES = 4 * 1024 * 1024
POLL_SECONDS = 15
ONLINE_SECONDS = 40
CAPTURE_SECONDS = 20


class GlassesBridge:
    def __init__(self, token=None):
        self.token = os.environ.get("GLASSES_TOKEN", "") if token is None else token
        self._commands = asyncio.Queue(maxsize=4)
        self._pending = {}
        self._request_id = 0
        self._last_seen = 0.0

    @property
    def enabled(self):
        return len(self.token) >= 32

    def online(self):
        return self.enabled and (time.monotonic() - self._last_seen) < ONLINE_SECONDS

    def _check_auth(self, request):
        if not self.enabled:
            raise web.HTTPServiceUnavailable(text="Glasses integration not configured")
        value = request.headers.get("Authorization", "")
        scheme, sep, supplied = value.partition(" ")
        if not (sep and scheme.lower() == "bearer"
                and secrets.compare_digest(supplied, self.token)):
            raise web.HTTPForbidden(text="Invalid device credential")

    async def next_command(self, request):
        """Long-poll: plain integer request id, 0 when idle."""
        self._check_auth(request)
        self._last_seen = time.monotonic()
        until = time.monotonic() + POLL_SECONDS
        while time.monotonic() < until:
            try:
                request_id = await asyncio.wait_for(
                    self._commands.get(), timeout=max(0.05, until - time.monotonic()))
            except asyncio.TimeoutError:
                break
            self._last_seen = time.monotonic()
            future = self._pending.get(request_id)
            if future is not None and not future.done():
                return web.Response(text=str(request_id), content_type="text/plain",
                                    headers={"Cache-Control": "no-store"})
        return web.Response(text="0", content_type="text/plain",
                            headers={"Cache-Control": "no-store"})

    async def receive_photo(self, request):
        self._check_auth(request)
        try:
            request_id = int(request.match_info["request_id"])
        except (ValueError, KeyError):
            raise web.HTTPBadRequest(text="Invalid request id")
        future = self._pending.get(request_id)
        if future is None or future.done():
            raise web.HTTPConflict(text="Capture request expired")
        if request.content_type != "image/jpeg":
            raise web.HTTPUnsupportedMediaType(text="Expected image/jpeg")
        if request.content_length is not None and request.content_length > MAX_JPEG_BYTES:
            raise web.HTTPRequestEntityTooLarge(max_size=MAX_JPEG_BYTES,
                                                actual_size=request.content_length)
        chunks = []
        size = 0
        async for chunk in request.content.iter_chunked(65536):
            size += len(chunk)
            if size > MAX_JPEG_BYTES:
                raise web.HTTPRequestEntityTooLarge(
                    max_size=MAX_JPEG_BYTES, actual_size=size)
            chunks.append(chunk)
        image = b"".join(chunks)
        if len(image) < 1024 or not image.startswith(b"\xff\xd8") or not image.endswith(b"\xff\xd9"):
            raise web.HTTPBadRequest(text="Invalid or incomplete JPEG")
        future.set_result(image)
        return web.json_response({"ok": True})

    async def capture_photo(self, timeout=CAPTURE_SECONDS):
        """Request exactly one photo. Returns bytes or None on timeout/error."""
        if not self.online() or len(self._pending) >= 2:
            return None
        self._request_id += 1
        request_id = self._request_id
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            self._commands.put_nowait(request_id)
        except asyncio.QueueFull:
            self._pending.pop(request_id, None)
            return None
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self._pending.pop(request_id, None)

    def register(self, app):
        app.router.add_get("/glasses/next", self.next_command)
        app.router.add_post("/glasses/photo/{request_id}", self.receive_photo)
