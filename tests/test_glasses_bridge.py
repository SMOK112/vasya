import asyncio
import time
import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from glasses_bridge import GlassesBridge


TOKEN = "1" * 64
HEADERS = {"Authorization": "Bearer " + TOKEN}


class GlassesBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bridge = GlassesBridge(token=TOKEN)
        app = web.Application(client_max_size=5 * 1024 * 1024)
        self.bridge.register(app)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_rejects_missing_and_wrong_device_token(self):
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            async with self.client.get("/glasses/next", headers=headers) as response:
                self.assertEqual(response.status, 403)
        async with self.client.post("/glasses/photo/1", headers={},
                                    data=b"x") as response:
            self.assertEqual(response.status, 403)

    async def test_single_photo_request_roundtrip(self):
        self.bridge._last_seen = time.monotonic()
        requested = asyncio.create_task(self.bridge.capture_photo(timeout=5))
        async with self.client.get("/glasses/next", headers=HEADERS) as response:
            self.assertEqual(response.status, 200)
            request_id = int((await response.text()).strip())
        self.assertGreater(request_id, 0)
        image = b"\xff\xd8" + b"a" * 1024 + b"\xff\xd9"
        async with self.client.post(
            f"/glasses/photo/{request_id}", headers={**HEADERS, "Content-Type": "image/jpeg"},
            data=image
        ) as response:
            self.assertEqual(response.status, 200)
        self.assertEqual(await requested, image)

    async def test_rejects_malformed_image_without_fulfilling_request(self):
        self.bridge._last_seen = time.monotonic()
        requested = asyncio.create_task(self.bridge.capture_photo(timeout=5))
        async with self.client.get("/glasses/next", headers=HEADERS) as response:
            request_id = int((await response.text()).strip())
        async with self.client.post(
            f"/glasses/photo/{request_id}", headers={**HEADERS, "Content-Type": "image/jpeg"},
            data=b"bad_jpeg" * 300
        ) as response:
            self.assertEqual(response.status, 400)
        self.assertFalse(requested.done())
        requested.cancel()
        try:
            await requested
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    unittest.main()
