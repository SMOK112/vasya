# Glasses camera bridge (experimental)

This is a separate review branch. **Not deployed to the VPS.** Do not merge/deploy until tested. It does not replace the existing phone microphone or audio output.

## Architecture

- XIAO ESP32S3 Sense / OV3660 connects to **2.4 GHz Wi-Fi** and independently initiates HTTPS requests to Vasya.
- It long-polls `GET /glasses/next` for a capture job. Idle reply is `0`.
- When requested, it takes **one JPEG** and sends authenticated `POST /glasses/photo/{id}`.
- Server keeps the JPEG in RAM, validates its size and JPEG markers, and provides it to the existing AI photo handler.
- Existing phone UI shows `Очки: выкл/вкл`. Glasses selection is **off by default**. If device capture times out, the server falls back to the phone camera.
- Audio input/output stays on the phone. Later the phone can route audio to a conventional Bluetooth earbud. ESP32-S3 is BLE-only and does not directly support classic Bluetooth A2DP.
- The glasses do **not** continuously transmit camera footage. Do not record people without their knowledge; plan a prominent recording indicator for the eventual enclosure.

## Before connecting the device

**Security:** The public server's existing `VASYA_TOKEN` remains independent. Make a new random 32-byte `GLASSES_TOKEN`; do not reuse the website token, Gemini/Polza key, or Wi-Fi password. The device token belongs only in the server's private environment and the local Arduino `secrets.h`.

Generate once, locally on the VPS:

```sh
openssl rand -hex 32
```

Add `GLASSES_TOKEN=<the-generated-value>` to `/opt/vasya/.env` using a private editor. Never paste it into chat, logs, GitHub, screenshots or shell history. Keep file permissions `600`. The token enables the two new HTTPS device endpoints; when unset or shorter than 32 characters, they return unavailable and the old application continues to work.

The existing deployment uses `bash /opt/vasya/update.sh` (git pull + restart). **Do not run it on this branch yet.** First audit the pull request, back up your server and verify that the production TLS certificate chain is signed by ISRG Root X1; otherwise update the trusted roots in the device firmware. TLS verification is intentional: don't switch to `setInsecure()`.

## Arduino setup (after server is configured)

1. Arduino IDE: select **XIAO_ESP32S3**, serial COM port, **OPI PSRAM**, Arduino ESP32 by Espressif Systems 3.3.12.
2. Save the entire `firmware/GlassesBridge` directory as an Arduino sketch folder.
3. Copy `secrets.example.h` to `secrets.h` in the same directory and fill the private Wi-Fi password and the **same unique** device token as the VPS `GLASSES_TOKEN`.
4. The ESP32 firmware uses a dedicated camera pin map for XIAO, with JPEG UXGA 1600×1200 to prioritize stability and reading small text over high-frame-rate streaming.
5. Flash via USB-C with no battery attached. Keep the board on a nonconductive bench while testing and open Serial Monitor at 115200 baud.
6. Verify logs: Wi-Fi connected, camera initialized, and `Glasses ready (no photo until requested)`.

The code ships the public ISRG Root X1 certificate. HTTPS time validation requires NTP; it fails closed if time sync fails. If the VPS later uses a different certificate authority, update the CA rather than disabling verification.

## User acceptance test

1. With the original phone site running, verify its normal voice/text/photo functions still work while **glasses mode is off**.
2. With the powered ESP32 online, enable `Очки: вкл` in the phone UI.
3. Point the camera at a non-sensitive printed sample page, tap `Глянь`, and confirm the server answers about the ESP32 photo (not the phone).
4. Disable glasses and confirm phone photos work again.
5. Unplug ESP32, enable glasses, trigger `Глянь`; phone should be used as fallback after a device timeout.
6. Test wrong or absent device token: `/glasses/next` must be rejected, with no image returned. Verify the certificate is checked.
7. Check server memory, logs and excessive request sizes. Watch for upload failure on poor Wi-Fi. Check the local camera's image focus separately.

**Known limitations:** This prototype has no on-glasses physical button, no local battery, no wireless charging, no microphone capture on ESP32, no confirmed hardware enclosure, and no build/test run against a live VPS yet. The UI currently lets you select the source even if the device is offline; it reports fallback. For final glasses use, add explicit on-device capture indication, measured thermal and power tests, a secure case insulating contacts, and professional LiPo battery integration.

## Rollback

Since changes are staged in a separate branch, production stays unchanged unless merged and deployed. If deployment fails, restore the previous Git revision (after backing up your VPS) and restart the original `vasya` service. Never remove `.env` or API keys during rollback.
