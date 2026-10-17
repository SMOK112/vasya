/*
 * Vasya glasses camera bridge — XIAO ESP32S3 Sense / OV3660.
 * Only captures on an explicit request from the authorized Vasya server.
 * No always-on image stream. HTTPS certificate and hostname are verified.
 * Arduino IDE: Tools > Board XIAO_ESP32S3, PSRAM OPI PSRAM, monitor 115200.
 * Fill your own local secrets.h; never commit secrets.h to GitHub.
 */
#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <time.h>
#include "esp_camera.h"
#include "secrets.h"
#include "root_ca.h"

static_assert(sizeof(GLASSES_TOKEN) >= 33, "Provide >= 32 character unique GLASSES_TOKEN");

static const uint32_t MAX_FRAME_SIZE = 4UL * 1024 * 1024;
static unsigned long lastReconnect = 0;

// XIAO ESP32S3 Sense camera pinout (not the generic ESP32-CAM pinout).
bool startCamera() {
  if (!psramFound()) { Serial.println("PSRAM required"); return false; }
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = 15; c.pin_d1 = 17; c.pin_d2 = 18; c.pin_d3 = 16;
  c.pin_d4 = 14; c.pin_d5 = 12; c.pin_d6 = 11; c.pin_d7 = 48;
  c.pin_xclk = 10; c.pin_pclk = 13; c.pin_vsync = 38; c.pin_href = 47;
  c.pin_sccb_sda = 40; c.pin_sccb_scl = 39;
  c.pin_pwdn = -1; c.pin_reset = -1;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = FRAMESIZE_UXGA; // 1600x1200; QXGA often unstable on this board
  c.jpeg_quality = 12;
  c.fb_count = 1;
  c.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK) {
    Serial.printf("Camera failed: 0x%x\n", err);
    return false;
  }
  sensor_t* s = esp_camera_sensor_get();
  if (s && s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);
  }
  Serial.println("Camera initialized");
  return true;
}

bool connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
    delay(250);
  }
  if (WiFi.status() != WL_CONNECTED) {
    Serial.printf("Wi-Fi failed: %d\n", WiFi.status());
    return false;
  }
  Serial.print("Wi-Fi IP: "); Serial.println(WiFi.localIP());
  return true;
}

bool syncClock() {
  configTime(0, 0, "pool.ntp.org", "time.google.com");
  time_t now = time(nullptr);
  unsigned long start = millis();
  while (now < 1700000000 && millis() - start < 20000) {
    delay(300);
    now = time(nullptr);
  }
  if (now < 1700000000) {
    Serial.println("No trusted clock; refusing HTTPS");
    return false;
  }
  return true;
}

void captureAndSend(long id) {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) { Serial.println("Capture failed"); return; }
  if (fb->format != PIXFORMAT_JPEG || fb->len < 1024 || fb->len > MAX_FRAME_SIZE) {
    Serial.println("Invalid JPEG");
    esp_camera_fb_return(fb);
    return;
  }

  WiFiClientSecure client;
  client.setCACert(ISRG_ROOT_X1);
  client.setTimeout(15000);
  HTTPClient http;
  String url = String(SERVER_URL) + "/glasses/photo/" + String(id);
  if (!http.begin(client, url)) {
    Serial.println("Could not create HTTPS request");
    esp_camera_fb_return(fb);
    return;
  }
  http.setTimeout(20000);
  http.addHeader("Authorization", String("Bearer ") + GLASSES_TOKEN);
  http.addHeader("Content-Type", "image/jpeg");
  int code = http.POST(fb->buf, fb->len);
  Serial.printf("Photo sent; HTTP %d; bytes %u\n", code, (unsigned)fb->len);
  http.end();
  esp_camera_fb_return(fb);
}

void setup() {
  Serial.begin(115200);
  delay(700);
  if (!connectWifi()) return;
  if (!syncClock()) return;
  if (!startCamera()) return;
  Serial.println("Glasses ready (no photo until requested)");
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    if (millis() - lastReconnect > 5000) {
      lastReconnect = millis();
      WiFi.reconnect();
    }
    delay(500);
    return;
  }
  if (time(nullptr) < 1700000000) {
    syncClock();
    delay(1000);
    return;
  }
  WiFiClientSecure client;
  client.setCACert(ISRG_ROOT_X1);
  client.setTimeout(20000);
  HTTPClient http;
  if (!http.begin(client, String(SERVER_URL) + "/glasses/next")) {
    delay(2000);
    return;
  }
  http.setTimeout(20000);
  http.addHeader("Authorization", String("Bearer ") + GLASSES_TOKEN);
  int code = http.GET();
  if (code == HTTP_CODE_OK) {
    String response = http.getString();
    response.trim();
    long id = response.toInt();
    http.end();
    if (id > 0) captureAndSend(id);
  } else {
    Serial.printf("Poll HTTP %d\n", code);
    http.end();
    delay(2000);
  }
}
