// esp32cam_ws_camera_min.ino
// Minimal firmware for AI-Thinker ESP32-CAM (OV2640):
// - Connect to Wi-Fi
// - Stream JPEG frames to FastAPI server via WebSocket: /ws/camera
//
// Tested target: "AI Thinker ESP32-CAM" board in Arduino IDE.

#include <WiFi.h>
#include <esp_camera.h>
#include <ArduinoWebsockets.h>

using namespace websockets;

// Select camera model pins
#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

// ===== WiFi / Server (EDIT THESE) =====
static const char* WIFI_SSID = "2018";
static const char* WIFI_PASS = "20182018";

// Use your PC LAN IP (NOT 127.0.0.1). Example: "192.168.1.100"
static const char* SERVER_HOST = "192.168.1.106";
static const uint16_t SERVER_PORT = 8081;
static const char* CAM_WS_PATH = "/ws/camera";

// ===== Video params =====
static const framesize_t FRAME_SIZE = FRAMESIZE_VGA; // 640x480
static const int JPEG_QUALITY = 15;                  // 0-63, lower is better quality
static const int FB_COUNT = 2;
static const int TARGET_FPS = 15;

WebsocketsClient wsCam;

static void connect_wifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.print("[WIFI] Connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print('.');
  }

  Serial.println();
  Serial.print("[WIFI] Connected. IP=");
  Serial.println(WiFi.localIP());
}

static bool init_camera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;

  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  config.frame_size = FRAME_SIZE;
  config.jpeg_quality = JPEG_QUALITY;
  config.fb_count = FB_COUNT;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[CAM] init failed: 0x%x\n", err);
    return false;
  }

  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    s->set_hmirror(s, 1);
    s->set_vflip(s, 0);
  }

  Serial.println("[CAM] init OK");
  return true;
}

static bool ensure_ws_connected() {
  if (wsCam.available()) return true;

  Serial.printf("[WS] Connecting ws://%s:%u%s\n", SERVER_HOST, SERVER_PORT, CAM_WS_PATH);
  bool ok = wsCam.connect(SERVER_HOST, SERVER_PORT, CAM_WS_PATH);
  Serial.println(ok ? "[WS] connected" : "[WS] connect failed");
  return ok;
}

void setup() {
  Serial.begin(115200);
  delay(500);

  wsCam.onEvent([](WebsocketsEvent ev, String) {
    if (ev == WebsocketsEvent::ConnectionOpened) Serial.println("[WS] open");
    if (ev == WebsocketsEvent::ConnectionClosed) Serial.println("[WS] closed");
  });

  connect_wifi();

  if (!init_camera()) {
    Serial.println("[FATAL] camera init failed. Rebooting in 3s...");
    delay(3000);
    ESP.restart();
  }
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WIFI] disconnected. Reconnecting...");
    WiFi.disconnect();
    delay(300);
    connect_wifi();
  }

  ensure_ws_connected();
  wsCam.poll();

  if (!wsCam.available()) {
    delay(200);
    return;
  }

  static unsigned long lastSendMs = 0;
  const unsigned long intervalMs = (TARGET_FPS > 0) ? (1000UL / (unsigned long)TARGET_FPS) : 0;
  unsigned long now = millis();
  if (intervalMs > 0 && (now - lastSendMs) < intervalMs) {
    delay(1);
    return;
  }

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb || fb->format != PIXFORMAT_JPEG) {
    if (fb) esp_camera_fb_return(fb);
    delay(5);
    return;
  }

  bool ok = wsCam.sendBinary((const char*)fb->buf, fb->len);
  esp_camera_fb_return(fb);

  if (!ok) {
    Serial.println("[WS] send failed. closing...");
    wsCam.close();
    delay(200);
  }

  lastSendMs = now;
}
