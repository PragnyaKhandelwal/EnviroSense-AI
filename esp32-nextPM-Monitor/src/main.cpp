#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include "credentials.h"

#define LED_PIN 2
#define NEXTPM_RX_PIN 16
#define NEXTPM_TX_PIN 17
#define NEXTPM_BAUD 9600

const char* mqtt_topic = "esp32/sensor/data";
const char* mqtt_client_id = "ESP32Client_";

WiFiClient espClient;
PubSubClient client(espClient);
HardwareSerial nextPmSerial(2);

unsigned long lastMsg = 0;

struct NextPmReading {
  uint16_t pm1_0 = 0;
  uint16_t pm2_5 = 0;
  uint16_t pm10 = 0;
  uint32_t lastUpdateMs = 0;
  bool valid = false;
};

NextPmReading latestReading;

bool readNextPmFrame(NextPmReading& output) {
  static uint8_t frame[32];
  static uint8_t index = 0;

  while (nextPmSerial.available() > 0) {
    uint8_t byteIn = static_cast<uint8_t>(nextPmSerial.read());

    if (index == 0) {
      if (byteIn != 0x42) {
        continue;
      }
      frame[index++] = byteIn;
      continue;
    }

    if (index == 1) {
      if (byteIn != 0x4D) {
        index = 0;
        continue;
      }
      frame[index++] = byteIn;
      continue;
    }

    frame[index++] = byteIn;

    if (index < sizeof(frame)) {
      continue;
    }

    index = 0;

    const uint16_t frameLength = (static_cast<uint16_t>(frame[2]) << 8) | frame[3];
    if (frameLength != 28) {
      continue;
    }

    uint16_t checksum = 0;
    for (uint8_t i = 0; i < 30; i++) {
      checksum += frame[i];
    }

    const uint16_t expectedChecksum = (static_cast<uint16_t>(frame[30]) << 8) | frame[31];
    if (checksum != expectedChecksum) {
      continue;
    }

    output.pm1_0 = (static_cast<uint16_t>(frame[10]) << 8) | frame[11];
    output.pm2_5 = (static_cast<uint16_t>(frame[12]) << 8) | frame[13];
    output.pm10 = (static_cast<uint16_t>(frame[14]) << 8) | frame[15];
    output.lastUpdateMs = millis();
    output.valid = true;
    return true;
  }

  return false;
}

void updateSensorReading() {
  NextPmReading reading;
  if (readNextPmFrame(reading)) {
    latestReading = reading;
    Serial.print("NextPM updated PM1.0=");
    Serial.print(latestReading.pm1_0);
    Serial.print(" PM2.5=");
    Serial.print(latestReading.pm2_5);
    Serial.print(" PM10=");
    Serial.println(latestReading.pm10);
  }
}

String buildMqttPayload() {
  String payload = "{";
  payload += "\"sensor\":\"nextpm\"";
  payload += ",\"pm1_0\":";
  payload += latestReading.pm1_0;
  payload += ",\"pm2_5\":";
  payload += latestReading.pm2_5;
  payload += ",\"pm10\":";
  payload += latestReading.pm10;
  payload += ",\"valid\":";
  payload += latestReading.valid ? "true" : "false";
  payload += ",\"uptime_ms\":";
  payload += millis();
  payload += "}";
  return payload;
}

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(WIFI_SSID);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    
    String clientId = mqtt_client_id;
    clientId += String(random(0xffff), HEX);
    
    if (client.connect(clientId.c_str(), MQTT_USER, MQTT_PASSWORD)) {
      Serial.println("connected");
      
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(115200);
  Serial.println("Hello, ESP32!");

  nextPmSerial.begin(NEXTPM_BAUD, SERIAL_8N1, NEXTPM_RX_PIN, NEXTPM_TX_PIN);
  Serial.print("NextPM UART ready on RX=");
  Serial.print(NEXTPM_RX_PIN);
  Serial.print(" TX=");
  Serial.println(NEXTPM_TX_PIN);
  
  setup_wifi();
  client.setServer(MQTT_SERVER, MQTT_PORT);
  
}

void loop() {
  updateSensorReading();

  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  unsigned long now = millis();
  if (now - lastMsg > 5000) {  
    lastMsg = now;
    if (!latestReading.valid) {
      Serial.println("No valid NextPM frame yet; skipping MQTT publish");
      return;
    }
    
    digitalWrite(LED_PIN, HIGH);
    Serial.println("LED is ON");
    
    String payload = buildMqttPayload();
    
    Serial.print("Publishing message: ");
    Serial.println(payload);
    
    if (client.publish(mqtt_topic, payload.c_str())) {
      Serial.println("Message published successfully");
    } else {
      Serial.println("Message publish failed");
    }

    delay(500);
    digitalWrite(LED_PIN, LOW);
    Serial.println("LED is OFF");
  }
}
