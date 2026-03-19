#include <Arduino.h>
#include <WiFi.h>
#include <ModbusMaster.h>
#include <PubSubClient.h>
#include "credentials.h"

#define LED_PIN 2
#define NEXTPM_RX_PIN 17
#define NEXTPM_TX_PIN 16
#define NEXTPM_BAUD 115200
#define NEXTPM_MODBUS_SLAVE_ID 1

const char* mqtt_topic = "esp32/sensor/data";
const char* mqtt_client_id = "ESP32Client_";
const uint16_t MQTT_BUFFER_SIZE = 1024;

const unsigned long SENSOR_POLL_INTERVAL_MS = 300000;  // 5 minutes
const unsigned long MQTT_PUBLISH_INTERVAL_MS = 300000; // 5 minutes

#define REG_FIRMWARE_VERSION 0x01
#define REG_STATUS 0x13
#define REG_START_PM10_CLOGGING 109

#define REG_START_PM1_10SEC_AVG_CNT 50
#define REG_START_PM2_5_10SEC_AVG_CNT 52
#define REG_START_PM10_10SEC_AVG_CNT 54

#define REG_START_PM1_10SEC_AVG_MASS 56
#define REG_START_PM2_5_10SEC_AVG_MASS 58
#define REG_START_PM10_10SEC_AVG_MASS 60

#define REG_START_PM1_60SEC_AVG_CNT 62
#define REG_START_PM2_5_60SEC_AVG_CNT 64
#define REG_START_PM10_60SEC_AVG_CNT 66

#define REG_START_PM1_60SEC_AVG_MASS 68
#define REG_START_PM2_5_60SEC_AVG_MASS 70
#define REG_START_PM10_60SEC_AVG_MASS 72

#define REG_START_PM1_15MIN_AVG_CNT 74
#define REG_START_PM2_5_15MIN_AVG_CNT 76
#define REG_START_PM10_15MIN_AVG_CNT 78

#define REG_START_PM1_15MIN_AVG_MASS 80
#define REG_START_PM2_5_15MIN_AVG_MASS 82
#define REG_START_PM10_15MIN_AVG_MASS 84

#define REG_START_02_05_QTY_10SEC_AVG 128
#define REG_START_05_10_QTY_10SEC_AVG 130
#define REG_START_10_25_QTY_10SEC_AVG 132
#define REG_START_25_50_QTY_10SEC_AVG 134
#define REG_START_50_100_QTY_10SEC_AVG 136

#define REG_FAN_RATIO 100
#define REG_HEATER_RATIO 101
#define REG_FAN_SPEED 102
#define REG_LASER_STATUS 103

#define REG_PM_HUMIDITY 106
#define REG_PM_TEMPERATURE 107

#define REG_EXT_CLC_TEMP 145
#define REG_EXT_CLC_RELATIVE_HUMIDITY 146

struct NextPmReading {
  uint16_t pm1_0_pcs;
  uint16_t pm2_5_pcs;
  uint16_t pm10_pcs;
  uint16_t pm1_0_ugm3;
  uint16_t pm2_5_ugm3;
  uint16_t pm10_ugm3;
  uint8_t state;
  unsigned long lastUpdateMs;
  bool valid;
};

HardwareSerial nextPmSerial(2);
ModbusMaster nextPmModbus;

NextPmReading latestReading = {0, 0, 0, 0, 0, 0, 0, 0, false};

float latestTemperature = NAN;
float latestHumidity = NAN;

float latestExtClcTemp = NAN;
float latestExtClcHumidity = NAN;

float latestFanRatio = NAN;
float latestHeaterRatio = NAN;
float latestFanSpeed = NAN;
float latestLaserStatus = NAN;

float latestPm10CloggingMg = NAN;

float pmCount10s_pm1_0 = NAN;
float pmCount10s_pm2_5 = NAN;
float pmCount10s_pm10 = NAN;
float pmMass10s_pm1_0 = NAN;
float pmMass10s_pm2_5 = NAN;
float pmMass10s_pm10 = NAN;

float pmCount60s_pm1_0 = NAN;
float pmCount60s_pm2_5 = NAN;
float pmCount60s_pm10 = NAN;
float pmMass60s_pm1_0 = NAN;
float pmMass60s_pm2_5 = NAN;
float pmMass60s_pm10 = NAN;

float pmCount15m_pm1_0 = NAN;
float pmCount15m_pm2_5 = NAN;
float pmCount15m_pm10 = NAN;
float pmMass15m_pm1_0 = NAN;
float pmMass15m_pm2_5 = NAN;
float pmMass15m_pm10 = NAN;

float bin_0_3_0_5 = NAN;
float bin_0_5_1_0 = NAN;
float bin_1_0_2_5 = NAN;
float bin_2_5_5_0 = NAN;
float bin_5_0_10_0 = NAN;

uint16_t latestFirmwareVersion = 0;
uint16_t latestStatusCode = 0;

bool trhValid = false;
bool binsValid = false;

unsigned long lastSensorPoll = 0;
unsigned long lastMsg = 0;

WiFiClient espClient;
PubSubClient client(espClient);

bool readU16Register(uint16_t reg, uint16_t& outValue) {
  uint8_t result = nextPmModbus.readHoldingRegisters(reg, 1);
  if (result != nextPmModbus.ku8MBSuccess) {
    return false;
  }
  outValue = nextPmModbus.getResponseBuffer(0);
  return true;
}

bool readU16Div1000(uint16_t reg, float& outValue) {
  uint16_t raw = 0;
  if (!readU16Register(reg, raw)) {
    return false;
  }
  outValue = raw / 1000.0f;
  return true;
}

bool readU16Div100(uint16_t reg, float& outValue) {
  uint16_t raw = 0;
  if (!readU16Register(reg, raw)) {
    return false;
  }
  outValue = raw / 100.0f;
  return true;
}

bool readAveragedPMFloat(uint16_t reg, float& outValue) {
  uint8_t result = nextPmModbus.readHoldingRegisters(reg, 2);
  if (result != nextPmModbus.ku8MBSuccess) {
    return false;
  }

  uint32_t hi = nextPmModbus.getResponseBuffer(0);
  uint32_t lo = nextPmModbus.getResponseBuffer(1);
  outValue = ((lo << 16) | hi) / 1000.0f;
  return true;
}

bool readPM10Clogging(float& outValue) {
  uint8_t result = nextPmModbus.readHoldingRegisters(REG_START_PM10_CLOGGING, 3);
  if (result != nextPmModbus.ku8MBSuccess) {
    return false;
  }

  uint32_t part1 = nextPmModbus.getResponseBuffer(0);
  uint32_t part2 = nextPmModbus.getResponseBuffer(1);
  uint32_t part3 = nextPmModbus.getResponseBuffer(2);
  uint64_t raw = ((uint64_t)part1 << 32) | ((uint64_t)part2 << 16) | part3;

  outValue = raw / 10.0f;
  if (outValue > 12.0f) {
    outValue = 12.0f;
  }
  return true;
}

bool pollSensorModbus() {
  bool ok = true;

  ok = ok && readU16Register(REG_FIRMWARE_VERSION, latestFirmwareVersion);
  ok = ok && readU16Register(REG_STATUS, latestStatusCode);
  ok = ok && readPM10Clogging(latestPm10CloggingMg);

  ok = ok && readAveragedPMFloat(REG_START_PM1_10SEC_AVG_CNT, pmCount10s_pm1_0);
  ok = ok && readAveragedPMFloat(REG_START_PM2_5_10SEC_AVG_CNT, pmCount10s_pm2_5);
  ok = ok && readAveragedPMFloat(REG_START_PM10_10SEC_AVG_CNT, pmCount10s_pm10);
  ok = ok && readAveragedPMFloat(REG_START_PM1_10SEC_AVG_MASS, pmMass10s_pm1_0);
  ok = ok && readAveragedPMFloat(REG_START_PM2_5_10SEC_AVG_MASS, pmMass10s_pm2_5);
  ok = ok && readAveragedPMFloat(REG_START_PM10_10SEC_AVG_MASS, pmMass10s_pm10);

  ok = ok && readAveragedPMFloat(REG_START_PM1_60SEC_AVG_CNT, pmCount60s_pm1_0);
  ok = ok && readAveragedPMFloat(REG_START_PM2_5_60SEC_AVG_CNT, pmCount60s_pm2_5);
  ok = ok && readAveragedPMFloat(REG_START_PM10_60SEC_AVG_CNT, pmCount60s_pm10);
  ok = ok && readAveragedPMFloat(REG_START_PM1_60SEC_AVG_MASS, pmMass60s_pm1_0);
  ok = ok && readAveragedPMFloat(REG_START_PM2_5_60SEC_AVG_MASS, pmMass60s_pm2_5);
  ok = ok && readAveragedPMFloat(REG_START_PM10_60SEC_AVG_MASS, pmMass60s_pm10);

  ok = ok && readAveragedPMFloat(REG_START_PM1_15MIN_AVG_CNT, pmCount15m_pm1_0);
  ok = ok && readAveragedPMFloat(REG_START_PM2_5_15MIN_AVG_CNT, pmCount15m_pm2_5);
  ok = ok && readAveragedPMFloat(REG_START_PM10_15MIN_AVG_CNT, pmCount15m_pm10);
  ok = ok && readAveragedPMFloat(REG_START_PM1_15MIN_AVG_MASS, pmMass15m_pm1_0);
  ok = ok && readAveragedPMFloat(REG_START_PM2_5_15MIN_AVG_MASS, pmMass15m_pm2_5);
  ok = ok && readAveragedPMFloat(REG_START_PM10_15MIN_AVG_MASS, pmMass15m_pm10);

  ok = ok && readU16Div1000(REG_START_02_05_QTY_10SEC_AVG, bin_0_3_0_5);
  ok = ok && readU16Div1000(REG_START_05_10_QTY_10SEC_AVG, bin_0_5_1_0);
  ok = ok && readU16Div1000(REG_START_10_25_QTY_10SEC_AVG, bin_1_0_2_5);
  ok = ok && readU16Div1000(REG_START_25_50_QTY_10SEC_AVG, bin_2_5_5_0);
  ok = ok && readU16Div1000(REG_START_50_100_QTY_10SEC_AVG, bin_5_0_10_0);

  ok = ok && readU16Div100(REG_PM_HUMIDITY, latestHumidity);
  ok = ok && readU16Div100(REG_PM_TEMPERATURE, latestTemperature);
  ok = ok && readU16Div100(REG_FAN_RATIO, latestFanRatio);
  ok = ok && readU16Div100(REG_HEATER_RATIO, latestHeaterRatio);
  ok = ok && readU16Div100(REG_FAN_SPEED, latestFanSpeed);
  ok = ok && readU16Div100(REG_LASER_STATUS, latestLaserStatus);
  ok = ok && readU16Div100(REG_EXT_CLC_TEMP, latestExtClcTemp);
  ok = ok && readU16Div100(REG_EXT_CLC_RELATIVE_HUMIDITY, latestExtClcHumidity);

  if (!ok) {
    latestReading.valid = false;
    trhValid = false;
    binsValid = false;
    return false;
  }

  trhValid = true;
  binsValid = true;

  latestReading.pm1_0_pcs = (uint16_t)roundf(pmCount60s_pm1_0);
  latestReading.pm2_5_pcs = (uint16_t)roundf(pmCount60s_pm2_5);
  latestReading.pm10_pcs = (uint16_t)roundf(pmCount60s_pm10);
  latestReading.pm1_0_ugm3 = (uint16_t)roundf(pmMass60s_pm1_0 * 10.0f);
  latestReading.pm2_5_ugm3 = (uint16_t)roundf(pmMass60s_pm2_5 * 10.0f);
  latestReading.pm10_ugm3 = (uint16_t)roundf(pmMass60s_pm10 * 10.0f);
  latestReading.state = (uint8_t)(latestStatusCode & 0xFF);
  latestReading.lastUpdateMs = millis();
  latestReading.valid = true;

  return true;
}

void logPublishedSnapshot() {
  Serial.print("[PUBLISHED] NextPM | Status: 0x");
  Serial.print(latestStatusCode, HEX);
  Serial.print(" | PM60s μg/m³ [1.0,2.5,10]: ");
  Serial.print(pmMass60s_pm1_0, 3);
  Serial.print(", ");
  Serial.print(pmMass60s_pm2_5, 3);
  Serial.print(", ");
  Serial.println(pmMass60s_pm10, 3);

  Serial.print("  Bins pcs/ml [0.3-0.5,0.5-1,1-2.5,2.5-5,5-10]: ");
  Serial.print(bin_0_3_0_5, 3);
  Serial.print(", ");
  Serial.print(bin_0_5_1_0, 3);
  Serial.print(", ");
  Serial.print(bin_1_0_2_5, 3);
  Serial.print(", ");
  Serial.print(bin_2_5_5_0, 3);
  Serial.print(", ");
  Serial.println(bin_5_0_10_0, 3);

  Serial.print("  Temp/Humidity: ");
  Serial.print(latestTemperature, 2);
  Serial.print(" C, ");
  Serial.print(latestHumidity, 2);
  Serial.println(" %");
}

void logExtraDiagnostics() {
  Serial.println("[EXTRA READS - NOT PUBLISHED] Diagnostics snapshot:");
  Serial.print("  Firmware: ");
  Serial.print(latestFirmwareVersion);
  Serial.print(" | PM10 clogging mg: ");
  Serial.println(latestPm10CloggingMg, 2);

  Serial.print("  PM10s μg/m³ [1.0,2.5,10]: ");
  Serial.print(pmMass10s_pm1_0, 3);
  Serial.print(", ");
  Serial.print(pmMass10s_pm2_5, 3);
  Serial.print(", ");
  Serial.println(pmMass10s_pm10, 3);

  Serial.print("  PM15m μg/m³ [1.0,2.5,10]: ");
  Serial.print(pmMass15m_pm1_0, 3);
  Serial.print(", ");
  Serial.print(pmMass15m_pm2_5, 3);
  Serial.print(", ");
  Serial.println(pmMass15m_pm10, 3);

  Serial.print("  Fan/Heater/FanSpeed/Laser: ");
  Serial.print(latestFanRatio, 2);
  Serial.print(" / ");
  Serial.print(latestHeaterRatio, 2);
  Serial.print(" / ");
  Serial.print(latestFanSpeed, 2);
  Serial.print(" / ");
  Serial.println(latestLaserStatus, 2);

  Serial.print("  External Compensated Temp/Humidity: ");
  Serial.print(latestExtClcTemp, 2);
  Serial.print(" C, ");
  Serial.print(latestExtClcHumidity, 2);
  Serial.println(" %");
}

void updateSensorReading() {
  if (millis() - lastSensorPoll <= SENSOR_POLL_INTERVAL_MS) {
    return;
  }

  lastSensorPoll = millis();

  if (!pollSensorModbus()) {
    Serial.println("NextPM Modbus read failed for one or more registers");
    return;
  }

  logPublishedSnapshot();
  logExtraDiagnostics();
}

String buildMqttPayload() {
  String payload = "{";
  payload += "\"sensor\":\"nextpm\"";

  // Publish order mirrors subscriber.py and PostgreSQL schema:
  // counts -> mass concentration -> bins -> temperature -> humidity -> state -> valid
  payload += ",\"pm1_0_pcs\":";
  payload += latestReading.pm1_0_pcs;
  payload += ",\"pm2_5_pcs\":";
  payload += latestReading.pm2_5_pcs;
  payload += ",\"pm10_pcs\":";
  payload += latestReading.pm10_pcs;

  payload += ",\"pm1_0_ugm3\":";
  payload += (latestReading.pm1_0_ugm3 / 10.0f);
  payload += ",\"pm2_5_ugm3\":";
  payload += (latestReading.pm2_5_ugm3 / 10.0f);
  payload += ",\"pm10_ugm3\":";
  payload += (latestReading.pm10_ugm3 / 10.0f);

  payload += ",\"bin_0_3_0_5\":";
  payload += isnan(bin_0_3_0_5) ? "null" : String(bin_0_3_0_5);
  payload += ",\"bin_0_5_1_0\":";
  payload += isnan(bin_0_5_1_0) ? "null" : String(bin_0_5_1_0);
  payload += ",\"bin_1_0_2_5\":";
  payload += isnan(bin_1_0_2_5) ? "null" : String(bin_1_0_2_5);
  payload += ",\"bin_2_5_5_0\":";
  payload += isnan(bin_2_5_5_0) ? "null" : String(bin_2_5_5_0);
  payload += ",\"bin_5_0_10_0\":";
  payload += isnan(bin_5_0_10_0) ? "null" : String(bin_5_0_10_0);

  payload += ",\"temperature\":";
  payload += trhValid ? String(latestTemperature) : "null";
  payload += ",\"humidity\":";
  payload += trhValid ? String(latestHumidity) : "null";

  payload += ",\"state\":";
  payload += (int)latestReading.state;
  payload += ",\"valid\":";
  payload += latestReading.valid ? "true" : "false";
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
  Serial.println("IP address:");
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

  nextPmSerial.begin(NEXTPM_BAUD, SERIAL_8E1, NEXTPM_RX_PIN, NEXTPM_TX_PIN);
  nextPmModbus.begin(NEXTPM_MODBUS_SLAVE_ID, nextPmSerial);
  Serial.print("NextPM Modbus ready on RX=");
  Serial.print(NEXTPM_RX_PIN);
  Serial.print(" TX=");
  Serial.println(NEXTPM_TX_PIN);

  setup_wifi();
  client.setServer(MQTT_SERVER, MQTT_PORT);
  bool mqttBufferOk = client.setBufferSize(MQTT_BUFFER_SIZE);
  Serial.print("MQTT buffer set to ");
  Serial.print(MQTT_BUFFER_SIZE);
  Serial.print(" bytes: ");
  Serial.println(mqttBufferOk ? "OK" : "FAILED");
}

void loop() {
  updateSensorReading();

  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  unsigned long now = millis();
  if (now - lastMsg > MQTT_PUBLISH_INTERVAL_MS) {
    lastMsg = now;
    if (!latestReading.valid) {
      Serial.println("No valid NextPM Modbus data yet; skipping MQTT publish");
      return;
    }

    digitalWrite(LED_PIN, HIGH);

    String payload = buildMqttPayload();
    Serial.print("Publishing message: ");
    Serial.println(payload);

    if (client.publish(mqtt_topic, payload.c_str())) {
      Serial.println("Message published successfully");
    } else {
      Serial.print("Message publish failed | connected=");
      Serial.print(client.connected() ? "true" : "false");
      Serial.print(" | state=");
      Serial.print(client.state());
      Serial.print(" | payload_len=");
      Serial.println(payload.length());

      if (!client.connected()) {
        reconnect();
      }

      if (client.publish(mqtt_topic, payload.c_str())) {
        Serial.println("Message publish retry successful");
      } else {
        Serial.println("Message publish retry failed");
      }
    }

    delay(500);
    digitalWrite(LED_PIN, LOW);
  }
}
