#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include "credentials.h"

#define LED_PIN 2
#define NEXTPM_RX_PIN 17     
#define NEXTPM_TX_PIN 16    
#define NEXTPM_BAUD 115200

const char* mqtt_topic = "esp32/sensor/data";
const char* mqtt_client_id = "ESP32Client_";

// NextPM Protocol Constants 
#define NEXTPM_ADDRESS 0x81
#define NEXTPM_CMD_PM10S 0x11      // 10-second average
#define NEXTPM_CMD_PM60S 0x12      // 60-second average
#define NEXTPM_CMD_PM900S 0x13     // 900-second average
#define NEXTPM_CMD_TRH 0x14        // Temperature & humidity
#define NEXTPM_CMD_STATE 0x16      // Sensor state readings
#define NEXTPM_PM_FRAME_LENGTH 16  // PM response is 16 bytes
#define NEXTPM_TRH_FRAME_LENGTH 8  // T/RH response is 8 bytes
#define NEXTPM_STATE_FRAME_LENGTH 4 // State response: ADDR CMD STATE CHK

const unsigned long SENSOR_POLL_INTERVAL_MS = 60000;
const unsigned long MQTT_PUBLISH_INTERVAL_MS = 60000;

// Struct for NextPM data
struct NextPmReading {
  uint16_t pm1_0_pcs;    // pcs/L
  uint16_t pm2_5_pcs;    // pcs/L
  uint16_t pm10_pcs;     // pcs/L
  uint16_t pm1_0_ugm3;   // μg/m³
  uint16_t pm2_5_ugm3;   // μg/m³
  uint16_t pm10_ugm3;    // μg/m³
  uint8_t state;
  unsigned long lastUpdateMs;
  bool valid;
};

// Hardware Serial for NextPM
HardwareSerial nextPmSerial(2);  // UART2
NextPmReading latestReading = {0, 0, 0, 0, 0, 0, 0, 0, false};
float latestTemperature = NAN;
float latestHumidity = NAN;
bool trhValid = false;
unsigned long lastModbusRequest = 0;
unsigned long lastMsg = 0;

// WiFi & MQTT
WiFiClient espClient;
PubSubClient client(espClient);

// Calculate NextPM checksum: 0x100 - ((sum of all bytes) % 256)
uint8_t calculateNextPMChecksum(uint8_t cmd) {
  uint16_t sum = NEXTPM_ADDRESS + cmd;
  return (0x100 - (sum % 256)) & 0xFF;
}

void sendNextPMCommand(uint8_t cmd) {
  uint8_t command[3];
  command[0] = NEXTPM_ADDRESS;
  command[1] = cmd;
  command[2] = calculateNextPMChecksum(command[1]);
  
  nextPmSerial.write(command, 3);
  Serial.print("NextPM command sent: 0x");
  Serial.print(command[0], HEX);
  Serial.print(" 0x");
  Serial.print(command[1], HEX);
  Serial.print(" 0x");
  Serial.println(command[2], HEX);
}

// Parse NextPM PM frame response (16 bytes)
// Format: 0x81 CMD STATE PM1_pcs(2) PM2.5_pcs(2) PM10_pcs(2) PM1_ugm3(2) PM2.5_ugm3(2) PM10_ugm3(2) CHECKSUM
bool parseNextPMPmFrame(NextPmReading& output) {
  if (nextPmSerial.available() < NEXTPM_PM_FRAME_LENGTH) {
    return false;
  }
  
  // Look for frame start (0x81)
  while (nextPmSerial.available() > 0) {
    uint8_t byte = nextPmSerial.peek();
    if (byte == NEXTPM_ADDRESS) {
      break;
    }
    nextPmSerial.read();  // Skip garbage bytes
  }
  
  if (nextPmSerial.available() < NEXTPM_PM_FRAME_LENGTH) {
    return false;
  }
  
  // Read 16-byte frame
  uint8_t frame[16];
  for (uint8_t i = 0; i < 16; i++) {
    frame[i] = nextPmSerial.read();
  }
  
  // Debug: Log frame
  Serial.print("NextPM frame (16 bytes): ");
  for (uint8_t i = 0; i < 16; i++) {
    Serial.print("0x");
    if (frame[i] < 16) Serial.print("0");
    Serial.print(frame[i], HEX);
    Serial.print(" ");
  }
  Serial.println();
  
  // Validate frame structure
  if (frame[0] != NEXTPM_ADDRESS) {
    Serial.println("Invalid frame start");
    return false;
  }
  
  if (frame[1] != NEXTPM_CMD_PM10S && frame[1] != NEXTPM_CMD_PM60S && frame[1] != NEXTPM_CMD_PM900S) {
    Serial.print("Invalid command ID: 0x");
    Serial.println(frame[1], HEX);
    return false;
  }
  
  // Verify checksum
  uint16_t sum = 0;
  for (uint8_t i = 0; i < 15; i++) {
    sum += frame[i];
  }
  uint8_t expectedChecksum = (0x100 - (sum % 256)) & 0xFF;
  if (frame[15] != expectedChecksum) {
    Serial.print("Checksum mismatch: got 0x");
    Serial.print(frame[15], HEX);
    Serial.print(" expected 0x");
    Serial.println(expectedChecksum, HEX);
    return false;
  }
  
  // Extract state code (byte 2)
  output.state = frame[2];
  
  // Extract PM values (big-endian 16-bit values)
  // Bytes 3-4: PM1 pcs/L
  output.pm1_0_pcs = ((uint16_t)frame[3] << 8) | frame[4];
  
  // Bytes 5-6: PM2.5 pcs/L
  output.pm2_5_pcs = ((uint16_t)frame[5] << 8) | frame[6];
  
  // Bytes 7-8: PM10 pcs/L
  output.pm10_pcs = ((uint16_t)frame[7] << 8) | frame[8];
  
  // Bytes 9-10: PM1 μg/m³ (divide by 10)
  output.pm1_0_ugm3 = ((uint16_t)frame[9] << 8) | frame[10];
  
  // Bytes 11-12: PM2.5 μg/m³ (divide by 10)
  output.pm2_5_ugm3 = ((uint16_t)frame[11] << 8) | frame[12];
  
  // Bytes 13-14: PM10 μg/m³ (divide by 10)
  output.pm10_ugm3 = ((uint16_t)frame[13] << 8) | frame[14];
  
  output.lastUpdateMs = millis();
  output.valid = true;
  
  return true;
}

bool parseNextPMTrhFrame(float& temperature, float& humidity) {
  if (nextPmSerial.available() < NEXTPM_TRH_FRAME_LENGTH) {
    return false;
  }

  while (nextPmSerial.available() > 0) {
    uint8_t byte = nextPmSerial.peek();
    if (byte == NEXTPM_ADDRESS) {
      break;
    }
    nextPmSerial.read();
  }

  if (nextPmSerial.available() < NEXTPM_TRH_FRAME_LENGTH) {
    return false;
  }

  uint8_t frame[8];
  for (uint8_t i = 0; i < 8; i++) {
    frame[i] = nextPmSerial.read();
  }

  if (frame[0] != NEXTPM_ADDRESS) {
    return false;
  }

  if (frame[1] == NEXTPM_CMD_STATE) {
    if (nextPmSerial.available() < (NEXTPM_STATE_FRAME_LENGTH - 1)) {
      return false;
    }

    uint8_t stateFrame[4];
    stateFrame[0] = frame[0];
    stateFrame[1] = frame[1];
    for (uint8_t i = 2; i < 4; i++) {
      stateFrame[i] = nextPmSerial.read();
    }

    uint16_t stateSum = stateFrame[0] + stateFrame[1] + stateFrame[2];
    uint8_t expectedStateChecksum = (0x100 - (stateSum % 256)) & 0xFF;
    if (stateFrame[3] == expectedStateChecksum) {
      latestReading.state = stateFrame[2];
      Serial.print("NextPM state frame during T/RH read, state=0x");
      Serial.println(latestReading.state, HEX);
    }
    return false;
  }

  if (frame[1] != NEXTPM_CMD_TRH) {
    return false;
  }

  uint16_t sum = 0;
  for (uint8_t i = 0; i < 7; i++) {
    sum += frame[i];
  }
  uint8_t expectedChecksum = (0x100 - (sum % 256)) & 0xFF;
  if (frame[7] != expectedChecksum) {
    Serial.print("T/RH checksum mismatch: got 0x");
    Serial.print(frame[7], HEX);
    Serial.print(" expected 0x");
    Serial.println(expectedChecksum, HEX);
    return false;
  }

  uint16_t rawTemp = ((uint16_t)frame[3] << 8) | frame[4];
  uint16_t rawHumidity = ((uint16_t)frame[5] << 8) | frame[6];

  temperature = rawTemp / 100.0;
  humidity = rawHumidity / 100.0;
  return true;
}

bool waitForNextPMPmResponse(uint16_t timeoutMs) {
  unsigned long start = millis();
  while (millis() - start < timeoutMs) {
    if (parseNextPMPmFrame(latestReading)) {
      return true;
    }
    delay(2);
  }
  return false;
}

bool waitForNextPMTrhResponse(uint16_t timeoutMs) {
  float temperature = NAN;
  float humidity = NAN;
  unsigned long start = millis();
  while (millis() - start < timeoutMs) {
    if (parseNextPMTrhFrame(temperature, humidity)) {
      latestTemperature = temperature;
      latestHumidity = humidity;
      trhValid = true;
      return true;
    }
    delay(2);
  }
  return false;
}

void updateSensorReading() {
  bool gotNewPm = false;
  bool gotNewTrh = false;

  // Poll PM + T/RH every 60 seconds
  if (millis() - lastModbusRequest > SENSOR_POLL_INTERVAL_MS) {
    sendNextPMCommand(NEXTPM_CMD_PM60S);
    gotNewPm = waitForNextPMPmResponse(400);

    sendNextPMCommand(NEXTPM_CMD_TRH);
    gotNewTrh = waitForNextPMTrhResponse(250);

    lastModbusRequest = millis();
  }

  if (gotNewPm || gotNewTrh) {
    if (!latestReading.valid) {
      Serial.println("NextPM poll completed, but PM frame is not valid yet");
      return;
    }

    Serial.print("NextPM updated | State: 0x");
    Serial.print(latestReading.state, HEX);
    Serial.print(" | PM1.0(pcs/L): ");
    Serial.print(latestReading.pm1_0_pcs);
    Serial.print(" | PM2.5(pcs/L): ");
    Serial.print(latestReading.pm2_5_pcs);
    Serial.print(" | PM10(pcs/L): ");
    Serial.println(latestReading.pm10_pcs);
    
    Serial.print("  PM1.0(μg/m³): ");
    Serial.print(latestReading.pm1_0_ugm3 / 10.0);
    Serial.print(" | PM2.5(μg/m³): ");
    Serial.print(latestReading.pm2_5_ugm3 / 10.0);
    Serial.print(" | PM10(μg/m³): ");
    Serial.println(latestReading.pm10_ugm3 / 10.0);

    if (trhValid) {
      Serial.print("  Temp(°C): ");
      Serial.print(latestTemperature);
      Serial.print(" | RH(%): ");
      Serial.println(latestHumidity);
    } else {
      Serial.println("  Temp/RH not available yet");
    }
  }
}

String buildMqttPayload() {
  String payload = "{";
  payload += "\"sensor\":\"nextpm\"";
  payload += ",\"pm1_0_pcs\":";
  payload += latestReading.pm1_0_pcs;
  payload += ",\"pm2_5_pcs\":";
  payload += latestReading.pm2_5_pcs;
  payload += ",\"pm10_pcs\":";
  payload += latestReading.pm10_pcs;
  payload += ",\"pm1_0_ugm3\":";
  payload += (latestReading.pm1_0_ugm3 / 10.0);
  payload += ",\"pm2_5_ugm3\":";
  payload += (latestReading.pm2_5_ugm3 / 10.0);
  payload += ",\"pm10_ugm3\":";
  payload += (latestReading.pm10_ugm3 / 10.0);
  payload += ",\"state\":";
  payload += (int)latestReading.state;
  payload += ",\"valid\":";
  payload += latestReading.valid ? "true" : "false";
  payload += ",\"temperature\":";
  if (trhValid) {
    payload += latestTemperature;
  } else {
    payload += "null";
  }
  payload += ",\"humidity\":";
  if (trhValid) {
    payload += latestHumidity;
  } else {
    payload += "null";
  }
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

  // NextPM requires: 115200 baud, 8 bits, EVEN parity, 1 stop bit
  nextPmSerial.begin(NEXTPM_BAUD, SERIAL_8E1, NEXTPM_RX_PIN, NEXTPM_TX_PIN);
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
  if (now - lastMsg > MQTT_PUBLISH_INTERVAL_MS) {  
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
