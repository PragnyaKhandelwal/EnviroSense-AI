#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include "credentials.h"

#define LED_PIN 2

const char* mqtt_topic = "esp32/sensor/data";
const char* mqtt_client_id = "ESP32Client_";

WiFiClient espClient;
PubSubClient client(espClient);

unsigned long lastMsg = 0;
int value = 0;

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
  
  setup_wifi();
  client.setServer(MQTT_SERVER, MQTT_PORT);
  
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  unsigned long now = millis();
  if (now - lastMsg > 5000) {  
    lastMsg = now;
    
    digitalWrite(LED_PIN, HIGH);
    Serial.println("LED is ON");
    
    String payload = "{";
    payload += "\"temperature\":"; payload += random(20, 30);
    payload += ",\"humidity\":"; payload += random(40, 80);
    payload += ",\"value\":"; payload += value;
    payload += "}";
    
    Serial.print("Publishing message: ");
    Serial.println(payload);
    
    // Publish to MQTT
    if (client.publish(mqtt_topic, payload.c_str())) {
      Serial.println("Message published successfully");
    } else {
      Serial.println("Message publish failed");
    }
    
    value++;
    
    delay(500);
    digitalWrite(LED_PIN, LOW);
    Serial.println("LED is OFF");
  }
}
