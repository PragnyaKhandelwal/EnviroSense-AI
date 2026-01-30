#include <Arduino.h>

#define LED_PIN 2

int myFunction(int, int);

void setup() {
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(115200);
  Serial.println("Hello, ESP32!");
}

void loop() {
  digitalWrite(LED_PIN, HIGH);
  Serial.println("LED is ON");
  delay(1000);

  digitalWrite(LED_PIN, LOW);
  Serial.println("LED is OFF");
  delay(2000);
}
