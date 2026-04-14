# ESP32 Next PM Monitor

## Setup Instructions

### 1. Configure Credentials

Before building and uploading the code:

1. Copy the example credentials file:
   ```bash
   cp include/credentials.example.h include/credentials.h
   ```

2. Edit `include/credentials.h` and add your WiFi credentials:
   ```cpp
   const char* WIFI_SSID = "YOUR_WIFI_SSID";
   const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
   ```

3. (Optional) If using a private MQTT broker, update the MQTT settings in `credentials.h`

### 2. Build and Upload

Build the project:
```bash
pio run
```

Upload to ESP32:
```bash
pio run --target upload
```

### 3. Monitor Serial Output

```bash
pio device monitor
```

## MQTT Topics

- **Publish**: `esp32/sensor/data` - Publishes sensor data every 5 seconds in JSON format

## Notes

- The `credentials.h` file is in `.gitignore` and will not be committed to version control
- Always use the example file as a template when setting up a new environment
