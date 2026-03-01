import json
from datetime import datetime

import paho.mqtt.client as mqtt
import psycopg2

DB_CONFIG = {
    "dbname": "{db_name}",
    "user": "{user_name}",
    "password": "{password}",
    "host": "localhost",
    "port": port_no,
}

MQTT_CONFIG = {
    "broker": "localhost",
    "port": port_no,
    "user": "",
    "pass": "",
    "topic": "{topic_name}",
}


def pick(payload, *keys):
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def normalize_payload(data):
    pm1_0_pcs = pick(data, "pm1_0_pcs")
    pm2_5_pcs = pick(data, "pm2_5_pcs")
    pm10_pcs = pick(data, "pm10_pcs")

    bin_0_3_0_5 = pick(data, "bin_0_3_0_5", "bin1", "channel1")
    bin_0_5_1_0 = pick(data, "bin_0_5_1_0", "bin2", "channel2")
    bin_1_0_2_5 = pick(data, "bin_1_0_2_5", "bin3", "channel3")
    bin_2_5_5_0 = pick(data, "bin_2_5_5_0", "bin4", "channel4")
    bin_5_0_10_0 = pick(data, "bin_5_0_10_0", "bin5", "channel5")

    if bin_1_0_2_5 is None and pm1_0_pcs is not None and pm2_5_pcs is not None:
        bin_1_0_2_5 = max(float(pm2_5_pcs) - float(pm1_0_pcs), 0.0)

    return {
        "device_id": pick(data, "device_id") or "ESP32_Node_1",

        "pm1_0_pcs": pm1_0_pcs,
        "pm2_5_pcs": pm2_5_pcs,
        "pm10_pcs": pm10_pcs,

        "pm1_0": pick(data, "pm1_0_ugm3", "pm1_0", "pm1"),
        "pm2_5": pick(data, "pm2_5_ugm3", "pm2_5", "pm25"),
        "pm10_0": pick(data, "pm10_ugm3", "pm10_0", "pm10"),

        "bin_0_3_0_5": bin_0_3_0_5,
        "bin_0_5_1_0": bin_0_5_1_0,
        "bin_1_0_2_5": bin_1_0_2_5,
        "bin_2_5_5_0": bin_2_5_5_0,
        "bin_5_0_10_0": bin_5_0_10_0,

        "temperature": pick(data, "temperature", "temp"),
        "humidity": pick(data, "humidity", "hum"),

        "state_code": pick(data, "state"),
        "valid": pick(data, "valid"),
    }


def save_to_db(data):
    try:
        row = normalize_payload(data)
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO sensor_data (
                time, device_id,
                pm1_0_pcs, pm2_5_pcs, pm10_pcs,
                pm1_0, pm2_5, pm10_0,
                bin_0_3_0_5, bin_0_5_1_0, bin_1_0_2_5, bin_2_5_5_0, bin_5_0_10_0,
                temperature, humidity,
                state_code, valid
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                datetime.now(),
                row["device_id"],
                row["pm1_0_pcs"],
                row["pm2_5_pcs"],
                row["pm10_pcs"],
                row["pm1_0"],
                row["pm2_5"],
                row["pm10_0"],
                row["bin_0_3_0_5"],
                row["bin_0_5_1_0"],
                row["bin_1_0_2_5"],
                row["bin_2_5_5_0"],
                row["bin_5_0_10_0"],
                row["temperature"],
                row["humidity"],
                row["state_code"],
                row["valid"],
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
        print(f"Data saved to DB: {row}")
    except Exception as e:
        print(f"Database Error: {e}")


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(MQTT_CONFIG["topic"])
    else:
        print(f"Failed to connect, return code {rc}")


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        print(f"Received: {payload}")
        save_to_db(payload)
    except Exception as e:
        print(f"Error parsing JSON: {e}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(MQTT_CONFIG["user"], MQTT_CONFIG["pass"])
client.on_connect = on_connect
client.on_message = on_message

print("Starting EnviroSense Subscriber...")
client.connect(MQTT_CONFIG["broker"], MQTT_CONFIG["port"], 60)
client.loop_forever()

