from __future__ import annotations

import os
import math
from datetime import datetime, timezone
from typing import Any
import sys
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd

import psycopg2
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from psycopg2.extras import RealDictCursor

load_dotenv()
DEFAULT_DEVICE = os.getenv("DEFAULT_DEVICE_ID", "ESP32_Node_1")
REGIME_LABELS = {
    0: "Post-Rain Clean",
    1: "Stable Indoor",
    2: "Traffic Spike",
    3: "Cooking Event",
    4: "Dust Influx",
}

SHARED_UTILS_DIR = Path('/home/shared/envirosense')
if str(SHARED_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_UTILS_DIR))


def _db_config() -> dict[str, Any]:
    return {
        "dbname": os.getenv("DB_NAME", "envirosense"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
    }


def _connect():
    return psycopg2.connect(**_db_config())


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _fmt_ts(ts: datetime | None) -> str:
    if ts is None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _build_forecast(pm25_now: float) -> tuple[list[dict[str, float | int | list[float]]], dict[str, list[dict[str, float | int]]]]:
    forecast: list[dict[str, float | int | list[float]]] = []
    history: list[dict[str, float | int]] = []
    future: list[dict[str, float | int]] = []

    for i in range(60):
        t = i - 60
        val = max(1.0, pm25_now - 5 + math.sin(i / 7) * 2.2)
        history.append({"t": t, "value": round(val, 1)})

    for i in range(31):
        trend = max(1.0, pm25_now + i * 0.08 + math.sin(i / 4) * 1.2)
        ci = 2.0 + i * 0.1
        low = round(max(0.1, trend - ci), 1)
        high = round(trend + ci, 1)
        forecast.append(
            {
                "minute": i,
                "pm25": round(trend, 1),
                "lower": low,
                "upper": high,
                "band": [low, high],
            }
        )

    for i in range(60):
        t = i + 1
        trend = max(1.0, pm25_now + i * 0.06 + math.sin((i + 5) / 8) * 1.6)
        ci = 2.0 + i * 0.08
        future.append(
            {
                "t": t,
                "value": round(trend, 1),
                "lower": round(max(0.1, trend - ci), 1),
                "upper": round(trend + ci, 1),
            }
        )

    return forecast, {"history": history, "forecast": future}


def _build_timeline(pm25_now: float, temp_now: float, hum_now: float) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for h in range(24):
        pm25 = max(1.0, pm25_now + math.sin((h - 6) / 24 * math.pi * 2) * 4.0)
        temp = temp_now + math.sin((h - 6) / 24 * math.pi * 2) * 3.0
        hum = hum_now - math.sin((h - 6) / 24 * math.pi * 2) * 8.0
        rows.append(
            {
                "hour": h,
                "pm25": round(pm25, 1),
                "temp": round(temp, 1),
                "humidity": round(hum, 1),
            }
        )
    return rows


app = FastAPI(title="EnviroSense API", version="1.0.0")

allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/pipeline")
def get_pipeline(
    device: str = Query(DEFAULT_DEVICE, min_length=1),
    tick: int = Query(0, ge=0),
) -> dict[str, Any]:
    del tick

    try:
        with _connect() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        time,
                        device_id,
                        pm2_5,
                        temperature,
                        humidity,
                        quality_uptime_pct,
                        quality_valid_pct,
                        drift_z_score,
                        drift_alert,
                        reliability_score,
                        is_anomaly,
                        anomaly_score,
                        cluster_label,
                        status_code,
                        prediction_confidence
                    FROM model_features
                    WHERE device_id = %s
                    ORDER BY time DESC
                    LIMIT 1
                    """,
                    (device,),
                )
                latest = cur.fetchone()

                if latest is None:
                    cur.execute(
                        """
                        SELECT time, device_id, pm2_5, temperature, humidity
                        FROM clean_data
                        WHERE device_id = %s
                        ORDER BY time DESC
                        LIMIT 1
                        """,
                        (device,),
                    )
                    latest_clean = cur.fetchone()
                    if latest_clean is None:
                        raise HTTPException(status_code=404, detail=f"No data available for device: {device}")
                    latest = {
                        **latest_clean,
                        "quality_uptime_pct": 99.0,
                        "quality_valid_pct": 98.0,
                        "drift_z_score": 0.0,
                        "drift_alert": False,
                        "reliability_score": 95.0,
                        "is_anomaly": False,
                        "anomaly_score": 0.0,
                        "cluster_label": "Stable Indoor",
                        "status_code": "ok",
                        "prediction_confidence": 0.9,
                    }

                cur.execute(
                    """
                    SELECT pm2_5
                    FROM clean_data
                    WHERE device_id = %s
                    ORDER BY time DESC
                    LIMIT 60
                    """,
                    (device,),
                )
                sparkline_rows = cur.fetchall()
                sparkline = [round(_to_float(r["pm2_5"]), 1) for r in reversed(sparkline_rows)]
                if not sparkline:
                    sparkline = [round(_to_float(latest.get("pm2_5"), 10.0), 1)] * 60

                cur.execute(
                    """
                    SELECT regime_id, regime_label
                    FROM regime_profiles
                    WHERE device_id = %s
                    ORDER BY time DESC
                    LIMIT 1
                    """,
                    (device,),
                )
                regime_row = cur.fetchone() or {}
                regime_id = int(regime_row.get("regime_id") or 1)
                regime_label = regime_row.get("regime_label") or REGIME_LABELS.get(regime_id, "Stable Indoor")

                cur.execute(
                    """
                    SELECT from_regime, to_regime, transition_prob
                    FROM regime_transitions
                    """
                )
                transition_rows = cur.fetchall()

                matrix_size = 5
                matrix = [[0.0 for _ in range(matrix_size)] for _ in range(matrix_size)]
                for row in transition_rows:
                    i = int(row["from_regime"] or 0)
                    j = int(row["to_regime"] or 0)
                    if 0 <= i < matrix_size and 0 <= j < matrix_size:
                        matrix[i][j] = round(_to_float(row["transition_prob"]), 4)

                current_row = matrix[regime_id] if 0 <= regime_id < len(matrix) else matrix[1]
                if sum(current_row) <= 0:
                    current_row = [0.1, 0.62, 0.14, 0.1, 0.04]
                    matrix[1] = current_row
                    regime_id = 1
                    regime_label = REGIME_LABELS[1]

                next_idx = max(range(len(current_row)), key=lambda idx: current_row[idx] if idx != regime_id else -1.0)

                cur.execute(
                    """
                    SELECT avg_duration, entropy
                    FROM regime_stability
                    WHERE device_id = %s AND regime_id = %s
                    LIMIT 1
                    """,
                    (device, regime_id),
                )
                stability_row = cur.fetchone() or {}
                entropy = round(_to_float(stability_row.get("entropy"), 1.0), 2)
                duration_minutes = int(round(_to_float(stability_row.get("avg_duration"), 30.0)))

                cur.execute(
                    """
                    SELECT measured_at, z_score, drift_alert
                    FROM pragnya_drift_metrics
                    WHERE device_id = %s
                    ORDER BY measured_at DESC
                    LIMIT 60
                    """,
                    (device,),
                )
                drift_rows = cur.fetchall()
                drift_series = [round(_to_float(r["z_score"]), 2) for r in reversed(drift_rows)]
                if not drift_series:
                    drift_series = [round(_to_float(latest.get("drift_z_score"), 0.0), 2)]

                cur.execute(
                    """
                    SELECT window_end, uptime_pct, valid_pct
                    FROM pragnya_quality_metrics
                    WHERE device_id = %s
                    ORDER BY window_end DESC
                    LIMIT 24
                    """,
                    (device,),
                )
                metric_rows = cur.fetchall()

                model_metrics: list[dict[str, float | int]] = []
                for idx, row in enumerate(reversed(metric_rows)):
                    uptime = _to_float(row["uptime_pct"], 99.0)
                    valid = _to_float(row["valid_pct"], 98.0)
                    mae = round(max(0.05, (100.0 - valid) / 20.0), 2)
                    rmse = round(max(0.08, (100.0 - uptime) / 15.0 + 0.2), 2)
                    model_metrics.append({"t": idx, "mae": mae, "rmse": rmse})

                if not model_metrics:
                    model_metrics = [
                        {"t": i, "mae": round(0.4 + abs(math.sin(i / 5)) * 0.2, 2), "rmse": round(0.7 + abs(math.cos(i / 6)) * 0.3, 2)}
                        for i in range(24)
                    ]

                pm25_now = round(_to_float(latest.get("pm2_5"), 10.0), 1)
                temp_now = round(_to_float(latest.get("temperature"), 25.0), 1)
                hum_now = round(_to_float(latest.get("humidity"), 55.0), 1)
                forecast, detailed_forecast = _build_forecast(pm25_now)
                timeline = _build_timeline(pm25_now, temp_now, hum_now)

                anomaly = _to_bool(latest.get("is_anomaly"), False)
                anomaly_score = round(_to_float(latest.get("anomaly_score"), 0.0), 3)
                ts_text = _fmt_ts(latest.get("time"))

                reliability_score = _to_float(latest.get("reliability_score"), 95.0)
                quality_uptime = _to_float(latest.get("quality_uptime_pct"), 99.0)
                quality_valid = _to_float(latest.get("quality_valid_pct"), 98.0)
                drift_sigma = abs(_to_float(latest.get("drift_z_score"), 0.0))

                alerts = [
                    {
                        "id": f"ALT-{datetime.now(timezone.utc).strftime('%H%M%S')}",
                        "ts": ts_text,
                        "severity": "critical" if anomaly else "info",
                        "title": "Live Anomaly Detected" if anomaly else "System Within Expected Range",
                        "description": (
                            f"Model flagged anomaly score {anomaly_score}."
                            if anomaly
                            else "No critical anomaly in latest inference window."
                        ),
                        "recommendation": (
                            "Inspect sensor placement and verify calibration baseline."
                            if anomaly
                            else "Continue normal monitoring cadence."
                        ),
                        "status": "open" if anomaly else "acknowledged",
                    }
                ]

                return {
                    "device": {
                        "id": device,
                        "label": device,
                        "seedOffset": 0,
                    },
                    "sensor": {
                        "device_id": device,
                        "device_label": device,
                        "ts": int((latest.get("time") or datetime.now(timezone.utc)).timestamp() * 1000),
                        "pm25": pm25_now,
                        "pm10": round(pm25_now * 1.35, 1),
                        "cityAvg": round(pm25_now * 0.9, 1),
                        "delta": round(pm25_now - (sum(sparkline[-5:]) / max(1, min(5, len(sparkline)))), 1),
                        "temperature": temp_now,
                        "humidity": hum_now,
                        "sparkline": sparkline,
                    },
                    "regime": {
                        "current": regime_label,
                        "confidence": round(_to_float(latest.get("prediction_confidence"), 0.86), 2),
                        "statusLabel": (
                            "Stable Clean"
                            if pm25_now < 15
                            else "Moderate Fluctuation"
                            if pm25_now < 35
                            else "Unstable / Polluted"
                        ),
                        "windowMinutes": 5,
                    },
                    "transitions": {
                        "matrix": matrix,
                        "nextLikely": {
                            "regime": REGIME_LABELS.get(next_idx, "Stable Indoor"),
                            "probability": round(current_row[next_idx], 4),
                        },
                    },
                    "stability": {
                        "durationMinutes": max(1, duration_minutes),
                        "entropy": entropy,
                        "trend": "Stable" if entropy < 0.9 else "Drifting" if entropy < 1.3 else "Volatile",
                    },
                    "anomaly": {
                        "anomalous": anomaly,
                        "severity": "critical" if anomaly else "info",
                        "detectedAt": ts_text,
                        "code": f"ANM-{int(anomaly_score * 1000):03d}",
                        "title": "Anomalous Pattern" if anomaly else "All Clear",
                    },
                    "reliability": {
                        "trust": round(max(0.0, min(100.0, reliability_score))),
                        "uptime": round(max(0.0, min(100.0, quality_uptime)), 1),
                        "validity": round(max(0.0, min(100.0, quality_valid)), 1),
                        "driftSigma": round(drift_sigma, 2),
                    },
                    "alerts": alerts,
                    "drift": drift_series,
                    "modelMetrics": model_metrics,
                    "forecast": forecast,
                    "detailedForecast": detailed_forecast,
                    "timeline24h": timeline,
                }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline query failed: {exc}") from exc


@app.get("/api/forecast")
def get_forecast():
    try:
        with _connect() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT time, y_true, y_pred
                    FROM ashu_model_predictions
                    WHERE model_id = 'pm25_model'
                    ORDER BY time DESC
                    LIMIT 120;
                """)
                rows = cur.fetchall()

        
        rows = list(reversed(rows))

        return rows

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/regime-intelligence")
def get_regime_intelligence(device: str = Query(DEFAULT_DEVICE, min_length=1)):
    try:
        with _connect() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:

                cur.execute("""
                    SELECT from_regime, to_regime, transition_prob
                    FROM regime_transitions
                """)
                rows = cur.fetchall()

                matrix_size = 3
                matrix = [[0]*matrix_size for _ in range(matrix_size)]

                for r in rows:
                    i = int(r["from_regime"])
                    j = int(r["to_regime"])
                    if i < matrix_size and j < matrix_size:
                        matrix[i][j] = float(r["transition_prob"])

                return {
                    "transitions": {
                        "matrix": matrix
                    }
                }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/api/particle-physics")
def particle_physics(device: str = DEFAULT_DEVICE):
    try:
        with _connect() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:

                cur.execute("""
                    SELECT time, cluster, label
                    FROM pratishtha_features
                    WHERE device_id = %s
                    ORDER BY time ASC
                    LIMIT 200
                """, (device,))

                rows = cur.fetchall()

                processed = []
                for i, r in enumerate(rows):
                    cluster = r.get("cluster", 0)

                    pm25 = 10 + cluster * 8 + math.sin(i / 5) * 2
                    pm10 = pm25 * 1.3

                    processed.append({
                        "time": r["time"],
                        "cluster": cluster,
                        "label": r["label"],
                        "pm2_5": round(pm25, 2),
                        "pm10_0": round(pm10, 2),
                    })

                return {
                    "particleSize": processed,
                    "densityScatter": processed,
                    "clusters": processed
                }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
