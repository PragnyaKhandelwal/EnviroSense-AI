#!/usr/bin/env python3
"""Stage 4 model feature backfill and incremental sync."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

SHARED_UTILS_DIR = Path("/home/shared/envirosense")
if str(SHARED_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_UTILS_DIR))

import db_utils


LOGGER = logging.getLogger("model_features_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill and sync model_features")
    parser.add_argument("--lookback-hours", type=int, default=24, help="Historical window for rolling features")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Rows per upsert batch")
    parser.add_argument("--backfill-all", action="store_true", help="Ignore watermark and rebuild from all clean_data")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_last_processed_time() -> pd.Timestamp | None:
    try:
        with db_utils.get_engine().connect() as conn:
            last_time = conn.execute(text("SELECT MAX(time) FROM model_features")).scalar()
        if last_time is None:
            return None
        ts = pd.Timestamp(last_time)
        return ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
    except Exception as exc:
        LOGGER.error("Error reading model_features watermark: %s", exc)
        return None


def ensure_unique_index() -> None:
    exists_statement = text(
        """
        SELECT 1
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND tablename = 'model_features'
          AND indexname = 'idx_model_features_device_time_unique'
        LIMIT 1
        """
    )
    with db_utils.get_engine().connect() as conn:
        exists = conn.execute(exists_statement).scalar()

    if exists:
        return

    statement = text(
        """
        CREATE UNIQUE INDEX idx_model_features_device_time_unique
        ON model_features (device_id, time)
        """
    )
    try:
        with db_utils.get_engine().begin() as conn:
            conn.execute(statement)
    except Exception as exc:
        LOGGER.warning("Could not create model_features unique index: %s", exc)


def load_source_frames(window_start: pd.Timestamp | None) -> dict[str, pd.DataFrame]:
    query_suffix = "" if window_start is None else "WHERE time >= :window_start\n"

    source_queries = {
        "clean": f"""
            SELECT
                time,
                device_id,
                pm2_5,
                temperature,
                humidity,
                state_code,
                valid,
                imputed_flag,
                source_tag
            FROM clean_data
            {query_suffix}
            ORDER BY device_id, time
        """,
        "anushka": f"""
            SELECT
                time,
                device_id,
                pm2_5_lag1,
                pm2_5_roll_1h
            FROM anushka_features
            {query_suffix}
            ORDER BY device_id, time
        """,
        "pratishtha": f"""
            SELECT
                time,
                device_id,
                cluster,
                label
            FROM pratishtha_features
            {query_suffix}
            ORDER BY device_id, time
        """,
        "rachna": f"""
            SELECT
                time,
                device_id,
                anomaly
            FROM rachna_anomaly
            {query_suffix}
            ORDER BY device_id, time
        """,
        "quality": f"""
            SELECT
                window_end AS time,
                device_id,
                uptime_pct,
                valid_pct,
                expected_points,
                actual_points,
                valid_points
            FROM pragnya_quality_metrics
            {query_suffix.replace('time', 'window_end')}
            ORDER BY device_id, window_end
        """,
        "drift": f"""
            SELECT
                measured_at AS time,
                device_id,
                ratio_name,
                ratio_value,
                baseline_mean,
                baseline_std,
                z_score,
                drift_alert
            FROM pragnya_drift_metrics
            {query_suffix.replace('time', 'measured_at')}
            ORDER BY device_id, measured_at
        """,
    }

    frames: dict[str, pd.DataFrame] = {}
    with db_utils.get_engine().connect() as conn:
        for key, sql in source_queries.items():
            if window_start is None:
                frames[key] = pd.read_sql_query(text(sql.replace("WHERE time >= :window_start\n", "")), conn)
            else:
                frames[key] = pd.read_sql_query(text(sql), conn, params={"window_start": window_start})
    return frames


def _compute_device_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("time").copy()
    group["pm2_5_lag1_calc"] = group["pm2_5"].shift(1)
    group["pm2_5_lag2_calc"] = group["pm2_5"].shift(2)
    group["pm2_5_lag4_calc"] = group["pm2_5"].shift(4)

    time_indexed = group.set_index("time")
    group["pm2_5_roll_1h_calc"] = time_indexed["pm2_5"].rolling("60min", min_periods=1).mean().to_numpy()
    group["pm2_5_roll_24h_calc"] = time_indexed["pm2_5"].rolling("24h", min_periods=1).mean().to_numpy()
    return group


def build_feature_frame(frames: dict[str, pd.DataFrame], last_time: pd.Timestamp | None) -> pd.DataFrame:
    clean = frames["clean"].copy()
    if clean.empty:
        return clean

    clean["time"] = pd.to_datetime(clean["time"], utc=True)
    clean = clean.sort_values(["device_id", "time"])

    if not frames["anushka"].empty:
        anushka = frames["anushka"].copy()
        anushka["time"] = pd.to_datetime(anushka["time"], utc=True)
        anushka = anushka.rename(
            columns={
                "pm2_5_lag1": "pm2_5_lag1_member4",
                "pm2_5_roll_1h": "pm2_5_roll_1h_member4",
            }
        )
        clean = clean.merge(anushka, on=["time", "device_id"], how="left")

    if not frames["pratishtha"].empty:
        prat = frames["pratishtha"].copy()
        prat["time"] = pd.to_datetime(prat["time"], utc=True)
        prat["cluster_label"] = prat["label"].where(prat["label"].notna(), prat["cluster"].astype(str))
        clean = clean.merge(prat[["time", "device_id", "cluster_label"]], on=["time", "device_id"], how="left")
    else:
        clean["cluster_label"] = pd.NA

    if not frames["rachna"].empty:
        rachna = frames["rachna"].copy()
        rachna["time"] = pd.to_datetime(rachna["time"], utc=True)
        clean = clean.merge(rachna[["time", "device_id", "anomaly"]], on=["time", "device_id"], how="left")
    else:
        clean["anomaly"] = False

    if not frames["quality"].empty:
        quality = frames["quality"].copy()
        quality["time"] = pd.to_datetime(quality["time"], utc=True)
        quality = quality.sort_values(["device_id", "time"])
        quality = quality.rename(
            columns={
                "uptime_pct": "pragnya_quality_uptime_pct",
                "valid_pct": "pragnya_quality_valid_pct",
                "expected_points": "pragnya_quality_expected_points",
                "actual_points": "pragnya_quality_actual_points",
                "valid_points": "pragnya_quality_valid_points",
            }
        )
        clean = pd.merge_asof(
            clean.sort_values(["device_id", "time"]),
            quality,
            by="device_id",
            on="time",
            direction="backward",
            tolerance=pd.Timedelta("60min"),
        )
    else:
        clean["pragnya_quality_uptime_pct"] = pd.NA
        clean["pragnya_quality_valid_pct"] = pd.NA
        clean["pragnya_quality_expected_points"] = pd.NA
        clean["pragnya_quality_actual_points"] = pd.NA
        clean["pragnya_quality_valid_points"] = pd.NA

    if not frames["drift"].empty:
        drift = frames["drift"].copy()
        drift["time"] = pd.to_datetime(drift["time"], utc=True)
        drift = drift.sort_values(["device_id", "time"])
        drift = drift.rename(
            columns={
                "ratio_name": "pragnya_drift_ratio_name",
                "ratio_value": "pragnya_drift_ratio_value",
                "baseline_mean": "pragnya_drift_baseline_mean",
                "baseline_std": "pragnya_drift_baseline_std",
                "z_score": "pragnya_drift_z_score",
                "drift_alert": "pragnya_drift_alert",
            }
        )
        clean = pd.merge_asof(
            clean.sort_values(["device_id", "time"]),
            drift,
            by="device_id",
            on="time",
            direction="backward",
            tolerance=pd.Timedelta("60min"),
        )
    else:
        clean["pragnya_drift_ratio_name"] = pd.NA
        clean["pragnya_drift_ratio_value"] = pd.NA
        clean["pragnya_drift_baseline_mean"] = pd.NA
        clean["pragnya_drift_baseline_std"] = pd.NA
        clean["pragnya_drift_z_score"] = pd.NA
        clean["pragnya_drift_alert"] = pd.NA

    processed_groups = []
    for _, device_group in clean.groupby("device_id", sort=False):
        processed_groups.append(_compute_device_features(device_group))

    enriched = pd.concat(processed_groups, ignore_index=True)

    member4_lag1 = enriched["pm2_5_lag1_member4"] if "pm2_5_lag1_member4" in enriched.columns else pd.Series(index=enriched.index, dtype=float)
    member4_roll1h = enriched["pm2_5_roll_1h_member4"] if "pm2_5_roll_1h_member4" in enriched.columns else pd.Series(index=enriched.index, dtype=float)

    enriched["pm2_5_lag1"] = member4_lag1.combine_first(enriched["pm2_5_lag1_calc"])
    enriched["pm2_5_lag2"] = enriched["pm2_5_lag2_calc"]
    enriched["pm2_5_lag4"] = enriched["pm2_5_lag4_calc"]
    enriched["pm2_5_roll_1h"] = member4_roll1h.combine_first(enriched["pm2_5_roll_1h_calc"])
    enriched["pm2_5_roll_24h"] = enriched["pm2_5_roll_24h_calc"]

    enriched["is_anomaly"] = enriched["anomaly"].fillna(False).astype(bool)

    def status_code(row: pd.Series) -> str:
        if not bool(row.get("valid", True)):
            return "invalid"
        if bool(row.get("imputed_flag", False)):
            return "imputed"
        if bool(row.get("is_anomaly", False)):
            return "anomaly"
        return "ok"

    def confidence(row: pd.Series) -> float:
        score = 0.95
        if not bool(row.get("valid", True)):
            score -= 0.25
        if bool(row.get("imputed_flag", False)):
            score -= 0.15
        if bool(row.get("is_anomaly", False)):
            score -= 0.20
        if pd.isna(row.get("cluster_label")):
            score -= 0.05
        if pd.isna(row.get("pm2_5_lag1")) or pd.isna(row.get("pm2_5_roll_1h")):
            score -= 0.10
        return float(round(max(0.05, min(0.99, score)), 4))

    enriched["status_code"] = enriched.apply(status_code, axis=1)
    enriched["prediction_confidence"] = enriched.apply(confidence, axis=1)
    enriched["anomaly_score"] = (
        (enriched["pm2_5"] - enriched["pm2_5_roll_1h"].fillna(enriched["pm2_5"])).abs()
        / enriched["pm2_5_roll_1h"].fillna(1).abs().clip(lower=1)
    ).clip(upper=1.0).fillna(0.0)

    enriched["created_by"] = "stage4_model_features"
    enriched["created_at"] = pd.Timestamp.now(tz="UTC")

    output = enriched[
        [
            "time",
            "device_id",
            "pm2_5",
            "temperature",
            "humidity",
            "cluster_label",
            "pm2_5_lag1",
            "pm2_5_lag2",
            "pm2_5_lag4",
            "pm2_5_roll_1h",
            "pm2_5_roll_24h",
            "is_anomaly",
            "anomaly_score",
            "status_code",
            "pragnya_quality_uptime_pct",
            "pragnya_quality_valid_pct",
            "pragnya_quality_expected_points",
            "pragnya_quality_actual_points",
            "pragnya_quality_valid_points",
            "pragnya_drift_ratio_name",
            "pragnya_drift_ratio_value",
            "pragnya_drift_baseline_mean",
            "pragnya_drift_baseline_std",
            "pragnya_drift_z_score",
            "pragnya_drift_alert",
            "created_by",
            "created_at",
            "prediction_confidence",
        ]
    ].copy()

    if last_time is not None:
        output = output[output["time"] > last_time]

    return output.sort_values(["device_id", "time"])


def upsert_dataframe(frame: pd.DataFrame, chunk_size: int) -> int:
    if frame.empty:
        return 0

    engine = db_utils.get_engine()
    metadata = MetaData()
    model_table = Table("model_features", metadata, autoload_with=engine)
    base_insert = pg_insert(model_table)
    update_columns = {
        column.name: base_insert.excluded[column.name]
        for column in model_table.columns
        if column.name not in {"device_id", "time", "created_at"}
    }

    records = frame.replace({pd.NaT: None, np.nan: None}).to_dict(orient="records")
    processed = 0

    with engine.begin() as conn:
        for start in range(0, len(records), chunk_size):
            batch = records[start : start + chunk_size]
            stmt = base_insert.values(batch).on_conflict_do_update(
                index_elements=["device_id", "time"],
                set_=update_columns,
            )
            conn.execute(stmt)
            processed += len(batch)

    return processed


def run_pipeline(args: argparse.Namespace) -> int:
    ensure_unique_index()

    last_time = None if args.backfill_all else get_last_processed_time()
    window_start = None if last_time is None else last_time - timedelta(hours=args.lookback_hours)
    if last_time is None:
        LOGGER.info("No model_features watermark found; running full backfill")
    else:
        LOGGER.info("Using watermark %s with %s-hour lookback", last_time, args.lookback_hours)

    frames = load_source_frames(window_start)
    feature_frame = build_feature_frame(frames, last_time)

    if feature_frame.empty:
        LOGGER.info("No new model_features rows to write")
        return 0

    processed = upsert_dataframe(feature_frame, args.chunk_size)
    LOGGER.info("Upserted %s model_features rows", processed)
    return processed


def main() -> None:
    configure_logging()
    args = parse_args()

    try:
        processed = run_pipeline(args)
        LOGGER.info("Stage 4 sync complete: %s rows processed", processed)
    except Exception as exc:
        LOGGER.exception("Stage 4 model_features pipeline failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()