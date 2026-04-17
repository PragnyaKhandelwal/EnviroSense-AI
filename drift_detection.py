#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

SHARED_UTILS_DIR = Path('/home/shared/envirosense')
if str(SHARED_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_UTILS_DIR))

import db_utils  # type: ignore


RATIO_COLUMNS = {
    'ratio_pm1_pcs_mass': ('pm1_0_pcs', 'pm1_0'),
    'ratio_pm25_pcs_mass': ('pm2_5_pcs', 'pm2_5'),
    'ratio_pm10_pcs_mass': ('pm10_pcs', 'pm10_0'),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Pragnya analytics PM ratio drift detector')
    parser.add_argument('--lookback-hours', type=int, default=24, help='Lookback window for baseline statistics')
    parser.add_argument('--z-threshold', type=float, default=3.0, help='Absolute z-score threshold for drift alert')
    parser.add_argument('--source-table', default='sensor_data', help='Source table name')
    parser.add_argument('--target-table', default='pragnya_drift_metrics', help='Destination table name')
    parser.add_argument('--ensure-table', action='store_true', help='Attempt to create target table before writing')
    parser.add_argument('--dry-run', action='store_true', help='Print drift metrics without writing to DB')
    return parser.parse_args()


def ensure_table(target_table: str) -> None:
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {target_table} (
        id BIGSERIAL PRIMARY KEY,
        measured_at TIMESTAMPTZ NOT NULL,
        device_id TEXT NOT NULL,
        ratio_name TEXT NOT NULL,
        ratio_value DOUBLE PRECISION NOT NULL,
        baseline_mean DOUBLE PRECISION NOT NULL,
        baseline_std DOUBLE PRECISION NOT NULL,
        z_score DOUBLE PRECISION NOT NULL,
        drift_alert BOOLEAN NOT NULL,
        source_tag TEXT NOT NULL DEFAULT 'pragnya_analytics',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """
    with db_utils.get_engine().begin() as conn:
        conn.execute(text(create_sql))


def compute_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for ratio_name, (pcs_col, mass_col) in RATIO_COLUMNS.items():
        pcs = pd.to_numeric(out[pcs_col], errors='coerce')
        mass = pd.to_numeric(out[mass_col], errors='coerce')
        out[ratio_name] = np.where(mass > 0, pcs / mass, np.nan)
    return out


def build_drift_rows(source_table: str, lookback_hours: int, z_threshold: float) -> pd.DataFrame:
    query = f"""
        SELECT
            time,
            device_id,
            pm1_0_pcs,
            pm2_5_pcs,
            pm10_pcs,
            pm1_0,
            pm2_5,
            pm10_0
        FROM {source_table}
        WHERE time >= NOW() - (:hours || ' hours')::interval
        ORDER BY time ASC
    """
    with db_utils.get_engine().connect() as conn:
        source = pd.read_sql_query(text(query), conn, params={'hours': lookback_hours})

    if source.empty:
        return pd.DataFrame()

    source['time'] = pd.to_datetime(source['time'], utc=True, errors='coerce')
    source = compute_ratios(source)

    rows: list[dict[str, object]] = []
    for device_id, group in source.groupby('device_id'):
        if group.empty:
            continue
        latest = group.iloc[-1]
        baseline = group.iloc[:-1] if len(group) > 1 else group

        for ratio_name in RATIO_COLUMNS:
            baseline_series = pd.to_numeric(baseline[ratio_name], errors='coerce').dropna()
            if baseline_series.empty:
                continue

            mean = float(baseline_series.mean())
            std = float(baseline_series.std(ddof=0))
            if std == 0:
                std = 1e-9

            ratio_value = float(latest[ratio_name]) if pd.notna(latest[ratio_name]) else np.nan
            if np.isnan(ratio_value):
                continue

            z_score = (ratio_value - mean) / std
            rows.append(
                {
                    'measured_at': latest['time'],
                    'device_id': device_id,
                    'ratio_name': ratio_name,
                    'ratio_value': ratio_value,
                    'baseline_mean': mean,
                    'baseline_std': std,
                    'z_score': float(z_score),
                    'drift_alert': bool(abs(z_score) >= z_threshold),
                    'source_tag': 'pragnya_analytics',
                }
            )

    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if args.ensure_table:
        ensure_table(args.target_table)

    drift = build_drift_rows(
        source_table=args.source_table,
        lookback_hours=args.lookback_hours,
        z_threshold=args.z_threshold,
    )
    if drift.empty:
        print('No drift metrics produced in selected lookback window.')
        return 0

    print(drift.to_string(index=False))
    if args.dry_run:
        print('Dry run enabled; nothing written to database.')
        return 0

    inserted = db_utils.save_dataframe(drift, args.target_table)
    alerts = int(drift['drift_alert'].sum())
    print(f'Inserted {inserted} drift rows into {args.target_table}; alerts={alerts}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
