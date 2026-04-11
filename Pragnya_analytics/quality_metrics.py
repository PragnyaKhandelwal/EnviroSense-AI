#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

SHARED_UTILS_DIR = Path('/home/shared/envirosense')
if str(SHARED_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_UTILS_DIR))

import db_utils  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Pragnya analytics quality metric generator')
    parser.add_argument('--window-minutes', type=int, default=60, help='Window size used to compute quality metrics')
    parser.add_argument('--expected-interval-seconds', type=int, default=60, help='Expected sensor publish interval')
    parser.add_argument('--source-table', default='sensor_data', help='Source table name')
    parser.add_argument('--target-table', default='pragnya_quality_metrics', help='Destination table name')
    parser.add_argument('--ensure-table', action='store_true', help='Attempt to create target table before writing')
    parser.add_argument('--dry-run', action='store_true', help='Print metrics without writing to DB')
    return parser.parse_args()


def ensure_table(target_table: str) -> None:
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {target_table} (
        id BIGSERIAL PRIMARY KEY,
        window_end TIMESTAMPTZ NOT NULL,
        window_minutes INTEGER NOT NULL,
        device_id TEXT NOT NULL,
        expected_points INTEGER NOT NULL,
        actual_points INTEGER NOT NULL,
        valid_points INTEGER NOT NULL,
        uptime_pct NUMERIC(6,2) NOT NULL,
        valid_pct NUMERIC(6,2) NOT NULL,
        source_tag TEXT NOT NULL DEFAULT 'pragnya_analytics',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """
    with db_utils.get_engine().begin() as conn:
        conn.execute(text(create_sql))


def build_metrics(window_minutes: int, expected_interval_seconds: int, source_table: str) -> pd.DataFrame:
    query = f"""
        SELECT
            time,
            device_id,
            valid
        FROM {source_table}
        WHERE time >= NOW() - (:minutes || ' minutes')::interval
    """
    with db_utils.get_engine().connect() as conn:
        source = pd.read_sql_query(text(query), conn, params={'minutes': window_minutes})

    if source.empty:
        return pd.DataFrame()

    expected_points = max(1, round((window_minutes * 60) / expected_interval_seconds))
    source['valid'] = source['valid'].fillna(False).astype(bool)

    grouped = (
        source.groupby('device_id', as_index=False)
        .agg(actual_points=('time', 'count'), valid_points=('valid', 'sum'))
        .sort_values('device_id')
    )
    grouped['expected_points'] = expected_points
    grouped['uptime_pct'] = (grouped['actual_points'] / expected_points * 100.0).clip(upper=100).round(2)
    grouped['valid_pct'] = (grouped['valid_points'] / grouped['actual_points'] * 100.0).fillna(0.0).round(2)
    grouped['window_end'] = pd.Timestamp.now(tz='UTC')
    grouped['window_minutes'] = window_minutes
    grouped['source_tag'] = 'pragnya_analytics'

    return grouped[
        [
            'window_end',
            'window_minutes',
            'device_id',
            'expected_points',
            'actual_points',
            'valid_points',
            'uptime_pct',
            'valid_pct',
            'source_tag',
        ]
    ]


def main() -> int:
    args = parse_args()
    if args.ensure_table:
        ensure_table(args.target_table)

    metrics = build_metrics(
        window_minutes=args.window_minutes,
        expected_interval_seconds=args.expected_interval_seconds,
        source_table=args.source_table,
    )
    if metrics.empty:
        print('No source rows in selected window; no quality metrics generated.')
        return 0

    print(metrics.to_string(index=False))
    if args.dry_run:
        print('Dry run enabled; nothing written to database.')
        return 0

    inserted = db_utils.save_dataframe(metrics, args.target_table)
    print(f'Inserted {inserted} quality metric rows into {args.target_table}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
