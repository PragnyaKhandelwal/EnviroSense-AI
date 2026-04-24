#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

SHARED_UTILS_DIR = Path('/home/shared/envirosense')
if str(SHARED_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_UTILS_DIR))

import db_utils  # type: ignore


NUMERIC_BOUNDS = {
    'pm1_0': (0.0, 2000.0),
    'pm2_5': (0.0, 2000.0),
    'pm10_0': (0.0, 2000.0),
    'temperature': (-30.0, 80.0),
    'humidity': (0.0, 100.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Pragnya data QA validator')
    parser.add_argument('--table', default='clean_data', help='Table to validate')
    parser.add_argument('--lookback-hours', type=int, default=24, help='Time window for QA checks')
    return parser.parse_args()


def query_frame(table: str, lookback_hours: int) -> pd.DataFrame:
    sql = text(
        f"""
        SELECT *
        FROM {table}
        WHERE time >= NOW() - (:hours || ' hours')::interval
        """
    )
    with db_utils.get_engine().connect() as conn:
        return pd.read_sql_query(sql, conn, params={'hours': lookback_hours})


def main() -> int:
    args = parse_args()
    frame = query_frame(args.table, args.lookback_hours)

    summary: dict[str, object] = {
        'table': args.table,
        'lookback_hours': args.lookback_hours,
        'row_count': int(len(frame)),
        'checks': {},
        'status': 'pass',
    }

    checks: dict[str, object] = {}

    if frame.empty:
        checks['empty_window'] = {'status': 'warn', 'message': 'No rows found in lookback window'}
        summary['checks'] = checks
        summary['status'] = 'warn'
        print(json.dumps(summary, indent=2, default=str))
        return 0

    null_rates = {}
    for column in ['time', 'device_id', 'pm2_5', 'temperature', 'humidity', 'valid']:
        if column in frame.columns:
            null_rates[column] = round(float(frame[column].isna().mean() * 100.0), 4)

    checks['null_rate_pct'] = {'status': 'pass', 'values': null_rates}

    out_of_range = {}
    for column, (lower, upper) in NUMERIC_BOUNDS.items():
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors='coerce')
        bad = ((series < lower) | (series > upper)).fillna(False)
        out_of_range[column] = int(bad.sum())

    range_status = 'pass' if sum(out_of_range.values()) == 0 else 'fail'
    checks['out_of_range'] = {'status': range_status, 'counts': out_of_range}

    duplicate_count = 0
    if {'time', 'device_id'}.issubset(frame.columns):
        dedup = frame[['time', 'device_id']].copy()
        dedup['time'] = pd.to_datetime(dedup['time'], utc=True, errors='coerce')
        duplicate_count = int(dedup.duplicated(subset=['time', 'device_id']).sum())

    dup_status = 'pass' if duplicate_count == 0 else 'fail'
    checks['duplicate_time_device'] = {'status': dup_status, 'count': duplicate_count}

    expected_columns = {'time', 'device_id', 'pm2_5', 'temperature', 'humidity', 'valid'}
    missing_columns = sorted(list(expected_columns - set(frame.columns)))
    schema_status = 'pass' if not missing_columns else 'fail'
    checks['schema'] = {'status': schema_status, 'missing_columns': missing_columns}

    failing = [v for v in checks.values() if isinstance(v, dict) and v.get('status') == 'fail']
    if failing:
        summary['status'] = 'fail'

    summary['checks'] = checks
    print(json.dumps(summary, indent=2, default=str))
    return 1 if summary['status'] == 'fail' else 0


if __name__ == '__main__':
    raise SystemExit(main())
