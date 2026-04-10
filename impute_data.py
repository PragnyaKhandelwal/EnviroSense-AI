#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SHARED_UTILS_DIR = Path('/home/shared/envirosense')
if str(SHARED_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_UTILS_DIR))

import db_utils  # type: ignore


NUMERIC_COLUMNS = ['pm1_0', 'pm2_5', 'pm10_0', 'temperature', 'humidity']
OUTPUT_COLUMNS = [
    'time',
    'device_id',
    'pm1_0',
    'pm2_5',
    'pm10_0',
    'temperature',
    'humidity',
    'state_code',
    'valid',
    'imputed_flag',
    'source_tag',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Member 1 micro-batch imputation job')
    parser.add_argument('--minutes', type=int, default=60, help='Lookback window in minutes')
    parser.add_argument('--source-table', default='sensor_data', help='Source table name')
    parser.add_argument('--target-table', default='clean_data', help='Destination table name')
    parser.add_argument('--backfill-all', action='store_true', help='Ignore watermark and backfill any missing historical rows')
    parser.add_argument('--dry-run', action='store_true', help='Do not write to the database')
    return parser.parse_args()


def get_last_processed_time(target_table: str):
    query = f'SELECT MAX(time) AS last_time FROM {target_table}'
    with db_utils.get_engine().connect() as conn:
        result = pd.read_sql_query(query, conn)
    if result.empty:
        return None
    value = result.iloc[0]['last_time']
    if pd.isna(value):
        return None
    return pd.to_datetime(value, utc=True)


def coerce_boolean(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    normalized = series.astype('string').str.lower().str.strip()
    return normalized.isin({'1', 'true', 't', 'yes', 'y'})


def build_clean_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame['time'] = pd.to_datetime(frame['time'], utc=True, errors='coerce')
    frame = frame.dropna(subset=['time', 'device_id'])
    frame = frame.sort_values(['time', 'device_id']).drop_duplicates(subset=['time', 'device_id'], keep='last')

    for column in NUMERIC_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors='coerce')

    if 'state_code' not in frame.columns:
        frame['state_code'] = pd.NA
    frame['state_code'] = pd.to_numeric(frame['state_code'], errors='coerce').astype('Int64')

    if 'valid' not in frame.columns:
        frame['valid'] = True
    frame['valid'] = coerce_boolean(frame['valid']).fillna(True)

    before_missing = frame[NUMERIC_COLUMNS].isna().any(axis=1)
    state_missing = frame['state_code'].isna()

    for column in NUMERIC_COLUMNS:
        median_value = frame[column].median(skipna=True)
        if pd.isna(median_value):
            median_value = 0.0
        frame[column] = frame[column].fillna(median_value)

    if frame['state_code'].isna().all():
        frame['state_code'] = 0
    else:
        state_mode = frame['state_code'].mode(dropna=True)
        frame['state_code'] = frame['state_code'].fillna(state_mode.iloc[0] if not state_mode.empty else 0)

    frame['source_tag'] = 'member1'
    frame['imputed_flag'] = before_missing | state_missing

    clean = frame[OUTPUT_COLUMNS].copy()
    clean['state_code'] = clean['state_code'].fillna(0).astype(int)
    clean['valid'] = clean['valid'].astype(bool)
    clean['imputed_flag'] = clean['imputed_flag'].astype(bool)
    return clean


def filter_new_rows(raw: pd.DataFrame, last_processed_time):
    if last_processed_time is None:
        return raw
    return raw[pd.to_datetime(raw['time'], utc=True, errors='coerce') > last_processed_time]


def filter_existing_rows(raw: pd.DataFrame, target_table: str) -> pd.DataFrame:
    if raw.empty:
        return raw

    with db_utils.get_engine().connect() as conn:
        existing = pd.read_sql_query(
            f'SELECT time, device_id FROM {target_table}',
            conn,
        )

    if existing.empty:
        return raw

    merged = raw.copy()
    merged['time'] = pd.to_datetime(merged['time'], utc=True, errors='coerce')
    existing['time'] = pd.to_datetime(existing['time'], utc=True, errors='coerce')
    merged = merged.merge(existing.drop_duplicates(), on=['time', 'device_id'], how='left', indicator=True)
    return merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])


def main() -> int:
    args = parse_args()
    last_processed_time = get_last_processed_time(args.target_table)
    raw = db_utils.get_latest_batch(minutes=args.minutes, source_table=args.source_table)
    if raw.empty:
        print('No source rows found in the requested window.')
        return 0

    if args.backfill_all:
        raw = filter_existing_rows(raw, args.target_table)
    else:
        raw = filter_new_rows(raw, last_processed_time)
    if raw.empty:
        print('No new rows to process.')
        return 0

    clean = build_clean_frame(raw)
    print(f'Prepared {len(clean)} rows from {len(raw)} source rows.')
    print(clean.head().to_string(index=False))

    if args.dry_run:
        print('Dry run enabled; nothing written to the database.')
        return 0

    inserted = db_utils.save_dataframe(clean, args.target_table)
    print(f'Inserted {inserted} rows into {args.target_table}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())