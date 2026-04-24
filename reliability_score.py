#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import text

SHARED_UTILS_DIR = Path('/home/shared/envirosense')
if str(SHARED_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_UTILS_DIR))

import db_utils  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Pragnya reliability score updater for model_features')
    parser.add_argument('--target-table', default='model_features', help='Destination feature table')
    parser.add_argument('--dry-run', action='store_true', help='Preview rows to be updated without writing')
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    score_expr = """
        ROUND(
            (
                0.35 * (COALESCE(quality_valid_pct, 0) / 100.0) +
                0.30 * (COALESCE(quality_uptime_pct, 0) / 100.0) +
                0.25 * (COALESCE(quality_completeness_pct, 0) / 100.0) +
                0.10 * GREATEST(0.0, LEAST(1.0, 1.0 - (ABS(COALESCE(drift_z_score, 0)) / 5.0)))
            )::numeric,
            4
        )
    """

    preview_sql = text(
        f"""
        SELECT COUNT(*) AS candidate_rows
        FROM {args.target_table}
        WHERE quality_valid_pct IS NOT NULL
           OR quality_uptime_pct IS NOT NULL
           OR quality_completeness_pct IS NOT NULL
           OR drift_z_score IS NOT NULL
        """
    )

    update_sql = text(
        f"""
        UPDATE {args.target_table}
        SET reliability_score = {score_expr}
        WHERE quality_valid_pct IS NOT NULL
           OR quality_uptime_pct IS NOT NULL
           OR quality_completeness_pct IS NOT NULL
           OR drift_z_score IS NOT NULL
        """
    )

    with db_utils.get_engine().begin() as conn:
        candidate_rows = conn.execute(preview_sql).scalar() or 0
        print(f'Candidate rows for reliability update: {candidate_rows}')

        if args.dry_run:
            print('Dry run enabled; skipped UPDATE.')
            return 0

        result = conn.execute(update_sql)
        print(f'Updated reliability_score for {result.rowcount} rows in {args.target_table}.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
