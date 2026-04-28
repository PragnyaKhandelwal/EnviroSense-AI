import sys
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text

SHARED_UTILS_DIR = Path('/home/shared/envirosense')
if str(SHARED_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_UTILS_DIR))

from db_utils import get_engine


SLEEP_SECONDS = 60
ANOMALY_THRESHOLD = 200.0


def get_last_processed_time():
    with get_engine().connect() as conn:
        result = conn.execute(text('SELECT MAX(time) FROM rachna_anomaly'))
        return result.scalar()


def fetch_new_rows(last_time):
    query = text(
        """
        WITH ordered AS (
            SELECT
                time,
                device_id,
                pm2_5,
                LAG(pm2_5) OVER (PARTITION BY device_id ORDER BY time) AS prev_pm2_5
            FROM sensor_data
        )
        SELECT
            time,
            device_id,
            pm2_5,
            CASE
                WHEN prev_pm2_5 IS NULL THEN FALSE
                WHEN ABS(pm2_5 - prev_pm2_5) > :threshold THEN TRUE
                ELSE FALSE
            END AS anomaly
        FROM ordered
        WHERE (:last_time IS NULL OR time > :last_time)
        ORDER BY time ASC
        """
    )

    with get_engine().connect() as conn:
        return pd.read_sql_query(
            query,
            conn,
            params={'threshold': ANOMALY_THRESHOLD, 'last_time': last_time},
        )


def upsert_rows(rows: pd.DataFrame) -> int:
    if rows.empty:
        return 0

    payload = rows[['time', 'device_id', 'pm2_5', 'anomaly']].to_dict('records')

    statement = text(
        """
        INSERT INTO rachna_anomaly (time, device_id, pm2_5, anomaly)
        VALUES (:time, :device_id, :pm2_5, :anomaly)
        ON CONFLICT (time, device_id)
        DO UPDATE SET
            pm2_5 = EXCLUDED.pm2_5,
            anomaly = EXCLUDED.anomaly
        """
    )

    with get_engine().begin() as conn:
        conn.execute(statement, payload)

    return len(payload)


def main():
    while True:
        last_time = get_last_processed_time()
        rows = fetch_new_rows(last_time)

        inserted = upsert_rows(rows)
        print(f'Processed {len(rows)} rows; upserted {inserted} rows.')

        time.sleep(SLEEP_SECONDS)


if __name__ == '__main__':
    main()
