import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
def get_recent_data(limit=200):

    conn = psycopg2.connect(
         host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

    query = f"""
    SELECT *
    FROM sensor_data
    ORDER BY time DESC
    LIMIT {limit}
    """
    df = pd.read_sql(query, conn)
    print(f"DEBUG: Found {len(df)} rows")
    conn.close()

    return df