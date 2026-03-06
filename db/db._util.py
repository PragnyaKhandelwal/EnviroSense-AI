import psycopg2
import pandas as pd

def get_data():
    
    conn=psycopg2.connect(
        host="localhost",
        database="envirosense",
        user="pratishtha",
        password=""
    )
    query="SELECT * FROM sensor_data"
    df=pd.read_sql(query,conn)
    conn.close()
    return df