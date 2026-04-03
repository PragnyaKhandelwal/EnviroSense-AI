import psycopg2
import pandas as pd

conn = psycopg2.connect(
    host="69.62.83.135",
    database="envirosense",
    user="postgres",
    password="rachna",
    port="5432",
    sslmode="disable"
)

query = "SELECT * FROM sensor_data ORDER BY time;"
df = pd.read_sql(query, conn)

print(df.head())