import pandas as pd
import psycopg2
from statsmodels.tsa.arima.model import ARIMA
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import time

engine = create_engine("postgresql+psycopg2://ashu:ashu@localhost:5432/envirosense")
POLL_INTERVAL = 60 
pollutants = ["pm1_0", "pm2_5", "pm10_0"]

def process_data():

    query = """
    SELECT time, pm1_0, pm2_5, pm10_0
    FROM sensor_data
    ORDER BY time
    """

    df = pd.read_sql(query, engine)

    if df.empty:
        print("No data available")
        return

    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    df = df.resample("15min").mean()
    
    df = df.interpolate()
    df = df.asfreq("15min")
    
    df["hour"] = df.index.hour
    df["day"] = df.index.day_name()

    forecasts = {}
    heatmaps = {}


    for col in pollutants:
        print(f"\nProcessing {col.upper()}...")
        
        # -------- TREND --------
        df[f"{col}_smooth"] = df[col].rolling(window=8).mean()

        mean = df[col].mean()
        std = df[col].std()

        df[f"{col}_peak"] = df[col] > (mean + 2 * std)
        
        # -------- HEATMAP --------
        heatmaps[col] = df.pivot_table(
            values=col,
            index="hour",
            columns="day",
            aggfunc="mean"
        )
        
        # -------- ARIMA --------
        try:
            model = ARIMA(df[col], order=(2,1,2))
            model_fit = model.fit()

            forecasts[col] = model_fit.forecast(steps=4)

            print(f"\nForecast for {col}:")
            print(forecasts[col])

        except Exception as e:
            print(f"ARIMA failed for {col}: {e}")
            
        threshold = 15 if col == "pm2_5" else 25
        
        df[f"{col}_exceed"] = df[col] > threshold

        df[f"{col}_group"] = (
            df[f"{col}_exceed"] != df[f"{col}_exceed"].shift()
        ).cumsum()

        durations = df[df[f"{col}_exceed"]].groupby(f"{col}_group").size()

        print(f"\n{col.upper()} Exceedance Durations:")
        print(durations)
        
        
        # -------- ANOMALY --------
        df[f"{col}_zscore"] = (df[col] - mean) / std
        df[f"{col}_anomaly"] = df[f"{col}_zscore"].abs() > 3

        # -------- VOLATILITY --------
        df[f"{col}_volatility"] = df[col].rolling(8).std()
        
        
        
        
        plt.figure(figsize=(12, 6))

    for col in pollutants:
        plt.plot(df.index, df[col], label=f"{col} raw")
        plt.plot(df.index, df[f"{col}_smooth"], linestyle='--', label=f"{col} smooth")

    plt.legend()
    plt.title("Pollutant Trends")
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.grid()
    plt.show()
    plt.savefig("output.png")
    
    print("\nProcessing completed.\n")






def run_pipeline():
    print("Starting real-time analytics pipeline...\n")

    while True:
        try:
            process_data()
        except Exception as e:
            print("Error:", e)

        print(f"Waiting {POLL_INTERVAL} seconds...\n")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run_pipeline()