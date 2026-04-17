import sys
sys.path.insert(0, "/home/shared/envirosense")
import pandas as pd
import time
import os

from statsmodels.tsa.arima.model import ARIMA
from db_utils import get_engine
import matplotlib.pyplot as plt


engine = get_engine()

query = """
SELECT time, pm1_0, pm2_5, pm10_0
FROM clean_data
WHERE time < '2026-04-17 08:45:00'
ORDER BY time
"""

os.makedirs("Ashu", exist_ok=True)

# ------------------ MAIN PROCESS ------------------

def process_data():

    df = pd.read_sql(query, engine)

    if df.empty:
        return

    # -------- TIME SERIES PREP --------
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    df = df.resample("1min").mean()
    df = df.interpolate()
    df = df.asfreq("15min")

    # -------- FEATURE EXTRACTION --------
    df["hour"] = df.index.hour
    df["day"] = df.index.day_name()

    result = {}

    # -------- ANALYTICS LOOP --------
    for col in ["pm1_0", "pm2_5", "pm10_0"]:

        # Trend smoothing
        df[f"{col}_smooth"] = df[col].rolling(8).mean()

        mean = df[col].mean()
        std = df[col].std()

        # Threshold
        threshold = 15 if col == "pm2_5" else 25
        df[f"{col}_exceed"] = df[col] > threshold

        # Duration
        df[f"{col}_group"] = (df[f"{col}_exceed"] != df[f"{col}_exceed"].shift()).cumsum()
        durations = df[df[f"{col}_exceed"]].groupby(f"{col}_group").size()
        last_duration = int(durations.iloc[-1]) if not durations.empty else 0

        # Forecast
        try:
            model = ARIMA(df[col], order=(1,1,1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=4)
        except:
            continue

        # Naming
        if col == "pm2_5":
            key = "pm25"
        elif col == "pm10_0":
            key = "pm10"
        else:
            key = "pm1"

        # Store results
        result[f"{key}_forecast_1"] = float(forecast.iloc[0])
        result[f"{key}_forecast_2"] = float(forecast.iloc[1])
        result[f"{key}_forecast_3"] = float(forecast.iloc[2])
        result[f"{key}_forecast_4"] = float(forecast.iloc[3])

        result[f"{key}_avg"] = float(mean)
        result[f"{key}_exceed"] = bool(df[f"{col}_exceed"].iloc[-1])
        result[f"{key}_duration"] = last_duration

    # -------- INSERT INTO DB --------
    rows = []

    for i in range(len(df)):
        row = {"time": df.index[i]}

        for key in result:
            row[key] = result[key]

        rows.append(row)

    output_df = pd.DataFrame(rows)

    from db_utils import save_dataframe
    save_dataframe(output_df, "ashu_features")

    for col in ["pm1_0", "pm2_5", "pm10_0"]:

        heatmap = df.pivot_table(
            values=col,
            index="hour",
            columns="day",
            aggfunc="mean"
        )

        # naming
        if col == "pm2_5":
            name = "pm25"
        elif col == "pm10_0":
            name = "pm10"
        else:
            name = "pm1"

        
        heatmap.to_csv(f"Ashu/{name}_heatmap.csv")

    # -------- TREND PLOT  --------
    plt.figure(figsize=(12, 6))

    for col in ["pm1_0", "pm2_5", "pm10_0"]:
        plt.plot(df.index, df[col], label=f"{col} raw")
        plt.plot(df.index, df[f"{col}_smooth"], linestyle="--", label=f"{col} smooth")

    plt.legend()
    plt.title("Pollution Trends")
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.grid()

    plt.savefig("Ashu/combined_plot.png")
    plt.close()


def main():
    process_data()
    time.sleep(60)


if __name__ == "__main__":
    main()