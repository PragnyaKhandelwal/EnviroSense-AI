#!/usr/bin/env python
"""
Ashu Analytics Pipeline (member2): ARIMA Forecasting
- Reads from clean_data 
- Generates 4-step PM forecasts using ARIMA
- Writes to ashu_features table
- Watermark-based incremental processing
"""

import sys
import traceback
import logging

sys.path.insert(0, "/home/shared/envirosense")

import pandas as pd
import time
import os
from pathlib import Path
from sqlalchemy import text

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    logger.warning("⚠️ statsmodels not installed, ARIMA forecasting disabled")
    ARIMA = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    logger.warning("⚠️ matplotlib not available, plots disabled")
    HAS_MATPLOTLIB = False

import db_utils
from db_utils import get_engine

# Ensure output directory exists
os.makedirs("Ashu", exist_ok=True)


def get_last_processed_time():
    """Get MAX(time) from ashu_features to use as watermark."""
    try:
        with db_utils.get_engine().connect() as conn:
            result = conn.execute(text("SELECT MAX(time) FROM ashu_features"))
            last_time = result.scalar()
            if last_time is None:
                return None
            return last_time
    except Exception as e:
        logger.error(f"Error getting last processed time: {e}")
        return None


def fetch_historical_window():
    """Fetch historical window for ARIMA model training (last 24 hours)."""
    try:
        query = text("""
            SELECT time, pm1_0, pm2_5, pm10_0
            FROM clean_data
            WHERE time >= NOW() - INTERVAL '24 hours'
            ORDER BY time ASC
        """)
        with db_utils.get_engine().connect() as conn:
            df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()


def generate_forecasts_for_timestamp(timestamp, historical_df, pm_columns=["pm1_0", "pm2_5", "pm10_0"]):
    """
    Generate ARIMA forecasts for a specific timestamp.
    
    Args:
        timestamp: The time to generate forecasts for
        historical_df: Historical data including and before this timestamp
        pm_columns: List of PM columns to forecast
    
    Returns:
        Dictionary of forecast results
    """
    result = {}
    
    if historical_df.empty:
        logger.warning(f"Empty historical data for {timestamp}")
        return None
    
    # Convert time column to datetime and set as index
    historical_df = historical_df.copy()
    historical_df['time'] = pd.to_datetime(historical_df['time'])
    historical_df.set_index('time', inplace=True)
    
    # Resample to consistent frequency
    try:
        historical_df = historical_df.resample("1min").mean()
        historical_df = historical_df.interpolate()
    except Exception as e:
        logger.warning(f"Resampling error for {timestamp}: {e}")
        pass
    
    # Time-based features
    try:
        result["hour"] = int(pd.Timestamp(timestamp).hour)
        result["day"] = pd.Timestamp(timestamp).day_name()
    except:
        pass
    
    # Process each PM pollutant
    for col in pm_columns:
        if col not in historical_df.columns:
            continue
        
        try:
            series = historical_df[col].dropna()
            
            if len(series) < 5:  # Need minimum data
                logger.warning(f"Insufficient data for {col} at {timestamp}")
                continue
            
            # Compute basic statistics
            mean_val = float(series.mean())
            std_val = float(series.std())
            
            # Threshold detection
            threshold = 15.0 if col == "pm2_5" else 25.0
            last_value = float(series.iloc[-1])
            is_exceeding = last_value > threshold
            
            # Naming convention
            if col == "pm2_5":
                key = "pm25"
            elif col == "pm10_0":
                key = "pm10"
            else:
                key = "pm1"
            
            # Store current stats
            result[f"{key}_avg"] = mean_val
            result[f"{key}_exceed"] = is_exceeding
            result[f"{key}_duration"] = 1  # Placeholder: could enhance with run-length encoding
            
            # ARIMA Forecasting (if statsmodels available)
            if ARIMA is not None and len(series) >= 10:
                try:
                    model = ARIMA(series, order=(1, 1, 1), suppress_warnings=True)
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=4)
                    
                    result[f"{key}_forecast_1"] = float(forecast.iloc[0])
                    result[f"{key}_forecast_2"] = float(forecast.iloc[1])
                    result[f"{key}_forecast_3"] = float(forecast.iloc[2])
                    result[f"{key}_forecast_4"] = float(forecast.iloc[3])
                except Exception as e:
                    logger.debug(f"ARIMA failed for {col} at {timestamp}: {e}")
                    # Use naive forecast (repeat last value)
                    for i in range(1, 5):
                        result[f"{key}_forecast_{i}"] = last_value
            else:
                # Fallback: naive forecast
                for i in range(1, 5):
                    result[f"{key}_forecast_{i}"] = last_value
        
        except Exception as e:
            logger.error(f"Error processing {col}: {e}")
            traceback.print_exc()
    
    return result


def run_pipeline():
    """Main analytics pipeline."""
    logger.info("🚀 Starting Ashu Analytics Pipeline")
    
    try:
        # Get last processed timestamp (watermark)
        last_time = get_last_processed_time()
        logger.info(f"Watermark: {last_time}")
        
        # Fetch historical window for model training
        historical_df = fetch_historical_window()
        
        if historical_df.empty:
            logger.warning("No historical data available")
            return
        
        logger.info(f"Loaded {len(historical_df)} historical rows")
        
        # Fetch NEW rows since watermark to process
        try:
            if last_time is None:
                # First run: process last 12 hours of data
                query = text("""
                    SELECT time FROM clean_data
                    WHERE time >= NOW() - INTERVAL '12 hours'
                    ORDER BY time ASC
                """)
            else:
                # Incremental: only NEW data
                query = text("""
                    SELECT time FROM clean_data
                    WHERE time > :last_time
                    ORDER BY time ASC
                """)
            
            with db_utils.get_engine().connect() as conn:
                if last_time is None:
                    times_df = pd.read_sql(query, conn)
                else:
                    times_df = pd.read_sql(query, conn, params={'last_time': last_time})
        
        except Exception as e:
            logger.error(f"Error fetching new timestamps: {e}")
            return
        
        if times_df.empty:
            logger.info("✅ No new data to process")
            return
        
        timestamps = times_df['time'].unique()
        logger.info(f"📊 Processing {len(timestamps)} new timestamps")
        
        # Process each timestamp and generate forecasts
        forecast_rows = []
        
        for ts in timestamps:
            try:
                forecast = generate_forecasts_for_timestamp(ts, historical_df)
                
                if forecast is not None:
                    forecast['time'] = ts
                    forecast_rows.append(forecast)
            except Exception as e:
                logger.warning(f"Error processing timestamp {ts}: {e}")
        
        if not forecast_rows:
            logger.warning("No forecasts generated")
            return
        
        # Insert into ashu_features
        forecast_df = pd.DataFrame(forecast_rows)
        logger.info(f"Generated {len(forecast_df)} forecast rows")
        logger.info(f"Time range: {forecast_df['time'].min()} → {forecast_df['time'].max()}")
        
        # Use save_dataframe which handles ON CONFLICT
        inserted = db_utils.save_dataframe(forecast_df, "ashu_features")
        logger.info(f"✅ Inserted {inserted} rows to ashu_features")
        
        # Optional: Generate visualization (if matplotlib available)
        if HAS_MATPLOTLIB and len(historical_df) > 0:
            try:
                historical_df_copy = historical_df.copy()
                plt.figure(figsize=(12, 6))
                
                for col in ["pm1_0", "pm2_5", "pm10_0"]:
                    if col in historical_df_copy.columns:
                        data = historical_df_copy[col].dropna()
                        if len(data) > 0:
                            plt.plot(data.index, data, label=col, alpha=0.7)
                
                plt.legend()
                plt.title("Historical Pollution Trends (24h)")
                plt.xlabel("Time")
                plt.ylabel("Concentration (µg/m³)")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig("Ashu/combined_plot.png", dpi=150)
                plt.close()
                logger.info("✅ Plot saved")
            except Exception as e:
                logger.warning(f"Plot generation failed: {e}")
        
        logger.info("✅ Cycle complete\n")
    
    except Exception as e:
        logger.error(f"❌ PIPELINE ERROR: {e}")
        traceback.print_exc()


def main():
    """Main loop with 10-minute cadence (forecast every 10 min, not every 60s)."""
    logger.info("🔥 Ashu Analytics Service Started (10-minute cadence)")
    
    while True:
        try:
            run_pipeline()
        except Exception as e:
            logger.error(f"❌ OUTER LOOP ERROR: {e}")
            traceback.print_exc()
        
        # 10-minute cadence (forecasting is less frequent than real-time features)
        logger.info("Sleeping for 10 minutes before next cycle...\n")
        time.sleep(600)


if __name__ == "__main__":
    main()