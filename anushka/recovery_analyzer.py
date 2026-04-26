#!/usr/bin/env python3
"""
Recovery Analyzer Module for Temporal Recovery Analysis

This module provides functions to detect peaks, compute recovery slopes,
half-lives, and return-to-baseline times.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import optimize


def detect_peak(series: pd.Series, timestamps) -> Tuple[Optional[datetime], Optional[float]]:
    """
    Detect the peak value and time in a time series.

    Args:
        series: Pandas Series of numeric values
        timestamps: Pandas Series or DatetimeIndex of timestamps

    Returns:
        Tuple of (peak_time, peak_value)
    """
    if series.empty:
        return None, None

    peak_idx = series.idxmax()
    peak_value = series.loc[peak_idx]
    # Handle both Series and DatetimeIndex
    if hasattr(timestamps, 'loc'):
        peak_time = timestamps.loc[peak_idx]
    else:
        # For DatetimeIndex, peak_idx is already the timestamp
        peak_time = peak_idx

    return peak_time, peak_value


def compute_recovery_slope(series: pd.Series, timestamps: pd.Series,
                          peak_time: datetime, baseline_window: int = 10) -> Optional[float]:
    """
    Compute the recovery slope after a peak.

    Args:
        series: Pandas Series of numeric values
        timestamps: Pandas Series of timestamps
        peak_time: Time of the detected peak
        baseline_window: Number of points to use for baseline estimation

    Returns:
        Recovery slope (negative for decreasing)
    """
    # Find data after peak
    post_peak = series[timestamps > peak_time]
    post_timestamps = timestamps[timestamps > peak_time]

    if len(post_peak) < 2:
        return None

    # Estimate baseline as mean of first few points after peak
    baseline = post_peak.iloc[:baseline_window].mean()

    # Fit linear regression to recovery phase
    if len(post_peak) >= 2:
        x = np.arange(len(post_peak))
        slope, _ = np.polyfit(x, post_peak.values, 1)
        return slope

    return None


def compute_half_life(series: pd.Series, timestamps: pd.Series,
                     peak_time: datetime, peak_value: float) -> Optional[float]:
    """
    Compute the half-life of recovery (time to reach half of peak value).

    Args:
        series: Pandas Series of numeric values
        timestamps: Pandas Series of timestamps
        peak_time: Time of the detected peak
        peak_value: Value at the peak

    Returns:
        Half-life in time units
    """
    half_value = peak_value / 2
    post_peak = series[timestamps > peak_time]

    if post_peak.empty:
        return None

    # Find first point below half value
    below_half = post_peak[post_peak <= half_value]
    if below_half.empty:
        return None

    half_time_idx = below_half.index[0]
    # Handle both Series and DatetimeIndex
    if hasattr(timestamps, 'loc'):
        half_time = timestamps.loc[half_time_idx]
    else:
        half_time = half_time_idx

    # Calculate time difference
    time_diff = (half_time - peak_time).total_seconds() / 3600  # hours

    return time_diff


def return_to_baseline(series: pd.Series, timestamps: pd.Series,
                      peak_time: datetime, baseline_value: float,
                      tolerance: float = 0.1) -> Optional[datetime]:
    """
    Find the time when the series returns to baseline.

    Args:
        series: Pandas Series of numeric values
        timestamps: Pandas Series of timestamps
        peak_time: Time of the detected peak
        baseline_value: Baseline value to return to
        tolerance: Tolerance for baseline detection

    Returns:
        Time when returned to baseline
    """
    post_peak = series[timestamps > peak_time]
    post_timestamps = timestamps[timestamps > peak_time]

    if post_peak.empty:
        return None

    # Find points within tolerance of baseline
    baseline_range = baseline_value * (1 + tolerance)
    within_baseline = post_peak[post_peak <= baseline_range]

    if within_baseline.empty:
        return None

    return_time_idx = within_baseline.index[0]
    # Handle both Series and DatetimeIndex
    if hasattr(post_timestamps, 'loc'):
        return_time = post_timestamps.loc[return_time_idx]
    else:
        return_time = return_time_idx

    return return_time


def analyze_recovery_features(data: pd.DataFrame, target_col: str, time_col: str,
                            baseline_window: int = 10) -> pd.DataFrame:
    """
    Analyze recovery features for a dataset.

    Args:
        data: DataFrame with time series data
        target_col: Column name of the target variable
        time_col: Column name of timestamps
        baseline_window: Window for baseline estimation

    Returns:
        DataFrame with recovery features added
    """
    result = data.copy()
    result[time_col] = pd.to_datetime(result[time_col])

    device_results = {}

    for device_id, group in result.groupby("device_id"):
        series = group[target_col]
        timestamps = group[time_col]

        peak_time, peak_value = detect_peak(series, timestamps)

        if peak_time is None:
            device_results[device_id] = {
                'recovery_slope': np.nan,
                'half_life': np.nan,
                'peak_time': None,
                'return_to_baseline_time': None
            }
            continue

        # Estimate baseline as mean before peak
        pre_peak = series[timestamps < peak_time]
        baseline = pre_peak.tail(baseline_window).mean() if len(pre_peak) >= baseline_window else series.mean()

        slope = compute_recovery_slope(series, timestamps, peak_time, baseline_window)
        half_life = compute_half_life(series, timestamps, peak_time, peak_value)
        return_time = return_to_baseline(series, timestamps, peak_time, baseline)

        device_results[device_id] = {
            'recovery_slope': slope,
            'half_life': half_life,
            'peak_time': peak_time,
            'return_to_baseline_time': return_time
        }

    # Assign to DataFrame
    result["recovery_slope"] = result["device_id"].map(lambda x: device_results[x]['recovery_slope'])
    result["half_life"] = result["device_id"].map(lambda x: device_results[x]['half_life'])
    result["peak_time"] = result["device_id"].map(lambda x: device_results[x]['peak_time'])
    result["return_to_baseline_time"] = result["device_id"].map(lambda x: device_results[x]['return_to_baseline_time'])

    return result
