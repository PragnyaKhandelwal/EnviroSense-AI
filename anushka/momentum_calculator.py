#!/usr/bin/env python3
"""
Momentum Calculator Module for Temporal Momentum Analysis

This module provides functions to compute momentum indices, trend strength,
and classify patterns as spikes or sustained changes.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats


def compute_momentum_index(series: pd.Series, window: int = 5) -> pd.Series:
    """
    Compute momentum index as the rate of change over a rolling window.

    Args:
        series: Pandas Series of numeric values
        window: Rolling window size for momentum calculation

    Returns:
        Series of momentum indices
    """
    # Momentum as the slope of linear regression over window
    momentum = pd.Series(index=series.index, dtype=float)

    for i in range(window - 1, len(series)):
        window_data = series.iloc[i - window + 1:i + 1]
        if len(window_data) >= 2:
            x = np.arange(len(window_data))
            slope, _ = np.polyfit(x, window_data.values, 1)
            momentum.iloc[i] = slope
        else:
            momentum.iloc[i] = np.nan

    return momentum


def compute_trend_strength(series: pd.Series, window: int = 10) -> pd.Series:
    """
    Compute trend strength as the R-squared of linear fit over window.

    Args:
        series: Pandas Series of numeric values
        window: Rolling window size for trend calculation

    Returns:
        Series of trend strength values (0-1)
    """
    trend_strength = pd.Series(index=series.index, dtype=float)

    for i in range(window - 1, len(series)):
        window_data = series.iloc[i - window + 1:i + 1]
        if len(window_data) >= 2:
            x = np.arange(len(window_data))
            slope, intercept = np.polyfit(x, window_data.values, 1)
            predicted = slope * x + intercept
            ss_res = np.sum((window_data.values - predicted) ** 2)
            ss_tot = np.sum((window_data.values - np.mean(window_data.values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            trend_strength.iloc[i] = max(0, min(1, r_squared))
        else:
            trend_strength.iloc[i] = np.nan

    return trend_strength


def classify_pattern(momentum: float, trend_strength: float,
                    spike_threshold: float = 2.0, sustained_threshold: float = 0.7) -> Tuple[bool, bool]:
    """
    Classify a pattern as spike or sustained based on momentum and trend strength.

    Args:
        momentum: Momentum index value
        trend_strength: Trend strength value (0-1)
        spike_threshold: Momentum threshold for spike detection
        sustained_threshold: Trend strength threshold for sustained detection

    Returns:
        Tuple of (spike_flag, sustained_flag)
    """
    spike_flag = abs(momentum) > spike_threshold
    sustained_flag = trend_strength > sustained_threshold

    return spike_flag, sustained_flag


def analyze_momentum_features(data: pd.DataFrame, target_col: str,
                            window: int = 5) -> pd.DataFrame:
    """
    Analyze momentum features for a dataset.

    Args:
        data: DataFrame with time series data
        target_col: Column name of the target variable
        window: Window size for momentum calculations

    Returns:
        DataFrame with momentum features added
    """
    result = data.copy()

    # Compute momentum features per device
    momentum_indices = []
    trend_strengths = []
    spike_flags = []
    sustained_flags = []

    for device_id, group in result.groupby("device_id"):
        series = group[target_col]
        momentum = compute_momentum_index(series, window)
        trend = compute_trend_strength(series, window)

        momentum_indices.extend(momentum.values)
        trend_strengths.extend(trend.values)

        for m, t in zip(momentum, trend):
            spike, sustained = classify_pattern(m, t)
            spike_flags.append(spike)
            sustained_flags.append(sustained)

    result["momentum_index"] = momentum_indices
    result["trend_strength"] = trend_strengths
    result["spike_flag"] = spike_flags
    result["sustained_flag"] = sustained_flags

    return result
