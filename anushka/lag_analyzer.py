#!/usr/bin/env python3
"""
Lag Analyzer Module for Temporal Memory Analysis

This module provides functions to analyze temporal lag relationships in time-series data,
compute autocorrelation, estimate memory length, and generate importance reports.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats


def compute_autocorrelation(series: pd.Series, max_lags: int = 20) -> Dict[int, float]:
    """
    Compute autocorrelation for a time series up to max_lags.

    Args:
        series: Pandas Series of numeric values
        max_lags: Maximum number of lags to compute

    Returns:
        Dictionary mapping lag to autocorrelation coefficient
    """
    if len(series) < max_lags + 1:
        max_lags = len(series) - 1

    autocorr = {}
    for lag in range(1, max_lags + 1):
        corr = series.corr(series.shift(lag))
        if not np.isnan(corr):
            autocorr[lag] = corr

    return autocorr


def estimate_memory_length(autocorr: Dict[int, float], threshold: float = 0.1) -> int:
    """
    Estimate the memory length of a time series based on autocorrelation decay.

    Args:
        autocorr: Dictionary of lag to autocorrelation values
        threshold: Correlation threshold below which memory is considered lost

    Returns:
        Estimated memory length (number of significant lags)
    """
    significant_lags = [lag for lag, corr in autocorr.items() if abs(corr) > threshold]
    return max(significant_lags) if significant_lags else 0


def lag_importance_report(data: pd.DataFrame, target_col: str, device_col: str = "device_id",
                         max_lags: int = 10) -> pd.DataFrame:
    """
    Generate a report of lag feature importance for each device.

    Args:
        data: DataFrame with time series data
        target_col: Column name of the target variable
        device_col: Column name of device identifier
        max_lags: Maximum lags to analyze

    Returns:
        DataFrame with lag importance metrics per device
    """
    reports = []

    for device_id, group in data.groupby(device_col):
        if len(group) < max_lags + 1:
            continue

        series = group[target_col].dropna()
        if len(series) < max_lags + 1:
            continue

        autocorr = compute_autocorrelation(series, max_lags)
        memory_length = estimate_memory_length(autocorr)

        # Compute lag correlations with target
        lag_features = {}
        for lag in range(1, max_lags + 1):
            lagged = series.shift(lag)
            corr = series.corr(lagged)
            lag_features[f"lag_{lag}_corr"] = corr if not np.isnan(corr) else 0.0

        report = {
            "device_id": device_id,
            "memory_length": memory_length,
            "sample_size": len(series),
            **lag_features
        }
        reports.append(report)

    return pd.DataFrame(reports)
