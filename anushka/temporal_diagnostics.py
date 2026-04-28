#!/usr/bin/env python3
"""
Temporal Diagnostics Module for Feature Stability and Predictive Utility Analysis

This module provides functions to check feature stability, generate predictive
utility reports, and export diagnostic summaries.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import json


def feature_stability_check(data: pd.DataFrame, feature_cols: List[str],
                          time_col: str, window_size: int = 100) -> pd.DataFrame:
    """
    Check the stability of features over time using rolling statistics.

    Args:
        data: DataFrame with features and timestamps
        feature_cols: List of feature column names
        time_col: Column name of timestamps
        window_size: Size of rolling window for stability calculation

    Returns:
        DataFrame with stability metrics for each feature
    """
    data = data.sort_values(time_col).copy()
    stability_metrics = []

    for feature in feature_cols:
        if feature not in data.columns:
            continue

        series = data[feature].dropna()
        if len(series) < window_size:
            continue

        # Rolling mean and std
        rolling_mean = series.rolling(window=window_size).mean()
        rolling_std = series.rolling(window=window_size).std()

        # Coefficient of variation (stability measure)
        cv = (rolling_std / rolling_mean).mean()

        # Trend in mean (drift detection)
        mean_trend = np.polyfit(np.arange(len(rolling_mean.dropna())), rolling_mean.dropna(), 1)[0]

        # Autocorrelation stability
        autocorr = series.corr(series.shift(1))

        stability_metrics.append({
            'feature': feature,
            'coefficient_of_variation': cv,
            'mean_trend': mean_trend,
            'autocorrelation': autocorr,
            'sample_size': len(series)
        })

    return pd.DataFrame(stability_metrics)


def predictive_utility_report(data: pd.DataFrame, feature_cols: List[str],
                            target_col: str, time_col: str) -> pd.DataFrame:
    """
    Generate a report on the predictive utility of features.

    Args:
        data: DataFrame with features, target, and timestamps
        feature_cols: List of feature column names
        target_col: Target column name
        time_col: Column name of timestamps

    Returns:
        DataFrame with predictive metrics for each feature
    """
    data = data.dropna(subset=[target_col] + feature_cols).copy()
    if data.empty:
        return pd.DataFrame()

    utility_metrics = []

    for feature in feature_cols:
        if feature not in data.columns:
            continue

        # Simple linear regression
        X = data[[feature]]
        y = data[target_col]

        if len(X) < 10:  # Minimum sample size
            continue

        model = LinearRegression()
        scores = cross_val_score(model, X, y, cv=min(5, len(X)), scoring='r2')

        # Fit on full data for additional metrics
        model.fit(X, y)
        y_pred = model.predict(X)

        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        correlation = data[feature].corr(data[target_col])

        utility_metrics.append({
            'feature': feature,
            'r2_score': r2,
            'mean_cv_r2': scores.mean(),
            'mse': mse,
            'correlation_with_target': correlation,
            'sample_size': len(X)
        })

    return pd.DataFrame(utility_metrics)


def export_summary(stability_df: pd.DataFrame, utility_df: pd.DataFrame,
                  output_path: str) -> None:
    """
    Export diagnostic summary to JSON file.

    Args:
        stability_df: DataFrame from feature_stability_check
        utility_df: DataFrame from predictive_utility_report
        output_path: Path to output JSON file
    """
    summary = {
        stability_analysis: stability_df.to_dict(records),
        utility_analysis: utility_df.to_dict(records),
        summary_stats: {
            total_features_analyzed: len(stability_df),
            features_with_good_stability: len(stability_df[stability_df[coefficient_of_variation] < 0.5]),
            features_with_high_utility: len(utility_df[utility_df[r2_score] > 0.5])
        }
    }

    with open(output_path, w) as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Diagnostic summary exported to {output_path}")


def comprehensive_temporal_diagnostic(data: pd.DataFrame, feature_cols: List[str],
                                    target_col: str, time_col: str,
                                    output_path: Optional[str] = None) -> Dict:
    """
    Run comprehensive temporal diagnostics.

    Args:
        data: DataFrame with time series data
        feature_cols: List of feature column names
        target_col: Target column name
        time_col: Column name of timestamps
        output_path: Optional path to export results

    Returns:
        Dictionary with all diagnostic results
    """
    print("Running feature stability check...")
    stability = feature_stability_check(data, feature_cols, time_col)

    print("Running predictive utility analysis...")
    utility = predictive_utility_report(data, feature_cols, target_col, time_col)

    results = {
        stability_metrics: stability,
        utility_metrics: utility
    }

    if output_path:
        export_summary(stability, utility, output_path)

    return results
