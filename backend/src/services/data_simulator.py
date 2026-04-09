"""
Synthetic Data Simulator
Generates realistic, privacy-safe time-series data using:
  - Linear/logistic trend (g(t))
  - Fourier Series for weekly & yearly seasonality (s(t))
  - Gaussian noise
  - Deliberate anomaly injection for demo purposes
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Literal

# Trend configurations
TREND_CONFIGS = {
    "aggressive_growth": {"slope": 3.5, "base": 1000},
    "stable": {"slope": 0.3, "base": 1000},
    "declining": {"slope": -2.0, "base": 1500},
}

# Dataset context metadata for LLM prompting
CONTEXT_META = {
    "ecommerce_sales": {
        "label": "E-commerce Daily Sales",
        "unit": "USD",
        "description": "Daily revenue from an e-commerce platform",
    },
    "server_load": {
        "label": "Cloud Server CPU Load",
        "unit": "%",
        "description": "Average daily CPU utilization of cloud servers",
    },
    "user_signups": {
        "label": "App User Signups",
        "unit": "users",
        "description": "Daily new user registrations for a SaaS application",
    },
    "support_tickets": {
        "label": "Customer Support Tickets",
        "unit": "tickets",
        "description": "Daily incoming customer support requests",
    },
}


def _fourier_seasonality(t: np.ndarray, period: float, order: int = 3) -> np.ndarray:
    """
    Generate a Fourier Series-based seasonal pattern.
    This is mathematically identical to how Facebook Prophet computes seasonality.

    s(t) = Σ [a_n * cos(2πnt/P) + b_n * sin(2πnt/P)]  for n = 1..order

    Args:
        t: Array of time indices (days).
        period: The repeating cycle length (e.g., 7 for weekly).
        order: Number of Fourier harmonics (higher = more complex pattern).

    Returns:
        Seasonal component array of the same shape as t.
    """
    np.random.seed(42)  # Reproducible but realistic coefficients
    seasonal = np.zeros_like(t, dtype=float)
    for n in range(1, order + 1):
        a_n = np.random.normal(0, 5)
        b_n = np.random.normal(0, 5)
        seasonal += a_n * np.cos(2 * np.pi * n * t / period)
        seasonal += b_n * np.sin(2 * np.pi * n * t / period)
    return seasonal


def generate_synthetic_data(
    context: str = "ecommerce_sales",
    trend_type: Literal["aggressive_growth", "stable", "declining"] = "aggressive_growth",
    days: int = 730,
    inject_anomalies: bool = True,
) -> dict:
    """
    Generate a synthetic time-series dataset for predictive forecasting.

    Args:
        context: Business domain (ecommerce_sales, server_load, user_signups, support_tickets).
        trend_type: Overall direction of the trend.
        days: Total number of days of historical data to generate (default 730 = 2 years).
        inject_anomalies: Whether to inject deliberate outlier events into the data.

    Returns:
        A dictionary containing:
          - 'data': list of {ds: date_str, y: numeric_value} dicts (Prophet-compatible format)
          - 'context_meta': metadata about the dataset for LLM prompting
          - 'injected_anomaly_dates': list of date strings where anomalies were injected
    """
    config = TREND_CONFIGS.get(trend_type, TREND_CONFIGS["stable"])
    meta = CONTEXT_META.get(context, CONTEXT_META["ecommerce_sales"])

    # Time index
    t = np.arange(days)

    # --- g(t): Piecewise Linear Trend ---
    trend = config["base"] + config["slope"] * t

    # --- s(t): Fourier Seasonality ---
    # Weekly pattern (e.g., lower on weekends)
    weekly_seasonality = _fourier_seasonality(t, period=7, order=3)
    # Yearly pattern (e.g., holiday spikes)
    yearly_seasonality = _fourier_seasonality(t, period=365.25, order=5) * 2

    # --- Noise: Gaussian random ---
    np.random.seed(99)
    noise = np.random.normal(0, config["base"] * 0.03, days)

    # Combine all components: y(t) = g(t) + s(t) + h(t) + ε
    y = trend + weekly_seasonality + yearly_seasonality + noise

    # Clip to realistic positive values
    y = np.clip(y, 1, None)

    # --- Anomaly Injection ---
    injected_anomaly_indices = []
    if inject_anomalies:
        np.random.seed(7)
        # Inject 4 anomalies: 2 spikes, 2 drops
        spike_indices = np.random.choice(range(60, days - 60), size=2, replace=False)
        drop_indices = np.random.choice(range(60, days - 60), size=2, replace=False)

        for idx in spike_indices:
            y[idx] *= np.random.uniform(2.5, 4.0)  # Spike: 2.5x to 4x
            injected_anomaly_indices.append(int(idx))

        for idx in drop_indices:
            y[idx] *= np.random.uniform(0.1, 0.3)  # Drop: 70-90% crash
            injected_anomaly_indices.append(int(idx))

    # Build date series starting 'days' ago
    start_date = datetime.today() - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]

    data = [
        {"ds": d.strftime("%Y-%m-%d"), "y": round(float(v), 2)}
        for d, v in zip(dates, y)
    ]

    injected_anomaly_dates = [
        dates[i].strftime("%Y-%m-%d") for i in injected_anomaly_indices
    ]

    return {
        "data": data,
        "context_meta": meta,
        "injected_anomaly_dates": injected_anomaly_dates,
    }
