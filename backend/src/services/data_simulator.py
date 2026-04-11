"""
Synthetic Data Simulator
Generates realistic, privacy-safe time-series data using:
  - Linear trend with optional mid-series changepoint (g(t))
  - Fourier Series for weekly & yearly seasonality (s(t))
  - Gaussian noise
  - Deliberate anomaly injection for demo purposes

Design decisions
----------------
* A SINGLE isolated numpy RNG (seeded from microsecond clock) is created at
  the top of generate_synthetic_data() and threaded through every random
  operation.  No call touches np.random.<global> — so the global NumPy state
  is never mutated and forecast bootstrap CIs remain reproducible.

* _fourier_seasonality() accepts a random PHASE OFFSET per harmonic.
  Without this, the yearly sinusoid always starts at t=0 (= dataset start
  date) with zero phase, causing constructive interference of 5 harmonics
  to always peak at roughly the same calendar month regardless of the RNG
  seed.  Adding φ ∈ [0, 2π) per harmonic randomises which part of the
  year experiences the seasonal peak on every simulation run.
"""

import time
import numpy as np
from datetime import datetime, timedelta
from typing import Literal


# ─── Trend Configurations ─────────────────────────────────────────────────────

TREND_CONFIGS = {
    "aggressive_growth": {"slope": 3.5,  "base": 1000},
    "stable":            {"slope": 0.3,  "base": 1000},
    "declining":         {"slope": -2.0, "base": 1500},
}

# ─── Dataset Context Metadata (for LLM prompting) ─────────────────────────────

CONTEXT_META = {
    "ecommerce_sales": {
        "label":       "E-commerce Daily Sales",
        "unit":        "USD",
        "description": "Daily revenue from an e-commerce platform",
    },
    "server_load": {
        "label":       "Cloud Server CPU Load",
        "unit":        "%",
        "description": "Average daily CPU utilization of cloud servers",
    },
    "user_signups": {
        "label":       "App User Signups",
        "unit":        "users",
        "description": "Daily new user registrations for a SaaS application",
    },
    "support_tickets": {
        "label":       "Customer Support Tickets",
        "unit":        "tickets",
        "description": "Daily incoming customer support requests",
    },
}


# ─── Fourier Seasonality ──────────────────────────────────────────────────────

def _fourier_seasonality(
    t: np.ndarray,
    period: float,
    order: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a Fourier Series-based seasonal pattern.

    Mathematically identical to Facebook Prophet's seasonality:
        s(t) = Σ [a_n·cos(2πnt/P + φ_n) + b_n·sin(2πnt/P + φ_n)]

    The random phase offset φ_n ∈ [0, 2π) is the key addition over the
    naive implementation.  Without it, the fundamental harmonic always
    starts at cos(0)=1 at t=0, so constructive interference of N harmonics
    reliably creates a large peak at the same relative position in every
    dataset — making the yearly spike appear at the same calendar month
    every time.  Randomising φ_n per harmonic breaks this alignment.

    Args:
        t:      Array of time indices (days 0 … n-1).
        period: Repeating cycle length (7 = weekly, 365.25 = yearly).
        order:  Number of Fourier harmonics.
        rng:    Isolated numpy Generator — never touches global state.
    """
    seasonal = np.zeros_like(t, dtype=float)
    for n in range(1, order + 1):
        a_n     = rng.normal(0, 5)
        b_n     = rng.normal(0, 5)
        phi     = rng.uniform(0, 2 * np.pi)          # random phase offset ← key fix
        angle   = 2 * np.pi * n * t / period + phi
        seasonal += a_n * np.cos(angle) + b_n * np.sin(angle)
    return seasonal


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_synthetic_data(
    context: str = "ecommerce_sales",
    trend_type: Literal["aggressive_growth", "stable", "declining"] = "aggressive_growth",
    days: int = 730,
    inject_anomalies: bool = True,
) -> dict:
    """
    Generate a synthetic time-series dataset for predictive forecasting.

    Every call is independent and produces a genuinely different dataset:
      - Different seasonal peak months (random Fourier phase per harmonic)
      - Different anomaly positions (random from middle 80% of series)
      - Different noise pattern (same isolated RNG, freshly seeded each call)

    Args:
        context:          Business domain (ecommerce_sales | server_load |
                          user_signups | support_tickets).
        trend_type:       Overall direction (aggressive_growth | stable | declining).
        days:             Length of the historical series (default 730 = 2 years).
        inject_anomalies: Whether to inject deliberate outlier events.

    Returns:
        dict with keys:
          'data'                  — list of {ds: str, y: float} (Prophet-compatible)
          'context_meta'          — metadata dict for LLM prompting
          'injected_anomaly_dates'— dates where anomalies were injected
    """
    # ── Single isolated RNG for the entire generation ─────────────────────────
    # Microsecond-clock seed → every function call is genuinely independent.
    # NEVER touches np.random.<global> so forecast bootstrap stays reproducible.
    rng = np.random.default_rng(int(time.time() * 1_000_000) % (2 ** 31))

    config = TREND_CONFIGS.get(trend_type, TREND_CONFIGS["stable"])
    meta   = CONTEXT_META.get(context, CONTEXT_META["ecommerce_sales"])

    # ── g(t): Linear Trend ───────────────────────────────────────────────────
    t     = np.arange(days)
    trend = config["base"] + config["slope"] * t

    # ── s(t): Fourier Seasonality ─────────────────────────────────────────────
    # Weekly pattern  (period=7, 3 harmonics)
    weekly_seasonality  = _fourier_seasonality(t, period=7,      order=3, rng=rng)
    # Yearly pattern  (period=365.25, 5 harmonics) — scaled for visibility
    yearly_seasonality  = _fourier_seasonality(t, period=365.25, order=5, rng=rng) * 2

    # ── ε: Gaussian Noise ─────────────────────────────────────────────────────
    noise = rng.normal(0, config["base"] * 0.03, days)

    # ── Combine: y(t) = g(t) + s(t) + ε ──────────────────────────────────────
    y = trend + weekly_seasonality + yearly_seasonality + noise
    y = np.clip(y, 1, None)   # enforce positive values

    # ── Anomaly Injection ─────────────────────────────────────────────────────
    injected_anomaly_indices = []
    if inject_anomalies:
        # 2 spikes + 2 drops, all in the middle 80% of the dataset so they're
        # visible in the chart and don't overlap the forecast region.
        pool         = np.arange(60, days - 60)
        spike_indices = rng.choice(pool, size=2, replace=False)
        drop_indices  = rng.choice(pool, size=2, replace=False)

        for idx in spike_indices:
            y[idx] *= rng.uniform(2.5, 4.0)        # Spike: 2.5× – 4.0×
            injected_anomaly_indices.append(int(idx))

        for idx in drop_indices:
            y[idx] *= rng.uniform(0.1, 0.3)        # Drop: 70 – 90% crash
            injected_anomaly_indices.append(int(idx))

    # ── Build output ──────────────────────────────────────────────────────────
    start_date = datetime.today() - timedelta(days=days)
    dates      = [start_date + timedelta(days=i) for i in range(days)]

    data = [
        {"ds": d.strftime("%Y-%m-%d"), "y": round(float(v), 2)}
        for d, v in zip(dates, y)
    ]

    injected_anomaly_dates = [
        dates[i].strftime("%Y-%m-%d") for i in injected_anomaly_indices
    ]

    return {
        "data":                   data,
        "context_meta":           meta,
        "injected_anomaly_dates": injected_anomaly_dates,
    }
