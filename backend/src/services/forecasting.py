"""
Forecasting Engine
Uses a Decomposable Time-Series approach (mirroring Facebook Prophet's methodology)
implemented with NumPy, Pandas, and statsmodels — no C++ build required.

Mathematical approach:
  y(t) = g(t) + s(t) + ε

  g(t) : Linear trend via linear regression (OLS)
  s(t) : Fourier Series seasonality  Σ [aₙ·cos(2πnt/P) + bₙ·sin(2πnt/P)]
  CI   : Bootstrap resampling for 95% confidence intervals
  Anomalies: Points where actual y is outside the 95% CI band
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from typing import Optional


def _add_fourier_features(t: np.ndarray, period: float, order: int = 3) -> np.ndarray:
    """
    Generates Fourier Series features for capturing seasonality in regression.
    Identical mathematical foundation used by Facebook Prophet.

    For each harmonic n=1..order:
      cos(2πnt / P)  and  sin(2πnt / P)

    Args:
        t: Integer time index array.
        period: Season length (7 for weekly, 365.25 for yearly).
        order: Number of Fourier harmonics.

    Returns:
        Array of shape (len(t), 2*order) — Fourier feature matrix.
    """
    features = []
    for n in range(1, order + 1):
        features.append(np.cos(2 * np.pi * n * t / period))
        features.append(np.sin(2 * np.pi * n * t / period))
    return np.column_stack(features)


def _fit_decomposable_model(df: pd.DataFrame):
    """
    Fits a Decomposable Time-Series model:
      y(t) = β₀ + β₁·t + Fourier_weekly(t) + Fourier_yearly(t) + ε

    Returns the fitted model and feature matrix for prediction.
    """
    n = len(df)
    t = np.arange(n, dtype=float)

    # Build feature matrix: [trend, weekly Fourier, yearly Fourier]
    trend_feat = t.reshape(-1, 1)
    weekly_feat = _add_fourier_features(t, period=7, order=3)    # 6 features
    yearly_feat = _add_fourier_features(t, period=365.25, order=5)  # 10 features

    X = np.hstack([trend_feat, weekly_feat, yearly_feat])
    y = df["y"].values

    model = LinearRegression()
    model.fit(X, y)

    return model, X, t, y


def _bootstrap_confidence_interval(
    residuals: np.ndarray,
    yhat: np.ndarray,
    n_bootstrap: int = 500,
    ci: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimates confidence intervals via bootstrap resampling of residuals.
    This is analogous to Prophet's Bayesian posterior sampling.

    Args:
        residuals: In-sample residuals (y - yhat).
        yhat: Fitted/predicted values.
        n_bootstrap: Number of bootstrap iterations.
        ci: Confidence level (0.95 = 95%).

    Returns:
        (lower_bound, upper_bound) arrays.
    """
    np.random.seed(42)
    n = len(yhat)
    boot_preds = np.zeros((n_bootstrap, n))

    for i in range(n_bootstrap):
        sampled_residuals = np.random.choice(residuals, size=n, replace=True)
        boot_preds[i] = yhat + sampled_residuals

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_preds, alpha * 100, axis=0)
    upper = np.percentile(boot_preds, (1 - alpha) * 100, axis=0)

    return lower, upper


def run_forecast(
    data: list[dict],
    forecast_weeks: int = 4,
    context_label: str = "Metric",
) -> dict:
    """
    Full forecasting pipeline — pure Python/NumPy, no external binaries needed.

    Args:
        data: List of {ds: 'YYYY-MM-DD', y: float} dicts.
        forecast_weeks: Number of future weeks to forecast (1–6).
        context_label: Human-readable name of the metric (for LLM prompts).

    Returns:
        Dictionary with:
          - historical_fit: fitted values on historical data
          - forecast: future predictions with 95% confidence intervals
          - anomalies: historical outliers outside the CI band
          - summary_stats: key numbers for LLM insight generation
    """
    df = pd.DataFrame(data)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    df["y"] = df["y"].astype(float)

    n_hist = len(df)

    # Fit the model on historical data
    model, X_hist, t_hist, y_hist = _fit_decomposable_model(df)
    yhat_hist = model.predict(X_hist)

    # Cap predictions to non-negative (realistic metrics)
    yhat_hist = np.clip(yhat_hist, 0, None)

    # Bootstrap CI on historical fit
    residuals = y_hist - yhat_hist
    lower_hist, upper_hist = _bootstrap_confidence_interval(residuals, yhat_hist)

    # --- Anomaly Detection ---
    anomaly_mask = (y_hist > upper_hist) | (y_hist < lower_hist)
    anomaly_df = df[anomaly_mask].copy()
    anomaly_df["yhat"] = yhat_hist[anomaly_mask]
    anomaly_df["yhat_lower"] = lower_hist[anomaly_mask]
    anomaly_df["yhat_upper"] = upper_hist[anomaly_mask]
    anomaly_df["direction"] = np.where(
        anomaly_df["y"] > anomaly_df["yhat_upper"], "spike", "drop"
    )
    anomaly_df["pct_deviation"] = (
        (anomaly_df["y"] - anomaly_df["yhat"]) / anomaly_df["yhat"] * 100
    ).round(1)
    anomaly_df["ds"] = anomaly_df["ds"].dt.strftime("%Y-%m-%d")

    # --- Future Forecast ---
    forecast_days = forecast_weeks * 7
    t_future = np.arange(n_hist, n_hist + forecast_days, dtype=float)

    weekly_feat_future = _add_fourier_features(t_future, period=7, order=3)
    yearly_feat_future = _add_fourier_features(t_future, period=365.25, order=5)
    X_future = np.hstack([t_future.reshape(-1, 1), weekly_feat_future, yearly_feat_future])

    yhat_future = model.predict(X_future)
    yhat_future = np.clip(yhat_future, 0, None)

    # For future CI, use wider intervals (uncertainty grows with time)
    # Scale residual std by sqrt(steps) — standard statistical convention
    residual_std = np.std(residuals)
    horizon_scale = np.sqrt(np.arange(1, forecast_days + 1) / forecast_days)
    lower_future = yhat_future - 1.96 * residual_std * (1 + horizon_scale)
    upper_future = yhat_future + 1.96 * residual_std * (1 + horizon_scale)
    lower_future = np.clip(lower_future, 0, None)

    # Build future date list
    last_date = df["ds"].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D")

    forecast_records = [
        {
            "ds": d.strftime("%Y-%m-%d"),
            "yhat": round(float(y), 2),
            "yhat_lower": round(float(lo), 2),
            "yhat_upper": round(float(hi), 2),
        }
        for d, y, lo, hi in zip(future_dates, yhat_future, lower_future, upper_future)
    ]

    # Historical fit records (attach actual y for chart)
    hist_fit_records = [
        {
            "ds": df["ds"].iloc[i].strftime("%Y-%m-%d"),
            "y": round(float(y_hist[i]), 2),
            "yhat": round(float(yhat_hist[i]), 2),
            "yhat_lower": round(float(lower_hist[i]), 2),
            "yhat_upper": round(float(upper_hist[i]), 2),
        }
        for i in range(n_hist)
    ]

    # --- Summary Stats for LLM ---
    last_actual = float(y_hist[-1])
    forecast_end = float(yhat_future[-1])
    forecast_lower_end = float(lower_future[-1])
    forecast_upper_end = float(upper_future[-1])
    growth_pct = round((forecast_end - last_actual) / max(last_actual, 1) * 100, 1)

    peak_idx = np.argmax(yhat_future)
    peak_date = future_dates[peak_idx].strftime("%Y-%m-%d")

    anomaly_records = anomaly_df.to_dict(orient="records")

    summary_stats = {
        "context_label": context_label,
        "forecast_weeks": forecast_weeks,
        "current_value": round(last_actual, 2),
        "forecast_end_value": round(forecast_end, 2),
        "forecast_end_lower": round(forecast_lower_end, 2),
        "forecast_end_upper": round(forecast_upper_end, 2),
        "growth_pct_over_period": growth_pct,
        "peak_predicted_date": peak_date,
        "anomaly_count": len(anomaly_records),
        "anomaly_spike_count": int(sum(1 for a in anomaly_records if a["direction"] == "spike")),
        "anomaly_drop_count": int(sum(1 for a in anomaly_records if a["direction"] == "drop")),
    }

    return {
        "historical_fit": hist_fit_records,
        "forecast": forecast_records,
        "anomalies": anomaly_records,
        "summary_stats": summary_stats,
    }


def run_scenario_forecast(
    data: list[dict],
    growth_multiplier: float = 1.0,
    seasonality_strength: float = 1.0,
    forecast_weeks: int = 4,
) -> dict:
    """
    Runs a 'What-If' scenario by modifying the underlying data trend
    and comparing against the unmodified baseline.

    Args:
        data: Original historical data.
        growth_multiplier: e.g., 1.10 = +10% overall scale shift.
        seasonality_strength: Amplifies seasonal deviations around the mean.
        forecast_weeks: How many weeks forward to project.

    Returns:
        Both baseline and scenario forecast dicts for side-by-side comparison.
    """
    # Baseline (no modification)
    baseline = run_forecast(data, forecast_weeks)

    # Scenario: Scale y values
    df_mod = pd.DataFrame(data).copy()
    df_mod["y"] = pd.to_numeric(df_mod["y"])

    # Apply trend multiplier
    df_mod["y"] = df_mod["y"] * growth_multiplier

    # Apply seasonality strength (amplify/dampen deviations around the mean)
    if seasonality_strength != 1.0:
        mean_val = df_mod["y"].mean()
        deviation = df_mod["y"] - mean_val
        df_mod["y"] = mean_val + deviation * seasonality_strength

    df_mod["y"] = df_mod["y"].clip(lower=0)

    scenario = run_forecast(df_mod.to_dict(orient="records"), forecast_weeks)

    return {
        "baseline_forecast": baseline["forecast"],
        "scenario_forecast": scenario["forecast"],
        "baseline_stats": baseline["summary_stats"],
        "scenario_stats": scenario["summary_stats"],
    }
