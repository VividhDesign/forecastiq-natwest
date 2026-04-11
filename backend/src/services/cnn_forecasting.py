"""
1D CNN Time-Series Forecaster
=============================
A lightweight Convolutional Neural Network for short-term time-series forecasting.
Uses PyTorch with a sliding-window approach to learn local temporal patterns.

Architecture:
    Input: [batch, 1, window_size]  →  Conv1D(32, k=3)  →  ReLU
                                    →  Conv1D(16, k=3)  →  ReLU
                                    →  Flatten  →  Dense(32)  →  Dropout(0.2)
                                    →  Dense(1)

Why 1D CNN for time series?
    - Captures local patterns (spikes, dips, weekly shapes) via convolutional filters
    - Much faster to train than LSTM/GRU (no sequential bottleneck)
    - Works well with limited data (our 730-day dataset)
    - ~2,000 parameters — trains in seconds on CPU

Confidence Intervals:
    Uses MC Dropout — runs the model N times at inference with dropout enabled,
    producing a distribution of predictions. The 2.5th and 97.5th percentiles
    give us a 95% confidence band (analogous to Bayesian uncertainty).

Comparison with Classical Model:
    The classical OLS+Fourier model is mathematically optimal for linear trends
    and periodic seasonality. The CNN model excels at capturing non-linear
    patterns, sudden regime changes, and complex interactions that the classical
    model may miss. Running both and comparing lets us determine which approach
    is more appropriate for a given dataset.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ─── Model Architecture ────────────────────────────────────────────────────────

class TimeSeriesCNN(nn.Module):
    """
    Lightweight 1D CNN for univariate time-series forecasting.
    Takes a window of past values and predicts the next value.
    """

    def __init__(self, window_size: int = 28, dropout_rate: float = 0.2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * window_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x shape: (batch, 1, window_size)
        out = self.conv_block(x)
        out = self.regressor(out)
        return out.squeeze(-1)


# ─── Data Preparation ──────────────────────────────────────────────────────────

def _create_sliding_windows(y: np.ndarray, window_size: int = 28):
    """
    Creates sliding window input-output pairs for supervised learning.

    Given a time series [v0, v1, v2, ..., vN], generates:
        X[i] = [v_i, v_{i+1}, ..., v_{i+window-1}]
        y[i] = v_{i+window}

    Args:
        y: 1D array of time-series values.
        window_size: Number of past values to use as input.

    Returns:
        X: Array of shape (n_samples, window_size)
        targets: Array of shape (n_samples,)
    """
    X, targets = [], []
    for i in range(len(y) - window_size):
        X.append(y[i : i + window_size])
        targets.append(y[i + window_size])
    return np.array(X), np.array(targets)


def _normalize(y: np.ndarray):
    """Min-max normalization to [0, 1] range for stable CNN training."""
    y_min, y_max = np.min(y), np.max(y)
    scale = y_max - y_min if y_max != y_min else 1.0
    return (y - y_min) / scale, y_min, scale


def _denormalize(y_norm: np.ndarray, y_min: float, scale: float):
    """Reverse the min-max normalization."""
    return y_norm * scale + y_min


# ─── Training ──────────────────────────────────────────────────────────────────

def _train_cnn(
    y: np.ndarray,
    window_size: int = 28,
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 32,
):
    """
    Trains the 1D CNN model on historical data.

    Args:
        y: Historical time-series values (1D array).
        window_size: Lookback window size.
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.
        batch_size: Mini-batch size.

    Returns:
        Trained model, normalization parameters, and training loss history.
    """
    # Normalize
    y_norm, y_min, y_scale = _normalize(y.astype(float))

    # Create windows
    X, targets = _create_sliding_windows(y_norm, window_size)

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X).unsqueeze(1)  # (N, 1, window_size)
    y_tensor = torch.FloatTensor(targets)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = TimeSeriesCNN(window_size=window_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_history.append(epoch_loss / len(loader))

    return model, y_min, y_scale, loss_history


# ─── Inference with MC Dropout ──────────────────────────────────────────────────

def _mc_dropout_predict(
    model: TimeSeriesCNN,
    last_window: np.ndarray,
    forecast_days: int,
    y_min: float,
    y_scale: float,
    n_samples: int = 100,
):
    """
    Multi-step forecast using Monte Carlo Dropout for uncertainty estimation.

    At each step:
      1. Run the model N times with dropout ENABLED (not eval mode)
      2. Get N different predictions → distribution
      3. Mean = point forecast, 2.5th/97.5th percentile = 95% CI
      4. Append mean to window and slide forward

    Args:
        model: Trained CNN model.
        last_window: Last `window_size` normalized values.
        forecast_days: Number of days to forecast.
        y_min, y_scale: Normalization parameters for denormalization.
        n_samples: Number of MC dropout forward passes per step.

    Returns:
        yhat, yhat_lower, yhat_upper — all denormalized.
    """
    model.train()  # Keep dropout active for MC sampling

    window = last_window.copy()
    predictions = []
    lowers = []
    uppers = []

    for step in range(forecast_days):
        x = torch.FloatTensor(window[-len(last_window):]).unsqueeze(0).unsqueeze(0)

        # MC Dropout: multiple forward passes
        mc_preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                # Re-enable dropout manually for each pass
                model.train()
                pred = model(x).item()
                mc_preds.append(pred)

        mc_preds = np.array(mc_preds)
        mean_pred = np.mean(mc_preds)
        lower_pred = np.percentile(mc_preds, 2.5)
        upper_pred = np.percentile(mc_preds, 97.5)

        predictions.append(mean_pred)
        lowers.append(lower_pred)
        uppers.append(upper_pred)

        # Slide window forward
        window = np.append(window, mean_pred)

    # Denormalize
    yhat = _denormalize(np.array(predictions), y_min, y_scale)
    yhat_lower = _denormalize(np.array(lowers), y_min, y_scale)
    yhat_upper = _denormalize(np.array(uppers), y_min, y_scale)

    # Clip to non-negative
    yhat = np.clip(yhat, 0, None)
    yhat_lower = np.clip(yhat_lower, 0, None)

    return yhat, yhat_lower, yhat_upper


# ─── Public API ────────────────────────────────────────────────────────────────

def run_cnn_forecast(
    data: list[dict],
    forecast_weeks: int = 4,
    context_label: str = "Metric",
    window_size: int = 28,
    epochs: int = 50,
) -> dict:
    """
    Full CNN forecasting pipeline.

    Args:
        data: List of {ds: 'YYYY-MM-DD', y: float} dicts.
        forecast_weeks: Number of future weeks to forecast (1-6).
        context_label: Human-readable name of the metric.
        window_size: CNN lookback window (default 28 days = 4 weeks).
        epochs: Training epochs (default 50, enough for small data).

    Returns:
        Dictionary with forecast, accuracy metrics, and training info.
    """
    df = pd.DataFrame(data)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    df["y"] = df["y"].astype(float)

    y = df["y"].values
    n_hist = len(y)
    forecast_days = forecast_weeks * 7

    # ─── Train/Validation Split for accuracy metrics ───
    # Use last 20% as holdout for computing metrics
    holdout_size = max(int(n_hist * 0.2), window_size + 1)
    train_y = y[:-holdout_size]
    holdout_y = y[-holdout_size:]

    # Train the model on training split
    model, y_min, y_scale, loss_history = _train_cnn(
        train_y, window_size=window_size, epochs=epochs
    )

    # Validate on holdout
    y_norm_full, _, _ = _normalize(y.astype(float))
    # Use the same normalization params from training
    y_norm_train = (train_y - y_min) / y_scale

    # Get holdout predictions (one-step-ahead on holdout)
    holdout_preds = []
    model.eval()
    with torch.no_grad():
        for i in range(len(holdout_y)):
            idx = len(train_y) - window_size + i
            if idx < 0:
                continue
            window_data = (y[:len(train_y) + i] - y_min) / y_scale
            w = window_data[-window_size:]
            x = torch.FloatTensor(w).unsqueeze(0).unsqueeze(0)
            pred = model(x).item()
            holdout_preds.append(_denormalize(pred, y_min, y_scale))

    holdout_actual = holdout_y[:len(holdout_preds)]
    holdout_preds = np.array(holdout_preds)

    # ─── Accuracy Metrics ───
    mae = float(np.mean(np.abs(holdout_actual - holdout_preds)))
    rmse = float(np.sqrt(np.mean((holdout_actual - holdout_preds) ** 2)))
    mape = float(np.mean(np.abs((holdout_actual - holdout_preds) / np.clip(holdout_actual, 1, None))) * 100)

    # ─── Retrain on FULL data for final forecast ───
    model_full, y_min_full, y_scale_full, _ = _train_cnn(
        y, window_size=window_size, epochs=epochs
    )

    # Normalize full series for last window
    y_norm_full_final = (y - y_min_full) / y_scale_full
    last_window = y_norm_full_final[-window_size:]

    # MC Dropout forecast
    yhat_future, yhat_lower, yhat_upper = _mc_dropout_predict(
        model_full, last_window, forecast_days, y_min_full, y_scale_full
    )

    # Build future dates
    last_date = df["ds"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D"
    )

    forecast_records = [
        {
            "ds": d.strftime("%Y-%m-%d"),
            "yhat": round(float(yh), 2),
            "yhat_lower": round(float(lo), 2),
            "yhat_upper": round(float(hi), 2),
        }
        for d, yh, lo, hi in zip(future_dates, yhat_future, yhat_lower, yhat_upper)
    ]

    # ─── Historical fitted values (one-step predictions on full training data) ───
    model_full.eval()
    hist_preds = []
    y_norm_for_hist = (y - y_min_full) / y_scale_full
    with torch.no_grad():
        for i in range(window_size, n_hist):
            w = y_norm_for_hist[i - window_size : i]
            x = torch.FloatTensor(w).unsqueeze(0).unsqueeze(0)
            pred = model_full(x).item()
            hist_preds.append(_denormalize(pred, y_min_full, y_scale_full))

    hist_fit_records = [
        {
            "ds": df["ds"].iloc[i + window_size].strftime("%Y-%m-%d"),
            "y": round(float(y[i + window_size]), 2),
            "yhat": round(float(hp), 2),
        }
        for i, hp in enumerate(hist_preds)
        if i + window_size < n_hist
    ]

    # ─── Summary Stats ───
    last_actual = float(y[-1])
    forecast_end = float(yhat_future[-1])
    growth_pct = round((forecast_end - last_actual) / max(last_actual, 1) * 100, 1)

    summary_stats = {
        "model_name": "1D CNN",
        "context_label": context_label,
        "forecast_weeks": forecast_weeks,
        "current_value": round(last_actual, 2),
        "forecast_end_value": round(forecast_end, 2),
        "forecast_end_lower": round(float(yhat_lower[-1]), 2),
        "forecast_end_upper": round(float(yhat_upper[-1]), 2),
        "growth_pct_over_period": growth_pct,
        "window_size": window_size,
        "epochs": epochs,
        "final_train_loss": round(loss_history[-1], 6),
        "total_parameters": sum(p.numel() for p in model_full.parameters()),
        # Accuracy metrics (on holdout set)
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
    }

    return {
        "forecast": forecast_records,
        "historical_fit": hist_fit_records,
        "summary_stats": summary_stats,
        "accuracy_metrics": {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
            "holdout_size": len(holdout_preds),
        },
    }
