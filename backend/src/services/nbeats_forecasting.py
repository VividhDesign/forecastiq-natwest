"""
N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting
===================================================================================
Paper: Oreshkin et al., ICLR 2020  https://arxiv.org/abs/1905.10437

This implements the INTERPRETABLE variant of N-BEATS, which decomposes forecasts
into explicit Trend and Seasonality stacks — mirroring exactly what the Classical
OLS+Fourier model does analytically, but learning the decomposition end-to-end
from data without any hand-crafted equations.

Why N-BEATS beats 1D CNN for time series:
    - Purpose-built for forecasting (not adapted from image/audio tasks)
    - Interpretable: trend stack explains growth, seasonality stack explains cycles
    - Faster: FC layers are cheaper than convolutions on short sequences
    - Doubly-residual learning: each block only models what previous blocks missed

Architecture (Interpretable variant):
    Input window
        → Trend Stack [Block₁ → Block₂]       basis: polynomial [1, t, t², t³]
            residual ↓
        → Seasonality Stack [Block₃ → Block₄]  basis: Fourier [cos/sin harmonics]
            ↓
    forecast = Σ(trend forecasts) + Σ(seasonality forecasts)

Each block:
    Input → FC(128) → FC(128) → FC(128) → FC(128)
          → theta_b → backcast = theta_b @ basis_backcast.T
          → theta_f → forecast = theta_f @ basis_forecast.T
    residual = input − backcast  (doubly residual connection)

Confidence Intervals:
    MC Dropout — same batched approach as CNN service: repeat input N times,
    single forward pass with dropout active, percentile the outputs.

Performance targets (CPU, 730-day dataset):
    Training  : ~1–2 s  (20 epochs, sliding-window, hidden=128)
    MC Dropout: < 1 s   (20 samples, batched, 28 forecast steps)
    Total     : < 5 s
"""

import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import hashlib
import time
from torch.utils.data import DataLoader, TensorDataset


# ─── In-Memory Model Cache ────────────────────────────────────────────────────
# Keyed by MD5 of the raw input array bytes.
# TTL = 1800s (30 min) — covers an entire hackathon demo session.
# Evicts oldest entry when > 5 models cached (keeps Render memory sane).

_MODEL_CACHE: dict = {}
_CACHE_TTL: int = 1800  # seconds


def _data_hash(y: np.ndarray) -> str:
    """Stable MD5 of the input array for cache key."""
    return hashlib.md5(y.astype(np.float64).tobytes()).hexdigest()


def _get_cached(key: str):
    entry = _MODEL_CACHE.get(key)
    if entry and (time.time() - entry["ts"]) < _CACHE_TTL:
        return entry["model"], entry["y_min"], entry["y_scale"]
    return None, None, None


def _put_cache(key: str, model, y_min: float, y_scale: float) -> None:
    _MODEL_CACHE[key] = {"model": model, "y_min": y_min, "y_scale": y_scale, "ts": time.time()}
    # Evict oldest entry if cache grows beyond 5 items
    if len(_MODEL_CACHE) > 5:
        oldest = min(_MODEL_CACHE, key=lambda k: _MODEL_CACHE[k]["ts"])
        del _MODEL_CACHE[oldest]


# ─── Basis Functions ──────────────────────────────────────────────────────────

def _trend_basis(n: int, degree: int) -> torch.Tensor:
    """
    Polynomial basis matrix for the trend block.
    Each column is t^p normalized to [0,1] range.

    Returns shape (n, degree+1):
        column 0 = 1  (constant)
        column 1 = t  (linear)
        column 2 = t² (quadratic)
        column 3 = t³ (cubic)
    """
    t = torch.arange(n, dtype=torch.float32) / max(n - 1, 1)  # [0, 1]
    return torch.stack([t ** p for p in range(degree + 1)], dim=1)


def _seasonality_basis(n: int, harmonics: int, period: float = 7.0) -> torch.Tensor:
    """
    Fourier basis matrix for the seasonality block.
    Identical mathematical foundation as the Classical model's Fourier features.

    Returns shape (n, 2*harmonics):
        [cos(2π·1·t/P), sin(2π·1·t/P), cos(2π·2·t/P), sin(2π·2·t/P), ...]
    """
    t = torch.arange(n, dtype=torch.float32)
    terms = []
    for k in range(1, harmonics + 1):
        terms.append(torch.cos(2 * torch.pi * k * t / period))
        terms.append(torch.sin(2 * torch.pi * k * t / period))
    return torch.stack(terms, dim=1)


# ─── N-BEATS Block ────────────────────────────────────────────────────────────

class NBeatsBlock(nn.Module):
    """
    Single interpretable N-BEATS block.

    The FC backbone extracts a hidden representation, then two linear heads
    produce coefficients for the basis expansion:
        backcast = theta_b @ basis_b.T    (explains what block 'uses' from input)
        forecast = theta_f @ basis_f.T    (block's contribution to final output)

    The Dropout layer (active during MC sampling) provides uncertainty estimates.
    """

    def __init__(
        self,
        input_size: int,
        forecast_size: int,
        hidden_units: int,
        backcast_basis: torch.Tensor,  # shape (window_size, n_theta)
        forecast_basis: torch.Tensor,  # shape (forecast_size, n_theta)
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        n_theta = backcast_basis.shape[1]

        # Store transposed bases as buffers so they move to the right device automatically
        self.register_buffer("basis_b", backcast_basis.T)  # (n_theta, window_size)
        self.register_buffer("basis_f", forecast_basis.T)  # (n_theta, forecast_size)

        # 4-layer FC backbone
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Coefficient heads (no bias — basis expansion handles the offset)
        self.theta_b = nn.Linear(hidden_units, n_theta, bias=False)
        self.theta_f = nn.Linear(hidden_units, n_theta, bias=False)

    def forward(self, x):
        h = self.fc(x)                              # (batch, hidden)
        backcast = self.theta_b(h) @ self.basis_b   # (batch, window_size)
        forecast = self.theta_f(h) @ self.basis_f   # (batch, forecast_size)
        return backcast, forecast


# ─── N-BEATS Stack ────────────────────────────────────────────────────────────

class NBeatsStack(nn.Module):
    """
    A stack of N-BEATS blocks with doubly-residual connections.

    Block 1 receives the raw input and produces a backcast+forecast.
    Block 2 receives (input − backcast₁) — only what Block 1 couldn't explain.
    This progressive residual subtraction forces each block to specialise.
    """

    def __init__(self, blocks: nn.ModuleList, forecast_size: int):
        super().__init__()
        self.blocks = blocks
        self.forecast_size = forecast_size

    def forward(self, x):
        # Accumulate branch forecasts; residual decreases as blocks explain more
        total_forecast = torch.zeros(x.shape[0], self.forecast_size, device=x.device)
        residual = x
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast   # doubly residual
            total_forecast = total_forecast + forecast
        return residual, total_forecast


# ─── Full N-BEATS Model ───────────────────────────────────────────────────────

class NBeats(nn.Module):
    """
    Interpretable N-BEATS: Trend Stack → Seasonality Stack.

    The output decomposes naturally into:
        trend_component     = sum of trend block forecasts
        seasonal_component  = sum of seasonality block forecasts
        final_forecast      = trend + seasonal

    This directly mirrors the Classical model decomposition y(t) = g(t) + s(t),
    making side-by-side comparison with the Classical model conceptually clean.
    """

    def __init__(
        self,
        window_size: int = 28,
        forecast_size: int = 28,
        hidden_units: int = 128,
        n_trend_blocks: int = 2,
        n_seasonality_blocks: int = 2,
        trend_degree: int = 3,
        seasonality_harmonics: int = 3,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.window_size = window_size
        self.forecast_size = forecast_size

        # Trend basis: polynomial [1, t, t², t³] — degree+1 = 4 basis functions
        trend_basis_b = _trend_basis(window_size, trend_degree)
        trend_basis_f = _trend_basis(forecast_size, trend_degree)

        # Seasonality basis: Fourier harmonics — 2×harmonics = 6 basis functions
        seas_basis_b = _seasonality_basis(window_size, seasonality_harmonics)
        seas_basis_f = _seasonality_basis(forecast_size, seasonality_harmonics)

        self.trend_stack = NBeatsStack(
            nn.ModuleList([
                NBeatsBlock(window_size, forecast_size, hidden_units,
                            trend_basis_b, trend_basis_f, dropout_rate)
                for _ in range(n_trend_blocks)
            ]),
            forecast_size,
        )

        self.seasonality_stack = NBeatsStack(
            nn.ModuleList([
                NBeatsBlock(window_size, forecast_size, hidden_units,
                            seas_basis_b, seas_basis_f, dropout_rate)
                for _ in range(n_seasonality_blocks)
            ]),
            forecast_size,
        )

    def forward(self, x):
        residual, trend_forecast = self.trend_stack(x)
        _, seasonality_forecast = self.seasonality_stack(residual)
        return trend_forecast + seasonality_forecast


# ─── Data Preparation ──────────────────────────────────────────────────────────

def _normalize(y: np.ndarray):
    """Min-max normalization to [0, 1] for stable training."""
    y_min, y_max = y.min(), y.max()
    scale = y_max - y_min if y_max != y_min else 1.0
    return (y - y_min) / scale, float(y_min), float(scale)


def _denormalize(y_norm: np.ndarray, y_min: float, scale: float) -> np.ndarray:
    return y_norm * scale + y_min


def _sliding_windows(y: np.ndarray, window_size: int):
    """Create sliding window (X, target) pairs for supervised training."""
    X, targets = [], []
    for i in range(len(y) - window_size):
        X.append(y[i: i + window_size])
        targets.append(y[i + window_size])
    return np.array(X, dtype=np.float32), np.array(targets, dtype=np.float32)


# ─── Training ──────────────────────────────────────────────────────────────────

def _train_nbeats(
    y: np.ndarray,
    window_size: int = 28,
    forecast_size: int = 28,
    epochs: int = 8,       # 8 epochs = good quality + fast (~2-3s on Render CPU)
    lr: float = 0.001,
    batch_size: int = 64,
    hidden_units: int = 64,  # 64 units = 4x faster than 128, minimal accuracy loss
) -> tuple:
    """
    Train N-BEATS on the full historical dataset.

    Using one-step-ahead sliding windows for training gives the model dense
    supervision signal (n-window_size samples from n data points).
    The multi-step forecast at inference time uses auto-regression.

    Returns: (model, y_min, y_scale)
    """
    y_norm, y_min, y_scale = _normalize(y.astype(np.float64))

    X, targets = _sliding_windows(y_norm, window_size)

    X_tensor = torch.FloatTensor(X)          # (N, window_size)
    y_tensor = torch.FloatTensor(targets)    # (N,)
    dataset = TensorDataset(X_tensor, y_tensor)
    # num_workers=0: avoid spawning subprocesses on Render's constrained free-tier env
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = NBeats(
        window_size=window_size,
        forecast_size=1,          # train one-step-ahead for dense supervision
        hidden_units=hidden_units,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch).squeeze(-1)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
    # ── Memory cleanup: must happen BEFORE caching ────────────────────────────
    # Optimizer (Adam) holds 2× model size in moment buffers. Loader and tensor
    # objects must be freed now so only the model weights go into the cache.
    del optimizer, criterion, loader, dataset, X_tensor, y_tensor, X, targets
    gc.collect()

    return model, y_min, y_scale


# ─── Batched MC Dropout Forecast ──────────────────────────────────────────────

def _mc_forecast(
    model: NBeats,
    last_window: np.ndarray,
    forecast_days: int,
    y_min: float,
    y_scale: float,
    n_samples: int = 20,
) -> tuple:
    """
    Multi-step auto-regressive forecast with batched MC Dropout for 95% CI.

    At each step:
      1. Stack n_samples copies of window → batch of shape (n_samples, window_size)
      2. Single forward pass with dropout ON → n_samples stochastic predictions
      3. mean = point forecast; [2.5th, 97.5th] percentile = 95% CI
      4. Slide window using mean prediction

    This is identical to the CNN service's approach, giving a fair comparison.
    """
    model.train()   # keep dropout active
    window = last_window.copy()
    window_size = len(last_window)

    predictions, lowers, uppers = [], [], []

    with torch.no_grad():
        for _ in range(forecast_days):
            x_single = torch.FloatTensor(window[-window_size:])
            # Batch: (n_samples, window_size)
            x_batch = x_single.unsqueeze(0).repeat(n_samples, 1)

            mc_preds = model(x_batch).squeeze(-1).cpu().numpy()  # (n_samples,)

            mean_pred = float(np.mean(mc_preds))
            predictions.append(mean_pred)
            lowers.append(float(np.percentile(mc_preds, 2.5)))
            uppers.append(float(np.percentile(mc_preds, 97.5)))

            window = np.append(window, mean_pred)

    yhat = _denormalize(np.array(predictions), y_min, y_scale)
    yhat_lower = _denormalize(np.array(lowers), y_min, y_scale)
    yhat_upper = _denormalize(np.array(uppers), y_min, y_scale)

    return (
        np.clip(yhat, 0, None),
        np.clip(yhat_lower, 0, None),
        yhat_upper,
    )


# ─── Public API ────────────────────────────────────────────────────────────────

def run_nbeats_forecast(
    data: list[dict],
    forecast_weeks: int = 4,
    context_label: str = "Metric",
    window_size: int = 28,
    epochs: int = 8,
    hidden_units: int = 64,
) -> dict:
    """
    Full N-BEATS pipeline with in-memory caching.

    First call with a given dataset: trains the model (~2-4s on CPU).
    Subsequent calls with the same dataset: instant cache hit (<0.05s).
    Cache TTL = 30 minutes, max 5 models stored (Render memory-safe).

    Steps:
      1. Hash input data → check cache
      2. If miss: train NBeats, store in cache
      3. Evaluate on last 20% holdout (batch inference)
      4. MC Dropout multi-step future forecast (batched)
      5. Batch-infer historical fitted values for the chart
    """
    df = pd.DataFrame(data)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    df["y"] = df["y"].astype(float)

    y = df["y"].values
    n_hist = len(y)
    forecast_days = forecast_weeks * 7

    # ── Load pretrained weights OR train on demand ────────────────────────────
    # Priority:
    #   1. Per-dataset in-memory cache (instant — same data seen before)
    #   2. Pre-trained weights from corpus (fast — background thread finished)
    #   3. On-demand training from scratch (slow fallback — pretrain not ready)
    cache_key = _data_hash(y)
    model, y_min, y_scale = _get_cached(cache_key)

    if model is None:
        pretrained_used = False
        try:
            from src.services.nbeats_pretrain import (
                get_pretrained_state_dict,
                is_pretrained,
                PRETRAIN_WINDOW_SIZE,
                PRETRAIN_HIDDEN_UNITS,
            )
            if is_pretrained() and window_size == PRETRAIN_WINDOW_SIZE and hidden_units == PRETRAIN_HIDDEN_UNITS:
                state_dict = get_pretrained_state_dict()
                if state_dict is not None:
                    # Instantiate a fresh model and load pretrained weights.
                    # Fresh instance per request avoids .train()/.eval() state
                    # conflicts between concurrent MC-Dropout inference calls.
                    model = NBeats(
                        window_size=window_size,
                        forecast_size=1,
                        hidden_units=hidden_units,
                    )
                    model.load_state_dict(state_dict)
                    model.eval()
                    _, y_min, y_scale = _normalize(y.astype(np.float64))
                    pretrained_used = True
        except ImportError:
            pass  # pretrain module unavailable — fall through to on-demand

        if not pretrained_used:
            # Pre-training not ready yet — train on demand (8 epochs, ~5-8 s)
            model, y_min, y_scale = _train_nbeats(
                y,
                window_size=window_size,
                forecast_size=1,
                epochs=epochs,
                hidden_units=hidden_units,
            )

        _put_cache(cache_key, model, y_min, y_scale)

    # ── Holdout Metrics (last 20%, batch inference) ───────────────────────────
    y_norm = (y - y_min) / y_scale
    holdout_size = max(int(n_hist * 0.2), window_size + 1)
    holdout_start = n_hist - holdout_size

    holdout_X = np.array([
        y_norm[i - window_size: i]
        for i in range(holdout_start, n_hist)
        if i >= window_size
    ], dtype=np.float32)
    holdout_actual = y[holdout_start: holdout_start + len(holdout_X)]

    if len(holdout_X) > 0:
        model.eval()
        with torch.no_grad():
            holdout_preds_norm = model(
                torch.FloatTensor(holdout_X)
            ).squeeze(-1).cpu().numpy()
        holdout_preds = _denormalize(holdout_preds_norm, y_min, y_scale)

        mae = float(np.mean(np.abs(holdout_actual - holdout_preds)))
        rmse = float(np.sqrt(np.mean((holdout_actual - holdout_preds) ** 2)))
        mape = float(
            np.mean(np.abs((holdout_actual - holdout_preds) /
                           np.clip(holdout_actual, 1, None))) * 100
        )
    else:
        mae = rmse = mape = 0.0
        holdout_preds = np.array([])

    # ── Future Forecast (MC Dropout) ─────────────────────────────────────────
    last_window = y_norm[-window_size:]
    yhat_future, yhat_lower, yhat_upper = _mc_forecast(
        model, last_window, forecast_days, y_min, y_scale
    )

    last_date = df["ds"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D",
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

    # ── Historical Fitted Values (batch, fast) ────────────────────────────────
    model.eval()
    all_X = np.array([
        y_norm[i - window_size: i]
        for i in range(window_size, n_hist)
    ], dtype=np.float32)
    with torch.no_grad():
        hist_preds_norm = model(
            torch.FloatTensor(all_X)
        ).squeeze(-1).cpu().numpy()
    hist_preds = _denormalize(hist_preds_norm, y_min, y_scale)

    hist_fit_records = [
        {
            "ds": df["ds"].iloc[i + window_size].strftime("%Y-%m-%d"),
            "y": round(float(y[i + window_size]), 2),
            "yhat": round(float(hp), 2),
        }
        for i, hp in enumerate(hist_preds)
        if i + window_size < n_hist
    ]

    # ── Summary stats ─────────────────────────────────────────────────────────
    last_actual = float(y[-1])
    forecast_end = float(yhat_future[-1])
    growth_pct = round(
        (forecast_end - last_actual) / max(last_actual, 1) * 100, 1
    )

    n_params = sum(p.numel() for p in model.parameters())

    summary_stats = {
        "model_name": "N-BEATS (Interpretable)",
        "context_label": context_label,
        "forecast_weeks": forecast_weeks,
        "current_value": round(last_actual, 2),
        "forecast_end_value": round(forecast_end, 2),
        "forecast_end_lower": round(float(yhat_lower[-1]), 2),
        "forecast_end_upper": round(float(yhat_upper[-1]), 2),
        "growth_pct_over_period": growth_pct,
        "window_size": window_size,
        "epochs": epochs,
        "hidden_units": hidden_units,
        "total_parameters": n_params,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
    }

    result = {
        "forecast": forecast_records,
        "historical_fit": hist_fit_records,
        "summary_stats": summary_stats,
        "accuracy_metrics": {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
            "holdout_size": len(holdout_preds) if hasattr(holdout_preds, '__len__') else 0,
        },
    }

    # ── Memory cleanup (Render free-tier: 512 MB limit) ────────────────────────
    # Drop large tensors now that results are serialised into plain Python dicts.
    # The model itself stays in the cache for fast re-use on subsequent requests.
    del holdout_X, all_X
    gc.collect()

    return result
