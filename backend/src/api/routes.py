"""
API Routes
Exposes all backend endpoints for the frontend to consume.
"""

import io
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Literal, Optional, List

from src.services.data_simulator import generate_synthetic_data
from src.services.forecasting import run_forecast, run_scenario_forecast
from src.services.cnn_forecasting import run_cnn_forecast
from src.services.nbeats_forecasting import run_nbeats_forecast
from src.services.llm_service import (
    generate_forecast_insight,
    generate_anomaly_insight,
    generate_scenario_insight,
    generate_chat_insight,
    generate_comparison_insight,
)

router = APIRouter()

# ─── Request / Response Models ───────────────────────────────────────────────

class SimulateRequest(BaseModel):
    context: Literal["ecommerce_sales", "server_load", "user_signups", "support_tickets"] = "ecommerce_sales"
    trend_type: Literal["aggressive_growth", "stable", "declining"] = "aggressive_growth"
    inject_anomalies: bool = True
    days: int = Field(default=730, ge=90, le=1095)


class ForecastRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    data: List[dict]
    forecast_weeks: int = Field(default=4, ge=1, le=6)
    context_label: str = "Metric"
    model_choice: Literal["gemini", "groq"] = "gemini"


class ScenarioRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    data: List[dict]
    growth_multiplier: float = Field(default=1.1, ge=0.1, le=5.0)
    seasonality_strength: float = Field(default=1.0, ge=0.1, le=3.0)
    forecast_weeks: int = Field(default=4, ge=1, le=6)
    model_choice: Literal["gemini", "groq"] = "gemini"
    remove_outliers: bool = False


class ForecastInsightRequest(BaseModel):
    """Lightweight request to generate an AI insight from pre-computed stats."""
    model_config = {"protected_namespaces": ()}
    summary_stats: dict
    model_choice: Literal["gemini", "groq"] = "gemini"


class AnomalyInsightRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    anomaly: dict
    context_label: str
    model_choice: Literal["gemini", "groq"] = "gemini"


class ChatRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    question: str
    summary_stats: dict
    anomaly_count: int = 0
    context_label: str = "Metric"
    model_choice: Literal["gemini", "groq"] = "groq"


class CompareRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    data: List[dict]
    forecast_weeks: int = Field(default=4, ge=1, le=6)
    context_label: str = "Metric"
    model_choice: Literal["gemini", "groq"] = "gemini"


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/simulate", tags=["Data"])
def simulate_data(req: SimulateRequest):
    """
    Generate synthetic time-series data on-the-fly.
    Used by the Sandbox onboarding — no CSV needed.
    """
    result = generate_synthetic_data(
        context=req.context,
        trend_type=req.trend_type,
        inject_anomalies=req.inject_anomalies,
        days=req.days,
    )
    return result


@router.post("/upload", tags=["Data"])
async def upload_csv(file: UploadFile = File(...)):
    """
    Accept a user-uploaded CSV file with columns: ds (date), y (numeric value).
    Returns the parsed data as a list of {ds, y} dicts.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    if "ds" not in df.columns or "y" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="CSV must have columns: 'ds' (date, YYYY-MM-DD) and 'y' (numeric).",
        )

    df["ds"] = pd.to_datetime(df["ds"]).dt.strftime("%Y-%m-%d")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"])

    return {"data": df[["ds", "y"]].to_dict(orient="records"), "context_meta": None}


@router.post("/forecast", tags=["Forecasting"])
def forecast(req: ForecastRequest):
    """
    Run the math-only forecasting pipeline (fast path, no LLM):
    - OLS + Fourier decomposition
    - Vectorized bootstrap confidence intervals
    - Anomaly detection on historical data
    - Naive baseline comparison

    The AI insight is intentionally excluded here so this endpoint returns
    in ~0.3s. Call /forecast-insight separately (in parallel) for the LLM text.
    """
    if len(req.data) < 30:
        raise HTTPException(
            status_code=400, detail="At least 30 data points are required for forecasting."
        )

    result = run_forecast(
        data=req.data,
        forecast_weeks=req.forecast_weeks,
        context_label=req.context_label,
    )

    # Insight is fetched separately by the frontend via /forecast-insight
    result["forecast_insight"] = None
    return result


@router.post("/forecast-insight", tags=["Forecasting"])
def forecast_insight(req: ForecastInsightRequest):
    """
    Generate an AI insight from pre-computed forecast stats.
    Called by the frontend in parallel with /forecast so the dashboard
    renders immediately with data while the AI text loads in the background.
    """
    try:
        insight = generate_forecast_insight(
            stats=req.summary_stats,
            model_choice=req.model_choice,
        )
        return {"forecast_insight": insight}
    except Exception as e:
        return {"forecast_insight": f"[AI insight unavailable: {str(e)}]"}


@router.post("/anomaly-insight", tags=["Forecasting"])
def anomaly_insight(req: AnomalyInsightRequest):
    """
    Generate an AI explanation for a specific anomaly point.
    Called when a user clicks on a red anomaly dot on the chart.
    """
    try:
        insight = generate_anomaly_insight(
            anomaly=req.anomaly,
            context_label=req.context_label,
            model_choice=req.model_choice,
        )
        return {"insight": insight}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scenario", tags=["Forecasting"])
def scenario(req: ScenarioRequest):
    """
    Run a 'What-If' scenario forecast (e.g., +10% growth).
    Returns both baseline and modified scenario for side-by-side comparison.
    """
    if len(req.data) < 30:
        raise HTTPException(
            status_code=400, detail="At least 30 data points are required."
        )

    result = run_scenario_forecast(
        data=req.data,
        growth_multiplier=req.growth_multiplier,
        seasonality_strength=req.seasonality_strength,
        forecast_weeks=req.forecast_weeks,
        remove_outliers=req.remove_outliers,
    )

    # Generate LLM comparison insight
    try:
        result["scenario_insight"] = generate_scenario_insight(
            baseline_stats=result["baseline_stats"],
            scenario_stats=result["scenario_stats"],
            model_choice=req.model_choice,
        )
    except Exception as e:
        result["scenario_insight"] = f"[AI insight unavailable: {str(e)}]"

    return result


@router.post("/model-comparison", tags=["Forecasting"])
def model_comparison(req: CompareRequest):
    """
    Run ALL THREE models on the same dataset for a head-to-head comparison:
      1. Classical (OLS + Fourier Decomposition) — interpretable baseline
      2. 1D CNN (Deep Learning)                  — local pattern recognition
      3. N-BEATS (Interpretable Deep Learning)   — state-of-the-art decomposition

    Returns side-by-side forecasts and accuracy metrics (MAE/RMSE/MAPE on a
    20% holdout set) for all three. The winner is the model with the lowest MAE.

    Directly addresses Hackathon Learning Outcome #1:
    'When more advanced models are justified — demonstrated empirically.'
    """
    if len(req.data) < 60:
        raise HTTPException(
            status_code=400,
            detail="At least 60 data points are required for model comparison (need holdout set).",
        )

    # ── Run all three models ──────────────────────────────────────────────────
    classical_result = run_forecast(
        data=req.data,
        forecast_weeks=req.forecast_weeks,
        context_label=req.context_label,
    )

    try:
        cnn_result = run_cnn_forecast(
            data=req.data,
            forecast_weeks=req.forecast_weeks,
            context_label=req.context_label,
        )
    except Exception as e:
        cnn_result = {
            "forecast": [],
            "historical_fit": [],
            "summary_stats": {"model_name": "1D CNN", "error": str(e)},
            "accuracy_metrics": {"mae": None, "rmse": None, "mape": None, "holdout_size": 0},
        }

    try:
        nbeats_result = run_nbeats_forecast(
            data=req.data,
            forecast_weeks=req.forecast_weeks,
            context_label=req.context_label,
        )
    except Exception as e:
        nbeats_result = {
            "forecast": [],
            "historical_fit": [],
            "summary_stats": {"model_name": "N-BEATS", "error": str(e)},
            "accuracy_metrics": {"mae": None, "rmse": None, "mape": None, "holdout_size": 0},
        }

    # ── Determine winner (lowest MAE across all three) ─────────────────────────
    def _safe_mae(result):
        v = result.get("accuracy_metrics", {}).get("mae", None)
        return float(v) if v is not None else float("inf")

    maes = {
        "classical": _safe_mae(classical_result),
        "cnn":       _safe_mae(cnn_result),
        "nbeats":    _safe_mae(nbeats_result),
    }
    winner = min(maes, key=maes.get)

    comparison = {
        "classical": {
            "forecast": classical_result["forecast"],
            "summary_stats": classical_result["summary_stats"],
            "accuracy_metrics": classical_result.get("accuracy_metrics", {}),
        },
        "cnn": {
            "forecast": cnn_result["forecast"],
            "summary_stats": cnn_result["summary_stats"],
            "accuracy_metrics": cnn_result.get("accuracy_metrics", {}),
        },
        "nbeats": {
            "forecast": nbeats_result["forecast"],
            "summary_stats": nbeats_result["summary_stats"],
            "accuracy_metrics": nbeats_result.get("accuracy_metrics", {}),
        },
        "winner": winner,
        "all_maes": maes,
        "naive_baseline": classical_result.get("naive_baseline", []),
    }

    # ── AI insight (3-way comparison) ─────────────────────────────────────────
    try:
        comparison["comparison_insight"] = generate_comparison_insight(
            classical_stats=classical_result["summary_stats"],
            cnn_stats=cnn_result["summary_stats"],
            winner=winner,
            model_choice=req.model_choice,
            nbeats_stats=nbeats_result["summary_stats"],
        )
    except Exception as e:
        comparison["comparison_insight"] = f"[AI insight unavailable: {str(e)}]"

    return comparison


@router.post("/chat", tags=["Forecasting"])
def chat(req: ChatRequest):
    """
    Free-form Q&A grounded in verified forecast data.
    Users can ask any question about their data and the AI answers
    using only the mathematically verified summary stats.
    """
    try:
        answer = generate_chat_insight(
            question=req.question,
            stats=req.summary_stats,
            anomaly_count=req.anomaly_count,
            context_label=req.context_label,
            model_choice=req.model_choice,
        )
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
