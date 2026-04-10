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
from src.services.llm_service import (
    generate_forecast_insight,
    generate_anomaly_insight,
    generate_scenario_insight,
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


class AnomalyInsightRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    anomaly: dict
    context_label: str
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
    Run the full Prophet forecasting pipeline:
    - Short-term forecast with confidence intervals
    - Anomaly detection on historical data
    - AI-generated insight (via selected LLM)
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

    # Generate LLM insight for the forecast
    try:
        result["forecast_insight"] = generate_forecast_insight(
            stats=result["summary_stats"],
            model_choice=req.model_choice,
        )
    except Exception as e:
        result["forecast_insight"] = f"[AI insight unavailable: {str(e)}]"

    return result


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
