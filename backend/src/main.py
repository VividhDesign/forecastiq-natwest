"""
ForecastIQ — Predictive Forecasting API
NatWest Code for Purpose Hackathon 2026
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router

app = FastAPI(
    title="ForecastIQ API",
    description="Short-term time-series forecasting with anomaly detection and AI-generated insight summaries.",
    version="1.0.0",
)

# Allow frontend (React on Vite/Vercel) to call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "ForecastIQ API"}
