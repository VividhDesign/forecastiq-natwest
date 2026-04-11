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


@app.on_event("startup")
def startup_event():
    """
    Server startup hook.
    N-BEATS pre-training removed — the /model-comparison endpoint now compares
    Classical (OLS+Fourier) vs the 28-day Naive Baseline, both computed analytically
    with no training step. This keeps cold-start time under 3 seconds.
    """
    print("[Startup] ForecastIQ API ready.", flush=True)

@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "ForecastIQ API"}


@app.get("/ping", tags=["Health"])
def ping():
    """
    Ultra-lightweight warm-up endpoint.
    The frontend pings this on page load to wake the Render free-tier instance
    from sleep before the user clicks 'Launch Dashboard', eliminating the
    cold-start delay (~30-60 s) from the user-visible critical path.
    """
    return {"pong": True}


