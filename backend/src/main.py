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
    Launch N-BEATS background pre-training on a synthetic corpus.

    Runs in a daemon thread — all API endpoints are immediately available.
    Pre-training takes ~40s on Render free-tier CPU; after that, N-BEATS
    model weights are held in memory for any endpoint that needs them.
    Check /pretrain-status to confirm completion.
    """
    try:
        from src.services.nbeats_pretrain import launch_pretrain_background
        launch_pretrain_background()
    except Exception as exc:
        # Non-fatal — inference will fall back to on-demand training
        print(f"[Startup] N-BEATS pretrain could not be scheduled: {exc}", flush=True)

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


@app.get("/pretrain-status", tags=["Health"])
def pretrain_status():
    """
    Reports whether the N-BEATS background pre-training has completed.
    Useful for debugging Render cold-starts and verifying the speedup is active.
    """
    try:
        from src.services.nbeats_pretrain import is_pretrained
        ready = is_pretrained()
    except Exception:
        ready = False
    return {
        "pretrained": ready,
        "message": "N-BEATS pretrained weights ready — inference will be fast (~0.5 s)."
                   if ready else
                   "Pre-training still running (~40 s after boot). First Compare Models call will train on demand.",
    }

