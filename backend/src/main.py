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
async def startup_event():
    """
    Fire N-BEATS pre-training in a background daemon thread the moment
    Render boots the service.

    Why background thread (not async task):
        PyTorch training is CPU-bound and blocks the event loop if awaited.
        A daemon thread keeps the FastAPI event loop free so /ping, /simulate,
        and /forecast are all instantly available while pre-training runs.

    Timeline:
        t=0 s  : Server boots, /ping available immediately
        t=0 s  : Pre-training thread starts (60 synthetic series × 5 epochs)
        t=~40 s: Pre-training finishes, weights stored in memory
        t=??   : User opens app, warm-up ping resolves, they pick settings
        t=??+  : User clicks "Compare Models" — pretrained weights ready,
                 N-BEATS inference completes in ~0.5 s instead of ~8 s
    """
    from src.services.nbeats_pretrain import launch_pretrain_background
    launch_pretrain_background()


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

