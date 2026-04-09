"""
AI Predictive Forecasting — NatWest Code for Purpose Hackathon
FastAPI Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router

app = FastAPI(
    title="AI Predictive Forecasting API",
    description="Forecast future trends, detect anomalies, and run scenario planning using Facebook Prophet with multi-LLM insight generation.",
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
    return {"status": "ok", "message": "AI Predictive Forecasting API is running."}
