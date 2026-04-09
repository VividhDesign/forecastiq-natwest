# ForecastIQ — AI Predictive Forecasting

> **NatWest Code for Purpose Hackathon** · Theme: AI Predictive Forecasting

A full-stack, production-quality AI forecasting platform that transforms time-series data into actionable business insights — powered by **Facebook Prophet** (mathematical forecasting) and **Generative AI** (plain-English summaries).

---

## Overview

ForecastIQ helps teams **look ahead instead of backwards** by providing:

1. **Short-term forecasting** (1–6 weeks) with uncertainty ranges — not just a single number.
2. **Anomaly detection** — automatic flagging of unexpected spikes and drops in historical data.
3. **Scenario planning** — interactive "What-if" playground with side-by-side comparison.

All mathematical computations are performed locally using **Facebook Prophet** (a decomposable time-series model using Fourier Series for seasonality and Bayesian Inference for confidence intervals). **Generative AI (Gemini Pro or Groq/Llama-3)** is used exclusively to translate statistical outputs into concise, plain-English summaries for non-technical users.

**Intended users:** Business analysts, operations teams, and non-technical decision-makers who need forward-looking insight without data science expertise.

---

## Features

- ✅ **On-the-fly Synthetic Data Sandbox** — Generate realistic datasets (e-commerce, server load, signups) without needing real data. Customizable trend, seasonality, and anomaly injection.
- ✅ **CSV Upload** — Upload any `ds` / `y` time-series CSV for instant forecasting.
- ✅ **Short-Term Forecasting** — 1–6 week forecasts with shaded 95% confidence intervals (Bayesian posterior).
- ✅ **Anomaly Detection** — Flags historical data points outside the confidence band; highlights spikes vs drops.
- ✅ **Scenario Playground** — Growth multiplier and seasonality strength sliders; side-by-side baseline vs scenario chart.
- ✅ **Multi-LLM Insight Selector** — Switch between Google Gemini Pro and Groq (Llama-3) from the navbar. The AI model chosen generates all textual summaries.
- ✅ **Zero Hallucination on Numbers** — LLMs receive only verified numerical outputs from Prophet. They write English, not maths.
- ✅ **Premium Dark-Mode UI** — Glassmorphism design, responsive, accessible.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 18 (Vite), Recharts, Axios |
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **Forecasting** | Facebook Prophet (Decomposable TS + Fourier Seasonality + Bayesian CI) |
| **Anomaly Detection** | Statistical outlier detection using Prophet's confidence bands |
| **Data Generation** | NumPy, Pandas (Fourier Series synthetic data) |
| **Generative AI** | Google Gemini 1.5 Pro (`google-generativeai`) + Groq Llama-3 (`groq`) |
| **Deployment** | Render (backend) + Vercel (frontend) |

---

## Architecture

```
User Browser (Vercel)
       ↓ HTTPS
React Frontend (Vite)
       ↓ REST API calls
FastAPI Backend (Render)
   ├── /api/simulate  → data_simulator.py (NumPy Fourier synthetic data)
   ├── /api/upload    → Pandas CSV parser
   ├── /api/forecast  → forecasting.py (Prophet model) → llm_service.py (Gemini/Groq)
   ├── /api/anomaly-insight → llm_service.py
   └── /api/scenario  → forecasting.py (scenario variant) → llm_service.py
```

**Key design decision:** All statistical computation happens in the backend (Prophet). The LLMs are called only once per request, with a structured numerical prompt, ensuring factual accuracy and preventing hallucinated figures.

---

## Install & Run

### Prerequisites
- Python 3.11+
- Node.js 18+
- At least one API key: [Google AI Studio](https://aistudio.google.com) (Gemini) or [Groq Console](https://console.groq.com)

### Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# Start the server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`  
Auto-generated docs: `http://localhost:8000/docs`

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Set up environment
cp .env.example .env
# Edit .env: VITE_API_URL=http://localhost:8000/api

# Start dev server
npm run dev
```

The app will be available at: `http://localhost:5173`

---

## Usage Example

1. Open the app → you see the **Onboarding** screen.
2. Select **"Generate Sandbox Data"** → choose `E-commerce Sales`, `Aggressive Growth`, toggle **Inject Anomalies ON**.
3. Click **"Generate & Launch Dashboard"** → within ~2 seconds, the dashboard appears.
4. **Forecast tab:** Inspect the shaded confidence interval chart and the AI-generated 3-sentence insight.
5. **Anomalies tab:** Click any anomaly row in the table → AI generates a 2-sentence explanation.
6. **Scenario tab:** Drag the growth multiplier to `+20%` → run scenario → compare charts side-by-side.
7. Switch the AI model from `Gemini Pro` → `Groq` in the navbar. Re-fetch insights to compare.

---

## Running Tests

```bash
cd backend
# Ensure the venv is activated
pytest tests/ -v
```

---

## Mathematics Reference

ForecastIQ uses **Facebook Prophet's Decomposable Time-Series Model**:

> `y(t) = g(t) + s(t) + h(t) + ε`

- **`g(t)`** — Piecewise Linear Growth (trend modelling)
- **`s(t)`** — Fourier Series seasonality: `Σ [aₙ·cos(2πnt/P) + bₙ·sin(2πnt/P)]`
- **`h(t)`** — Holiday/event effects (optional)
- **Confidence Intervals** — Derived from Bayesian posterior sampling (MCMC)

Anomalies are defined as: `y_actual > yhat_upper` OR `y_actual < yhat_lower`

---

## Limitations

- CSV upload requires exactly two columns: `ds` (date, `YYYY-MM-DD`) and `y` (numeric). Other formats are not yet supported.
- LLM insights require a valid API key for the selected model. If not configured, a fallback message is shown.
- Forecasting accuracy degrades on datasets with fewer than 90 data points.

## Future Improvements

- Support for multi-column / multi-metric datasets.
- Export forecast data as CSV/PDF report.
- Add OpenAI GPT-4 as a third LLM option.
- Historical comparison across multiple time windows.

---

## License & Compliance

This project is submitted under **Apache License 2.0** in compliance with NatWest Code for Purpose hackathon rules and the Developer Certificate of Origin (DCO).

All data used in this project is **synthetically generated** using Python's NumPy library. No real, personal, or proprietary data is used or stored.
