<div align="center">

<img src="https://img.shields.io/badge/NatWest-Code%20for%20Purpose-d97706?style=for-the-badge" alt="NatWest Badge"/>
<img src="https://img.shields.io/badge/Theme-Predictive%20Forecasting-f59e0b?style=for-the-badge" alt="Theme"/>
<img src="https://img.shields.io/badge/Status-Live-22c55e?style=for-the-badge" alt="Status"/>

# 📊 ForecastIQ

**Turn historical data into trustworthy short-term forecasts.**

🌐 [Live App](https://forecastiq-natwest.vercel.app/)&nbsp;·&nbsp;🔌 [API Docs](https://forecastiq-natwest.onrender.com/docs)&nbsp;·&nbsp;📦 [Quick Start](#install--run)

</div>

---

## The Problem

Business teams need to plan ahead — for staffing, inventory, budgets, and capacity.
But most forecasting tools are either too technical (Python notebooks, ARIMA tuning) or too opaque (black-box predictions with no confidence range).

**ForecastIQ** provides a middle ground: a web-based tool that gives non-technical users
honest short-term forecasts with clear uncertainty ranges, anomaly detection, and what-if
scenario modelling — all in plain English.

---

## How It Works

The core idea is simple: **separate the maths from the language**.

1. A **deterministic forecasting engine** (Python, NumPy, scikit-learn) computes all numbers — trend, seasonality, confidence intervals, anomalies.
2. A **generative AI layer** (Gemini or Llama-3) receives only those verified numbers and translates them into business-friendly sentences.

The AI never invents figures. It only writes English around numbers that have already been calculated and validated. This prevents hallucinated statistics while still giving non-technical users an accessible summary.

---

## The Forecasting Model

We implement a **decomposable time-series model** from first principles — the same mathematical framework used by Facebook Prophet, but built entirely with NumPy and scikit-learn so there are no C++ compilation requirements on the server.

### Core equation

```
y(t) = g(t) + s(t) + ε
```

| Symbol | Component | Method |
|---|---|---|
| `g(t)` | **Trend** — overall direction | OLS linear regression: `y = β₀ + β₁·t` |
| `s(t)` | **Seasonality** — repeating patterns | Fourier series: `Σ [aₙ cos(2πnt/P) + bₙ sin(2πnt/P)]` |
| `ε` | **Residuals** — unexplained variance | Used for confidence interval estimation |

### Why Fourier series for seasonality

Most real-world metrics have multiple overlapping seasonal patterns (e.g. weekday vs weekend, summer vs winter). Fourier decomposition captures these as a sum of sine and cosine waves at different frequencies.

We use **order-3 harmonics for weekly patterns** (period = 7) and **order-5 harmonics for annual patterns** (period = 365.25). The higher order for annual patterns captures more complex shapes like holiday spikes.

The feature matrix for a single data point at time `t` looks like:

```
[t, cos(2π·1·t/7), sin(2π·1·t/7), cos(2π·2·t/7), sin(2π·2·t/7), ...,
    cos(2π·1·t/365.25), sin(2π·1·t/365.25), ..., cos(2π·5·t/365.25), sin(2π·5·t/365.25)]
```

This gives us 1 trend feature + 6 weekly features + 10 yearly features = **17 features** per time step, all fitted via ordinary least squares.

### Confidence intervals

For historical data, we use **bootstrap resampling** (500 iterations) of the in-sample residuals to estimate the 95% confidence band around the fitted curve.

For future forecasts, we use a **growing uncertainty model** — the confidence interval widens with each step into the future:

```
CI(step) = ŷ ± 1.96 · σ_residual · (1 + √(step / horizon))
```

This reflects the intuitive truth that predictions further into the future carry more uncertainty.

### Anomaly detection

A historical data point is flagged as anomalous if it falls outside the 95% bootstrap confidence band:

```
y > upper_band → Spike (unexpected surge)
y < lower_band → Drop  (unexpected crash)
```

Each anomaly records its direction, date, and percentage deviation from the model's expectation.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│         User's Browser (Vercel)             │
│         React + Vite + Recharts             │
│  ┌──────────┐ ┌──────────┐ ┌─────────────┐ │
│  │Onboarding│ │Dashboard │ │ AI Model    │ │
│  │ Sandbox /│ │Forecast /│ │ Selector    │ │
│  │CSV Upload│ │Anomalies/│ │ Gemini ↔    │ │
│  └──────────┘ │Scenarios │ │ Groq        │ │
│               └──────────┘ └─────────────┘ │
└────────────────────┬────────────────────────┘
                     │ HTTPS (REST API)
┌────────────────────▼────────────────────────┐
│         FastAPI Backend (Render)             │
│                                             │
│  POST /api/simulate                         │
│    → data_simulator.py                      │
│      Linear trend + Fourier seasonality     │
│      + Gaussian noise + anomaly injection   │
│                                             │
│  POST /api/upload                           │
│    → CSV parser (Pandas)                    │
│                                             │
│  POST /api/forecast                         │
│    → forecasting.py                         │
│      OLS trend + Fourier seasonality        │
│      + Bootstrap CI + anomaly detection     │
│    → llm_service.py (verified stats only)   │
│                                             │
│  POST /api/anomaly-insight                  │
│    → llm_service.py (single anomaly)        │
│                                             │
│  POST /api/scenario                         │
│    → scenario forecast (baseline vs what-if)│
│    → llm_service.py (comparison summary)    │
│                                             │
│  LLM Router (model-agnostic):               │
│    "gemini" → Gemini 2.0 Flash              │
│    "groq"   → Llama-3.3-70B via Groq       │
└─────────────────────────────────────────────┘
```

### Key design decision

The LLM receives a **fully numerical prompt** — every figure in the prompt comes from the deterministic engine. The LLM's only job is to convert those numbers into readable sentences. This means:

- Numbers are always correct (computed, never generated)
- Switching between Gemini and Llama-3 changes only the writing style, not the data
- The system remains trustworthy even if the LLM occasionally misinterprets a figure

---

## Features

| Feature | Description |
|---|---|
| **Data Simulator** | Generate realistic synthetic datasets (4 business contexts, 3 trend types, optional anomaly injection) on-the-fly — no real data required |
| **CSV Upload** | Upload any time-series CSV with `ds` (date) and `y` (numeric) columns |
| **Short-Term Forecast** | 1–6 week forecasts with shaded 95% confidence intervals |
| **Anomaly Detection** | Flags historical data points outside the bootstrap CI band, classified as spikes or drops |
| **Scenario Playground** | Interactive sliders for growth and seasonality multipliers with side-by-side baseline vs scenario comparison |
| **Raw Data Explorer** | Searchable, sortable table with anomaly highlights, % deviation, and one-click CSV export |
| **Multi-LLM Insights** | Switch between Gemini 2.0 Flash and Groq (Llama-3.3-70B) live — the AI writes, the maths calculates |
| **Dark / Light Mode** | Theme toggle with localStorage persistence |

---

## Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| **Frontend** | React 18, Vite | Fast HMR, lightweight production bundles |
| **Charts** | Recharts | Supports shaded confidence bands and interactive tooltips |
| **Styling** | Vanilla CSS (DM Sans + DM Mono) | Full control, no framework lock-in |
| **Backend** | Python 3.11, FastAPI | Async, auto-generated OpenAPI docs |
| **Forecasting** | NumPy, scikit-learn, statsmodels | Pure Python — no C++ compilation required at deploy time |
| **Data Generation** | NumPy (Fourier series) | Realistic synthetic data, zero privacy risk |
| **AI — Gemini** | `google-generativeai` SDK | gemini-2.0-flash: fast, free-tier |
| **AI — Groq** | `groq` SDK | Llama-3.3-70B at ultra-low latency |
| **Deployment** | Render (backend) + Vercel (frontend) | Free-tier, CI/CD from GitHub |

---

## Project Structure

```
ForecastIQ/
├── README.md
├── render.yaml                         # Render deployment config
├── .gitignore
│
├── backend/
│   ├── .python-version                 # Pins Python 3.11.9 for Render
│   ├── requirements.txt
│   ├── .env.example
│   └── src/
│       ├── main.py                     # FastAPI entry point + CORS
│       ├── api/
│       │   └── routes.py               # REST endpoint definitions
│       └── services/
│           ├── data_simulator.py       # Synthetic data (Fourier + noise)
│           ├── forecasting.py          # Core model (OLS + Bootstrap CI)
│           └── llm_service.py          # LLM router (Gemini / Groq)
│   └── tests/
│       └── test_forecasting.py         # 11 test assertions
│
└── frontend/
    ├── index.html
    ├── package.json
    ├── .env.production                 # Production backend URL
    └── src/
        ├── main.jsx
        ├── App.jsx                     # Root: theme + routing state
        ├── index.css                   # Design system (dark + light)
        └── components/
            ├── Onboarding/             # Data input: sandbox or CSV
            ├── Dashboard/              # Main analytics layout
            ├── Charts/
            │   ├── ForecastChart.jsx    # Recharts ComposedChart + CI band
            │   ├── AnomalyPanel.jsx     # Anomaly table + AI explanations
            │   ├── ScenarioPlayground.jsx # What-if sliders + comparison
            │   └── DataExplorer.jsx     # Filterable data table + CSV export
            └── Shared/
                ├── NavBar.jsx           # Model selector + theme toggle
                └── InsightCard.jsx      # AI-generated insight display
```

---

## Install & Run

### Prerequisites

- Python 3.11+
- Node.js 18+
- At least one free API key:
  - [Google AI Studio (Gemini)](https://aistudio.google.com/apikey)
  - [Groq Console (Llama-3)](https://console.groq.com/keys)

### Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Edit .env — add your GEMINI_API_KEY and/or GROQ_API_KEY

uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

API at `http://localhost:8000` · Docs at `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
cp .env.example .env
# Edit .env → VITE_API_URL=http://localhost:8000/api

npm run dev
```

App at `http://localhost:5173`

---

## Usage

### Quick demo (no data required)

1. Open the app → **Onboarding** screen
2. Select a business context (e.g. E-commerce Sales)
3. Choose a trend direction and toggle anomaly injection
4. Click **Launch Dashboard →**

### On the dashboard

| Tab | What it shows |
|---|---|
| **Forecast** | Line chart with 95% confidence band + AI insight card |
| **Anomalies** | Detected spikes and drops — click any row for an AI explanation |
| **Scenario** | Growth & seasonality sliders — compare baseline vs what-if |
| **Raw Data** | Full data table with search, sort, anomaly highlights, and CSV export |

Switch AI models from the navbar. Toggle dark/light mode with the ☀/☽ button.

### CSV upload format

```csv
ds,y
2024-01-01,1200.5
2024-01-02,1345.2
```

Minimum 30 rows. Dates in YYYY-MM-DD format.

---

## Tests

```bash
cd backend
pytest tests/ -v
```

11 test assertions covering data generation, forecast structure, anomaly detection, scenario comparison, and summary statistics shape.

---

## Environment Variables

### Backend (`backend/.env`)

```
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

### Frontend (`frontend/.env`)

```
VITE_API_URL=http://localhost:8000/api
```

> `.env` files are gitignored. Only `.env.example` and `.env.production` are tracked.

---

## Deployment

> **Live now:**
> - Frontend: [forecastiq-natwest.vercel.app](https://forecastiq-natwest.vercel.app/)
> - Backend API: [forecastiq-natwest.onrender.com](https://forecastiq-natwest.onrender.com)
> - API Docs: [forecastiq-natwest.onrender.com/docs](https://forecastiq-natwest.onrender.com/docs)

### Backend → Render

This repo includes `render.yaml` and `backend/.python-version` for zero-config deployment.

1. Create a Web Service on [render.com](https://render.com) and connect this repository
2. Set `GEMINI_API_KEY` and `GROQ_API_KEY` in the Render environment
3. Deploy — build and start commands are pre-configured

> Free-tier services sleep after 15 min of inactivity. First request after sleep takes ~30s.

### Frontend → Vercel

1. Import this repo on [vercel.com](https://vercel.com)
2. Set root directory to `frontend`, framework preset to Vite
3. Add `VITE_API_URL=https://forecastiq-natwest.onrender.com/api`
4. Deploy

---

## Known Limitations

- CSV upload supports only two columns: `ds` (date) and `y` (numeric)
- Forecasting accuracy degrades below 90 data points
- LLM insights require a valid API key — a fallback message is shown if missing
- Bootstrap CI runs 500 iterations — may take 1–2 seconds on large datasets

---

## License & Compliance

Submitted under the **Apache License 2.0** in compliance with NatWest Code for Purpose hackathon rules and DCO requirements. All commits are signed off with a single email identity.

All data used in this project is **synthetically generated** using NumPy. No real, personal, or proprietary data is used or stored.

---

<div align="center">

**Built for NatWest Code for Purpose India Hackathon 2026**

*Making data-driven decisions accessible — not just to data scientists.*

</div>
