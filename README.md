<div align="center">

<img src="https://img.shields.io/badge/NatWest-Code%20for%20Purpose-6366f1?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTV6TTIgMTdsOCA0IDgtNFY3bC04IDQtOC00eiIvPjwvc3ZnPg==" alt="NatWest Badge"/>
<img src="https://img.shields.io/badge/Theme-AI%20Predictive%20Forecasting-818cf8?style=for-the-badge" alt="Theme"/>
<img src="https://img.shields.io/badge/Status-Live%20Deployed-10b981?style=for-the-badge" alt="Status"/>
<img src="https://img.shields.io/badge/Frontend-Vercel-000000?style=for-the-badge&logo=vercel" alt="Vercel"/>
<img src="https://img.shields.io/badge/Backend-Render-46E3B7?style=for-the-badge&logo=render" alt="Render"/>

# 🔮 ForecastIQ — AI Predictive Forecasting

**Transform historical data into trustworthy, actionable forecasts.**  
Built for the **NatWest Code for Purpose India Hackathon 2026**.

🌐 **[Live App → forecastiq-natwest.vercel.app](https://forecastiq-natwest.vercel.app/)** &nbsp;|&nbsp; 🔌 **[API Docs → forecastiq-natwest.onrender.com/docs](https://forecastiq-natwest.onrender.com/docs)**

[📐 Architecture](#architecture) · [🧮 Mathematics](#mathematics-reference) · [📦 Setup](#install--run) · [☁️ Deployment](#deployment)

</div>

---

## 📌 Overview

Many teams today rely only on past data and lack accessible, trustworthy forecasting tools. **ForecastIQ** solves this by helping users:

- **Look ahead instead of backwards** — short-term forecasts for 1–6 weeks
- **Get honest signals, not overconfident predictions** — every forecast shows a range (low, likely, high)
- **Understand uncertainty and take early action** — anomaly detection flags sudden changes before they become crises
- **Test what-if scenarios** — interactive playground to compare "what happens if growth increases by 10%?"

All mathematical computations are performed **locally** using a **Decomposable Time-Series Model** (Fourier Series seasonality + OLS trend + Bootstrap CI). **Generative AI (Gemini 2.0 Flash or Groq Llama-3)** is used exclusively to translate statistical outputs into plain-English summaries — ensuring **zero hallucination on numbers**.

**Intended users:** Business analysts, operations teams, and non-technical decision-makers who need forward-looking insight without data science expertise.

---

## ✨ Features

| Feature | Description |
|---|---|
| ✅ **On-the-fly Data Simulator** | Generate realistic synthetic datasets (e-commerce, server load, signups, support tickets) with customizable trend, seasonality & injected anomalies — no real data needed |
| ✅ **CSV Upload** | Upload any time-series CSV (`ds`, `y` columns) for instant analysis |
| ✅ **Short-Term Forecasting** | 1–6 week forecasts with shaded 95% confidence intervals, showing a range (low, likely, high) |
| ✅ **Anomaly Detection** | Automatically flags historical data points outside the 95% confidence band — highlights spikes vs drops |
| ✅ **Scenario Playground** | Interactive sliders for growth multiplier & seasonality strength — side-by-side baseline vs scenario comparison chart |
| ✅ **📊 Raw Data Explorer** | Full searchable, filterable, paginated data table with anomaly highlights, % deviation, change-from-previous column and one-click CSV export |
| ✅ **Multi-LLM Insight Selector** | Switch between **Google Gemini 2.0 Flash** and **Groq (Llama-3.3-70B)** live from the navbar — AI writes the explanation, maths does the calculation |
| ✅ **Key Trend Patterns** | AI highlights trend, seasonality, and peak periods in plain language |
| ✅ **Zero Hallucination on Numbers** | LLMs only receive verified numerical outputs from the model; they write English, not maths |
| ✅ **Premium Dark-Mode UI** | Glassmorphism design, responsive, fully interactive Recharts dashboard |

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Frontend** | React 18 + Vite | Fast HMR, lightweight modern SPA |
| **UI** | Recharts + Vanilla CSS | Shaded confidence bands, interactive tooltips |
| **Backend** | Python 3.11 + FastAPI | Async, OpenAPI docs auto-generated |
| **Forecasting** | NumPy + scikit-learn + statsmodels | Pure Python — no C++ build required |
| **Data Generation** | NumPy (Fourier Series) | Realistic synthetic data, zero privacy risk |
| **AI — Gemini** | `google-generativeai` (Gemini 2.0 Flash) | Fast, free-tier, precise insight generation |
| **AI — Groq** | `groq` (Llama-3.3-70B) | Ultra-fast inference, open-source model |
| **Deployment** | Render (backend) + Vercel (frontend) | Free-tier cloud, CI/CD ready |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│           User Browser (Vercel)             │
│           React + Vite Frontend             │
│  ┌──────────┐ ┌──────────┐ ┌─────────────┐ │
│  │Onboarding│ │Dashboard │ │ LLM Switcher│ │
│  │ Sandbox /│ │Forecast /│ │ Gemini ↔   │ │
│  │CSV Upload│ │Anomalies/│ │ Groq        │ │
│  └──────────┘ │Scenarios │ └─────────────┘ │
│               └──────────┘                 │
└────────────────────┬────────────────────────┘
                     │ HTTPS REST API
┌────────────────────▼────────────────────────┐
│         FastAPI Backend (Render)            │
│                                             │
│  /api/simulate  → data_simulator.py         │
│    NumPy: trend + Fourier seasonality       │
│    + Gaussian noise + anomaly injection     │
│                                             │
│  /api/upload    → CSV parser (Pandas)       │
│                                             │
│  /api/forecast  → forecasting.py           │
│    y(t) = g(t) + s(t) + ε                  │
│    g(t) = OLS Linear Regression (trend)     │
│    s(t) = Fourier Series (seasonality)      │
│    CI   = Bootstrap Resampling (95%)        │
│    Anomalies = |y| outside CI band          │
│    → llm_service.py (insight generation)    │
│                                             │
│  /api/anomaly-insight → llm_service.py      │
│  /api/scenario  → scenario forecast         │
│                 → llm_service.py            │
│                                             │
│  llm_service.py (Model-Agnostic Router)     │
│    "gemini" → Google Gemini 2.0 Flash API   │
│    "groq"   → Groq Llama-3.3-70B API        │
└─────────────────────────────────────────────┘
```

**Key design decision:** All statistics come from the backend model. LLMs are called once per request with a fully numerical prompt, preventing hallucinated figures. The result is a **trustworthy, fast, and transparent** system.

---

## 🧮 Mathematics Reference

ForecastIQ implements a **Decomposable Time-Series Model** — the same mathematical foundation used by Facebook Prophet:

### The Core Equation

```
y(t) = g(t) + s(t) + ε
```

| Component | Name | Formula | Purpose |
|---|---|---|---|
| `g(t)` | **Trend** | `β₀ + β₁·t` (OLS Linear Regression) | Captures the overall direction (growth, decline, flat) |
| `s(t)` | **Seasonality** | `Σ [aₙ·cos(2πnt/P) + bₙ·sin(2πnt/P)]` | Captures repeating weekly and yearly patterns using Fourier harmonics |
| `ε` | **Noise** | Gaussian residuals | Random variance; used to estimate confidence intervals |

### Confidence Intervals (Uncertainty Band)

```
CI = Bootstrap Resampling of residuals (n=500 iterations, 95% α-level)
     Future CI widens: ± 1.96 · σ · √(step/horizon)
```

This gives the **low, likely, high** range shown on every forecast chart. Anomalies are flagged when:

```
y_actual > yhat_upper  →  SPIKE (unexpected surge)
y_actual < yhat_lower  →  DROP  (unexpected crash)
```

### Synthetic Data Generation

The Data Simulator also uses Fourier Series to create realistic fake datasets:

```python
y(t) = base + slope·t              # trend
     + Σ [aₙ·cos(2πnt/7)]          # weekly seasonality (Fourier order 3)
     + Σ [bₙ·cos(2πnt/365.25)]     # yearly seasonality (Fourier order 5)
     + N(0, σ)                     # Gaussian noise
     + spike_events                # 4 deliberate anomaly injections
```

---

## 📂 Project Structure

```
ForecastIQ/
├── README.md                          # This file
├── .gitignore
│
├── backend/                           # Python FastAPI service
│   ├── requirements.txt               # pip dependencies
│   ├── .env.example                   # Environment variable template
│   └── src/
│       ├── main.py                    # FastAPI app entry point + CORS
│       ├── api/
│       │   └── routes.py              # All REST endpoints
│       └── services/
│           ├── data_simulator.py      # Synthetic data generation (Fourier + NumPy)
│           ├── forecasting.py         # Core time-series engine (OLS + Bootstrap CI)
│           └── llm_service.py         # Multi-LLM router (Gemini + Groq)
│   └── tests/
│       └── test_forecasting.py        # Unit tests (11 assertions)
│
└── frontend/                          # React + Vite SPA
    ├── index.html                     # SEO-optimised entry point
    ├── package.json
    ├── .env.example                   # Frontend env template
    └── src/
        ├── main.jsx                   # React entry point
        ├── App.jsx                    # Root component (Onboarding → Dashboard)
        ├── index.css                  # Premium dark-mode design system
        └── components/
            ├── Onboarding/            # Welcome screen (Sandbox + Upload)
            ├── Dashboard/             # Main analytics page
            ├── Charts/
            │   ├── ForecastChart.jsx  # Recharts ComposedChart + shaded CI
            │   ├── AnomalyPanel.jsx   # Clickable anomaly table + AI explanations
            │   └── ScenarioPlayground.jsx # What-if sliders + comparison chart
            └── Shared/
                ├── NavBar.jsx         # LLM selector dropdown
                └── InsightCard.jsx    # AI-generated insight cards
```

---

## 🚀 Install & Run

### Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- At least one API key (free):
  - [Google AI Studio → Gemini](https://aistudio.google.com/apikey)
  - [Groq Console](https://console.groq.com/keys)

---

### Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — add your GEMINI_API_KEY and/or GROQ_API_KEY

# Start the API server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

✅ API runs at: `http://localhost:8000`  
📚 Auto-docs at: `http://localhost:8000/docs`

---

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.example .env
# Edit .env → VITE_API_URL=http://localhost:8000/api

# Start dev server
npm run dev
```

✅ App runs at: `http://localhost:5173`

---

## 📋 Usage Example

### Using the Sandbox (Recommended for demo)

1. Open the app → **Onboarding** screen appears
2. Select **"✨ Generate Sandbox Data"** tab
3. Choose:
   - Context: `🛒 E-commerce Sales`
   - Trend: `📈 Aggressive Growth`
   - Toggle **Inject Anomalies → ON**
4. Click **"🚀 Generate & Launch Dashboard"**

### On the Dashboard

| Tab | What you see |
|---|---|
| **📈 Forecast** | Line chart with shaded 95% confidence band + AI insight card |
| **🚨 Anomalies** | Table of detected spikes/drops — click any row for AI explanation |
| **🎰 Scenario** | Drag sliders for growth & seasonality — compare baseline vs scenario |

**Switch AI models** anytime using the navbar dropdown:  
`✨ Google Gemini 2.0 Flash` ↔ `⚡ Groq / Llama-3.3-70B`

### Using CSV Upload

Your CSV must have exactly two columns:

```csv
ds,y
2024-01-01,1200.5
2024-01-02,1345.2
...
```

Minimum 30 rows required for model fitting.

---

## 🧪 Running Tests

```bash
cd backend
# Activate venv first
pytest tests/ -v
```

**11 test assertions** covering:
- Data simulator output shape and value correctness
- Anomaly injection and detection
- Forecast output structure
- Scenario comparison correctness
- Summary statistics completeness

---

## ⚙️ Environment Variables

### Backend (`backend/.env`)

```env
# Required: At least one must be set
GEMINI_API_KEY=your_google_ai_studio_key_here
GROQ_API_KEY=your_groq_console_key_here
```

### Frontend (`frontend/.env`)

```env
# URL of your running backend
VITE_API_URL=http://localhost:8000/api
```

> ⚠️ Never commit `.env` files. Only `.env.example` is tracked in git.

---

## ☁️ Deployment

> **Live deployments:**
> - 🌐 Frontend: [forecastiq-natwest.vercel.app](https://forecastiq-natwest.vercel.app/)
> - 🔌 Backend API: [forecastiq-natwest.onrender.com](https://forecastiq-natwest.onrender.com)
> - 📚 API Docs: [forecastiq-natwest.onrender.com/docs](https://forecastiq-natwest.onrender.com/docs)

### Backend → Render

This repo includes a `render.yaml` file for zero-config deployment.

1. Create a new **Web Service** on [render.com](https://render.com)
2. Connect this GitHub repository — Render auto-detects `render.yaml`
3. Set environment variables in the Render dashboard:
   - `GEMINI_API_KEY` — from [Google AI Studio](https://aistudio.google.com/apikey)
   - `GROQ_API_KEY` — from [Groq Console](https://console.groq.com/keys)
4. Deploy — the start command `uvicorn src.main:app --host 0.0.0.0 --port $PORT` is pre-configured

> ⚠️ **Free tier note:** Render free services spin down after 15 minutes of inactivity. The first request after sleep takes ~30 seconds to wake up. This is expected behaviour.

### Frontend → Vercel

1. Import this repo on [vercel.com](https://vercel.com)
2. Set:
   - **Root directory:** `frontend`
   - **Framework Preset:** Vite
3. Add environment variable:
   ```
   VITE_API_URL=https://forecastiq-natwest.onrender.com/api
   ```
4. Deploy — Vercel auto-builds with `npm run build`

---

## ⚠️ Known Limitations

- CSV upload supports only `ds` (date, `YYYY-MM-DD`) and `y` (numeric) columns
- Forecasting accuracy degrades with fewer than 90 data points
- LLM insights require a valid API key; a fallback message is shown if unconfigured
- The bootstrap CI computation runs `n=500` iterations — may take ~1–2s on large datasets

## 🔮 Future Improvements

- Multi-metric dashboard (compare multiple `y` columns simultaneously)
- Export forecast as PDF / CSV report
- Add OpenAI GPT-4o as a third LLM option
- Real-time streaming for LLM insight generation
- User-defined holiday events for improved seasonality modelling

---

## 📜 License & Open Source Compliance

This project is submitted under the **Apache License 2.0** in compliance with NatWest Code for Purpose hackathon rules and the **Developer Certificate of Origin (DCO)** requirements.

All commits are signed off (`git commit -s`) using a single email identity as required.

All data used in this project is **synthetically generated** using Python's NumPy library. No real, personal, or proprietary data is used or stored at any point.

---

<div align="center">

**Built with ❤️ for NatWest Code for Purpose India Hackathon 2026**

*Making data-driven decisions accessible to everyone — not just data scientists.*

</div>
