<div align="center">

<img src="https://img.shields.io/badge/NatWest-Code%20for%20Purpose%202026-003D7B?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2Zy8+" alt="NatWest"/>
<img src="https://img.shields.io/badge/Status-Live%20%26%20Deployed-22c55e?style=for-the-badge" alt="Live"/>
<img src="https://img.shields.io/badge/Theme-AI%20Predictive%20Forecasting-f59e0b?style=for-the-badge" alt="Theme"/>
<img src="https://img.shields.io/badge/License-Apache%202.0-6366f1?style=for-the-badge" alt="License"/>

# 📊 ForecastIQ

### *Predictive Analytics for Everyone — Not Just Data Scientists*

> **Turn any time-series into trustworthy short-term forecasts with confidence intervals, anomaly detection, what-if scenarios, and AI summaries in plain English.**

<br/>

🌐 **[Live Demo](https://forecastiq-natwest.vercel.app/)** &nbsp;·&nbsp; 🔌 **[API Docs](https://forecastiq-natwest.onrender.com/docs)** &nbsp;·&nbsp; ⚡ **[Quick Start (3 commands)](#-quick-start)**

</div>

---

## The Problem We're Solving

Business teams — procurement, operations, finance — need to plan weeks ahead. They're often stuck between two bad options:

- **Too technical:** Python notebooks, ARIMA tuning, manual feature engineering — inaccessible to non-experts
- **Too simplistic:** Spreadsheet averages with no uncertainty range, no anomaly flags, no scenario modelling

**ForecastIQ** sits exactly in the middle: a web tool that gives non-technical users **honest, mathematically rigorous forecasts** with clear uncertainty ranges, anomaly detection, what-if scenario modelling, and a conversational AI layer — all without writing a single line of code.

---

## ✅ How We Meet Every Problem Statement Requirement

| Requirement | Our Implementation | Where to See It |
|---|---|---|
| *Predict likely values for future periods* | OLS + Fourier decomposition — 1 to 6 week forecasts | **📈 Forecast tab** |
| *Show a range of outcomes, not just a single number* | Shaded 95% bootstrap confidence band that widens over time | **Forecast chart** |
| *Compare against a simple baseline to avoid over-fitting* | 28-day rolling mean baseline plotted on every chart | **Pattern Breakdown card** |
| *Detect early warning signs — sudden changes* | Anomalies flagged where actuals break the 95% CI band | **🚨 Anomalies tab** |
| *Explain findings in non-technical language* | AI-generated 2–3 sentence summaries for every forecast | **Insight cards** |
| *Suggest concrete next steps* | Each anomaly gives a one-sentence recommended action | **Anomaly detail view** |
| *Highlight key patterns — trend, seasonality* | Decomposition card: trend slope %, seasonal amplitude | **Pattern Breakdown card** |
| *Let users adjust growth rate* | Growth multiplier slider (0.1× to 5.0×) | **🎰 Scenario tab** |
| *Remove recent outliers scenario* | Winsorization toggle that clips top/bottom 5% before fitting | **Scenario → Remove Outliers** |
| *Apply flat or seasonal patterns* | Seasonality strength slider (0.1× to 3.0×) | **Scenario tab** |
| *Side-by-side scenario comparisons* | Baseline vs scenario on same chart + AI summary of difference | **Scenario tab** |
| *Keep experience lightweight and trustworthy* | Pure Python, no C++ deps, deterministic math + grounded AI | **Architecture** |

---

## 🎬 Two-Minute Demo Path (for judges)

> **No setup needed — the live app is at [forecastiq-natwest.vercel.app](https://forecastiq-natwest.vercel.app/)**

1. **Open the app** → choose **🛒 E-commerce Sales** + **📈 Aggressive Growth** → click **Launch Dashboard →**
2. On the **📈 Forecast** tab — see the 95% confidence band, naive baseline, and AI-generated insight
3. Click the **🚨 Anomalies** tab → click any red dot for a 3-sentence AI explanation + recommended action
4. Switch to **🎰 Scenario** → drag the growth slider to 1.25× and hit **Run Scenario**
5. Go to **📈 Live Stock** on the home screen → load **NWG.L (NatWest Group)** for real market data
6. Click **💬 Ask** → type *"When is the peak and how confident are you?"* — grounded AI answers live

---

## 🏗️ How It Works

```
User Browser (Vercel CDN)
│
│  Step 1 — Pick data source:
│    🧪 Sample Data  →  /api/simulate   (seed=42, cached, instant)
│    📈 Live Stock   →  /api/fetch-stock (yfinance → Yahoo Finance)
│    📄 Upload CSV   →  /api/upload      (any ds/y CSV, ≥30 rows)
│
│  Step 2 — Dashboard loads in parallel:
│    /api/forecast         →  Math results in ~0.3 s
│    /api/forecast-insight →  LLM insight fires in parallel (non-blocking)
│
└───────────────────────────────────────────────────────────►

FastAPI Backend (Render)
│
│  Deterministic Forecasting Engine (NumPy + scikit-learn)
│    y(t) = g(t)  +  s(t)  +  ε
│            │        │
│            │        └─ Fourier Series seasonality
│            │              Σ [aₙ cos(2πnt/P) + bₙ sin(2πnt/P)]
│            │              Weekly:  period=7,  order=3
│            │              Yearly:  period=365.25, order=5
│            └─ OLS linear trend:  y = β₀ + β₁·t
│
│  Confidence Intervals
│    Historical: 500-iteration vectorized bootstrap resampling
│    Future CI:  grows as ŷ ± 1.96·σ·(1 + √(step/horizon))
│
│  Anomaly Detection
│    Spike: y_actual > upper_band  →  flagged with pct_deviation
│    Drop:  y_actual < lower_band  →  flagged with pct_deviation
│
│  AI Layer (grounded — zero hallucination by design)
│    LLM receives:  fully pre-computed stats dict (no raw data)
│    LLM produces:  English sentences around verified numbers only
│    Primary:  Groq (Llama-3.3-70B) — no daily limits
│    Fallback: Gemini 2.0 Flash — auto-switches on 429
│
│  Model Comparison (Compare Models tab)
│    Classical (OLS+Fourier) vs 28-day Naive Baseline
│    Evaluated on 20% holdout: MAE / RMSE / MAPE
│    Winner declared live — purely analytical, <1 s
```

---

## 🔬 The Forecasting Model — From First Principles

We implement a **decomposable time-series model** identical in concept to Facebook Prophet — but built entirely with NumPy and scikit-learn so there are no C++ compilation requirements anywhere in the stack.

### Core Decomposition

```
y(t) = g(t)  +  s(t)  +  ε
```

| Symbol | Component | Method |
|---|---|---|
| `g(t)` | **Trend** | OLS linear regression: `y = β₀ + β₁·t` |
| `s(t)` | **Seasonality** | Fourier: `Σ [ aₙ cos(2πnt/P) + bₙ sin(2πnt/P) ]` |
| `ε` | **Residuals** | Used for bootstrap CI estimation |

The feature matrix per time step (17 features total):

```
[t,  cos(2π·1·t/7),   sin(2π·1·t/7),   ...  cos(2π·3·t/7),   sin(2π·3·t/7),   ← weekly (6)
     cos(2π·1·t/365), sin(2π·1·t/365), ...  cos(2π·5·t/365), sin(2π·5·t/365)]  ← yearly (10)
```

All 17 features fit in a single `LinearRegression().fit()` — transparent, interpretable, fast.

### Why Fourier Over Moving Averages / Smoothing

| Method | Limitation | Our Upgrade |
|---|---|---|
| Moving average | Repeats past — can't model trend direction | `β₁·t` OLS term models growth/decline explicitly |
| Exponential smoothing | Single-frequency only | Fourier captures weekly **and** yearly patterns simultaneously |
| ARIMA | Requires manual (p,d,q) tuning | Auto-fits via OLS, zero hyperparameters |

### Naive Baseline

Every forecast runs a **28-day rolling mean** in parallel — the simplest possible "prediction". The chart plots it as a dotted grey line. The **Pattern Breakdown card** shows `model_vs_naive_%` — the empirical answer to *"is this model actually learning anything?"*

```
ŷ_naive(t) = mean(y[n-28 : n])   for all future t
```

### Growing Confidence Intervals

```
CI(step) = ŷ ± 1.96 · σ_residual · (1 + √(step / horizon))
```

Widens automatically to show that *next week is more predictable than next month* — making uncertainty intuitive without statistics jargon.

---

## 📊 Model Comparison — Classical vs Naive Baseline

The **Compare Models** tab runs a head-to-head comparison that directly addresses **Hackathon Learning Outcome #2: "Why simple comparisons matter when testing ideas."**

```
Classical (OLS + Fourier)        vs        Naive Baseline (28-day avg)
──────────────────────────────            ────────────────────────────
y(t) = β₀ + β₁·t                          ŷ(t) = mean(y[n-28 : n])
     + Σ Fourier seasonality               flat line for all future t
     + 95% bootstrap CI band
```

**How evaluation works:**
- Both models are evaluated on the same **20% holdout set** (held back from model fitting)
- Metrics computed: **MAE**, **RMSE**, **MAPE**
- Winner = lowest MAE — declared live in the UI with a trophy badge

**What this shows judges:**
- If our Classical model clearly beats the flat rolling average → it's genuinely learning trend + seasonal patterns
- If the naive baseline is competitive → honest signal that the data may be too noisy for meaningful forecasting
- Either outcome is educational: the tool shows the truth, not just what looks impressive

**Performance:** Both models are purely analytical — no training, no PyTorch, no GPU. The entire comparison runs in **< 1 second** after data is loaded.

---

## ⚡ Speed Engineering — Why the Dashboard is Fast

We spent significant effort making every user interaction feel immediate:

| Optimization | Result |
|---|---|
| **`seed=42` deterministic caching** | Sandbox `/simulate` cached after 1st call — sub-millisecond repeated reads |
| **Parallel forecast + LLM** | `/api/forecast` returns math (~0.3 s), LLM insight fires in parallel (non-blocking) |
| **Vectorized bootstrap CI** | 500-iteration CI computed in <50 ms — single `rng.choice(size=(500,n))` matrix op |
| **Classical vs Naive comparison** | Both models are analytical — no training, no PyTorch. Comparison runs in <1 s |
| **Backend /ping warm-up** | Frontend pings the Render instance on page load, eliminating cold-start during demo |
| **Isolated RNG** | No global NumPy state mutation — forecast bootstrap stays reproducible every run |

> **Render free-tier note:** Free services sleep after 15 min of inactivity. The first request after sleep takes ~30 s to wake up (shown as a banner in the UI). After that, all operations are fast.

---

## 📤 CSV Upload

Upload any time-series CSV to run the full forecasting pipeline on your own data.

**Required format (minimum 30 rows):**

```csv
ds,y
2024-01-01,1200.50
2024-01-02,1345.20
2024-01-03,980.75
```

A ready-to-use **sample CSV** (`sample_upload.csv`) is included in the repository root — 365 days of synthetic retail data with injected spikes and drops for anomaly detection demo purposes. Upload it directly to test the CSV workflow end-to-end.

---

## 🌐 Live Stock Data

Fetch real historical stock prices from Yahoo Finance and run the full pipeline on live market data:

| Ticker | Description |
|---|---|
| **NWG.L** | NatWest Group — London Stock Exchange (GBp) |
| **AAPL** | Apple Inc. — NASDAQ (USD) |
| **TSLA** | Tesla Inc. — NASDAQ (USD) |
| **^FTSE** | FTSE 100 Index |
| **^GSPC** | S&P 500 Index |

Or enter any Yahoo Finance symbol. Supports 1-year, 2-year, or 5-year history windows.

> 🔒 **Privacy:** Stock prices are publicly available market data. No user data is stored, logged, or transmitted.

---

## ⚡ Quick Start

> **Three terminal commands from a fresh clone to a running app.**

### Prerequisites

- Python 3.11+  
- Node.js 18+  
- A free API key from [Groq Console](https://console.groq.com/keys) (takes 30 seconds, no credit card)

### 1. Clone

```bash
git clone https://github.com/your-username/forecastiq-natwest.git
cd forecastiq-natwest
```

### 2. Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Open .env and add: GROQ_API_KEY=your_key_here

uvicorn src.main:app --reload --port 8000
```

API: `http://localhost:8000` · Interactive docs: `http://localhost:8000/docs`

### 3. Frontend

```bash
# In a new terminal tab:
cd frontend
npm install
cp .env.example .env
# .env already contains: VITE_API_URL=http://localhost:8000/api

npm run dev
```

App: `http://localhost:5173`

> **That's it.** No Docker, no database, no build step. The backend starts serving `/ping` in under 3 seconds.

---

## 🗂️ Project Structure

```
forecastiq-natwest/
│
├── README.md
├── render.yaml                      # Zero-config Render deployment
├── sample_upload.csv                # 365-day demo CSV for upload testing
│
├── backend/
│   ├── .python-version              # Pins Python 3.11 for Render
│   ├── requirements.txt
│   ├── .env.example                 # Copy → .env, add GROQ_API_KEY
│   └── src/
│       ├── main.py                  # FastAPI app + CORS + startup pre-training
│       ├── api/
│       │   └── routes.py            # All REST endpoints
│       └── services/
│           ├── data_simulator.py    # Fourier synthetic data + seed caching
│           ├── forecasting.py       # OLS + Bootstrap CI + anomaly detection
│           ├── nbeats_forecasting.py # N-BEATS (ICLR 2020) + MC Dropout CI
│           ├── nbeats_pretrain.py   # Background pre-training at startup
│           └── llm_service.py       # Groq/Gemini router with auto-fallback
│
└── frontend/
    ├── index.html
    ├── package.json
    ├── .env.example
    └── src/
        ├── App.jsx                  # Root state, theme
        ├── index.css                # Design system (dark + light)
        └── components/
            ├── Onboarding/          # Data source picker (Sandbox / Stock / CSV)
            ├── Dashboard/           # Main analytics layout
            ├── Charts/
            │   ├── ForecastChart.jsx     # Recharts + CI band + naive baseline
            │   ├── AnomalyPanel.jsx      # Anomaly table + AI per-anomaly insights
            │   ├── ScenarioPlayground.jsx # What-if sliders + side-by-side chart
            │   ├── ModelComparison.jsx   # Classical vs Naive Baseline comparison
            │   ├── ChatPanel.jsx         # Grounded Q&A — verified stats only
            │   └── DataExplorer.jsx      # Searchable table + CSV export
            └── Shared/
                ├── NavBar.jsx            # Model selector + theme toggle
                └── InsightCard.jsx       # AI insight display
```

---

## 🌍 Environment Variables

**Backend** (`backend/.env`):

```env
GROQ_API_KEY=your_groq_key_here       # Required — fast, no daily limits
GEMINI_API_KEY=your_gemini_key_here   # Optional — auto-fallback from Groq on 429
```

**Frontend** (`frontend/.env`):

```env
VITE_API_URL=http://localhost:8000/api
```

> `.env` files are gitignored. Only `.env.example` files are tracked. **Never commit API keys.**

---

## 🚀 Deployment

> **Already live:**
> - Frontend: [forecastiq-natwest.vercel.app](https://forecastiq-natwest.vercel.app/)
> - Backend API: [forecastiq-natwest.onrender.com](https://forecastiq-natwest.onrender.com)
> - API Docs: [forecastiq-natwest.onrender.com/docs](https://forecastiq-natwest.onrender.com/docs)

### Deploy Your Own

**Backend → Render** (free tier):
1. Connect this repo on [render.com](https://render.com) → New Web Service
2. Add `GROQ_API_KEY` (+ optionally `GEMINI_API_KEY`) in Render's Environment tab
3. Deploy — `render.yaml` pre-configures build and start commands

**Frontend → Vercel** (free):
1. Import repo on [vercel.com](https://vercel.com)
2. Root directory: `frontend`, Framework: Vite
3. Add `VITE_API_URL=https://your-render-url.onrender.com/api`
4. Deploy

---

## 🧪 Tests

```bash
cd backend
# activate your venv first, then:
pip install pytest
pytest tests/ -v
```

11 assertions covering: data generation, forecast structure, anomaly detection, scenario comparison, and summary stats shape.

---

## 🧠 Grounded AI — Zero Hallucination by Design

The most important engineering decision in this project:

**The LLM never sees raw data. It never does arithmetic.**

Every AI insight is generated from a pre-computed `summary_stats` dict:

```python
{
  "current_value": 3247.5,
  "forecast_end_value": 3421.2,
  "forecast_end_lower": 2891.0,
  "forecast_end_upper": 3951.4,
  "growth_pct_over_period": 5.3,
  "peak_predicted_date": "2026-04-18",
  "anomaly_count": 6,
  "trend_slope_pct": 12.4,
  ...
}
```

The LLM prompt says: *"Write exactly 3 sentences using ONLY the numbers below. Do not invent or assume any figures."*

This eliminates the most common failure mode in AI analytics tools — hallucinated statistics — while still making the output accessible to non-technical users.

---

## 📚 How We Address the Learning Outcomes

### 1. Ways to look ahead — and how to judge if the approach helps

- Classical OLS+Fourier decomposes the series into trend (OLS) + seasonality (Fourier) + noise — analytically, no training required
- **Compare Models** tab runs both Classical and Naive Baseline on the actual user dataset with MAE/RMSE/MAPE on a 20% holdout — users see empirically whether the model is adding value over a trivial rolling average
- Forecast accuracy is shown honestly: if the model barely beats the naive baseline, the UI says so

### 2. Why simple comparisons matter

- 28-day naive baseline is computed and plotted on **every** forecast — not just the comparison tab
- `model_vs_naive_%` in the Pattern Breakdown card gives a single number: "is this model adding value over a rolling average?"
- The **Compare Models** tab makes this a formal, quantified head-to-head on a held-out 20% test set
- Sometimes the naive baseline wins on very noisy data — the tool shows this honestly

### 3. Communicating uncertainty effectively

- Shaded confidence band (not error bars) makes ranges intuitive
- Band widens visually as the forecast extends further — communicating *future = less certain*
- Every AI insight explicitly states the 95% range in natural language
- CI uses bootstrap resampling on historical residuals — principled, not just ± one standard deviation

---

## 📋 Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Frontend | React 18, Vite | Fast HMR, minimal bundle |
| Charts | Recharts | Shaded CI bands + interactive tooltips |
| Styling | Vanilla CSS (DM Sans + DM Mono) | Zero framework lock-in |
| Backend | Python 3.11, FastAPI | Async, auto OpenAPI docs |
| Forecasting | NumPy, scikit-learn | Pure Python OLS+Fourier, no C++ required |
| Live Data | yfinance | Yahoo Finance — public market data |
| AI (default) | Groq SDK | Llama-3.3-70B — no daily quota limits |
| AI (optional) | google-generativeai | Gemini 2.0 Flash — auto-fallback on 429 |
| Backend hosting | Render | Free tier, CI/CD from GitHub |
| Frontend hosting | Vercel | CDN, instant deploys from GitHub |

---

## ⚠️ Known Limitations

- CSV upload requires exactly two columns: `ds` (YYYY-MM-DD date) and `y` (numeric)
- Forecast accuracy degrades below 90 data points
- Model comparison tab requires ≥60 data points (needs a meaningful holdout)
- Render free-tier cold starts (~30 s after 15 min idle) — open the app a minute before a live demo
- Financial time-series (stock data) have wider confidence intervals by nature — this is accurate, not a bug

---

## 📄 License & Compliance

Released under **Apache License 2.0** in compliance with NatWest Code for Purpose hackathon rules and DCO requirements. All commits signed off under a single author identity.

All synthetic data is generated locally using NumPy — no real, personal, proprietary, or financial data is stored or transmitted by this application.

---

<div align="center">

**Built for NatWest Code for Purpose India Hackathon 2026**

*Making data-driven forecasting accessible — not just to data scientists.*

</div>
