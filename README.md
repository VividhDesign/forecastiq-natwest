<div align="center">

<img src="https://img.shields.io/badge/NatWest-Code%20for%20Purpose%202026-003D7B?style=for-the-badge" alt="NatWest Code for Purpose 2026"/>
<img src="https://img.shields.io/badge/Status-Live%20%26%20Deployed-22c55e?style=for-the-badge" alt="Live & Deployed"/>
<img src="https://img.shields.io/badge/Theme-AI%20Predictive%20Forecasting-f59e0b?style=for-the-badge" alt="AI Predictive Forecasting"/>
<img src="https://img.shields.io/badge/License-Apache%202.0-6366f1?style=for-the-badge" alt="Apache 2.0"/>

# 📊 ForecastIQ

### *Predictive Analytics for Everyone — Not Just Data Scientists*

> Turn any time-series into trustworthy short-term forecasts with confidence intervals,
> anomaly detection, what-if scenario planning, and AI-generated summaries in plain English.

<br/>

🌐 **[Live Demo](https://forecastiq-natwest.vercel.app/)** &nbsp;·&nbsp; 🔌 **[API Docs](https://forecastiq-natwest.onrender.com/docs)** &nbsp;·&nbsp; ⚡ **[Quick Start](#-quick-start-run-locally)**

</div>

---

## The Problem We're Solving

Business teams in procurement, operations, and finance need to plan weeks ahead. They're usually stuck between two frustrating options:

- **Too technical** — Python notebooks, ARIMA tuning, manual feature engineering. Inaccessible to anyone without a data science background.
- **Too simplistic** — Spreadsheet averages with no uncertainty range, no anomaly flags, no scenario modelling.

**ForecastIQ** sits exactly in between. It's a web tool that gives non-technical users honest, mathematically rigorous forecasts with clear uncertainty ranges, anomaly detection, "what-if" scenario modelling, and a conversational AI layer — without writing a single line of code.

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

> **No setup required — the live app is at [forecastiq-natwest.vercel.app](https://forecastiq-natwest.vercel.app/)**
>
> ⚠️ The backend runs on Render's free tier and sleeps after 15 minutes of inactivity. If a loading banner appears, wait ~30 seconds for the server to wake up, then proceed normally.

1. **Open the app** → choose **🛒 E-commerce Sales** + **📈 Aggressive Growth** → click **Launch Dashboard →**
2. On the **📈 Forecast** tab — see the 95% confidence band, naive baseline, and AI-generated insight
3. Click the **🚨 Anomalies** tab → click any red dot for a 3-sentence AI explanation + recommended action
4. Switch to **🎰 Scenario** → drag the growth slider to **1.25×** and hit **Run Scenario**
5. Back on the home screen, click **📈 Live Stock** → enter **NWG.L** (NatWest's own stock) to run the full pipeline on real market data
6. Click **💬 Ask** → type *"When is the peak and how confident are you?"* — the AI answers using only verified numbers

---

## 🏗️ Architecture

```
User Browser (Vercel CDN)
│
│  Step 1 — Pick a data source:
│    🧪 Sample Data  →  POST /api/simulate   (seed=42, in-memory cache, instant)
│    📈 Live Stock   →  GET  /api/fetch-stock (yfinance → Yahoo Finance)
│    📄 Upload CSV   →  POST /api/upload      (any ds/y CSV, ≥30 rows)
│
│  Step 2 — Dashboard loads in parallel:
│    POST /api/forecast         →  Math results in ~0.3 s
│    POST /api/forecast-insight →  LLM insight fires in parallel (non-blocking)
│
└──────────────────────────────────────────────────────────────►

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
│    LLM produces:  English sentences using verified numbers only
│    Primary:  Groq (Llama-3.3-70B) — no daily quota limits
│    Fallback: Gemini 2.0 Flash — auto-switches on rate limit (429)
│
│  Naive Baseline (always shown alongside the main forecast)
│    28-day rolling mean — simple, interpretable reference line
│    Plotted on every forecast chart; model_vs_naive_% shown in Pattern Breakdown card
```

---

## 🔬 The Forecasting Model — From First Principles

We built a **decomposable time-series model** using only NumPy and scikit-learn — no Prophet, no C++ compilation, no binary dependencies of any kind.

### Core Equation

```
y(t) = g(t)  +  s(t)  +  ε
```

| Symbol | Component | Method |
|---|---|---|
| `g(t)` | **Trend** | OLS linear regression: `y = β₀ + β₁·t` |
| `s(t)` | **Seasonality** | Fourier: `Σ [ aₙ cos(2πnt/P) + bₙ sin(2πnt/P) ]` |
| `ε` | **Residuals** | Used for bootstrap CI estimation |

The feature matrix per time step has 17 features total:

```
[t,  cos(2π·1·t/7),  sin(2π·1·t/7),  …  cos(2π·3·t/7),  sin(2π·3·t/7),   ← weekly  (6 features)
     cos(2π·1·t/365), sin(2π·1·t/365), … cos(2π·5·t/365), sin(2π·5·t/365)] ← yearly (10 features)
```

All 17 features fit in a single `LinearRegression().fit()` call — transparent, interpretable, and fast.

### Why Not Just Use a Moving Average or ARIMA?

| Method | Limitation | Our Approach |
|---|---|---|
| Moving average | Repeats past — can't model trend direction | `β₁·t` OLS term explicitly models growth or decline |
| Exponential smoothing | Captures single frequency only | Fourier captures weekly **and** yearly patterns simultaneously |
| ARIMA | Requires manual (p,d,q) tuning per dataset | Auto-fits via OLS — zero hyperparameters |

### Naive Baseline

Every forecast also computes a **28-day rolling mean** — the simplest possible "prediction". It's plotted as a dotted grey line on every chart. The Pattern Breakdown card shows `model_vs_naive_%`, giving a single number that answers: *"Is this model actually better than just taking an average?"*

```
ŷ_naive(t) = mean(y[n-28 : n])   for all future t
```

### Growing Confidence Intervals

```
CI(step) = ŷ ± 1.96 · σ_residual · (1 + √(step / horizon))
```

The band widens automatically as the forecast looks further ahead — communicating *next week is more predictable than next month* without needing any statistics knowledge.

---

## 📊 Naive Baseline — Built-in Sanity Check

Every forecast automatically includes a **28-day rolling mean baseline** — the simplest possible predictor — plotted as a dotted grey line alongside the main OLS+Fourier forecast. This directly addresses Hackathon Learning Outcome #2: *"Why simple comparisons matter when testing ideas."*

```
Classical (OLS + Fourier)                    Naive Baseline (28-day rolling avg)
──────────────────────────────              ────────────────────────────────────
y(t) = β₀ + β₁·t                            ŷ(t) = mean(y[n-28 : n])
     + Σ Fourier seasonality                 flat constant for all future t
     + 95% bootstrap CI band
```

**How it works in practice:**
- The naive baseline runs on every dataset automatically — no extra step needed
- The **Pattern Breakdown card** displays `model_vs_naive_%`: a single number showing how much the OLS+Fourier model outperforms the rolling average
- On very noisy datasets this number can be small or negative — the tool reports it honestly, because accuracy matters more than looking impressive

**Performance:** The naive baseline is a single `numpy.mean()` call — it adds zero latency.

---

## 🧠 Grounded AI — Zero Hallucination by Design

The most important engineering decision in this project:

**The LLM never sees raw data. It never does arithmetic.**

Every AI insight is generated from a pre-computed `summary_stats` dictionary:

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

The LLM prompt explicitly says: *"Write exactly 3 sentences using ONLY the numbers listed below. Do not invent, assume, or calculate any figures."*

This eliminates the most common failure mode in AI analytics — hallucinated statistics — while making every output readable and useful for non-technical users.

---

## ⚡ Speed Engineering

Every interaction is designed to feel immediate:

| Optimization | Result |
|---|---|
| **`seed=42` deterministic caching** | `/simulate` cached after 1st call — sub-millisecond repeated reads |
| **Parallel forecast + LLM** | `/api/forecast` returns math in ~0.3 s; LLM insight fires in parallel (non-blocking) |
| **Vectorized bootstrap CI** | 500-iteration CI computed in < 50 ms — single `rng.choice(size=(500,n))` matrix op |
| **Naive baseline pre-computation** | Rolling mean computed inline with the forecast — adds ~0 ms overhead |
| **Backend `/ping` warm-up** | Frontend pings Render on page load, eliminating cold-start from the user's critical path |
| **Isolated RNG** | No global NumPy state mutation — bootstrap stays fully reproducible every run |

> **Render free-tier note:** The backend sleeps after 15 minutes of inactivity. The first request after sleep takes ~30 s to wake up (the UI shows a banner). All subsequent operations are fast.

---

## 📤 CSV Upload

Upload any time-series CSV to run the full pipeline on your own data.

**Required format (minimum 30 rows):**

```csv
ds,y
2024-01-01,1200.50
2024-01-02,1345.20
2024-01-03,980.75
```

A ready-to-use **`sample_upload.csv`** is included in the repository root — 365 days of synthetic retail data with injected spikes and drops, ready to upload and demonstrate the anomaly detection end-to-end.

---

## 🌐 Live Stock Data

Fetch real historical closing prices from Yahoo Finance and run the full forecasting pipeline on live market data:

| Ticker | Description |
|---|---|
| **NWG.L** | NatWest Group — London Stock Exchange (GBp) |
| **AAPL** | Apple Inc. — NASDAQ (USD) |
| **TSLA** | Tesla Inc. — NASDAQ (USD) |
| **^FTSE** | FTSE 100 Index |
| **^GSPC** | S&P 500 Index |

Or enter any Yahoo Finance symbol. Supports 1-year, 2-year, or 5-year history windows.

> 🔒 **Privacy:** Stock prices are public market data sourced from Yahoo Finance. No user data is stored, logged, or transmitted by this application.

---

## ⚡ Quick Start — Run Locally

> **Don't want to clone? The live app works without any setup at [forecastiq-natwest.vercel.app](https://forecastiq-natwest.vercel.app/) — it connects to our hosted backend automatically.**

If you'd prefer to run everything locally from a fresh clone:

### Prerequisites

- Python 3.11+
- Node.js 18+
- A free API key from [Groq Console](https://console.groq.com/keys) — takes 30 seconds, no credit card required

### 1. Clone the repository

```bash
git clone https://github.com/VividhDesign/forecastiq-natwest.git
cd forecastiq-natwest
```

### 2. Start the backend

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Now open backend/.env and set:
#   GROQ_API_KEY=your_groq_key_here

# Start the server
uvicorn src.main:app --reload --port 8000
```

Backend API: `http://localhost:8000`
Interactive API docs: `http://localhost:8000/docs`

### 3. Start the frontend

Open a **new terminal tab**, then:

```bash
cd forecastiq-natwest/frontend

npm install

# Create a local .env pointing to your local backend:
echo "VITE_API_URL=http://localhost:8000/api" > .env

npm run dev
```

App runs at: `http://localhost:5173`

> **That's it.** No Docker, no database, no build step. The backend starts serving `/ping` in under 3 seconds.

---

## 🧪 Running the Tests

```bash
cd backend
# Activate your venv first, then:
pytest tests/ -v
```

The test suite covers 9 assertions across the core pipeline:
- Synthetic data generation (length, keys, positive values, anomaly injection)
- Forecast structure (required keys, exact row count, anomaly detection)
- Summary stats schema (all keys required by LLM prompts)
- Scenario forecast (baseline vs scenario keys, growth multiplier effect)

---

## 🗂️ Project Structure

```
forecastiq-natwest/
│
├── README.md
├── render.yaml                      # Zero-config Render deployment descriptor
├── sample_upload.csv                # 365-day demo CSV for upload feature testing
│
├── backend/
│   ├── .python-version              # Pins Python 3.11 for Render
│   ├── requirements.txt
│   ├── .env.example                 # Safe template — copy to .env and add your key
│   ├── tests/
│   │   └── test_forecasting.py      # Pytest suite (9 assertions)
│   └── src/
│       ├── main.py                  # FastAPI app entry point + CORS + health endpoints
│       ├── api/
│       │   └── routes.py            # All REST endpoints with docstrings
│       └── services/
│           ├── data_simulator.py    # Fourier synthetic data generator + seed caching
│           ├── forecasting.py       # OLS + Bootstrap CI + anomaly detection engine
│           └── llm_service.py       # Groq / Gemini router with automatic fallback
│
└── frontend/
    ├── index.html
    ├── package.json
    ├── .env.example                 # Contains production backend URL (Render)
    └── src/
        ├── App.jsx                  # Root component — global state + theme management
        ├── index.css                # Full design system (dark + light mode tokens)
        └── components/
            ├── Onboarding/          # Landing page + data source selector
            ├── Dashboard/           # Main analytics layout and tab routing
            ├── Charts/
            │   ├── ForecastChart.jsx      # Recharts line chart + shaded CI band
            │   ├── AnomalyPanel.jsx       # Anomaly table + AI per-anomaly insights
            │   ├── ScenarioPlayground.jsx # What-if sliders + side-by-side chart
            │   ├── ModelComparison.jsx    # Classical vs Naive Baseline comparison
            │   ├── ChatPanel.jsx          # Grounded Q&A — verified stats only
            │   └── DataExplorer.jsx       # Searchable raw data table + CSV export
            └── Shared/
                ├── NavBar.jsx             # Model selector (Groq / Gemini) + theme toggle
                └── InsightCard.jsx        # Reusable AI insight display card
```

---

## 🌍 Environment Variables

**Backend** (`backend/.env` — copy from `backend/.env.example`):

```env
GROQ_API_KEY=your_groq_key_here       # Required — fast inference, no daily quota limits
GEMINI_API_KEY=your_gemini_key_here   # Optional — used as automatic fallback if Groq returns 429
```

**Frontend** (`frontend/.env` — create manually for local development):

```env
VITE_API_URL=http://localhost:8000/api
```

> `frontend/.env.example` contains the **production Render URL** and is used by the deployed Vercel app. For local development, create your own `frontend/.env` pointing to `localhost:8000` as shown above.

> `.env` files are gitignored. Only `.env.example` files are committed. **Never commit real API keys.**

---

## 🚀 Deployment

> **Already live:**
> - Frontend: [forecastiq-natwest.vercel.app](https://forecastiq-natwest.vercel.app/)
> - Backend API: [forecastiq-natwest.onrender.com](https://forecastiq-natwest.onrender.com)
> - Interactive API Docs: [forecastiq-natwest.onrender.com/docs](https://forecastiq-natwest.onrender.com/docs)

### Deploy Your Own Instance

**Backend → Render** (free tier):
1. Connect this repo on [render.com](https://render.com) → New Web Service
2. Add `GROQ_API_KEY` (and optionally `GEMINI_API_KEY`) in Render's Environment tab
3. Deploy — `render.yaml` pre-configures the build and start commands automatically

**Frontend → Vercel** (free):
1. Import repo on [vercel.com](https://vercel.com)
2. Set root directory to `frontend`, framework preset to **Vite**
3. Add environment variable: `VITE_API_URL=https://your-render-url.onrender.com/api`
4. Deploy

---

## 📚 How We Address the Three Learning Outcomes

### 1. Ways to look ahead — and how to judge if the approach helps

Our OLS+Fourier model decomposes the series into a trend component (OLS) and a seasonality component (Fourier) without any hyperparameter tuning. Every forecast also computes a 28-day naive rolling average alongside it, so users can immediately see whether the model adds value over a trivial benchmark. The Pattern Breakdown card reports `model_vs_naive_%` as a single, honest number — if the baseline is competitive on noisy data, the tool says so.

### 2. Why simple comparisons matter

The 28-day naive baseline runs on every dataset automatically — it's never hidden on a separate tab. The Pattern Breakdown card surfaces the `model_vs_naive_%` difference at a glance. On very noisy datasets this number can be low or even negative, because showing the truth is more useful than showing only results that look impressive.

### 3. Communicating uncertainty effectively

We use a shaded confidence band (not error bars) so the uncertainty range is immediately intuitive. The band widens as the forecast extends further, so even a non-technical user can see that next week is more predictable than next month. Every AI-generated insight explicitly states the 95% probability range in plain English. The confidence intervals are computed via bootstrap resampling of historical residuals — principled, not just ± one standard deviation.

---

## 📋 Tech Stack

| Layer | Technology | Why We Chose It |
|---|---|---|
| Frontend | React 18 + Vite | Fast hot-reload dev server, minimal production bundle |
| Charts | Recharts | Composable — easy to add shaded CI bands and interactive tooltips |
| Styling | Vanilla CSS (DM Sans + DM Mono) | Full control, zero framework lock-in |
| Backend | Python 3.11 + FastAPI | Async-native, generates interactive OpenAPI docs automatically |
| Forecasting | NumPy + scikit-learn | Pure Python OLS+Fourier — no C++ compilation anywhere |
| Live Data | yfinance | Public Yahoo Finance API — real stock prices, zero auth required |
| AI (default) | Groq SDK (Llama-3.3-70B) | Fast inference, no daily quota limits |
| AI (fallback) | google-generativeai (Gemini 2.0 Flash) | Auto-switches when Groq returns a 429 rate limit |
| Backend hosting | Render | Free tier, automatic CI/CD deploys from GitHub |
| Frontend hosting | Vercel | Global CDN, instant preview deployments from GitHub |

---

## Known Limitations

Being honest about limitations is part of good engineering — and the judges' guidelines ask for it explicitly.

- **CSV format:** The upload endpoint expects exactly two columns — `ds` (date string, YYYY-MM-DD) and `y` (numeric). Any other column names or date formats will be rejected with a descriptive error message.
- **Short histories:** Forecast accuracy is noticeably lower below ~90 data points. The OLS+Fourier model needs enough history to fit weekly and yearly Fourier terms; with very few rows, those patterns are underfit.
- **Cold starts on Render free tier:** The backend sleeps after 15 minutes of inactivity. The first request after sleep takes around 30 seconds; the UI shows a "Waking up..." banner during this window. All subsequent requests are fast.
- **Stock data confidence intervals:** Financial time-series are genuinely noisy, so the bootstrap-derived confidence bands are naturally wider than on smoother operational data. This is mathematically expected behaviour, not a model defect.

---

## 📄 License & Compliance

Released under the **Apache License 2.0** in compliance with NatWest Code for Purpose hackathon rules and DCO requirements.

All commits include a `Signed-off-by` DCO sign-off under a single author identity. All synthetic data is generated locally using NumPy — no real, personal, proprietary, or financial data is stored, logged, or transmitted by this application.

---

<div align="center">

**Built for NatWest Code for Purpose India Hackathon 2026**

*Making data-driven forecasting accessible — not just to data scientists.*

</div>
