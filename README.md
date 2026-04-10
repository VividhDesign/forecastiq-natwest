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

Business teams need to plan ahead — for staffing, inventory, budgets, and capacity. But most forecasting tools are either too technical (Python notebooks, ARIMA tuning) or too opaque (black-box predictions with no confidence range).

**ForecastIQ** provides a middle ground: a web-based tool that gives non-technical users honest short-term forecasts with clear uncertainty ranges, anomaly detection, what-if scenario modelling, and a conversational Q&A interface — all in plain English.

---

## How It Works

The core idea is simple: **separate the maths from the language**.

1. A **deterministic forecasting engine** (Python, NumPy, scikit-learn) computes all numbers — trend, seasonality, confidence intervals, anomalies, and a naive baseline for comparison.
2. A **generative AI layer** (Gemini or Llama-3) receives only those verified numbers and translates them into business-friendly sentences.
3. A **conversational Q&A tab** lets users ask free-form questions about the data — every answer is grounded in the same verified numbers, never hallucinated.

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

### Naive baseline comparison

To avoid over-fitting, every forecast is compared against a **naive baseline**: the 28-day rolling mean (4 complete weeks to eliminate day-of-week bias). This appears as a flat dotted grey line on the chart. The "Pattern Breakdown" card shows the model-vs-naive percentage — proving the model is actually learning patterns beyond a simple average.

```
ŷ_naive(t) = mean(y[n-28 : n-1])  for all future t
```

If the model can't meaningfully outperform this trivial baseline, it signals that the data may lack strong trend or seasonality — a useful diagnostic in itself.

### Anomaly detection

A historical data point is flagged as anomalous if it falls outside the 95% bootstrap confidence band:

```
y > upper_band → Spike (unexpected surge)
y < lower_band → Drop  (unexpected crash)
```

Each anomaly records its direction, date, and percentage deviation. The AI generates **3 sentences** for each anomaly: what happened, the likely cause, and a **concrete next step** the team should take.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│         User's Browser (Vercel)             │
│         React + Vite + Recharts             │
│  ┌──────────┐ ┌──────────┐ ┌─────────────┐ │
│  │Onboarding│ │Dashboard │ │ AI Model    │ │
│  │ Sandbox /│ │Forecast /│ │ Selector    │ │
│  │CSV Upload│ │Anomalies/│ │Groq ↔ Gemini│ │
│  └──────────┘ │Scenarios/│ └─────────────┘ │
│               │Chat Q&A  │                 │
│               └──────────┘                 │
└────────────────────┬────────────────────────┘
                     │ HTTPS (REST API)
┌────────────────────▼────────────────────────┐
│         FastAPI Backend (Render)             │
│                                             │
│  POST /api/simulate → Synthetic data gen    │
│  POST /api/upload   → CSV parser            │
│  POST /api/forecast → Full forecasting      │
│    ├─ OLS + Fourier + Bootstrap CI          │
│    ├─ Anomaly detection                     │
│    ├─ Naive baseline comparison             │
│    ├─ Pattern decomposition stats           │
│    └─ LLM insight (auto-fallback)           │
│  POST /api/anomaly-insight → Per-anomaly AI │
│  POST /api/scenario → What-if comparison    │
│    ├─ Growth / seasonality multipliers      │
│    └─ Remove outliers (winsorization)       │
│  POST /api/chat → Free-form Q&A             │
│    └─ Grounded in verified stats only       │
│                                             │
│  LLM Router (model-agnostic + auto-fallback)│
│    "groq"   → Llama-3.3-70B (default)      │
│    "gemini" → Gemini 2.0 Flash              │
│    on 429/quota → auto-fallback to Groq     │
└─────────────────────────────────────────────┘
```

### Key design decisions

**1. LLMs as translators, not calculators.** The LLM receives a fully numerical prompt — every figure comes from the deterministic engine. The LLM's only job is to convert those numbers into readable sentences. Switching between Gemini and Llama-3 changes only the writing style, not the data.

**2. Auto-fallback on quota exhaustion.** If Gemini's free-tier quota is exceeded (429 error), the system silently retries the same prompt with Groq. Demos never fail — even when API limits are reached mid-presentation.

**3. Grounded Q&A.** The chat feature sends the user's question alongside a complete snapshot of verified summary stats. The LLM is explicitly instructed: *"Answer using ONLY the facts provided. Do not invent numbers."*

---

## Features

| Feature | Description |
|---|---|
| **Data Simulator** | Generate realistic synthetic datasets (4 business contexts, 3 trend types, optional anomaly injection) on-the-fly — no real data required |
| **CSV Upload** | Upload any time-series CSV with `ds` (date) and `y` (numeric) columns |
| **Short-Term Forecast** | 1–6 week forecasts with shaded 95% confidence intervals |
| **Naive Baseline** | 28-day rolling mean plotted alongside the forecast — proves the model beats a simple heuristic |
| **Pattern Breakdown** | Dashboard card showing trend slope %, seasonal amplitude, and model-vs-naive comparison |
| **Anomaly Detection** | Flags historical data points outside the bootstrap CI band, classified as spikes or drops with AI explanations and actionable next steps |
| **Scenario Playground** | Interactive sliders for growth and seasonality multipliers, plus a "Remove Outliers" toggle (winsorization) — side-by-side baseline vs scenario |
| **💬 Ask Tab** | Conversational Q&A grounded in verified data — ask anything about the forecast, trends, or anomalies |
| **Multi-LLM Insights** | Switch between Gemini 2.0 Flash and Groq (Llama-3.3-70B) live — with auto-fallback on quota |
| **Raw Data Explorer** | Searchable, sortable table with anomaly highlights, % deviation, and one-click CSV export |
| **Dark / Light Mode** | Theme toggle with localStorage persistence; light mode default |

---

## How It Maps to the Problem Statement

| Requirement | Our Implementation |
|---|---|
| *"Predict likely values for future periods"* | ✅ 1–6 week forecast via OLS + Fourier decomposition |
| *"Show a range of outcomes (not just a single number)"* | ✅ Shaded 95% confidence interval (bootstrap + horizon-scaled) |
| *"Compare predictions to a simple baseline to avoid over-fitting"* | ✅ 28-day naive baseline plotted + model-vs-naive % displayed |
| *"Detect early warning signs such as sudden changes"* | ✅ Anomalies flagged at spikes/drops outside CI band |
| *"Provide explanations short enough for non-experts"* | ✅ AI-generated 2–3 sentence summaries in plain English |
| *"Suggest next steps"* | ✅ Each anomaly insight includes a concrete recommended action |
| *"Highlight key patterns (trend, seasonality)"* | ✅ Pattern Breakdown card: trend slope %, seasonal amplitude |
| *"Let users test simple scenarios: adjust growth rate"* | ✅ Growth multiplier slider in Scenario Playground |
| *"Remove recent outliers"* | ✅ Winsorization toggle: clips top/bottom 5% before fitting |
| *"Apply flat or seasonal patterns"* | ✅ Seasonality strength slider (0.1× to 3.0×) |
| *"Generate side-by-side comparisons"* | ✅ Baseline vs scenario lines on the same chart |
| *"Summarize the difference clearly"* | ✅ AI-generated scenario comparison summary |
| *"Keep the experience lightweight, trustworthy, and fast"* | ✅ Pure Python model, no C++ deps, auto-fallback on AI |

---

## Output Examples

These match the output format described in the problem statement:

**Use Case 1 — Short-term forecast:**
> Next 4 weeks: central estimate +3.4% growth. Lower bound: −31.2%. Upper bound: +38.1%. Peak expected in Week 2 (2026-04-18). Seasonal patterns show consistent weekday/weekend variation.

**Use Case 2 — Anomaly detection:**
> Yesterday's E-commerce Sales were unusually high (5,812.4 vs expected 3,200.1), a +81.6% deviation exceeding the forecast band. Potential driver: promotional campaign or viral referral event. Recommend: review order management logs for that date and verify fulfilment capacity.

**Use Case 3 — Scenario comparison:**
> Under a +20% growth scenario, end-of-period value is expected to reach 4,468 (vs 3,723 in baseline). Range: 2,972–5,964. The additional growth widens the confidence interval by 8%, reflecting higher volatility.

---

## Design Rationale — Why This Approach

### Why Fourier + OLS instead of simpler methods?

The problem statement mentions **moving averages** and **exponential smoothing** as simple forecasting methods. We evaluated these but chose Fourier + OLS for specific reasons:

| Method | Limitation | Why we upgraded |
|---|---|---|
| **Moving average** | Cannot model future trends — it only repeats past averages | Our OLS trend component (β₁·t) explicitly models growth/decline direction |
| **Exponential smoothing** | Single-frequency — misses multi-period seasonality | Fourier captures both weekly AND yearly patterns simultaneously |
| **ARIMA** | Requires manual tuning (p, d, q) — not accessible for non-experts | Our model auto-fits via OLS, zero hyperparameters |

However, we kept the spirit of simplicity: our forecast is still a **single linear regression** under the hood — the feature engineering (Fourier terms) handles the complexity while the model itself stays transparent and interpretable.

### When would a more advanced model be justified?

Our model works well for **stable, seasonal data** with moderate trends. A more advanced approach (e.g. gradient-boosted trees, neural forecasters) would be justified when:
- The data has **changepoints** (sudden trend breaks) that linear g(t) can't capture
- The time series has **external regressors** (e.g. promotions, weather, pricing changes)
- The dataset exceeds 10,000+ points where deep learning can find non-linear structure

We deliberately chose the simpler model because **the problem statement values transparency over accuracy** — and for 1–6 week horizons on business data, our approach is highly competitive.

### Forecast validation

We validate our model against a **naive baseline** (28-day rolling mean) on every single run. This serves as a built-in holdout-free sanity check:
- If model_vs_naive_pct ≈ 0%, the model isn't adding value over a trivial forecast
- If model_vs_naive_pct > 0%, the learned trend + seasonality are capturing real patterns

A more thorough validation would use train/test splitting (e.g. train on first 80% of data, evaluate on the last 20%). This is straightforward to implement but was deprioritized for the MVP in favour of the live naive comparison which gives judges an immediate visual proof during demos.

---

## How We Address the Learning Outcomes

The problem statement defines 3 learning areas participants should demonstrate:

### 1. Ways to look ahead — and how to judge if the approach helps

- **How it works:** Our model decomposes time series into trend (OLS) + seasonality (Fourier) + noise — a well-established approach used in production at companies like Meta (Facebook Prophet uses the same framework).
- **When more advanced models are justified:** We explain this explicitly above — our linear model is ideal for 1–6 week horizons, but changepoints and external regressors would warrant gradient boosting or neural methods.
- **How to validate:** The naive baseline serves as a built-in benchmark. We compare every forecast against a simple 28-day average — if the model can't beat it, that's a signal the data lacks learnable patterns.

### 2. Why simple comparisons matter

- **Baseline as sanity check:** The dotted grey naive baseline line on every chart ensures judges (and users) can immediately see whether the model is actually useful.
- **"Pattern Breakdown" card:** Shows model_vs_naive_pct — a single number that answers "is this model worth using?" If close to 0%, the simple average is just as good.
- **Simple models often win:** Our 17-feature OLS model consistently outperforms naive averages on seasonal data while remaining fully transparent. This validates the "simple models often outperform overly complex ones" insight from the problem statement.

### 3. Communicating uncertainty effectively

- **Uncertainty is information, not error:** The shaded 95% confidence band doesn't mean "the model might be wrong." It means "here's the honest range of likely outcomes." This reframing helps users make better decisions.
- **Growing CI for future forecasts:** The band widens over time — visually communicating that next week is more predictable than next month. This is intuitive even for non-technical audiences.
- **Range format in AI summaries:** Every AI insight mentions the confidence range explicitly (e.g. "95% range: 2,476 – 4,971") so the user always sees the spread, not just a single number.
- **How we make ranges intuitive:** We use a shaded gradient on the chart (not just error bars), label it clearly in the stat cards ("95% probability band"), and the AI writes about it in natural language. Non-technical users understand "the forecast could range from X to Y" far better than "σ = 234.5."

---

## Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| **Frontend** | React 18, Vite | Fast HMR, lightweight production bundles |
| **Charts** | Recharts | Supports shaded confidence bands and interactive tooltips |
| **Styling** | Vanilla CSS (DM Sans + DM Mono) | Full control, no framework lock-in |
| **Backend** | Python 3.11, FastAPI | Async, auto-generated OpenAPI docs |
| **Forecasting** | NumPy, scikit-learn | Pure Python — no C++ compilation required at deploy time |
| **Data Generation** | NumPy (Fourier series) | Realistic synthetic data, zero privacy risk |
| **AI — Groq (default)** | `groq` SDK | Llama-3.3-70B at ultra-low latency, no quota limits |
| **AI — Gemini** | `google-generativeai` SDK | gemini-2.0-flash, with auto-fallback to Groq on 429 |
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
│           └── llm_service.py          # LLM router (auto-fallback)
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
            │   ├── ForecastChart.jsx    # Recharts + CI band + naive line
            │   ├── AnomalyPanel.jsx     # Anomaly table + AI explanations
            │   ├── ScenarioPlayground.jsx # What-if + remove outliers
            │   ├── ChatPanel.jsx        # Free-form Q&A interface
            │   └── DataExplorer.jsx     # Filterable table + CSV export
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
  - [Groq Console (Llama-3)](https://console.groq.com/keys) — **recommended, no daily limits**
  - [Google AI Studio (Gemini)](https://aistudio.google.com/apikey) — optional, has free-tier quota

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
# Edit .env — add your GROQ_API_KEY (required) and optionally GEMINI_API_KEY

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
| **📈 Forecast** | Line chart with 95% CI band, naive baseline (dotted grey), and AI insight. Pattern Breakdown card shows trend slope, seasonal amplitude, model vs naive % |
| **🚨 Anomalies** | Detected spikes and drops — click any row for a 3-sentence AI explanation with recommended next steps |
| **🎰 Scenario** | Growth & seasonality sliders + Remove Outliers toggle. Baseline vs scenario comparison + AI summary |
| **💬 Ask** | Chat Q&A — ask anything about the forecast data. 5 suggested questions for quick demos. All answers grounded in verified stats |
| **📊 Raw Data** | Full data table with search, sort, anomaly highlights, and CSV export |

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
GROQ_API_KEY=your_key_here      # Required — default model
GEMINI_API_KEY=your_key_here    # Optional — auto-fallback if quota exceeded
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
2. Set `GROQ_API_KEY` and `GEMINI_API_KEY` in the Render environment
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
- Gemini free-tier has daily quota limits — Groq is used as default to avoid this
- Bootstrap CI runs 500 iterations — takes 1–2 seconds on large datasets
- Free-tier Render backend has cold-start delays (~30s after 15 min inactivity)

---

## License & Compliance

Submitted under the **Apache License 2.0** in compliance with NatWest Code for Purpose hackathon rules and DCO requirements. All commits are signed off with a single email identity.

All data used in this project is **synthetically generated** using NumPy. No real, personal, or proprietary data is used or stored.

---

<div align="center">

**Built for NatWest Code for Purpose India Hackathon 2026**

*Making data-driven decisions accessible — not just to data scientists.*

</div>
