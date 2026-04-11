"""
Multi-LLM Insight Service
=========================
A model-agnostic router that takes verified numerical outputs from the
forecasting engine and translates them into plain-English summaries for
non-technical business users.

Supported models:
  - Google Gemini 2.0 Flash  (via google-generativeai)
  - Groq / Llama-3.3-70B     (via groq)

Design principle: LLMs here are *translators*, not *calculators*.
All numbers come from the deterministic forecasting engine — the LLM
only converts those numbers into readable sentences. This prevents
hallucinated figures and ensures factual accuracy.
"""

import os
from dotenv import load_dotenv
from typing import Literal

load_dotenv()

ModelChoice = Literal["gemini", "groq"]


def _build_forecast_prompt(stats: dict) -> str:
    """
    Construct a structured prompt that feeds only verified numbers to the LLM.
    The LLM's job is ONLY to convert numbers → business-friendly English.
    """
    return f"""You are a concise, expert data analyst summarizing a forecast for a business audience.
Use only the numbers provided. Do NOT invent or assume any additional figures.

Dataset: {stats['context_label']}
Forecast Period: Next {stats['forecast_weeks']} weeks
Current Value: {stats['current_value']}
Forecasted End Value: {stats['forecast_end_value']} (Range: {stats['forecast_end_lower']} – {stats['forecast_end_upper']})
Projected Growth: {stats['growth_pct_over_period']}%
Peak Expected On: {stats['peak_predicted_date']}
Anomalies Detected in History: {stats['anomaly_count']} total ({stats['anomaly_spike_count']} spikes, {stats['anomaly_drop_count']} drops)

Write exactly 3 sentences:
1. Summarize the overall forecast outlook and growth.
2. Describe the confidence range so a non-expert understands uncertainty.
3. Mention the peak date and historical anomaly count and what that implies for vigilance.

Be direct, professional, and avoid jargon."""


def _build_anomaly_prompt(anomaly: dict, context_label: str) -> str:
    """Build a prompt for explaining a single detected anomaly."""
    direction_word = "unexpected spike" if anomaly["direction"] == "spike" else "sudden drop"
    return f"""You are a concise data analyst helping a business team act on an anomaly.
An {direction_word} was detected in {context_label} data.

Date: {anomaly['ds']}
Actual Value: {anomaly['y']}
Expected Value (Model Forecast): {anomaly['yhat']}
Deviation: {anomaly['pct_deviation']}%

Write exactly 3 sentences:
1. Describe what happened in plain language (the anomaly itself).
2. Suggest the most likely business cause or contributing factor.
3. Recommend one concrete next step the team should take (e.g., check system logs, review regional data, contact operations, monitor the next 48 hours).

Be specific and actionable. Avoid technical jargon."""


def _build_scenario_prompt(baseline_stats: dict, scenario_stats: dict) -> str:
    """Build a prompt comparing baseline vs scenario forecasts."""
    return f"""You are a concise business analyst comparing two forecast scenarios.

Dataset: {baseline_stats['context_label']}
Baseline Forecast End Value: {baseline_stats['forecast_end_value']}
Scenario Forecast End Value: {scenario_stats['forecast_end_value']}
Baseline Growth: {baseline_stats['growth_pct_over_period']}%
Scenario Growth: {scenario_stats['growth_pct_over_period']}%

Write exactly 2 sentences:
1. Compare the two scenarios and highlight the key difference in numbers.
2. Provide a clear business recommendation based on the comparison.

Be direct and professional."""


def _call_gemini(prompt: str) -> str:
    """Call Google Gemini 2.0 Flash API."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment variables.")

    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


def _call_groq(prompt: str) -> str:
    """Call Groq (Llama-3) API."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in environment variables.")

    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def _call_with_fallback(prompt: str, model_choice: ModelChoice) -> str:
    """
    Call the selected LLM. If Gemini returns a quota/rate-limit error (429),
    automatically fall back to Groq so demos never fail.
    """
    if model_choice == "groq":
        return _call_groq(prompt)

    try:
        return _call_gemini(prompt)
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "quota" in err_str.lower() or "RESOURCE_EXHAUSTED" in err_str:
            # Silently fall back to Groq on quota exhaustion
            return "" + _call_groq(prompt)
        raise


def generate_forecast_insight(stats: dict, model_choice: ModelChoice = "gemini") -> str:
    prompt = _build_forecast_prompt(stats)
    return _call_with_fallback(prompt, model_choice)


def generate_anomaly_insight(anomaly: dict, context_label: str, model_choice: ModelChoice = "gemini") -> str:
    prompt = _build_anomaly_prompt(anomaly, context_label)
    return _call_with_fallback(prompt, model_choice)


def generate_scenario_insight(
    baseline_stats: dict, scenario_stats: dict, model_choice: ModelChoice = "gemini"
) -> str:
    prompt = _build_scenario_prompt(baseline_stats, scenario_stats)
    return _call_with_fallback(prompt, model_choice)


def _build_chat_prompt(question: str, stats: dict, anomaly_count: int, context_label: str) -> str:
    """Build a grounded Q&A prompt — LLM answers from verified data only."""
    return f"""You are a data analyst assistant for {context_label} forecasting data.
Answer the user's question using ONLY the facts provided below. Do not invent numbers.

Verified Data Context:
- Current value: {stats.get('current_value')}
- Forecasted end value ({stats.get('forecast_weeks')} weeks): {stats.get('forecast_end_value')}
- 95% confidence range: {stats.get('forecast_end_lower')} to {stats.get('forecast_end_upper')}
- Projected growth: {stats.get('growth_pct_over_period')}%
- Peak expected on: {stats.get('peak_predicted_date')}
- Anomalies in history: {anomaly_count} total ({stats.get('anomaly_spike_count')} spikes, {stats.get('anomaly_drop_count')} drops)
- Trend direction over history: {stats.get('trend_slope_pct')}% overall change
- Seasonal amplitude: {stats.get('seasonal_amplitude')}
- Naive baseline end: {stats.get('naive_forecast_end')}
- Model vs naive: {stats.get('model_vs_naive_pct')}%

User question: {question}

Answer in 2-3 clear sentences. If the question cannot be answered from the data above, say so honestly."""


def generate_chat_insight(
    question: str,
    stats: dict,
    anomaly_count: int,
    context_label: str,
    model_choice: ModelChoice = "groq",
) -> str:
    """Answer a free-form user question grounded in the verified forecast data."""
    prompt = _build_chat_prompt(question, stats, anomaly_count, context_label)
    return _call_with_fallback(prompt, model_choice)


def _build_comparison_prompt(classical_stats: dict, nbeats_stats: dict, winner: str) -> str:
    """
    Build a prompt for the Classical vs N-BEATS head-to-head comparison.
    Both models share the same decomposition philosophy (trend + seasonality),
    but Classical does it analytically while N-BEATS learns it from data.
    """
    winner_name = (
        "Classical (OLS + Fourier)"
        if winner == "classical"
        else "N-BEATS (Interpretable Deep Learning)"
    )

    nbeats_block = ""
    if nbeats_stats and "error" not in nbeats_stats:
        nbeats_block = f"""
Model B — N-BEATS (Interpretable Deep Learning, ICLR 2020):
- Forecast End Value: {nbeats_stats.get('forecast_end_value', 'N/A')}
- Growth: {nbeats_stats.get('growth_pct_over_period', 'N/A')}%
- MAE: {nbeats_stats.get('mae', 'N/A')}
- RMSE: {nbeats_stats.get('rmse', 'N/A')}
- MAPE: {nbeats_stats.get('mape', 'N/A')}%
- Architecture: Trend Stack (polynomial basis) → Seasonality Stack (Fourier basis)
  Learned end-to-end via doubly-residual backpropagation."""
    else:
        nbeats_block = "\nModel B — N-BEATS: failed to run (see error)."

    return f"""You are a data science expert explaining model comparison results to a business audience.
Two forecasting models that share the same conceptual approach were tested head-to-head:
both decompose the time series into trend + seasonality — but one uses analytical equations,
the other learns the decomposition from data.

Model A — Classical (OLS + Fourier Decomposition):
- Forecast End Value: {classical_stats.get('forecast_end_value')}
- Growth: {classical_stats.get('growth_pct_over_period')}%
- MAE: {classical_stats.get('mae')}
- RMSE: {classical_stats.get('rmse')}
- MAPE: {classical_stats.get('mape')}%
- Architecture: Linear regression (OLS) + Fourier seasonality. No training required.
  Fully analytical and interpretable.{nbeats_block}

Winner (lowest MAE on 20% holdout set): {winner_name}

Write exactly 4 sentences:
1. State which model performed better and by how much on MAE and RMSE.
2. Explain WHY in plain language — relate it to the data's pattern (e.g. regularity,
   seasonality strength, noise level, dataset size).
3. Describe when the other model would be the better choice (different data characteristics).
4. Give a concrete recommendation: for this dataset, which model should the team use and why?

Be professional, educational, and concise. Avoid jargon."""


def generate_comparison_insight(
    classical_stats: dict,
    nbeats_stats: dict,
    winner: str,
    model_choice: ModelChoice = "gemini",
) -> str:
    """Generate an AI insight comparing Classical (analytical) vs N-BEATS (learned)."""
    prompt = _build_comparison_prompt(classical_stats, nbeats_stats, winner)
    return _call_with_fallback(prompt, model_choice)

