"""
Multi-LLM Insight Service
Model-agnostic router that translates raw forecast statistics into
plain English summaries for non-technical users.

Supported models:
  - Google Gemini Pro  (via google-generativeai)
  - Groq / Llama-3     (via groq)

The LLMs are only used for text translation — all numbers come from
Prophet's deterministic maths, ensuring zero hallucination on figures.
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
    return f"""You are a concise data analyst.
A {direction_word} was detected in {context_label} data.

Date: {anomaly['ds']}
Actual Value: {anomaly['y']}
Expected Value (Model Forecast): {anomaly['yhat']}
Deviation: {anomaly['pct_deviation']}%

Write 2 sentences:
1. Describe the anomaly in plain language.
2. Suggest 1 possible business cause or recommended next action.

Be concise and avoid technical jargon."""


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
    """Call Google Gemini Pro API."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment variables.")

    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro")
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
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def generate_forecast_insight(stats: dict, model_choice: ModelChoice = "gemini") -> str:
    """
    Generate a plain-English forecast summary using the chosen LLM.

    Args:
        stats: Summary statistics from forecasting.py (summary_stats).
        model_choice: 'gemini' or 'groq'.

    Returns:
        A 3-sentence business-friendly forecast summary string.
    """
    prompt = _build_forecast_prompt(stats)
    if model_choice == "gemini":
        return _call_gemini(prompt)
    elif model_choice == "groq":
        return _call_groq(prompt)
    else:
        raise ValueError(f"Unsupported model_choice: {model_choice}")


def generate_anomaly_insight(anomaly: dict, context_label: str, model_choice: ModelChoice = "gemini") -> str:
    """
    Generate a plain-English explanation for a single anomaly.

    Args:
        anomaly: Anomaly record from forecasting.py.
        context_label: Human-readable dataset name.
        model_choice: 'gemini' or 'groq'.

    Returns:
        A 2-sentence anomaly explanation string.
    """
    prompt = _build_anomaly_prompt(anomaly, context_label)
    if model_choice == "gemini":
        return _call_gemini(prompt)
    elif model_choice == "groq":
        return _call_groq(prompt)
    else:
        raise ValueError(f"Unsupported model_choice: {model_choice}")


def generate_scenario_insight(
    baseline_stats: dict, scenario_stats: dict, model_choice: ModelChoice = "gemini"
) -> str:
    """
    Generate a plain-English comparison for baseline vs scenario.

    Args:
        baseline_stats: Summary stats from the original data forecast.
        scenario_stats: Summary stats from the modified scenario forecast.
        model_choice: 'gemini' or 'groq'.

    Returns:
        A 2-sentence scenario comparison string.
    """
    prompt = _build_scenario_prompt(baseline_stats, scenario_stats)
    if model_choice == "gemini":
        return _call_gemini(prompt)
    elif model_choice == "groq":
        return _call_groq(prompt)
    else:
        raise ValueError(f"Unsupported model_choice: {model_choice}")
