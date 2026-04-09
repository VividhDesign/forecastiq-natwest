"""
Unit tests for the forecasting service.
Run with: pytest tests/ -v

Uses a pure-Python Decomposable Time-Series model (Fourier + OLS + Bootstrap CI).
No external binary dependencies (no CmdStan / C++ build required).
"""

import pytest
import pandas as pd
from src.services.data_simulator import generate_synthetic_data
from src.services.forecasting import run_forecast, run_scenario_forecast



@pytest.fixture
def sample_data():
    """Generate 2 years of synthetic e-commerce data for testing."""
    result = generate_synthetic_data(
        context="ecommerce_sales",
        trend_type="stable",
        days=365,
        inject_anomalies=True,
    )
    return result["data"]


def test_data_simulator_returns_correct_length(sample_data):
    """Verify the simulator generates the expected number of data points."""
    assert len(sample_data) == 365


def test_data_simulator_has_required_keys(sample_data):
    """Each data point must have 'ds' and 'y' keys (Prophet format)."""
    for point in sample_data[:5]:
        assert "ds" in point
        assert "y" in point


def test_data_simulator_no_negative_values(sample_data):
    """All generated values must be positive (clipped by simulator)."""
    for point in sample_data:
        assert point["y"] > 0


def test_anomaly_injection():
    """Verify that anomalies are actually injected when requested."""
    result = generate_synthetic_data(
        context="ecommerce_sales",
        trend_type="stable",
        days=365,
        inject_anomalies=True,
    )
    assert len(result["injected_anomaly_dates"]) == 4  # 2 spikes + 2 drops


def test_no_anomaly_injection():
    """Verify no anomaly dates returned when flag is off."""
    result = generate_synthetic_data(
        context="ecommerce_sales",
        trend_type="stable",
        days=365,
        inject_anomalies=False,
    )
    assert len(result["injected_anomaly_dates"]) == 0


def test_forecast_returns_expected_keys(sample_data):
    """Verify forecast output contains all required sections."""
    result = run_forecast(sample_data, forecast_weeks=4)
    assert "historical_fit" in result
    assert "forecast" in result
    assert "anomalies" in result
    assert "summary_stats" in result


def test_forecast_future_rows_count(sample_data):
    """Verify forecast returns ~28 future rows for 4 weeks."""
    result = run_forecast(sample_data, forecast_weeks=4)
    future_count = len(result["forecast"])
    assert 26 <= future_count <= 30  # Allow minor buffer for Prophet internals


def test_forecast_detects_anomalies(sample_data):
    """Confirm Prophet detects anomalies in data with injected outliers."""
    result = run_forecast(sample_data, forecast_weeks=4)
    # With injected anomalies, must detect at least 1
    assert len(result["anomalies"]) >= 1


def test_forecast_summary_stats_structure(sample_data):
    """Summary stats must have all keys required by LLM prompts."""
    result = run_forecast(sample_data, forecast_weeks=4, context_label="Test Metric")
    stats = result["summary_stats"]
    required_keys = [
        "context_label", "forecast_weeks", "current_value",
        "forecast_end_value", "forecast_end_lower", "forecast_end_upper",
        "growth_pct_over_period", "peak_predicted_date",
        "anomaly_count", "anomaly_spike_count", "anomaly_drop_count",
    ]
    for key in required_keys:
        assert key in stats, f"Missing key: {key}"


def test_scenario_forecast_returns_both(sample_data):
    """Scenario endpoint must return both baseline and scenario forecasts."""
    result = run_scenario_forecast(
        data=sample_data,
        growth_multiplier=1.1,
        forecast_weeks=4,
    )
    assert "baseline_forecast" in result
    assert "scenario_forecast" in result
    assert "baseline_stats" in result
    assert "scenario_stats" in result


def test_scenario_growth_increases_forecast(sample_data):
    """A +20% growth multiplier must result in higher scenario end value."""
    result = run_scenario_forecast(
        data=sample_data,
        growth_multiplier=1.2,
        forecast_weeks=4,
    )
    baseline_end = result["baseline_stats"]["forecast_end_value"]
    scenario_end = result["scenario_stats"]["forecast_end_value"]
    assert scenario_end > baseline_end, "Scenario +20% should yield higher forecast"
