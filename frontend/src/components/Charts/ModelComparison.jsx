import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Area,
} from 'recharts'
import InsightCard from '../Shared/InsightCard'
import './ModelComparison.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

/**
 * Custom tooltip for the comparison chart.
 * Shows values from both models side by side.
 */
const ComparisonTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="custom-tooltip">
      <div className="tt-label">{label}</div>
      {payload.map(p => (
        <div key={p.name} className="tt-row">
          <span style={{ color: p.color || p.fill }}>■</span>
          <span className="tt-value">
            {p.name}: {typeof p.value === 'number'
              ? p.value.toLocaleString(undefined, { maximumFractionDigits: 1 })
              : '—'}
          </span>
        </div>
      ))}
    </div>
  )
}

/**
 * ModelComparison — Side-by-side comparison of Classical vs CNN forecasting models.
 *
 * Directly addresses Hackathon Learning Outcome #1:
 * "When more advanced models are justified."
 *
 * Shows:
 *   - Both forecast lines on the same chart
 *   - Accuracy metrics (MAE, RMSE, MAPE) for each model
 *   - Winner determination and AI-generated explanation
 */
export default function ModelComparison({ data, forecastWeeks, contextLabel, selectedModel }) {
  const [comparison, setComparison] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const fetchComparison = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const res = await axios.post(`${API_BASE}/model-comparison`, {
        data,
        forecast_weeks: forecastWeeks,
        context_label: contextLabel,
        model_choice: selectedModel,
      })
      setComparison(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Model comparison failed. Check backend connection.')
    } finally {
      setLoading(false)
    }
  }, [data, forecastWeeks, contextLabel, selectedModel])

  useEffect(() => {
    if (data?.length >= 60) {
      fetchComparison()
    }
  }, []) // Run once on mount, user can refresh manually

  // Build chart data merging both forecasts
  const buildChartData = () => {
    if (!comparison) return []

    const classicalForecast = comparison.classical?.forecast || []
    const cnnForecast = comparison.cnn?.forecast || []
    const naive = comparison.naive_baseline || []

    const maxLen = Math.max(classicalForecast.length, cnnForecast.length)
    const chartData = []

    for (let i = 0; i < maxLen; i++) {
      const entry = {
        ds: classicalForecast[i]?.ds || cnnForecast[i]?.ds,
      }
      if (classicalForecast[i]) {
        entry.classical = classicalForecast[i].yhat
        entry.classical_lower = classicalForecast[i].yhat_lower
        entry.classical_upper = classicalForecast[i].yhat_upper
      }
      if (cnnForecast[i]) {
        entry.cnn = cnnForecast[i].yhat
        entry.cnn_lower = cnnForecast[i].yhat_lower
        entry.cnn_upper = cnnForecast[i].yhat_upper
      }
      if (naive[i]) {
        entry.naive = naive[i].yhat_naive
      }
      chartData.push(entry)
    }

    return chartData
  }

  const renderMetricRow = (label, classical, cnn, lowerIsBetter = true) => {
    const cVal = classical ?? Infinity
    const dVal = cnn ?? Infinity
    const classicalBetter = lowerIsBetter ? cVal <= dVal : cVal >= dVal
    const cnnBetter = lowerIsBetter ? dVal < cVal : dVal > cVal

    return (
      <div className="metric-row">
        <span className="label">{label}</span>
        <span className={`value ${classicalBetter ? 'best' : ''}`}>
          {classical != null ? classical.toLocaleString(undefined, { maximumFractionDigits: 2 }) : 'N/A'}
        </span>
        <span className={`value ${cnnBetter ? 'best' : ''}`}>
          {cnn != null ? cnn.toLocaleString(undefined, { maximumFractionDigits: 2 }) : 'N/A'}
        </span>
      </div>
    )
  }

  // ── Loading State ──
  if (loading) {
    return (
      <div className="glass-card comparison-loading">
        <div className="pulse-icon">🧠</div>
        <p>Running both models on your dataset...</p>
        <p className="sub">Classical (OLS+Fourier) + 1D CNN (Deep Learning) — this may take 10-15 seconds</p>
        <div className="spinner" style={{ width: '32px', height: '32px', borderWidth: '3px' }} />
      </div>
    )
  }

  // ── Error State ──
  if (error) {
    return (
      <div className="glass-card" style={{ padding: '32px', textAlign: 'center' }}>
        <div style={{ fontSize: '2rem', marginBottom: '12px' }}>⚠️</div>
        <p style={{ color: 'var(--red)' }}>{error}</p>
        <button className="btn btn-ghost" style={{ marginTop: '16px' }} onClick={fetchComparison}>
          Retry Comparison
        </button>
      </div>
    )
  }

  // ── Not enough data ──
  if (!data || data.length < 60) {
    return (
      <div className="glass-card" style={{ padding: '32px', textAlign: 'center' }}>
        <div style={{ fontSize: '2rem', marginBottom: '12px' }}>📊</div>
        <p style={{ color: 'var(--text-secondary)' }}>
          Model comparison requires at least 60 data points for a meaningful holdout evaluation.
          Your dataset has {data?.length || 0} points.
        </p>
      </div>
    )
  }

  if (!comparison) return null

  const classicalMetrics = comparison.classical?.accuracy_metrics || {}
  const cnnMetrics = comparison.cnn?.accuracy_metrics || {}
  const classicalStats = comparison.classical?.summary_stats || {}
  const cnnStats = comparison.cnn?.summary_stats || {}
  const winner = comparison.winner
  const chartData = buildChartData()

  const winnerName = winner === 'classical' ? 'Classical (OLS + Fourier)' : '1D CNN (Deep Learning)'
  const winnerEmoji = winner === 'classical' ? '📐' : '🧠'

  return (
    <div className="comparison-container fade-in">
      {/* Winner Banner */}
      <div className="glass-card winner-banner">
        <div className="trophy">🏆</div>
        <div className="winner-info">
          <h3>{winnerEmoji} {winnerName} wins on this dataset</h3>
          <p>
            Based on lowest Mean Absolute Error (MAE) on a {classicalMetrics.holdout_size || '20%'}-point holdout validation set.
            Lower error = more accurate predictions.
          </p>
        </div>
      </div>

      {/* Metrics Comparison Cards */}
      <div className="metrics-grid">
        {/* Classical Model Card */}
        <div className={`glass-card model-metrics-card ${winner === 'classical' ? 'is-winner' : ''}`}>
          <div className="card-header">
            <h4>📐 Classical (OLS + Fourier)</h4>
            {winner === 'classical' && <span className="winner-badge">✓ Winner</span>}
          </div>
          <div className="metric-rows">
            <div className="metric-row">
              <span className="label">MAE</span>
              <span className={`value ${winner === 'classical' ? 'best' : ''}`}>
                {classicalMetrics.mae?.toLocaleString() ?? 'N/A'}
              </span>
            </div>
            <div className="metric-row">
              <span className="label">RMSE</span>
              <span className={`value ${winner === 'classical' ? 'best' : ''}`}>
                {classicalMetrics.rmse?.toLocaleString() ?? 'N/A'}
              </span>
            </div>
            <div className="metric-row">
              <span className="label">MAPE</span>
              <span className={`value ${winner === 'classical' ? 'best' : ''}`}>
                {classicalMetrics.mape != null ? `${classicalMetrics.mape}%` : 'N/A'}
              </span>
            </div>
            <div style={{ height: '1px', background: 'var(--border)', margin: '4px 0' }} />
            <div className="metric-row">
              <span className="label">Forecast End</span>
              <span className="value">{classicalStats.forecast_end_value?.toLocaleString()}</span>
            </div>
            <div className="metric-row">
              <span className="label">Growth</span>
              <span className="value" style={{ color: classicalStats.growth_pct_over_period >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {classicalStats.growth_pct_over_period >= 0 ? '+' : ''}{classicalStats.growth_pct_over_period}%
              </span>
            </div>
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '12px' }}>
            Strengths: Interpretable, fast, works with small datasets, mathematically optimal for linear trends + periodic seasonality.
          </p>
        </div>

        {/* CNN Model Card */}
        <div className={`glass-card model-metrics-card ${winner === 'cnn' ? 'is-winner' : ''}`}>
          <div className="card-header">
            <h4>🧠 1D CNN (Deep Learning)</h4>
            {winner === 'cnn' && <span className="winner-badge">✓ Winner</span>}
          </div>
          <div className="metric-rows">
            <div className="metric-row">
              <span className="label">MAE</span>
              <span className={`value ${winner === 'cnn' ? 'best' : ''}`}>
                {cnnMetrics.mae?.toLocaleString() ?? 'N/A'}
              </span>
            </div>
            <div className="metric-row">
              <span className="label">RMSE</span>
              <span className={`value ${winner === 'cnn' ? 'best' : ''}`}>
                {cnnMetrics.rmse?.toLocaleString() ?? 'N/A'}
              </span>
            </div>
            <div className="metric-row">
              <span className="label">MAPE</span>
              <span className={`value ${winner === 'cnn' ? 'best' : ''}`}>
                {cnnMetrics.mape != null ? `${cnnMetrics.mape}%` : 'N/A'}
              </span>
            </div>
            <div style={{ height: '1px', background: 'var(--border)', margin: '4px 0' }} />
            <div className="metric-row">
              <span className="label">Forecast End</span>
              <span className="value">{cnnStats.forecast_end_value?.toLocaleString() ?? 'N/A'}</span>
            </div>
            <div className="metric-row">
              <span className="label">Growth</span>
              <span className="value" style={{ color: (cnnStats.growth_pct_over_period ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {(cnnStats.growth_pct_over_period ?? 0) >= 0 ? '+' : ''}{cnnStats.growth_pct_over_period ?? 'N/A'}%
              </span>
            </div>
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '12px' }}>
            Strengths: Captures non-linear patterns, regime changes, and complex interactions. Better with large datasets (1000+ rows).
          </p>
        </div>
      </div>

      {/* Side-by-Side Forecast Chart */}
      <div className="glass-card comparison-chart-container">
        <div className="chart-header">
          <div>
            <h3>Forecast Comparison — Classical vs CNN</h3>
            <p style={{ fontSize: '0.8rem', marginTop: '4px', color: 'var(--text-muted)' }}>
              Both models forecasting the next {forecastWeeks} week{forecastWeeks > 1 ? 's' : ''}. Shaded areas = 95% confidence intervals.
            </p>
          </div>
          <div className="chart-legend-pills">
            <span className="legend-pill" style={{ borderColor: '#f59e0b' }}>— Classical</span>
            <span className="legend-pill" style={{ borderColor: '#8b5cf6' }}>— CNN</span>
            <span className="legend-pill" style={{ borderColor: '#94a3b8', borderStyle: 'dotted' }}>··· Naive</span>
          </div>
        </div>

        <div className="divider" style={{ margin: '16px 0' }} />

        <ResponsiveContainer width="100%" height={350}>
          <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
            <defs>
              <linearGradient id="classicalBand" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.12} />
                <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.02} />
              </linearGradient>
              <linearGradient id="cnnBand" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.12} />
                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.02} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis
              dataKey="ds"
              tick={{ fontSize: 10, fill: '#475569' }}
              tickFormatter={v => v?.slice(5)}
              axisLine={{ stroke: 'rgba(255,255,255,0.05)' }}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 10, fill: '#475569' }}
              tickFormatter={v => v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v}
              axisLine={false}
              tickLine={false}
              width={56}
            />
            <Tooltip content={<ComparisonTooltip />} />

            {/* Classical Confidence Band */}
            <Area type="monotone" dataKey="classical_upper" stroke="none" fill="url(#classicalBand)" fillOpacity={1} dot={false} name="Classical Upper" legendType="none" />
            <Area type="monotone" dataKey="classical_lower" stroke="none" fill="var(--bg-base)" fillOpacity={1} dot={false} name="Classical Lower" legendType="none" />

            {/* CNN Confidence Band */}
            <Area type="monotone" dataKey="cnn_upper" stroke="none" fill="url(#cnnBand)" fillOpacity={1} dot={false} name="CNN Upper" legendType="none" />
            <Area type="monotone" dataKey="cnn_lower" stroke="none" fill="var(--bg-base)" fillOpacity={1} dot={false} name="CNN Lower" legendType="none" />

            {/* Classical Forecast Line */}
            <Line
              type="monotone"
              dataKey="classical"
              stroke="#f59e0b"
              strokeWidth={2.5}
              dot={false}
              name="Classical (OLS+Fourier)"
              activeDot={{ r: 4, fill: '#f59e0b' }}
            />

            {/* CNN Forecast Line */}
            <Line
              type="monotone"
              dataKey="cnn"
              stroke="#8b5cf6"
              strokeWidth={2.5}
              dot={false}
              name="1D CNN"
              activeDot={{ r: 4, fill: '#8b5cf6' }}
            />

            {/* Naive Baseline */}
            <Line
              type="monotone"
              dataKey="naive"
              stroke="#94a3b8"
              strokeWidth={1.5}
              strokeDasharray="3 5"
              dot={false}
              name="Naive Baseline"
              activeDot={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* AI Insight */}
      {comparison.comparison_insight && (
        <InsightCard
          icon="🧠"
          title="AI Model Comparison Analysis"
          model={selectedModel}
          text={comparison.comparison_insight}
        />
      )}

      {/* Methodology Note */}
      <div className="glass-card" style={{ padding: '16px 20px' }}>
        <h4 style={{ fontSize: '0.85rem', marginBottom: '8px', color: 'var(--text-secondary)' }}>
          📋 How This Comparison Works
        </h4>
        <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
          Both models are evaluated on the same holdout set (last 20% of your data). The data is hidden from both models during training,
          then each model's predictions are compared against actual values using three standard error metrics:
          <strong> MAE</strong> (average absolute error),
          <strong> RMSE</strong> (penalizes large errors more),
          <strong> MAPE</strong> (percentage-based, scale-independent).
          The model with the lowest MAE is declared the winner for this specific dataset.
          <em> Different datasets may yield different winners</em> — this is exactly what "knowing when advanced models are justified" means.
        </p>
      </div>
    </div>
  )
}
