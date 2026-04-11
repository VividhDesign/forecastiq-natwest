import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Area,
} from 'recharts'
import InsightCard from '../Shared/InsightCard'
import './ModelComparison.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

/** Custom tooltip — shows all three model values side by side. */
const ComparisonTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="custom-tooltip">
      <div className="tt-label">{label}</div>
      {payload.map(p => (
        <div key={p.name} className="tt-row">
          <span style={{ color: p.color || p.fill }}>■</span>
          <span className="tt-value">
            {p.name}:{' '}
            {typeof p.value === 'number'
              ? p.value.toLocaleString(undefined, { maximumFractionDigits: 1 })
              : '—'}
          </span>
        </div>
      ))}
    </div>
  )
}

/**
 * ModelComparison — 3-way head-to-head: Classical vs 1D CNN vs N-BEATS.
 *
 * Directly addresses Hackathon Learning Outcome #1:
 * "When more advanced models are justified" — demonstrated empirically on the
 * user's own data, not just stated as theory.
 *
 * Shows:
 *   - All three forecast lines on the same chart with CI bands
 *   - Accuracy metrics (MAE, RMSE, MAPE) for each model on a 20% holdout
 *   - Winner badge + AI-generated explanation
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
    if (data?.length >= 60) fetchComparison()
  }, []) // Run once on mount; user can refresh manually

  // ── Build merged chart data ───────────────────────────────────────────────
  const buildChartData = () => {
    if (!comparison) return []
    const cf = comparison.classical?.forecast || []
    const cnn = comparison.cnn?.forecast || []
    const nb = comparison.nbeats?.forecast || []
    const naive = comparison.naive_baseline || []

    const maxLen = Math.max(cf.length, cnn.length, nb.length)
    return Array.from({ length: maxLen }, (_, i) => ({
      ds:               cf[i]?.ds || cnn[i]?.ds || nb[i]?.ds,
      classical:        cf[i]?.yhat,
      classical_lower:  cf[i]?.yhat_lower,
      classical_upper:  cf[i]?.yhat_upper,
      cnn:              cnn[i]?.yhat,
      cnn_lower:        cnn[i]?.yhat_lower,
      cnn_upper:        cnn[i]?.yhat_upper,
      nbeats:           nb[i]?.yhat,
      nbeats_lower:     nb[i]?.yhat_lower,
      nbeats_upper:     nb[i]?.yhat_upper,
      naive:            naive[i]?.yhat_naive,
    }))
  }

  // ── Model config (colour + labels) ───────────────────────────────────────
  const MODELS = [
    {
      key: 'classical',
      label: 'Classical (OLS + Fourier)',
      emoji: '📐',
      color: '#f59e0b',
      desc: 'Strengths: Fully interpretable, instant (no training), optimal for linear trends + periodic seasonality. Best when data follows clean, regular patterns.',
    },
    {
      key: 'cnn',
      label: '1D CNN (Deep Learning)',
      emoji: '🔷',
      color: '#8b5cf6',
      desc: 'Strengths: Learns local temporal patterns (spikes, weekly shapes) via convolutional filters. Better than Classical on noisy or irregular data.',
    },
    {
      key: 'nbeats',
      label: 'N-BEATS (Interpretable DL)',
      emoji: '🧠',
      color: '#34d399',
      desc: 'Strengths: State-of-the-art (ICLR 2020). Decomposes into trend + seasonality stacks like Classical — but learns the decomposition from data. Best of both worlds.',
    },
  ]

  // ── Loading ───────────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="glass-card comparison-loading">
        <div className="pulse-icon">🧠</div>
        <p>Running all 3 models on your dataset…</p>
        <p className="sub">
          Classical · 1D CNN · N-BEATS — this takes 10–20 seconds on first run
        </p>
        <div className="spinner" style={{ width: '32px', height: '32px', borderWidth: '3px' }} />
      </div>
    )
  }

  // ── Error ─────────────────────────────────────────────────────────────────
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

  // ── Not enough data ───────────────────────────────────────────────────────
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

  const winner = comparison.winner
  const winnerModel = MODELS.find(m => m.key === winner) || MODELS[0]
  const chartData = buildChartData()

  return (
    <div className="comparison-container fade-in">

      {/* ── Winner Banner ─────────────────────────────────────────────────── */}
      <div className="glass-card winner-banner">
        <div className="trophy">🏆</div>
        <div className="winner-info">
          <h3>{winnerModel.emoji} {winnerModel.label} wins on this dataset</h3>
          <p>
            Based on lowest Mean Absolute Error (MAE) measured on a 20% held-out test set.
            {' '}The winner varies by dataset — which is precisely what
            {' '}<em>"knowing when advanced models are justified"</em> means.
          </p>
        </div>
      </div>

      {/* ── Three Model Metric Cards ──────────────────────────────────────── */}
      <div className="metrics-grid metrics-grid-3">
        {MODELS.map(m => {
          const metrics = comparison[m.key]?.accuracy_metrics || {}
          const stats   = comparison[m.key]?.summary_stats || {}
          const isWinner = winner === m.key

          return (
            <div
              key={m.key}
              className={`glass-card model-metrics-card ${isWinner ? 'is-winner' : ''}`}
              style={{ borderColor: isWinner ? m.color : undefined }}
            >
              <div className="card-header">
                <h4 style={{ color: isWinner ? m.color : undefined }}>
                  {m.emoji} {m.label}
                </h4>
                {isWinner && (
                  <span className="winner-badge" style={{ background: m.color }}>
                    ✓ Winner
                  </span>
                )}
              </div>

              <div className="metric-rows">
                {[
                  { label: 'MAE',  val: metrics.mae },
                  { label: 'RMSE', val: metrics.rmse },
                  { label: 'MAPE', val: metrics.mape != null ? `${metrics.mape}%` : null },
                ].map(({ label, val }) => (
                  <div key={label} className="metric-row">
                    <span className="label">{label}</span>
                    <span className={`value ${isWinner ? 'best' : ''}`}>
                      {val != null ? val.toLocaleString?.() ?? val : 'N/A'}
                    </span>
                  </div>
                ))}

                <div style={{ height: '1px', background: 'var(--border)', margin: '4px 0' }} />

                <div className="metric-row">
                  <span className="label">Forecast End</span>
                  <span className="value">{stats.forecast_end_value?.toLocaleString() ?? 'N/A'}</span>
                </div>
                <div className="metric-row">
                  <span className="label">Growth</span>
                  <span
                    className="value"
                    style={{
                      color: (stats.growth_pct_over_period ?? 0) >= 0
                        ? 'var(--green)' : 'var(--red)',
                    }}
                  >
                    {(stats.growth_pct_over_period ?? 0) >= 0 ? '+' : ''}
                    {stats.growth_pct_over_period ?? 'N/A'}%
                  </span>
                </div>
              </div>

              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '12px' }}>
                {m.desc}
              </p>
            </div>
          )
        })}
      </div>

      {/* ── 3-Way Forecast Chart ─────────────────────────────────────────── */}
      <div className="glass-card comparison-chart-container">
        <div className="chart-header">
          <div>
            <h3>3-Way Forecast Comparison</h3>
            <p style={{ fontSize: '0.8rem', marginTop: '4px', color: 'var(--text-muted)' }}>
              All three models forecasting the next {forecastWeeks} week{forecastWeeks > 1 ? 's' : ''}.
              Shaded areas = 95% confidence intervals.
            </p>
          </div>
          <div className="chart-legend-pills">
            {MODELS.map(m => (
              <span key={m.key} className="legend-pill" style={{ borderColor: m.color }}>
                — {m.emoji} {m.key === 'classical' ? 'Classical' : m.key === 'cnn' ? 'CNN' : 'N-BEATS'}
              </span>
            ))}
            <span className="legend-pill" style={{ borderColor: '#94a3b8', borderStyle: 'dotted' }}>
              ··· Naive
            </span>
          </div>
        </div>

        <div className="divider" style={{ margin: '16px 0' }} />

        <ResponsiveContainer width="100%" height={370}>
          <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
            <defs>
              {MODELS.map(m => (
                <linearGradient key={m.key} id={`band_${m.key}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor={m.color} stopOpacity={0.1} />
                  <stop offset="95%" stopColor={m.color} stopOpacity={0.01} />
                </linearGradient>
              ))}
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

            {/* CI bands for each model */}
            {MODELS.map(m => (
              <>
                <Area key={`${m.key}_upper`} type="monotone" dataKey={`${m.key}_upper`}
                  stroke="none" fill={`url(#band_${m.key})`} fillOpacity={1}
                  dot={false} legendType="none" name={`${m.label} Upper`} />
                <Area key={`${m.key}_lower`} type="monotone" dataKey={`${m.key}_lower`}
                  stroke="none" fill="var(--bg-base)" fillOpacity={1}
                  dot={false} legendType="none" name={`${m.label} Lower`} />
              </>
            ))}

            {/* Forecast lines */}
            {MODELS.map(m => (
              <Line
                key={m.key}
                type="monotone"
                dataKey={m.key}
                stroke={m.color}
                strokeWidth={winner === m.key ? 2.8 : 1.8}
                strokeOpacity={winner === m.key ? 1 : 0.7}
                dot={false}
                name={m.label}
                activeDot={{ r: 4, fill: m.color }}
              />
            ))}

            {/* Naive baseline */}
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

        {/* Re-run button */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '12px' }}>
          <button className="btn btn-ghost" onClick={fetchComparison}>
            ↻ Re-run Comparison
          </button>
        </div>
      </div>

      {/* ── AI Insight ───────────────────────────────────────────────────── */}
      {comparison.comparison_insight && (
        <InsightCard
          icon="🧠"
          title="AI Model Comparison Analysis"
          model={selectedModel}
          text={comparison.comparison_insight}
        />
      )}

      {/* ── Methodology Note ─────────────────────────────────────────────── */}
      <div className="glass-card" style={{ padding: '16px 20px' }}>
        <h4 style={{ fontSize: '0.85rem', marginBottom: '8px', color: 'var(--text-secondary)' }}>
          📋 How This Comparison Works
        </h4>
        <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
          All three models train on the same data and are evaluated on the same hidden
          20% holdout set. Winner = lowest <strong>MAE</strong> (Mean Absolute Error).
          {' '}<strong>RMSE</strong> penalises large errors more heavily.
          {' '}<strong>MAPE</strong> is scale-independent (useful across different metrics).
          <br />
          The winner changes by dataset — on clean seasonal data Classical usually wins;
          on noisy, non-linear series N-BEATS or CNN may win.{' '}
          <em>This is exactly what "knowing when more advanced models are justified" means.</em>
        </p>
      </div>
    </div>
  )
}
