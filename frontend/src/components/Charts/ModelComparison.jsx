import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Area,
} from 'recharts'
import InsightCard from '../Shared/InsightCard'
import './ModelComparison.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

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
 * ModelComparison — Classical (OLS+Fourier) vs Naive Baseline.
 *
 * Directly addresses Hackathon Learning Outcome #2:
 * "Why simple comparisons matter when testing ideas"
 *   - Baseline forecasts provide a sanity check
 *   - Accuracy can be misleading without a reference
 *   - Simple models often outperform overly complex ones
 *
 * Both signals are computed analytically — no ML training, instant response.
 */
export default function ModelComparison({ data, forecastWeeks, contextLabel, selectedModel }) {
  const [comparison, setComparison] = useState(null)
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState('')

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
  }, []) // run once on mount

  // ── Model config ───────────────────────────────────────────────────────────
  const MODELS = [
    {
      key:   'classical',
      label: 'OLS + Fourier (Classical)',
      emoji: '📐',
      color: '#f59e0b',
      desc:  'Analytical decomposition: Linear trend via OLS regression + Fourier series for seasonality. Instant — no training required. Fully interpretable and mathematically exact.',
    },
    {
      key:   'naive',
      label: 'Naive Baseline (28-day Avg)',
      emoji: '📏',
      color: '#94a3b8',
      desc:  'The simplest possible forecast — a flat rolling average of the last 28 days. Used as a sanity check: if our model cannot beat this, it is not learning anything useful.',
    },
  ]

  // ── Build chart data ───────────────────────────────────────────────────────
  const buildChartData = () => {
    if (!comparison) return []
    const cf    = comparison.classical?.forecast || []
    const naive = comparison.naive_baseline       || []
    const maxLen = Math.max(cf.length, naive.length)
    return Array.from({ length: maxLen }, (_, i) => ({
      ds:              cf[i]?.ds || naive[i]?.ds,
      classical:       cf[i]?.yhat,
      classical_lower: cf[i]?.yhat_lower,
      classical_upper: cf[i]?.yhat_upper,
      naive:           naive[i]?.yhat_naive,
    }))
  }

  // ── Loading ────────────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="glass-card comparison-loading">
        <div className="pulse-icon">📐</div>
        <p>Running Classical vs Naive Baseline comparison…</p>
        <p className="sub">Pure analytics — no training needed, almost instant</p>
        <div className="spinner" style={{ width: '32px', height: '32px', borderWidth: '3px' }} />
      </div>
    )
  }

  // ── Error ──────────────────────────────────────────────────────────────────
  if (error) {
    return (
      <div className="glass-card" style={{ padding: '32px', textAlign: 'center' }}>
        <div style={{ fontSize: '2rem', marginBottom: '12px' }}>⚠️</div>
        <p style={{ color: 'var(--red)' }}>{error}</p>
        <button className="btn btn-ghost" style={{ marginTop: '16px' }} onClick={fetchComparison}>
          Retry
        </button>
      </div>
    )
  }

  // ── Not enough data ────────────────────────────────────────────────────────
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

  const winner      = comparison.winner
  const winnerModel = MODELS.find(m => m.key === winner) || MODELS[0]
  const chartData   = buildChartData()

  return (
    <div className="comparison-container fade-in">

      {/* ── Winner Banner ───────────────────────────────────────────────────── */}
      <div className="glass-card winner-banner">
        <div className="trophy">🏆</div>
        <div className="winner-info">
          <h3>{winnerModel.emoji} {winnerModel.label} wins on this dataset</h3>
          <p>
            Measured by lowest MAE on a held-out 20% test set.{' '}
            {winner === 'classical'
              ? 'Our model beats the naive baseline — it is genuinely learning patterns from the data.'
              : 'The naive baseline is competitive here — a signal to investigate the data further.'}
          </p>
        </div>
      </div>

      {/* ── Two Model Cards ─────────────────────────────────────────────────── */}
      <div className="metrics-grid">
        {MODELS.map(m => {
          const metrics  = comparison[m.key]?.accuracy_metrics || {}
          const stats    = comparison[m.key]?.summary_stats    || {}
          const isWinner = winner === m.key

          return (
            <div
              key={m.key}
              className={`glass-card model-metrics-card ${isWinner ? 'is-winner' : ''}`}
              style={{ borderColor: isWinner ? m.color : undefined,
                       boxShadow: isWinner ? `0 0 24px ${m.color}18` : undefined }}
            >
              <div className="card-header">
                <h4 style={{ color: isWinner ? m.color : undefined }}>
                  {m.emoji} {m.label}
                </h4>
                {isWinner && (
                  <span className="winner-badge" style={{ background: m.color, color: '#0f172a' }}>
                    ✓ Winner
                  </span>
                )}
              </div>

              <div className="metric-rows">
                {[
                  { label: 'MAE',  val: metrics.mae  },
                  { label: 'RMSE', val: metrics.rmse },
                  { label: 'MAPE', val: metrics.mape != null ? `${metrics.mape}%` : null },
                ].map(({ label, val }) => (
                  <div key={label} className="metric-row">
                    <span className="label">{label}</span>
                    <span className={`value ${isWinner ? 'best' : ''}`}>
                      {val != null ? String(val) : 'N/A'}
                    </span>
                  </div>
                ))}

                {stats.forecast_end_value != null && (
                  <>
                    <div style={{ height: '1px', background: 'var(--border)', margin: '4px 0' }} />
                    <div className="metric-row">
                      <span className="label">Forecast End</span>
                      <span className="value">{stats.forecast_end_value?.toLocaleString() ?? 'N/A'}</span>
                    </div>
                    <div className="metric-row">
                      <span className="label">Growth</span>
                      <span className="value" style={{
                        color: (stats.growth_pct_over_period ?? 0) >= 0 ? 'var(--green)' : 'var(--red)',
                      }}>
                        {(stats.growth_pct_over_period ?? 0) >= 0 ? '+' : ''}
                        {stats.growth_pct_over_period ?? 'N/A'}%
                      </span>
                    </div>
                  </>
                )}
              </div>

              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '14px', lineHeight: 1.55 }}>
                {m.desc}
              </p>
            </div>
          )
        })}
      </div>

      {/* ── Forecast Chart ──────────────────────────────────────────────────── */}
      <div className="glass-card comparison-chart-container">
        <div className="chart-header">
          <div>
            <h3>Forecast Comparison — Classical vs Naive Baseline</h3>
            <p style={{ fontSize: '0.8rem', marginTop: '4px', color: 'var(--text-muted)' }}>
              Both models forecasting the next {forecastWeeks} week{forecastWeeks > 1 ? 's' : ''}.
              Shaded area = 95% confidence interval (Classical model only).
            </p>
          </div>
          <div className="chart-legend-pills">
            <span className="legend-pill" style={{ borderColor: '#f59e0b' }}>— 📐 Classical</span>
            <span className="legend-pill" style={{ borderColor: '#94a3b8', borderStyle: 'dotted' }}>··· 📏 Naive</span>
          </div>
        </div>

        <div className="divider" style={{ margin: '16px 0' }} />

        <ResponsiveContainer width="100%" height={360}>
          <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
            <defs>
              <linearGradient id="classicalBand" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#f59e0b" stopOpacity={0.15} />
                <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.02} />
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

            {/* Classical CI band */}
            <Area type="monotone" dataKey="classical_upper" stroke="none" fill="url(#classicalBand)"
              fillOpacity={1} dot={false} legendType="none" name="Classical Upper" />
            <Area type="monotone" dataKey="classical_lower" stroke="none" fill="var(--bg-base)"
              fillOpacity={1} dot={false} legendType="none" name="Classical Lower" />

            {/* Classical forecast line */}
            <Line type="monotone" dataKey="classical" stroke="#f59e0b"
              strokeWidth={2.5} dot={false} name="Classical (OLS+Fourier)"
              activeDot={{ r: 4, fill: '#f59e0b' }} />

            {/* Naive baseline */}
            <Line type="monotone" dataKey="naive" stroke="#94a3b8"
              strokeWidth={1.8} strokeDasharray="4 5"
              dot={false} name="Naive Baseline" activeDot={false} />
          </ComposedChart>
        </ResponsiveContainer>

        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '12px' }}>
          <button className="btn btn-ghost" onClick={fetchComparison}>
            ↻ Re-run Comparison
          </button>
        </div>
      </div>

      {/* ── AI Insight ────────────────────────────────────────────────────── */}
      {comparison.comparison_insight && (
        <InsightCard
          icon="📐"
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
          The <strong>Classical model</strong> decomposes the series into{' '}
          <strong>trend + seasonality</strong> using OLS regression and Fourier series — the same
          mathematics used by Facebook Prophet, computed analytically in milliseconds.
          <br /><br />
          The <strong>Naive Baseline</strong> simply predicts the 28-day rolling average flat into the future.
          It requires zero computation. If our model cannot beat this,
          it is not learning anything meaningful from the data.
          <br /><br />
          Both are evaluated on the same hidden <strong>20% holdout set</strong>.
          Winner = lowest <strong>MAE</strong>.{' '}
          <em>This directly demonstrates Hackathon Learning Outcome #2: "Why simple comparisons matter."</em>
        </p>
      </div>
    </div>
  )
}
