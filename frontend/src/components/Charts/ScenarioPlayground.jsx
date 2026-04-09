import { useState, useCallback, useEffect } from 'react'
import axios from 'axios'
import {
  ComposedChart, Line, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import InsightCard from '../Shared/InsightCard'
import './ScenarioPlayground.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

export default function ScenarioPlayground({ data, contextLabel, selectedModel }) {
  const [growthMultiplier, setGrowthMultiplier] = useState(1.1)
  const [seasonalityStrength, setSeasonalityStrength] = useState(1.0)
  const [forecastWeeks, setForecastWeeks] = useState(4)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const runScenario = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const res = await axios.post(`${API_BASE}/scenario`, {
        data,
        growth_multiplier: growthMultiplier,
        seasonality_strength: seasonalityStrength,
        forecast_weeks: forecastWeeks,
        model_choice: selectedModel,
      })
      setResult(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Scenario run failed.')
    } finally {
      setLoading(false)
    }
  }, [data, growthMultiplier, seasonalityStrength, forecastWeeks, selectedModel])

  // Auto-run on mount
  useEffect(() => { runScenario() }, [])

  // Merge baseline + scenario data for chart
  const chartData = result ? result.baseline_forecast.map((b, i) => ({
    ds: b.ds,
    baseline: b.yhat,
    scenario: result.scenario_forecast[i]?.yhat,
    scenario_lower: result.scenario_forecast[i]?.yhat_lower,
    scenario_upper: result.scenario_forecast[i]?.yhat_upper,
  })) : []

  const growthPct = Math.round((growthMultiplier - 1) * 100)

  return (
    <div className="scenario-layout">
      {/* Controls */}
      <div className="glass-card scenario-controls">
        <h3>🎰 Scenario Playground</h3>
        <p style={{ fontSize: '0.85rem', marginTop: '4px', marginBottom: '20px' }}>
          Adjust the sliders to simulate "What-if" scenarios. The model will re-forecast based on your inputs.
        </p>

        <div className="grid-2" style={{ gap: '24px' }}>
          <div>
            <label className="label">Growth Multiplier: <strong style={{ color: 'var(--accent-light)' }}>{growthPct >= 0 ? '+' : ''}{growthPct}%</strong></label>
            <input
              type="range" min={0.5} max={2.0} step={0.05}
              value={growthMultiplier}
              onChange={e => setGrowthMultiplier(Number(e.target.value))}
            />
            <div className="range-labels">
              <span>−50%</span><span style={{ color: 'var(--text-secondary)' }}>Baseline</span><span>+100%</span>
            </div>
          </div>

          <div>
            <label className="label">Seasonality Strength: <strong style={{ color: 'var(--accent-light)' }}>{seasonalityStrength.toFixed(1)}×</strong></label>
            <input
              type="range" min={0.1} max={3.0} step={0.1}
              value={seasonalityStrength}
              onChange={e => setSeasonalityStrength(Number(e.target.value))}
            />
            <div className="range-labels">
              <span>0.1×</span><span style={{ color: 'var(--text-secondary)' }}>Normal</span><span>3.0×</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3" style={{ marginTop: '20px', flexWrap: 'wrap' }}>
          <div className="flex items-center gap-2">
            <label className="label" style={{ margin: 0 }}>Forecast Weeks:</label>
            <select className="input" style={{ width: 'auto', padding: '7px 10px' }} value={forecastWeeks} onChange={e => setForecastWeeks(Number(e.target.value))}>
              {[1,2,3,4,5,6].map(w => <option key={w} value={w}>{w}w</option>)}
            </select>
          </div>
          <button className="btn btn-primary" onClick={runScenario} disabled={loading} style={{ flexShrink: 0 }}>
            {loading ? <><span className="spinner" /> Running...</> : '▶ Run Scenario'}
          </button>
        </div>
      </div>

      {error && <div style={{ color: 'var(--red)', fontSize: '0.85rem', padding: '10px', background: 'rgba(239,68,68,0.08)', borderRadius: 'var(--radius-sm)', marginTop: '12px' }}>⚠️ {error}</div>}

      {/* Comparison Chart */}
      {result && !loading && (
        <div className="fade-in">
          <div className="glass-card" style={{ padding: '24px', marginTop: '16px' }}>
            <div className="chart-stats-comparison">
              <div className="stat-card">
                <div className="stat-label">Baseline End Value</div>
                <div className="stat-value">{result.baseline_stats.forecast_end_value?.toLocaleString()}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                  Growth: {result.baseline_stats.growth_pct_over_period}%
                </div>
              </div>
              <div className="stat-card" style={{ borderColor: 'rgba(99,102,241,0.4)' }}>
                <div className="stat-label">Scenario End Value</div>
                <div className="stat-value" style={{ color: 'var(--accent-light)' }}>{result.scenario_stats.forecast_end_value?.toLocaleString()}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                  Growth: {result.scenario_stats.growth_pct_over_period}%
                </div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Difference</div>
                <div className="stat-value" style={{ color: result.scenario_stats.forecast_end_value > result.baseline_stats.forecast_end_value ? 'var(--green)' : 'var(--red)' }}>
                  {result.scenario_stats.forecast_end_value > result.baseline_stats.forecast_end_value ? '+' : ''}
                  {(result.scenario_stats.forecast_end_value - result.baseline_stats.forecast_end_value).toLocaleString(undefined, { maximumFractionDigits: 1 })}
                </div>
              </div>
            </div>

            <div className="divider" />

            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
                <defs>
                  <linearGradient id="scenarioArea" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6366f1" stopOpacity={0.15} />
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="ds" tick={{ fontSize: 10, fill: '#475569' }} tickFormatter={v => v?.slice(5)} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: '#475569' }} tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(1)}k` : v} axisLine={false} tickLine={false} width={56} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: '0.82rem' }} />
                <Legend wrapperStyle={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }} />
                <Area type="monotone" dataKey="scenario_upper" stroke="none" fill="url(#scenarioArea)" legendType="none" dot={false} activeDot={false} name="Scenario Band" />
                <Area type="monotone" dataKey="scenario_lower" stroke="none" fill="var(--bg-base)" legendType="none" dot={false} activeDot={false} />
                <Line type="monotone" dataKey="baseline" stroke="#94a3b8" strokeWidth={2} strokeDasharray="5 3" dot={false} name="Baseline" />
                <Line type="monotone" dataKey="scenario" stroke="#818cf8" strokeWidth={2.5} dot={false} name={`Scenario (${growthPct >= 0 ? '+' : ''}${growthPct}%)`} activeDot={{ r: 4 }} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* AI Comparison Insight */}
          <div style={{ marginTop: '16px' }}>
            <InsightCard
              icon="🧠"
              title="AI Scenario Analysis"
              text={result.scenario_insight}
              model={selectedModel}
            />
          </div>
        </div>
      )}

      {loading && (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '200px', gap: '16px' }}>
          <div className="spinner" style={{ width: '32px', height: '32px', borderWidth: '3px' }} />
          <span style={{ color: 'var(--text-secondary)' }}>Running scenario forecast...</span>
        </div>
      )}
    </div>
  )
}
