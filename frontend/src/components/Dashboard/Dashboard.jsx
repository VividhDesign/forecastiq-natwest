import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import NavBar from '../Shared/NavBar'
import ForecastChart from '../Charts/ForecastChart'
import AnomalyPanel from '../Charts/AnomalyPanel'
import ScenarioPlayground from '../Charts/ScenarioPlayground'
import DataExplorer from '../Charts/DataExplorer'
import InsightCard from '../Shared/InsightCard'
import './Dashboard.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

export default function Dashboard({ datasetPayload, selectedModel, onModelChange, onReset }) {
  const [activeTab, setActiveTab] = useState('forecast')
  const [forecastData, setForecastData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [forecastWeeks, setForecastWeeks] = useState(4)

  const contextLabel = datasetPayload?.context_meta?.label || 'Metric'

  const fetchForecast = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const res = await axios.post(`${API_BASE}/forecast`, {
        data: datasetPayload.data,
        forecast_weeks: forecastWeeks,
        context_label: contextLabel,
        model_choice: selectedModel,
      })
      setForecastData(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Forecast failed. Check backend connection.')
    } finally {
      setLoading(false)
    }
  }, [datasetPayload, forecastWeeks, selectedModel, contextLabel])

  useEffect(() => {
    fetchForecast()
  }, [fetchForecast])

  const stats = forecastData?.summary_stats

  return (
    <div className="dashboard-layout">
      <NavBar
        contextLabel={contextLabel}
        contextDesc={datasetPayload?.context_meta?.description || ''}
        selectedModel={selectedModel}
        onModelChange={onModelChange}
        onReset={onReset}
      />

      <div className="dashboard-content">
        {/* Stat Cards */}
        {stats && (
          <div className="grid-4 fade-in" style={{ marginBottom: '24px' }}>
            <div className="stat-card">
              <div className="stat-label">Current Value</div>
              <div className="stat-value">{stats.current_value?.toLocaleString()}</div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '4px' }}>{contextLabel}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">{stats.forecast_weeks}-Week Forecast</div>
              <div className="stat-value">{stats.forecast_end_value?.toLocaleString()}</div>
              <div style={{ fontSize: '0.78rem', color: stats.growth_pct_over_period >= 0 ? 'var(--green)' : 'var(--red)', marginTop: '4px' }}>
                {stats.growth_pct_over_period >= 0 ? '▲' : '▼'} {Math.abs(stats.growth_pct_over_period)}%
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Confidence Range</div>
              <div className="stat-value" style={{ fontSize: '1.2rem' }}>
                {stats.forecast_end_lower?.toLocaleString()} – {stats.forecast_end_upper?.toLocaleString()}
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '4px' }}>95% probability band</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Anomalies Detected</div>
              <div className="stat-value" style={{ color: stats.anomaly_count > 0 ? 'var(--red)' : 'var(--green)' }}>
                {stats.anomaly_count}
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                {stats.anomaly_spike_count} spikes · {stats.anomaly_drop_count} drops
              </div>
            </div>
          </div>
        )}

        {/* Week Selector + Tabs */}
        <div className="flex justify-between items-center" style={{ marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
          <div className="tabs" style={{ flex: '1', minWidth: '340px' }}>
            <button className={`tab ${activeTab === 'forecast' ? 'active' : ''}`} onClick={() => setActiveTab('forecast')}>📈 Forecast</button>
            <button className={`tab ${activeTab === 'anomalies' ? 'active' : ''}`} onClick={() => setActiveTab('anomalies')}>🚨 Anomalies</button>
            <button className={`tab ${activeTab === 'scenario' ? 'active' : ''}`} onClick={() => setActiveTab('scenario')}>🎰 Scenario</button>
            <button className={`tab ${activeTab === 'data' ? 'active' : ''}`} onClick={() => setActiveTab('data')}>📊 Raw Data</button>
          </div>
          <div className="flex items-center gap-2" style={{ whiteSpace: 'nowrap' }}>
            <label className="label" style={{ margin: 0 }}>Forecast weeks:</label>
            <select
              className="input"
              style={{ width: 'auto', padding: '8px 12px' }}
              value={forecastWeeks}
              onChange={e => setForecastWeeks(Number(e.target.value))}
            >
              {[1,2,3,4,5,6].map(w => <option key={w} value={w}>{w} week{w > 1 ? 's' : ''}</option>)}
            </select>
          </div>
        </div>

        {/* Main Content Area */}
        {loading ? (
          <div className="loading-state">
            <div className="spinner" style={{ width: '40px', height: '40px', borderWidth: '4px' }} />
            <p>Running forecast model, this may take a moment...</p>
          </div>
        ) : error ? (
          <div className="glass-card" style={{ padding: '32px', textAlign: 'center' }}>
            <div style={{ fontSize: '2rem', marginBottom: '12px' }}>⚠️</div>
            <p style={{ color: 'var(--red)' }}>{error}</p>
            <button className="btn btn-ghost" style={{ marginTop: '16px' }} onClick={fetchForecast}>Retry</button>
          </div>
        ) : (
          <div className="fade-in">
            {activeTab === 'forecast' && (
              <div className="grid-2" style={{ gap: '20px' }}>
                <div style={{ gridColumn: '1 / -1' }}>
                  <ForecastChart
                    historicalFit={forecastData.historical_fit}
                    forecast={forecastData.forecast}
                    anomalies={forecastData.anomalies}
                    contextLabel={contextLabel}
                  />
                </div>
                <InsightCard
                  icon="🔮"
                  title="AI Forecast Insight"
                  model={selectedModel}
                  text={forecastData.forecast_insight}
                />
                <div className="glass-card">
                  <h3 style={{ marginBottom: '14px' }}>Peak Prediction</h3>
                  <p style={{ fontSize: '0.88rem' }}>
                    Highest forecasted value expected around <strong style={{ color: 'var(--accent-light)' }}>{stats?.peak_predicted_date}</strong>.
                    Plan ahead for this period to capitalize on the opportunity or prepare for the load.
                  </p>
                </div>
              </div>
            )}

            {activeTab === 'anomalies' && (
              <AnomalyPanel
                anomalies={forecastData.anomalies}
                contextLabel={contextLabel}
                selectedModel={selectedModel}
              />
            )}

            {activeTab === 'scenario' && (
              <ScenarioPlayground
                data={datasetPayload.data}
                contextLabel={contextLabel}
                selectedModel={selectedModel}
              />
            )}

            {activeTab === 'data' && (
              <DataExplorer
                data={datasetPayload.data}
                anomalies={forecastData?.anomalies}
                contextLabel={contextLabel}
              />
            )}
          </div>
        )}
      </div>
    </div>
  )
}
