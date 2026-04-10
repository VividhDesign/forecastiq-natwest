import { useState, useRef } from 'react'
import axios from 'axios'
import './Onboarding.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

const CONTEXTS = [
  { value: 'ecommerce_sales', label: '🛒 E-commerce Sales',   desc: 'Daily revenue data from an online store' },
  { value: 'server_load',     label: '💻 Cloud Server Load',  desc: 'Average daily CPU utilisation (%)' },
  { value: 'user_signups',    label: '👥 App User Signups',   desc: 'Daily new user registrations' },
  { value: 'support_tickets', label: '🎫 Support Tickets',    desc: 'Daily incoming customer queries' },
]

const TRENDS = [
  { value: 'aggressive_growth', label: '📈 Aggressive Growth' },
  { value: 'stable',            label: '➡️ Stable / Flat' },
  { value: 'declining',         label: '📉 Declining' },
]

export default function Onboarding({ onDataReady, theme, onThemeToggle }) {
  const [tab, setTab] = useState('sandbox') // 'sandbox' | 'upload'
  const [context, setContext] = useState('ecommerce_sales')
  const [trend, setTrend] = useState('aggressive_growth')
  const [injectAnomalies, setInjectAnomalies] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const fileRef = useRef()

  const handleGenerate = async () => {
    setLoading(true)
    setError('')
    try {
      const res = await axios.post(`${API_BASE}/simulate`, {
        context,
        trend_type: trend,
        inject_anomalies: injectAnomalies,
        days: 730,
      })
      onDataReady({ data: res.data.data, context_meta: res.data.context_meta, source: 'sandbox' })
    } catch (e) {
      setError('Could not reach the backend. Please try again in a moment.')
    } finally {
      setLoading(false)
    }
  }

  const handleFileUpload = async (file) => {
    if (!file || !file.name.endsWith('.csv')) {
      setError('Please upload a valid .csv file with columns: ds, y')
      return
    }
    setLoading(true)
    setError('')
    const formData = new FormData()
    formData.append('file', file)
    try {
      const res = await axios.post(`${API_BASE}/upload`, formData)
      onDataReady({ data: res.data.data, context_meta: null, source: 'upload' })
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to parse CSV. Ensure columns: ds (date), y (number).')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="onboarding-page">
      {/* Floating theme toggle — top right */}
      <button
        className="theme-toggle"
        onClick={onThemeToggle}
        title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
        aria-label="Toggle theme"
        style={{ position: 'fixed', top: '16px', right: '16px', zIndex: 10 }}
      >
        {theme === 'dark' ? '☀' : '☽'}
      </button>

      <div className="onboarding-container fade-in">

        {/* Header */}
        <div className="onboarding-header">
          <div className="onboarding-logo">
            <span className="logo-icon">📊</span>
            <span className="logo-text gradient-text">ForecastIQ</span>
          </div>
          <div className="onboarding-tagline">NatWest · Code for Purpose 2026</div>
          <h1 className="onboarding-title">Predictive Forecasting Tool</h1>
          <p className="onboarding-subtitle">
            Upload your time-series data or generate a realistic dataset to explore short-term
            forecasts, anomaly detection, and scenario modelling.
          </p>
        </div>

        {/* Main Card */}
        <div className="glass-card onboarding-card">
          {/* Tabs */}
          <div className="tabs">
            <button
              className={`tab ${tab === 'sandbox' ? 'active' : ''}`}
              onClick={() => setTab('sandbox')}
            >
              Generate Sample Data
            </button>
            <button
              className={`tab ${tab === 'upload' ? 'active' : ''}`}
              onClick={() => setTab('upload')}
            >
              Upload CSV
            </button>
          </div>

          <div className="divider" />

          {tab === 'sandbox' ? (
            <div className="sandbox-form fade-in">
              <p style={{ marginBottom: '20px', fontSize: '0.88rem' }}>
                Generate a realistic synthetic dataset on-the-fly. No real data needed — ideal for
                exploring the platform's forecasting and anomaly detection capabilities.
              </p>

              {/* Context Selection */}
              <div className="form-group">
                <label className="label">Business Context</label>
                <div className="context-grid">
                  {CONTEXTS.map(c => (
                    <div
                      key={c.value}
                      className={`context-card ${context === c.value ? 'selected' : ''}`}
                      onClick={() => setContext(c.value)}
                    >
                      <div className="context-label">{c.label}</div>
                      <div className="context-desc">{c.desc}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Trend Selection */}
              <div className="form-group">
                <label className="label">Trend Direction</label>
                <div className="trend-options">
                  {TRENDS.map(t => (
                    <button
                      key={t.value}
                      className={`trend-btn ${trend === t.value ? 'active' : ''}`}
                      onClick={() => setTrend(t.value)}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Inject Anomalies */}
              <div className="form-group">
                <div className="toggle-wrapper">
                  <label className="toggle">
                    <input
                      type="checkbox"
                      checked={injectAnomalies}
                      onChange={e => setInjectAnomalies(e.target.checked)}
                    />
                    <span className="toggle-slider" />
                  </label>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.88rem' }}>Inject Anomalies</div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                      Adds realistic spikes & drops for anomaly detection demo
                    </div>
                  </div>
                </div>
              </div>

              {error && <div className="error-msg" style={{ marginBottom: '12px' }}>⚠️ {error}</div>}

              <button
                className="btn btn-primary generate-btn"
                onClick={handleGenerate}
                disabled={loading}
              >
                {loading
                  ? <><span className="spinner" />Generating dataset...</>
                  : <>Launch Dashboard →</>
                }
              </button>
            </div>
          ) : (
            <div className="upload-form fade-in">
              <p style={{ marginBottom: '20px', fontSize: '0.88rem' }}>
                Upload a CSV file with two columns: <code>ds</code> (date, YYYY-MM-DD) and{' '}
                <code>y</code> (numeric value). Minimum 30 rows required.
              </p>

              <div
                className={`drop-zone ${dragOver ? 'drag-over' : ''}`}
                onClick={() => fileRef.current.click()}
                onDragOver={e => { e.preventDefault(); setDragOver(true) }}
                onDragLeave={() => setDragOver(false)}
                onDrop={e => { e.preventDefault(); setDragOver(false); handleFileUpload(e.dataTransfer.files[0]) }}
              >
                <div style={{ fontSize: '2rem', marginBottom: '10px' }}>📄</div>
                <div style={{ fontWeight: 600, marginBottom: '5px', fontSize: '0.95rem' }}>
                  Drop your CSV here
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>or click to browse</div>
                <input
                  ref={fileRef}
                  type="file"
                  accept=".csv"
                  hidden
                  onChange={e => handleFileUpload(e.target.files[0])}
                />
              </div>

              <div className="glass-card" style={{ marginTop: '14px', padding: '12px 16px' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  <strong>Required format:</strong> Two columns — <code>ds</code> (date) and{' '}
                  <code>y</code> (numeric). Minimum 30 rows.
                </div>
              </div>

              {error && <div className="error-msg" style={{ marginTop: '12px' }}>⚠️ {error}</div>}
              {loading && (
                <div style={{ display: 'flex', justifyContent: 'center', marginTop: '20px' }}>
                  <span className="spinner" />
                </div>
              )}
            </div>
          )}
        </div>

        {/* Feature Pills */}
        <div className="feature-pills">
          {[
            '📐 Time-Series Decomposition',
            '📊 Bootstrap Confidence Intervals',
            '🚨 Anomaly Detection',
            '🎰 Scenario Analysis',
            '🤖 AI-Generated Insights',
          ].map(f => (
            <span key={f} className="feature-pill">{f}</span>
          ))}
        </div>

      </div>
    </div>
  )
}
