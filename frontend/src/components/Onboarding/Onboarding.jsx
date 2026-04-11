import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import './Onboarding.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

const CONTEXTS = [
  { value: 'ecommerce_sales', label: '🛒 E-commerce Sales',   desc: 'Daily revenue from an online store' },
  { value: 'server_load',     label: '💻 Cloud Server Load',  desc: 'Average daily CPU utilisation (%)' },
  { value: 'user_signups',    label: '👥 App User Signups',   desc: 'Daily new user registrations' },
  { value: 'support_tickets', label: '🎫 Support Tickets',    desc: 'Daily incoming customer queries' },
]

const TRENDS = [
  { value: 'aggressive_growth', label: '📈 Aggressive Growth' },
  { value: 'stable',            label: '➡️ Stable / Flat' },
  { value: 'declining',         label: '📉 Declining' },
]

/** Popular tickers shown as one-click chips in the Live Stock tab */
const POPULAR_TICKERS = [
  { symbol: 'NWG.L',  label: '🏦 NatWest (NWG.L)',   note: 'LSE · GBp' },
  { symbol: 'AAPL',   label: '🍎 Apple (AAPL)',        note: 'NASDAQ · USD' },
  { symbol: 'TSLA',   label: '⚡ Tesla (TSLA)',        note: 'NASDAQ · USD' },
  { symbol: 'MSFT',   label: '🪟 Microsoft (MSFT)',    note: 'NASDAQ · USD' },
  { symbol: '^FTSE',  label: '🇬🇧 FTSE 100 (^FTSE)',  note: 'LSE Index' },
  { symbol: '^GSPC',  label: '🇺🇸 S&P 500 (^GSPC)',   note: 'NYSE Index' },
]

export default function Onboarding({ onDataReady, theme, onThemeToggle }) {
  const [tab, setTab]               = useState('sandbox')  // 'sandbox' | 'stock' | 'upload'
  const [context, setContext]       = useState('ecommerce_sales')
  const [trend, setTrend]           = useState('aggressive_growth')
  const [injectAnomalies, setInjectAnomalies] = useState(true)
  const [ticker, setTicker]         = useState('NWG.L')
  const [period, setPeriod]         = useState('2y')
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState('')
  const [dragOver, setDragOver]     = useState(false)
  const fileRef = useRef()

  // ── Backend warm-up ping ────────────────────────────────────────────────────
  const [backendStatus, setBackendStatus] = useState('waking')
  useEffect(() => {
    let slowTimer
    const ping = async () => {
      try {
        slowTimer = setTimeout(() => setBackendStatus('slow'), 5000)
        await axios.get(`${API_BASE.replace('/api', '')}/ping`, { timeout: 60000 })
        clearTimeout(slowTimer)
        setBackendStatus('ready')
      } catch {
        clearTimeout(slowTimer)
        setBackendStatus('ready')
      }
    }
    ping()
    return () => clearTimeout(slowTimer)
  }, [])

  // ── Sandbox handler ─────────────────────────────────────────────────────────
  const handleGenerate = async () => {
    setLoading(true); setError('')
    try {
      const res = await axios.post(`${API_BASE}/simulate`, {
        context, trend_type: trend, inject_anomalies: injectAnomalies, days: 730,
      })
      onDataReady({ data: res.data.data, context_meta: res.data.context_meta, source: 'sandbox' })
    } catch {
      setError('Could not reach the backend. Please try again in a moment.')
    } finally { setLoading(false) }
  }

  // ── Live Stock handler ──────────────────────────────────────────────────────
  const handleFetchStock = async () => {
    const sym = ticker.trim().toUpperCase()
    if (!sym) { setError('Please enter a ticker symbol (e.g. NWG.L or AAPL).'); return }
    setLoading(true); setError('')
    try {
      const res = await axios.get(`${API_BASE}/fetch-stock`, {
        params: { ticker: sym, period },
      })
      onDataReady({
        data:         res.data.data,
        context_meta: res.data.context_meta,
        source:       'stock',
        ticker:       res.data.ticker,
      })
    } catch (e) {
      setError(e.response?.data?.detail || `Could not fetch data for "${sym}". Check the symbol and try again.`)
    } finally { setLoading(false) }
  }

  // ── CSV handler ─────────────────────────────────────────────────────────────
  const handleFileUpload = async (file) => {
    if (!file || !file.name.endsWith('.csv')) {
      setError('Please upload a valid .csv file with columns: ds, y'); return
    }
    setLoading(true); setError('')
    const formData = new FormData()
    formData.append('file', file)
    try {
      const res = await axios.post(`${API_BASE}/upload`, formData)
      onDataReady({ data: res.data.data, context_meta: null, source: 'upload' })
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to parse CSV. Ensure columns: ds (date), y (number).')
    } finally { setLoading(false) }
  }

  return (
    <div className="onboarding-page">
      {/* Theme toggle */}
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
            Upload your data, fetch live stock prices, or generate a realistic synthetic dataset
            to explore short-term forecasts, anomaly detection, and scenario modelling.
          </p>

          {/* Backend warm-up badge */}
          {backendStatus === 'waking' && (
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: '8px',
              marginTop: '12px', padding: '6px 14px', borderRadius: '20px',
              background: 'var(--glass-bg)', border: '1px solid var(--border)',
              fontSize: '0.78rem', color: 'var(--text-muted)'
            }}>
              <span className="spinner" style={{ width: '12px', height: '12px', borderWidth: '2px' }} />
              Connecting to server…
            </div>
          )}
          {backendStatus === 'slow' && (
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: '8px',
              marginTop: '12px', padding: '6px 14px', borderRadius: '20px',
              background: 'rgba(251,191,36,0.1)', border: '1px solid rgba(251,191,36,0.3)',
              fontSize: '0.78rem', color: 'var(--amber, #fbbf24)'
            }}>
              ⏳ Backend waking up (free tier cold start — takes ~30 s once, then stays fast)
            </div>
          )}
          {backendStatus === 'ready' && (
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: '6px',
              marginTop: '12px', padding: '6px 14px', borderRadius: '20px',
              background: 'rgba(34,197,94,0.1)', border: '1px solid rgba(34,197,94,0.3)',
              fontSize: '0.78rem', color: 'var(--green, #22c55e)'
            }}>
              ✓ Server ready
            </div>
          )}
        </div>

        {/* Main Card */}
        <div className="glass-card onboarding-card">

          {/* Tabs */}
          <div className="tabs">
            <button
              className={`tab ${tab === 'sandbox' ? 'active' : ''}`}
              onClick={() => { setTab('sandbox'); setError('') }}
            >
              🧪 Sample Data
            </button>
            <button
              className={`tab ${tab === 'stock' ? 'active' : ''}`}
              onClick={() => { setTab('stock'); setError('') }}
              style={{ position: 'relative' }}
            >
              📈 Live Stock
              <span style={{
                position: 'absolute', top: '-6px', right: '-4px',
                background: '#34d399', color: '#0f172a', fontSize: '0.6rem',
                fontWeight: 700, padding: '1px 5px', borderRadius: '8px',
                letterSpacing: '0.3px',
              }}>NEW</span>
            </button>
            <button
              className={`tab ${tab === 'upload' ? 'active' : ''}`}
              onClick={() => { setTab('upload'); setError('') }}
            >
              📄 Upload CSV
            </button>
          </div>

          <div className="divider" />

          {/* ── SANDBOX TAB ──────────────────────────────────────────────────── */}
          {tab === 'sandbox' && (
            <div className="sandbox-form fade-in">
              <p style={{ marginBottom: '20px', fontSize: '0.88rem' }}>
                Generate a realistic synthetic dataset on-the-fly. No real data needed — ideal for
                exploring the platform's forecasting and anomaly detection capabilities.
              </p>

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

              <div className="form-group">
                <div className="toggle-wrapper">
                  <label className="toggle">
                    <input type="checkbox" checked={injectAnomalies} onChange={e => setInjectAnomalies(e.target.checked)} />
                    <span className="toggle-slider" />
                  </label>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.88rem' }}>Inject Anomalies</div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                      Adds realistic spikes &amp; drops for anomaly detection demo
                    </div>
                  </div>
                </div>
              </div>

              {error && <div className="error-msg" style={{ marginBottom: '12px' }}>⚠️ {error}</div>}

              <button className="btn btn-primary generate-btn" onClick={handleGenerate} disabled={loading}>
                {loading ? <><span className="spinner" />Generating dataset...</> : <>Launch Dashboard →</>}
              </button>
            </div>
          )}

          {/* ── LIVE STOCK TAB ───────────────────────────────────────────────── */}
          {tab === 'stock' && (
            <div className="sandbox-form fade-in">
              <p style={{ marginBottom: '16px', fontSize: '0.88rem' }}>
                Fetch real historical stock prices from Yahoo Finance and run the full
                forecasting pipeline on live market data.{' '}
                <strong>NWG.L is NatWest Group's own stock</strong> — a powerful demo for the judges.
              </p>

              {/* Popular tickers */}
              <div className="form-group">
                <label className="label">Popular Tickers</label>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                  {POPULAR_TICKERS.map(t => (
                    <button
                      key={t.symbol}
                      onClick={() => setTicker(t.symbol)}
                      style={{
                        padding: '7px 14px', borderRadius: '20px', fontSize: '0.82rem',
                        border: `1px solid ${ticker === t.symbol ? '#34d399' : 'var(--border)'}`,
                        background: ticker === t.symbol ? 'rgba(52,211,153,0.12)' : 'var(--glass-bg)',
                        color: ticker === t.symbol ? '#34d399' : 'var(--text-secondary)',
                        cursor: 'pointer', fontWeight: ticker === t.symbol ? 700 : 400,
                        transition: 'all 0.2s',
                      }}
                    >
                      {t.label}
                      <span style={{ marginLeft: '5px', fontSize: '0.7rem', opacity: 0.6 }}>{t.note}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Custom ticker input */}
              <div className="form-group">
                <label className="label">Or Enter Any Ticker Symbol</label>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                  <input
                    type="text"
                    value={ticker}
                    onChange={e => setTicker(e.target.value.toUpperCase())}
                    onKeyDown={e => e.key === 'Enter' && handleFetchStock()}
                    placeholder="e.g. NWG.L, AAPL, TSLA, ^FTSE"
                    style={{
                      flex: 1, padding: '10px 14px', borderRadius: '10px',
                      border: '1px solid var(--border)', background: 'var(--glass-bg)',
                      color: 'var(--text-primary)', fontSize: '0.9rem',
                      fontFamily: 'var(--font-mono, monospace)',
                    }}
                  />
                  <select
                    value={period}
                    onChange={e => setPeriod(e.target.value)}
                    style={{
                      padding: '10px 12px', borderRadius: '10px',
                      border: '1px solid var(--border)', background: 'var(--glass-bg)',
                      color: 'var(--text-secondary)', fontSize: '0.85rem', cursor: 'pointer',
                    }}
                  >
                    <option value="1y">1 Year</option>
                    <option value="2y">2 Years</option>
                    <option value="5y">5 Years</option>
                  </select>
                </div>
              </div>

              {/* Privacy note */}
              <div style={{
                padding: '10px 14px', borderRadius: '10px', marginBottom: '16px',
                background: 'rgba(52,211,153,0.06)', border: '1px solid rgba(52,211,153,0.18)',
                fontSize: '0.78rem', color: 'var(--text-muted)',
              }}>
                🔒 <strong>Privacy:</strong> Stock prices are publicly available market data from Yahoo Finance.
                No user data is stored. Past price patterns do not guarantee future performance.
              </div>

              {error && <div className="error-msg" style={{ marginBottom: '12px' }}>⚠️ {error}</div>}

              <button className="btn btn-primary generate-btn" onClick={handleFetchStock} disabled={loading}
                style={{ background: 'linear-gradient(135deg, #34d399, #059669)' }}>
                {loading
                  ? <><span className="spinner" />Fetching {ticker} from Yahoo Finance...</>
                  : <>Load {ticker} Stock Data →</>
                }
              </button>
            </div>
          )}

          {/* ── CSV UPLOAD TAB ───────────────────────────────────────────────── */}
          {tab === 'upload' && (
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
                <div style={{ fontWeight: 600, marginBottom: '5px', fontSize: '0.95rem' }}>Drop your CSV here</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>or click to browse</div>
                <input ref={fileRef} type="file" accept=".csv" hidden onChange={e => handleFileUpload(e.target.files[0])} />
              </div>

              <div className="glass-card" style={{ marginTop: '14px', padding: '12px 16px' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  <strong>Required format:</strong> Two columns — <code>ds</code> (date) and{' '}
                  <code>y</code> (numeric). Minimum 30 rows.
                </div>
              </div>

              {error && <div className="error-msg" style={{ marginTop: '12px' }}>⚠️ {error}</div>}
              {loading && <div style={{ display: 'flex', justifyContent: 'center', marginTop: '20px' }}><span className="spinner" /></div>}
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
            '📈 Live Stock Data',
            '🤖 AI-Generated Insights',
          ].map(f => (
            <span key={f} className="feature-pill">{f}</span>
          ))}
        </div>

      </div>
    </div>
  )
}
