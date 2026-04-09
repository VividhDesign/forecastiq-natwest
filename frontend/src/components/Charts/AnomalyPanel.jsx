import { useState } from 'react'
import axios from 'axios'
import InsightCard from '../Shared/InsightCard'
import './AnomalyPanel.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

export default function AnomalyPanel({ anomalies, contextLabel, selectedModel }) {
  const [selectedAnomaly, setSelectedAnomaly] = useState(null)
  const [insight, setInsight] = useState('')
  const [insightLoading, setInsightLoading] = useState(false)

  const handleSelectAnomaly = async (anomaly) => {
    setSelectedAnomaly(anomaly)
    setInsight('')
    setInsightLoading(true)
    try {
      const res = await axios.post(`${API_BASE}/anomaly-insight`, {
        anomaly: {
          ds: anomaly.ds,
          y: anomaly.y,
          yhat: anomaly.yhat,
          direction: anomaly.direction,
          pct_deviation: anomaly.pct_deviation,
        },
        context_label: contextLabel,
        model_choice: selectedModel,
      })
      setInsight(res.data.insight)
    } catch {
      setInsight('[AI insight unavailable. Check API key setup.]')
    } finally {
      setInsightLoading(false)
    }
  }

  if (!anomalies?.length) {
    return (
      <div className="glass-card anomaly-empty">
        <div style={{ fontSize: '2rem', marginBottom: '12px' }}>✅</div>
        <h3>No Anomalies Detected</h3>
        <p>All historical data points fall within the 95% confidence band. The model sees no unexpected behavior.</p>
      </div>
    )
  }

  return (
    <div className="anomaly-layout">
      {/* Summary */}
      <div className="grid-2" style={{ marginBottom: '20px' }}>
        <div className="stat-card">
          <div className="stat-label">Total Anomalies</div>
          <div className="stat-value" style={{ color: 'var(--red)' }}>{anomalies.length}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Breakdown</div>
          <div className="flex gap-3 items-center" style={{ marginTop: '8px' }}>
            <span className="badge badge-red">▲ {anomalies.filter(a => a.direction === 'spike').length} spikes</span>
            <span className="badge badge-yellow">▼ {anomalies.filter(a => a.direction === 'drop').length} drops</span>
          </div>
        </div>
      </div>

      <div className="grid-2">
        {/* Anomaly Table */}
        <div className="glass-card" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)' }}>
            <h3>Detected Anomaly Events</h3>
            <p style={{ fontSize: '0.8rem', marginTop: '4px' }}>Click a row for an AI explanation</p>
          </div>
          <div className="anomaly-table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Actual</th>
                  <th>Expected</th>
                  <th>Deviation</th>
                  <th>Type</th>
                </tr>
              </thead>
              <tbody>
                {anomalies.map((a, i) => (
                  <tr
                    key={i}
                    className={`anomaly-row ${selectedAnomaly?.ds === a.ds ? 'selected' : ''}`}
                    onClick={() => handleSelectAnomaly(a)}
                  >
                    <td>{a.ds}</td>
                    <td>{Number(a.y).toLocaleString(undefined, { maximumFractionDigits: 1 })}</td>
                    <td style={{ color: 'var(--text-muted)' }}>{Number(a.yhat).toLocaleString(undefined, { maximumFractionDigits: 1 })}</td>
                    <td style={{ color: a.direction === 'spike' ? 'var(--red)' : 'var(--yellow)', fontWeight: 600 }}>
                      {a.pct_deviation > 0 ? '+' : ''}{a.pct_deviation}%
                    </td>
                    <td>
                      <span className={`badge ${a.direction === 'spike' ? 'badge-red' : 'badge-yellow'}`}>
                        {a.direction === 'spike' ? '▲ Spike' : '▼ Drop'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* AI Insight Pane */}
        {selectedAnomaly ? (
          <div className="anomaly-detail fade-in">
            <div className="glass-card" style={{ marginBottom: '16px', padding: '16px 20px' }}>
              <h3>📅 {selectedAnomaly.ds}</h3>
              <div className="flex gap-3 items-center" style={{ marginTop: '10px', flexWrap: 'wrap' }}>
                <span className={`badge ${selectedAnomaly.direction === 'spike' ? 'badge-red' : 'badge-yellow'}`}>
                  {selectedAnomaly.direction === 'spike' ? '▲ Spike' : '▼ Drop'}
                </span>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  Actual: <strong style={{ color: 'var(--text-primary)' }}>{Number(selectedAnomaly.y).toLocaleString(undefined, { maximumFractionDigits: 1 })}</strong>
                </span>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  vs Expected: <strong style={{ color: 'var(--text-primary)' }}>{Number(selectedAnomaly.yhat).toLocaleString(undefined, { maximumFractionDigits: 1 })}</strong>
                </span>
                <span style={{ fontSize: '0.85rem', fontWeight: 700, color: selectedAnomaly.direction === 'spike' ? 'var(--red)' : 'var(--yellow)' }}>
                  {selectedAnomaly.pct_deviation > 0 ? '+' : ''}{selectedAnomaly.pct_deviation}%
                </span>
              </div>
            </div>
            <InsightCard
              icon="🤖"
              title="AI Anomaly Explanation"
              text={insight}
              model={selectedModel}
              loading={insightLoading}
            />
          </div>
        ) : (
          <div className="glass-card anomaly-select-prompt">
            <span style={{ fontSize: '2rem' }}>👆</span>
            <p style={{ marginTop: '12px' }}>Select an anomaly from the table to get an AI-powered explanation.</p>
          </div>
        )}
      </div>
    </div>
  )
}
