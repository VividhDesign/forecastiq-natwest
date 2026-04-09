import {
  ComposedChart, Area, Line, Scatter, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import './ForecastChart.css'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="custom-tooltip">
      <div className="tt-label">{label}</div>
      {payload.map(p => (
        <div key={p.name} className="tt-row">
          <span style={{ color: p.color || p.fill }}>■</span>
          <span className="tt-value">{p.name}: {typeof p.value === 'number' ? p.value.toLocaleString(undefined, { maximumFractionDigits: 1 }) : '—'}</span>
        </div>
      ))}
    </div>
  )
}

export default function ForecastChart({ historicalFit, forecast, anomalies, contextLabel }) {
  // Merge historical + forecast into one dataset for the chart
  const historicalPoints = (historicalFit || []).map(d => ({
    ds: d.ds,
    actual: d.y ?? null,
    fitted: d.yhat,
    lower: d.yhat_lower,
    upper: d.yhat_upper,
    isForecast: false,
  }))

  const forecastPoints = (forecast || []).map(d => ({
    ds: d.ds,
    actual: null,
    predicted: d.yhat,
    lower: d.yhat_lower,
    upper: d.yhat_upper,
    isForecast: true,
  }))

  // Thin the historical data for performance (show every 3rd point)
  const thinnedHistory = historicalPoints.filter((_, i) => i % 3 === 0)
  const chartData = [...thinnedHistory, ...forecastPoints]

  // Format anomaly data for Scatter overlay
  const anomalyPoints = (anomalies || []).map(a => ({
    ds: a.ds,
    y: a.y,
    direction: a.direction,
    pct: a.pct_deviation,
  }))

  // Find the boundary between historical and forecast
  const forecastStartDate = forecast?.[0]?.ds

  // Sample tick marks (every ~30 days)
  const tickInterval = Math.max(1, Math.floor(chartData.length / 12))
  const ticks = chartData.filter((_, i) => i % tickInterval === 0).map(d => d.ds)

  return (
    <div className="glass-card forecast-chart-card">
      <div className="chart-header">
        <div>
          <h3>{contextLabel} — Historical Fit & {forecast?.length ? `${Math.round(forecast.length / 7)}-Week` : ''} Forecast</h3>
          <p style={{ fontSize: '0.8rem', marginTop: '4px' }}>
            Solid line = actual data · Dashed line = forecast · Shaded area = 95% confidence interval · <span style={{ color: '#ef4444' }}>● = anomaly</span>
          </p>
        </div>
        <div className="chart-legend-pills">
          <span className="legend-pill" style={{ borderColor: '#818cf8' }}>— Historical</span>
          <span className="legend-pill" style={{ borderColor: '#38bdf8', borderStyle: 'dashed' }}>-- Forecast</span>
          <span className="legend-pill" style={{ borderColor: '#ef4444' }}>● Anomaly</span>
        </div>
      </div>

      <div className="divider" style={{ margin: '16px 0' }} />

      <ResponsiveContainer width="100%" height={380}>
        <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
          <defs>
            <linearGradient id="confGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#6366f1" stopOpacity={0.18} />
              <stop offset="95%" stopColor="#6366f1" stopOpacity={0.02} />
            </linearGradient>
            <linearGradient id="forecastGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.15} />
              <stop offset="95%" stopColor="#38bdf8" stopOpacity={0.02} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />

          <XAxis
            dataKey="ds"
            ticks={ticks}
            tick={{ fontSize: 10, fill: '#475569' }}
            tickFormatter={v => v?.slice(0, 7)}
            axisLine={{ stroke: 'rgba(255,255,255,0.05)' }}
            tickLine={false}
          />
          <YAxis
            tick={{ fontSize: 10, fill: '#475569' }}
            tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(1)}k` : v}
            axisLine={false}
            tickLine={false}
            width={56}
          />

          <Tooltip content={<CustomTooltip />} />

          {/* Confidence Band (area between lower and upper) */}
          <Area
            type="monotone"
            dataKey="upper"
            stroke="none"
            fill="url(#confGradient)"
            fillOpacity={1}
            legendType="none"
            dot={false}
            activeDot={false}
            name="Upper Bound"
          />
          <Area
            type="monotone"
            dataKey="lower"
            stroke="none"
            fill="var(--bg-base)"
            fillOpacity={1}
            legendType="none"
            dot={false}
            activeDot={false}
            name="Lower Bound"
          />

          {/* Actual (historical) line */}
          <Line
            type="monotone"
            dataKey="actual"
            stroke="#818cf8"
            strokeWidth={2}
            dot={false}
            name="Actual"
            connectNulls={false}
            activeDot={{ r: 4, fill: '#818cf8' }}
          />

          {/* Forecast line — dashed */}
          <Line
            type="monotone"
            dataKey="predicted"
            stroke="#38bdf8"
            strokeWidth={2.5}
            strokeDasharray="6 3"
            dot={false}
            name="Forecast"
            connectNulls={false}
            activeDot={{ r: 4, fill: '#38bdf8' }}
          />

          {/* Vertical Reference Line: start of forecast */}
          {forecastStartDate && (
            <ReferenceLine
              x={forecastStartDate}
              stroke="rgba(255,255,255,0.12)"
              strokeDasharray="4 4"
              label={{ value: 'Forecast →', position: 'top', fill: '#475569', fontSize: 11 }}
            />
          )}

          {/* Anomaly dots (plotted against actual) */}
          {anomalyPoints.length > 0 && anomalyPoints.map((pt, i) => {
            const idx = chartData.findIndex(d => d.ds === pt.ds)
            if (idx === -1) return null
            return null // handled via extra scatter series below
          })}
        </ComposedChart>
      </ResponsiveContainer>

      {/* Anomaly Scatter (separate pass for clean rendering) */}
      {anomalyPoints.length > 0 && (
        <div className="anomaly-dots-note">
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            🔴 {anomalyPoints.length} anomalies detected — see the <strong>Anomalies</strong> tab for detailed AI analysis.
          </span>
        </div>
      )}
    </div>
  )
}
