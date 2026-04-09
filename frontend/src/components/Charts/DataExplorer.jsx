import { useState, useMemo } from 'react'
import './DataExplorer.css'

/**
 * DataExplorer — Premium raw data viewer for ForecastIQ
 * Shows: summary stats, searchable & paginated table, anomaly highlights, CSV export
 */
export default function DataExplorer({ data, anomalies, contextLabel }) {
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState('all')   // 'all' | 'spike' | 'drop' | 'normal'
  const [sortDir, setSortDir] = useState('asc')  // 'asc' | 'desc'
  const [page, setPage] = useState(1)
  const PAGE_SIZE = 20

  // Build anomaly lookup map
  const anomalyMap = useMemo(() => {
    const map = {}
    ;(anomalies || []).forEach(a => { map[a.ds] = a })
    return map
  }, [anomalies])

  // Compute summary stats
  const stats = useMemo(() => {
    if (!data?.length) return null
    const values = data.map(d => d.y)
    const mean = values.reduce((a, b) => a + b, 0) / values.length
    const sorted = [...values].sort((a, b) => a - b)
    const median = sorted.length % 2 === 0
      ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
      : sorted[Math.floor(sorted.length / 2)]
    const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length
    const std = Math.sqrt(variance)
    return {
      count: values.length,
      dateFrom: data[0]?.ds,
      dateTo: data[data.length - 1]?.ds,
      min: Math.min(...values),
      max: Math.max(...values),
      mean: mean.toFixed(1),
      median: median.toFixed(1),
      std: std.toFixed(1),
      anomalyCount: anomalies?.length || 0,
      spikes: (anomalies || []).filter(a => a.direction === 'spike').length,
      drops: (anomalies || []).filter(a => a.direction === 'drop').length,
    }
  }, [data, anomalies])

  // Filter + search + sort
  const filtered = useMemo(() => {
    let rows = (data || []).map(d => ({
      ...d,
      anomaly: anomalyMap[d.ds] || null,
    }))

    if (filter === 'spike') rows = rows.filter(r => r.anomaly?.direction === 'spike')
    else if (filter === 'drop') rows = rows.filter(r => r.anomaly?.direction === 'drop')
    else if (filter === 'normal') rows = rows.filter(r => !r.anomaly)

    if (search) rows = rows.filter(r => r.ds.includes(search))

    if (sortDir === 'desc') rows = [...rows].reverse()

    return rows
  }, [data, anomalyMap, filter, search, sortDir])

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const paginated = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  const handleFilterChange = (f) => { setFilter(f); setPage(1) }
  const handleSearch = (e) => { setSearch(e.target.value); setPage(1) }

  // Export to CSV
  const handleExport = () => {
    const header = 'date,value,anomaly_type,deviation_pct\n'
    const rows = filtered.map(r => {
      const a = r.anomaly
      return `${r.ds},${r.y},${a ? a.direction : ''},${a ? (a.pct_deviation || '').toFixed(1) : ''}`
    }).join('\n')
    const blob = new Blob([header + rows], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `forecastiq_data_${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (!data?.length) return (
    <div className="glass-card de-empty">
      <div style={{ fontSize: '2rem', marginBottom: '12px' }}>📭</div>
      <p>No data loaded yet.</p>
    </div>
  )

  return (
    <div className="de-wrapper fade-in">

      {/* ── Summary Stats Bar ── */}
      <div className="de-stats-grid">
        <div className="de-stat-card">
          <div className="de-stat-icon">📅</div>
          <div className="de-stat-body">
            <div className="de-stat-label">Date Range</div>
            <div className="de-stat-value">{stats?.dateFrom} → {stats?.dateTo}</div>
          </div>
        </div>
        <div className="de-stat-card">
          <div className="de-stat-icon">📊</div>
          <div className="de-stat-body">
            <div className="de-stat-label">Total Data Points</div>
            <div className="de-stat-value">{stats?.count?.toLocaleString()}</div>
          </div>
        </div>
        <div className="de-stat-card">
          <div className="de-stat-icon">📉</div>
          <div className="de-stat-body">
            <div className="de-stat-label">Min / Max</div>
            <div className="de-stat-value">{Number(stats?.min).toLocaleString()} / {Number(stats?.max).toLocaleString()}</div>
          </div>
        </div>
        <div className="de-stat-card">
          <div className="de-stat-icon">📐</div>
          <div className="de-stat-body">
            <div className="de-stat-label">Mean / Median</div>
            <div className="de-stat-value">{Number(stats?.mean).toLocaleString()} / {Number(stats?.median).toLocaleString()}</div>
          </div>
        </div>
        <div className="de-stat-card">
          <div className="de-stat-icon">〰️</div>
          <div className="de-stat-body">
            <div className="de-stat-label">Std Deviation</div>
            <div className="de-stat-value">± {Number(stats?.std).toLocaleString()}</div>
          </div>
        </div>
        <div className="de-stat-card" style={{ borderColor: stats?.anomalyCount > 0 ? 'rgba(239,68,68,0.3)' : undefined }}>
          <div className="de-stat-icon">🔴</div>
          <div className="de-stat-body">
            <div className="de-stat-label">Anomalies</div>
            <div className="de-stat-value" style={{ color: stats?.anomalyCount > 0 ? 'var(--red)' : 'var(--green)' }}>
              {stats?.anomalyCount} &nbsp;
              <span style={{ fontSize: '0.78rem', fontWeight: 400, color: 'var(--text-muted)' }}>
                ({stats?.spikes} spikes · {stats?.drops} drops)
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ── Controls ── */}
      <div className="de-controls glass-card">
        <div className="de-search-wrap">
          <span className="de-search-icon">🔍</span>
          <input
            id="de-search"
            className="de-search-input"
            type="text"
            placeholder="Search by date (e.g. 2024-03)…"
            value={search}
            onChange={handleSearch}
          />
        </div>

        <div className="de-filter-pills">
          {[
            { key: 'all', label: '📋 All', count: data.length },
            { key: 'spike', label: '🔴 Spikes', count: stats?.spikes },
            { key: 'drop', label: '🟡 Drops', count: stats?.drops },
            { key: 'normal', label: '✅ Normal', count: (data.length || 0) - (stats?.anomalyCount || 0) },
          ].map(f => (
            <button
              key={f.key}
              id={`de-filter-${f.key}`}
              className={`de-pill ${filter === f.key ? 'active' : ''}`}
              onClick={() => handleFilterChange(f.key)}
            >
              {f.label} <span className="de-pill-count">{f.count}</span>
            </button>
          ))}
        </div>

        <div className="de-actions">
          <button
            id="de-sort-btn"
            className="btn btn-ghost"
            style={{ padding: '8px 14px', fontSize: '0.82rem' }}
            onClick={() => setSortDir(d => d === 'asc' ? 'desc' : 'asc')}
          >
            {sortDir === 'asc' ? '⬆ Oldest First' : '⬇ Newest First'}
          </button>
          <button
            id="de-export-btn"
            className="btn btn-primary"
            style={{ padding: '8px 16px', fontSize: '0.82rem' }}
            onClick={handleExport}
          >
            ⬇ Export CSV
          </button>
        </div>
      </div>

      {/* ── Table ── */}
      <div className="glass-card de-table-card">
        <div className="de-table-header-row">
          <div>
            <h3>Raw Data Table</h3>
            <p style={{ fontSize: '0.8rem', marginTop: '4px' }}>
              Showing {filtered.length.toLocaleString()} of {data.length.toLocaleString()} rows
              {filter !== 'all' ? ` · filtered: ${filter}` : ''}
              {search ? ` · search: "${search}"` : ''}
            </p>
          </div>
          <div className="de-legend">
            <span className="de-legend-item spike">▲ Spike</span>
            <span className="de-legend-item drop">▼ Drop</span>
            <span className="de-legend-item normal">● Normal</span>
          </div>
        </div>

        <div className="divider" style={{ margin: '16px 0' }} />

        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Date</th>
                <th>{contextLabel}</th>
                <th>Status</th>
                <th>Deviation</th>
                <th>Change from Prev.</th>
              </tr>
            </thead>
            <tbody>
              {paginated.length === 0 ? (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', padding: '32px', color: 'var(--text-muted)' }}>
                    No matching rows found
                  </td>
                </tr>
              ) : paginated.map((row, i) => {
                const globalIdx = (page - 1) * PAGE_SIZE + i
                const anom = row.anomaly
                const isSpike = anom?.direction === 'spike'
                const isDrop = anom?.direction === 'drop'

                // Calculate change from previous row in FILTERED list
                const prevRow = filtered[globalIdx > 0 ? globalIdx - 1 : 0]
                const delta = globalIdx > 0 ? row.y - prevRow.y : null
                const deltaColor = delta === null ? '' : delta >= 0 ? 'var(--green)' : 'var(--red)'
                const deltaSym = delta === null ? '—' : delta >= 0 ? `▲ +${delta.toFixed(1)}` : `▼ ${delta.toFixed(1)}`

                return (
                  <tr
                    key={row.ds}
                    className={`de-row ${isSpike ? 'row-spike' : isDrop ? 'row-drop' : ''}`}
                  >
                    <td className="de-row-num">{globalIdx + 1}</td>
                    <td className="de-date">{row.ds}</td>
                    <td className="de-value">{row.y.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                    <td>
                      {anom ? (
                        <span className={`badge ${isSpike ? 'badge-red' : 'badge-yellow'}`}>
                          {isSpike ? '🔴 Spike' : '🟡 Drop'}
                        </span>
                      ) : (
                        <span className="badge badge-green">✅ Normal</span>
                      )}
                    </td>
                    <td style={{ color: isSpike ? 'var(--red)' : isDrop ? 'var(--yellow)' : 'var(--text-muted)' }}>
                      {anom ? `${anom.pct_deviation > 0 ? '+' : ''}${anom.pct_deviation?.toFixed(1)}%` : '—'}
                    </td>
                    <td style={{ color: deltaColor, fontVariantNumeric: 'tabular-nums' }}>
                      {deltaSym}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="de-pagination">
            <button
              id="de-prev-btn"
              className="btn btn-ghost"
              style={{ padding: '6px 14px', fontSize: '0.82rem' }}
              disabled={page === 1}
              onClick={() => setPage(p => p - 1)}
            >
              ← Prev
            </button>
            <div className="de-page-numbers">
              {Array.from({ length: totalPages }, (_, i) => i + 1)
                .filter(p => p === 1 || p === totalPages || Math.abs(p - page) <= 2)
                .reduce((acc, p, idx, arr) => {
                  if (idx > 0 && p - arr[idx - 1] > 1) acc.push('...')
                  acc.push(p)
                  return acc
                }, [])
                .map((p, i) => (
                  typeof p === 'number'
                    ? <button
                        key={i}
                        className={`de-page-btn ${p === page ? 'active' : ''}`}
                        onClick={() => setPage(p)}
                      >{p}</button>
                    : <span key={i} className="de-page-ellipsis">…</span>
                ))}
            </div>
            <button
              id="de-next-btn"
              className="btn btn-ghost"
              style={{ padding: '6px 14px', fontSize: '0.82rem' }}
              disabled={page === totalPages}
              onClick={() => setPage(p => p + 1)}
            >
              Next →
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
