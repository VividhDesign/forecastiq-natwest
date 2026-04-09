import './InsightCard.css'

export default function InsightCard({ icon, title, text, model, loading }) {
  const modelLabel = model === 'gemini' ? 'Google Gemini Pro' : 'Groq / Llama-3'

  return (
    <div className="insight-card-wrapper glass-card">
      <div className="insight-header">
        <div className="flex items-center gap-2">
          <span className="insight-icon-lg">{icon}</span>
          <h3>{title}</h3>
        </div>
        <span className="badge badge-blue" style={{ fontSize: '0.7rem' }}>
          {model === 'gemini' ? '✨' : '⚡'} {modelLabel}
        </span>
      </div>
      <div className="divider" style={{ margin: '14px 0' }} />
      {loading ? (
        <div className="flex items-center gap-3">
          <div className="spinner" />
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>AI is generating insight...</span>
        </div>
      ) : (
        <p className="insight-body">{text || 'No insight available. Check your API key configuration.'}</p>
      )}
    </div>
  )
}
