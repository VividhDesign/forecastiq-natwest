import './NavBar.css'

const MODEL_OPTIONS = [
  { value: 'gemini', label: '✨ Gemini 2.5 Flash' },
  { value: 'groq',   label: '⚡ Llama-3.3 (Groq)' },
]

export default function NavBar({ contextLabel, contextDesc, selectedModel, onModelChange, onReset }) {
  return (
    <nav className="navbar">
      <div className="navbar-inner">
        {/* Logo */}
        <div className="navbar-logo">
          <span className="logo-icon-sm">🔮</span>
          <span className="gradient-text" style={{ fontWeight: 800, fontSize: '1.1rem' }}>ForecastIQ</span>
        </div>

        {/* Context Info */}
        <div className="navbar-context">
          <div className="context-name">{contextLabel}</div>
          {contextDesc && <div className="context-desc-nav">{contextDesc}</div>}
        </div>

        {/* Right Controls */}
        <div className="navbar-controls">
          {/* Model Selector */}
          <div className="model-selector-wrapper">
            <label className="label" style={{ margin: 0, fontSize: '0.7rem' }}>AI Model</label>
            <div className="model-selector">
              {MODEL_OPTIONS.map(m => (
                <button
                  key={m.value}
                  className={`model-btn ${selectedModel === m.value ? 'active' : ''}`}
                  onClick={() => onModelChange(m.value)}
                  title={`Switch to ${m.label}`}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          {/* Reset */}
          <button className="btn btn-ghost reset-btn" onClick={onReset} title="Start over with new data">
            ↺ New Data
          </button>
        </div>
      </div>
    </nav>
  )
}
