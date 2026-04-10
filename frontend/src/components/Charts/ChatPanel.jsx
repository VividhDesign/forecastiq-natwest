import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import './ChatPanel.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

const SUGGESTED_QUESTIONS = [
  'What is the overall growth trend?',
  'How confident is this forecast?',
  'What caused the anomalies?',
  'When should I prepare for the peak?',
  'How does this compare to the naive baseline?',
]

export default function ChatPanel({ summaryStats, contextLabel, selectedModel }) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      text: `I can answer questions about your **${contextLabel}** forecast using only the verified data from the model. Ask me anything about the trend, confidence range, anomalies, or peak forecasts.`,
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = async (question) => {
    const q = question || input.trim()
    if (!q || loading) return

    setInput('')
    setMessages(prev => [...prev, { role: 'user', text: q }])
    setLoading(true)

    try {
      const res = await axios.post(`${API_BASE}/chat`, {
        question: q,
        summary_stats: summaryStats,
        anomaly_count: summaryStats?.anomaly_count || 0,
        context_label: contextLabel,
        model_choice: selectedModel,
      })
      setMessages(prev => [...prev, { role: 'assistant', text: res.data.answer }])
    } catch (e) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: '⚠️ Could not reach the AI. Please check your connection.',
        isError: true,
      }])
    } finally {
      setLoading(false)
    }
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="chat-panel">
      {/* Header */}
      <div className="chat-header">
        <div>
          <h3>Ask About Your Data</h3>
          <p className="chat-subtitle">
            Questions are answered using only verified numbers from the forecasting model — no hallucinated stats.
          </p>
        </div>
        <span className="chat-model-badge">
          {selectedModel === 'groq' ? '⚡ Llama-3.3' : '✦ Gemini'}
        </span>
      </div>

      {/* Suggested Questions */}
      <div className="chat-suggestions">
        {SUGGESTED_QUESTIONS.map((q) => (
          <button
            key={q}
            className="suggestion-pill"
            onClick={() => sendMessage(q)}
            disabled={loading}
          >
            {q}
          </button>
        ))}
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-bubble ${msg.role} ${msg.isError ? 'error' : ''}`}>
            {msg.role === 'assistant' && (
              <div className="bubble-label">📊 ForecastIQ</div>
            )}
            <div className="bubble-text">
              {msg.text.split('**').map((part, j) =>
                j % 2 === 1 ? <strong key={j}>{part}</strong> : part
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="chat-bubble assistant">
            <div className="bubble-label">📊 ForecastIQ</div>
            <div className="typing-dots">
              <span /><span /><span />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="chat-input-row">
        <input
          className="chat-input"
          type="text"
          placeholder="e.g. What is the expected growth over the next 4 weeks?"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
          disabled={loading}
        />
        <button
          className="chat-send"
          onClick={() => sendMessage()}
          disabled={loading || !input.trim()}
        >
          {loading ? <span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} /> : '→'}
        </button>
      </div>
      <div className="chat-hint">Press Enter to send · Shift+Enter for new line</div>
    </div>
  )
}
