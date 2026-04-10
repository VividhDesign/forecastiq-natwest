import { useState, useEffect } from 'react'
import Onboarding from './components/Onboarding/Onboarding'
import Dashboard from './components/Dashboard/Dashboard'
import './index.css'

/**
 * Root application component.
 * Manages the two-phase flow: Onboarding → Dashboard.
 * All dataset, LLM model, and theme state lives here.
 */
function App() {
  const [appState, setAppState] = useState('onboarding') // 'onboarding' | 'dashboard'
  const [datasetPayload, setDatasetPayload] = useState(null)
  const [selectedModel, setSelectedModel] = useState('groq')

  // Theme: persisted in localStorage, defaults to 'dark'
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('fiq-theme') || 'light'
  })

  // Apply theme to <html> data-theme attribute
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('fiq-theme', theme)
  }, [theme])

  const toggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark')

  const handleDataReady = (payload) => {
    setDatasetPayload(payload)
    setAppState('dashboard')
  }

  const handleReset = () => {
    setDatasetPayload(null)
    setAppState('onboarding')
  }

  return (
    <div className="app">
      {appState === 'onboarding' ? (
        <Onboarding onDataReady={handleDataReady} theme={theme} onThemeToggle={toggleTheme} />
      ) : (
        <Dashboard
          datasetPayload={datasetPayload}
          selectedModel={selectedModel}
          onModelChange={setSelectedModel}
          onReset={handleReset}
          theme={theme}
          onThemeToggle={toggleTheme}
        />
      )}
    </div>
  )
}

export default App
