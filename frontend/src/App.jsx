import { useState } from 'react'
import Onboarding from './components/Onboarding/Onboarding'
import Dashboard from './components/Dashboard/Dashboard'
import './index.css'

/**
 * Root application component.
 * Manages the two-phase flow: Onboarding → Dashboard.
 * All dataset and LLM model state lives here and is passed down.
 */
function App() {
  const [appState, setAppState] = useState('onboarding') // 'onboarding' | 'dashboard'
  const [datasetPayload, setDatasetPayload] = useState(null)
  const [selectedModel, setSelectedModel] = useState('gemini')

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
        <Onboarding onDataReady={handleDataReady} />
      ) : (
        <Dashboard
          datasetPayload={datasetPayload}
          selectedModel={selectedModel}
          onModelChange={setSelectedModel}
          onReset={handleReset}
        />
      )}
    </div>
  )
}

export default App
