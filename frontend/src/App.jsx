import React from 'react';
import './App.css'
import { useHandTracker } from './hooks/useHandTracker';
import { usePredictor } from './hooks/usePredictor';
import HandTracker from './components/HandTracker';
import PredictionPanel from './components/PredictionPanel';

function App() {
  const { videoRef, latestFrameFeatures, rawMultiHandLandmarks, rawMultiHandedness, mpHands } = useHandTracker();
  const { predictions } = usePredictor(latestFrameFeatures);

  return (
    <div className="App" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '20px' }}>
      <h1>ASL Translator</h1>
      <HandTracker videoRef={videoRef} rawMultiHandLandmarks={rawMultiHandLandmarks} rawMultiHandedness={rawMultiHandedness} mpHands={mpHands} />
      <PredictionPanel predictions={predictions} />
    </div>
  )
}

export default App
