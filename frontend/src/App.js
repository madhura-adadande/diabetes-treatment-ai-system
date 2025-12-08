import React, { useState } from 'react';
import './App.css';

function App() {
  const [patientData, setPatientData] = useState({
    glucose: '',
    bmi: '',
    age: '',
    blood_pressure: '',
    pregnancies: '0',
    diabetes_pedigree: '0.5'
  });
  
  const [recommendation, setRecommendation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    // Convert all string inputs to proper numbers
    const dataToSend = {
      glucose: parseFloat(patientData.glucose),
      bmi: parseFloat(patientData.bmi),
      age: parseInt(patientData.age),
      blood_pressure: parseFloat(patientData.blood_pressure),
      pregnancies: parseInt(patientData.pregnancies) || 0,
      diabetes_pedigree: parseFloat(patientData.diabetes_pedigree) || 0.5
    };
    
    // Validate data
    if (isNaN(dataToSend.glucose) || isNaN(dataToSend.bmi) || isNaN(dataToSend.age) || isNaN(dataToSend.blood_pressure)) {
      setError('Please fill in all required fields with valid numbers');
      setLoading(false);
      return;
    }
    
    console.log('Sending data:', dataToSend);
    
    try {
      const response = await fetch('http://localhost:8000/recommend_treatment', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(dataToSend)
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }
      
      const result = await response.json();
      console.log('Received:', result);
      setRecommendation(result);
    } catch (error) {
      console.error('Error:', error);
      setError('Error connecting to AI system: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field, value) => {
    setPatientData({
      ...patientData,
      [field]: value
    });
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🏥 Diabetes Treatment AI System</h1>
        <p>AI-Powered Treatment Recommendations with DQN & REINFORCE</p>
      </header>
      
      <div className="main-container">
        <div className="patient-form">
          <h2>📋 Patient Information</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label>Glucose Level (mg/dL): *</label>
              <input
                type="number"
                step="0.1"
                placeholder="Normal: 70-100, Diabetes: >126"
                value={patientData.glucose}
                onChange={(e) => handleInputChange('glucose', e.target.value)}
                required
              />
            </div>
            
            <div className="form-group">
              <label>BMI (Body Mass Index): *</label>
              <input
                type="number"
                step="0.1"
                placeholder="Normal: 18.5-24.9, Overweight: 25-29.9"
                value={patientData.bmi}
                onChange={(e) => handleInputChange('bmi', e.target.value)}
                required
              />
            </div>
            
            <div className="form-group">
              <label>Age (years): *</label>
              <input
                type="number"
                placeholder="Enter patient age"
                value={patientData.age}
                onChange={(e) => handleInputChange('age', e.target.value)}
                required
              />
            </div>
            
            <div className="form-group">
              <label>Blood Pressure (systolic): *</label>
              <input
                type="number"
                placeholder="Normal: <120, High: >140"
                value={patientData.blood_pressure}
                onChange={(e) => handleInputChange('blood_pressure', e.target.value)}
                required
              />
            </div>
            
            <div className="form-group">
              <label>Number of Pregnancies:</label>
              <input
                type="number"
                placeholder="0 for males, actual count for females"
                value={patientData.pregnancies}
                onChange={(e) => handleInputChange('pregnancies', e.target.value)}
              />
            </div>
            
            <div className="form-group">
              <label>Diabetes Pedigree Function:</label>
              <input
                type="number"
                step="0.01"
                placeholder="Family history factor (0.0-2.0)"
                value={patientData.diabetes_pedigree}
                onChange={(e) => handleInputChange('diabetes_pedigree', e.target.value)}
              />
            </div>
            
            <button type="submit" className="submit-btn" disabled={loading}>
              {loading ? '🔄 AI Processing...' : '🤖 Get AI Treatment Recommendation'}
            </button>
          </form>
          
          {error && (
            <div className="error-message" style={{
              background: '#fee', 
              border: '1px solid #fcc', 
              padding: '15px', 
              borderRadius: '8px', 
              marginTop: '15px',
              color: '#c00'
            }}>
              {error}
            </div>
          )}
        </div>
        
        {recommendation && (
          <div className="recommendations">
            <h2>🧠 AI Treatment Recommendations</h2>
            <div id="results">
              <div className="algorithm-card dqn-card">
                <h3>🎯 Deep Q-Network (DQN)</h3>
                <p className="treatment">{recommendation.recommendations?.dqn?.treatment || 'No recommendation'}</p>
                <p className="confidence">Confidence: {((recommendation.recommendations?.dqn?.confidence || 0) * 100).toFixed(1)}%</p>
                <small style={{color: '#666'}}>Trained on 1,000 episodes • 5.4M parameters</small>
              </div>
              <div className="algorithm-card pg-card">
                <h3>🎲 Policy Gradient (REINFORCE)</h3>
                <p className="treatment">{recommendation.recommendations?.policy_gradient?.treatment || 'No recommendation'}</p>
                <p className="confidence">Confidence: {((recommendation.recommendations?.policy_gradient?.confidence || 0) * 100).toFixed(1)}%</p>
                <small style={{color: '#666'}}>Trained on 500 episodes • 347K parameters</small>
              </div>
            </div>
            <div className="system-info">
              <p>📊 <strong>Trained on 883,825 real patients</strong> from CDC BRFSS dataset</p>
              <p>🧠 <strong>5.4M+ neural network parameters</strong> across two RL algorithms</p>
              <p>⚡ <strong>Real-time clinical decision support</strong> with confidence scoring</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;