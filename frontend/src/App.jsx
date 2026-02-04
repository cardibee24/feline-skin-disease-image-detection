import React, { useState } from 'react';
import ImageUpload from './components/ImageUpload';
import PredictionResult from './components/PredictionResult';

// Base URL for API - change if deployed elsewhere
const API_URL = 'http://localhost:8000';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleUpload = (selectedFile, previewData) => {
    setFile(selectedFile);
    setPreview(previewData);
    setResult(null);
    setError(null);
  };

  const analyzeImage = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to analyze image. Is the backend running?');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError(err.message || 'An error occurred during analysis.');
    } finally {
      setLoading(false);
    }
  };

  const resetApp = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="app-container">
      <header>
        <h1>Feline Skin Guard</h1>
        <p className="subtitle">AI-Powered Skin Disease Detection</p>
      </header>

      <main className="w-full flex-center flex-col">
        {!result && (
          <div className="glass-card fade-in-up">
            <ImageUpload onUpload={handleUpload} isLoading={loading} />

            {file && (
              <div className="mt-4 text-center">
                <button
                  className="btn"
                  onClick={analyzeImage}
                  disabled={loading}
                  style={{ width: '100%', marginTop: '1rem' }}
                >
                  {loading ? 'Analyzing...' : 'Analyze Image'}
                </button>
              </div>
            )}

            {error && (
              <div className="mt-4" style={{ color: 'var(--error)', textAlign: 'center' }}>
                {error}
              </div>
            )}
          </div>
        )}

        {result && (
          <PredictionResult result={result} onReset={resetApp} />
        )}
      </main>

      <footer style={{ marginTop: 'auto', paddingTop: '2rem', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
        <p>© 2026 Feline Skin Guard Project</p>
      </footer>
    </div>
  );
}

export default App;
