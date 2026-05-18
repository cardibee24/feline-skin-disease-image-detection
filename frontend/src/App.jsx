import React, { useState, useEffect } from 'react';
import ImageUpload from './components/ImageUpload';
import PredictionResult from './components/PredictionResult';
import ScanHistory from './components/ScanHistory';
import AboutModal from './components/AboutModal';

const API_URL = 'http://localhost:8000';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [errorType, setErrorType] = useState(null);
  const [history, setHistory] = useState([]);
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem('theme') === 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
    localStorage.setItem('theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  const handleUpload = (selectedFile, previewData) => {
    setFile(selectedFile);
    setPreview(previewData);
    setResult(null);
    setError(null);
    setErrorType(null);
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
        const errData = await response.json().catch(() => null);
        if (errData?.detail?.error === 'not_feline') {
          setError(errData.detail.message);
          setErrorType('not_feline');
          return;
        }
        throw new Error('Failed to analyze image. Is the backend running?');
      }

      const data = await response.json();
      setResult(data);

      // Add to history
      const now = new Date();
      const timestamp = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) +
        ' · ' + now.toLocaleDateString([], { day: 'numeric', month: 'short' });

      setHistory(prev => [{
        prediction: data.prediction,
        confidence: data.confidence,
        preview: preview,
        timestamp,
      }, ...prev]);

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
    setErrorType(null);
  };

  return (
    <div className="app-container">
      {/* Theme toggle */}
      <button
        onClick={() => setDarkMode(!darkMode)}
        title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
        style={{
          position: 'fixed',
          top: '1.2rem',
          right: '1.5rem',
          background: 'var(--glass-bg)',
          border: '1px solid var(--glass-border)',
          borderRadius: '50%',
          width: '42px',
          height: '42px',
          fontSize: '1.2rem',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
          transition: 'all 0.2s ease',
          zIndex: 100,
        }}
      >
        {darkMode ? '☀️' : '🌙'}
      </button>

      <header>
        <h1>🐾 Feline Skin Guard</h1>
        <p className="subtitle">AI-Powered Skin Disease Detection</p>
      </header>

      <main className="w-full flex-center flex-col">
        {!result && (
          <div className="glass-card fade-in-up">
            <ImageUpload onUpload={handleUpload} isLoading={loading} />

            {file && (
              <div style={{ marginTop: '1.5rem' }}>
                <button
                  className="btn"
                  onClick={analyzeImage}
                  disabled={loading}
                  style={{ width: '100%' }}
                >
                  {loading ? 'Analyzing...' : 'Analyze Image'}
                </button>
              </div>
            )}

            {errorType === 'not_feline' && (
              <div style={{
                marginTop: '1.2rem',
                padding: '1rem 1.2rem',
                background: 'rgba(139,74,60,0.08)',
                border: '1px solid rgba(139,74,60,0.35)',
                borderRadius: '12px',
                textAlign: 'center',
              }}>
                <div style={{ fontSize: '2.2rem', marginBottom: '0.4rem' }}>🐾</div>
                <p style={{ fontWeight: '600', color: 'var(--text-main)', marginBottom: '0.3rem' }}>
                  No Cat Detected
                </p>
                <p style={{ fontSize: '0.87rem', color: 'var(--text-muted)', lineHeight: '1.6' }}>
                  {error}
                </p>
              </div>
            )}

            {error && errorType !== 'not_feline' && (
              <div style={{ color: 'var(--error)', textAlign: 'center', marginTop: '1rem' }}>
                {error}
              </div>
            )}
          </div>
        )}

        {result && (
          <PredictionResult result={result} onReset={resetApp} />
        )}

        {/* Scan history shown below result or upload card */}
        <ScanHistory
          history={history}
          onClear={() => setHistory([])}
        />
        <AboutModal />
      </main>

      <footer style={{ marginTop: 'auto', paddingTop: '2rem', paddingBottom: '1rem', color: 'var(--text-muted)', fontSize: '0.85rem', textAlign: 'center' }}>
        <p>© 22ACB05340_AU_YING_SEE</p>
      </footer>
    </div>
  );
}

export default App;
