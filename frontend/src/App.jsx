import React, { useState, useEffect } from 'react';
import ImageUpload from './components/ImageUpload';
import PredictionResult from './components/PredictionResult';
import ScanHistory from './components/ScanHistory';
import AboutModal from './components/AboutModal';

const API_URL = 'http://localhost:8000';

function App() {
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [results, setResults] = useState([]); // [{...prediction} | {_error, message}]
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [history, setHistory] = useState([]);
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem('theme') === 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
    localStorage.setItem('theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  const handleUpload = (selectedFiles, previewData) => {
    setFiles(selectedFiles);
    setPreviews(previewData);
    setResults([]);
  };

  const analyzeImages = async () => {
    if (!files.length) return;
    setLoading(true);
    setResults([]);
    setProgress({ current: 0, total: files.length });

    const allResults = [];

    for (let i = 0; i < files.length; i++) {
      setProgress({ current: i + 1, total: files.length });
      const formData = new FormData();
      formData.append('file', files[i]);

      try {
        const response = await fetch(`${API_URL}/predict`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => null);
          if (errData?.detail?.error === 'not_feline') {
            allResults.push({ _error: 'not_feline', message: errData.detail.message });
          } else {
            allResults.push({ _error: 'generic', message: 'Failed to analyze image. Is the backend running?' });
          }
        } else {
          allResults.push(await response.json());
        }
      } catch (err) {
        allResults.push({ _error: 'generic', message: err.message || 'An error occurred during analysis.' });
      }
    }

    setResults(allResults);
    setLoading(false);
    setProgress({ current: 0, total: 0 });

    const now = new Date();
    const timestamp =
      now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) +
      ' · ' +
      now.toLocaleDateString([], { day: 'numeric', month: 'short' });

    const historyEntries = allResults
      .map((r, i) => ({ ...r, _preview: previews[i] }))
      .filter(r => !r._error)
      .map(r => ({ prediction: r.prediction, confidence: r.confidence, preview: r._preview, timestamp }));

    if (historyEntries.length) setHistory(prev => [...historyEntries, ...prev]);
  };

  const resetApp = () => {
    setFiles([]);
    setPreviews([]);
    setResults([]);
    setProgress({ current: 0, total: 0 });
  };

  const hasResults = results.length > 0;
  const isMulti = results.length > 1;

  const analyzeLabel = () => {
    if (!loading) return files.length > 1 ? `Analyze ${files.length} Images` : 'Analyze Image';
    if (progress.total > 1) return `Analyzing ${progress.current}/${progress.total}...`;
    return 'Analyzing...';
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
        {!hasResults && (
          <div className="glass-card fade-in-up">
            <ImageUpload onUpload={handleUpload} isLoading={loading} />

            {files.length > 0 && (
              <div style={{ marginTop: '1.5rem' }}>
                <button
                  className="btn"
                  onClick={analyzeImages}
                  disabled={loading}
                  style={{ width: '100%' }}
                >
                  {analyzeLabel()}
                </button>
              </div>
            )}
          </div>
        )}

        {hasResults && (
          <div className="fade-in-up" style={{ width: '100%' }}>
            {isMulti && (
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '1.5rem',
                padding: '0.85rem 1.1rem',
                background: 'var(--glass-bg)',
                border: '1px solid var(--glass-border)',
                borderRadius: '14px',
                boxShadow: '0 2px 12px rgba(103,114,81,0.1)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                  <span style={{ fontSize: '1.3rem' }}>🐾</span>
                  <div>
                    <div style={{ fontWeight: '700', fontSize: '1rem', color: 'var(--text-main)', lineHeight: 1.2 }}>
                      Analysis Complete
                    </div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                      {results.length} image{results.length !== 1 ? 's' : ''} processed
                    </div>
                  </div>
                </div>
                <button
                  className="btn"
                  onClick={resetApp}
                  style={{ background: '#4a5240', border: 'none', color: '#fff', fontSize: '0.85rem', padding: '0.45rem 1rem' }}
                >
                  + New Batch
                </button>
              </div>
            )}

            <div style={isMulti ? {
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              columnGap: '1rem',
              rowGap: '2rem',
              alignItems: 'start',
            } : {}}>
              {results.map((result, i) => (
                <div key={i}>
                  {isMulti && (
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.65rem',
                      marginBottom: '0.25rem',
                      padding: '0.45rem 0.7rem',
                      background: 'var(--glass-bg)',
                      border: '1px solid var(--glass-border)',
                      borderRadius: '10px',
                    }}>
                      <img
                        src={previews[i]}
                        alt={`Image ${i + 1}`}
                        style={{ width: '32px', height: '32px', objectFit: 'cover', borderRadius: '6px', flexShrink: 0 }}
                      />
                      <span style={{ fontWeight: '600', color: 'var(--primary-color)', fontSize: '0.88rem' }}>
                        Image {i + 1}
                      </span>
                      <span style={{
                        marginLeft: 'auto',
                        fontSize: '0.75rem',
                        color: 'var(--text-muted)',
                        background: 'rgba(103,114,81,0.12)',
                        borderRadius: '20px',
                        padding: '2px 8px',
                      }}>
                        {result._error ? 'error' : `${(result.confidence * 100).toFixed(0)}% confidence`}
                      </span>
                    </div>
                  )}

                  {result._error === 'not_feline' ? (
                    <div className="glass-card" style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '2.2rem', marginBottom: '0.4rem' }}>🐾</div>
                      <p style={{ fontWeight: '600', color: 'var(--text-main)', marginBottom: '0.3rem' }}>No Cat Detected</p>
                      <p style={{ fontSize: '0.87rem', color: 'var(--text-muted)', lineHeight: '1.6' }}>{result.message}</p>
                      {!isMulti && (
                        <button onClick={resetApp} className="btn" style={{ background: '#4a5240', border: 'none', color: '#fff', marginTop: '1rem', width: '100%' }}>
                          Try Another Image
                        </button>
                      )}
                    </div>
                  ) : result._error ? (
                    <div className="glass-card" style={{ textAlign: 'center' }}>
                      <p style={{ color: 'var(--error)' }}>{result.message}</p>
                      {!isMulti && (
                        <button onClick={resetApp} className="btn" style={{ background: '#4a5240', border: 'none', color: '#fff', marginTop: '1rem', width: '100%' }}>
                          Try Another Image
                        </button>
                      )}
                    </div>
                  ) : (
                    <PredictionResult
                      result={result}
                      onReset={resetApp}
                      showReset={!isMulti}
                      outerStyle={isMulti ? { marginTop: 0 } : {}}
                    />
                  )}
                </div>
              ))}
            </div>

            {isMulti && (
              <button
                className="btn"
                onClick={resetApp}
                style={{ background: '#4a5240', border: 'none', color: '#fff', width: '100%', marginTop: '0.5rem' }}
              >
                Analyze Another Batch
              </button>
            )}
          </div>
        )}

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
