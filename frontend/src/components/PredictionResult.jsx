import React from 'react';

const PredictionResult = ({ result, onReset }) => {
    if (!result) return null;

    const { prediction, confidence, mode } = result;
    const percentage = (confidence * 100).toFixed(1);

    // Color coding based on confidence
    let color = 'var(--success)';
    if (confidence < 0.7) color = 'var(--warning)';
    if (confidence < 0.4) color = 'var(--error)';

    return (
        <div className="glass-card fade-in-up" style={{ marginTop: '2rem', textAlign: 'center' }}>
            <h2 style={{ marginBottom: '1rem', color: 'var(--text-main)' }}>Analysis Result</h2>

            <div style={{ marginBottom: '2rem' }}>
                <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Detected Condition</div>
                <div style={{ fontSize: '2.5rem', fontWeight: 'bold', color: 'var(--primary-color)' }}>
                    {prediction}
                </div>
                {mode === 'mock_model' && (
                    <div style={{ fontSize: '0.7rem', color: 'var(--warning)', marginTop: '0.2rem' }}>
                        (Simulated Result - Model Not Loaded)
                    </div>
                )}
            </div>

            <div style={{ marginBottom: '2rem', background: 'rgba(255,255,255,0.05)', padding: '1rem', borderRadius: '12px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <span>Confidence</span>
                    <span>{percentage}%</span>
                </div>
                <div style={{ width: '100%', height: '10px', background: 'rgba(255,255,255,0.1)', borderRadius: '5px', overflow: 'hidden' }}>
                    <div
                        style={{
                            width: `${percentage}%`,
                            height: '100%',
                            background: color,
                            transition: 'width 1s ease-out'
                        }}
                    />
                </div>
            </div>

            <button onClick={onReset} className="btn" style={{ background: 'transparent', border: '1px solid var(--text-muted)' }}>
                Analyze Another Image
            </button>
        </div>
    );
};

export default PredictionResult;
