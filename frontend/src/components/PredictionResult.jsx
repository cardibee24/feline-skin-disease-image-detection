import React, { useState, useEffect } from 'react';

const DISEASE_INFO = {
    scabies: {
        icon: '🦠',
        description: 'Scabies is caused by tiny mites that burrow into the skin, causing intense itching and crusty sores.',
        action: 'Consult a vet immediately. Treatment typically involves prescribed antiparasitic medication and keeping the cat isolated to prevent spreading.',
        severity: 'high',
    },
    ringworm: {
        icon: '🍄',
        description: 'Ringworm is a contagious fungal infection that causes circular patches of hair loss and scaly skin.',
        action: 'Visit a vet for antifungal treatment (topical or oral). Avoid contact with other pets as it can spread to humans too.',
        severity: 'medium',
    },
    flea: {
        icon: '🪲',
        description: 'Flea infestation causes excessive scratching, hair loss, and small red bumps on the skin.',
        action: 'Apply vet-recommended flea treatment. Wash bedding and vacuum the home thoroughly to eliminate fleas from the environment.',
        severity: 'medium',
    },
    healthy: {
        icon: '✅',
        description: "No signs of skin disease detected. Your cat's skin appears to be in good condition.",
        action: "Keep up with regular grooming and annual vet check-ups to maintain your cat's skin health.",
        severity: 'none',
    },
};

const severityColor = { high: '#8b4a3c', medium: '#a1907c', none: '#677251' };
const severityBg = { high: 'rgba(139,74,60,0.08)', medium: 'rgba(161,144,124,0.1)', none: 'rgba(103,114,81,0.08)' };

const PredictionResult = ({ result, onReset }) => {
    if (!result) return null;
    const { prediction, confidence, all_probabilities, gradcam_image, mode } = result;
    const percentage = (confidence * 100).toFixed(1);
    const isUnclear = confidence < 0.5;
    const isLowConfidence = confidence >= 0.5 && confidence < 0.7;

    const info = DISEASE_INFO[prediction?.toLowerCase()] || {
        icon: '🔍',
        description: 'Condition detected by the model.',
        action: 'Please consult a veterinarian for a professional diagnosis.',
        severity: 'medium',
    };

    // Reject unclear predictions
    if (isUnclear) {
        return (
            <div className="glass-card fade-in-up" style={{ marginTop: '2rem', textAlign: 'center' }}>
                <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🔎</div>
                <h2 style={{ marginBottom: '0.75rem', color: 'var(--text-main)' }}>Unable to Determine</h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', lineHeight: '1.7', marginBottom: '1.5rem' }}>
                    The model confidence is too low ({percentage}%) to make a reliable prediction.
                    This may be due to image quality, lighting, or an unclear skin condition.
                </p>
                <div style={{
                    background: 'rgba(139,74,60,0.08)',
                    border: '1px solid rgba(139,74,60,0.3)',
                    borderRadius: '12px',
                    padding: '0.9rem 1rem',
                    marginBottom: '1.5rem',
                    display: 'flex',
                    gap: '0.6rem',
                    alignItems: 'flex-start',
                    textAlign: 'left',
                }}>
                    <span style={{ fontSize: '1.1rem' }}>💡</span>
                    <p style={{ fontSize: '0.88rem', color: 'var(--text-main)', lineHeight: '1.6' }}>
                        Please retake the photo with better lighting and ensure the affected skin area is clearly visible. Always consult a veterinarian for a proper diagnosis.
                    </p>
                </div>
                <button onClick={onReset} className="btn" style={{ background: '#4a5240', border: 'none', color: '#fff', width: '100%' }}>
                    Try Another Image
                </button>
            </div>
        );
    }

    return (
        <div className="glass-card fade-in-up" style={{ marginTop: '2rem', textAlign: 'center' }}>
            <h2 style={{ marginBottom: '1rem', color: 'var(--text-main)' }}>Analysis Result</h2>

            {/* Low confidence warning */}
            {isLowConfidence && (
                <div style={{
                    background: 'rgba(161,144,124,0.15)',
                    border: '1px solid #a1907c',
                    borderRadius: '10px',
                    padding: '0.7rem 1rem',
                    marginBottom: '1.2rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    textAlign: 'left',
                }}>
                    <span style={{ fontSize: '1.1rem' }}>⚠️</span>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-main)', lineHeight: '1.5' }}>
                        <strong>Low confidence detection ({percentage}%).</strong> Result may not be accurate — please consult a vet for a proper diagnosis.
                    </p>
                </div>
            )}

            {/* Detected condition */}
            <div style={{ marginBottom: '1.5rem' }}>
                <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Detected Condition</div>
                <div style={{ fontSize: '2.8rem', marginBottom: '0.25rem' }}>{info.icon}</div>
                <div style={{ fontSize: '2rem', fontWeight: 'bold', color: 'var(--primary-color)', textTransform: 'capitalize' }}>
                    {prediction}
                </div>
                {mode === 'mock_model' && (
                    <div style={{ fontSize: '0.7rem', color: 'var(--warning)', marginTop: '0.2rem' }}>
                        (Simulated Result — Model Not Loaded)
                    </div>
                )}
            </div>

            {/* Grad-CAM heatmap */}
            {gradcam_image && (
                <div style={{ marginBottom: '1.5rem' }}>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.6rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        AI Focus Area (Grad-CAM)
                    </div>
                    <img
                        src={`data:image/png;base64,${gradcam_image}`}
                        alt="Grad-CAM heatmap"
                        style={{
                            width: '100%',
                            maxWidth: '224px',
                            borderRadius: '12px',
                            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                        }}
                    />
                    <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                        🔴 Red areas = regions the model focused on most
                    </p>
                </div>
            )}

            {/* All probabilities breakdown */}
            {all_probabilities && (
                <div style={{ marginBottom: '1.5rem', background: 'rgba(103,114,81,0.08)', padding: '1rem', borderRadius: '12px', textAlign: 'left' }}>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.8rem', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        All Probabilities
                    </div>
                    {all_probabilities.map((item, index) => (
                        <ProbBar
                            key={item.label}
                            label={item.label}
                            confidence={item.confidence}
                            isTop={index === 0}
                            delay={index * 150}
                        />
                    ))}
                </div>
            )}

            {/* Disease info card */}
            <div style={{
                background: severityBg[info.severity],
                border: `1px solid ${severityColor[info.severity]}40`,
                borderRadius: '12px',
                padding: '1rem 1.2rem',
                marginBottom: '1.5rem',
                textAlign: 'left',
            }}>
                <p style={{ fontSize: '0.95rem', color: 'var(--text-main)', marginBottom: '0.6rem', lineHeight: '1.6' }}>
                    {info.description}
                </p>
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
                    <span style={{ fontSize: '1rem', marginTop: '2px' }}>💡</span>
                    <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', lineHeight: '1.6' }}>
                        {info.action}
                    </p>
                </div>
            </div>

            {/* Disclaimer */}
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '1.2rem', fontStyle: 'italic' }}>
                This tool is for informational purposes only. Always consult a veterinarian for a professional diagnosis.
            </p>

            <button onClick={onReset} className="btn" style={{ background: '#4a5240', border: 'none', color: '#fff', width: '100%' }}>
                Analyze Another Image
            </button>
        </div>
    );
};

// Individual animated probability bar
const ProbBar = ({ label, confidence, isTop, delay }) => {
    const [width, setWidth] = useState(0);
    const percentage = (confidence * 100).toFixed(1);

    useEffect(() => {
        setWidth(0);
        const timer = setTimeout(() => setWidth(parseFloat(percentage)), 100 + delay);
        return () => clearTimeout(timer);
    }, [percentage, delay]);

    return (
        <div style={{ marginBottom: '0.65rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                <span style={{
                    fontSize: '0.9rem',
                    fontWeight: isTop ? '700' : '400',
                    color: isTop ? 'var(--primary-color)' : 'var(--text-main)',
                    textTransform: 'capitalize',
                }}>
                    {isTop ? `${label} ★` : label}
                </span>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>{percentage}%</span>
            </div>
            <div style={{ width: '100%', height: '8px', background: 'rgba(103,114,81,0.15)', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{
                    width: `${width}%`,
                    height: '100%',
                    background: isTop
                        ? 'linear-gradient(to right, var(--primary-color), var(--secondary-color))'
                        : 'rgba(103,114,81,0.4)',
                    borderRadius: '4px',
                    transition: 'width 0.8s ease-out',
                }} />
            </div>
        </div>
    );
};

export default PredictionResult;
