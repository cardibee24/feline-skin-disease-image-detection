import React, { useState, useEffect } from 'react';

const API_URL = 'http://localhost:8000';

const AboutModal = () => {
    const [open, setOpen] = useState(false);
    const [metrics, setMetrics] = useState(null);

    useEffect(() => {
        document.body.style.overflow = open ? 'hidden' : '';
        return () => { document.body.style.overflow = ''; };
    }, [open]);

    useEffect(() => {
        fetch(`${API_URL}/metrics`)
            .then(res => res.json())
            .then(data => { if (!data.error) setMetrics(data); })
            .catch(() => {});
    }, []);

    return (
        <>
            {/* Side button — below history */}
            <button
                onClick={() => setOpen(true)}
                title="About this project"
                style={{
                    position: 'fixed',
                    left: 0,
                    top: 'calc(50% + 80px)',
                    transform: 'translateY(-50%)',
                    background: 'var(--secondary-color)',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '0 12px 12px 0',
                    padding: '0.8rem 0.5rem',
                    cursor: 'pointer',
                    zIndex: 200,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    fontSize: '0.7rem',
                    fontWeight: '600',
                    boxShadow: '2px 0 12px rgba(0,0,0,0.15)',
                    transition: 'background 0.2s',
                    writingMode: 'vertical-rl',
                    letterSpacing: '0.05em',
                }}
            >
                ⓘ About
            </button>

            {/* Backdrop */}
            {open && (
                <div
                    onClick={() => setOpen(false)}
                    style={{
                        position: 'fixed',
                        inset: 0,
                        background: 'rgba(0,0,0,0.4)',
                        zIndex: 300,
                        backdropFilter: 'blur(4px)',
                    }}
                />
            )}

            {/* Modal */}
            {open && (
                <div style={{
                    position: 'fixed',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    width: '90%',
                    maxWidth: '520px',
                    maxHeight: '85vh',
                    background: 'var(--background-start)',
                    border: '1px solid var(--glass-border)',
                    borderRadius: '20px',
                    boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
                    zIndex: 500,
                    animation: 'modalIn 0.25s ease-out',
                    overflow: 'hidden',
                    display: 'flex',
                    flexDirection: 'column',
                }}>
                <div style={{ overflowY: 'auto', padding: '1.5rem', flex: 1 }}>
                    {/* Header */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.2rem' }}>
                        <h2 style={{ fontSize: '1.1rem', fontWeight: '800', color: 'var(--text-main)' }}>
                            🐾 About This Project
                        </h2>
                        <button
                            onClick={() => setOpen(false)}
                            style={{
                                background: 'transparent',
                                border: '1px solid var(--glass-border)',
                                borderRadius: '8px',
                                padding: '0.3rem 0.6rem',
                                fontSize: '0.85rem',
                                color: 'var(--text-muted)',
                                cursor: 'pointer',
                            }}
                        >
                            ✕
                        </button>
                    </div>

                    {/* Project title */}
                    <div style={{
                        background: 'var(--glass-bg)',
                        border: '1px solid var(--glass-border)',
                        borderRadius: '12px',
                        padding: '0.9rem 1rem',
                        marginBottom: '1rem',
                    }}>
                        <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.2rem' }}>FYP Title</p>
                        <p style={{ fontSize: '0.9rem', fontWeight: '600', color: 'var(--text-main)', lineHeight: '1.5' }}>
                            Deep Learning for Feline Disease Detection: A Preliminary Study on Image Classification and Analysis
                        </p>
                        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>
                            Au Ying See · 22ACB05340 · UTAR
                        </p>
                    </div>

                    {/* Model info */}
                    <Section title="🤖 Model">
                        <Row label="Architecture" value="ResNet18 (Transfer Learning)" />
                        <Row label="Pretrained On" value="ImageNet" />
                        <Row label="Classes" value="Scabies · Ringworm · Flea · Healthy" />
                        <Row label="Input Size" value="224 × 224 px" />
                    </Section>

                    {/* Dataset */}
                    <Section title="🗂 Dataset">
                        <Row label="Scabies" value="~641 images" />
                        <Row label="Ringworm" value="~615 images" />
                        <Row label="Flea" value="~694 images" />
                        <Row label="Healthy" value="~642 images" />
                        <Row label="Split" value="80% Train / 20% Validation" />
                    </Section>

                    {/* Preprocessing */}
                    <Section title="⚙️ Preprocessing">
                        <Row label="CLAHE" value="clipLimit=2.0, tileGridSize=(8×8)" />
                        <Row label="Colour Space" value="LAB (applied on L channel)" />
                        <Row label="Normalization" value="ImageNet mean & std" />
                        <Row label="Augmentation" value="Flip · Rotation · Colour jitter" />
                    </Section>

                    {/* Explainability */}
                    <Section title="🔍 Explainability">
                        <Row label="Method" value="Grad-CAM (Gradient-weighted Class Activation Mapping)" />
                        <Row label="Target Layer" value="ResNet18 layer4 (last conv layer)" />
                        <Row label="Purpose" value="Highlights skin regions that influenced the prediction" />
                    </Section>

                    {/* Model Performance — best model */}
                    {metrics && (
                        <Section title="📊 Model Performance (Best Model)">
                            <Row label="Model"            value={metrics.best_model ? metrics.best_model.toUpperCase() : 'ResNet18'} />
                            <Row label="Overall Accuracy" value={`${(metrics.overall_accuracy * 100).toFixed(2)}%`} />
                            <Row label="Macro Precision"  value={`${(metrics.macro_precision * 100).toFixed(2)}%`} />
                            <Row label="Macro Recall"     value={`${(metrics.macro_recall * 100).toFixed(2)}%`} />
                            <Row label="Macro F1-Score"   value={`${(metrics.macro_f1 * 100).toFixed(2)}%`} />
                            {metrics.roc_auc != null && (
                                <Row label="ROC-AUC (macro)" value={`${(metrics.roc_auc * 100).toFixed(2)}%`} />
                            )}
                            {Object.entries(metrics.per_class).map(([cls, m]) => (
                                <Row
                                    key={cls}
                                    label={`${cls.charAt(0).toUpperCase() + cls.slice(1)} F1`}
                                    value={`${(m.f1_score * 100).toFixed(2)}%`}
                                />
                            ))}
                        </Section>
                    )}

                    {/* Model Comparison Table */}
                    {metrics && metrics.comparison && Object.keys(metrics.comparison).length > 1 && (
                        <div style={{ marginBottom: '1rem' }}>
                            <p style={{
                                fontSize: '0.78rem',
                                fontWeight: '700',
                                color: 'var(--primary-color)',
                                textTransform: 'uppercase',
                                letterSpacing: '0.06em',
                                marginBottom: '0.5rem',
                            }}>
                                ⚖️ Model Comparison
                            </p>
                            <div style={{ overflowX: 'auto' }}>
                                <table style={{
                                    width: '100%',
                                    borderCollapse: 'collapse',
                                    fontSize: '0.8rem',
                                    background: 'var(--glass-bg)',
                                    border: '1px solid var(--glass-border)',
                                    borderRadius: '12px',
                                    overflow: 'hidden',
                                }}>
                                    <thead>
                                        <tr style={{ background: 'rgba(103,114,81,0.12)' }}>
                                            {['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'].map(h => (
                                                <th key={h} style={{
                                                    padding: '0.5rem 0.7rem',
                                                    textAlign: 'center',
                                                    color: 'var(--text-muted)',
                                                    fontWeight: '600',
                                                    borderBottom: '1px solid var(--glass-border)',
                                                }}>
                                                    {h}
                                                </th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {Object.entries(metrics.comparison).map(([arch, m]) => {
                                            const isBest = arch === metrics.best_model;
                                            return (
                                                <tr key={arch} style={{
                                                    background: isBest ? 'rgba(103,114,81,0.1)' : 'transparent',
                                                    borderBottom: '1px solid var(--glass-border)',
                                                }}>
                                                    <td style={{ padding: '0.45rem 0.7rem', textAlign: 'center', fontWeight: isBest ? '700' : '400', color: isBest ? 'var(--primary-color)' : 'var(--text-main)' }}>
                                                        {arch.toUpperCase()}{isBest ? ' ★' : ''}
                                                    </td>
                                                    <td style={{ padding: '0.45rem 0.7rem', textAlign: 'center', color: 'var(--text-main)' }}>{(m.overall_accuracy * 100).toFixed(2)}%</td>
                                                    <td style={{ padding: '0.45rem 0.7rem', textAlign: 'center', color: 'var(--text-main)' }}>{(m.macro_precision * 100).toFixed(2)}%</td>
                                                    <td style={{ padding: '0.45rem 0.7rem', textAlign: 'center', color: 'var(--text-main)' }}>{(m.macro_recall * 100).toFixed(2)}%</td>
                                                    <td style={{ padding: '0.45rem 0.7rem', textAlign: 'center', color: 'var(--text-main)' }}>{(m.macro_f1 * 100).toFixed(2)}%</td>
                                                    <td style={{ padding: '0.45rem 0.7rem', textAlign: 'center', color: 'var(--text-main)' }}>{m.roc_auc != null ? `${(m.roc_auc * 100).toFixed(2)}%` : '—'}</td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                            <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>
                                ★ = deployed model &nbsp;|&nbsp; Evaluated on 20% held-out validation set
                            </p>
                        </div>
                    )}


                </div>
                </div>
            )}
        </>
    );
};

const Section = ({ title, children }) => (
    <div style={{ marginBottom: '1rem' }}>
        <p style={{
            fontSize: '0.78rem',
            fontWeight: '700',
            color: 'var(--primary-color)',
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
            marginBottom: '0.5rem',
        }}>
            {title}
        </p>
        <div style={{
            background: 'var(--glass-bg)',
            border: '1px solid var(--glass-border)',
            borderRadius: '12px',
            overflow: 'hidden',
        }}>
            {children}
        </div>
    </div>
);

const Row = ({ label, value }) => (
    <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        padding: '0.55rem 0.9rem',
        borderBottom: '1px solid var(--glass-border)',
        gap: '1rem',
    }}>
        <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)', flexShrink: 0 }}>{label}</span>
        <span style={{ fontSize: '0.82rem', color: 'var(--text-main)', textAlign: 'right' }}>{value}</span>
    </div>
);

export default AboutModal;
