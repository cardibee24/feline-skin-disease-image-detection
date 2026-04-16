import React, { useState } from 'react';

const DISEASE_ICONS = {
    scabies:  '🦠',
    ringworm: '🍄',
    flea:     '🪲',
    healthy:  '✅',
};

const SEVERITY_COLOR = {
    scabies:  '#8b4a3c',
    ringworm: '#a1907c',
    flea:     '#a1907c',
    healthy:  '#677251',
};

const ScanHistory = ({ history, onClear }) => {
    const [open, setOpen] = useState(false);

    // Session summary counts
    const counts = history.reduce((acc, item) => {
        const key = item.prediction.toLowerCase();
        acc[key] = (acc[key] || 0) + 1;
        return acc;
    }, {});

    return (
        <>
            {/* Side toggle button */}
            <button
                onClick={() => setOpen(true)}
                title="Scan History"
                style={{
                    position: 'fixed',
                    left: 0,
                    top: '50%',
                    transform: 'translateY(-50%)',
                    background: 'var(--primary-color)',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '0 12px 12px 0',
                    padding: '0.8rem 0.5rem',
                    cursor: 'pointer',
                    zIndex: 200,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '0.4rem',
                    fontSize: '0.7rem',
                    fontWeight: '600',
                    boxShadow: '2px 0 12px rgba(0,0,0,0.15)',
                    transition: 'background 0.2s',
                    writingMode: 'vertical-rl',
                    letterSpacing: '0.05em',
                }}
            >
                🕓 History {history.length > 0 && `(${history.length})`}
            </button>

            {/* Backdrop */}
            {open && (
                <div
                    onClick={() => setOpen(false)}
                    style={{
                        position: 'fixed',
                        inset: 0,
                        background: 'rgba(0,0,0,0.3)',
                        zIndex: 300,
                        backdropFilter: 'blur(2px)',
                    }}
                />
            )}

            {/* Drawer */}
            <div style={{
                position: 'fixed',
                left: 0,
                top: 0,
                height: '100vh',
                width: '320px',
                background: 'var(--background-start)',
                borderRight: '1px solid var(--glass-border)',
                boxShadow: '4px 0 24px rgba(0,0,0,0.2)',
                zIndex: 400,
                transform: open ? 'translateX(0)' : 'translateX(-100%)',
                transition: 'transform 0.3s ease',
                display: 'flex',
                flexDirection: 'column',
                overflowY: 'auto',
            }}>
                {/* Drawer header */}
                <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '1.2rem 1rem',
                    borderBottom: '1px solid var(--glass-border)',
                    position: 'sticky',
                    top: 0,
                    background: 'var(--background-start)',
                    zIndex: 1,
                }}>
                    <h3 style={{ fontSize: '1rem', fontWeight: '700', color: 'var(--text-main)' }}>
                        🕓 Scan History
                    </h3>
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                        {history.length > 0 && (
                            <button
                                onClick={onClear}
                                style={{
                                    background: 'transparent',
                                    border: '1px solid var(--glass-border)',
                                    borderRadius: '8px',
                                    padding: '0.3rem 0.6rem',
                                    fontSize: '0.78rem',
                                    color: 'var(--text-muted)',
                                    cursor: 'pointer',
                                }}
                            >
                                Clear
                            </button>
                        )}
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
                </div>

                <div style={{ padding: '1rem', flex: 1 }}>
                    {history.length === 0 ? (
                        <div style={{ textAlign: 'center', color: 'var(--text-muted)', marginTop: '3rem', fontSize: '0.9rem' }}>
                            <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🐾</div>
                            No scans yet.<br />Analyze an image to get started.
                        </div>
                    ) : (
                        <>
                            {/* Session summary */}
                            <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
                                {Object.entries(counts).map(([label, count]) => (
                                    <div key={label} style={{
                                        background: 'var(--glass-bg)',
                                        border: '1px solid var(--glass-border)',
                                        borderRadius: '20px',
                                        padding: '0.2rem 0.65rem',
                                        fontSize: '0.78rem',
                                        color: 'var(--text-main)',
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '0.3rem',
                                    }}>
                                        <span>{DISEASE_ICONS[label] || '🔍'}</span>
                                        <span style={{ textTransform: 'capitalize' }}>{label}</span>
                                        <span style={{
                                            background: 'var(--primary-color)',
                                            color: '#fff',
                                            borderRadius: '50%',
                                            width: '16px',
                                            height: '16px',
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            fontSize: '0.65rem',
                                            fontWeight: '700',
                                        }}>{count}</span>
                                    </div>
                                ))}
                            </div>

                            {/* Scan list */}
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                                {history.map((item, index) => (
                                    <div
                                        key={index}
                                        style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '0.75rem',
                                            background: 'var(--glass-bg)',
                                            border: '1px solid var(--glass-border)',
                                            borderRadius: '12px',
                                            padding: '0.65rem 0.75rem',
                                        }}
                                    >
                                        <img
                                            src={item.preview}
                                            alt="scan"
                                            style={{
                                                width: '46px',
                                                height: '46px',
                                                borderRadius: '8px',
                                                objectFit: 'cover',
                                                flexShrink: 0,
                                            }}
                                        />
                                        <div style={{ flex: 1, minWidth: 0 }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                                                <span>{DISEASE_ICONS[item.prediction.toLowerCase()] || '🔍'}</span>
                                                <span style={{
                                                    fontWeight: '700',
                                                    fontSize: '0.9rem',
                                                    color: SEVERITY_COLOR[item.prediction.toLowerCase()] || 'var(--primary-color)',
                                                    textTransform: 'capitalize',
                                                }}>
                                                    {item.prediction}
                                                </span>
                                            </div>
                                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
                                                {item.timestamp}
                                            </div>
                                        </div>
                                        <div style={{
                                            background: item.confidence >= 0.7
                                                ? 'rgba(103,114,81,0.12)'
                                                : 'rgba(161,144,124,0.15)',
                                            color: item.confidence >= 0.7
                                                ? 'var(--primary-color)'
                                                : 'var(--warning)',
                                            borderRadius: '8px',
                                            padding: '0.25rem 0.5rem',
                                            fontSize: '0.8rem',
                                            fontWeight: '600',
                                            flexShrink: 0,
                                        }}>
                                            {(item.confidence * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </>
                    )}
                </div>
            </div>
        </>
    );
};

export default ScanHistory;
