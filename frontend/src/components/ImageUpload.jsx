import React, { useState, useRef } from 'react';
import CameraCapture from './CameraCapture';

const MAX_IMAGES = 6;
let _id = 0;
const nextId = () => ++_id;

const ImageUpload = ({ onUpload, isLoading }) => {
    const [items, setItems] = useState([]); // [{id, file, url}]
    const [dragActive, setDragActive] = useState(false);
    const [isCameraMode, setIsCameraMode] = useState(false);
    const inputRef = useRef(null);

    const notify = (nextItems) => {
        onUpload(nextItems.map(i => i.file), nextItems.map(i => i.url));
    };

    const processFiles = (fileList) => {
        const candidates = Array.from(fileList).filter(f => {
            if (!f.type.startsWith('image/')) { alert(`"${f.name}" is not an image.`); return false; }
            if (f.size > 10 * 1024 * 1024) { alert(`"${f.name}" exceeds the 10 MB limit.`); return false; }
            return true;
        });

        Promise.all(candidates.map(file => new Promise(resolve => {
            const reader = new FileReader();
            reader.onloadend = () => {
                const img = new window.Image();
                img.onload = () => {
                    if (img.width < 64 || img.height < 64) {
                        alert(`"${file.name}" is too small (min 64×64 px).`);
                        resolve(null);
                    } else {
                        resolve({ id: nextId(), file, url: reader.result });
                    }
                };
                img.src = reader.result;
            };
            reader.readAsDataURL(file);
        }))).then(loaded => {
            const valid = loaded.filter(Boolean);
            setItems(prev => {
                const slots = MAX_IMAGES - prev.length;
                if (slots <= 0) { alert(`Maximum ${MAX_IMAGES} images already selected.`); return prev; }
                if (valid.length > slots) alert(`Only ${slots} slot(s) remaining — first ${slots} image(s) added.`);
                const next = [...prev, ...valid.slice(0, slots)];
                notify(next);
                return next;
            });
        });
    };

    const addCameraCapture = (file, url) => {
        setItems(prev => {
            if (prev.length >= MAX_IMAGES) { alert(`Maximum ${MAX_IMAGES} images already selected.`); return prev; }
            const next = [...prev, { id: nextId(), file, url }];
            notify(next);
            return next;
        });
    };

    const removeItem = (id) => {
        setItems(prev => {
            const next = prev.filter(i => i.id !== id);
            notify(next);
            return next;
        });
    };

    const clearAll = () => {
        setItems([]);
        onUpload([], []);
    };

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(e.type === 'dragenter' || e.type === 'dragover');
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files?.length) processFiles(e.dataTransfer.files);
    };

    const handleChange = (e) => {
        if (e.target.files?.length) processFiles(e.target.files);
        e.target.value = '';
    };

    if (isCameraMode) {
        return (
            <div className="upload-container w-full glass-card p-4">
                <CameraCapture
                    onCapture={(file, url) => { setIsCameraMode(false); addCameraCapture(file, url); }}
                    onCancel={() => setIsCameraMode(false)}
                />
            </div>
        );
    }

    const canAddMore = items.length < MAX_IMAGES && !isLoading;

    return (
        <div className="upload-container w-full">
            <div
                className="upload-area glass-card"
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={canAddMore ? handleDrop : undefined}
                style={{
                    minHeight: '300px',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    borderStyle: 'dashed',
                    borderWidth: '2px',
                    borderColor: dragActive ? 'var(--primary-color)' : 'var(--glass-border)',
                    background: dragActive ? 'rgba(103,114,81,0.12)' : 'var(--glass-bg)',
                    transform: dragActive ? 'scale(1.02)' : 'scale(1)',
                    boxShadow: dragActive
                        ? '0 0 0 4px rgba(103,114,81,0.25), 0 8px 32px rgba(103,114,81,0.2)'
                        : '0 8px 32px 0 rgba(103,114,81,0.15)',
                    transition: 'all 0.25s ease',
                    padding: '1.5rem',
                    position: 'relative',
                }}
            >
                {items.length > 0 && !isLoading && (
                    <button
                        onClick={clearAll}
                        title="Clear all"
                        style={{
                            position: 'absolute',
                            top: '10px',
                            right: '10px',
                            width: '28px',
                            height: '28px',
                            borderRadius: '50%',
                            background: 'rgba(0,0,0,0.45)',
                            border: 'none',
                            color: '#fff',
                            fontSize: '16px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            zIndex: 10,
                            lineHeight: 1,
                            padding: 0,
                        }}
                    >
                        ×
                    </button>
                )}
                <input
                    ref={inputRef}
                    type="file"
                    onChange={handleChange}
                    accept="image/*"
                    multiple
                    style={{ display: 'none' }}
                />

                {items.length > 0 ? (
                    <div style={{ width: '100%' }}>
                        {/* Thumbnail grid */}
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(2, 1fr)',
                            gap: '0.75rem',
                            marginBottom: '1rem',
                        }}>
                            {items.map((item, idx) => (
                                <div key={item.id} style={{ position: 'relative', borderRadius: '10px', overflow: 'hidden', aspectRatio: '4/3', boxShadow: '0 2px 8px rgba(0,0,0,0.18)' }}>
                                    <img
                                        src={item.url}
                                        alt={`Image ${idx + 1}`}
                                        style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                                    />
                                    {!isLoading && (
                                        <button
                                            onClick={() => removeItem(item.id)}
                                            title="Remove"
                                            style={{
                                                position: 'absolute',
                                                top: '4px',
                                                right: '4px',
                                                width: '22px',
                                                height: '22px',
                                                borderRadius: '50%',
                                                background: 'rgba(0,0,0,0.6)',
                                                border: 'none',
                                                color: '#fff',
                                                fontSize: '14px',
                                                cursor: 'pointer',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                lineHeight: 1,
                                                padding: 0,
                                            }}
                                        >
                                            ×
                                        </button>
                                    )}
                                    <div style={{
                                        position: 'absolute',
                                        bottom: '3px',
                                        left: '5px',
                                        background: 'rgba(0,0,0,0.5)',
                                        color: '#fff',
                                        fontSize: '10px',
                                        borderRadius: '4px',
                                        padding: '1px 4px',
                                    }}>
                                        {idx + 1}
                                    </div>
                                </div>
                            ))}
                        </div>

                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '0.25rem' }}>
                            {/* slot counter */}
                            <span style={{
                                fontSize: '0.78rem',
                                color: 'var(--text-muted)',
                                background: 'rgba(103,114,81,0.1)',
                                borderRadius: '20px',
                                padding: '3px 10px',
                            }}>
                                {items.length} / {MAX_IMAGES}
                            </span>

                            {/* add more button */}
                            {canAddMore && (
                                <button
                                    onClick={() => inputRef.current.click()}
                                    style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '5px',
                                        background: 'rgba(103,114,81,0.12)',
                                        border: '1px dashed var(--primary-color)',
                                        borderRadius: '8px',
                                        color: 'var(--primary-color)',
                                        fontSize: '0.82rem',
                                        fontWeight: '600',
                                        cursor: 'pointer',
                                        padding: '5px 12px',
                                        transition: 'background 0.2s',
                                    }}
                                    onMouseEnter={e => e.currentTarget.style.background = 'rgba(103,114,81,0.22)'}
                                    onMouseLeave={e => e.currentTarget.style.background = 'rgba(103,114,81,0.12)'}
                                >
                                    <span style={{ fontSize: '1rem', lineHeight: 1 }}>+</span> Add photos
                                </button>
                            )}
                        </div>

                    </div>
                ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                        <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--primary-color)', marginBottom: '1rem' }}>
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="17 8 12 3 7 8"></polyline>
                            <line x1="12" y1="3" x2="12" y2="15"></line>
                        </svg>
                        <h3 style={{ fontSize: '1.25rem', marginBottom: '0.4rem' }}>
                            {dragActive ? 'Drop it!' : 'Upload Feline Images'}
                        </h3>

                        {dragActive ? (
                            <p style={{ color: 'var(--primary-color)', fontWeight: '600' }}>Release to upload</p>
                        ) : (
                            <>
                                <div style={{ display: 'flex', gap: '0.65rem', alignItems: 'center', marginBottom: '0.85rem' }}>
                                    <button
                                        onClick={() => inputRef.current.click()}
                                        style={{
                                            display: 'flex', alignItems: 'center', gap: '6px',
                                            background: 'var(--primary-color)', border: 'none',
                                            borderRadius: '10px', color: '#fff',
                                            fontSize: '0.88rem', fontWeight: '600',
                                            cursor: 'pointer', padding: '0.55rem 1.2rem',
                                            boxShadow: '0 2px 8px rgba(103,114,81,0.3)',
                                            transition: 'opacity 0.2s',
                                        }}
                                        onMouseEnter={e => e.currentTarget.style.opacity = '0.85'}
                                        onMouseLeave={e => e.currentTarget.style.opacity = '1'}
                                    >
                                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
                                        </svg>
                                        Browse
                                    </button>

                                    <button
                                        onClick={(e) => { e.stopPropagation(); setIsCameraMode(true); }}
                                        style={{
                                            display: 'flex', alignItems: 'center', gap: '6px',
                                            background: 'transparent',
                                            border: '1.5px solid var(--glass-border)',
                                            borderRadius: '10px', color: 'var(--text-main)',
                                            fontSize: '0.88rem', fontWeight: '600',
                                            cursor: 'pointer', padding: '0.55rem 1.2rem',
                                            transition: 'border-color 0.2s',
                                        }}
                                        onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--primary-color)'}
                                        onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--glass-border)'}
                                    >
                                        📷 Camera
                                    </button>
                                </div>

                                <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                                    or drag & drop · up to {MAX_IMAGES} images · max 10 MB
                                </p>
                            </>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default ImageUpload;
