import React, { useState, useRef } from 'react';
import CameraCapture from './CameraCapture';

const ImageUpload = ({ onUpload, isLoading }) => {
    const [preview, setPreview] = useState(null);
    const [dragActive, setDragActive] = useState(false);
    const [isCameraMode, setIsCameraMode] = useState(false);

    const inputRef = useRef(null);

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    };

    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    };

    const handleFile = (file) => {
        if (!file.type.startsWith('image/')) {
            alert("Please upload an image file.");
            return;
        }

        const reader = new FileReader();
        reader.onloadend = () => {
            setPreview(reader.result);
            onUpload(file, reader.result);
        };
        reader.readAsDataURL(file);
    };

    const handleCameraCapture = (file, previewUrl) => {
        setPreview(previewUrl);
        setIsCameraMode(false);
        onUpload(file, previewUrl);
    };

    const onButtonClick = () => {
        inputRef.current.click();
    };

    if (isCameraMode) {
        return (
            <div className="upload-container w-full glass-card p-4">
                <CameraCapture
                    onCapture={handleCameraCapture}
                    onCancel={() => setIsCameraMode(false)}
                />
            </div>
        );
    }

    return (
        <div className="upload-container w-full">

            <div
                className="upload-area glass-card"
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
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
                    position: 'relative',
                }}
            >
                <input
                    ref={inputRef}
                    type="file"
                    onChange={handleChange}
                    accept="image/*"
                    style={{ display: 'none' }}
                />

                {preview ? (
                    <div style={{ position: 'relative', width: '100%', display: 'flex', justifyContent: 'center' }}>
                        <div style={{ position: 'relative', display: 'inline-block' }}>
                            <img
                                src={preview}
                                alt="Preview"
                                style={{
                                    maxWidth: '100%',
                                    maxHeight: '260px',
                                    borderRadius: '12px',
                                    boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
                                    display: 'block',
                                }}
                            />
                        </div>
                        <button
                            className="btn"
                            style={{
                                position: 'absolute',
                                bottom: '-50px',
                                background: '#4a5240',
                            }}
                            onClick={() => {
                                setPreview(null);
                                onUpload(null, null);
                            }}
                        >
                            Reset
                        </button>
                    </div>
                ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                        <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--primary-color)', marginBottom: '1rem', transition: 'transform 0.2s' }}>
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="17 8 12 3 7 8"></polyline>
                            <line x1="12" y1="3" x2="12" y2="15"></line>
                        </svg>
                        <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>
                            {dragActive ? 'Drop it!' : 'Upload Feline Image'}
                        </h3>
                        <p style={{ color: 'var(--text-muted)' }}>
                            {dragActive
                                ? 'Release to upload'
                                : <>Drag & drop or <span onClick={onButtonClick} style={{ color: 'var(--primary-color)', cursor: 'pointer', textDecoration: 'underline' }}>browse</span></>
                            }
                        </p>

                        <div style={{ borderBottom: '1px solid var(--glass-border)', width: '50%', margin: '1rem 0' }}></div>

                        <button
                            className="btn"
                            style={{ marginTop: '0.5rem' }}
                            onClick={(e) => {
                                e.stopPropagation();
                                setIsCameraMode(true);
                            }}
                        >
                            <span style={{ marginRight: '8px' }}>📷</span> Take Photo
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
};

export default ImageUpload;
