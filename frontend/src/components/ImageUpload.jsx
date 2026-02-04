import React, { useState, useRef } from 'react';

const ImageUpload = ({ onUpload, isLoading }) => {
    const [preview, setPreview] = useState(null);
    const [dragActive, setDragActive] = useState(false);
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
        // Validate file type
        if (!file.type.startsWith('image/')) {
            alert("Please upload an image file.");
            return;
        }

        const reader = new FileReader();
        reader.onloadend = () => {
            setPreview(reader.result);
            // Automatically trigger upload or wait for button?
            // Let's wait for button for better UX (confirmation)
            onUpload(file, reader.result);
        };
        reader.readAsDataURL(file);
    };

    const onButtonClick = () => {
        inputRef.current.click();
    };

    return (
        <div className="upload-container w-full">
            <div
                className={`upload-area glass-card ${dragActive ? 'drag-active' : ''}`}
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
                    borderColor: dragActive ? 'var(--accent-color)' : 'var(--glass-border)',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease'
                }}
                onClick={onButtonClick}
            >
                <input
                    ref={inputRef}
                    type="file"
                    className="file-input"
                    onChange={handleChange}
                    accept="image/*"
                    style={{ display: 'none' }}
                />

                {preview ? (
                    <div className="preview-container" style={{ position: 'relative', width: '100%', height: '100%', display: 'flex', justifyContent: 'center' }}>
                        <img
                            src={preview}
                            alt="Preview"
                            style={{
                                maxWidth: '100%',
                                maxHeight: '300px',
                                borderRadius: '12px',
                                boxShadow: '0 4px 12px rgba(0,0,0,0.2)'
                            }}
                        />
                        <div style={{
                            position: 'absolute',
                            bottom: '-40px',
                            color: 'var(--text-muted)',
                            fontSize: '0.9rem'
                        }}>
                            Click or Drop to change image
                        </div>
                    </div>
                ) : (
                    <>
                        <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--primary-color)', marginBottom: '1rem' }}>
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="17 8 12 3 7 8"></polyline>
                            <line x1="12" y1="3" x2="12" y2="15"></line>
                        </svg>
                        <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>Upload Feline Image</h3>
                        <p style={{ color: 'var(--text-muted)' }}>Drag & drop or click to browse</p>
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '0.5rem' }}>Supported: JPEG, PNG</p>
                    </>
                )}
            </div>
        </div>
    );
};

export default ImageUpload;
