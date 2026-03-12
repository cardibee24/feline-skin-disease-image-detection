import React, { useRef, useState, useCallback } from 'react';
import Webcam from 'react-webcam';

const videoConstraints = {
    width: 720,
    height: 720,
    facingMode: "user"
};

const CameraCapture = ({ onCapture, onCancel }) => {
    const webcamRef = useRef(null);
    const [imgSrc, setImgSrc] = useState(null);

    const capture = useCallback(() => {
        const imageSrc = webcamRef.current.getScreenshot();
        setImgSrc(imageSrc);
    }, [webcamRef]);

    const retake = () => {
        setImgSrc(null);
    };

    const confirm = () => {
        if (imgSrc) {
            // Convert base64 to File object
            fetch(imgSrc)
                .then(res => res.blob())
                .then(blob => {
                    const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
                    onCapture(file, imgSrc);
                });
        }
    };

    return (
        <div className="camera-container flex flex-col items-center w-full">
            {!imgSrc ? (
                <div className="webcam-wrapper rounded-xl overflow-hidden shadow-lg border-2 border-[var(--primary-color)] bg-black">
                    <Webcam
                        audio={false}
                        height={300}
                        ref={webcamRef}
                        screenshotFormat="image/jpeg"
                        width={400}
                        videoConstraints={videoConstraints}
                        style={{ display: 'block' }}
                    />
                </div>
            ) : (
                <div className="preview-wrapper rounded-xl overflow-hidden shadow-lg border-2 border-[var(--accent-color)]">
                    <img src={imgSrc} alt="captured" style={{ maxHeight: '300px', maxWidth: '100%' }} />
                </div>
            )}

            <div className="controls mt-4 flex gap-4">
                {!imgSrc ? (
                    <>
                        <button className="btn btn-secondary" onClick={onCancel}>
                            Cancel
                        </button>
                        <button className="btn" onClick={capture}>
                            Capture Photo
                        </button>
                    </>
                ) : (
                    <>
                        <button className="btn btn-secondary" onClick={retake}>
                            Retake
                        </button>
                        <button className="btn" onClick={confirm}>
                            Use Photo
                        </button>
                    </>
                )}
            </div>
        </div>
    );
};

export default CameraCapture;
