import React, { useState, useRef } from 'react';

function UploadButton({ handleFile }) {
    const inputRef = useRef(null);

    const [progress, setProgress] = useState(null);

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setProgress(0);
        const interval = setInterval(() => {
            setProgress(prev => {
                if (prev >= 100) {
                    clearInterval(interval);
                    setTimeout(() => setProgress(null), 1000);
                    return 100;
                }
                return prev + 2;
            });
        }, 50);
    };

    return (
        <>
            <input 
                ref={inputRef}
                type='file'
                className='hidden'
                onChange={handleFileChange}
            />

            {progress !== null && (
                <div className='fixed z-50 w-full bg-slate-800 rounded-xl p-3 shadow-lg'>
                    <p className='text-sm text-orange-200 mb-2'>Uploading... {progress}%</p>
                    <div className='w-full bg-slate-600 rounded-full h-2'>
                        <div
                            className='bg-orange-300 h-2 rounded-full transition-all duration-150'
                            style={{ width: `${progress}%` }}
                        />
                    </div>
                </div>
            )}

            <button
                onClick={() => inputRef.current.click()}
                className='fixed cursor-pointer bottom-12 right-12 z-50 w-12 h-12 bg-white rounded-full hover:scale-110 transition-transform duration-150'
            >
                <span className="text-4xl font-normal">+</span>
            </button>
        </>
    );
}

export default UploadButton;