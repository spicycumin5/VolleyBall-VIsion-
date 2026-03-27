import React, { useState, useRef } from 'react';
import UploadBar from './UploadBar';

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

        e.target.value = null; // reset input so the same file can be uploaded again
    };

    return (
        <>
            <input 
                ref={inputRef}
                type='file'
                className='hidden'
                onChange={handleFileChange}
            />

            <UploadBar progress={progress}/>

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