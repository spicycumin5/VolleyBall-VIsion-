import React from 'react';

function UploadBar({ progress=100 }){
    return(
        <div>
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
        </div>
    );
}

export default UploadBar;