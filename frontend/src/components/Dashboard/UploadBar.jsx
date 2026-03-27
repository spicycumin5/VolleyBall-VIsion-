import React from 'react';

function UploadBar({ progress=100 }){
    return(
        <div>
            {progress !== null && (
                <div className='fixed bottom-0 left-0 right-0 z-50 bg-black p-1 shadow-lg'>
                    <p className='text-sm text-orange-200 mb-2 text-center'>
                        {progress < 100 ? `Uploading... ${progress}%` : 'Done!'}
                    </p>
                    <div className='w-full bg-slate-600 h-1'>
                        <div
                            className='bg-orange-300 h-1 transition-all rounded-sm duration-150'
                            style={{ width: `${progress}%` }}
                        />
                    </div>
                </div>
            )}
        </div>
    );
}

export default UploadBar;