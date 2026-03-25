import React, { useEffect, useRef } from 'react';
import videos from '../../assets/videos/volleyball-footage.mp4';

function BackgroundVideo(){
    const videoRef = useRef(null);

    useEffect(() => {
        if (videoRef.current) {
            videoRef.current.playbackRate = 0.65;
        }
    }, []);

    return (
        <video ref={videoRef} className='absolute z-0 w-auto min-w-full min-h-full max-w-none blur-xs' autoPlay loop muted>
            <source src={videos} type='video/mp4' />
        </video>    
    );
}

export default BackgroundVideo;