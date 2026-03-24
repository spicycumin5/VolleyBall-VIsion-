import React from 'react';
import videos from '../assets/videos/volleyball-footage.mp4';

function BackgroundVideo(){
    return (
        <video className='absolute z-0 w-auto min-w-full min-h-full max-w-none blur-xs' autoPlay loop muted>
            <source src={videos} type='video/mp4' />
        </video>    
    );
}

export default BackgroundVideo;