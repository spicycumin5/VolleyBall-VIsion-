import React from 'react';
import videos from '../assets/videos/volleyball-sample.mp4';


function VideoPlayer({ mp4, webm }){
    return (
        <div className="pl-2">
            <video className="border-2 border-black-900 bg-slate-900" controls width="80%">
                <source src={videos} type="video/mp4" />
                {/* <source src={url.webm} type="video/webm" /> */}
                Sorry, your browser doesn't support videos.
            </video>
        </div>
    );
}

export default VideoPlayer;