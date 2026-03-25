import React from 'react';
// import videos from '../../assets/videos/volleyball-sample.mp4';
'../../../demo/one_play_boxmot_motion.mp4'
// import videos from '../../../../demo/one_play_boxmot_motion.mp4';

function VideoPlayer({ url }){
    return (
        <div className="pl-2">
            <video className="border-2 border-black-900 bg-slate-900" controls width="80%">
                <source src={url} type="video/mp4" />
                Sorry, your browser doesn't support videos.
            </video>
        </div>
    );
}

export default VideoPlayer;