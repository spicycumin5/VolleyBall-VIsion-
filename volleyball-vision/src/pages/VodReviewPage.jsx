import React from 'react';
import VideoPlayer from '../components/VodReviewPage/VideoPlayer';
import Header from '../components/VodReviewPage/Header';

import VideoAnnotator from '../components/VodReviewPage/VideoAnnotator';
// import annotations from '../../../demo/one_play.json'

function VodReviewPage( { videoRef }){

    
    return(
        <div className="min-h-screen min-w-screen bg-app-ice-blue gap-0">
            <div className="flex flex-col">
                <Header />
                <div className="flex flex-row">
                    <div className="flex-1">
                        <VideoAnnotator url={'/videos/one_play.mp4'} annotations={'./annotations/one_play.json'} />
                        <VideoPlayer url={'/videos/one_play.mp4'} />
                    </div>
                    <div className="flex-1 bg-amber-100">

                    </div>
                </div>
            </div>
        </div>
    );
}


export default VodReviewPage;