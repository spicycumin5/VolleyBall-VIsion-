import React from 'react';
import VideoPlayer from '../components/VodReviewPage/VideoPlayer';
import Header from '../components/VodReviewPage/Header';

function VodReviewPage( { videoRef }){

    
    return(
        <div className="min-h-screen min-w-screen bg-app-ice-blue gap-0">
            <div className="flex flex-col">
                <Header />
                <div className="flex flex-row">
                    <div className="flex-1">
                        <VideoPlayer mp4="./assets/videos/volleyball-sample.mp4"/>
                    </div>
                    <div className="flex-1 bg-amber-100">

                    </div>
                </div>
            </div>
        </div>
    );
}


export default VodReviewPage;