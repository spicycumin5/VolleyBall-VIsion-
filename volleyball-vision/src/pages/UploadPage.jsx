import React from 'react';
import VideoPlayer from '../components/VideoPlayer';
import Header from '../components/Header';

function UploadPage(){
    return(
        <div className="min-h-screen min-w-screen bg-blue-400 gap-0">
            <div className="flex flex-col">
                <Header />
                <div className="flex flex-row">
                    <div className="flex-1">
                        <h2> Upload files here: </h2>
                        <VideoPlayer mp4="./assets/videos/volleyball-sample.mp4"/>
                    </div>
                    <div className="flex-1 bg-amber-100">
                        
                    </div>
                </div>
            </div>
        </div>
    );
}


export default UploadPage;