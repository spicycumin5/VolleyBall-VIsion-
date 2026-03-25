import React from 'react';

function Thumbnail({ video={
    src: "https://www.fivb.com/wp-content/uploads/2025/07/102195-1.jpeg", 
    title: "Replay 01",
    date: "March 25, 10:15 AM",
} }){
    return (
        <div className='hover:bg-blue-800/20 rounded-xl transition-all duration-150 p-2'>
            <img src={video.src} className='rounded-xl'/>
            <div>
                <h1> {video.title}</h1>
                <h2> {video.date} </h2>
            </div>
        </div>
    );
}

export default Thumbnail;
