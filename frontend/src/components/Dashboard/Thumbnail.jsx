import React from 'react';
import { useNavigate } from 'react-router-dom';


function Thumbnail({ id, video={
    src: "https://www.fivb.com/wp-content/uploads/2025/07/102195-1.jpeg", 
    title: "Replay 01",
    date: "March 25, 10:15 AM",
} }){
    const videoURL = `/thumbnails/thumbnail${(id % 8)}.png`;
    console.log(videoURL);
    const navigate = useNavigate();
    return (
        <div 
            className='cursor-pointer hover:bg-slate-600/10 rounded-xl transition-all duration-150 p-2'
            onClick={() => navigate(`/video/${video.id}`, { 
                state: { 
                videoSrc: video.src, 
                videoName: video.title 
                }
        })}
        >
            <img src={videoURL} className='w-full aspect-video object-cover rounded-xl'/>
            <div className='flex gap-2 items-center justify-between'>
                <h1 className='text-base font-semibold'> {video.title}</h1>
                <h2 className='text-sm font-base'> {video.date} </h2>
            </div>
        </div>
    );
}

export default Thumbnail;
