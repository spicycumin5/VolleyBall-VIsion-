import React from 'react';
import { useNavigate } from 'react-router-dom';


function Thumbnail({ video={
    src: "https://www.fivb.com/wp-content/uploads/2025/07/102195-1.jpeg", 
    title: "Replay 01",
    date: "March 25, 10:15 AM",
} }){
    const navigate = useNavigate();
    return (
        <div 
            className='cursor-pointer hover:bg-blue-800/20 hover:scale-101 rounded-xl transition-all duration-150 p-2'
            onClick={() => navigate(`/video/${video.id}`)}
        >
            <img src={video.src} className='rounded-xl'/>
            <div className='flex gap-2 items-center'>
                <h1 className='text-xl font-semibold'> {video.title}</h1>
                <h2> {video.date} </h2>
            </div>
        </div>
    );
}

export default Thumbnail;
