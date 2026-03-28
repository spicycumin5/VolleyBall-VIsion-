import React from 'react';


function Thumbnail({ id, video={
    src: "https://www.fivb.com/wp-content/uploads/2025/07/102195-1.jpeg", 
    title: "Replay 01",
    date: "March 25, 10:15 AM",
}, onClick, onDelete }) {
    const videoURL = `/thumbnails/thumbnail${(id % 8)}.png`;
    return (
        <div 
            className='relative group cursor-pointer hover:bg-slate-600/10 rounded-xl transition-all duration-150 p-2'>
                <button 
                onClick={(e) => {
                    e.stopPropagation(); // Prevents navigating to the VOD page
                    if (onDelete) {
                        onDelete();
                    }
                }}
                className="absolute top-4 right-4 z-20 bg-red-500 text-white w-8 h-8 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-200 shadow-lg hover:bg-red-600 flex items-center justify-center"
                >
                    ✕
                </button>
            <div onClick = {onClick}>
                <img src={videoURL} className='w-full aspect-video object-cover rounded-xl'/>
                <div className='flex gap-2 items-center justify-between'>
                    <h1 className='text-base font-semibold'> {video.title}</h1>
                    <h2 className='text-sm font-base'> {video.date} </h2>
                </div>
            </div>
        </div>
    );
}

export default Thumbnail;
