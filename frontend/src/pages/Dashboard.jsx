// ./volleyball-vision/src/pages/Dashboard.jsx
// Created March 23, 4:10 PM
// Evan Inrig

import React, { useState } from 'react';
import Toolbar from '../components/Dashboard/Toolbar';
import Header from '../components/Header';
import Thumbnail from '../components/Dashboard/Thumbnail';
import Playlist from '../components/Dashboard/Playlist';

function Dashboard(){

    const [searchQuery, setSearch] = useState("");
    
    const playlists = ["All Videos", "Favorites", "Unreviewed", "Processing.."]
    const videos = [{
    id: 0,
    src: "https://www.fivb.com/wp-content/uploads/2025/07/102195-1.jpeg", 
    title: "Replay 01",
    date: "March 25, 10:15 AM",
}, {
    id: 1,
    src: "https://www.fivb.com/wp-content/uploads/2025/07/102195-1.jpeg", 
    title: "Replay 02",
    date: "Jan 25, 10:15 AM",
}, {
    id: 2,
    src: "https://www.fivb.com/wp-content/uploads/2025/07/102195-1.jpeg", 
    title: "Replay 02",
    date: "March 28, 10:15 AM",
}]
    return (
    <div className="flex flex-col h-screen min-w-screen bg-app-ice-blue overflow-hidden">
        
        {/* Fixed Header */}
        <div className="flex-none z-10">
            <Header value={searchQuery} onChange={(value) => {setSearch(value)}}/>
        </div>

        <div className="flex flex-row flex-1 overflow-hidden">

            {/* Fixed Left Bar */}
            <div className="flex-none sm:hidden md:block md:w-1/5 rounded-tr-xl bg-app-dark-blue/70 overflow-y-auto">
                {playlists.map((title,index) => (
                    <Playlist key={index} title={title}/>
                ))}
            </div>

            {/* Scrollable Videos Grid */}
            <div className="flex-1 grid overflow-y-auto sm:grid-cols-2 md:grid-cols-2 xl:grid-cols-3 gap-2 px-4 py-1 content-start">
                {videos
                    .filter(video => (
                        video.title.toLowerCase().includes(searchQuery.toLowerCase()) || video.date.toLowerCase().includes(searchQuery.toLowerCase())
                    ))
                    .map((video, index) => (
                    <Thumbnail key={index} video={video} />
                ))}
            </div>
        </div>
    </div>
);
}

export default Dashboard;