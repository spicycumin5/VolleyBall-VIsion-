// ./volleyball-vision/src/pages/Dashboard.jsx
// Created March 23, 4:10 PM
// Evan Inrig

import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import Toolbar from '../components/Dashboard/Toolbar';
import Header from '../components/Header';
import Thumbnail from '../components/Dashboard/Thumbnail';
import Playlist from '../components/Dashboard/Playlist';
import UploadButton from '../components/Dashboard/UploadButton';
import sample_videos from '../assets/tests/sample_videos';
import sample_playlists from '../assets/tests/sample_playlists';
import UploadBar from '../components/Dashboard/UploadBar';
import SESSIONS from '../assets/videos/sessions.js';

function Dashboard(){
    const inputRef = useRef(null);
    const navigate = useNavigate(); 
    const [searchQuery, setSearch] = useState("");
    const [fileSubmit, setSubmit] = useState(false);
    
    const playlists = sample_playlists;
    const videos = sample_videos;
    return (
    <div>
    <div className="flex flex-col h-screen min-w-screen bg-white overflow-hidden">
        {/* Fixed Header */}
        <div className="flex-none z-10">
            <Header value={searchQuery} onChange={(value) => {setSearch(value)}}/>
        </div>

        <div className="flex flex-row flex-1 overflow-hidden">

            {/* Fixed Left Bar */}
            <div className="flex-none sm:hidden md:block md:w-1/5 bg-black/90 overflow-y-auto">
                {playlists.map((title,index) => (
                    <Playlist key={index} title={title}/>
                ))}
            </div>

            {/* Scrollable Videos Grid */}
            <div className="flex-1 grid overflow-y-auto sm:grid-cols-2 md:grid-cols-2 xl:grid-cols-3 p-2 content-start">
                {SESSIONS
                    .filter(session => (
                        session.title.toLowerCase().includes(searchQuery.toLowerCase()) || session.date.toLowerCase().includes(searchQuery.toLowerCase())
                    ))
                    .map((session, index) => (
                    <Thumbnail
                        key={index}
                        id={index}
                        video={{ src: session.thumbnail, title: session.title, date: session.date }}
                        onClick={() => navigate("/vod", { state: { sessionKey: session.key } })}  
                    /> ))}
            </div>
        </div>
    </div>
    <UploadButton handleFile={(file) => {}}/>
    {/* <UploadBar /> */}
    {fileSubmit && (
            <div>
                <input type="file">
                </input>
            </div>
        )}
    </div>
);
}

export default Dashboard;