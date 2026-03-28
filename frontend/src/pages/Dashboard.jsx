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
import RenameModule from '../components/Dashboard/RenameModule';

function Dashboard(){
    const inputRef = useRef(null);
    const navigate = useNavigate(); 
    const [searchQuery, setSearch] = useState("");
    const [fileSubmit, setSubmit] = useState(false);
    const [sessions, setSessions] = useState(SESSIONS);
    const playlists = sample_playlists;
    const videos = sample_videos;
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [tempFileName, setTempFileName] = useState("");

    const handleFileUploadComplete = (file) => {
        // Triggered after UploadButton finishes its "fake" progress
        setTempFileName(file.name.split('.')[0]); // Default to filename
        setIsModalOpen(true);
    };

    const deleteSession = (key) => {
    if (window.confirm("Are you sure you want to delete this session?")) {
        const updatedSessions = sessions.filter(s => s.key !== key);
        setSessions(updatedSessions);
    }
    };

    const saveNewSession = (newTitle) => {
        const newSession = {
            key: `new-${Date.now()}`, // Unique key
            title: newTitle,
            date: new Date().toLocaleDateString('en-US', { month: 'long', day: 'numeric' }),
            thumbnail: "/thumbnails/one_play.jpg", // Placeholder
            videoSrc: "/videos/one_play.mp4",      // Placeholder
            annotationUrl: "/annotations/one_play.json",
            clips: []
        };

        setSessions([newSession, ...sessions]); // Add to the front of the list
        setIsModalOpen(false);
    };

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
                {sessions
                    .filter(session => (
                        (session.title || "").toLowerCase().includes(searchQuery.toLowerCase()) ||
                        (session.date || "").toLowerCase().includes(searchQuery.toLowerCase())
                    ))
                    .map((session, index) => (
                    <Thumbnail
                        key={session.key}
                        id={index}
                        video={{ src: session.thumbnail, title: session.title, date: session.date }}
                        onClick={() => navigate("/vod", { state: { sessionKey: session.key } })}
                        onDelete={() => deleteSession(session.key)} // Pass the function here
                    /> 
                ))}
            </div>
        </div>
    </div>
    <UploadButton onUploadComplete={handleFileUploadComplete}/>
        <RenameModule 
                isOpen={isModalOpen}
                initialName={tempFileName}
                onSave={saveNewSession}
                onCancel={() => setIsModalOpen(false)}
            />
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