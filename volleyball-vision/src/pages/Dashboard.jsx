// ./volleyball-vision/src/pages/Dashboard.jsx
// Created March 23, 4:10 PM
// Evan Inrig

import React from 'react';
import Toolbar from '../components/Dashboard/Toolbar';
import Thumbnail from '../components/Dashboard/Thumbnail';

function Dashboard(){
    return (
    <div className="flex flex-col h-screen min-w-screen bg-app-ice-blue overflow-hidden">
        
        {/* Fixed Header */}
        <div className="flex-none z-10">
            <Toolbar />
        </div>

        <div className="flex flex-row flex-1 overflow-hidden">

            {/* Fixed Left Bar */}
            <div className="flex-none sm:hidden md:block md:w-1/5 rounded-tr-xl bg-white overflow-y-auto">
                02
            </div>

            {/* Scrollable Videos Grid */}
            <div className="flex-1 grid overflow-y-auto sm:grid-cols-2 md:grid-cols-2 xl:grid-cols-3 gap-2 px-4 py-1 content-start">
                <Thumbnail />
                <Thumbnail />
                <Thumbnail />
                <Thumbnail />
                <Thumbnail />
                <Thumbnail />
                <Thumbnail />
                <Thumbnail />
                <Thumbnail />
                <Thumbnail />
                <Thumbnail />
                <Thumbnail />
            </div>
        </div>
    </div>
);
}

export default Dashboard;