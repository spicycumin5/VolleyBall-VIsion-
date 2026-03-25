// ./volleyball-vision/src/pages/Dashboard.jsx
// Created March 23, 4:10 PM
// Evan Inrig

import React from 'react';
import Toolbar from '../components/Dashboard/Toolbar';

function Dashboard(){
    return(
        <div className="flex flex-col min-h-screen min-w-screen bg-app-ice-blue gap-0">
            <Toolbar />
            <div className='grow flex flex-row gap-2'>
                <div className='basis-1/5 bg-white'> 02 </div>
                <div className='basis-4/5 bg-white'> 03 </div>
            </div>
        </div>
        );
}

export default Dashboard;