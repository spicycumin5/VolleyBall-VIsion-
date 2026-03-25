// ./volleyball-vision/src/pages/LandingPage.jsx
// Created March 23, 4:10 PM
// Evan Inrig

import React from 'react';
import BackgroundVideo from '../components/LandingPage/BackgroundVideo';
import { useNavigate } from 'react-router-dom';

function LandingPage(){
    const navigate = useNavigate();
    return (
        <>
            <div id="background-video" className="relative flex flex-col items-center justify-center h-screen overflow-hidden">
                <BackgroundVideo />
                <div className="z-10 p-4 text-center shadow-lg-20">
                    <h1 className="text-orange-300 text-shadow-lg/20 text-8xl font-bold">VolleyVision</h1>
                    <h2 className="text-orange-200/90 text-shadow-lg/20 text-2xl font-semibold">By Evan Inrig, Kyumin Han, and Jason Press</h2>
                </div>
                <button 
                    className="z-10 bg-blue-300 p-5 cursor-pointer rounded-md hover:bg-blue-400 hover:scale-102 transform transition-all duration-150"
                    onClick={() => navigate(`/home`)}
                >
                    <h1 className="text-2xl font-bold">
                        Get Started
                    </h1>
                </button>
            </div>
        </>
    );
}

export default LandingPage;