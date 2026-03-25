import React from 'react';

function Thumbnail({ video }){
    return (
        <div className='grid'>
            <img></img>
            <div>
                <h1> {video.title}</h1>
                <h2> {video.date} </h2>
            </div>
        </div>
    );
}

export default Thumbnail;