import React from 'react';

function Playlist({ title }){
    return (
        <div className='cursor-pointer text-white p-2 bg-transparent hover:bg-blue-200/30 hover:scale-101 transition-all duration-200 text-base font-normal'>
            {title}
        </div>
    );
}

export default Playlist;