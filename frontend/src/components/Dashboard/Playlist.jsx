import React from 'react';

function Playlist({ title }){
    return (
        <div className='cursor-pointer text-slate-900/70 border-b-2 border-slate-500/50 p-2 bg-transparent hover:bg-blue-200/30 hover:text-slate-900/90 hover:scale-101 transition-all duration-200 text-xl font-semibold'>
            {title}
        </div>
    );
}

export default Playlist;