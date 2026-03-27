import React from 'react';

function Header(){
    return (
        <header className="flex justify-between mb-2 bg-app-dark-blue p-4 sm:p-4 border-b-2 border-slate-900">
            <h2 className="text-orange-300 font-bold text-4xl tracking-tighter">
                VolleyVision
            </h2>
        </header>
    );
}

export default Header;