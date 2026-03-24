import React from 'react';

function Header(){
    return (
        <header className="flex justify-between mb-2 bg-app-dark-blue shadow-sm p-6 sm:p-8">
            <h2 className="text-orange-300 font-bold text-4xl tracking-tighter">
                VolleyVision
            </h2>
        </header>
    );
}

export default Header;