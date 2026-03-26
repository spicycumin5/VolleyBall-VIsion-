import React from 'react';
import { useNavigate } from 'react-router-dom';

function Header({ value, onChange }){
    const navigate = useNavigate();
    return (
        <header className="flex justify-between mb-2 bg-app-dark-blue p-4 sm:p-4 border-b-2 border-slate-900">
            <h2 
                className="cursor-pointer text-orange-300 font-bold text-4xl tracking-tighter"
                onClick={() => {navigate("/home")}}
            >
                VolleyVision
            </h2>
            <input
                type='text'
                id='search'
                name='search'
                onChange={(event) => onChange(event.target.value)}
                value={value}
                className='w-2/5 bg-white border-black rounded-xl p-3'
                placeholder='Search'
            >
            
            </input>
        </header>
    );
}

export default Header;