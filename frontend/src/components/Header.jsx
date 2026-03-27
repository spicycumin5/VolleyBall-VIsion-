import React from 'react';
import { useNavigate } from 'react-router-dom';

function Header({ value, onChange }){
    const navigate = useNavigate();
    return (
        <header className="flex justify-between bg-black py-2 px-4 border-b-2 border-slate-900">
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
                className='w-3/10 m-2 bg-white text-sm border-black rounded-xl p-1 pl-2'
                placeholder='Search'
            >
            
            </input>
        </header>
    );
}

export default Header;