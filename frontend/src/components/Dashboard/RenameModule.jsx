import React, { useState, useEffect } from 'react';

function RenameModule({ isOpen, initialName, onSave, onCancel }) {
    const [title, setTitle] = useState(initialName);

    useEffect(() => {
        if (isOpen) {
            setTitle(initialName);
        }
    }, [isOpen, initialName]);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <div className="bg-white p-8 rounded-2xl shadow-2xl w-96 flex flex-col gap-4">
                <h2 className="text-2xl font-bold text-slate-800">New Session</h2>
                <p className="text-sm text-slate-500">Give your practice session a name:</p>
                
                <input 
                    type="text" 
                    value={title} 
                    onChange={(e) => setTitle(e.target.value)}
                    className="w-full p-3 border-2 border-slate-200 rounded-lg focus:border-blue-400 outline-none"
                    autoFocus
                />

                <div className="flex gap-3 mt-4">
                    <button 
                        onClick={onCancel}
                        className="flex-1 px-4 py-2 bg-slate-100 text-slate-600 rounded-lg hover:bg-slate-200 transition-colors"
                    >
                        Cancel
                    </button>
                    <button 
                        onClick={() => onSave(title)}
                        className="flex-1 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-semibold"
                    >
                        Save
                    </button>
                </div>
            </div>
        </div>
    );
}

export default RenameModule;