import { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import VodReviewPage from './pages/VodReviewPage';
import LandingPage from './pages/LandingPage';
import Dashboard from './pages/Dashboard';

function App() {

  return (
    <BrowserRouter >
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/home" element={<LandingPage />} />
        <Route path="/video" element={<VodReviewPage /> } />
      </Routes>
    </BrowserRouter>
  );
}

export default App
