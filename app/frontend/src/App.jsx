import React from 'react';
import VideoUploader from './VideoUploader';
import './App.css';

function App() {
  return (
    <div className="min-h-screen bg-gray-950 text-white font-sans flex flex-col">
      
      <header className="px-6 py-4">
        <h1 className="text-green-400 text-2xl font-bold">
           TennUS: Tu analista personal
        </h1>
      </header>

      <div className="flex-grow flex items-center justify-center px-4">
        <VideoUploader />
      </div>
    </div>
  );
}

export default App;
