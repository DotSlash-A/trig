import React from 'react';
import { Compass } from 'lucide-react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import Navbar from './components/layout/Navbar';
import Sidebar from './components/layout/Sidebar';
import Dashboard from './pages/Dashboard';
import Differentiation from './pages/Differentiation';
import Integration from './pages/Integration';
import Slope from './pages/Slope';
import ComplexNumbers from './pages/ComplexNumbers';
import Circle from './pages/Circle';
import Prog from './pages/Prog'; // Ensure the file exists at ./pages/Prog.tsx

function App() {
  return (
    <Router>
      <div className="flex flex-col min-h-screen bg-slate-50">
        <Navbar />
        <div className="flex flex-1 overflow-hidden">
          <Sidebar />
          <main className="flex-1 overflow-y-auto p-4 md:p-6">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/differentiation" element={<Differentiation />} />
              <Route path="/integration" element={<Integration />} />
              <Route path="/slope" element={<Slope />} />
              <Route path="/complex" element={<ComplexNumbers />} />
              <Route path="/circle" element={<Circle />} />
              <Route path='/prog' element={<Prog />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;