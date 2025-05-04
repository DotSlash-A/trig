import React from 'react';
import { Compass } from 'lucide-react';
import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <header className="bg-white border-b border-slate-200 shadow-sm">
      <div className="px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center">
              <Compass className="h-8 w-8 text-blue-600" />
              <span className="ml-2 text-xl font-bold text-slate-800">MathCalc</span>
            </Link>
          </div>
          <div className="hidden md:block">
            <div className="ml-4 flex items-center space-x-4">
              <Link 
                to="/" 
                className="px-3 py-2 rounded-md text-sm font-medium text-slate-700 hover:text-blue-600 hover:bg-slate-100"
              >
                Dashboard
              </Link>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Navbar;