import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { FunctionSquare as Function, Hash, Circle, TrendingUp, PlusSquare, ChevronLeft, ChevronRight, LayoutDashboard } from 'lucide-react';

const Sidebar = () => {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  const navItems = [
    { 
      path: '/', 
      name: 'Dashboard', 
      icon: <LayoutDashboard size={20} /> 
    },
    { 
      path: '/differentiation', 
      name: 'Differentiation', 
      icon: <Function size={20} /> 
    },
    { 
      path: '/integration', 
      name: 'Integration', 
      icon: <Hash size={20} /> 
    },
    { 
      path: '/slope', 
      name: 'Slope', 
      icon: <TrendingUp size={20} /> 
    },
    { 
      path: '/complex', 
      name: 'Complex Numbers', 
      icon: <PlusSquare size={20} /> 
    },
    { 
      path: '/circle', 
      name: 'Circle', 
      icon: <Circle size={20} /> 
    },
  ];

  const toggleSidebar = () => {
    setCollapsed(!collapsed);
  };

  return (
    <nav 
      className={`bg-white border-r border-slate-200 transition-all duration-300 ${
        collapsed ? 'w-16' : 'w-64'
      } hidden md:block`}
    >
      <div className="h-full flex flex-col">
        <div className="flex-1 py-4 overflow-y-auto">
          <ul className="space-y-1 px-2">
            {navItems.map((item) => (
              <li key={item.path}>
                <Link
                  to={item.path}
                  className={`flex items-center px-3 py-2 rounded-md text-sm font-medium ${
                    location.pathname === item.path
                      ? 'bg-blue-50 text-blue-600'
                      : 'text-slate-700 hover:bg-slate-100 hover:text-blue-600'
                  } transition-colors duration-200`}
                >
                  <span className="mr-3">{item.icon}</span>
                  {!collapsed && <span>{item.name}</span>}
                </Link>
              </li>
            ))}
          </ul>
        </div>
        <div className="p-2 border-t border-slate-200">
          <button
            onClick={toggleSidebar}
            className="w-full flex items-center justify-center p-2 rounded-md text-slate-500 hover:bg-slate-100 hover:text-blue-600 transition-colors duration-200"
          >
            {collapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
          </button>
        </div>
      </div>
    </nav>
  );
};

export default Sidebar;