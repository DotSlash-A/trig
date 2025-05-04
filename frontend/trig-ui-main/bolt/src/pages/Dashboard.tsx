import React from 'react';
import { Link } from 'react-router-dom';
import { FunctionSquare as Function, Hash, Circle, TrendingUp, PlusSquare, ArrowRight } from 'lucide-react';

const Dashboard = () => {
  const tools = [
    {
      title: 'Differentiation',
      description: 'Calculate derivatives with optional step-by-step solutions',
      path: '/differentiation',
      icon: <Function className="h-8 w-8 text-blue-600" />,
      examples: ['sin(x)', 'x^2 * ln(x)', 'e^x * cos(x)']
    },
    {
      title: 'Integration',
      description: 'Find indefinite integrals with optional step-by-step solutions',
      path: '/integration',
      icon: <Hash className="h-8 w-8 text-purple-600" />,
      examples: ['x^2', 'sin(x)', '1/x']
    },
    {
      title: 'Slope Calculator',
      description: 'Calculate the slope between two points',
      path: '/slope',
      icon: <TrendingUp className="h-8 w-8 text-green-600" />,
      examples: ['(1,2) to (5,7)', '(0,0) to (3,4)', '(-2,-3) to (5,1)']
    },
    {
      title: 'Complex Numbers',
      description: 'Perform operations with complex numbers and calculate powers of i',
      path: '/complex',
      icon: <PlusSquare className="h-8 w-8 text-amber-600" />,
      examples: ['2+3i + 1+2i', '(3+4i) × (2-i)', 'i^4']
    },
    {
      title: 'Circle Calculator',
      description: 'Find circle equations from points or calculate details from equations',
      path: '/circle',
      icon: <Circle className="h-8 w-8 text-teal-600" />,
      examples: ['Center (1,2), r=5', 'Three points', 'x^2 + y^2 - 4x - 6y - 12 = 0']
    }
  ];

  return (
    <div className="max-w-7xl mx-auto">
      <div className="text-center mb-12">
        <h1 className="text-3xl md:text-4xl font-bold text-slate-800 mb-4">
          Advanced Mathematics Calculator
        </h1>
        <p className="text-lg text-slate-600 max-w-3xl mx-auto">
          Solve complex calculus problems, work with geometric equations, and perform 
          operations on complex numbers with step-by-step solutions.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {tools.map((tool) => (
          <Link 
            key={tool.path} 
            to={tool.path}
            className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 hover:shadow-md transition-shadow duration-300"
          >
            <div className="flex items-start mb-4">
              <div className="bg-slate-50 rounded-lg p-3 mr-4">{tool.icon}</div>
              <div>
                <h2 className="text-xl font-semibold text-slate-800">{tool.title}</h2>
                <p className="text-slate-600 mt-1">{tool.description}</p>
              </div>
            </div>
            <div className="mt-4">
              <h3 className="text-sm font-medium text-slate-700 mb-2">Example inputs:</h3>
              <div className="flex flex-wrap gap-2">
                {tool.examples.map((example, index) => (
                  <span 
                    key={index} 
                    className="inline-block px-2 py-1 bg-slate-100 text-slate-700 text-sm rounded-md"
                  >
                    {example}
                  </span>
                ))}
              </div>
            </div>
            <div className="mt-4 text-blue-600 font-medium flex items-center justify-end">
              Try it <ArrowRight className="ml-1 h-4 w-4" />
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
};

export default Dashboard;