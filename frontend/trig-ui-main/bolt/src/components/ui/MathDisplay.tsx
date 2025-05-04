import React from 'react';

interface MathDisplayProps {
  expression: string;
  isCode?: boolean;
  className?: string;
}

const MathDisplay: React.FC<MathDisplayProps> = ({ 
  expression, 
  isCode = true,
  className = '' 
}) => {
  return (
    <div className={`${className}`}>
      {isCode ? (
        <pre className="bg-slate-50 rounded-md p-3 overflow-x-auto text-slate-800 font-mono whitespace-pre-wrap">
          {expression}
        </pre>
      ) : (
        <div className="text-slate-800">
          {expression}
        </div>
      )}
    </div>
  );
};

export default MathDisplay;