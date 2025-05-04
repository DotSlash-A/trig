import React from 'react';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  title?: string;
  description?: string;
}

const Card: React.FC<CardProps> = ({ 
  children, 
  className = '', 
  title, 
  description 
}) => {
  return (
    <div className={`bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden ${className}`}>
      {(title || description) && (
        <div className="px-6 py-4 border-b border-slate-200">
          {title && <h3 className="text-lg font-semibold text-slate-800">{title}</h3>}
          {description && <p className="mt-1 text-sm text-slate-600">{description}</p>}
        </div>
      )}
      <div className="p-6">{children}</div>
    </div>
  );
};

export default Card;