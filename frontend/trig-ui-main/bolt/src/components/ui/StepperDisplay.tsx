import React from 'react';
import { CheckCircle } from 'lucide-react';
import MathDisplay from './MathDisplay';

interface Step {
  step_description: string;
  expression_pretty: string;
  expression_latex?: string;
}

interface StepperDisplayProps {
  steps: Step[];
}

const StepperDisplay: React.FC<StepperDisplayProps> = ({ steps }) => {
  return (
    <div className="space-y-6">
      {steps.map((step, index) => (
        <div key={index} className="flex">
          <div className="mr-4 flex flex-col items-center">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-100 text-blue-600">
              {index + 1}
            </div>
            {index < steps.length - 1 && (
              <div className="h-full w-0.5 bg-blue-200"></div>
            )}
          </div>
          <div className="pt-1 pb-8">
            <h3 className="text-md font-medium text-slate-800 mb-2">
              {step.step_description}
            </h3>
            {step.expression_pretty && (
              <MathDisplay 
                expression={step.expression_pretty} 
                className="mt-2" 
              />
            )}
          </div>
        </div>
      ))}
      <div className="flex">
        <div className="mr-4">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-green-100">
            <CheckCircle className="h-5 w-5 text-green-600" />
          </div>
        </div>
        <div className="pt-1">
          <h3 className="text-md font-medium text-slate-800">Calculation complete</h3>
        </div>
      </div>
    </div>
  );
};

export default StepperDisplay;