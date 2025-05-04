import React, { useState } from 'react';
import { Calculator } from 'lucide-react';
import Card from '../components/ui/Card';
import TextField from '../components/ui/TextField';
import Button from '../components/ui/Button';
import MathDisplay from '../components/ui/MathDisplay';
import StepperDisplay from '../components/ui/StepperDisplay';
import { 
  differentiateSimple, 
  differentiateWithSteps, 
  DifferentiateRequest, 
  SimpleDerivativeResponse, 
  StepDerivativeResponse 
} from '../services/api';

const Differentiation = () => {
  const [formData, setFormData] = useState<DifferentiateRequest>({
    expression: '',
    variable: 'x',
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSteps, setShowSteps] = useState(false);
  const [result, setResult] = useState<SimpleDerivativeResponse | null>(null);
  const [stepResult, setStepResult] = useState<StepDerivativeResponse | null>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    
    try {
      if (showSteps) {
        const response = await differentiateWithSteps(formData);
        setStepResult(response);
      } else {
        const response = await differentiateSimple(formData);
        setResult(response);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const examples = [
    'x**2 * sin(x)',
    'exp(x) * cos(x)',
    'ln(x) * x**3',
    'sin(x**2)',
  ];

  const handleUseExample = (example: string) => {
    setFormData({
      ...formData,
      expression: example,
    });
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-800 flex items-center">
          <Calculator className="mr-2 h-6 w-6 text-blue-600" />
          Differentiation Calculator
        </h1>
        <p className="text-slate-600 mt-1">
          Calculate derivatives with respect to a variable, with optional step-by-step solutions.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="md:col-span-2">
          <form onSubmit={handleSubmit}>
            <div className="space-y-4">
              <TextField
                label="Expression"
                name="expression"
                value={formData.expression}
                onChange={handleInputChange}
                placeholder="Enter a mathematical expression (e.g., x**2 * sin(x))"
                required
              />
              <TextField
                label="Variable"
                name="variable"
                value={formData.variable}
                onChange={handleInputChange}
                placeholder="Variable to differentiate with respect to"
                required
              />
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="showSteps"
                  checked={showSteps}
                  onChange={() => setShowSteps(!showSteps)}
                  className="rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                />
                <label htmlFor="showSteps" className="text-sm text-slate-700">
                  Show step-by-step solution
                </label>
              </div>
              <div>
                <Button type="submit" isLoading={isLoading}>
                  Calculate Derivative
                </Button>
              </div>
            </div>
          </form>

          <div className="mt-6">
            <p className="text-sm text-slate-600 mb-2">Try these examples:</p>
            <div className="flex flex-wrap gap-2">
              {examples.map((example) => (
                <button
                  key={example}
                  onClick={() => handleUseExample(example)}
                  className="px-3 py-1 text-sm bg-slate-100 text-slate-700 rounded-md hover:bg-slate-200 transition-colors"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>

          {error && (
            <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-md">
              {error}
            </div>
          )}
        </Card>

        {(result || stepResult) && (
          <Card 
            className="md:col-span-2" 
            title="Derivative Result"
          >
            {showSteps && stepResult ? (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="text-sm font-medium text-slate-700 mb-1">Original Expression</h4>
                    <MathDisplay expression={stepResult.original_expression} />
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-slate-700 mb-1">Final Derivative</h4>
                    <MathDisplay 
                      expression={stepResult.final_derivative}
                      className="bg-blue-50" 
                    />
                  </div>
                </div>
                
                <div className="mt-6">
                  <h4 className="text-sm font-medium text-slate-700 mb-3">Solution Steps</h4>
                  <div className="bg-slate-50 rounded-lg p-4">
                    <StepperDisplay steps={stepResult.steps} />
                  </div>
                </div>
              </div>
            ) : result ? (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="text-sm font-medium text-slate-700 mb-1">Original Expression</h4>
                    <MathDisplay expression={result.original_expression} />
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-slate-700 mb-1">Derivative</h4>
                    <MathDisplay 
                      expression={result.derivative} 
                      className="bg-blue-50"
                    />
                  </div>
                </div>
              </div>
            ) : null}
          </Card>
        )}
      </div>
    </div>
  );
};

export default Differentiation;