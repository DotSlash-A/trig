import React, { useState } from 'react';
import { TrendingUp } from 'lucide-react';
import Card from '../components/ui/Card';
import TextField from '../components/ui/TextField';
import Button from '../components/ui/Button';
import { calculateSlope, SlopeRequest, SlopeResponse } from '../services/api';

const Slope = () => {
  const [formData, setFormData] = useState<SlopeRequest>({
    x1: 0,
    y1: 0,
    x2: 0,
    y2: 0,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SlopeResponse | null>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: parseFloat(value) || 0,
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await calculateSlope(formData);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const examples = [
    { x1: 1, y1: 2, x2: 5, y2: 7 },
    { x1: 0, y1: 0, x2: 3, y2: 4 },
    { x1: -2, y1: -3, x2: 5, y2: 1 },
  ];

  const handleUseExample = (example: SlopeRequest) => {
    setFormData(example);
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-800 flex items-center">
          <TrendingUp className="mr-2 h-6 w-6 text-green-600" />
          Slope Calculator
        </h1>
        <p className="text-slate-600 mt-1">
          Calculate the slope between two points on a coordinate plane.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <form onSubmit={handleSubmit}>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <TextField
                  label="X₁"
                  name="x1"
                  type="number"
                  step="any"
                  value={formData.x1.toString()}
                  onChange={handleInputChange}
                  placeholder="X coordinate of point 1"
                  required
                />
                <TextField
                  label="Y₁"
                  name="y1"
                  type="number"
                  step="any"
                  value={formData.y1.toString()}
                  onChange={handleInputChange}
                  placeholder="Y coordinate of point 1"
                  required
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <TextField
                  label="X₂"
                  name="x2"
                  type="number"
                  step="any"
                  value={formData.x2.toString()}
                  onChange={handleInputChange}
                  placeholder="X coordinate of point 2"
                  required
                />
                <TextField
                  label="Y₂"
                  name="y2"
                  type="number"
                  step="any"
                  value={formData.y2.toString()}
                  onChange={handleInputChange}
                  placeholder="Y coordinate of point 2"
                  required
                />
              </div>
              <div>
                <Button type="submit" isLoading={isLoading} variant="outline">
                  Calculate Slope
                </Button>
              </div>
            </div>
          </form>

          <div className="mt-6">
            <p className="text-sm text-slate-600 mb-2">Try these examples:</p>
            <div className="flex flex-wrap gap-2">
              {examples.map((example, index) => (
                <button
                  key={index}
                  onClick={() => handleUseExample(example)}
                  className="px-3 py-1 text-sm bg-slate-100 text-slate-700 rounded-md hover:bg-slate-200 transition-colors"
                >
                  ({example.x1}, {example.y1}) to ({example.x2}, {example.y2})
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

        {result && (
          <Card title="Calculation Result">
            <div className="space-y-4">
              <div>
                <p className="text-sm font-medium text-slate-700 mb-2">
                  Points: ({formData.x1}, {formData.y1}) and ({formData.x2}, {formData.y2})
                </p>

                <div className="flex items-center">
                  <div className="text-lg text-green-600 font-bold">
                    Slope = {result.slope}
                  </div>
                </div>
              </div>

              <div className="mt-4 pt-4 border-t border-slate-200">
                <h4 className="text-sm font-medium text-slate-700 mb-2">Formula</h4>
                <div className="bg-slate-50 p-3 rounded-md text-slate-800">
                  m = (y₂ - y₁) / (x₂ - x₁) = ({formData.y2} - {formData.y1}) / ({formData.x2} - {formData.x1}) = {result.slope}
                </div>
              </div>

              <div className="mt-4">
                <h4 className="text-sm font-medium text-slate-700 mb-2">Point-Slope Form</h4>
                <div className="bg-slate-50 p-3 rounded-md text-slate-800">
                  y - {formData.y1} = {result.slope} ⋅ (x - {formData.x1})
                </div>
              </div>

              <div className="mt-4">
                <h4 className="text-sm font-medium text-slate-700 mb-2">Slope-Intercept Form</h4>
                {(() => {
                  const b = formData.y1 - result.slope * formData.x1;
                  const sign = b >= 0 ? '+' : '';
                  return (
                    <div className="bg-slate-50 p-3 rounded-md text-slate-800">
                      y = {result.slope}x {sign} {b}
                    </div>
                  );
                })()}
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
};

export default Slope;