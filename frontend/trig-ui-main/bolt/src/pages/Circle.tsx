import React, { useState } from 'react';
import { Circle as CircleIcon } from 'lucide-react';
import Card from '../components/ui/Card';
import TextField from '../components/ui/TextField';
import Button from '../components/ui/Button';
import MathDisplay from '../components/ui/MathDisplay';
import { 
  calculateCircleEquation, 
  getCircleDetails,
  calculateCircleFrom3Points,
  calculateCircleFrom3PointsDet,
  CircleEqnRequest,
  CircleEqnResponse,
  CircleDetailsRequest,
  CircleDetailsResponse,
  Circle3PointsRequest,
  Circle3PointsResponse,
  Circle3PointsDetResponse
} from '../services/api';

type TabType = 'center-radius' | 'equation' | 'three-points';

const Circle = () => {
  const [activeTab, setActiveTab] = useState<TabType>('center-radius');
  
  const [centerRadiusData, setCenterRadiusData] = useState<CircleEqnRequest>({
    h: 0,
    k: 0,
    r: 1,
  });
  
  const [equationData, setEquationData] = useState<CircleDetailsRequest>({
    equation: '',
  });
  
  const [threePointsData, setThreePointsData] = useState<Circle3PointsRequest>({
    p: { x: 0, y: 0 },
    q: { x: 0, y: 0 },
    r: { x: 0, y: 0 },
  });
  
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [centerRadiusResult, setCenterRadiusResult] = useState<CircleEqnResponse | null>(null);
  const [equationResult, setEquationResult] = useState<CircleDetailsResponse | null>(null);
  const [threePointsResult, setThreePointsResult] = useState<{ standard: Circle3PointsResponse, determinant: Circle3PointsDetResponse } | null>(null);
  
  const handleCenterRadiusInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setCenterRadiusData({
      ...centerRadiusData,
      [name]: parseFloat(value) || 0,
    });
  };

  const handleEquationInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEquationData({
      ...equationData,
      [e.target.name]: e.target.value,
    });
  };

  const handleThreePointsInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    const [point, coord] = name.split('.');
    
    setThreePointsData({
      ...threePointsData,
      [point]: {
        ...threePointsData[point as keyof Circle3PointsRequest],
        [coord]: parseFloat(value) || 0,
      },
    });
  };

  const handleCenterRadiusSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await calculateCircleEquation(centerRadiusData);
      setCenterRadiusResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleEquationSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await getCircleDetails(equationData);
      setEquationResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleThreePointsSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    
    try {
      const [standardResponse, determinantResponse] = await Promise.all([
        calculateCircleFrom3Points(threePointsData),
        calculateCircleFrom3PointsDet(threePointsData)
      ]);
      
      setThreePointsResult({
        standard: standardResponse,
        determinant: determinantResponse
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const centerRadiusExamples = [
    { h: 0, k: 0, r: 5 },
    { h: 1, k: 2, r: 3 },
    { h: -3, k: 4, r: 2 },
  ];

  const equationExamples = [
    { equation: "x^2 + y^2 = 25" },
    { equation: "x^2 - 2x + y^2 - 4y - 12 = 0" },
  ];

  const threePointsExamples = [
    { p: { x: 4, y: 3 }, q: { x: -4, y: 3 }, r: { x: 4, y: -3 } },
    { p: { x: 1, y: 1 }, q: { x: 4, y: 4 }, r: { x: 7, y: 1 } },
  ];

  const handleUseCenterRadiusExample = (example: CircleEqnRequest) => {
    setCenterRadiusData(example);
  };

  const handleUseEquationExample = (example: CircleDetailsRequest) => {
    setEquationData(example);
  };

  const handleUseThreePointsExample = (example: Circle3PointsRequest) => {
    setThreePointsData(example);
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-800 flex items-center">
          <CircleIcon className="mr-2 h-6 w-6 text-teal-600" />
          Circle Calculator
        </h1>
        <p className="text-slate-600 mt-1">
          Calculate circle equations and properties using different methods.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="md:col-span-2">
          <div className="mb-6">
            <div className="flex border-b border-slate-200">
              <button
                className={`px-4 py-2 text-sm font-medium ${
                  activeTab === 'center-radius'
                    ? 'text-teal-600 border-b-2 border-teal-600'
                    : 'text-slate-600 hover:text-slate-800'
                }`}
                onClick={() => setActiveTab('center-radius')}
              >
                Center & Radius
              </button>
              <button
                className={`px-4 py-2 text-sm font-medium ${
                  activeTab === 'equation'
                    ? 'text-teal-600 border-b-2 border-teal-600'
                    : 'text-slate-600 hover:text-slate-800'
                }`}
                onClick={() => setActiveTab('equation')}
              >
                From Equation
              </button>
              <button
                className={`px-4 py-2 text-sm font-medium ${
                  activeTab === 'three-points'
                    ? 'text-teal-600 border-b-2 border-teal-600'
                    : 'text-slate-600 hover:text-slate-800'
                }`}
                onClick={() => setActiveTab('three-points')}
              >
                Three Points
              </button>
            </div>
          </div>

          {activeTab === 'center-radius' && (
            <form onSubmit={handleCenterRadiusSubmit}>
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-4">
                  <TextField
                    label="Center X (h)"
                    name="h"
                    type="number"
                    step="any"
                    value={centerRadiusData.h.toString()}
                    onChange={handleCenterRadiusInputChange}
                    required
                  />
                  <TextField
                    label="Center Y (k)"
                    name="k"
                    type="number"
                    step="any"
                    value={centerRadiusData.k.toString()}
                    onChange={handleCenterRadiusInputChange}
                    required
                  />
                  <TextField
                    label="Radius (r)"
                    name="r"
                    type="number"
                    step="any"
                    min="0"
                    value={centerRadiusData.r.toString()}
                    onChange={handleCenterRadiusInputChange}
                    required
                  />
                </div>
                <div>
                  <Button type="submit" isLoading={isLoading}>
                    Calculate Circle Equation
                  </Button>
                </div>
              </div>

              <div className="mt-6">
                <p className="text-sm text-slate-600 mb-2">Try these examples:</p>
                <div className="flex flex-wrap gap-2">
                  {centerRadiusExamples.map((example, index) => (
                    <button
                      key={index}
                      onClick={() => handleUseCenterRadiusExample(example)}
                      className="px-3 py-1 text-sm bg-slate-100 text-slate-700 rounded-md hover:bg-slate-200 transition-colors"
                    >
                      Center ({example.h}, {example.k}), r = {example.r}
                    </button>
                  ))}
                </div>
              </div>
            </form>
          )}

          {activeTab === 'equation' && (
            <form onSubmit={handleEquationSubmit}>
              <div className="space-y-4">
                <TextField
                  label="Circle Equation"
                  name="equation"
                  value={equationData.equation}
                  onChange={handleEquationInputChange}
                  placeholder="e.g., x^2 + y^2 = 25 or x^2 + y^2 - 4x - 6y - 12 = 0"
                  required
                />
                <div>
                  <Button type="submit" isLoading={isLoading}>
                    Find Circle Properties
                  </Button>
                </div>
              </div>

              <div className="mt-6">
                <p className="text-sm text-slate-600 mb-2">Try these examples:</p>
                <div className="flex flex-wrap gap-2">
                  {equationExamples.map((example, index) => (
                    <button
                      key={index}
                      onClick={() => handleUseEquationExample(example)}
                      className="px-3 py-1 text-sm bg-slate-100 text-slate-700 rounded-md hover:bg-slate-200 transition-colors"
                    >
                      {example.equation}
                    </button>
                  ))}
                </div>
              </div>
            </form>
          )}

          {activeTab === 'three-points' && (
            <form onSubmit={handleThreePointsSubmit}>
              <div className="space-y-4">
                <div>
                  <h3 className="text-sm font-medium text-slate-700 mb-2">Point P</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <TextField
                      label="X₁"
                      name="p.x"
                      type="number"
                      step="any"
                      value={threePointsData.p.x.toString()}
                      onChange={handleThreePointsInputChange}
                      required
                    />
                    <TextField
                      label="Y₁"
                      name="p.y"
                      type="number"
                      step="any"
                      value={threePointsData.p.y.toString()}
                      onChange={handleThreePointsInputChange}
                      required
                    />
                  </div>
                </div>
                <div>
                  <h3 className="text-sm font-medium text-slate-700 mb-2">Point Q</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <TextField
                      label="X₂"
                      name="q.x"
                      type="number"
                      step="any"
                      value={threePointsData.q.x.toString()}
                      onChange={handleThreePointsInputChange}
                      required
                    />
                    <TextField
                      label="Y₂"
                      name="q.y"
                      type="number"
                      step="any"
                      value={threePointsData.q.y.toString()}
                      onChange={handleThreePointsInputChange}
                      required
                    />
                  </div>
                </div>
                <div>
                  <h3 className="text-sm font-medium text-slate-700 mb-2">Point R</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <TextField
                      label="X₃"
                      name="r.x"
                      type="number"
                      step="any"
                      value={threePointsData.r.x.toString()}
                      onChange={handleThreePointsInputChange}
                      required
                    />
                    <TextField
                      label="Y₃"
                      name="r.y"
                      type="number"
                      step="any"
                      value={threePointsData.r.y.toString()}
                      onChange={handleThreePointsInputChange}
                      required
                    />
                  </div>
                </div>
                <div>
                  <Button type="submit" isLoading={isLoading}>
                    Calculate Circle
                  </Button>
                </div>
              </div>

              <div className="mt-6">
                <p className="text-sm text-slate-600 mb-2">Try these examples:</p>
                <div className="flex flex-wrap gap-2">
                  {threePointsExamples.map((example, index) => (
                    <button
                      key={index}
                      onClick={() => handleUseThreePointsExample(example)}
                      className="px-3 py-1 text-sm bg-slate-100 text-slate-700 rounded-md hover:bg-slate-200 transition-colors"
                    >
                      ({example.p.x}, {example.p.y}), ({example.q.x}, {example.q.y}), ({example.r.x}, {example.r.y})
                    </button>
                  ))}
                </div>
              </div>
            </form>
          )}

          {error && (
            <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-md">
              {error}
            </div>
          )}
        </Card>

        {centerRadiusResult && activeTab === 'center-radius' && (
          <Card className="md:col-span-2" title="Circle Equation Results">
            <div className="space-y-4">
              <div>
                <h4 className="text-sm font-medium text-slate-700 mb-1">Circle Properties</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                  <div className="p-3 bg-teal-50 rounded-md">
                    <span className="text-sm text-slate-600">Center (h, k)</span>
                    <p className="text-lg font-semibold text-slate-800">({centerRadiusResult.center_h}, {centerRadiusResult.center_k})</p>
                  </div>
                  <div className="p-3 bg-teal-50 rounded-md">
                    <span className="text-sm text-slate-600">Radius</span>
                    <p className="text-lg font-semibold text-slate-800">{centerRadiusResult.radius}</p>
                  </div>
                  <div className="p-3 bg-teal-50 rounded-md">
                    <span className="text-sm text-slate-600">Area</span>
                    <p className="text-lg font-semibold text-slate-800">{(Math.PI * Math.pow(centerRadiusResult.radius, 2)).toFixed(2)} square units</p>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium text-slate-700 mb-1">Standard Form</h4>
                <MathDisplay expression={centerRadiusResult.standard_form} />
              </div>

              <div>
                <h4 className="text-sm font-medium text-slate-700 mb-1">General Form</h4>
                <MathDisplay expression={centerRadiusResult.general_form} />
              </div>

              <div className="mt-4 pt-4 border-t border-slate-200">
                <h4 className="text-sm font-medium text-slate-700 mb-2">General Form Coefficients</h4>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                  <div className="p-2 bg-slate-50 rounded-md">
                    <span className="text-sm text-slate-600">A</span>
                    <p className="font-medium text-slate-800">{centerRadiusResult.A}</p>
                  </div>
                  <div className="p-2 bg-slate-50 rounded-md">
                    <span className="text-sm text-slate-600">B</span>
                    <p className="font-medium text-slate-800">{centerRadiusResult.B}</p>
                  </div>
                  <div className="p-2 bg-slate-50 rounded-md">
                    <span className="text-sm text-slate-600">C</span>
                    <p className="font-medium text-slate-800">{centerRadiusResult.C}</p>
                  </div>
                  <div className="p-2 bg-slate-50 rounded-md">
                    <span className="text-sm text-slate-600">D</span>
                    <p className="font-medium text-slate-800">{centerRadiusResult.D}</p>
                  </div>
                  <div className="p-2 bg-slate-50 rounded-md">
                    <span className="text-sm text-slate-600">E</span>
                    <p className="font-medium text-slate-800">{centerRadiusResult.E}</p>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        )}

        {equationResult && activeTab === 'equation' && (
          <Card className="md:col-span-2" title="Circle Properties Results">
            <div className="space-y-4">
              <div>
                <h4 className="text-sm font-medium text-slate-700 mb-1">Circle Properties</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                  <div className="p-3 bg-teal-50 rounded-md">
                    <span className="text-sm text-slate-600">Center (h, k)</span>
                    <p className="text-lg font-semibold text-slate-800">({equationResult.center_h}, {equationResult.center_k})</p>
                  </div>
                  <div className="p-3 bg-teal-50 rounded-md">
                    <span className="text-sm text-slate-600">Radius</span>
                    <p className="text-lg font-semibold text-slate-800">{equationResult.radius}</p>
                  </div>
                  <div className="p-3 bg-teal-50 rounded-md">
                    <span className="text-sm text-slate-600">Area</span>
                    <p className="text-lg font-semibold text-slate-800">{(Math.PI * Math.pow(equationResult.radius, 2)).toFixed(2)} square units</p>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium text-slate-700 mb-1">Input Equation</h4>
                <MathDisplay expression={equationResult.input_equation} />
              </div>

              <div>
                <h4 className="text-sm font-medium text-slate-700 mb-1">Normalized Equation</h4>
                <MathDisplay expression={equationResult.normalized_equation} />
              </div>

              <div>
                <h4 className="text-sm font-medium text-slate-700 mb-1">Standard Form</h4>
                <MathDisplay 
                  expression={`(x - ${equationResult.center_h})² + (y - ${equationResult.center_k})² = ${equationResult.radius}²`} 
                />
              </div>
            </div>
          </Card>
        )}

        {threePointsResult && activeTab === 'three-points' && (
          <Card className="md:col-span-2" title="Circle from Three Points Results">
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div className="p-3 bg-teal-50 rounded-md col-span-3">
                  <span className="text-sm text-slate-600">Points used</span>
                  <p className="text-lg font-semibold text-slate-800">
                    P({threePointsData.p.x}, {threePointsData.p.y}), 
                    Q({threePointsData.q.x}, {threePointsData.q.y}), 
                    R({threePointsData.r.x}, {threePointsData.r.y})
                  </p>
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium text-slate-700 mb-1">Regular Method Result</h4>
                <MathDisplay expression={threePointsResult.standard.standard_form} />
              </div>

              <div>
                <h4 className="text-sm font-medium text-slate-700 mb-1">Determinant Method Result</h4>
                <MathDisplay expression={threePointsResult.determinant.circle_equation} />
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
};

export default Circle;