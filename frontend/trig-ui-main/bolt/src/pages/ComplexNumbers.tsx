import React, { useState } from 'react';
import { PlusSquare } from 'lucide-react';
import Card from '../components/ui/Card';
import TextField from '../components/ui/TextField';
import Button from '../components/ui/Button';
import Select from '../components/ui/Select';
import { 
  performComplexArithmetic, 
  calculatePowerOfI,
  ArithmeticRequest, 
  ArithmeticResponse 
} from '../services/api';

const ComplexNumbers = () => {
  const [activeTab, setActiveTab] = useState<'arithmetic' | 'power'>('arithmetic');
  
  const [arithmeticData, setArithmeticData] = useState<ArithmeticRequest>({
    z1: { real: 0, img: 0 },
    z2: { real: 0, img: 0 },
    operation: 'add',
  });
  
  const [powerData, setPowerData] = useState({
    n: 1,
  });
  
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [arithmeticResult, setArithmeticResult] = useState<ArithmeticResponse | null>(null);
  const [powerResult, setPowerResult] = useState<{ result: number } | null>(null);

  const handleArithmeticInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    
    if (name === 'operation') {
      setArithmeticData({
        ...arithmeticData,
        operation: value as 'add' | 'subtract' | 'multiply' | 'divide',
      });
    } else {
      const [complex, part] = name.split('.');
      setArithmeticData({
        ...arithmeticData,
        [complex]: {
          ...arithmeticData[complex as 'z1' | 'z2'],
          [part]: parseFloat(value) || 0,
        },
      });
    }
  };

  const handlePowerInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setPowerData({
      ...powerData,
      [e.target.name]: parseInt(e.target.value) || 0,
    });
  };

  const handleArithmeticSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await performComplexArithmetic(arithmeticData);
      setArithmeticResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handlePowerSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await calculatePowerOfI(powerData.n);
      setPowerResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const arithmeticExamples = [
    { z1: { real: 2, img: 3 }, z2: { real: 1, img: 2 }, operation: 'add' },
    { z1: { real: 5, img: 1 }, z2: { real: 2, img: 4 }, operation: 'multiply' },
    { z1: { real: 4, img: 2 }, z2: { real: 1, img: 1 }, operation: 'divide' },
  ];

  const powerExamples = [1, 2, 3, 4, 5];

  const handleUseArithmeticExample = (example: ArithmeticRequest) => {
    setArithmeticData(example);
  };

  const handleUsePowerExample = (n: number) => {
    setPowerData({ n });
  };

  const operationSymbols: Record<string, string> = {
    add: '+',
    subtract: '-',
    multiply: '×',
    divide: '÷',
  };

  const formatComplex = (real: number, img: number): string => {
    if (real === 0 && img === 0) return '0';
    if (real === 0) return `${img}i`;
    if (img === 0) return `${real}`;
    
    const imgSign = img > 0 ? '+' : '';
    return `${real}${imgSign}${img}i`;
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-800 flex items-center">
          <PlusSquare className="mr-2 h-6 w-6 text-amber-600" />
          Complex Numbers
        </h1>
        <p className="text-slate-600 mt-1">
          Perform operations with complex numbers and calculate powers of i.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="md:col-span-2">
          <div className="mb-6">
            <div className="flex border-b border-slate-200">
              <button
                className={`px-4 py-2 text-sm font-medium ${
                  activeTab === 'arithmetic'
                    ? 'text-amber-600 border-b-2 border-amber-600'
                    : 'text-slate-600 hover:text-slate-800'
                }`}
                onClick={() => setActiveTab('arithmetic')}
              >
                Complex Arithmetic
              </button>
              <button
                className={`px-4 py-2 text-sm font-medium ${
                  activeTab === 'power'
                    ? 'text-amber-600 border-b-2 border-amber-600'
                    : 'text-slate-600 hover:text-slate-800'
                }`}
                onClick={() => setActiveTab('power')}
              >
                Powers of i
              </button>
            </div>
          </div>

          {activeTab === 'arithmetic' ? (
            <form onSubmit={handleArithmeticSubmit}>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h3 className="text-sm font-medium text-slate-700 mb-2">First Complex Number (z₁)</h3>
                    <div className="grid grid-cols-2 gap-4">
                      <TextField
                        label="Real Part"
                        name="z1.real"
                        type="number"
                        step="any"
                        value={arithmeticData.z1.real.toString()}
                        onChange={handleArithmeticInputChange}
                        required
                      />
                      <TextField
                        label="Imaginary Part"
                        name="z1.img"
                        type="number"
                        step="any"
                        value={arithmeticData.z1.img.toString()}
                        onChange={handleArithmeticInputChange}
                        required
                      />
                    </div>
                  </div>
                  <div>
                    <h3 className="text-sm font-medium text-slate-700 mb-2">Second Complex Number (z₂)</h3>
                    <div className="grid grid-cols-2 gap-4">
                      <TextField
                        label="Real Part"
                        name="z2.real"
                        type="number"
                        step="any"
                        value={arithmeticData.z2.real.toString()}
                        onChange={handleArithmeticInputChange}
                        required
                      />
                      <TextField
                        label="Imaginary Part"
                        name="z2.img"
                        type="number"
                        step="any"
                        value={arithmeticData.z2.img.toString()}
                        onChange={handleArithmeticInputChange}
                        required
                      />
                    </div>
                  </div>
                </div>
                <Select
                  label="Operation"
                  name="operation"
                  value={arithmeticData.operation}
                  onChange={handleArithmeticInputChange}
                  options={[
                    { value: 'add', label: 'Addition (+)' },
                    { value: 'subtract', label: 'Subtraction (-)' },
                    { value: 'multiply', label: 'Multiplication (×)' },
                    { value: 'divide', label: 'Division (÷)' },
                  ]}
                  required
                />
                <div>
                  <Button type="submit" isLoading={isLoading}>
                    Calculate
                  </Button>
                </div>
              </div>

              <div className="mt-6">
                <p className="text-sm text-slate-600 mb-2">Try these examples:</p>
                <div className="flex flex-wrap gap-2">
                  {arithmeticExamples.map((example, index) => (
                    <button
                      key={index}
                      onClick={() => handleUseArithmeticExample(example)}
                      className="px-3 py-1 text-sm bg-slate-100 text-slate-700 rounded-md hover:bg-slate-200 transition-colors"
                    >
                      {formatComplex(example.z1.real, example.z1.img)} {operationSymbols[example.operation]} {formatComplex(example.z2.real, example.z2.img)}
                    </button>
                  ))}
                </div>
              </div>
            </form>
          ) : (
            <form onSubmit={handlePowerSubmit}>
              <div className="space-y-4">
                <TextField
                  label="Power (n)"
                  name="n"
                  type="number"
                  value={powerData.n.toString()}
                  onChange={handlePowerInputChange}
                  helperText="Enter an integer to calculate i^n"
                  required
                />
                <div>
                  <Button type="submit" isLoading={isLoading}>
                    Calculate
                  </Button>
                </div>
              </div>

              <div className="mt-6">
                <p className="text-sm text-slate-600 mb-2">Try these examples:</p>
                <div className="flex flex-wrap gap-2">
                  {powerExamples.map((n) => (
                    <button
                      key={n}
                      onClick={() => handleUsePowerExample(n)}
                      className="px-3 py-1 text-sm bg-slate-100 text-slate-700 rounded-md hover:bg-slate-200 transition-colors"
                    >
                      i^{n}
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

        {arithmeticResult && activeTab === 'arithmetic' && (
          <Card className="md:col-span-2" title="Complex Arithmetic Result">
            <div className="space-y-4">
              <div className="p-4 bg-amber-50 rounded-lg">
                <p className="text-lg text-slate-800 font-medium mb-1">
                  {formatComplex(arithmeticData.z1.real, arithmeticData.z1.img)} {operationSymbols[arithmeticData.operation]} {formatComplex(arithmeticData.z2.real, arithmeticData.z2.img)} = {arithmeticResult.result}
                </p>
              </div>

              <div className="mt-4 border-t border-slate-200 pt-4">
                <h4 className="text-sm font-medium text-slate-700 mb-2">Formula Used</h4>
                <div className="bg-slate-50 p-3 rounded-md text-slate-800 font-mono">
                  {activeTab === 'arithmetic' && (
                    <div>
                      {arithmeticData.operation === 'add' && (
                        <p>(a + bi) + (c + di) = (a + c) + (b + d)i</p>
                      )}
                      {arithmeticData.operation === 'subtract' && (
                        <p>(a + bi) - (c + di) = (a - c) + (b - d)i</p>
                      )}
                      {arithmeticData.operation === 'multiply' && (
                        <p>(a + bi) × (c + di) = (ac - bd) + (ad + bc)i</p>
                      )}
                      {arithmeticData.operation === 'divide' && (
                        <p>(a + bi) ÷ (c + di) = ((ac + bd) + (bc - ad)i) / (c² + d²)</p>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </Card>
        )}

        {powerResult && activeTab === 'power' && (
          <Card className="md:col-span-2" title="Power of i Result">
            <div className="space-y-4">
              <div className="p-4 bg-amber-50 rounded-lg">
                <p className="text-lg text-slate-800 font-medium mb-1">
                  i^{powerData.n} = {powerResult.result === 1 ? '1' : powerResult.result === -1 ? '-1' : powerResult.result === 0 ? 'i' : '-i'}
                </p>
              </div>

              <div className="mt-4 border-t border-slate-200 pt-4">
                <h4 className="text-sm font-medium text-slate-700 mb-2">Pattern of Powers of i</h4>
                <div className="bg-slate-50 p-3 rounded-md text-slate-800">
                  <ul className="list-disc pl-5 space-y-1">
                    <li>i<sup>1</sup> = i</li>
                    <li>i<sup>2</sup> = -1</li>
                    <li>i<sup>3</sup> = -i</li>
                    <li>i<sup>4</sup> = 1</li>
                  </ul>
                  <p className="mt-2">The pattern repeats every 4 powers.</p>
                </div>
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
};

export default ComplexNumbers;