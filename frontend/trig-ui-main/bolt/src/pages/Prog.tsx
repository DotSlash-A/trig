import React, { useState } from 'react';
import { TrendingUp, ListFilter, LocateFixed } from 'lucide-react';
import Card from '../components/ui/Card';
import TextField from '../components/ui/TextField';
import Button from '../components/ui/Button';
// import MathDisplay from '../components/ui/MathDisplay'; // MathDisplay might not be needed if results are simple numbers
import {
  // Progression API imports
  APRequest,
  APResponse,
  calculateAP,
  APLastTermRequest,
  APLastTermResponse,
  calculateAPNthTermFromLast,
  APMiddleTermRequest,
  APMiddleTermResponse,
  findAPMiddleTerms
} from '../services/api';

type TabType = 'arithmetic-progression' | 'ap-nth-from-last' | 'ap-middle-term';

const Prog = () => {
  const [activeTab, setActiveTab] = useState<TabType>('arithmetic-progression');

  // Progression States
  const [apData, setApData] = useState<APRequest>({ a: 1, d: 2, n: 5 });
  const [apResult, setApResult] = useState<APResponse | null>(null);

  const [apLastTermData, setApLastTermData] = useState<APLastTermRequest>({ a: 1, d: 2, l: 9, n: 2 });
  const [apLastTermResult, setApLastTermResult] = useState<APLastTermResponse | null>(null);

  const [apMiddleTermData, setApMiddleTermData] = useState<APMiddleTermRequest>({ a: 1, d: 2, last_term: 9 });
  const [apMiddleTermResult, setApMiddleTermResult] = useState<APMiddleTermResponse | null>(null);
  
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Progression Input Handlers
  const handleApInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setApData({ ...apData, [name]: parseFloat(value) || 0 });
  };

  const handleApLastTermInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setApLastTermData({ ...apLastTermData, [name]: parseFloat(value) || 0 });
  };

  const handleApMiddleTermInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setApMiddleTermData({ ...apMiddleTermData, [name]: parseFloat(value) || 0 });
  };

  // Progression Submit Handlers
  const handleApSubmit = async (e: React.FormEvent) => {
    e.preventDefault(); setIsLoading(true); setError(null); setApResult(null);
    try {
      const response = await calculateAP(apData);
      setApResult(response);
    } catch (err) { setError(err instanceof Error ? err.message : 'An error occurred'); }
    finally { setIsLoading(false); }
  };

  const handleApLastTermSubmit = async (e: React.FormEvent) => {
    e.preventDefault(); setIsLoading(true); setError(null); setApLastTermResult(null);
    try {
      const response = await calculateAPNthTermFromLast(apLastTermData);
      setApLastTermResult(response);
    } catch (err) { setError(err instanceof Error ? err.message : 'An error occurred'); }
    finally { setIsLoading(false); }
  };

  const handleApMiddleTermSubmit = async (e: React.FormEvent) => {
    e.preventDefault(); setIsLoading(true); setError(null); setApMiddleTermResult(null);
    try {
      const response = await findAPMiddleTerms(apMiddleTermData);
      setApMiddleTermResult(response);
    } catch (err) { setError(err instanceof Error ? err.message : 'An error occurred'); }
    finally { setIsLoading(false); }
  };

  // Example Data
  const apExamples = [ { a: 1, d: 2, n: 5 }, { a: 10, d: -3, n: 7 }, { a: 0, d: 0.5, n: 10 } ];
  const apLastTermExamples = [ { a: 1, d: 2, l: 9, n: 2 }, { a: 20, d: -2, l: 2, n: 3 } ];
  const apMiddleTermExamples = [ { a: 1, d: 2, last_term: 9 }, { a: 2, d: 3, last_term: 14 }, { a: 5, d: 1, last_term: 10 } ];

  // Use Example Handlers
  const handleUseApExample = (example: APRequest) => setApData(example);
  const handleUseApLastTermExample = (example: APLastTermRequest) => setApLastTermData(example);
  const handleUseApMiddleTermExample = (example: APMiddleTermRequest) => setApMiddleTermData(example);

  const renderTabButton = (tab: TabType, label: string, Icon?: React.ElementType) => (
    <button
      className={`flex items-center px-4 py-2 text-sm font-medium ${
        activeTab === tab
          ? 'text-teal-600 border-b-2 border-teal-600'
          : 'text-slate-600 hover:text-slate-800'
      }`}
      onClick={() => {
        setActiveTab(tab);
        setError(null); // Clear error when switching tabs
        setApResult(null);
        setApLastTermResult(null);
        setApMiddleTermResult(null);
      }}
    >
      {Icon && <Icon className="mr-2 h-4 w-4" />}
      {label}
    </button>
  );

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-800 flex items-center">
          Progression Calculators
        </h1>
        <p className="text-slate-600 mt-1">
          Solve problems related to Arithmetic Progressions.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="md:col-span-2">
          <div className="mb-6">
            <div className="flex border-b border-slate-200 flex-wrap">
              {renderTabButton('arithmetic-progression', 'AP: Nth Term & Sum', TrendingUp)}
              {renderTabButton('ap-nth-from-last', 'AP: Nth Term from Last', ListFilter)}
              {renderTabButton('ap-middle-term', 'AP: Middle Term(s)', LocateFixed)}
            </div>
          </div>

          {/* Arithmetic Progression Forms */}
          {activeTab === 'arithmetic-progression' && (
            <form onSubmit={handleApSubmit}>
              <div className="space-y-4">
                <TextField label="First term (a)" name="a" type="number" step="any" value={apData.a.toString()} onChange={handleApInputChange} required />
                <TextField label="Common difference (d)" name="d" type="number" step="any" value={apData.d.toString()} onChange={handleApInputChange} required />
                <TextField label="Number of terms (n)" name="n" type="number" step="1" min="1" value={apData.n.toString()} onChange={handleApInputChange} required />
                <div><Button type="submit" isLoading={isLoading}>Calculate AP</Button></div>
              </div>
              <div className="mt-6"><p className="text-sm text-slate-600 mb-2">Try these examples:</p><div className="flex flex-wrap gap-2">
                  {apExamples.map((ex, i) => <button key={i} onClick={() => handleUseApExample(ex)} className="px-3 py-1 text-sm bg-slate-100 text-slate-700 rounded-md hover:bg-slate-200 transition-colors">a={ex.a}, d={ex.d}, n={ex.n}</button>)}
              </div></div>
            </form>
          )}

          {activeTab === 'ap-nth-from-last' && (
            <form onSubmit={handleApLastTermSubmit}>
              <div className="space-y-4">
                <TextField label="First term (a)" name="a" type="number" step="any" value={apLastTermData.a.toString()} onChange={handleApLastTermInputChange} required />
                <TextField label="Common difference (d)" name="d" type="number" step="any" value={apLastTermData.d.toString()} onChange={handleApLastTermInputChange} required />
                <TextField label="Last term (l)" name="l" type="number" step="any" value={apLastTermData.l.toString()} onChange={handleApLastTermInputChange} required />
                <TextField label="Nth term from end (n)" name="n" type="number" step="1" min="1" value={apLastTermData.n.toString()} onChange={handleApLastTermInputChange} required />
                <div><Button type="submit" isLoading={isLoading}>Calculate Nth Term from Last</Button></div>
              </div>
              <div className="mt-6"><p className="text-sm text-slate-600 mb-2">Try these examples:</p><div className="flex flex-wrap gap-2">
                  {apLastTermExamples.map((ex, i) => <button key={i} onClick={() => handleUseApLastTermExample(ex)} className="px-3 py-1 text-sm bg-slate-100 text-slate-700 rounded-md hover:bg-slate-200 transition-colors">a={ex.a}, d={ex.d}, l={ex.l}, n={ex.n}</button>)}
              </div></div>
            </form>
          )}

          {activeTab === 'ap-middle-term' && (
            <form onSubmit={handleApMiddleTermSubmit}>
              <div className="space-y-4">
                <TextField label="First term (a)" name="a" type="number" step="any" value={apMiddleTermData.a.toString()} onChange={handleApMiddleTermInputChange} required />
                <TextField label="Common difference (d)" name="d" type="number" step="any" value={apMiddleTermData.d.toString()} onChange={handleApMiddleTermInputChange} required />
                <TextField label="Value of Last term" name="last_term" type="number" step="any" value={apMiddleTermData.last_term.toString()} onChange={handleApMiddleTermInputChange} required />
                <div><Button type="submit" isLoading={isLoading}>Find Middle Term(s)</Button></div>
              </div>
              <div className="mt-6"><p className="text-sm text-slate-600 mb-2">Try these examples:</p><div className="flex flex-wrap gap-2">
                  {apMiddleTermExamples.map((ex, i) => <button key={i} onClick={() => handleUseApMiddleTermExample(ex)} className="px-3 py-1 text-sm bg-slate-100 text-slate-700 rounded-md hover:bg-slate-200 transition-colors">a={ex.a}, d={ex.d}, last_term={ex.last_term}</button>)}
              </div></div>
            </form>
          )}

          {error && (
            <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-md">
              {error}
            </div>
          )}
        </Card>

        {/* Progression Results */}
        {apResult && activeTab === 'arithmetic-progression' && (
          <Card className="md:col-span-2" title="Arithmetic Progression Results">
            <div className="space-y-2 p-4">
              <p className="text-slate-700"><strong>Nth Term:</strong> <span className="font-semibold text-teal-600">{apResult.nth_term}</span></p>
              <p className="text-slate-700"><strong>Sum of N Terms:</strong> <span className="font-semibold text-teal-600">{apResult.sum_n_terms}</span></p>
            </div>
          </Card>
        )}

        {apLastTermResult && activeTab === 'ap-nth-from-last' && (
          <Card className="md:col-span-2" title="AP Nth Term from Last Result">
            <div className="space-y-2 p-4">
              <p className="text-slate-700"><strong>Nth Term from Last:</strong> <span className="font-semibold text-teal-600">{apLastTermResult.nth_term_from_last}</span></p>
            </div>
          </Card>
        )}

        {apMiddleTermResult && activeTab === 'ap-middle-term' && (
          <Card className="md:col-span-2" title="AP Middle Term(s) Result">
            <div className="space-y-2 p-4">
              <p className="text-slate-700"><strong>Number of Terms:</strong> <span className="font-semibold text-teal-600">{apMiddleTermResult.number_of_terms}</span></p>
              <p className="text-slate-700"><strong>Middle Term(s):</strong> <span className="font-semibold text-teal-600">{apMiddleTermResult.middle_term_s.join(', ')}</span></p>
              {apMiddleTermResult.message && <p className="text-sm text-slate-600 mt-2"><em>{apMiddleTermResult.message}</em></p>}
            </div>
          </Card>
        )}
      </div>
    </div>
  );
};

export default Prog;