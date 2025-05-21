// API base URL
const API_BASE_URL = 'http://localhost:8000';

// Generic fetch function with error handling
async function fetchFromAPI(endpoint: string, options: RequestInit = {}) {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      throw new Error(errorData?.detail || `API error (${response.status})`);
    }

    return await response.json();
  } catch (error) {
    console.error('API request failed:', error);
    throw error;
  }
}

// Differentiation API
export interface DifferentiateRequest {
  expression: string;
  variable: string;
}

export interface SimpleDerivativeResponse {
  original_expression: string;
  variable: string;
  derivative: string;
}

export interface DerivativeStep {
  step_description: string;
  expression_latex: string;
  expression_pretty: string;
}

export interface StepDerivativeResponse extends SimpleDerivativeResponse {
  steps: DerivativeStep[];
  final_derivative: string;
}

export const differentiateSimple = (data: DifferentiateRequest): Promise<SimpleDerivativeResponse> => {
  return fetchFromAPI('/differentiate/simple', {
    method: 'POST',
    body: JSON.stringify(data),
  });
};

export const differentiateWithSteps = (data: DifferentiateRequest): Promise<StepDerivativeResponse> => {
  return fetchFromAPI('/differentiate/steps', {
    method: 'POST',
    body: JSON.stringify(data),
  });
};

// Integration API
export interface IntegrateRequest {
  expression: string;
  variable: string;
}

export interface SimpleIntegralResponse {
  original_expression: string;
  variable: string;
  integral_result: string;
  computation_notes: string | null;
}

export interface IntegrationStep {
  step_description: string;
  expression_latex: string;
  expression_pretty: string;
}

export interface StepIntegralResponse {
  original_expression: string;
  variable: string;
  steps: IntegrationStep[];
  final_integral: string;
  computation_notes: string | null;
}

export const integrateSimple = (data: IntegrateRequest): Promise<SimpleIntegralResponse> => {
  return fetchFromAPI('/integrate/integrate/simple', {
    method: 'POST',
    body: JSON.stringify(data),
  });
};

export const integrateWithSteps = (data: IntegrateRequest): Promise<StepIntegralResponse> => {
  return fetchFromAPI('/integrate/integrate/steps', {
    method: 'POST',
    body: JSON.stringify(data),
  });
};

// Slope API
export interface SlopeRequest {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface SlopeResponse {
  slope: number;
}

export const calculateSlope = (data: SlopeRequest): Promise<SlopeResponse> => {
  return fetchFromAPI('/SlopeCordiantes', {
    method: 'POST',
    body: JSON.stringify(data),
  });
};

// Powers of i API
export const calculatePowerOfI = (n: number): Promise<{ result: number }> => {
  return fetchFromAPI(`/powers?n=${n}`, {
    method: 'POST',
  });
};

// Complex Numbers API
export interface ComplexNumber {
  real: number;
  img: number;
}

export interface ArithmeticRequest {
  z1: ComplexNumber;
  z2: ComplexNumber;
  operation: 'add' | 'subtract' | 'multiply' | 'divide';
}

export interface ArithmeticResponse {
  result: string;
}

export const performComplexArithmetic = (data: ArithmeticRequest): Promise<ArithmeticResponse> => {
  return fetchFromAPI('/arithmetic', {
    method: 'POST',
    body: JSON.stringify(data),
  });
};

// Circle API
export interface CircleEqnRequest {
  r: number;
  h: number;
  k: number;
}

export interface CircleEqnResponse {
  standard_form: string;
  general_form: string;
  center_h: number;
  center_k: number;
  radius: number;
  A: number;
  B: number;
  C: number;
  D: number;
  E: number;
}

export const calculateCircleEquation = (data: CircleEqnRequest): Promise<CircleEqnResponse> => {
  return fetchFromAPI('/circle/eqn', {
    method: 'POST',
    body: JSON.stringify(data),
  });
};

export interface CircleDetailsRequest {
  equation: string;
}

export interface CircleDetailsResponse {
  center_h: number;
  center_k: number;
  radius: number;
  input_equation: string;
  normalized_equation: string;
}

export const getCircleDetails = (data: CircleDetailsRequest): Promise<CircleDetailsResponse> => {
  return fetchFromAPI('/circle/details', {
    method: 'POST',
    body: JSON.stringify(data),
  });
};

export interface Point {
  x: number;
  y: number;
}

export interface Circle3PointsRequest {
  p: Point;
  q: Point;
  r: Point;
}

export interface Circle3PointsResponse {
  standard_form: string;
}

export const calculateCircleFrom3Points = (data: Circle3PointsRequest): Promise<Circle3PointsResponse> => {
  return fetchFromAPI('/circle/3points', {
    method: 'POST',
    body: JSON.stringify(data),
  });
};

export interface Circle3PointsDetResponse {
  circle_equation: string;
}

export const calculateCircleFrom3PointsDet = (data: Circle3PointsRequest): Promise<Circle3PointsDetResponse> => {
  return fetchFromAPI('/circle/3points/det1', {
    method: 'POST',
    body: JSON.stringify(data),
  });

  
};




// Arithmetic Progression
export interface APRequest {
  a: number; // First term
  d: number; // Common difference
  n: number; // Number of terms
}

export interface APResponse {
  nth_term: number;
  sum_n_terms: number;
}

export const calculateAP = (data: APRequest): Promise<APResponse> => {
  return fetchFromAPI('/progressions/arithmetic_progression', {
    method: 'POST',
    body: JSON.stringify(data),
  });
};

// Nth Term from Last in AP
export interface APLastTermRequest {
  a: number; // First term
  d: number; // Common difference
  l: number; // Last term
  n: number; // Nth term from end
}

export interface APLastTermResponse {
  nth_term_from_last: number;
}

export const calculateAPNthTermFromLast = (data: APLastTermRequest): Promise<APLastTermResponse> => {
  return fetchFromAPI('/progressions/nth_term_from_last', {
    method: 'POST',
    body: JSON.stringify(data),
  });
};

// Middle Term(s) in AP
export interface APMiddleTermRequest {
  a: number; // First term
  d: number; // Common difference
  last_term: number; // Value of the last term
}

export interface APMiddleTermResponse {
  number_of_terms: number;
  middle_term_s: number[]; // Corresponds to middle_term(s) in Python, using _s for valid JS identifier
  message?: string;
}

export const findAPMiddleTerms = (data: APMiddleTermRequest): Promise<APMiddleTermResponse> => {
  return fetchFromAPI('/progressions/middle_term', {
    method: 'POST',
    body: JSON.stringify(data),
  });
};