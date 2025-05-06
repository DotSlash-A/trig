from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

# --- Common ---
class FunctionPiece(BaseModel):
    expression: str
    condition: str # e.g., "x < 0", "x >= 0"

class FunctionDefinition(BaseModel):
    type: str = Field(..., pattern="^(single|piecewise)$") # 'single' or 'piecewise'
    expression: Optional[str] = None # For type 'single'
    pieces: Optional[List[FunctionPiece]] = None # For type 'piecewise'
    variable: str = 'x'
    # Ensure either expression or pieces is provided based on type (can add custom validator)

class Point(BaseModel):
    # Allows flexibility: point can be a single number or coordinates
    values: Dict[str, float] # e.g. {"x": 1.0}, {"x": 3.0, "y": 4.0}, {"t": 3.14}

class Interval(BaseModel):
    start: Union[float, str] # Use str for "-oo", "oo"
    end: Union[float, str]
    start_inclusive: bool = False
    end_inclusive: bool = False

# --- Continuity ---
class ContinuityCheckRequest(BaseModel):
    function_definition: FunctionDefinition
    point: float

class ContinuityCheckResponse(BaseModel):
    point: float
    lhl: Optional[str] = None # Store sympy results as strings
    rhl: Optional[str] = None
    f_at_point: Optional[str] = None
    is_continuous: Optional[bool] = None
    reason: str

class ContinuityConstantsRequest(BaseModel):
    function_definition: FunctionDefinition
    point: float
    constants_to_find: List[str]

class ContinuityConstantsResponse(BaseModel):
    point: float
    conditions_for_continuity: List[str]
    equations_derived: List[str]
    simplified_equations: List[str]
    solutions: Optional[Dict[str, str]] = None # Constant: value (as string)
    is_possible: bool
    reason: Optional[str] = None

class ContinuityIntervalRequest(BaseModel):
    function_definition: FunctionDefinition
    interval: Interval
    # check_endpoints: bool # Implicit in Interval model

class DiscontinuityInfo(BaseModel):
    point: Union[float, str] # Point or potentially symbolic issue
    reason: str
    lhl: Optional[str] = None
    rhl: Optional[str] = None
    f_at_point: Optional[str] = None


class ContinuityIntervalResponse(BaseModel):
    interval_checked: Dict # Start, end, inclusive flags
    is_continuous_overall: bool
    potential_discontinuities_checked: List[Union[float, str]] # Points derived from function structure
    points_of_discontinuity_found: List[DiscontinuityInfo]
    continuity_intervals: List[Dict] # List of interval dicts where continuous

# --- Differentiability ---
class DifferentiabilityCheckRequest(BaseModel):
    function_definition: FunctionDefinition
    point: float

class DifferentiabilityCheckResponse(BaseModel):
    point: float
    continuity_check: ContinuityCheckResponse # Embed continuity result
    lhd: Optional[str] = None
    rhd: Optional[str] = None
    is_differentiable: bool
    derivative_value: Optional[str] = None
    reason: str

class DifferentiabilityConstantsRequest(BaseModel):
    function_definition: FunctionDefinition
    point: float
    constants_to_find: List[str]

class DifferentiabilityConstantsResponse(BaseModel):
    point: float
    continuity_equations: List[str]
    differentiability_equations: List[str]
    system_of_equations: List[str]
    solutions: Optional[Dict[str, str]] = None
    is_possible: bool
    reason: Optional[str] = None


# --- Rate Measure ---
class DirectRateRequest(BaseModel):
    function_str: str # e.g., "A = pi * r**2" or just "pi * r**2"
    dependent_var: str
    independent_var: str
    point: Dict[str, float] # Value of independent_var

class DirectRateResponse(BaseModel):
    derivative_expression: str # e.g., "dA/dr = 2*pi*r"
    rate_at_point: str

class RelatedRatesRequest(BaseModel):
    equation_str: str # e.g., "x**2 + y**2 = z**2"
    variables: List[str] # All variables involved (assumed funcs of 't')
    known_rates: Dict[str, float] # e.g., {"dx/dt": 2, "dy/dt": 3}
    target_rate: str # e.g., "dz/dt"
    instance_values: Dict[str, float] # e.g., {"x": 5, "y": 12}

class RelatedRatesResponse(BaseModel):
    original_equation: str
    differentiated_equation_wrt_t: str
    target_rate_solution_expr: str # Expression for target_rate solved symbolically
    required_values_at_instance: List[str] # Variables needed at the instance
    calculated_instance_values: Dict[str, str] # Values calculated from original eq if needed
    provided_values_used: Dict[str, float] # Combined known_rates and instance_values
    target_rate_value: str # Final numeric or symbolic value
    error: Optional[str] = None


# --- Approximations ---
class FindDifferentialRequest(BaseModel):
    function_str: str # "y = expr" or just "expr"
    variable: str = 'x'
    x_value: float
    dx_value: float

class FindDifferentialResponse(BaseModel):
    derivative_f_prime_x: str
    derivative_at_x_value: str
    differential_dy_formula: str # e.g., "dy = (expr)*dx"
    differential_dy_value: str

class ApproximateValueRequest(BaseModel):
    function_str: str
    target_x: float
    base_x: float
    variable: str = 'x'

class ApproximateValueResponse(BaseModel):
    base_x: float
    target_x: float
    delta_x: float
    f_base_x: str
    derivative_f_prime_x: str
    f_prime_base_x: str
    differential_dy: str
    approximation_formula: str
    approximate_value_symbolic: str
    approximate_value_numeric: Optional[str] = None # Use .evalf()

class CalculateErrorsRequest(BaseModel):
    function_str: str # e.g., "V = (4/3)*pi*r**3" or "(4/3)*pi*r**3"
    variable: str
    measured_value: float
    possible_error_dx: float # Error in the variable measurement

class CalculateErrorsResponse(BaseModel):
    measured_variable_value: float
    possible_error_dx: float
    function_value_y: str # y = f(measured_value)
    derivative_f_prime_x: str
    derivative_at_measured_value: str
    approximate_absolute_error_dy: str # dy = f'(x)dx
    approximate_relative_error_dy_y: str # dy / y
    approximate_percentage_error: str # rel_error * 100


# --- Tangents and Normals ---
class CurveDefinition(BaseModel):
    type: str = Field(..., pattern="^(explicit|implicit|parametric)$")
    equation: Optional[str] = None # For explicit (y=f(x)) or implicit (G(x,y)=0)
    x_eq: Optional[str] = None # For parametric x(t)
    y_eq: Optional[str] = None # For parametric y(t)
    variable: str = 'x' # Independent variable (x or t)
    dependent_variable: Optional[str] = 'y' # For implicit differentiation dy/dx

class TangentNormalRequest(BaseModel):
    curve_definition: CurveDefinition
    point: Dict[str, float] # e.g. {"x": 1, "y": 1} or {"t": 0.5}

class SlopesResponse(BaseModel):
    point: Dict[str, float]
    derivative_expression_dy_dx: Optional[str] = None
    tangent_slope: Optional[str] = None # Can be 'undefined'
    normal_slope: Optional[str] = None # Can be 'undefined' or 0
    error: Optional[str] = None

class EquationsResponse(SlopesResponse): # Inherits fields from SlopesResponse
    tangent_equation_point_slope: Optional[str] = None
    tangent_equation_simplified: Optional[str] = None
    normal_equation_point_slope: Optional[str] = None
    normal_equation_simplified: Optional[str] = None

class FindPointCondition(BaseModel):
    type: str = Field(..., pattern="^(parallel_to_line|perpendicular_to_line|slope_is|passes_through_point)$")
    line_equation: Optional[str] = None # e.g., "y = 2*x + 1" or "3*x + 2*y - 5 = 0"
    slope_value: Optional[float] = None
    external_point: Optional[Dict[str, float]] = None # e.g. {"x": 0, "y": 0}

class FindPointRequest(BaseModel):
    curve_definition: CurveDefinition
    condition: FindPointCondition

class FoundPoint(BaseModel):
    point: Dict[str, str] # Coordinates as strings
    tangent_slope_at_point: Optional[str] = None

class FindPointResponse(BaseModel):
    condition_description: str
    condition_equation: str
    solved_values: Optional[List[Dict[str, str]]] = None # Solved independent var values
    found_points: List[FoundPoint]
    error: Optional[str] = None

class AngleBetweenCurvesRequest(BaseModel):
    curve1_definition: CurveDefinition
    curve2_definition: CurveDefinition

class AngleAtPoint(BaseModel):
    point: Dict[str, str]
    curve1_slope_m1: Optional[str] = None
    curve2_slope_m2: Optional[str] = None
    angle_description: str # e.g., "Orthogonal", "Tangent", "Angle is ..."
    tan_theta: Optional[str] = None # |(m1-m2)/(1+m1m2)|
    angle_radians: Optional[str] = None
    angle_degrees: Optional[str] = None

class AngleBetweenCurvesResponse(BaseModel):
    intersection_points: List[Dict[str, str]]
    angles_at_points: List[AngleAtPoint]
    error: Optional[str] = None

# --- Monotonicity ---
class MonotonicityIntervalsRequest(BaseModel):
    function_str: str # e.g., "f(x) = ..." or just expr
    variable: str = 'x'
    domain_start: Union[float, str] = "-oo"
    domain_end: Union[float, str] = "oo"

class MonotonicitySignAnalysis(BaseModel):
    interval: List[Union[str, float]] # [start, end]
    test_point: Optional[float] = None
    f_prime_value_at_test: Optional[str] = None
    f_prime_sign: Optional[str] = None # '+', '-', '0', 'undefined'
    behavior: str # 'increasing', 'decreasing', 'constant', 'undefined'

class MonotonicityIntervalsResponse(BaseModel):
    function_str: str
    derivative_f_prime_x: str
    critical_points: List[str] # Points where f'=0 or undefined
    intervals_analyzed: List[List[Union[str, float]]]
    sign_analysis: List[MonotonicitySignAnalysis]
    strictly_increasing_intervals: List[List[Union[str, float]]]
    strictly_decreasing_intervals: List[List[Union[str, float]]]
    constant_intervals: List[List[Union[str, float]]]
    error: Optional[str] = None

class MonotonicityCheckIntervalRequest(BaseModel):
    function_str: str
    variable: str = 'x'
    interval: Interval

class MonotonicityCheckIntervalResponse(BaseModel):
    interval_checked: Dict
    derivative_f_prime_x: str
    derivative_analysis_in_interval: str # Text description
    sign_of_derivative_in_interval: str # '+', '-', '0', 'mixed', 'undefined'
    is_strictly_increasing: Optional[bool] = None
    is_strictly_decreasing: Optional[bool] = None
    is_constant: Optional[bool] = None
    conclusion: str
    error: Optional[str] = None