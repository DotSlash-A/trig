# models/linear_equations_models.py
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union

class LinearEquationInput(BaseModel):
    a: float = Field(..., description="Coefficient of x")
    b: float = Field(..., description="Coefficient of y")
    c: float = Field(..., description="Constant term")

class PairOfLinearEquationsRequest(BaseModel):
    eq1: LinearEquationInput = Field(..., description="First linear equation: a1x + b1y + c1 = 0 OR a1x + b1y = c1")
    eq2: LinearEquationInput = Field(..., description="Second linear equation: a2x + b2y + c2 = 0 OR a2x + b2y = c2")
    # Standardize to ax + by = c or ax + by + c = 0
    # Let's assume 'c' is on the RHS by default for ax+by=c form for direct use in solver
    # Or add a field to specify form. For cross-multiplication, ax+by+c=0 is standard.
    # Let's define the model input as ax+by=c, and adjust internally for methods like cross-multiplication
    # OR have a general input form where user specifies.
    # For simplicity: Let's ask for coefficients of a1x + b1y = c1 and a2x + b2y = c2 directly.
    # If ax+by+c=0 is given, user needs to shift c.
    # For cross-multiplication route, we can specify that input c should be from ax+by+c=0 form.

class EquationsCoeffs(BaseModel):
    a1: float
    b1: float
    c1: float # Represents the RHS constant for a1x + b1y = c1
    a2: float
    b2: float
    c2: float # Represents the RHS constant for a2x + b2y = c2

class ConsistencyCheckResponse(BaseModel):
    equations: EquationsCoeffs
    consistency_type: str = Field(..., examples=["consistent_unique", "consistent_infinite", "inconsistent_parallel"])
    description: str
    ratios: Dict[str, Union[float, str]] # e.g. {"a1/a2": 0.5, "b1/b2": "undefined (b1/0)"}
    graphical_interpretation: str

class SolutionResponse(BaseModel):
    equations: EquationsCoeffs
    consistency_type: str
    description: str
    solution_x: Optional[Union[float, str]] = None # Can be "Infinite"
    solution_y: Optional[Union[float, str]] = None # Can be "Infinite"
    method_used: Optional[str] = None
    steps: Optional[List[str]] = None # For methods like substitution, elimination

# For reducible equations
class ReducibleEquationPair(BaseModel):
    eq1_str: str = Field(..., example="2/x + 3/y = 13", description="First equation string")
    eq2_str: str = Field(..., example="5/x - 4/y = -2", description="Second equation string")
    # This is more complex as it requires parsing symbolic equations.
    # A simpler approach for the API is to ask the user for the substituted linear form.
    # e.g., "Let p=1/x, q=1/y. Enter equations for p and q."

class SubstitutedEquationsRequest(BaseModel):
    # Equations in terms of new variables u and v (e.g. u=1/x, v=1/y)
    # u_coeff1*u + v_coeff1*v = const1
    # u_coeff2*u + v_coeff2*v = const2
    u_coeff1: float
    v_coeff1: float
    const1: float
    u_coeff2: float
    v_coeff2: float
    const2: float
    original_var_u: str = Field("1/x", description="What u represents, e.g., '1/x'")
    original_var_v: str = Field("1/y", description="What v represents, e.g., '1/y'")

class ReducibleSolutionResponse(SolutionResponse):
    original_solution_x: Optional[Union[float, str]] = None
    original_solution_y: Optional[Union[float, str]] = None
    substituted_equations_details: Dict[str, Any]