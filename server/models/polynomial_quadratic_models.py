# models/polynomial_quadratic_models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

class PolynomialInput(BaseModel):
    coeffs: Optional[List[float]] = Field(None, example=[1, -3, 2], description="Coefficients from highest degree to constant (e.g., for x^2 - 3x + 2).")
    expression: Optional[str] = Field(None, example="x**2 - 3*x + 2", description="Polynomial expression string.")
    variable: str = Field('x', description="The variable symbol used in the expression.")

    # @validator('coeffs', 'expression', pre=True, always=True)
    def check_coeffs_or_expression(cls, v, values, field):
        # This custom validator is a bit tricky with Pydantic v1/v2 differences.
        # Basic idea: ensure at least one is provided.
        # A simpler way is to handle this logic in the router.
        # For now, allow both to be None and validate in the route.
        return v
    
    class Config:
        # For Pydantic V2, use model_validator
        # @model_validator(mode='before')
        # def check_at_least_one_representation(cls, data: Any) -> Any:
        #     if isinstance(data, dict):
        #         if not data.get('coeffs') and not data.get('expression'):
        #             raise ValueError('Either "coeffs" or "expression" must be provided.')
        #     return data
        pass


class PolynomialEvaluationRequest(PolynomialInput):
    x_value: Union[float, str] = Field(..., example=5, description="Value at which to evaluate the polynomial (can be complex like '2+3j').")

class PolynomialEvaluationResponse(BaseModel):
    polynomial_string: str
    coeffs: List[float]
    x_value: str # Store as string to handle complex numbers easily
    result: str  # Store as string for complex numbers

class PolynomialDivisionRequest(BaseModel):
    dividend: PolynomialInput
    divisor: PolynomialInput

class PolynomialDivisionResponse(BaseModel):
    dividend_str: str
    divisor_str: str
    quotient_coeffs: List[float]
    quotient_str: str
    remainder_coeffs: List[float]
    remainder_str: str
    equation: str # p(x) = g(x)q(x) + r(x)

class SyntheticDivisionRequest(PolynomialInput):
    a_value: Union[float, str] = Field(..., example=1, description="The value 'a' for the divisor (x - a). Can be complex.")

class SyntheticDivisionResponse(BaseModel):
    polynomial_str: str
    divisor_form: str # x - a
    quotient_coeffs: List[float]
    quotient_str: str
    remainder: str # P(a), can be complex
    
class QuadraticEquationCoeffs(BaseModel):
    a: float = Field(..., example=1)
    b: float = Field(..., example=-5)
    c: float = Field(..., example=6)

class QuadraticSolutionResponse(BaseModel):
    equation_string: str
    coefficients: QuadraticEquationCoeffs
    discriminant: Optional[float] = None
    nature_of_roots: str
    roots: List[Union[float, str]] # str for complex roots
    formula_used: Optional[str] = None

class RootsCoeffsRelationResponse(BaseModel):
    polynomial_string: str
    coeffs: List[float]
    degree: int
    relations: Dict[str, Any]
    verification_note: Optional[str] = None

class FindRootsResponse(BaseModel):
    polynomial_string: str
    coeffs: List[float]
    rational_roots_found: Optional[List[float]] = None
    all_numerical_roots: Optional[List[str]] = None # List of strings for complex numbers
    method_notes: Optional[str] = None

class FormPolynomialFromRootsRequest(BaseModel):
    roots: List[Union[float, str]] = Field(..., example=[2,3], description="List of roots (can be complex as strings e.g., '1+1j').")
    leading_coefficient: float = Field(1.0, description="Optional leading coefficient (k).")
    variable: str = Field('x', description="Variable for the polynomial.")

class FormPolynomialFromRootsResponse(BaseModel):
    roots_provided: List[str]
    polynomial_coeffs: List[Union[float,str]] # Coefficients can become complex
    polynomial_string: str
    leading_coefficient_used: float