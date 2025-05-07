# models/real_numbers_models.py
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Union

class EuclidLemmaResponse(BaseModel):
    dividend: int
    divisor: int
    quotient: int
    remainder: int
    equation: str

class HCFResponse(BaseModel):
    num1: int
    num2: int
    hcf: int

class PrimeFactorizationResponse(BaseModel):
    number: int
    factors: Dict[int, int] # prime: exponent

class HCFAndLCMResponse(BaseModel):
    num1: int
    num2: int
    prime_factorization_num1: Dict[int, int]
    prime_factorization_num2: Dict[int, int]
    hcf: int
    lcm: int

class IrrationalityCheckResponse(BaseModel):
    number_form: str # e.g. "sqrt(5)"
    is_irrational: bool
    reason: str

class DecimalExpansionResponse(BaseModel):
    numerator: int
    denominator: int
    fraction: str
    expansion_type: str # "terminating", "non-terminating recurring", "undefined"
    reason: str

class PolynomialAnalysisRequest(BaseModel):
    expression: str = Field(..., example="x**2 - 5*x + 6")

class PolynomialAnalysisResponse(BaseModel):
    expression: str
    degree: int
    coefficients: List[Union[int, float, str]] # Sympy coeffs can be fractions
    roots_found: List[str] # Roots can be complex, string representation
    sum_of_roots_vieta: str
    product_of_roots_vieta: str