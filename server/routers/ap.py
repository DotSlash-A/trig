# Existing imports...
from fastapi import FastAPI, Query, APIRouter, HTTPException
from sympy import symbols, Eq, solve, simplify, parse_expr
import math

from typing import Dict, Optional, List, Union, Tuple
from pydantic import BaseModel, Field
from fractions import Fraction
# Assuming matrix models are in a 'models' directory relative to this script
# from models.matrix_model import (
#     MatrixInputAPI,
#     DetInput,
#     DeterminantResponseAPI,
#     TwoMatrixInput,
#     MatrixEqualityResponse,
#     MatrixFormulaInput,
#     ConstructedMatrixResponse,
#     MinorsCofactorsResponse,
#     AdjInvResponse,
#     # MatrixInput, # Assuming this is also defined if needed elsewhere
# )

# import numpy as np # Not used in progressions code below
# import sympy # Not used in progressions code below

# --- Router Definition ---
# Using a more specific prefix like /progressions as in the original code
router = APIRouter(prefix="/progressions", tags=["Progressions"])

# --- Test Route ---
@router.get("/test")
async def test():
    return {"message": "Progressions API is working!"}

# --- Arithmetic Progression (AP) Section ---

class APRequest(BaseModel): # Renamed from APREquest for consistency
    """
    Request model for basic AP calculations.
    """
    a: float = Field(..., description="First term of the arithmetic progression")
    d: float = Field(..., description="Common difference of the arithmetic progression")
    n: int = Field(..., description="Number of terms in the arithmetic progression (must be positive integer)")

@router.post("/ap/basic") # Added /ap prefix for clarity
async def arithmetic_progression(request: APRequest):
    """
    Calculate the nth term and sum of the first n terms of an arithmetic progression.
    """
    a = request.a
    d = request.d
    n = request.n

    if n <= 0:
        raise HTTPException(status_code=400, detail="Number of terms (n) must be a positive integer.")

    # Calculate nth term
    nth_term = a + (n - 1) * d

    # Calculate sum of first n terms
    sum_n_terms = (n / 2) * (2 * a + (n - 1) * d)

    return {
        "nth_term": nth_term,
        "sum_n_terms": sum_n_terms
    }

class APLastTermRequest(BaseModel): # Renamed from ap_lastterm
    """
    Request model for nth term from last term calculation.
    """
    # a: float = Field(..., description="First term of the arithmetic progression") # 'a' is not needed if 'l' and 'd' are known
    d: float = Field(..., description="Common difference of the arithmetic progression")
    l: float = Field(..., description="Last term in the arithmetic progression")
    n: int = Field(..., description="Which term from the end (e.g., n=1 is the last term, n=2 is second last)")

@router.post("/ap/nth_term_from_last") # Added /ap prefix
async def nth_term_from_last(request: APLastTermRequest):
    """
    Calculate the nth term from the last term of an arithmetic progression.
    Example: If n=1, it returns the last term 'l'. If n=2, it returns the second last term.
    """
    # a = request.a # Not needed for this calculation
    d = request.d
    l = request.l
    n = request.n

    if n <= 0:
        raise HTTPException(status_code=400, detail="Term number from end (n) must be a positive integer.")

    # Calculate nth term from the last term: T_n_from_end = l - (n - 1) * d
    nth_term_from_last = l - (n - 1) * d

    return {
        "nth_term_from_last": nth_term_from_last
    }

class APMiddleTermInput(BaseModel): # Renamed from MiddleTermInput
    """Input model to find the middle term(s) given first term, common difference, and last term."""
    a: float = Field(..., description="First term of the arithmetic progression")
    d: float = Field(..., description="Common difference of the arithmetic progression")
    last_term: float = Field(..., description="Value of the last term in the arithmetic progression")

@router.post("/ap/middle_term") # Added /ap prefix
async def middle_term_ap(request: APMiddleTermInput): # Renamed function for clarity
    """
    Calculate the middle term(s) of an arithmetic progression given the first term,
    common difference, and the value of the last term.
    """
    a = request.a
    d = request.d
    an = request.last_term

    if d == 0:
        if a == an:
             # Infinite terms if a == an. Middle term concept is ill-defined.
             # Let's return the constant value 'a' but indicate the ambiguity.
             return {"middle_term(s)": [a], "number_of_terms": "Infinite (constant sequence)", "message": "Progression has constant terms. Any term can be considered 'middle'."}
        else:
            raise HTTPException(status_code=400, detail="Invalid progression: d is 0 but first and last terms differ.")

    # Calculate the number of terms n
    # Check if last_term is reachable with the given a and d
    # (an - a) must be a multiple of d, and (an - a)/d must be non-negative integer
    if abs(an - a) < 1e-9: # Handle floating point comparison for a == an
        n = 1
    elif (an - a) / d < -1e-9: # last term is before first term with positive d, or vice versa
        raise HTTPException(status_code=400, detail="Last term is not reachable from the first term with the given common difference.")
    else:
        n_float = (an - a) / d + 1
        # Check if n_float is close to an integer
        if abs(n_float - round(n_float)) > 1e-9:
             raise HTTPException(status_code=400, detail="The provided last term is not part of the arithmetic progression defined by 'a' and 'd'.")
        n = int(round(n_float)) # Use rounding to handle potential floating point inaccuracies
        if n <= 0:
             raise HTTPException(status_code=400, detail="Invalid progression parameters leading to non-positive number of terms.")


    # Calculate middle term(s)
    middle_terms = []
    middle_indices = [] # Store the index/indices (1-based)

    if n % 2 == 1: # Odd number of terms
        middle_index = (n + 1) // 2
        # Value: a + (middle_index - 1) * d
        middle_term_value = a + (middle_index - 1) * d
        middle_terms.append(middle_term_value)
        middle_indices.append(middle_index)
    else: # Even number of terms
        middle_index1 = n // 2
        middle_index2 = n // 2 + 1
        # Value 1: a + (middle_index1 - 1) * d
        middle_term1_value = a + (middle_index1 - 1) * d
        # Value 2: a + (middle_index2 - 1) * d
        middle_term2_value = a + (middle_index2 - 1) * d
        middle_terms.append(middle_term1_value)
        middle_terms.append(middle_term2_value)
        middle_indices.append(middle_index1)
        middle_indices.append(middle_index2)


    return {
        "number_of_terms": n,
        "middle_term_indices": middle_indices,
        "middle_term(s)": middle_terms
    }

# --- Geometric Progression (GP) Section ---

class GPRequest(BaseModel):
    """
    Request model for basic GP calculations.
    """
    a: float = Field(..., description="First term of the geometric progression (cannot be 0 if n > 1 and r=0)")
    r: float = Field(..., description="Common ratio of the geometric progression")
    n: int = Field(..., description="Number of terms in the geometric progression (must be positive integer)")

@router.post("/gp/basic") # Added /gp prefix
async def geometric_progression(request: GPRequest):
    """
    Calculate the nth term and sum of the first n terms of a geometric progression.
    Handles the case where r = 1 and potential floating point issues.
    """
    a = request.a
    r = request.r
    n = request.n

    if n <= 0:
        raise HTTPException(status_code=400, detail="Number of terms (n) must be a positive integer.")
    if a == 0 and n > 1 and r == 0:
         raise HTTPException(status_code=400, detail="Ambiguous case: a=0, r=0, n>1. Sequence is 0, 0,... but formulas can lead to 0^0.")
    if a == 0: # If a=0, all terms are 0 (unless n=1 and r=0)
        nth_term = 0.0
        sum_n_terms = 0.0
        return {
            "nth_term": nth_term,
            "sum_n_terms": sum_n_terms
        }

    # Calculate nth term: T_n = a * r^(n-1)
    # Handle 0^0 case which occurs if r=0 and n=1. Result should be 'a'.
    # math.pow(0, 0) is 1.0. a * math.pow(0, 0) = a. So it works.
    try:
        nth_term = a * math.pow(r, n - 1)
    except ValueError as e: # Handles potential issues like negative base to fractional power
        raise HTTPException(status_code=400, detail=f"Cannot calculate nth term: {e}")


    # Calculate sum of first n terms: S_n
    if abs(r - 1.0) < 1e-9: # Check if r is very close to 1
        sum_n_terms = n * a
    else:
        # S_n = a * (r^n - 1) / (r - 1)
        try:
            numerator = a * (math.pow(r, n) - 1)
            denominator = r - 1
            if abs(denominator) < 1e-15: # Avoid division by almost zero if r wasn't caught above
                 raise HTTPException(status_code=400, detail="Common ratio 'r' is too close to 1, leading to potential division instability.")
            sum_n_terms = numerator / denominator
        except ValueError as e: # Handles potential issues like negative base to fractional power if n is not int (though n is int here)
             raise HTTPException(status_code=400, detail=f"Cannot calculate sum: {e}")
        except OverflowError:
             raise HTTPException(status_code=400, detail="Calculation resulted in overflow. Check input values (large n or r).")


    return {
        "nth_term": nth_term,
        "sum_n_terms": sum_n_terms
    }

class GPSumInfinityRequest(BaseModel):
    """
    Request model for calculating the sum to infinity of a GP.
    """
    a: float = Field(..., description="First term of the geometric progression")
    r: float = Field(..., description="Common ratio of the geometric progression")

@router.post("/gp/sum_infinity") # Added /gp prefix
async def geometric_progression_sum_infinity(request: GPSumInfinityRequest):
    """
    Calculate the sum to infinity of a geometric progression.
    Requires the absolute value of the common ratio |r| to be less than 1.
    """
    a = request.a
    r = request.r

    if abs(r) >= 1:
        raise HTTPException(status_code=400, detail="Sum to infinity exists only if the absolute value of the common ratio |r| is less than 1.")

    # Formula: S_inf = a / (1 - r)
    # Check for denominator being zero (already covered by |r| < 1 check, but good practice)
    denominator = 1 - r
    if abs(denominator) < 1e-15: # Extremely unlikely given |r| < 1, but for safety
         raise HTTPException(status_code=500, detail="Internal calculation error: denominator is too close to zero.")

    sum_to_infinity = a / denominator

    return {
        "sum_to_infinity": sum_to_infinity
    }

class GeometricMeanRequest(BaseModel):
    """
    Request model for calculating the geometric mean of two numbers.
    """
    num1: float = Field(..., description="First number")
    num2: float = Field(..., description="Second number")

@router.post("/gp/geometric_mean") # Added /gp prefix
async def geometric_mean(request: GeometricMeanRequest):
    """
    Calculate the geometric mean (GM) of two numbers.
    For real results, both numbers must have the same sign (or one/both be zero).
    Typically, GM is defined for positive numbers. We'll handle non-negative inputs.
    """
    num1 = request.num1
    num2 = request.num2

    if num1 < 0 or num2 < 0:
        # Option 1: Raise error for negative inputs (common definition)
        raise HTTPException(status_code=400, detail="Geometric Mean is typically defined for non-negative numbers.")
        # Option 2: Allow calculation if product is non-negative (e.g., two negatives) -> GM = sqrt(num1*num2)
        # product = num1 * num2
        # if product < 0:
        #     raise HTTPException(status_code=400, detail="Cannot calculate real Geometric Mean for numbers with opposite signs.")
        # gm = math.copysign(math.sqrt(abs(product)), num1) # Preserve sign? Usually GM is positive. Let's stick to non-negative inputs.
    elif num1 == 0 or num2 == 0:
        gm = 0.0
    else:
        # GM = sqrt(num1 * num2)
        gm = math.sqrt(num1 * num2)

    return {
        "geometric_mean": gm
    }

class InsertGMeansRequest(BaseModel):
    """
    Request model for inserting geometric means between two numbers.
    """
    a: float = Field(..., description="The first number (start of the sequence)")
    b: float = Field(..., description="The last number (end of the sequence)")
    k: int = Field(..., description="The number of geometric means to insert (must be positive integer)")

@router.post("/gp/insert_means") # Added /gp prefix
async def insert_geometric_means(request: InsertGMeansRequest):
    """
    Inserts 'k' geometric means between two numbers 'a' and 'b'.
    Forms a GP: a, G1, G2, ..., Gk, b. Total n = k + 2 terms.
    """
    a = request.a
    b = request.b
    k = request.k

    if k <= 0:
        raise HTTPException(status_code=400, detail="Number of means to insert (k) must be a positive integer.")
    if a == 0:
        if b == 0:
             # Means between 0 and 0 are all 0
             means = [0.0] * k
             return {"geometric_means": means}
        else: # a=0, b!=0
             raise HTTPException(status_code=400, detail="Cannot insert geometric means starting from 0 to a non-zero number (requires infinite or zero ratio).")
    # If a != 0
    # The sequence is a, G1, ..., Gk, b. This has k+2 terms.
    # The last term b is the (k+2)th term.
    # b = a * r^((k+2) - 1) = a * r^(k+1)
    # r^(k+1) = b / a
    try:
        ratio_power_k_plus_1 = b / a
    except ZeroDivisionError: # Should be caught by a==0 check above, but for safety.
         raise HTTPException(status_code=400, detail="First number 'a' cannot be zero.")

    # Need to calculate r = (b/a)^(1/(k+1))
    # Handle potential complex numbers if b/a is negative and k+1 is even.
    # For typical 12th grade real context, raise error in this case.
    root_index = k + 1
    if ratio_power_k_plus_1 < 0 and root_index % 2 == 0:
        raise HTTPException(status_code=400, detail=f"Cannot compute real geometric means: requires taking an even root ({root_index}) of a negative number ({ratio_power_k_plus_1}).")

    try:
        # If ratio_power_k_plus_1 is negative and root_index is odd, result is negative.
        # Use formula that handles signs correctly: sign(x) * |x|^(1/n)
        if ratio_power_k_plus_1 < 0:
            r = -math.pow(abs(ratio_power_k_plus_1), 1.0 / root_index)
        elif ratio_power_k_plus_1 == 0: # This means b=0
            r = 0.0
        else: # Positive base
            r = math.pow(ratio_power_k_plus_1, 1.0 / root_index)

    except ValueError as e: # Catch potential math domain errors
         raise HTTPException(status_code=400, detail=f"Error calculating common ratio: {e}")
    except OverflowError:
         raise HTTPException(status_code=400, detail="Overflow error calculating common ratio. Check input values.")


    # Calculate the means: G_i = a * r^i for i = 1, 2, ..., k
    means = []
    current_term = a
    try:
        for i in range(k):
            current_term *= r
            means.append(current_term)
    except OverflowError:
         raise HTTPException(status_code=400, detail="Overflow error calculating means. Check input values or resulting ratio.")

    return {
        "common_ratio_used": r,
        "geometric_means": means
    }

# --- You would typically include this router in your main FastAPI app ---
# Example main app (assuming this code is in a file like 'progressions_routes.py'):
#
# from fastapi import FastAPI
# from . import progressions_routes # Import the router defined above
#
# app = FastAPI()
#
# app.include_router(progressions_routes.router)
#
# @app.get("/")
# async def root():
#     return {"message": "Welcome to the Math API"}