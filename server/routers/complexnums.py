from fastapi import FastAPI, Query, APIRouter, HTTPException, Body

# from models.complex import complex
from sympy import symbols, Eq, solve, simplify, parse_expr
import math
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication_application,
)
from typing import Dict, Optional, Literal
from pydantic import BaseModel, Field
from fractions import Fraction
import cmath
from models.complex import ComplexNumber, ArithmeticRequest, ArithmeticResponse

router = APIRouter()

i = 1j  # Use cmath's representation for complex numbers


@router.post("/powers")
async def powers(n: int = Query(..., description="Power of i")):
    """
    Calculate the power of i (imaginary unit) raised to n.
    Returns the result as a complex number string (e.g., "1+0i", "0+1i", "-1+0i", "0-1i")
    or a simple integer/float if the imaginary part is zero.
    """
    try:
        # The core logic remains the same, but we calculate the result first
        if n >= 0:
            res_mod = n % 4
            if res_mod == 0:
                result = complex(1, 0)  # 1
            elif res_mod == 1:
                result = complex(0, 1)  # i
            elif res_mod == 2:
                result = complex(-1, 0)  # -1
            else:  # res_mod == 3
                result = complex(0, -1)  # -i
        else:  # n < 0
            # For negative powers, i^(-n) = 1 / i^n
            # Calculate i^|n| first
            n_abs = abs(n)
            res_mod = n_abs % 4
            if res_mod == 0:
                denominator = complex(1, 0)  # i^0 = 1
            elif res_mod == 1:
                denominator = complex(0, 1)  # i^1 = i
            elif res_mod == 2:
                denominator = complex(-1, 0)  # i^2 = -1
            else:  # res_mod == 3
                denominator = complex(0, -1)  # i^3 = -i

            # Calculate 1 / denominator
            # Avoid division by zero, although denominator should never be 0+0j here
            if denominator == complex(0, 0):
                raise HTTPException(
                    status_code=500,
                    detail="Internal error: Calculated denominator is zero.",
                )
            result = complex(1, 0) / denominator

        # Format the result for consistent output
        # Return real numbers as floats/ints, complex numbers as standard string notation
        if result.imag == 0 or result.imag == -0:
            return {"result": result.real}
        else:
            # Use cmath's default string representation (e.g., "1j", "-1j", "(1+1j)")
            # Or format explicitly if needed: f"{result.real}{result.imag:+}j"
            return {
                "result": str(result).replace("j", "i")
            }  # Replace j with i for mathematical notation

    except ValueError as ve:
        # This might catch issues if n was somehow not validated as int by FastAPI
        raise HTTPException(status_code=400, detail=f"Invalid input value: {str(ve)}")
    except Exception as e:
        # Catch any other unexpected errors
        # Log the error e for debugging purposes
        print(f"Error calculating power of i: {e}")  # Basic logging
        raise HTTPException(
            status_code=500, detail=f"An unexpected server error occurred: {str(e)}"
        )




@router.post("/arithmetic", response_model=ArithmeticResponse)
async def perform_arithmetic(request: ArithmeticRequest):
    """
    Perform arithmetic operations on two complex numbers provided in the request body.

    Supported operations: `add`, `subtract`, `multiply`, `divide`.
    Returns the result of the operation.
    """
    try:
        # Convert the Pydantic models to Python's native complex type
        z1_native = complex(request.z1.real, request.z1.img)
        z2_native = complex(request.z2.real, request.z2.img)

        if request.operation == "add":
            result = z1_native + z2_native
        elif request.operation == "subtract":
            result = z1_native - z2_native
        elif request.operation == "multiply":
            result = z1_native * z2_native
        elif request.operation == "divide":
            if z2_native == 0:
                raise HTTPException(
                    status_code=400, detail="Division by zero is not allowed."
                )
            result = z1_native / z2_native
        else:
            # This case should ideally not be reached due to Pydantic validation
            raise HTTPException(
                status_code=400, detail=f"Unsupported operation: {request.operation}"
            )

        # Format the result string replacing 'j' with 'i' and cleaning up
        if result.imag == 0:
            result_str = str(result.real)
        elif result.real == 0:
            if result.imag == 1:
                result_str = "i"
            elif result.imag == -1:
                result_str = "-i"
            else:
                result_str = f"{result.imag}i"
        else:
            # Standard format a+bi or a-bi
            result_str = str(result).replace("(", "").replace(")", "").replace("j", "i")

        # Return the dictionary matching the ArithmeticResponse model
        return {"result": result_str}

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        # Log the unexpected error for debugging
        print(f"Error during arithmetic operation: {e}")  # Replace with proper logging
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        )
