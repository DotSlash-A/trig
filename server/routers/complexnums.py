from fastapi import FastAPI, Query, APIRouter, HTTPException, Body

# from models.complex import complex
import sympy
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
from models.complex import (
    ComplexNumber,
    ArithmeticRequest,
    ArithmeticResponse,
    PropertiesResponse,
    complexStringInput,
)

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


# witout sympy
@router.post("/properties")
async def complex_properties(z: ComplexNumber = Body(...)):
    """
    Calculate the modulus and conjugate of a complex number.

    Returns:
        - Modulus (absolute value) of the complex number.
        - Conjugate of the complex number.
    """
    try:
        # Convert to Python's native complex type
        z_native = complex(z.real, z.img)

        # Calculate modulus and conjugate
        mod = abs(z_native)
        conj = z_native.conjugate()

        # Format the results
        mod_str = str(mod).replace("j", "i")
        conj_str = str(conj).replace("j", "i")

        return {
            "modulus": mod_str,
            "conjugate": conj_str,
        }
    except Exception as e:
        print(f"Error calculating properties of complex number: {e}")
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        )


# @router.post("/modandconjugate", response_model=PropertiesResponse)
# async def complex_mod_and_conjugate(data:complexStringInput):
#     """
#     Calculate the modulus and conjugate of a complex number ans uses sympy for sqrt etc.

#     Returns:
#         - Modulus (absolute value) of the complex number.
#         - Conjugate of the complex number.
#     """
# input_str_raw = data.z
# try:
#     input_str_sympy= input_str_raw.replace('√', 'sqrt').replace('^', '**')
#     local_dict: {"i":sympy.I, "sqrt":sympy.sqrt, pi


@router.post("/properties", response_model=PropertiesResponse)
async def complex_properties_from_string(data: complexStringInput):
    """
    Calculate properties (modulus, conjugate, real/imaginary parts)
    of a complex number provided as a string (e.g., "a+bi", "a + sqrt(-b)").

    The endpoint parses the string using SymPy, allowing flexible input formats.

    Returns:
        - Original input expression.
        - Calculated real part.
        - Calculated imaginary part.
        - Modulus (absolute value) as a string.
        - Conjugate as a string in standard form.
        - Standard form (a+bi) of the input number.
    """
    input_str_raw = data.z
    try:
        # Prepare input string for SymPy (replace common variations)
        input_str_sympy = input_str_raw.replace("√", "sqrt").replace("^", "**")
        # Define 'i' for SymPy parsing
        local_dict = {"i": sympy.I, "sqrt": sympy.sqrt, "pi": sympy.pi, "E": sympy.E}
        # Add transformations for implicit multiplication (e.g., '5i')
        transformations = standard_transformations + (
            implicit_multiplication_application,
        )

        # Parse the expression using SymPy
        expr = parse_expr(
            input_str_sympy, local_dict=local_dict, transformations=transformations
        )

        # Evaluate the expression to a numerical complex form if possible
        # Use evalf() for numerical evaluation
        evaluated_expr = expr.evalf()

        # Extract real and imaginary parts using SymPy's methods
        real_part_sympy = sympy.re(evaluated_expr)
        imag_part_sympy = sympy.im(evaluated_expr)

        # Ensure the parts are numerical floats
        if not (real_part_sympy.is_Number and imag_part_sympy.is_Number):
            # If it didn't evaluate to a number, try simplifying first
            simplified_expr = sympy.simplify(expr)
            real_part_sympy = sympy.re(simplified_expr)
            imag_part_sympy = sympy.im(simplified_expr)
            # Try evaluating again after simplification
            real_part_sympy = real_part_sympy.evalf()
            imag_part_sympy = imag_part_sympy.evalf()
            if not (real_part_sympy.is_Number and imag_part_sympy.is_Number):
                raise ValueError(
                    "Expression could not be fully evaluated to a numerical complex number."
                )

        real_float = float(real_part_sympy)
        imag_float = float(imag_part_sympy)

        # Create Python complex number for standard calculations
        z_native = complex(real_float, imag_float)

        # Calculate modulus and conjugate
        mod = abs(z_native)
        conj = z_native.conjugate()

        # --- Format the results nicely ---
        # Use a helper function for consistent formatting
        def format_complex(c: complex) -> str:
            real = c.real
            imag = c.imag
            # Use .g format specifier for cleaner floats (removes trailing .0)
            real_str = f"{real:.15g}"
            imag_str = f"{abs(imag):.15g}"  # Use absolute value for coefficient

            if imag == 0:
                return real_str
            elif real == 0:
                if imag == 1:
                    return "i"
                elif imag == -1:
                    return "-i"
                else:
                    return f"{imag:.15g}i"
            else:
                sign = "+" if imag > 0 else "-"
                if abs(imag) == 1:
                    return f"{real_str}{sign}i"  # Show a+i or a-i
                else:
                    return f"{real_str}{sign}{imag_str}i"

        mod_str = f"{mod:.15g}"  # Format modulus cleanly
        conj_str = format_complex(conj)
        standard_form_str = format_complex(z_native)

        return PropertiesResponse(
            input_expression=input_str_raw,
            real_part=real_float,
            imaginary_part=imag_float,
            modulus=mod_str,
            conjugate=conj_str,
            standard_form=standard_form_str,
        )

    except (SyntaxError, TypeError, ValueError) as e:
        # Catch parsing errors or evaluation issues
        raise HTTPException(
            status_code=400,
            detail=f"Invalid complex number format or expression: '{input_str_raw}'. Error: {e}",
        )
    except Exception as e:
        # Catch unexpected errors
        print(
            f"Error calculating properties of complex number '{input_str_raw}': {e}"
        )  # Log the error
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred while processing the expression.",
        )
