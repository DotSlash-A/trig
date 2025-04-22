from fastapi import FastAPI, APIRouter, HTTPException
from models.shapes import (
    circleGenral,
    CircleEqnResponse,
    CircleGeneralFormInput,
    CircleDetailsResponse,
)
import sympy
from sympy import symbols, Eq, solve, simplify, parse_expr, expand, Poly, sqrt
import math
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication_application,
)
from typing import Dict, Optional
from pydantic import BaseModel, Field
from fractions import Fraction


router = APIRouter()


# @router.post("/circle/eqn")
# async def circle_eqn(info: circleGenral):
#     """
#     Calculate the equation of a circle given its center and radius.

#     solves it usign sympy library
#     Args:
#         info (circleGenral): Circle information containing center coordinates and radius.
#     Returns:
#         CircleEquationResponse: Object containing standard and general forms
#     """
#     try:
#         x, y = symbols("x y")
#         h = info.h
#         k = info.k
#         r = info.r
#         if r < 0:
#             raise HTTPException(status_code=400, detail="Radius cannot be negative")
#         lhs = (x - h) ** 2 + (y - k) ** 2
#         rhs = r**2
#         standard_eq_sympy = Eq(lhs, rhs)
#         expanded_lhs = expand(lhs)
#         # expanded_lhs_expr = parse_expr(str(expanded_lhs), transformations="all")
#         geneal_eq_sympy_str = expanded_lhs - rhs
#         geneal_eq_sympy = Eq(geneal_eq_sympy_str, 0)

#         def format_eq(eq):
#             eq_str = str(eq)
#             eq_str = eq_str.replace("**", "^")
#             if eq_str.startswith("Eq("):
#                 eq_str = eq_str[3:-1]
#             eq_str = eq_str.replace(",", "=", 1)
#             return eq_str

#         standard_eq_sympy_str = format_eq(standard_eq_sympy)
#         geneal_eq_sympy_str = format_eq(geneal_eq_sympy)

#         poly = Poly(geneal_eq_sympy_str, x, y)
#         coeffs = poly.coeffs()
#         terms = poly.monoms()
#         coeffs_dict = {monom: coeff for monom, coeff in zip(terms, coeffs)}
#         A = coeffs_dict.get((2, 0), 0)
#         B = coeffs_dict.get((0, 2), 0)
#         C = coeffs_dict.get((1, 0), 0)
#         D = coeffs_dict.get((0, 1), 0)
#         E = coeffs_dict.get((0, 0), 0)

#         # # Equation of the circle: (x - h)^2 + (y - k)^2 = r^2
#         # equation = Eq((x - h) ** 2 + (y - k) ** 2, r**2)
#         # # Simplify the equation
#         # simplified_eq = simplify(equation)
#         # # Convert to standard form
#         # standard_form = str(simplified_eq).replace("**", "^")
#         # Extract coefficients for the general form Ax^2 + By^2 + Cx + Dy + E = 0
#         return CircleEqnResponse(
#             standard_form=standard_eq_sympy_str,
#             general_form=geneal_eq_sympy_str,
#             center_h=info.h,
#             center_k=info.k,
#             radius=info.r,
#             A=A,
#             B=B,
#             C=C,
#             D=D,
#             E=E,
#         )
#     except HTTPException as http_exc:
#         raise http_exc  # Re-raise validation errors
#     except Exception as e:
#         print(f"Error calculating circle equation: {e}")  # Log error
#         raise HTTPException(
#             status_code=500, detail=f"An internal server error occurred: {e}"
#         )
# ... (imports and router setup) ...


@router.post("/circle/eqn")
async def circle_eqn(info: circleGenral):
    """
    Calculate the equation of a circle given its center and radius.

    solves it usign sympy library
    Args:
        info (circleGenral): Circle information containing center coordinates and radius.
    Returns:
        CircleEquationResponse: Object containing standard and general forms
    """
    try:
        x, y = symbols("x y")
        # Use sympify to ensure h, k, r are SymPy numbers if they aren't already
        h = sympy.sympify(info.h)
        k = sympy.sympify(info.k)
        r = sympy.sympify(info.r)

        if r < 0:
            raise HTTPException(status_code=400, detail="Radius cannot be negative")

        lhs = (x - h) ** 2 + (y - k) ** 2
        rhs = r**2
        standard_eq_sympy = Eq(lhs, rhs)
        expanded_lhs = expand(lhs)

        # This is the SymPy EXPRESSION for the general form polynomial (Ax^2 + ... + E)
        general_eq_expression = expanded_lhs - rhs
        # This is the SymPy EQUATION object (expression = 0)
        general_eq_sympy_eq_obj = Eq(general_eq_expression, 0)

        def format_eq(eq):
            eq_str = str(eq)
            eq_str = eq_str.replace("**", "^")
            # Handle Eq() formatting
            if eq_str.startswith("Eq("):
                # Extract LHS and RHS from Eq(LHS, RHS)
                parts = eq_str[3:-1].split(", ", 1)
                if len(parts) == 2:
                    return f"{parts[0]} = {parts[1]}"
            # Fallback for simple expression formatting if needed (though less likely for Eq)
            return eq_str.replace(
                ",", "=", 1
            )  # Keep this as a fallback? Might not be needed.

        standard_form_str = format_eq(standard_eq_sympy)
        general_form_str = format_eq(
            general_eq_sympy_eq_obj
        )  # Format the equation object

        # Use the EXPRESSION for Poly, not the formatted string or the equation object
        poly = Poly(general_eq_expression, x, y)
        coeffs = poly.coeffs()
        terms = poly.monoms()
        coeffs_dict = {monom: coeff for monom, coeff in zip(terms, coeffs)}
        # Extract coefficients as floats
        A = float(coeffs_dict.get((2, 0), 0))  # x^2
        B = float(coeffs_dict.get((0, 2), 0))  # y^2
        C = float(coeffs_dict.get((1, 0), 0))  # x
        D = float(coeffs_dict.get((0, 1), 0))  # y
        E = float(coeffs_dict.get((0, 0), 0))  # constant

        return CircleEqnResponse(
            standard_form=standard_form_str,
            general_form=general_form_str,
            center_h=float(h),  # Return as float
            center_k=float(k),  # Return as float
            radius=float(r),  # Return as float
            A=A,
            B=B,
            C=C,
            D=D,
            E=E,
        )
    except HTTPException as http_exc:
        raise http_exc  # Re-raise validation errors
    except Exception as e:
        print(f"Error calculating circle equation: {e}")  # Log error
        # Provide more specific error detail if possible
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred: {type(e).__name__} - {e}",
        )


@router.post("/circle/details", response_model=CircleDetailsResponse)
async def circle_details_from_general_form(data: CircleGeneralFormInput):
    """
    Calculate the center (h, k) and radius (r) of a circle from its
    general form equation (Ax^2 + By^2 + Cx + Dy + E = 0).

    Args:
        data (CircleGeneralFormInput): Object containing the equation string.

    Returns:
        CircleDetailsResponse: Object containing center coordinates and radius.
    """
    input_eq_str = data.equation
    try:
        x, y = symbols("x y")

        # Prepare string for parsing
        eq_str_sympy = input_eq_str.replace("^", "**")

        # Check if '=' is present and split
        if "=" not in eq_str_sympy:
            raise ValueError("Equation must contain '=' sign.")
        lhs_str, rhs_str = eq_str_sympy.split("=", 1)

        # Define local dict for parsing
        local_dict = {"x": x, "y": y}
        transformations = standard_transformations + (
            implicit_multiplication_application,
        )

        # Parse both sides and create the expression for LHS - RHS
        lhs_expr = parse_expr(
            lhs_str.strip(), local_dict=local_dict, transformations=transformations
        )
        rhs_expr = parse_expr(
            rhs_str.strip(), local_dict=local_dict, transformations=transformations
        )
        polynomial_expr = expand(lhs_expr - rhs_expr)  # This should equal 0

        # Create Poly object to extract coefficients
        poly = Poly(polynomial_expr, x, y)
        coeffs_dict = poly.as_dict()  # Get coeffs as {(x_pow, y_pow): coeff}

        # Extract coefficients A, B, C, D, E
        A = coeffs_dict.get((2, 0), 0)  # x^2
        B = coeffs_dict.get((0, 2), 0)  # y^2
        C = coeffs_dict.get((1, 0), 0)  # x
        D = coeffs_dict.get((0, 1), 0)  # y
        E = coeffs_dict.get((0, 0), 0)  # constant

        # --- Validate if it's a circle ---
        if A == 0 or B == 0:
            raise ValueError("Not a valid circle/ellipse equation (A or B is zero).")
        if A != B:
            raise ValueError(
                f"Not a circle equation (coefficient of x^2 ({A}) != coefficient of y^2 ({B}))."
            )

        # --- Normalize coefficients (divide by A if A is not 1) ---
        normalized_eq_str = None
        if A != 1:
            C = C / A
            D = D / A
            E = E / A
            # Optional: Create normalized equation string for output
            normalized_expr = x**2 + y**2 + C * x + D * y + E
            normalized_eq_str = f"{str(normalized_expr).replace('**','^')} = 0"
            A = 1  # A and B are now effectively 1
            B = 1

        # --- Calculate center (h, k) ---
        # h = -C / (2*A) -> h = -C / 2 (since A=1 now)
        # k = -D / (2*B) -> k = -D / 2 (since B=1 now)
        h = -C / 2
        k = -D / 2

        # --- Calculate radius (r) ---
        # r^2 = h^2 + k^2 - E/A -> r^2 = h^2 + k^2 - E (since A=1 now)
        radius_squared = h**2 + k**2 - E

        if radius_squared < 0:
            raise ValueError(
                f"Not a real circle (radius squared = {radius_squared} is negative)."
            )

        r = sqrt(radius_squared)

        # Convert SymPy numbers to Python floats for the response
        h_float = round(float(h), 3)
        k_float = round(float(k), 3)
        r_float = round(float(r), 3)

        return CircleDetailsResponse(
            center_h=h_float,
            center_k=k_float,
            radius=r_float,
            input_equation=input_eq_str,
            normalized_equation=normalized_eq_str,
        )

    except (SyntaxError, TypeError, ValueError) as e:
        # Catch parsing errors or validation errors
        raise HTTPException(
            status_code=400, detail=f"Invalid equation or not a valid circle: {e}"
        )
    except Exception as e:
        # Catch unexpected errors
        print(f"Error calculating circle details: {e}")  # Log error
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred: {type(e).__name__} - {e}",
        )

# def trying_matrices():
