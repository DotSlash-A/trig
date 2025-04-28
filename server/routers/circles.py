from fastapi import FastAPI, APIRouter, HTTPException
from models.shapes import (
    circleGenral,
    CircleEqnResponse,
    CircleGeneralFormInput,
    CircleDetailsResponse,
    circleWThreePointsInput,
    coordinates,
    linegeneral,
)
import sympy
from sympy import (
    symbols,
    Eq,
    solve,
    simplify,
    parse_expr,
    expand,
    Poly,
    sqrt,
    Matrix,
    sympify,
)
import math
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication_application,
)
from typing import Dict, Optional
from pydantic import BaseModel, Field
from fractions import Fraction
import numpy as np

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


@router.post("/circle/3points")
async def circle_from_three_points(data: circleWThreePointsInput):
    """
    Calculate the equation of a circle given three points on its circumference.

    Args:
        data (circleWThreePointsInput): Object containing three points.

    Returns:
        CircleEqnResponse: Object containing standard and general forms
    """
    try:
        # Extract points from the input data
        p = data.p
        q = data.q
        r = data.r

        # c = data.center
        def calc_c(x, y):
            return -(x**2 + y**2)

        # Extract coordinates from the points
        x1, y1 = p.x, p.y
        x2, y2 = q.x, q.y
        x3, y3 = r.x, r.y
        c1 = calc_c(x1, y1)
        c2 = calc_c(x2, y2)
        c3 = calc_c(x3, y3)

        # Calculate the determinants needed for the circle equation
        # A = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2

        A = np.array([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])  # Correct syntax
        B = np.array([[c1, y1, 1], [c2, y2, 1], [c3, y3, 1]])  # Correct syntax
        C = np.array([[x1, c1, 1], [x2, c2, 1], [x3, c3, 1]])  # Correct syntax
        D = np.array([[x1, y1, c1], [x2, y2, c2], [x3, y3, c3]])  # Correct syntax

        # Calculate the coefficients of the circle equation
        A_det = np.linalg.det(A)  # Renamed to avoid conflict with symbol A
        B_det = np.linalg.det(B)
        C_det = np.linalg.det(C)
        D_det = np.linalg.det(D)

        x, y = sympy.symbols("x y")  # Define symbols x, y

        # Define the equation using sympy.Eq and explicit multiplication
        # Equation is: A_det*(x**2 + y**2) + B_det*x + C_det*y + D_det = 0
        eqn = sympy.Eq(A_det * (x**2 + y**2) + B_det * x + C_det * y + D_det, 0)
        # Convert the equation to a string representation
        simplified_eq = sympy.simplify(eqn)
        simplified_eq = (
            str(eqn).replace("**", "^").replace("Eq(", "").replace(", 0)", "")
        )
        simplified_eq = simplified_eq.replace(
            " ", ""
        )  # Remove spaces for cleaner output
        print(f"Circle equation: {simplified_eq}")
        return {"standard_form": simplified_eq}
    except Exception as e:
        print(f"Error calculating circle equation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred: {type(e).__name__} - {e}",
        )


@router.post("/circle/3points/det")
async def circle_from_three_points_det(data: circleWThreePointsInput):
    """
    Calculate the equation of a circle given three points on its circumference
    uses determinants to calculate the circle equation.

    """
    try:
        # Extract points from the input data
        p = data.p
        q = data.q
        r = data.r

        # c = data.center
        # def calc_c(x, y):
        #     return (x**2 + y**2)
        x, y = symbols("x y")  # Define symbols x, y

        # Extract coordinates from the points
        x1, y1 = p.x, p.y
        x2, y2 = q.x, q.y
        x3, y3 = r.x, r.y
        # c1 = calc_c(x1, y1)
        # c2 = calc_c(x2, y2)
        # c3 = calc_c(x3, y3)
        matrix_sympy = Matrix(
            [
                [x**2 + y**2, x, y, 1],
                [sympy.sympify(x1**2 + y1**2), sympy.sympify(x1), sympy.sympify(y1), 1],
                [sympy.sympify(x2**2 + y2**2), sympy.sympify(x2), sympy.sympify(y2), 1],
                [sympy.sympify(x3**2 + y3**2), sympy.sympify(x3), sympy.sympify(y3), 1],
            ]
        )
        # Calculate the determinant of the matrix
        det_expr = matrix_sympy.det()
        # simplify sing sympy
        circle_eq_expr = sympy.expand(det_expr)
        # Convert the determinant to a string representation
        det_str = str(circle_eq_expr).replace("**", "^").replace(" ", "")
        det_str = det_str.replace("x**2", "x^2").replace("y**2", "y^2")
        det_str = det_str.replace("x**1", "x").replace("y**1", "y")
        return {"determinant": det_str}
    except Exception as e:
        print(f"Error calculating circle equation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred: {type(e).__name__} - {e}",
        )


@router.post("/circle/3points/det1")
async def circle_from_three_points_det1(data: circleWThreePointsInput):
    """
    Calculate the equation of a circle given three points on its circumference
    using the determinant method with SymPy. The determinant being zero
    represents the circle equation.
    """
    try:
        # Extract points from the input data
        p = data.p
        q = data.q
        r = data.r

        x, y = symbols("x y")  # Define symbols x, y

        # Extract coordinates from the points
        x1, y1 = p.x, p.y
        x2, y2 = q.x, q.y
        x3, y3 = r.x, r.y

        # Create the matrix using sympy.Matrix
        # Use sympify to ensure numbers are treated correctly by SymPy if needed
        matrix_sympy = Matrix(
            [
                [x**2 + y**2, x, y, 1],
                [sympy.sympify(x1**2 + y1**2), sympy.sympify(x1), sympy.sympify(y1), 1],
                [sympy.sympify(x2**2 + y2**2), sympy.sympify(x2), sympy.sympify(y2), 1],
                [sympy.sympify(x3**2 + y3**2), sympy.sympify(x3), sympy.sympify(y3), 1],
            ]
        )

        # Calculate the determinant using the .det() method of the SymPy Matrix
        det_expr = matrix_sympy.det()

        # The determinant expression itself represents the circle equation (det = 0)
        # Expand the determinant expression for a more standard polynomial form
        circle_eq_expr = sympy.expand(det_expr)

        # --- Optional: Simplify and Format ---
        # Simplify might try to factor, expand usually gives the Ax^2+By^2+... form
        # simplified_expr = sympy.simplify(circle_eq_expr) # Optional

        # Format the expression string
        # Helper to format coefficients nicely
        def format_coeff(value, var_name):
            if abs(value) < 1e-9:
                return ""
            sign = "-" if value < 0 else "+"
            num = abs(value)
            num_str = f"{num:.4g}" if abs(num - 1.0) > 1e-9 or not var_name else ""
            var_part = f"*{var_name}" if num_str and var_name else var_name
            return f" {sign} {num_str}{var_part}"

        # Extract coefficients from the expanded expression
        poly = Poly(circle_eq_expr, x, y)
        coeffs_dict = poly.as_dict()

        A = coeffs_dict.get((2, 0), 0)  # x^2 coeff
        B = coeffs_dict.get((0, 2), 0)  # y^2 coeff (Should equal A)
        C = coeffs_dict.get((1, 0), 0)  # x coeff
        D = coeffs_dict.get((0, 1), 0)  # y coeff
        E = coeffs_dict.get((0, 0), 0)  # constant

        # Check for collinearity (A should be non-zero)
        if abs(A) < 1e-9:
            raise ValueError("Points are collinear, cannot form a unique circle.")

        # Normalize (divide by A to get x^2 + y^2 + ...)
        C_prime = C / A
        D_prime = D / A
        E_prime = E / A

        # Build the string
        general_form_str = (
            (
                f"x^2 + y^2"
                f"{format_coeff(C_prime, 'x')}"
                f"{format_coeff(D_prime, 'y')}"
                f"{format_coeff(E_prime, '')}"
                " = 0"
            )
            .replace("+ -", "- ")
            .replace(" + ", " + ")
            .replace(" - ", " - ")
            .strip()
        )
        if general_form_str.startswith(" + "):
            general_form_str = general_form_str[3:]
        if general_form_str == "x^2 + y^2":
            general_form_str += " = 0"

        # Return the formatted equation string
        return {"circle_equation": general_form_str}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error calculating circle equation via determinant: {e}")
        # More specific error might be helpful depending on SymPy exceptions
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred: {type(e).__name__} - {e}",
        )


def substitute_point_in_circle_eqn(circle_eqn: str, point: coordinates):
    """
    Substitute a point into the circle equation and check if it satisfies the equation.

    Args:
        circle_eqn (str): Circle equation in string format.
        point (tuple): Point coordinates (x, y).

    Returns:
        bool: True if the point satisfies the equation, False otherwise.
    """
    x, y = symbols("x y")
    # Parse the circle equation
    circle_eqn = parse_expr(circle_eqn.replace("^", "**"))
    # Substitute the point into the equation
    substituted_eqn = circle_eqn.subs({x: point.x, y: point.y})
    # Check if the substituted equation is equal to 0
    return substituted_eqn


def format_sympy_equation(eq_expr, equals_zero=True):
    """Formats a SymPy expression into a string like 'expr = 0'."""
    eq_str = str(eq_expr).replace("**", "^")
    # Basic formatting for plus/minus signs
    eq_str = eq_str.replace("+ -", "- ")
    eq_str = eq_str.replace(" - ", " - ")
    eq_str = eq_str.replace(" + ", " + ")
    if eq_str.startswith(" + "):
        eq_str = eq_str[3:]
    if equals_zero:
        return f"{eq_str} = 0"
    else:
        # Handle Eq objects if needed, though here we primarily deal with expressions
        if eq_str.startswith("Eq("):
            parts = eq_str[3:-1].split(", ", 1)
            if len(parts) == 2:
                return f"{parts[0]} = {parts[1]}"
        return eq_str  # Fallback


@router.post("/circle/centeronline")
async def circle_center_online(p: coordinates, q: coordinates, line: linegeneral):
    """
    Finds the equation of the circle which passes through points p and q
    and has its center on the line Ax + By + C = 0.

    Args:
        p (coordinates): First point on the circle (x1, y1).
        q (coordinates): Second point on the circle (x2, y2).
        line (linegeneral): Line equation coefficients A, B, C for Ax + By + C = 0.
    Returns:
        dict: Containing the circle equation string.
    """
    try:
        # 1. Define Symbols
        x, y, g, f, c = symbols("x y g f c")

        # Use sympify to handle potential float inputs correctly in SymPy
        x1, y1 = sympify(p.x), sympify(p.y)
        x2, y2 = sympify(q.x), sympify(q.y)
        A, B, C = sympify(line.a), sympify(line.b), sympify(line.c)

        # 2. General Circle Equation (expression part)
        circle_expr = x**2 + y**2 + 2 * g * x + 2 * f * y + c

        # 3. Equation from Point p
        # x1^2 + y1^2 + 2*g*x1 + 2*f*y1 + c = 0
        eq1 = circle_expr.subs({x: x1, y: y1})
        # eq1 = Eq(circle_expr.subs({x: x1, y: y1}), 0) # Alternative: use Eq()

        # 4. Equation from Point q
        # x2^2 + y2^2 + 2*g*x2 + 2*f*y2 + c = 0
        eq2 = circle_expr.subs({x: x2, y: y2})
        # eq2 = Eq(circle_expr.subs({x: x2, y: y2}), 0) # Alternative: use Eq()

        # 5. Equation from Center (-g, -f) on Line Ax + By + C = 0
        # A*(-g) + B*(-f) + C = 0
        line_eq_expr = A * x + B * y + C
        eq3 = line_eq_expr.subs({x: -g, y: -f})
        # eq3 = Eq(line_eq_expr.subs({x: -g, y: -f}), 0) # Alternative: use Eq()

        # 6. Solve System for g, f, c
        # We have three linear equations (eq1=0, eq2=0, eq3=0) for g, f, c
        solution = solve([eq1, eq2, eq3], (g, f, c))

        if not solution:
            # Check if points are identical or other degenerate cases
            if x1 == x2 and y1 == y2:
                raise ValueError("Points p and q cannot be the same.")
            # Add more checks if needed (e.g., line perpendicular bisector issues)
            raise ValueError(
                "Could not find a unique solution. Check input points and line."
            )

        # Ensure solution is a dict if solve returns one solution
        if isinstance(solution, list) and len(solution) == 1:
            solution = solution[
                0
            ]  # Should not happen for linear system, but safe check
        elif not isinstance(solution, dict):
            raise ValueError(f"Unexpected solution format from SymPy: {solution}")

        g_sol, f_sol, c_sol = solution[g], solution[f], solution[c]

        # 7. Construct Final Equation
        final_circle_expr = circle_expr.subs({g: g_sol, f: f_sol, c: c_sol})

        # Format the final equation string
        equation_str = format_sympy_equation(
            final_circle_expr
        )  # Defaults to 'expr = 0'
        radius = sqrt(
            (g_sol**2 + f_sol**2 - c_sol)
        )  # Calculate radius from center coordinates and constant term

        return {
            "circle_equation": equation_str,
            "center_g": -float(g_sol),  # Return solved values if needed
            "center_f": -float(f_sol),
            "constant_c": float(radius),
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error in circle_center_online: {e}")  # Log the error
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred: {type(e).__name__} - {e}",
        )


def parametric_circle_eqn(h, k, r):
    """
    Generate the parametric equations for a circle.

    Args:
        h (float): X-coordinate of the center.
        k (float): Y-coordinate of the center.
        r (float): Radius of the circle.

    Returns:
        tuple: Parametric equations for x and y.
    """
    t = symbols("t")
    x = h + r * sympy.cos(t)
    y = k + r * sympy.sin(t)
    return x, y


def obtain_cetre_radius(circle_eqn: str):
    x, y = symbols("x y")
    exp = sympify(circle_eqn)
    expr_expanded = expand(exp)
    x_coeff = expr_expanded.coeff(x, 0)
    y_coeff = expr_expanded.coeff(y, 0)
    constant = expr_expanded.coeff(1, 0)
    # Calculate center coordinates
    h = -x_coeff / 2
    k = -y_coeff / 2
    # Calculate radius
    r = sqrt(h**2 + k**2 - constant)
    return h, k, r


class EqnInput(BaseModel):
    eqn: str = Field(
        ..., description="General form equation, e.g., 'x^2 + y^2 - 4*x + 6*y - 12 = 0'"
    )


@router.post("/circle/parametric")
async def circle_parametric(data: EqnInput):
    """
    Calculate the parametric equations of a circle given its general form equation.

    Args:
        data (parametricInput): Object containing the equation string.

    Returns:
        dict: Parametric equations for x and y.
    """
    try:
        # Extract the equation from the input data
        eqn = data.eqn

        # Obtain center and radius from the general form equation
        h, k, r = obtain_cetre_radius(eqn)

        # Generate parametric equations
        x_param, y_param = parametric_circle_eqn(h, k, r)

        # Format the parametric equations as strings
        x_param_str = format_sympy_equation(x_param, equals_zero=False)
        y_param_str = format_sympy_equation(y_param, equals_zero=False)

        return {
            "parametric_x": x_param_str,
            "parametric_y": y_param_str,
            "center_h": float(h),
            "center_k": float(k),
            "radius": float(r),
        }
    except Exception as e:
        print(f"Error calculating parametric equations: {e}")


def diameter_circle_eqn(p: coordinates, q: coordinates):
    """
    calculates the eqn of circle with diameter endpoints A(x1,y1) and B(x2,y2)
    """
    x, y, x1, y1, x2, y2 = symbols("x y x1 y1 x2 y2")
    x1, y1 = sympify(p.x), sympify(p.y)
    x2, y2 = sympify(q.x), sympify(q.y)
    formula = (x - x1) * (x - x2) + (y - y1) * (y - y2)
    eqn = formula.subs({x1: x1, y1: y1, x2: x2, y2: y2})
    # Expand the equation to standard form
    expanded_eqn = expand(eqn)
    # Convert to string and format
    circle_eqn = str(expanded_eqn).replace("**", "^")
    return circle_eqn


@router.post("/circle/diameter")
async def circle_diameter(p: coordinates, q: coordinates):
    """
    Calculate the equation of a circle given two points on its diameter.

    Args:
        p (coordinates): First point on the diameter (x1, y1).
        q (coordinates): Second point on the diameter (x2, y2).

    Returns:
        dict: Circle equation string.
    """
    try:
        # Calculate the circle equation using the diameter endpoints
        circle_eqn = diameter_circle_eqn(p, q)
        return {"circle_equation": circle_eqn}
    except Exception as e:
        print(f"Error calculating circle equation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred: {type(e).__name__} - {e}",
        )


# @router.post("/circle/pointposition")
# async def position_of_a_point(eqn: EqnInput, point: coordinates):
#     """
#     Determines the position of a point relative to a circle given its equation.

#     Args:
#         eqn (str): Circle equation in general form (e.g., 'x^2 + y^2 - 4*x + 6*y - 12 = 0').

#     Returns:
#         str: Position of the point relative to the circle ('inside', 'on', or 'outside').
#     """
#     try:
#         x, y = symbols("x y")
#         # eqn = eqn.replace(" ", "").replace("^", "**")
#         # Extract the coefficients from the equation
#         h, k, r = obtain_cetre_radius(eqn)
#         center = sympy.Point(h, r)
#         point = sympy.Point(point.x, point.y)
#         distance_formula = center.distance(point)
#         if distance_formula < r:
#             return "inside"
#         elif distance_formula == r:
#             return "on"
#         else:
#             return "outside"

#     except Exception as e:
#         print(f"Error determining point position: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"An internal server error occurred: {type(e).__name__} - {e}",
#         )
