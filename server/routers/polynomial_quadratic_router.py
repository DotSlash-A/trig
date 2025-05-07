# routers/polynomial_quadratic_router.py
from fastapi import APIRouter, Body, HTTPException, Query
from models import polynomial_quadratic_models as models
from services import polynomial_services as ps
from typing import List, Union
import cmath  # For parsing complex number strings
from sympy import symbols  # For symbolic variable creation

router = APIRouter(
    prefix="/polynomials-quadratic", tags=["Polynomials and Quadratic Equations"]
)


def _get_coeffs_from_poly_input(poly_input: models.PolynomialInput) -> List[float]:
    if poly_input.coeffs:
        return poly_input.coeffs
    elif poly_input.expression:
        try:
            return ps.parse_polynomial_string(
                poly_input.expression, poly_input.variable
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"Error parsing polynomial expression: {str(e)}"
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="Either 'coeffs' or 'expression' must be provided for the polynomial.",
        )


def _parse_value(val_str: Union[float, str]) -> Union[float, complex]:
    if isinstance(val_str, (int, float)):
        return float(val_str)
    try:
        return complex(val_str)  # Handles "2+3j", "5", etc.
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Invalid numeric value for evaluation: {val_str}"
        )


@router.post("/evaluate", response_model=models.PolynomialEvaluationResponse)
async def evaluate_poly(req: models.PolynomialEvaluationRequest = Body(...)):
    """Evaluates a polynomial P(x) at a given value of x."""
    coeffs = _get_coeffs_from_poly_input(req)
    poly_str = ps.polynomial_to_string(coeffs, req.variable)

    x_val_parsed = _parse_value(req.x_value)

    result = ps.evaluate_polynomial(coeffs, x_val_parsed)

    return models.PolynomialEvaluationResponse(
        polynomial_string=poly_str,
        coeffs=coeffs,
        x_value=str(x_val_parsed),
        result=str(result),
    )


@router.post("/divide", response_model=models.PolynomialDivisionResponse)
async def divide_poly(req: models.PolynomialDivisionRequest = Body(...)):
    """Performs polynomial division: dividend / divisor."""
    dividend_coeffs = _get_coeffs_from_poly_input(req.dividend)
    divisor_coeffs = _get_coeffs_from_poly_input(req.divisor)

    if not divisor_coeffs or all(abs(c) < 1e-9 for c in divisor_coeffs):
        raise HTTPException(
            status_code=400, detail="Divisor polynomial cannot be zero."
        )

    dividend_str = ps.polynomial_to_string(dividend_coeffs, req.dividend.variable)
    divisor_str = ps.polynomial_to_string(divisor_coeffs, req.divisor.variable)

    try:
        quotient_coeffs, remainder_coeffs = ps.polynomial_division(
            dividend_coeffs, divisor_coeffs
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    quotient_str = ps.polynomial_to_string(quotient_coeffs, req.dividend.variable)
    remainder_str = ps.polynomial_to_string(remainder_coeffs, req.dividend.variable)

    equation = (
        f"({dividend_str}) = ({divisor_str}) * ({quotient_str}) + ({remainder_str})"
    )

    return models.PolynomialDivisionResponse(
        dividend_str=dividend_str,
        divisor_str=divisor_str,
        quotient_coeffs=quotient_coeffs,
        quotient_str=quotient_str,
        remainder_coeffs=remainder_coeffs,
        remainder_str=remainder_str,
        equation=equation,
    )


@router.post("/synthetic-divide", response_model=models.SyntheticDivisionResponse)
async def synthetic_divide_poly(req: models.SyntheticDivisionRequest = Body(...)):
    """Performs synthetic division of P(x) by (x - a_value)."""
    coeffs = _get_coeffs_from_poly_input(req)
    poly_str = ps.polynomial_to_string(coeffs, req.variable)
    a_val_parsed = _parse_value(req.a_value)

    try:
        quotient_coeffs, remainder = ps.synthetic_division(coeffs, a_val_parsed)
    except ValueError as e:  # Should not happen if coeffs are valid
        raise HTTPException(status_code=400, detail=str(e))

    quotient_str = ps.polynomial_to_string(quotient_coeffs, req.variable)
    divisor_form = f"({req.variable} - ({str(a_val_parsed)}))"

    return models.SyntheticDivisionResponse(
        polynomial_str=poly_str,
        divisor_form=divisor_form,
        quotient_coeffs=quotient_coeffs,
        quotient_str=quotient_str,
        remainder=str(remainder),
    )


@router.post("/quadratic/solve", response_model=models.QuadraticSolutionResponse)
async def solve_quad_eq(coeffs_in: models.QuadraticEquationCoeffs = Body(...)):
    """Solves a quadratic equation ax^2 + bx + c = 0."""
    a, b, c = coeffs_in.a, coeffs_in.b, coeffs_in.c
    poly_coeffs_for_str = [a, b, c]
    eq_str = f"({ps.polynomial_to_string(poly_coeffs_for_str)}) = 0"

    solution = ps.solve_quadratic_equation(a, b, c)

    return models.QuadraticSolutionResponse(
        equation_string=eq_str,
        coefficients=coeffs_in,
        **solution,  # Unpack discriminant, nature_of_roots, roots, formula_used
    )


@router.post(
    "/roots-coefficients-relation", response_model=models.RootsCoeffsRelationResponse
)
async def get_roots_coeffs_relation(poly_input: models.PolynomialInput = Body(...)):
    """
    Shows the relationship between roots and coefficients for quadratic or cubic polynomials.
    (e.g., sum of roots, product of roots).
    """
    coeffs = _get_coeffs_from_poly_input(poly_input)
    poly_str = ps.polynomial_to_string(coeffs, poly_input.variable)
    degree = len(coeffs) - 1
    relations = None
    note = None

    if degree == 2:  # Quadratic
        relations = ps.relation_roots_coeffs_quadratic(coeffs)
        note = "For ax^2+bx+c=0: Sum (α+β) = -b/a, Product (αβ) = c/a."
    elif degree == 3:  # Cubic
        relations = ps.relation_roots_coeffs_cubic(coeffs)
        note = "For ax^3+bx^2+cx+d=0: Sum (α+β+γ) = -b/a, Sum of products pairwise (αβ+βγ+γα) = c/a, Product (αβγ) = -d/a."
    else:
        raise HTTPException(
            status_code=400,
            detail="This endpoint currently supports quadratic and cubic polynomials for root-coefficient relations.",
        )

    if (
        relations is None
    ):  # Should be caught by degree check or if service returns None for non-poly
        raise HTTPException(
            status_code=400,
            detail="Could not determine relations. Ensure valid polynomial.",
        )

    return models.RootsCoeffsRelationResponse(
        polynomial_string=poly_str,
        coeffs=coeffs,
        degree=degree,
        relations=relations,
        verification_note=note,
    )


@router.post("/find-roots", response_model=models.FindRootsResponse)
async def find_poly_roots(poly_input: models.PolynomialInput = Body(...)):
    """
    Attempts to find roots of a polynomial.
    - Finds rational roots using the Rational Root Theorem.
    - Attempts to find all roots numerically (real and complex) using sympy.nroots.
    """
    coeffs = _get_coeffs_from_poly_input(poly_input)
    poly_str = ps.polynomial_to_string(coeffs, poly_input.variable)

    # Attempt Rational Roots
    rational_roots = []
    rational_roots_note = "Rational Root Theorem applied."
    try:
        rational_roots = ps.find_rational_roots(coeffs)
        if not any(
            abs(c - round(c)) > 1e-6 for c in coeffs
        ):  # Check if coeffs are integers for strict RRT
            rational_roots_note += " (Strictly applicable for integer coefficients)."
        else:
            rational_roots_note += " (Applied to rounded integer coefficients; exercise caution for non-integer inputs)."

    except Exception as e:
        rational_roots_note = f"Error finding rational roots: {e}"

    # Attempt All Numerical Roots (using sympy.nroots via service)
    numerical_roots = []
    numerical_roots_note = (
        "Numerical root finding attempted (may include complex roots)."
    )
    try:
        numerical_roots = ps.find_all_roots_numeric(coeffs)
    except Exception as e:
        numerical_roots_note = f"Error finding numerical roots: {e}"

    return models.FindRootsResponse(
        polynomial_string=poly_str,
        coeffs=coeffs,
        rational_roots_found=rational_roots,
        all_numerical_roots=numerical_roots,
        method_notes=f"Rational Roots: {rational_roots_note} Numerical Roots: {numerical_roots_note}",
    )


@router.post(
    "/form-polynomial-from-roots", response_model=models.FormPolynomialFromRootsResponse
)
async def form_poly_from_roots(req: models.FormPolynomialFromRootsRequest = Body(...)):
    """
    Forms a polynomial P(x) = k(x-r1)(x-r2)...(x-rn) given its roots and an optional leading coefficient k.
    Handles real and complex roots.
    """
    parsed_roots = []
    for r_str in req.roots:
        parsed_roots.append(_parse_value(r_str))  # Handles complex numbers like "1+2j"

    # Start with P(x) = k
    # For each root r, multiply by (x - r)
    # P(x) = k * (x - r1) * (x - r2) * ...
    # Current polynomial coeffs, start with [k] (representing constant k)
    current_coeffs_sympy = [req.leading_coefficient]

    x_sym = symbols(req.variable)
    current_poly_sympy = ps.Poly(
        current_coeffs_sympy,
        x_sym,
        domain="RR" if all(isinstance(r, float) for r in parsed_roots) else "CC",
    )

    for root in parsed_roots:
        # (x - root) has coeffs [1, -root]
        # Multiply current_poly by (x - root)
        # Using sympy.expand for robust polynomial multiplication here as manual can be complex with complex numbers
        factor_poly = ps.Poly(
            [1, -root], x_sym, domain="CC"
        )  # Ensure complex domain for multiplication
        current_poly_sympy = ps.expand(current_poly_sympy * factor_poly)
        current_poly_sympy = ps.Poly(
            current_poly_sympy, x_sym, domain="CC"
        )  # Re-cast to Poly

    final_coeffs_sympy = current_poly_sympy.all_coeffs()

    # Convert Sympy numbers (potentially complex) to Python floats or complex string representations
    final_coeffs_py: List[Union[float, str]] = []
    for coeff_val in final_coeffs_sympy:
        if coeff_val.is_real:
            # Check if it's an integer
            if abs(coeff_val - round(coeff_val)) < 1e-9:
                final_coeffs_py.append(float(round(coeff_val)))
            else:
                final_coeffs_py.append(float(coeff_val))
        else:  # Complex coefficient
            final_coeffs_py.append(
                str(coeff_val)
            )  # Store complex as string "re+im*I" (sympy format)

    # Re-construct string from these possibly complex coeffs
    # The ps.polynomial_to_string might need adjustment for complex coefficients
    # For now, let sympy generate the string for the final polynomial
    final_poly_str = str(current_poly_sympy.as_expr())

    return models.FormPolynomialFromRootsResponse(
        roots_provided=[str(r) for r in parsed_roots],
        polynomial_coeffs=final_coeffs_py,
        polynomial_string=final_poly_str,
        leading_coefficient_used=req.leading_coefficient,
    )
