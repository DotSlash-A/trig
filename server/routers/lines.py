from fastapi import FastAPI, Query, APIRouter, HTTPException
from models.shapes import (
    SlopeCordiantes,
    SlopeInput,
    FindXRequest,
    SlopeIntercept,
    coordinates,
    LineInput,
)
from sympy import symbols, Eq, solve, simplify, parse_expr
import math
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication_application,
)
from typing import Dict, Optional
from pydantic import BaseModel, Field
from fractions import Fraction

router = APIRouter()


@router.post("/SlopeCordiantes")
async def slopecordinates(slopecordinates: SlopeCordiantes):
    try:
        x1 = slopecordinates.x1
        y1 = slopecordinates.y1
        x2 = slopecordinates.x2
        y2 = slopecordinates.y2
        m = (y2 - y1) / (x2 - x1)
        return {"slope": m}
    except Exception as e:
        return {"error": str(e)}


@router.get("/find_x")
async def find_x(find_var: str, known_values: Dict[str, float]) -> float | str:
    """
    solves for a missign variable int he slope formula m = (y2 - y1) / (x2 - x1).
    Args:
        find_var (str): The variable to find ('x1', 'y1', 'x2', 'y2', 'm').
        known_values: Dictionary containing the known variables and their values.

    Returns:
        float or str: The calculated value of the missing variable, or an error message string.
    """
    x1 = known_values.get("x1")
    y1 = known_values.get("y1")
    x2 = known_values.get("x2")
    y2 = known_values.get("y2")
    m = known_values.get("m")
    if m is None:
        raise HTTPException(status_code=400, detail="Error: Slope 'm' is required.")
    try:
        m_rad = math.radians(m)
        m_tan = math.tan(m_rad)
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Error: Invalid value for slope 'm': {m}"
        )

    try:
        if m_tan == 0 and find_var in ["x1", "x2"]:
            raise HTTPException(
                status_code=400,
                detail="Error: Cannot solve for x1 or x2 when slope (m) corresponds to a horizontal line.",
            )

        if find_var == "x2":
            if y2 is None or y1 is None or x1 is None:
                raise HTTPException(
                    status_code=400,
                    detail="Error: Missing required values (y2, y1, x1) to find 'x2'.",
                )
            if m_tan == 0:  # Should be caught above, but double-check
                raise HTTPException(
                    status_code=400,
                    detail="Error: Cannot solve for x2 with zero slope.",
                )
            result = x1 + (y2 - y1) / m_tan
        elif find_var == "x1":
            if y2 is None or y1 is None or x2 is None:
                raise HTTPException(
                    status_code=400,
                    detail="Error: Missing required values (y2, y1, x2) to find 'x1'.",
                )
            if m_tan == 0:  # Should be caught above, but double-check
                raise HTTPException(
                    status_code=400,
                    detail="Error: Cannot solve for x1 with zero slope.",
                )
            result = x2 - (y2 - y1) / m_tan
        elif find_var == "y2":
            if x2 is None or y1 is None or x1 is None:
                raise HTTPException(
                    status_code=400,
                    detail="Error: Missing required values (x2, y1, x1) to find 'y2'.",
                )
            result = y1 + m_tan * (x2 - x1)
        elif find_var == "y1":
            if x2 is None or y2 is None or x1 is None:
                raise HTTPException(
                    status_code=400,
                    detail="Error: Missing required values (x2, y2, x1) to find 'y1'.",
                )
            result = y2 - m_tan * (x2 - x1)
        elif find_var == "m":
            if x2 is None or y1 is None or x1 is None or y2 is None:  # Added y2 check
                raise HTTPException(
                    status_code=400,
                    detail="Error: Missing required values (x1, y1, x2, y2) to find 'm'.",
                )
            if x2 - x1 == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Error: Cannot calculate slope (m) when x1 and x2 are equal (vertical line).",
                )
            result = (y2 - y1) / (x2 - x1)
            # Convert slope back to degrees if needed, or return the tangent value?
            # For now, returning the tangent value calculated.
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Error: Invalid value for 'find_var': {find_var}. Must be one of 'x1', 'y1', 'x2', 'y2', 'm'.",
            )

        return {find_var: result}
    except ZeroDivisionError:
        # This might happen if m_tan is zero when calculating x1 or x2, although checked above.
        # Or if x2-x1 is zero when calculating m, also checked above.
        raise HTTPException(
            status_code=400,
            detail="Error: Calculation resulted in division by zero. Check input values (e.g., slope for horizontal line, or identical x values for vertical line).",
        )
    except Exception as e:
        # Catch any other unexpected errors
        # Consider logging the error 'e' here for debugging
        raise HTTPException(
            status_code=500, detail=f"An unexpected server error occurred: {str(e)}"
        )


def calculate_slope_intercept(
    point1: Optional[coordinates], point2: Optional[coordinates], slope: Optional[float]
) -> dict:
    """
    Reusable slope-intercept calculation logic
    """
    # Validate input parameters
    if slope is None:
        if not (point1 and point2):
            raise ValueError(
                "Both point1 and point2 are required when slope is not provided"
            )

        # Extract coordinates
        x1 = point1.x
        y1 = point1.y
        x2 = point2.x
        y2 = point2.y

        # Prevent division by zero
        if x2 - x1 == 0:
            raise ValueError("Cannot calculate slope for vertical line (x2 - x1 = 0)")

        m = (y2 - y1) / (x2 - x1)
    else:
        m = slope
        if not point1:
            raise ValueError("At least one point is required when slope is provided")

    # Calculate y-intercept
    b = point1.y - (m * point1.x)

    return {
        "slope": m,
        "y_intercept": b,
        "equation": f"y = {m}x + {b}" if m != 0 else f"y = {b}",
    }


@router.post("/slope_intercept")
async def slope_intercept(request: SlopeIntercept):
    try:
        return calculate_slope_intercept(
            point1=request.point1, point2=request.point2, slope=request.slope
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except AttributeError:
        raise HTTPException(422, "Missing required coordinates")
    except Exception as e:
        raise HTTPException(500, f"Unexpected error: {str(e)}")


@router.post("/perpendicular_slope_intercept")
async def perpendicular_slope_intercept(y: float, request: SlopeIntercept):
    """
    Find the eqn of a perpendicular line joining two points and given y-intercept.
    """
    try:
        # Calculate the slope and y-intercept of the original line
        original_line = calculate_slope_intercept(
            point1=request.point1, point2=request.point2, slope=request.slope
        )

        # The slope of the parallel line is the same as the original line
        original_slope = original_line["slope"]
        original_y_intercept = original_line["y_intercept"]
        perp_slope = (-1) / (original_slope) if original_slope != 0 else float("inf")
        perp_y_intercept = y
        eqn = f"y = {perp_slope}x + {perp_y_intercept}"

        return {"slope": perp_slope, "y_intercept": perp_y_intercept, "equation": eqn}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except AttributeError:
        raise HTTPException(422, "Missing required coordinates")
    except Exception as e:
        raise HTTPException(500, f"Unexpected error: {str(e)}")


def calculate_point_slope(point: coordinates, slope: float) -> coordinates:
    """
    reusabale point-slope calculation logic
    """
    if slope is None:
        raise ValueError("Slope is required")
    eqn = f"y - {point.y} = {slope}(x - {point.x})"
    return eqn


def calculate_two_point_form(point1: coordinates, point2: coordinates) -> str:
    """
    reusable two-point form calculation logic
    """
    if point1.x == point2.x:
        raise ValueError("Cannot calculate slope for vertical line (x2 - x1 = 0)")
    slope = (point2.y - point1.y) / (point2.x - point1.x)
    rounded_slope = round(slope, 2)  # round to 2 decimal places
    eqn = f"y - {point1.y} = {rounded_slope}(x - {point1.x})"
    return eqn


@router.post("/two_point_form")
async def two_point_form(request: SlopeIntercept):
    try:
        return calculate_two_point_form(request.point1, request.point2)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except AttributeError:
        raise HTTPException(422, "Missing required coordinates")
    except Exception as e:
        raise HTTPException(500, f"Unexpected error: {str(e)}")


def calculate_intercept(
    x_intercept: float, y_intercept: float, point: coordinates
) -> str:
    """
    reusable intercept calculation logic
    """
    if x_intercept is None or y_intercept is None:
        raise ValueError("Both x_intercept and y_intercept are required")

    const = (point.x / x_intercept) + (point.y / y_intercept)
    const = Fraction(const).limit_denominator()
    x_int = Fraction(x_intercept).limit_denominator()
    y_int = Fraction(y_intercept).limit_denominator()
    eqn = f"x/{x_int} + y/{y_int} = {const}"
    return eqn


@router.post("/intercept")
async def intercept(request: LineInput):
    try:
        return calculate_intercept(
            request.x_intercept, request.y_intercept, request.point
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except AttributeError:
        raise HTTPException(422, "Missing required coordinates")
    except Exception as e:
        raise HTTPException(500, f"Unexpected error: {str(e)}")


def calculate_symmetric_form(
    theta: float, point1: coordinates, point2: coordinates
) -> str:
    """
    reusable symmetric form calculation logic
    """
    if point1 is None:
        raise ValueError("Point is required")
    x = point1.x
    y = point1.y
    x1 = point2.x
    y1 = point2.y

    r = round(
        (x - x1) / math.cos(math.radians(theta))
        or (y - y1) / math.sin(math.radians(theta))
    )
    r = Fraction(r).limit_denominator()

    cos_theta = Fraction(math.cos(math.radians(theta))).limit_denominator()
    sin_theta = Fraction(math.sin(math.radians(theta))).limit_denominator()

    eqn = f"({x-x1})/cos({theta}) = ({y-y1})/sin({theta}) = {r}"
    return eqn


@router.post("/symmetry_form")
async def symmetry_form(theta: float, request: SlopeIntercept):
    try:
        return calculate_symmetric_form(theta, request.point1, request.point2)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except AttributeError:
        raise HTTPException(422, "Missing required coordinates")
    except Exception as e:
        raise HTTPException(500, f"Unexpected error: {str(e)}")


def calculate_normal_form(
    alpha: float, request: LineInput = None, p: float = None
) -> str:
    """
    reusable normal form calculation logic
    """
    if p is None:
        p = (request.point.x * math.cos(math.radians(alpha))) + (
            request.point.y * math.sin(math.radians(alpha))
        )
    p = Fraction(p).limit_denominator()
    eqn = f"x * cos({alpha}) + y * sin({alpha}) = {p}"
    return eqn


@router.post("/normal_form")
async def normal_form(request: LineInput, alpha: float, p: Optional[float] = None):
    try:
        return calculate_normal_form(alpha, request, p)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except AttributeError:
        raise HTTPException(422, "Missing required coordinates")
    except Exception as e:
        raise HTTPException(500, f"Unexpected error: {str(e)}")


def calculate_transformation(