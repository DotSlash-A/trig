from fastapi import FastAPI, Query, APIRouter, HTTPException
from models.shapes import SlopeCordiantes, SlopeInput, FindXRequest, SlopeIntercept
from sympy import symbols, Eq, solve, simplify, parse_expr
import math
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication_application,
)
from typing import Dict, Any
from pydantic import BaseModel, Field

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


@router.post("/slope_intercept")
async def slope_intercept(request: SlopeIntercept):
    """
    Calculate the slope and y-intercept of a line given two points.
    Args:
        request (SlopeIntercept): The request body containing the slope and y-intercept.

    Returns:
        dict: A dictionary containing the slope and y-intercept.
    """
    try:
        x1 = request.point1.x
        y1 = request.point1.y
        x2 = request.point2.x
        y2 = request.point2.y
        m = request.slope

        if m is None:
            if x1 is None or y1 is None or x2 is None or y2 is None:
                return "Error: Missing required values (x1, y1, x2, y2) to find 'm'."  # Should not happen
            m = (y2 - y1) / (x2 - x1)

        b = y1 - (m * x1)
        # y = mx + b => b = y - mx

        return {"slope": m, "y_intercept": b}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected server error occurred: {str(e)}"
        )
