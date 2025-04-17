from fastapi import FastAPI, Query, APIRouter
from models.shapes import SlopeCordiantes, SlopeInput, FindXRequest
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





@router.post("/find_x")
async def find_x(request: FindXRequest):
    """
    solves for a missign variable int he slope formula m = (y2 - y1) / (x2 - x1).
    Args:
        find_var (str): The variable to find ('x1', 'y1', 'x2', 'y2', 'm').
        known_values: Dictionary containing the known variables and their values.

    Returns:
        float or str: The calculated value of the missing variable, or an error message string.
    """
    known_values = request.known_values
    find_var = request.find_var.lower()
    x1 = known_values.get("x1")
    y1 = known_values.get("y1")
    x2 = known_values.get("x2")
    y2 = known_values.get("y2")
    m = known_values.get("m")
    angle = math.radians(m)
    m = math.tan(angle)
    try:
        if m is None:
            return "Error: Slope 'm' is required to find 'x2'."  # Should not happen if validation works
        if m == 0:
            return "Error: Cannot solve for x2 when slope (m) is 0 (horizontal line)."

        if find_var == "x2":
            if y2 is None or y1 is None or x1 is None:
                return "Error: Missing required values (y2, y1, x1) to find 'x2'."  # Should not happen
            result = x1 + (y2 - y1) / m
            return result
        elif find_var == "x1":
            if y2 is None or y1 is None or x2 is None:
                return "Error: Missing required values (y2, y1, x2) to find 'x1'."
            result = x2 - (y2 - y1) / m
            return result
        elif find_var == "y2":
            if x2 is None or y1 is None or x1 is None:
                return "Error: Missing required values (x2, y1, x1) to find 'y2'."
            result = y1 + m * (x2 - x1)
            return result
        elif find_var == "y1":
            if x2 is None or y2 is None or x1 is None:
                return "Error: Missing required values (x2, y2, x1) to find 'y1'."
            result = y2 - m * (x2 - x1)
            return result
        elif find_var == "m":
            if x2 is None or y2 is None or x1 is None:
                return "Error: Missing required values (x2, y2, x1) to find 'm'."
            if x2 - x1 == 0:
                return "Error: Cannot calculate slope (m) when x1 and x2 are equal (vertical line)."
            result = (y2 - y1) / (x2 - x1)
            return result
        return {find_var: result}
    except Exception as e:
        return {"error": str(e)}



