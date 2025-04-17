import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

# --- Pydantic Models ---
class SlopeInput(BaseModel):
    x1: Optional[float] = None
    y1: Optional[float] = None
    x2: Optional[float] = None
    y2: Optional[float] = None
    m: Optional[float] = None

class SlopeSolution(BaseModel):
    variable_found: str
    value: float

# --- FastAPI App ---
app = FastAPI(
    title="Slope Formula Solver API",
    description="API to solve for any variable (x1, y1, x2, y2, m) in the slope formula m = (y2 - y1) / (x2 - x1), given the other four.",
    version="1.0.0",
)

# --- Calculation Logic (kept mostly the same) ---
def solve_slope_formula(find_var: str, known_values: Dict[str, float]) -> float | str:
    """
    Solves for a missing variable in the slope formula m = (y2 - y1) / (x2 - x1).
    Internal function used by the API endpoint.

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

    # Input validation (presence of keys) is handled by the endpoint logic before calling this.
    # Type validation (float) is handled by Pydantic.

    try:
        if find_var == "x2":
            if m is None: return "Error: Slope 'm' is required to find 'x2'." # Should not happen if validation works
            if m == 0:
                return "Error: Cannot solve for x2 when slope (m) is 0 (horizontal line)."
            if y2 is None or y1 is None or x1 is None: return "Error: Missing required values (y2, y1, x1) to find 'x2'." # Should not happen
            return x1 + (y2 - y1) / m
        elif find_var == "x1":
            if m is None: return "Error: Slope 'm' is required to find 'x1'." # Should not happen
            if m == 0:
                return "Error: Cannot solve for x1 when slope (m) is 0 (horizontal line)."
            if y2 is None or y1 is None or x2 is None: return "Error: Missing required values (y2, y1, x2) to find 'x1'." # Should not happen
            return x2 - (y2 - y1) / m
        elif find_var == "y2":
            if y1 is None or m is None or x2 is None or x1 is None: return "Error: Missing required values (y1, m, x2, x1) to find 'y2'." # Should not happen
            return y1 + m * (x2 - x1)
        elif find_var == "y1":
            if y2 is None or m is None or x2 is None or x1 is None: return "Error: Missing required values (y2, m, x2, x1) to find 'y1'." # Should not happen
            return y2 - m * (x2 - x1)
        elif find_var == "m":
            if y2 is None or y1 is None or x2 is None or x1 is None: return "Error: Missing required values (y2, y1, x2, x1) to find 'm'." # Should not happen
            if x2 - x1 == 0:
                return "Error: Cannot calculate slope (m) when x1 and x2 are equal (vertical line)."
            return (y2 - y1) / (x2 - x1)
        else:
             # This case should technically not be reachable due to endpoint validation
             return f"Error: Invalid variable '{find_var}' requested."
    except ZeroDivisionError:
         # Should only happen for x1/x2 calculation if m is zero, which is checked above.
    for var in variables_needed:
        while True:
            try:
                value = float(input(f"Enter the value for {var}: "))
                known_values[var] = value
                break
            except ValueError:
                print("Invalid input. Please enter a number.")

    result = solve_slope_formula(find_variable, **known_values)

    if isinstance(result, str) and result.startswith("Error"):
        print(result)
    else:
        print(f"The calculated value for {find_variable} is: {result}")
