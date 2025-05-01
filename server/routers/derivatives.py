# from fastapi import FastAPI, Query, APIRouter

# from sympy import symbols, Eq, solve, simplify, parse_expr
# from sympy.parsing.sympy_parser import (
#     standard_transformations,
#     implicit_multiplication_application,
# )
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any, Tuple, Optional
import sympy
import traceback  # For more detailed error logging if needed


# router = APIRouter()


# --- Pydantic Models ---


class DiffInput(BaseModel):
    expression: str = Field(
        ...,
        description="The mathematical expression to differentiate (e.g., 'x**2 * sin(x)'). Use Python/SymPy syntax.",
    )
    variable: str = Field(
        ..., description="The variable to differentiate with respect to (e.g., 'x')."
    )


class DiffResponseSimple(BaseModel):
    original_expression: str = Field(..., description="The input expression.")
    variable: str = Field(..., description="The variable of differentiation.")
    derivative: str = Field(..., description="The calculated derivative (simplified).")


class StepDetail(BaseModel):
    step_description: str
    expression_latex: str  # Using LaTeX for better potential rendering
    expression_pretty: str  # Using pretty print for console/text readability


class DiffResponseSteps(BaseModel):
    original_expression: str = Field(..., description="The input expression.")
    variable: str = Field(..., description="The variable of differentiation.")
    steps: List[StepDetail] = Field(
        ..., description="A list of steps showing the differentiation process."
    )
    final_derivative: str = Field(
        ..., description="The final calculated derivative (simplified)."
    )  # Redundant but convenient


# --- Core Logic Functions ---


def parse_and_validate(
    expression_str: str, variable_str: str
) -> Tuple[sympy.Expr, sympy.Symbol]:
    """Parses the expression and variable, validates."""
    try:
        # Create the symbol for the differentiation variable
        variable_sym = sympy.symbols(variable_str)
        if not isinstance(variable_sym, sympy.Symbol):
            # Handle cases like "sin(x)" being passed as variable
            raise ValueError(f"'{variable_str}' is not a valid variable name.")

        # Define allowed functions/symbols for parsing (basic safety)
        # Allow standard functions, constants like pi, e
        allowed_globals = {
            "Symbol": sympy.Symbol,  # Needed potentially if expression uses Symbol()
            "Eq": sympy.Eq,  # Allow equations for potential future implicit diff
            # Add standard functions explicitly for clarity & control
            "sin": sympy.sin,
            "cos": sympy.cos,
            "tan": sympy.tan,
            "asin": sympy.asin,
            "acos": sympy.acos,
            "atan": sympy.atan,
            "exp": sympy.exp,
            "log": sympy.log,
            "ln": sympy.ln,  # ln is often alias for log
            "sqrt": sympy.sqrt,
            "pi": sympy.pi,
            "E": sympy.E,
            "I": sympy.I,  # Constants (I is imaginary unit)
            # Allow basic types
            "Integer": sympy.Integer,
            "Float": sympy.Float,
            "Rational": sympy.Rational,
            "True": True,
            "False": False,  # Booleans
        }
        # Use sympy's parse_expr for safer evaluation than eval()
        # It needs the variable symbol in the local dictionary to recognize it
        parsed_expr = sympy.parse_expr(
            expression_str,
            local_dict={variable_str: variable_sym},
            global_dict=allowed_globals,
        )
        return parsed_expr, variable_sym
    except (SyntaxError, TypeError, ValueError, sympy.SympifyError) as e:
        raise ValueError(f"Error parsing expression or variable: {e}")
    except Exception as e:
        # Catch other potential parsing issues
        raise ValueError(f"Unexpected parsing error: {e}")


def format_expression(expr: sympy.Expr) -> Tuple[str, str]:
    """Formats sympy expression into LaTeX and pretty string."""
    try:
        # Use sympy's built-in printers
        latex_str = sympy.latex(expr)
        pretty_str = sympy.pretty(
            expr, use_unicode=True
        )  # Use unicode for better console display
        return latex_str, pretty_str
    except Exception as e:
        print(f"Warning: Error formatting expression: {e}")
        # Fallback to basic string representation
        str_repr = str(expr)
        return str_repr, str_repr


def generate_differentiation_steps(
    expr: sympy.Expr, variable: sympy.Symbol
) -> Tuple[List[StepDetail], sympy.Expr]:
    """Generates simplified differentiation steps and the final result."""
    steps = []

    # --- Step 1: Original Expression ---
    latex_orig, pretty_orig = format_expression(expr)
    steps.append(
        StepDetail(
            step_description=f"Differentiate the expression with respect to {variable}:",
            expression_latex=f"\\frac{{d}}{{d{variable.name}}} \\left[ {latex_orig} \\right]",
            expression_pretty=f"d/d{variable.name}({pretty_orig})",
        )
    )

    # --- Step 2: Apply Differentiation Rules (Show Raw Result) ---
    try:
        # Calculate the derivative
        raw_derivative = sympy.diff(expr, variable)
        latex_raw, pretty_raw = format_expression(raw_derivative)

        # Attempt to identify the main rule applied at the top level (heuristic)
        rule_applied = "Applying differentiation rules"  # Default
        if isinstance(expr, sympy.Add):
            rule_applied = "Applying the Sum Rule"
        elif isinstance(expr, sympy.Mul):
            rule_applied = "Applying the Product Rule"
        elif isinstance(expr, sympy.Pow):
            # Check if base or exponent depends on variable
            base, expo = expr.args
            if variable in expo.free_symbols:
                # Potentially needs logarithmic differentiation or generalized power rule
                # Sympy handles x**x correctly with diff, but identifying the "method" is hard.
                rule_applied = "Applying Generalized Power Rule / Chain Rule (possibly involving logarithms)"
            else:
                rule_applied = "Applying the Power Rule"
        elif isinstance(expr, sympy.Function):
            # Check if argument depends on variable -> Chain rule
            if variable in expr.args[0].free_symbols:
                rule_applied = f"Applying the Chain Rule to {expr.func.__name__}(...)"

        # More sophisticated rule detection could be added here (e.g., quotient rule by checking for division)

        steps.append(
            StepDetail(
                step_description=f"{rule_applied} yields:",
                expression_latex=latex_raw,
                expression_pretty=pretty_raw,
            )
        )

    except Exception as e:
        raise ValueError(f"Error during differentiation calculation: {e}")

    # --- Step 3: Simplify the Result ---
    try:
        simplified_derivative = sympy.simplify(raw_derivative)
        latex_final, pretty_final = format_expression(simplified_derivative)

        # Only add simplification step if it actually changed the expression
        # Comparing SymPy expressions directly can be tricky, compare formatted strings for simplicity
        if latex_raw != latex_final:
            steps.append(
                StepDetail(
                    step_description="Simplifying the result:",
                    expression_latex=latex_final,
                    expression_pretty=pretty_final,
                )
            )
        final_result_expr = simplified_derivative
    except Exception as e:
        print(f"Warning: Error during simplification: {e}")
        # If simplification fails, use the raw derivative as the final result
        steps.append(
            StepDetail(
                step_description="Simplification failed, using unsimplified result:",
                expression_latex=latex_raw,  # Use raw from above
                expression_pretty=pretty_raw,
            )
        )
        final_result_expr = raw_derivative

    return steps, final_result_expr


# --- FastAPI Setup ---
# app = FastAPI(title="Symbolic Differentiation API")
# Or use a router
router = APIRouter(prefix="/differentiate", tags=["Differentiation"])


# --- API Endpoints ---


@router.post("/simple", response_model=DiffResponseSimple)
async def differentiate_simple(input_data: DiffInput):
    """
    Calculates the simplified derivative of an expression with respect to a variable.
    """
    try:
        parsed_expr, variable_sym = parse_and_validate(
            input_data.expression, input_data.variable
        )

        # Calculate derivative and simplify
        derivative_expr = sympy.diff(parsed_expr, variable_sym)
        simplified_derivative = sympy.simplify(derivative_expr)

        # Format for output
        _, pretty_derivative = format_expression(simplified_derivative)

        return DiffResponseSimple(
            original_expression=input_data.expression,
            variable=input_data.variable,
            derivative=pretty_derivative,
        )
    except ValueError as e:
        # Catch parsing/validation errors
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch unexpected errors during calculation/simplification
        print(f"Error in /simple: {e}\n{traceback.format_exc()}")  # Log detailed error
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred during differentiation.",
        )


@router.post("/steps", response_model=DiffResponseSteps)
async def differentiate_steps(input_data: DiffInput):
    """
    Calculates the derivative and provides a simplified step-by-step explanation.
    Note: Step generation is heuristic and may not perfectly capture all methods.
    """
    try:
        parsed_expr, variable_sym = parse_and_validate(
            input_data.expression, input_data.variable
        )

        # Generate steps and final result
        steps_list, final_expr = generate_differentiation_steps(
            parsed_expr, variable_sym
        )

        # Format final result for the separate field in response
        _, pretty_final = format_expression(final_expr)

        return DiffResponseSteps(
            original_expression=input_data.expression,
            variable=input_data.variable,
            steps=steps_list,
            final_derivative=pretty_final,
        )
    except ValueError as e:
        # Catch parsing/validation/calculation errors
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch unexpected errors
        print(f"Error in /steps: {e}\n{traceback.format_exc()}")  # Log detailed error
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred during step generation.",
        )


# --- Add router to app if using a router ---
# Example:
# app = FastAPI(title="Symbolic Differentiation API")
# app.include_router(router)

# --- Optional root endpoint ---
# @app.get("/")
# async def root():
#     return {"message": "Symbolic Differentiation API. POST to /differentiate/simple or /differentiate/steps"}
