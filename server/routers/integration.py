from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel, Field
from typing import List, Union, Dict, Any, Tuple, Optional # Added Optional
import sympy
import traceback

# --- Pydantic Models ---

class IntegrationInput(BaseModel):
    expression: str = Field(..., description="The mathematical expression to integrate (e.g., 'x**2 * sin(x)'). Use Python/SymPy syntax.")
    variable: str = Field(..., description="The variable to integrate with respect to (e.g., 'x').")

class IntegrationResponseSimple(BaseModel):
    original_expression: str = Field(..., description="The input expression.")
    variable: str = Field(..., description="The variable of integration.")
    integral_result: str = Field(..., description="The calculated indefinite integral (simplified, includes '+ C').")
    computation_notes: Optional[str] = Field(None, description="Notes about the computation (e.g., if integration failed).")

class StepDetail(BaseModel): # Reusing from differentiation example
    step_description: str
    expression_latex: str # Using LaTeX for better potential rendering
    expression_pretty: str # Using pretty print for console/text readability

class IntegrationResponseSteps(BaseModel):
    original_expression: str = Field(..., description="The input expression.")
    variable: str = Field(..., description="The variable of integration.")
    steps: List[StepDetail] = Field(..., description="A list of steps showing the integration attempt.")
    final_integral: str = Field(..., description="The final calculated integral (simplified, includes '+ C').")
    computation_notes: Optional[str] = Field(None, description="Notes about the computation (e.g., if integration failed or method guess).")


# --- Core Logic Functions ---

# Reusing parse_and_validate and format_expression from differentiation
def parse_and_validate(expression_str: str, variable_str: str) -> Tuple[sympy.Expr, sympy.Symbol]:
    """Parses the expression and variable, validates."""
    try:
        variable_sym = sympy.symbols(variable_str)
        if not isinstance(variable_sym, sympy.Symbol):
             raise ValueError(f"'{variable_str}' is not a valid variable name.")
        allowed_globals = { # Basic safe functions
            "Symbol": sympy.Symbol, "Eq": sympy.Eq,
            "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
            "asin": sympy.asin, "acos": sympy.acos, "atan": sympy.atan,
            "exp": sympy.exp, "log": sympy.log, "ln": sympy.ln,
            "sqrt": sympy.sqrt,
            "pi": sympy.pi, "E": sympy.E, "I": sympy.I,
            "Integer": sympy.Integer, "Float": sympy.Float, "Rational": sympy.Rational,
            "True": True, "False": False,
        }
        parsed_expr = sympy.parse_expr(
            expression_str,
            local_dict={variable_str: variable_sym},
            global_dict=allowed_globals
        )
        return parsed_expr, variable_sym
    except (SyntaxError, TypeError, ValueError, sympy.SympifyError) as e:
        raise ValueError(f"Error parsing expression or variable: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected parsing error: {e}")

def format_expression(expr: sympy.Expr) -> Tuple[str, str]:
    """Formats sympy expression into LaTeX and pretty string."""
    try:
        latex_str = sympy.latex(expr)
        pretty_str = sympy.pretty(expr, use_unicode=True)
        return latex_str, pretty_str
    except Exception as e:
        print(f"Warning: Error formatting expression: {e}")
        str_repr = str(expr)
        return str_repr, str_repr

# --- New Core Logic for Integration Steps ---

def guess_integration_method(expr: sympy.Expr, variable: sympy.Symbol) -> str:
    """Provides a heuristic guess about applicable integration methods based on integrand form."""
    # VERY basic heuristics - this is not reliable for complex cases!
    methods = []
    # Integration by Parts? (Look for product of polynomial/x and transcendentals)
    if isinstance(expr, sympy.Mul):
        poly_part = None
        trans_part = None
        for arg in expr.args:
            if variable in arg.free_symbols:
                if any(isinstance(arg, f) for f in [sympy.sin, sympy.cos, sympy.exp, sympy.log]):
                    trans_part = arg
                elif arg.is_polynomial(variable) or arg == variable:
                    poly_part = arg
        if poly_part is not None and trans_part is not None:
            methods.append("Integration by Parts")

    # Trig Substitution? (Look for sqrt(a^2 +/- x^2) etc.)
    if expr.has(sympy.sqrt):
       for sqrt_expr in expr.find(sympy.sqrt):
           inner = sqrt_expr.args[0]
           # Check for patterns like a**2 - var**2, a**2 + var**2, var**2 - a**2
           pattern_match = sympy.Wild('a', exclude=[variable])**2 - sympy.Wild('b', exclude=[variable])*variable**2
           match = inner.match(pattern_match)
           if match and match[sympy.Wild('b')] > 0: # Check coeff of var^2 is positive
               methods.append("Trigonometric Substitution")
               break # Found one pattern
           pattern_match = sympy.Wild('a', exclude=[variable])**2 + sympy.Wild('b', exclude=[variable])*variable**2
           match = inner.match(pattern_match)
           if match and match[sympy.Wild('b')] > 0:
               methods.append("Trigonometric Substitution")
               break
           pattern_match = sympy.Wild('b', exclude=[variable])*variable**2 - sympy.Wild('a', exclude=[variable])**2
           match = inner.match(pattern_match)
           if match and match[sympy.Wild('b')] > 0:
               methods.append("Trigonometric Substitution")
               break

    # Partial Fractions? (Check if it's a rational function)
    if expr.is_rational_function(variable):
        # More specific check: denominator degree >= numerator degree? (often needed)
        num, den = expr.as_numer_denom()
        if den.as_poly(variable).degree() >= num.as_poly(variable).degree():
             methods.append("Partial Fraction Decomposition")

    # Basic Substitution? (Hard to guess reliably - Chain rule in reverse)
    # Maybe look for f(g(x))*g'(x) patterns - too complex for simple heuristic

    if not methods:
        return "Standard integration rules or direct lookup."
    else:
        # Use 'or' as we don't know which one SymPy *actually* used
        return f"Possible applicable methods include: { ' or '.join(methods) }."


def generate_integration_steps(expr: sympy.Expr, variable: sympy.Symbol) -> Tuple[List[StepDetail], sympy.Expr, Optional[str]]:
    """Generates basic integration steps and the final result."""
    steps = []
    computation_notes = None

    # --- Step 1: Original Integral ---
    latex_orig, pretty_orig = format_expression(expr)
    steps.append(StepDetail(
        step_description=f"Integrate the expression with respect to {variable}:",
        expression_latex=f"\\int {latex_orig} \\, d{variable.name}",
        expression_pretty=f"∫({pretty_orig}) d{variable.name}"
    ))

    # --- Step 2: Method Guess (Heuristic) ---
    # Add this *before* calculating, as it's based on the integrand
    method_guess = guess_integration_method(expr, variable)
    # Add a note about the guess
    steps.append(StepDetail(
        step_description=f"Analysis of integrand suggests: ({method_guess} Note: This is a heuristic guess based on form, not necessarily the method SymPy uses internally).",
        expression_latex="", # No expression change here
        expression_pretty=""
    ))
    computation_notes = f"Heuristic method guess: {method_guess}"


    # --- Step 3: Attempt Integration ---
    integral_result_expr = None
    try:
        # Calculate the indefinite integral
        integral_result_expr = sympy.integrate(expr, variable)
        latex_raw, pretty_raw = format_expression(integral_result_expr)

        steps.append(StepDetail(
            step_description="Applying integration rules yields (before adding constant):",
            expression_latex=latex_raw,
            expression_pretty=pretty_raw
        ))

    except NotImplementedError:
        # SymPy couldn't find an elementary integral
        err_msg = f"SymPy could not find an elementary integral for the expression with respect to {variable}."
        steps.append(StepDetail(step_description=err_msg, expression_latex="-", expression_pretty="-"))
        computation_notes = err_msg
        # Return the original expression as a placeholder 'result' if integration fails
        return steps, expr, computation_notes
    except Exception as e:
        # Catch other potential errors during integration
        raise ValueError(f"Error during integration calculation: {e}")


    # --- Step 4: Simplification (Optional) ---
    # Simplification is often less dramatic/needed for integrals than derivatives
    # but we can include it for consistency.
    final_expr = integral_result_expr # Default to unsimplified
    try:
        simplified_integral = sympy.simplify(integral_result_expr)
        latex_final, pretty_final = format_expression(simplified_integral)
        latex_raw, pretty_raw = format_expression(integral_result_expr) # Get raw again for comparison

        if latex_raw != latex_final:
             steps.append(StepDetail(
                 step_description="Simplifying the result (before adding constant):",
                 expression_latex=latex_final,
                 expression_pretty=pretty_final
             ))
             final_expr = simplified_integral # Update final expression if simplified
        else:
             final_expr = integral_result_expr # Stick with original if no change

    except Exception as e:
        print(f"Warning: Error during integral simplification: {e}")
        # Use the unsimplified result if simplification fails
        final_expr = integral_result_expr


    # --- Final Step Placeholder (in response model, add "+ C") ---
    # The actual "+ C" is added textually in the endpoint

    return steps, final_expr, computation_notes


# --- FastAPI Setup ---
# app = FastAPI(title="Symbolic Integration API")
# Or use a router
router = APIRouter(prefix="/integrate", tags=["Integration"])


# --- API Endpoints ---

@router.post("/integrate/simple", response_model=IntegrationResponseSimple)
# @router.post("/simple", response_model=IntegrationResponseSimple)
async def integrate_simple(input_data: IntegrationInput):
    """
    Calculates the indefinite integral of an expression with respect to a variable.
    Result includes '+ C'. Returns note if integration fails.
    """
    computation_notes = None
    try:
        parsed_expr, variable_sym = parse_and_validate(input_data.expression, input_data.variable)

        # Calculate integral
        integral_expr = sympy.integrate(parsed_expr, variable_sym)

        # Optional: Simplify (less critical for integrals usually)
        # simplified_integral = sympy.simplify(integral_expr)
        simplified_integral = integral_expr # Keep unsimplified for now

        # Format for output
        _, pretty_integral = format_expression(simplified_integral)
        final_result_str = f"{pretty_integral} + C" # ADD CONSTANT

        return IntegrationResponseSimple(
            original_expression=input_data.expression,
            variable=input_data.variable,
            integral_result=final_result_str,
            computation_notes=computation_notes
        )
    except NotImplementedError:
         computation_notes = f"SymPy could not find an elementary integral for the expression with respect to {input_data.variable}."
         # Return something indicate failure, maybe original expression + note
         return IntegrationResponseSimple(
            original_expression=input_data.expression,
            variable=input_data.variable,
            integral_result="Integration Failed", # Or input_data.expression
            computation_notes=computation_notes
         )
    except ValueError as e:
        # Catch parsing/validation errors
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch unexpected errors during calculation/simplification
        print(f"Error in /integrate/simple: {e}\n{traceback.format_exc()}") # Log detailed error
        raise HTTPException(status_code=500, detail="An internal server error occurred during integration.")


@router.post("/integrate/steps", response_model=IntegrationResponseSteps)
# @router.post("/steps", response_model=IntegrationResponseSteps)
async def integrate_steps(input_data: IntegrationInput):
    """
    Attempts indefinite integration and provides basic steps/method guesses.
    Result includes '+ C'. Note: Step generation is basic.
    """
    try:
        parsed_expr, variable_sym = parse_and_validate(input_data.expression, input_data.variable)

        # Generate steps and final result expression
        steps_list, final_expr, notes = generate_integration_steps(parsed_expr, variable_sym)

        # Format final result expression for the separate field in response
        _, pretty_final = format_expression(final_expr)

        # Determine final string, handle integration failure case from steps
        if notes and "could not find an elementary integral" in notes:
             final_integral_str = "Integration Failed"
        else:
             final_integral_str = f"{pretty_final} + C" # ADD CONSTANT

        return IntegrationResponseSteps(
            original_expression=input_data.expression,
            variable=input_data.variable,
            steps=steps_list,
            final_integral=final_integral_str,
            computation_notes=notes
        )
    except ValueError as e:
        # Catch parsing/validation/calculation errors from core logic
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch unexpected errors
        print(f"Error in /integrate/steps: {e}\n{traceback.format_exc()}") # Log detailed error
        raise HTTPException(status_code=500, detail="An internal server error occurred during step generation.")


# --- Add router to app if using a router ---
# app.include_router(router)

# # --- Optional root endpoint ---
# @app.get("/")
# async def root():
#     return {"message": "Symbolic Integration API. POST to /integrate/simple or /integrate/steps"}