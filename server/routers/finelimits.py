from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import sympy
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SymPy Setup ---

# Define common functions and constants for sympify
SYMPY_LOCALS = {
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "cot": sympy.cot,
    "sec": sympy.sec,
    "csc": sympy.csc,
    "ln": sympy.log,  # Sympy uses log for natural log
    "log": lambda x, base=10: sympy.log(x, base), # Allow log(x) or log(x, base)
    "sqrt": sympy.sqrt,
    "exp": sympy.exp,
    "pi": sympy.pi,
    "e": sympy.E,
    # Add Abs for cases like limit x/|x| as x->0
    "abs": sympy.Abs,
}

# --- Helper Functions ---

def _parse_sympy_input(expression: str, variable_str: str, tending_to_str: str) -> tuple:
    """Parses input strings into SymPy objects."""
    try:
        # Sanitize variable name (basic)
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", variable_str):
             raise ValueError(f"Invalid variable name: {variable_str}")
        sympy_var = sympy.symbols(variable_str)

        # Prepare expression: replace ^ with **, handle potential issues
        parsed_expression = expression.replace('^', '**')
        # Add implicit multiplication for cases like '2x' -> '2*x'
        # Be careful with this - it can be tricky. Let's stick to requiring explicit '*' for now.
        # For robustness, users should write '2*x', not '2x'.

        sympy_expr = sympy.sympify(parsed_expression, locals=SYMPY_LOCALS)

        # Parse the limit point
        t_str = tending_to_str.lower().strip()
        parsed_limit_point = t_str.replace('^', '**') # Allow expressions like pi/2

        if t_str in ["inf", "infinity", "oo", "∞"]:
            sympy_limit_point = sympy.oo
        elif t_str in ["-inf", "-infinity", "-oo", "-∞"]:
            sympy_limit_point = -sympy.oo
        else:
            # Allow constants like pi, e in the limit point
            sympy_limit_point = sympy.sympify(parsed_limit_point, locals={"pi": sympy.pi, "e": sympy.E})

        # Ensure the expression actually contains the variable
        if sympy_var not in sympy_expr.free_symbols and not sympy_expr.is_constant():
             # Allow constant expressions, but warn if variable is specified but not used
             logger.warning(f"Variable '{variable_str}' not found in expression '{expression}'. Treating as constant.")
             # Note: sympy.limit handles this correctly, evaluating the constant expression.

        return sympy_expr, sympy_var, sympy_limit_point

    except (sympy.SympifyError, TypeError, SyntaxError) as e:
        logger.error(f"SymPy parsing error: {e}")
        raise ValueError(f"Could not parse expression, variable, or limit point: {e}. Ensure correct syntax (use '*' for multiplication, '**' or '^' for powers).")
    except ValueError as e: # Catch invalid variable name
        logger.error(f"Input validation error: {e}")
        raise ValueError(str(e)) # Re-raise specific error
    except Exception as e:
        logger.error(f"Unexpected error during SymPy parsing: {e}")
        raise ValueError(f"An unexpected error occurred during input parsing: {e}")


def _format_sympy_result(sympy_res: Any, use_pretty: bool = False) -> str:
    """Formats SymPy results into readable strings."""
    if isinstance(sympy_res, sympy.Limit):
        return f"Unevaluated Limit: {sympy.pretty(sympy_res) if use_pretty else str(sympy_res)}"
    elif sympy_res is sympy.oo:
        return "∞"
    elif sympy_res is -sympy.oo:
        return "-∞"
    elif sympy_res is sympy.zoo:
        return "Complex Infinity (zoo)"
    elif sympy_res is sympy.nan:
        return "NaN (Indeterminate)"
    elif isinstance(sympy_res, sympy.AccumBounds):
         return f"Bounded Value ({sympy_res.min}, {sympy_res.max}) (e.g., oscillation)"
    elif isinstance(sympy_res, (sympy.Rational, sympy.Integer, sympy.Float)):
        # Try to format nicely
        if isinstance(sympy_res, sympy.Rational) and sympy_res.q != 1:
            return f"{sympy_res.p}/{sympy_res.q}"
        elif isinstance(sympy_res, sympy.Float):
             # Attempt rational approximation for cleaner output, otherwise format float
             try:
                 rational_approx = sympy.Rational(sympy_res).limit_denominator(1000)
                 if rational_approx.q == 1:
                      return str(rational_approx.p)
                 elif abs(float(sympy_res) - float(rational_approx)) < 1e-10: # Check if approx is good
                      return f"{rational_approx.p}/{rational_approx.q}"
                 else:
                      return f"{sympy_res:.6g}" # Sensible float format
             except Exception:
                  return f"{sympy_res:.6g}" # Fallback float format
        else: # Integers
            return str(sympy_res)
    elif sympy_res is sympy.pi:
        return "pi"
    elif sympy_res is sympy.E:
        return "e"
    else:
        # General sympy expression
        try:
            return sympy.pretty(sympy_res) if use_pretty else str(sympy_res)
        except Exception:
            return str(sympy_res) # Fallback

# --- Pydantic Models ---

class LimitRequest(BaseModel):
    expression: str = Field(..., example="(x**2 - 1)/(x - 1)")
    variable: str = Field("x", example="x")
    tending_to: str = Field(..., example="1")

class LimitResponseSimple(BaseModel):
    expression: str
    variable: str
    tending_to: str
    limit_result: str
    explanation: Optional[str] = None # Add optional explanation for non-finite results

class Step(BaseModel):
    method: str
    description: str
    expression_before: Optional[str] = None
    expression_after: Optional[str] = None
    result: Optional[str] = None

class LimitResponseSteps(BaseModel):
    expression: str
    variable: str
    tending_to: str
    steps: List[Step]
    final_result: str


# --- Step-by-Step Calculator Class ---

class LimitCalculatorSteps:
    def __init__(self, expression: str, variable_str: str, tending_to_str: str):
        self.original_expression_str = expression
        self.variable_str = variable_str
        self.tending_to_str = tending_to_str
        self.steps: List[Step] = []
        self.final_result: Optional[sympy.Basic] = None # Store the final sympy object

        # Parse inputs
        self.sympy_expr, self.sympy_var, self.sympy_limit_point = _parse_sympy_input(
            expression, variable_str, tending_to_str
        )
        self.current_expr = self.sympy_expr # Start with the original expression

    def _add_step(self, method: str, description: str, expr_before: Optional[sympy.Basic] = None, expr_after: Optional[sympy.Basic] = None, result: Optional[Any] = None):
        self.steps.append(Step(
            method=method,
            description=description,
            expression_before=_format_sympy_result(expr_before, use_pretty=True) if expr_before is not None else None,
            expression_after=_format_sympy_result(expr_after, use_pretty=True) if expr_after is not None else None,
            result=_format_sympy_result(result) if result is not None else None,
        ))
        logger.info(f"Step ({method}): {description} | Result: {result}")

    def calculate(self):
        self._add_step(
            method="Initial Problem",
            description=f"Find the limit of f({self.variable_str}) as {self.variable_str} → {_format_sympy_result(self.sympy_limit_point)}",
            expr_after=self.current_expr
        )

        # 1. Try Direct Substitution
        limit_found = False
        try:
            self._add_step(
                method="Direct Substitution",
                description=f"Substitute {self.variable_str} = {_format_sympy_result(self.sympy_limit_point)} into the expression.",
                expr_before=self.current_expr
            )
            # Use subs and evaluate. Handle cases where subs returns unevaluated objects.
            # Using limit directly is often more robust than subs for checking forms.
            num, den = self.current_expr.as_numer_denom()
            num_limit_val = sympy.limit(num, self.sympy_var, self.sympy_limit_point)
            den_limit_val = sympy.limit(den, self.sympy_var, self.sympy_limit_point)

            is_indeterminate_00 = (num_limit_val == 0 and den_limit_val == 0)
            is_indeterminate_inf_inf = (num_limit_val.is_infinite and den_limit_val.is_infinite)
            is_indeterminate_form = is_indeterminate_00 or is_indeterminate_inf_inf

            # Evaluate the full expression limit for the substitution result
            sub_result = sympy.limit(self.current_expr, self.sympy_var, self.sympy_limit_point)

            if is_indeterminate_form:
                form_type = "0/0" if is_indeterminate_00 else "∞/∞"
                self._add_step(
                    method="Direct Substitution",
                    description=f"Substitution leads to an indeterminate form of type {form_type}.",
                    result=sympy.nan # Represent indeterminate form
                )
            elif sub_result.is_finite and not isinstance(sub_result, (sympy.Limit, sympy.AccumBounds)):
                self._add_step(
                    method="Direct Substitution",
                    description="Substitution yields a finite determinate value.",
                    result=sub_result
                )
                self.final_result = sub_result
                limit_found = True
            elif sub_result.is_infinite:
                 self._add_step(
                    method="Direct Substitution",
                    description="Substitution yields infinity.",
                    result=sub_result
                 )
                 self.final_result = sub_result
                 limit_found = True
            else: # Other cases like AccumBounds, unevaluated Limit, NaN
                 self._add_step(
                     method="Direct Substitution",
                     description=f"Substitution result is '{_format_sympy_result(sub_result)}'. May be indeterminate or require other methods.",
                     result=sub_result
                 )

        except Exception as e:
            logger.warning(f"Error during substitution/initial limit check: {e}")
            self._add_step("Direct Substitution", f"Error during substitution check: {e}")


        # 2. Try Simplification (if limit not found)
        if not limit_found:
            try:
                self._add_step(
                    method="Simplification",
                    description="Attempting to simplify the expression.",
                    expr_before=self.current_expr
                )
                simplified_expr = sympy.simplify(self.current_expr)
                # For rational functions, cancel might be better
                if self.current_expr.is_rational_function():
                    simplified_expr = sympy.cancel(self.current_expr)

                if simplified_expr != self.current_expr:
                    self._add_step(
                        method="Simplification",
                        description="Expression simplified.",
                        expr_before=self.current_expr,
                        expr_after=simplified_expr
                    )
                    self.current_expr = simplified_expr # Update expression

                    # Try substitution again on simplified expression
                    self._add_step(
                        method="Direct Substitution (Simplified)",
                        description=f"Substitute {self.variable_str} = {_format_sympy_result(self.sympy_limit_point)} into the simplified expression.",
                        expr_before=self.current_expr
                    )
                    sub_result_simplified = sympy.limit(self.current_expr, self.sympy_var, self.sympy_limit_point)

                    if sub_result_simplified.is_finite and not isinstance(sub_result_simplified, (sympy.Limit, sympy.AccumBounds)):
                        self._add_step(
                            method="Direct Substitution (Simplified)",
                            description="Substitution into simplified expression yields a finite value.",
                            result=sub_result_simplified
                        )
                        self.final_result = sub_result_simplified
                        limit_found = True
                    elif sub_result_simplified.is_infinite:
                         self._add_step(
                            method="Direct Substitution (Simplified)",
                            description="Substitution into simplified expression yields infinity.",
                            result=sub_result_simplified
                         )
                         self.final_result = sub_result_simplified
                         limit_found = True
                    else:
                         num_simp, den_simp = self.current_expr.as_numer_denom()
                         num_simp_lim = sympy.limit(num_simp, self.sympy_var, self.sympy_limit_point)
                         den_simp_lim = sympy.limit(den_simp, self.sympy_var, self.sympy_limit_point)
                         is_indeterminate_simp = (num_simp_lim == 0 and den_simp_lim == 0) or \
                                                 (num_simp_lim.is_infinite and den_simp_lim.is_infinite)
                         form_type = "0/0" if (num_simp_lim == 0 and den_simp_lim == 0) else "∞/∞" if is_indeterminate_simp else "other"

                         if is_indeterminate_simp:
                              self._add_step(
                                   method="Direct Substitution (Simplified)",
                                   description=f"Simplified expression still leads to indeterminate form ({form_type}).",
                                   result=sympy.nan
                              )
                         else:
                              self._add_step(
                                  method="Direct Substitution (Simplified)",
                                  description=f"Substitution result is '{_format_sympy_result(sub_result_simplified)}'.",
                                  result=sub_result_simplified
                              )

                else:
                    self._add_step(
                        method="Simplification",
                        description="Expression could not be simplified further.",
                        expr_after=self.current_expr
                    )

            except Exception as e:
                logger.warning(f"Error during simplification: {e}")
                self._add_step("Simplification", f"Error during simplification: {e}")

        # 3. Try L'Hôpital's Rule (if applicable and limit not found)
        # Check based on the state *after* potential simplification
        if not limit_found:
             num, den = self.current_expr.as_numer_denom()
             if den != 0: # Rule doesn't apply if denominator is constant 0
                 num_limit_val = sympy.limit(num, self.sympy_var, self.sympy_limit_point)
                 den_limit_val = sympy.limit(den, self.sympy_var, self.sympy_limit_point)

                 is_lhopital_applicable = (num_limit_val == 0 and den_limit_val == 0) or \
                                          (num_limit_val.is_infinite and den_limit_val.is_infinite)

                 if is_lhopital_applicable:
                     form_type = "0/0" if num_limit_val == 0 else "∞/∞"
                     self._add_step(
                         method="L'Hôpital's Rule",
                         description=f"Indeterminate form ({form_type}) detected. Applying L'Hôpital's Rule.",
                         expr_before=self.current_expr
                     )
                     try:
                         num_diff = sympy.diff(num, self.sympy_var)
                         den_diff = sympy.diff(den, self.sympy_var)
                         self._add_step(
                             method="L'Hôpital's Rule",
                             description=f"Numerator derivative: {_format_sympy_result(num_diff, use_pretty=True)}, Denominator derivative: {_format_sympy_result(den_diff, use_pretty=True)}."
                         )

                         if den_diff == 0:
                             # This case is tricky - could mean vertical asymptote or more complex behavior
                             self._add_step(
                                 method="L'Hôpital's Rule",
                                 description="Derivative of the denominator is zero. Rule cannot be applied directly in this form. Evaluating limit via other means."
                             )
                             # Fall through to final sympy.limit attempt
                         else:
                             lhopital_expr = num_diff / den_diff
                             self._add_step(
                                 method="L'Hôpital's Rule",
                                 description="Forming the ratio of derivatives.",
                                 expr_after=lhopital_expr
                             )
                             # Calculate limit of the new expression
                             lhopital_limit_val = sympy.limit(lhopital_expr, self.sympy_var, self.sympy_limit_point)
                             self._add_step(
                                 method="L'Hôpital's Rule",
                                 description=f"Calculating the limit of the new expression ({_format_sympy_result(lhopital_expr, use_pretty=True)}).",
                                 result=lhopital_limit_val
                             )

                             # Check if L'Hopital gave a conclusive answer
                             if not isinstance(lhopital_limit_val, sympy.Limit) and lhopital_limit_val is not sympy.nan:
                                 self.final_result = lhopital_limit_val
                                 limit_found = True
                                 self._add_step("L'Hôpital's Rule", "L'Hôpital's Rule yielded a conclusive result.")
                             else:
                                  self._add_step("L'Hôpital's Rule", "L'Hôpital's Rule did not yield a conclusive result in one step. Further analysis may be needed (handled by final limit call).")

                     except Exception as e:
                         logger.warning(f"Error applying L'Hopital's Rule: {e}")
                         self._add_step("L'Hôpital's Rule", f"Error applying rule: {e}")


        # 4. Final Attempt with sympy.limit (if not found by specific methods)
        if not limit_found:
            try:
                self._add_step(
                    method="General Limit Calculation",
                    description=f"Using sympy.limit directly on the expression: {_format_sympy_result(self.current_expr, use_pretty=True)}.",
                    expr_before=self.current_expr
                )
                final_limit_val = sympy.limit(self.current_expr, self.sympy_var, self.sympy_limit_point)
                self._add_step(
                    method="General Limit Calculation",
                    description="Result from sympy.limit:",
                    result=final_limit_val
                )
                self.final_result = final_limit_val # Store the result, even if unevaluated/NaN
                limit_found = True # Consider it 'found' even if NaN/Unevaluated for reporting
            except NotImplementedError as e:
                 logger.error(f"Sympy limit calculation failed (NotImplementedError): {e}")
                 self._add_step("General Limit Calculation", f"Sympy cannot compute this limit: {e}", result="Calculation Failed")
                 self.final_result = None # Indicate failure
            except Exception as e:
                logger.error(f"Error during final sympy.limit call: {e}")
                self._add_step("General Limit Calculation", f"Error during final limit calculation: {e}", result="Calculation Failed")
                self.final_result = None # Indicate failure

        # Format the final answer
        formatted_final_result = "Calculation Failed"
        if self.final_result is not None:
            formatted_final_result = _format_sympy_result(self.final_result)
        elif self.steps: # If calculation failed but we have steps, reflect last known state if possible
             last_step_res = self.steps[-1].result
             if last_step_res and last_step_res != "Calculation Failed":
                  formatted_final_result = f"Could not determine limit (last step result: {last_step_res})"
             else:
                  formatted_final_result = "Could not determine limit"


        self._add_step("Final Result", f"The calculated limit is: {formatted_final_result}")

        return {"steps": self.steps, "final_result": formatted_final_result}


# --- API Router and Endpoints ---

router = APIRouter()

@router.get("/limE")
async def info():
    return {"message": "Welcome! POST to /limit for direct result or /limit_steps for step-by-step solution."}

@router.post("/limit", response_model=LimitResponseSimple)
async def calculate_limit_direct(request: LimitRequest):
    """
    Calculates the limit of an expression directly using SymPy.

    - **expression**: Mathematical expression (e.g., `(x**2 - 1)/(x - 1)`). Use SymPy syntax.
                      Functions: `sin, cos, tan, cot, sec, csc, ln, log, sqrt, exp, abs`.
                      Constants: `pi, e`. Use `**` or `^` for powers. Use `*` for multiplication.
    - **variable**: Variable in the expression (default: `x`).
    - **tending_to**: Value the variable approaches (e.g., `1`, `0`, `oo`, `inf`, `-oo`, `pi/2`).
    """
    try:
        sympy_expr, sympy_var, sympy_limit_point = _parse_sympy_input(
            request.expression, request.variable, request.tending_to
        )

        limit_result = sympy.limit(sympy_expr, sympy_var, sympy_limit_point)
        formatted_result = _format_sympy_result(limit_result)

        explanation = None
        if limit_result is sympy.zoo:
            explanation = "The limit approaches complex infinity."
        elif limit_result is sympy.nan:
             explanation = "The limit is indeterminate."
        elif isinstance(limit_result, sympy.AccumBounds):
             explanation = "The limit does not exist (oscillates or is bounded)."
        elif isinstance(limit_result, sympy.Limit):
             explanation = "SymPy could not evaluate the limit."

        return LimitResponseSimple(
            expression=request.expression,
            variable=request.variable,
            tending_to=request.tending_to,
            limit_result=formatted_result,
            explanation=explanation
        )

    except ValueError as e: # Catch parsing errors
        raise HTTPException(status_code=400, detail=str(e))
    except NotImplementedError as e:
         logger.error(f"Sympy calculation failed (NotImplementedError): {request.expression}, var={request.variable}, to={request.tending_to} -> {e}")
         raise HTTPException(status_code=400, detail=f"Sympy cannot compute this limit type: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error in /limit endpoint for: {request.dict()}")
        raise HTTPException(status_code=500, detail=f"Internal server error during limit calculation: {e}")


@router.post("/limit_steps", response_model=LimitResponseSteps)
async def calculate_limit_steps_endpoint(request: LimitRequest):
    """
    Calculates the limit step-by-step, showing methods like Substitution,
    Simplification, and L'Hôpital's Rule where applicable.

    - **expression**: Mathematical expression (e.g., `(x**2 - 1)/(x - 1)`). Use SymPy syntax.
                      Functions: `sin, cos, tan, cot, sec, csc, ln, log, sqrt, exp, abs`.
                      Constants: `pi, e`. Use `**` or `^` for powers. Use `*` for multiplication.
    - **variable**: Variable in the expression (default: `x`).
    - **tending_to**: Value the variable approaches (e.g., `1`, `0`, `oo`, `inf`, `-oo`, `pi/2`).
    """
    try:
        calculator = LimitCalculatorSteps(
            expression=request.expression,
            variable_str=request.variable,
            tending_to_str=request.tending_to,
        )
        result_data = calculator.calculate()

        return LimitResponseSteps(
            expression=request.expression,
            variable=request.variable,
            tending_to=request.tending_to,
            steps=result_data["steps"],
            final_result=result_data["final_result"],
        )
    except ValueError as e: # Catch parsing errors from LimitCalculatorSteps init
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error in /limit_steps endpoint for: {request.dict()}")
        # Use the final_result field from the calculator if the error happened mid-calculation
        # For now, just raise a generic 500
        raise HTTPException(status_code=500, detail=f"Internal server error during step-by-step calculation: {e}")


# --- FastAPI App Setup (for running directly) ---
app = FastAPI(title="Advanced Limit Calculator API", version="1.0.0")
app.include_router(router, prefix="/mylimits", tags=["Limits"]) # Add prefix for clarity

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Limit Calculator API. See /docs for endpoints."}

# To run: uvicorn your_filename:app --reload
# Example: uvicorn main:app --reload (if you save this as main.py)