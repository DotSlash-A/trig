from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
import math
import re
import logging
import sympy # Import sympy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/test")
async def testing():
    return {"message": "Hello from mylimits!"}


# Keep original constants for potential numerical fallback
INF = float("inf")
NEG_INF = float("-inf")
NAN = float("nan")
E = math.e
PI = math.pi

# --- Shunting Yard and Numerical Evaluation (Kept for Fallback) ---
# OPERATORS, FUNCTIONS, tokenize, is_number, get_number, evaluate_expression
# remain the same as they are needed for the numerical fallback.
# ... (Keep existing OPERATORS, FUNCTIONS, tokenize, is_number, get_number, evaluate_expression code here) ...
OPERATORS = {
    "+": {"prec": 1, "assoc": "left"},
    "-": {"prec": 1, "assoc": "left"},
    "*": {"prec": 2, "assoc": "left"},
    "/": {"prec": 2, "assoc": "left"},
    "^": {"prec": 3, "assoc": "right"},
    "unary_minus": {"prec": 4, "assoc": "right"},  # High precedence for unary minus
}

FUNCTIONS = ["sin", "cos", "tan", "cot", "sec", "csc", "ln", "log", "sqrt", "exp"]


# CORRECTED Enhanced Tokenizer
def tokenize(expr: str, variable: str) -> List[str]:
    # Pattern remains the same with capturing groups
    pattern = rf"""
        (\d+\.\d*|\.\d+|\d+) |              # Group 1: Numbers (float or int)
        (\b(?:sin|cos|tan|cot|sec|csc|ln|log|sqrt|exp)\b) | # Group 2: Functions
        (\b{re.escape(variable)}\b) |       # Group 3: Variable
        (\b(?:pi|e)\b) |                    # Group 4: Constants pi, e
        (∞|inf|infinity) |                  # Group 5: Infinity symbol variants
        (\^) |                              # Group 6: Exponentiation
        ([\+\-\*\/]) |                      # Group 7: Basic operators
        ([\(\)])                            # Group 8: Parentheses
    """
    # Find all matches - returns list of tuples
    raw_matches = re.findall(pattern, expr, re.VERBOSE | re.IGNORECASE)

    # *** CORRECTED PART START ***
    # Flatten the list of tuples. Each tuple corresponds to one token match.
    # We iterate through the tuples and take the non-empty string which is the actual match.
    flat_tokens = []
    for match_tuple in raw_matches:
        for item in match_tuple:
            if item:  # Find the non-empty element in the tuple
                flat_tokens.append(item)
                break  # Move to the next tuple once the token is found
    # *** CORRECTED PART END ***

    # Normalize infinity and constants
    normalized_tokens = []
    for token in flat_tokens:
        lower_token = token.lower()
        if lower_token in ["∞", "inf", "infinity"]:
            normalized_tokens.append("inf")
        elif lower_token == "pi":
            normalized_tokens.append("pi")
        elif lower_token == "e":
            normalized_tokens.append("e")
        else:
            # Keep original case for functions/variable? Lowercase functions for consistency.
            if token.lower() in FUNCTIONS:
                normalized_tokens.append(token.lower())
            else:
                normalized_tokens.append(token)  # Keep case for variable, numbers etc.

    # Handle unary minus (applied to normalized tokens)
    processed_tokens = []
    for i, token in enumerate(normalized_tokens):
        is_unary = False
        if token == "-":
            if i == 0:
                is_unary = True
            else:
                prev_token = normalized_tokens[i - 1]
                # Check if previous is operator, '(', or known function
                if (
                    prev_token in OPERATORS
                    or prev_token == "("
                    or prev_token in FUNCTIONS
                ):
                    is_unary = True

        if is_unary:
            processed_tokens.append("unary_minus")
        else:
            processed_tokens.append(token)

    logger.info(f"Tokenized expression '{expr}' for fallback: {processed_tokens}")
    return processed_tokens


# helper functions
def is_number(token: str) -> bool:
    if token in ["inf", "-inf"]:  # Handle string representations
        return True
    try:
        float(token)
        return True
    except ValueError:
        return False


def get_number(token: str) -> float:
    if token == "inf":
        return INF
    if token == "-inf":
        return NEG_INF  # Need to handle this case
    return float(token)


# implements the shuttign yard algorithm to evaluate the expression
def evaluate_expression(tokens: List[str], var_value: float, variable: str) -> float:
    values: List[float] = []
    ops: List[str] = []

    # Map specific string constants to float values
    const_map = {"pi": PI, "e": E, "inf": INF}

    def apply_op():
        op = ops.pop()
        try:
            if op == "unary_minus":
                val = values.pop()
                values.append(-val)
            elif op in FUNCTIONS:
                val = values.pop()
                if op == "sin":
                    values.append(math.sin(val))
                elif op == "cos":
                    values.append(math.cos(val))
                elif op == "tan":
                    # Avoid large numbers near asymptotes for numerical stability
                    if abs(math.cos(val)) < 1e-15:
                         values.append(NAN) # Undefined at asymptote
                    else:
                         values.append(math.tan(val))
                elif op == "cot":
                    tan_val = math.tan(val)
                    if abs(tan_val) < 1e-15: # Check if tan is near zero
                        values.append(NAN) # Undefined where tan is zero
                    elif abs(math.cos(val)) < 1e-15: # Check if cos is near zero (tan undefined)
                        values.append(0.0) # Cot is zero where tan is infinite
                    else:
                        values.append(1 / tan_val)
                elif op == "sec":
                    cos_val = math.cos(val)
                    if abs(cos_val) < 1e-15:
                        values.append(NAN) # Undefined at asymptote
                    else:
                        values.append(1 / cos_val)
                elif op == "csc":
                    sin_val = math.sin(val)
                    if abs(sin_val) < 1e-15:
                        values.append(NAN) # Undefined at asymptote
                    else:
                        values.append(1 / sin_val)
                elif op == "ln":
                    values.append(math.log(val) if val > 0 else NAN)
                elif op == "log": # Assume log10
                    values.append(math.log10(val) if val > 0 else NAN)
                elif op == "sqrt":
                    values.append(math.sqrt(val) if val >= 0 else NAN)
                elif op == "exp":
                    # Prevent overflow in numerical evaluation
                    try:
                        res = math.exp(val)
                        values.append(res if abs(res) < 1e300 else (INF if res > 0 else NEG_INF))
                    except OverflowError:
                        values.append(INF)

            else:  # Binary operators
                b = values.pop()
                a = values.pop()
                if op == "+":
                    values.append(a + b)
                elif op == "-":
                    values.append(a - b)
                elif op == "*":
                    # Handle 0 * inf -> NaN
                    if (a == 0 and abs(b) == INF) or (b == 0 and abs(a) == INF):
                        values.append(NAN)
                    else:
                        values.append(a * b)
                elif op == "/":
                    if b == 0:
                        if a == 0:
                            values.append(NAN)  # 0/0
                        elif a > 0:
                            values.append(INF)
                        else:
                            values.append(NEG_INF)
                    elif abs(a) == INF and abs(b) == INF:
                         values.append(NAN) # inf/inf or -inf/-inf etc.
                    elif b == INF or b == NEG_INF:
                         values.append(0.0) # x / inf = 0
                    elif a == INF:
                         values.append(INF if b > 0 else NEG_INF)
                    elif a == NEG_INF:
                         values.append(NEG_INF if b > 0 else INF)
                    else:
                        values.append(a / b)
                elif op == "^":
                    # Handle 0^0 -> NaN, x^0 -> 1, 0^y (y>0) -> 0, 0^y (y<0) -> inf, 1^inf -> NaN
                    if a == 0 and b == 0:
                        values.append(NAN)
                    elif b == 0:
                        values.append(1.0)
                    elif a == 0:
                        values.append(0.0 if b > 0 else INF) # 0^neg = inf
                    elif a == 1 and abs(b) == INF:
                        values.append(NAN)  # 1^inf is indeterminate
                    elif abs(a) == INF and b == 0:
                         values.append(NAN) # inf^0 indeterminate
                    elif a < 0 and not float(b).is_integer():
                         values.append(NAN) # Negative base to non-integer power
                    else:
                        try:
                            # Prevent overflow/underflow
                            res = math.pow(a, b)
                            if abs(res) > 1e300:
                                values.append(INF if res > 0 else NEG_INF)
                            elif abs(res) < 1e-300 and res != 0:
                                values.append(0.0)
                            else:
                                values.append(res)
                        except (ValueError, OverflowError): # e.g., (-1)^0.5
                            values.append(NAN)


        except IndexError:
            # This error is more likely in the numerical fallback if tokenization/parsing failed subtly
            logger.error(f"Numerical Eval Error: Not enough operands for operator '{op}'. Values: {values}, Ops: {ops}")
            # Push NaN to signal failure clearly
            values.append(NAN)
            # Stop further processing by clearing ops, prevents cascading errors
            while ops: ops.pop()

        except Exception as e:
            # Catch potential math errors (e.g., log(-1))
            logger.error(f"Numerical Eval Error applying operator {op}: {e}")
            values.append(NAN) # Return NaN on math errors
            while ops: ops.pop() # Stop processing

    for i, token in enumerate(tokens):
        token_lower = token.lower()

        if is_number(token):
            values.append(get_number(token))
        elif token_lower in const_map:
            values.append(const_map[token_lower])
        elif token_lower == variable.lower():
            if var_value is None:
                 # Push NaN and clear ops to signal failure
                 logger.error("Numerical Eval Error: Variable value is None during evaluation.")
                 values.append(NAN)
                 while ops: ops.pop()
                 break # Stop processing tokens
            # Handle case where variable approaches infinity directly
            if var_value == INF:
                values.append(INF)
            elif var_value == NEG_INF:
                values.append(NEG_INF)
            elif math.isnan(var_value): # If NaN is passed in somehow
                values.append(NAN)
            else:
                values.append(var_value)
        elif token_lower in FUNCTIONS:
            ops.append(token_lower)  # Store function name
        elif token == "(":
            ops.append(token)
        elif token == ")":
            while ops and ops[-1] != "(":
                apply_op()
                # If apply_op resulted in NaN and cleared ops, break
                if not ops and values and math.isnan(values[-1]): break
            if not ops: # Error occurred in apply_op or mismatched parens
                 if not values or not math.isnan(values[-1]): # Avoid adding step if already handled
                      logger.error("Numerical Eval Error: Mismatched parentheses or error during application.")
                      values.append(NAN) # Ensure NaN is the result
                 break # Stop processing
            ops.pop()  # Pop '('
            # If the token before '(' was a function name, apply it
            if ops and ops[-1] in FUNCTIONS:
                apply_op()
                # If apply_op resulted in NaN and cleared ops, break
                if not ops and values and math.isnan(values[-1]): break
        elif token in OPERATORS or token == "unary_minus":
            # Check for unary minus explicitly
            current_op = "unary_minus" if token == "unary_minus" else token
            info = OPERATORS[current_op]
            while (
                ops
                and ops[-1] != "("
                and ops[-1] in OPERATORS # Ensure previous op is in OPERATORS dict
                and (
                    OPERATORS[ops[-1]]["prec"] > info["prec"]
                    or (
                        OPERATORS[ops[-1]]["prec"] == info["prec"]
                        and info["assoc"] == "left"
                    )
                )
            ):
                apply_op()
                 # If apply_op resulted in NaN and cleared ops, break
                if not ops and values and math.isnan(values[-1]): break
            # Break the outer loop as well if an error occurred
            if values and math.isnan(values[-1]) and not ops: break
            ops.append(current_op)
        else:
            logger.error(f"Numerical Eval Error: Unknown token '{token}'")
            values = [NAN] # Set result to NaN
            ops = [] # Clear ops
            break # Stop processing

    # After loop, apply remaining ops
    while ops:
        if ops[-1] == "(":
             logger.error("Numerical Eval Error: Mismatched parentheses at end.")
             values = [NAN] # Set result to NaN
             ops = [] # Clear ops
             break
        apply_op()
        # Check if error occurred during final applications
        if values and math.isnan(values[-1]) and not ops: break


    # Final result check
    if len(values) != 1 or math.isnan(values[0]):
        # Log if it wasn't an expected NaN from an error state
        if not (values and math.isnan(values[0])):
             logger.warning(
                 f"Numerical evaluation ended with values: {values}. Ops: {ops}. Tokens: {tokens}. Returning NaN."
             )
        return NAN # Return NaN if evaluation failed or ended with non-single value

    result = values[0]

    # Clamp large/small values ONLY if they are not INF/NEG_INF already
    if result != INF and result != NEG_INF:
        if abs(result) > 1e200: # Reduced threshold slightly
            return INF if result > 0 else NEG_INF
        # Handle very small numbers that should likely be zero
        if abs(result) < 1e-12:
            return 0.0

    return result


class LimitCalculator:
    def __init__(self, expression: str, variable: str, tending_to_str: str):
        self.original_expression = expression
        self.variable_str = variable
        self.tending_to_str = tending_to_str
        self.steps = []
        self.result = "Could not determine limit"
        self.epsilon = 1e-7 # For numerical checks
        self.large_number = 1e10 # For simulating infinity numerically

        # --- Sympy Initialization ---
        self.sympy_var = sympy.symbols(variable)
        self.sympy_expr = None
        self.sympy_limit_point = None
        self.sympy_limit_point_val = None # Store numerical value if finite

        try:
            # Define common functions for sympify
            local_dict = {
                "sin": sympy.sin,
                "cos": sympy.cos,
                "tan": sympy.tan,
                "cot": sympy.cot,
                "sec": sympy.sec,
                "csc": sympy.csc,
                "ln": sympy.log, # Sympy uses log for natural log
                "log": lambda x: sympy.log(x, 10), # Define log10 explicitly
                "sqrt": sympy.sqrt,
                "exp": sympy.exp,
                "pi": sympy.pi,
                "e": sympy.E,
                # Add other functions or constants if needed
            }
            # Use sympify to parse the expression safely
            # Replace ^ with ** for sympy compatibility if necessary
            parsed_expression = expression.replace('^', '**')
            self.sympy_expr = sympy.sympify(parsed_expression, locals=local_dict)

            # Parse the limit point for sympy
            t_str = tending_to_str.lower().strip()
            if t_str in ["inf", "infinity", "oo", "∞"]:
                self.sympy_limit_point = sympy.oo
                self.sympy_limit_point_val = INF # For numerical checks
            elif t_str in ["-inf", "-infinity", "-oo", "-∞"]:
                self.sympy_limit_point = -sympy.oo
                self.sympy_limit_point_val = NEG_INF # For numerical checks
            else:
                # Allow expressions like pi/2
                parsed_limit_point = t_str.replace('^', '**')
                self.sympy_limit_point = sympy.sympify(parsed_limit_point, locals={"pi": sympy.pi, "e": sympy.E})
                # Attempt to get a numerical value for checks
                try:
                    self.sympy_limit_point_val = float(self.sympy_limit_point.evalf())
                except TypeError:
                     # Limit point might be symbolic like 'a' if not careful, handle gracefully
                     logger.warning(f"Could not evaluate limit point {self.sympy_limit_point} numerically.")
                     self.sympy_limit_point_val = None # Cannot perform numerical checks easily

        except (sympy.SympifyError, SyntaxError, TypeError) as e:
            logger.error(f"Sympy parsing error: {e}")
            raise ValueError(f"Could not parse expression or limit point: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during sympy initialization: {e}")
            raise ValueError(f"An unexpected error occurred during setup: {e}")

        # --- Numerical Fallback Initialization ---
        self.fallback_tokens = []
        self.numerical_limit_point = None # Use sympy_limit_point_val if available
        try:
            # Tokenize for fallback evaluation if needed later
            self.fallback_tokens = tokenize(self.original_expression, self.variable_str) # Use original expression for fallback tokenizer
            # Use the already parsed numerical value if available
            if self.sympy_limit_point_val is not None:
                 self.numerical_limit_point = self.sympy_limit_point_val
            else:
                 # Try parsing again specifically for numerical if sympy eval failed
                 self.numerical_limit_point = self._parse_limit_point_numerical(tending_to_str)

        except ValueError as e:
             # If tokenization or numerical limit parsing fails, we can't use fallback
             logger.warning(f"Could not initialize numerical fallback: {e}")
             self.fallback_tokens = None
             self.numerical_limit_point = None
        except Exception as e:
             logger.warning(f"Unexpected error initializing numerical fallback: {e}")
             self.fallback_tokens = None
             self.numerical_limit_point = None

    def _parse_limit_point_numerical(self, limit_point_str: str) -> float:
        """Parses limit point for numerical evaluation."""
        t_str = limit_point_str.lower().strip()
        if t_str in ["inf", "infinity", "∞"]:
            return INF
        elif t_str in ["-inf", "-infinity", "-∞"]:
            return NEG_INF
        else:
            try:
                # Evaluate simple expressions like pi/2 for numerical value
                # Be careful with eval, use restricted scope
                val = eval(t_str, {"__builtins__": None}, {"pi": PI, "e": E, "sqrt": math.sqrt})
                if not isinstance(val, (int, float)):
                     raise ValueError("Invalid numerical limit point expression")
                return float(val)
            except Exception as e:
                logger.error(f"Failed to parse numerical limit point '{limit_point_str}': {e}")
                raise ValueError(f"Invalid numerical limit point: {limit_point_str}")

    def _add_step(self, step: str, explanation: str):
        self.steps.append({"step": step, "explanation": explanation})
        logger.info(f"Step added: {step} - {explanation}")

    def _format_sympy_result(self, sympy_res) -> str:
        """Formats sympy result, including fractions."""
        if sympy_res is sympy.oo:
            return "∞"
        elif sympy_res is -sympy.oo:
            return "-∞"
        elif sympy_res is sympy.zoo: # Complex infinity
            return "Complex Infinity (zoo)"
        elif sympy_res is sympy.nan:
            return "NaN (Indeterminate)"
        elif isinstance(sympy_res, sympy.Limit):
            return f"Unevaluated Limit: {sympy.pretty(sympy_res)}"
        elif isinstance(sympy_res, sympy.AccumBounds):
             # Handle cases like sin(oo) which results in AccumBounds(-1, 1)
             return f"Bounded Value ({sympy_res.min}, {sympy_res.max})"
        elif isinstance(sympy_res, sympy.Rational):
            # Format as fraction
            return f"{sympy_res.p}/{sympy_res.q}"
        elif isinstance(sympy_res, (sympy.Float, sympy.Integer, float, int)):
             # Attempt to convert floats to fractions if they are close to simple ones
             try:
                 # Convert to Rational if possible, limit denominator for cleaner output
                 rational_approx = sympy.Rational(sympy_res).limit_denominator(1000)
                 if rational_approx.q == 1:
                     return str(rational_approx.p) # Integer
                 else:
                     return f"{rational_approx.p}/{rational_approx.q}" # Fraction
             except (TypeError, ValueError):
                 # Fallback to string representation if conversion fails
                 return str(sympy_res)
        elif sympy_res is sympy.pi:
            return "pi"
        elif sympy_res is sympy.E:
            return "e"
        else:
            # General sympy expression, convert to string using pretty print
            try:
                return sympy.pretty(sympy_res)
            except Exception:
                 return str(sympy_res) # Fallback if pretty printing fails


    def _safe_numerical_evaluate(self, value: float) -> float:
        """Evaluate expression numerically using fallback tokens, returning NAN on error"""
        if self.fallback_tokens is None:
            logger.warning("Numerical fallback not available (initialization failed).")
            return NAN

        try:
            # Use the pre-tokenized list
            eval_result = evaluate_expression(self.fallback_tokens, value, self.variable_str)

            # Check for NaN or Inf explicitly from the numerical evaluator
            if isinstance(eval_result, float) and (math.isnan(eval_result) or math.isinf(eval_result)):
                return eval_result # Return NaN, INF, NEG_INF as is
            elif not isinstance(eval_result, (int, float)):
                 logger.error(f"Numerical evaluation returned non-numeric type: {type(eval_result)}")
                 return NAN
            return float(eval_result)

        except (ValueError, ZeroDivisionError, OverflowError, TypeError, IndexError) as e:
            logger.error(f"Error during numerical evaluation at value {value}: {e}")
            return NAN
        except Exception as e:
             logger.exception(f"Unexpected error during numerical evaluation at value {value}: {e}")
             return NAN


    def calculate(self) -> Dict[str, Any]:
        self._add_step(
            "Initial Expression",
            f"Find the limit of f({self.variable_str}) = {sympy.pretty(self.sympy_expr)} as {self.variable_str} → {self.tending_to_str}",
        )

        current_expr = self.sympy_expr
        limit_calculated = False
        final_result_obj = None # Store the sympy object result

        try:
            # 1. Attempt Direct Substitution
            self._add_step("Attempt Direct Substitution", f"Substitute {self.variable_str} = {self.tending_to_str} into the expression.")
            try:
                # Use subs for substitution, then evaluate if possible
                sub_result = current_expr.subs(self.sympy_var, self.sympy_limit_point)
                self._add_step("Substitution Result", f"Result: {sympy.pretty(sub_result)}")

                # Check if substitution yielded a direct answer (finite, oo, -oo)
                # Need to be careful with symbolic results like 'pi' vs indeterminate forms
                is_indeterminate = sub_result.is_infinite or sub_result is sympy.nan or sub_result.is_complex or \
                                   (isinstance(sub_result, sympy.Number) and not sub_result.is_finite and sub_result is not sympy.oo and sub_result is not -sympy.oo) or \
                                   sub_result.has(sympy.oo, -sympy.oo, sympy.zoo, sympy.nan) # More robust check for hidden indeterminacy

                if not is_indeterminate and sub_result.is_finite is not None: # Check if it's a determinate value
                    final_result_obj = sub_result
                    limit_calculated = True
                    self._add_step("Conclusion", f"Direct substitution yields a determinate value: {self._format_sympy_result(final_result_obj)}")
                else:
                    # Check for specific indeterminate forms like 0/0 or oo/oo for L'Hopital's later
                    num, den = current_expr.as_numer_denom()
                    num_limit = sympy.limit(num, self.sympy_var, self.sympy_limit_point)
                    den_limit = sympy.limit(den, self.sympy_var, self.sympy_limit_point)

                    indeterminate_form = None
                    if num_limit == 0 and den_limit == 0:
                        indeterminate_form = "0/0"
                    elif num_limit.is_infinite and den_limit.is_infinite:
                         # Check sign for oo/oo, -oo/oo etc.
                         if (num_limit == sympy.oo and den_limit == sympy.oo) or \
                            (num_limit == -sympy.oo and den_limit == -sympy.oo):
                             indeterminate_form = "∞/∞"
                         elif (num_limit == sympy.oo and den_limit == -sympy.oo) or \
                              (num_limit == -sympy.oo and den_limit == sympy.oo):
                             indeterminate_form = "-∞/∞" # Or similar
                         else: # Mixed infinities or complex infinity
                             indeterminate_form = "∞/∞ type"
                    # Add checks for other forms if needed (0*oo, oo-oo, 1^oo, 0^0, oo^0)

                    if indeterminate_form:
                         self._add_step("Indeterminate Form", f"Substitution results in an indeterminate form of type {indeterminate_form}.")
                    else:
                         self._add_step("Indeterminate Form", f"Substitution results in an indeterminate form ({sympy.pretty(sub_result)}).")

            except Exception as e:
                self._add_step("Substitution Error", f"Error during substitution: {e}")
                logger.warning(f"Error during substitution: {e}")

            # 2. Attempt Simplification (if substitution failed or was indeterminate)
            if not limit_calculated:
                self._add_step("Attempt Simplification", "Try simplifying the expression.")
                try:
                    # Use cancel for rational functions, simplify for general expressions
                    if current_expr.is_rational_function():
                        simplified_expr = sympy.cancel(current_expr)
                        simplification_method = "sympy.cancel"
                    else:
                        simplified_expr = sympy.simplify(current_expr)
                        simplification_method = "sympy.simplify"

                    if simplified_expr != current_expr:
                        self._add_step("Simplification Result", f"Using {simplification_method}: {sympy.pretty(simplified_expr)}")
                        current_expr = simplified_expr # Update current expression

                        # Try substitution again on the simplified expression
                        self._add_step("Attempt Substitution (Simplified)", f"Substitute {self.variable_str} = {self.tending_to_str} into the simplified expression.")
                        sub_result_simplified = current_expr.subs(self.sympy_var, self.sympy_limit_point)
                        self._add_step("Substitution Result (Simplified)", f"Result: {sympy.pretty(sub_result_simplified)}")

                        is_indeterminate_simplified = sub_result_simplified.is_infinite or sub_result_simplified is sympy.nan or sub_result_simplified.is_complex or \
                                           (isinstance(sub_result_simplified, sympy.Number) and not sub_result_simplified.is_finite and sub_result_simplified is not sympy.oo and sub_result_simplified is not -sympy.oo) or \
                                           sub_result_simplified.has(sympy.oo, -sympy.oo, sympy.zoo, sympy.nan)

                        if not is_indeterminate_simplified and sub_result_simplified.is_finite is not None:
                            final_result_obj = sub_result_simplified
                            limit_calculated = True
                            self._add_step("Conclusion", f"Substitution into simplified expression yields: {self._format_sympy_result(final_result_obj)}")
                        else:
                             # Check form again for L'Hopital's
                             num, den = current_expr.as_numer_denom()
                             num_limit = sympy.limit(num, self.sympy_var, self.sympy_limit_point)
                             den_limit = sympy.limit(den, self.sympy_var, self.sympy_limit_point)
                             if (num_limit == 0 and den_limit == 0) or (num_limit.is_infinite and den_limit.is_infinite):
                                 self._add_step("Indeterminate Form (Simplified)", "Simplified expression still results in an indeterminate form suitable for L'Hôpital's Rule.")
                             else:
                                 self._add_step("Indeterminate Form (Simplified)", f"Substitution results in an indeterminate form ({sympy.pretty(sub_result_simplified)}).")

                    else:
                        self._add_step("Simplification Result", "Expression could not be simplified further.")

                except Exception as e:
                    self._add_step("Simplification Error", f"Error during simplification: {e}")
                    logger.warning(f"Error during simplification: {e}")

            # 3. Attempt L'Hôpital's Rule (if form is 0/0 or oo/oo)
            if not limit_calculated:
                # Check the form AFTER potential simplification
                num, den = current_expr.as_numer_denom()
                num_limit = sympy.limit(num, self.sympy_var, self.sympy_limit_point)
                den_limit = sympy.limit(den, self.sympy_var, self.sympy_limit_point)

                # Check for 0/0 or (+/-)oo/(+/-)oo
                is_lhopital_applicable = (num_limit == 0 and den_limit == 0) or \
                                         (num_limit.is_infinite and den_limit.is_infinite)

                if is_lhopital_applicable:
                    self._add_step("Attempt L'Hôpital's Rule", "Indeterminate form (0/0 or ∞/∞) detected. Applying L'Hôpital's Rule.")
                    try:
                        num_diff = sympy.diff(num, self.sympy_var)
                        den_diff = sympy.diff(den, self.sympy_var)
                        self._add_step("Derivatives", f"Derivative of numerator: {sympy.pretty(num_diff)}\nDerivative of denominator: {sympy.pretty(den_diff)}")

                        if den_diff == 0:
                             self._add_step("L'Hôpital's Rule Error", "Denominator derivative is zero. Rule cannot be applied further in this way.")
                        else:
                            lhopital_expr = num_diff / den_diff
                            self._add_step("L'Hôpital's Expression", f"New expression is: {sympy.pretty(lhopital_expr)}")

                            # Calculate limit of the new expression
                            self._add_step("Calculate Limit (L'Hôpital)", f"Find limit of {sympy.pretty(lhopital_expr)} as {self.variable_str} → {self.tending_to_str}")
                            # Use sympy.limit directly on the L'Hopital expression
                            limit_result_lhopital = sympy.limit(lhopital_expr, self.sympy_var, self.sympy_limit_point)

                            formatted_lhopital_res = self._format_sympy_result(limit_result_lhopital)
                            self._add_step("L'Hôpital's Result", f"Result after applying L'Hôpital's Rule: {formatted_lhopital_res}")

                            # Check if L'Hopital gave a conclusive answer
                            if not isinstance(limit_result_lhopital, sympy.Limit) and limit_result_lhopital is not sympy.nan:
                                final_result_obj = limit_result_lhopital
                                limit_calculated = True
                                self._add_step("Conclusion", f"L'Hôpital's Rule yields the limit: {formatted_lhopital_res}")
                            else:
                                 self._add_step("L'Hôpital's Rule Inconclusive", "L'Hôpital's Rule did not yield a determinate limit. May need further application or other methods.")

                    except Exception as e:
                        self._add_step("L'Hôpital's Rule Error", f"Error applying L'Hôpital's Rule: {e}")
                        logger.warning(f"Error applying L'Hopital's Rule: {e}")

            # 4. Final Symbolic Attempt with sympy.limit (if other methods failed)
            if not limit_calculated:
                 self._add_step("Attempt General Symbolic Limit", f"Using sympy.limit on the expression: {sympy.pretty(current_expr)}")
                 try:
                     limit_result_sympy = sympy.limit(current_expr, self.sympy_var, self.sympy_limit_point)
                     formatted_sympy_res = self._format_sympy_result(limit_result_sympy)
                     self._add_step("Sympy Limit Result", f"sympy.limit result: {formatted_sympy_res}")

                     # Check if sympy could evaluate it
                     if not isinstance(limit_result_sympy, sympy.Limit) and limit_result_sympy is not sympy.nan:
                         final_result_obj = limit_result_sympy
                         limit_calculated = True
                         self._add_step("Conclusion", f"General symbolic calculation yields: {formatted_sympy_res}")
                     else:
                          self._add_step("Symbolic Limit Inconclusive", "sympy.limit returned an unevaluated or indeterminate result.")

                 except NotImplementedError as e:
                      self._add_step("Symbolic Limit Error", f"Sympy does not support calculating this limit type: {e}")
                      logger.warning(f"Sympy NotImplementedError: {e}")
                 except Exception as e:
                      # Catch potential errors within sympy's limit calculation
                      self._add_step("Symbolic Limit Error", f"An error occurred during sympy.limit: {e}")
                      logger.exception(f"Error during sympy.limit execution: {e}")


            # 5. Numerical Check (Fallback) - Only if all symbolic methods failed
            if not limit_calculated:
                 # Check if numerical fallback is possible
                 if self.fallback_tokens and self.numerical_limit_point is not None:
                    self._add_step(
                        "Attempting Numerical Check (Fallback)",
                        f"Symbolic methods failed. Evaluating function numerically near {self.variable_str} = {self.tending_to_str}.",
                    )

                    num_limit_point = self.numerical_limit_point
                    limit_approx_str = "Could not determine limit numerically"

                    if num_limit_point == INF:
                        # Evaluate at a large number
                        val = self._safe_numerical_evaluate(self.large_number)
                        self._add_step("Numerical Evaluation (∞)", f"f({self.large_number}) ≈ {val}")
                        if val == INF: limit_approx_str = "∞"
                        elif val == NEG_INF: limit_approx_str = "-∞"
                        elif not math.isnan(val): limit_approx_str = f"≈ {val}" # Indicate approximation

                    elif num_limit_point == NEG_INF:
                         # Evaluate at a large negative number
                        val = self._safe_numerical_evaluate(-self.large_number)
                        self._add_step("Numerical Evaluation (-∞)", f"f({-self.large_number}) ≈ {val}")
                        if val == INF: limit_approx_str = "∞"
                        elif val == NEG_INF: limit_approx_str = "-∞"
                        elif not math.isnan(val): limit_approx_str = f"≈ {val}" # Indicate approximation

                    else: # Finite limit point
                        try:
                            left_val = self._safe_numerical_evaluate(num_limit_point - self.epsilon)
                            right_val = self._safe_numerical_evaluate(num_limit_point + self.epsilon)
                            self._add_step(
                                "Numerical Evaluation (Finite)",
                                f"f({num_limit_point - self.epsilon:.2e}) ≈ {left_val}, f({num_limit_point + self.epsilon:.2e}) ≈ {right_val}",
                            )

                            # Check if left and right limits are close enough
                            tolerance = 1e-5 # Tolerance for numerical check agreement
                            if math.isnan(left_val) or math.isnan(right_val):
                                 limit_approx_str = "DNE (Numerical evaluation failed near point)"
                            elif math.isinf(left_val) and math.isinf(right_val):
                                 if left_val == right_val:
                                     limit_approx_str = "∞" if left_val > 0 else "-∞"
                                 else:
                                     limit_approx_str = "DNE (Infinite oscillation or different infinities)"
                            elif abs(left_val - right_val) < tolerance * (1 + abs(left_val)): # Relative tolerance
                                limit_approx = (left_val + right_val) / 2
                                # Refine approximation slightly
                                if abs(limit_approx) < 1e-9: limit_approx = 0.0
                                # Try to format as fraction if close
                                try:
                                     rational_approx = sympy.Rational(limit_approx).limit_denominator(1000)
                                     if rational_approx.q == 1:
                                         limit_approx_str = f"≈ {rational_approx.p}"
                                     else:
                                         limit_approx_str = f"≈ {rational_approx.p}/{rational_approx.q}"
                                except (TypeError, ValueError):
                                     limit_approx_str = f"≈ {limit_approx:.6g}" # Fallback to float formatting

                                self._add_step("Numerical Result", f"Left/Right values are close. Approximated limit {limit_approx_str}")
                            else:
                                limit_approx_str = "DNE (Left/Right limits differ)"
                                self._add_step("Numerical Result", f"Left ({left_val}) and Right ({right_val}) evaluations differ significantly.")

                        except Exception as e:
                            logger.error(f"Error during numerical check: {e}")
                            self._add_step("Numerical Check Failed", f"An error occurred during numerical evaluation: {e}")
                            limit_approx_str = "Error during numerical check"

                    # Use numerical result as the final answer
                    self.result = limit_approx_str
                    # Add a note that this is a numerical approximation
                    if "DNE" not in self.result and "Error" not in self.result and "∞" not in self.result:
                         self._add_step("Conclusion", f"Using numerical approximation as fallback: {self.result}")
                    else:
                         self._add_step("Conclusion", f"Numerical fallback result: {self.result}")

                 else:
                     # Numerical fallback was not possible
                     self._add_step("Fallback Failed", "Numerical fallback could not be attempted (initialization failed or limit point unsuitable).")
                     self.result = "Could not determine limit (Symbolic failed, Numerical unavailable)"


            # Set final result string if calculated symbolically
            if limit_calculated and final_result_obj is not None:
                 self.result = self._format_sympy_result(final_result_obj)
            elif not limit_calculated and self.result == "Could not determine limit": # If numerical fallback didn't run or failed indeterminately
                 # Check if a specific reason was logged
                 found_reason = any(
                     step["step"] == "Symbolic Limit Inconclusive" or
                     "DNE" in step["explanation"] or
                     "Error" in step["explanation"] or
                     "failed" in step["explanation"].lower() or
                     "Inconclusive" in step["step"]
                     for step in self.steps
                 )
                 if not found_reason:
                     self._add_step("Conclusion", "Failed to determine the limit using available symbolic and numerical methods.")
                 self.result = "Could not determine limit"


        except ValueError as e: # Catch parsing errors from __init__
            logger.error(f"Input Error: {e}")
            self._add_step("Error", f"Invalid input: {e}")
            self.result = f"Error: Invalid Input ({e})"
        except Exception as e:
            logger.exception("Unexpected error during limit calculation") # Log full traceback
            self._add_step("Error", f"An unexpected server error occurred: {e}")
            self.result = "Error: Calculation Failed"


        return {"steps": self.steps, "result": self.result}


# --- API Endpoint --- (Keep existing API endpoint code) ---
class LimitRequest(BaseModel):
    expression: str
    variable: str = "x"  # Default variable
    tending_to: str


class LimitResponse(BaseModel):
    steps: List[Dict[str, str]]
    result: str


@router.post("/calc_limit", response_model=LimitResponse)
async def calculate_limit_endpoint(request: LimitRequest):
    """
    Calculates the limit of an expression step-by-step using Sympy, with numerical fallback.

    - **expression**: The mathematical expression (e.g., `(x^2 - 1)/(x - 1)`, `sin(x)/x`, `(1 + 1/n)**n`). Use standard functions like `sin`, `cos`, `ln` (natural log), `log` (base 10), `exp`, `sqrt`. Use `pi` and `e` for constants. Use `oo`, `inf` or `∞` for infinity. Ensure explicit multiplication with `*` where needed (e.g., `2*x` not `2x`). Powers use `**` or `^`.
    - **variable**: The variable in the expression (default: `x`).
    - **tending_to**: The value the variable approaches (e.g., `1`, `0`, `oo`, `inf`, `-oo`, `-inf`, `pi/2`).
    """
    try:
        # Input validation (basic)
        if not request.expression or not request.variable or not request.tending_to:
             raise ValueError("Expression, variable, and tending_to fields are required.")
        # Basic check for potentially unsafe characters (though sympify is generally safe)
        # Allow common math symbols like +, -, *, /, ^, (, )
        # Disallow characters often used in injection attacks or causing parsing issues
        # This is a basic check, sympify handles most math syntax safely.
        if re.search(r"[;&|`$<>!#%\\\{\}\[\]~]", request.expression + request.variable + request.tending_to):
             raise ValueError("Input contains potentially unsafe or unsupported characters.")


        calculator = LimitCalculator(
            expression=request.expression,
            variable=request.variable,
            tending_to_str=request.tending_to,
        )
        result_data = calculator.calculate()
        return LimitResponse(**result_data)
    except ValueError as e:
        # Catch specific errors like invalid input or parsing errors
        logger.error(f"Input/Calculation ValueError: {e}")
        # Provide a slightly more informative error message back to the user
        error_detail = str(e)
        if "Could not parse" in error_detail:
            error_detail = "Failed to parse expression or limit point. Check syntax (e.g., use '*' for multiplication, '**' or '^' for powers)."
        elif "Invalid input" in error_detail:
             error_detail = "Invalid input provided. " + error_detail
        raise HTTPException(status_code=400, detail=error_detail)
    except Exception as e:
        # Catch unexpected errors during calculation
        logger.exception("Unhandled exception in /calculate_limit")
        # Avoid leaking internal error details unless necessary for debugging
        raise HTTPException(status_code=500, detail="Internal server error during limit calculation.")


@router.get("/limE", include_in_schema=False)
async def root():
    return {"message": "Welcome to the Limit Calculator API. POST to /calc_limit"}
