# main.py
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
import math
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


# app = FastAPI(
#     title="Advanced Limit Calculator API",
#     description="Calculates limits of functions step-by-step without external CAS libraries."
# )

# --- Core Calculation Logic (Moved to a separate module ideally) ---

# Constants
INF = float("inf")
NEG_INF = float("-inf")
NAN = float("nan")
E = math.e
PI = math.pi


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

    logger.info(f"Tokenized expression '{expr}' into: {processed_tokens}")
    return processed_tokens


# Helper Functions
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


OPERATORS = {
    "+": {"prec": 1, "assoc": "left"},
    "-": {"prec": 1, "assoc": "left"},
    "*": {"prec": 2, "assoc": "left"},
    "/": {"prec": 2, "assoc": "left"},
    "^": {"prec": 3, "assoc": "right"},
    "unary_minus": {"prec": 4, "assoc": "right"},  # High precedence for unary minus
}

FUNCTIONS = ["sin", "cos", "tan", "cot", "sec", "csc", "ln", "log", "sqrt", "exp"]


# Shunting-Yard based Expression Evaluator
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
                    values.append(math.tan(val))
                elif op == "cot":
                    values.append(
                        1 / math.tan(val)
                        if math.tan(val) != 0
                        else INF if math.cos(val) != 0 else NAN
                    )  # Handle cot(pi/2)=0 etc.
                elif op == "sec":
                    values.append(1 / math.cos(val) if math.cos(val) != 0 else INF)
                elif op == "csc":
                    values.append(1 / math.sin(val) if math.sin(val) != 0 else INF)
                elif op == "ln":
                    values.append(math.log(val) if val > 0 else NAN)
                elif op == "log":
                    values.append(math.log10(val) if val > 0 else NAN)  # log base 10
                elif op == "sqrt":
                    values.append(math.sqrt(val) if val >= 0 else NAN)
                elif op == "exp":
                    values.append(math.exp(val))
            else:  # Binary operators
                b = values.pop()
                a = values.pop()
                if op == "+":
                    values.append(a + b)
                elif op == "-":
                    values.append(a - b)
                elif op == "*":
                    values.append(a * b)
                elif op == "/":
                    if b == 0:
                        if a == 0:
                            values.append(NAN)  # 0/0
                        elif a > 0:
                            values.append(INF)
                        else:
                            values.append(NEG_INF)
                    elif a == INF and b == INF:
                        values.append(NAN)  # inf/inf
                    elif a == NEG_INF and b == NEG_INF:
                        values.append(NAN)  # -inf/-inf etc.
                    else:
                        values.append(a / b)
                elif op == "^":
                    # Handle 0^0 -> NaN, x^0 -> 1, 0^y (y>0) -> 0, 0^y (y<0) -> inf, 1^inf -> NaN (indeterminate)
                    if a == 0 and b == 0:
                        values.append(NAN)
                    elif b == 0:
                        values.append(1.0)
                    elif a == 0:
                        values.append(0.0 if b > 0 else INF)
                    elif a == 1 and abs(b) == INF:
                        values.append(NAN)  # 1^inf is indeterminate
                    elif a == INF and b == 0:
                        values.append(NAN)  # inf^0 indeterminate
                    # Add more inf^inf, 0^inf etc. handling if needed
                    else:
                        values.append(math.pow(a, b))

        except IndexError:
            raise ValueError(
                f"Invalid expression: Not enough operands for operator '{op}'"
            )
        except Exception as e:
            # Catch potential math errors (e.g., log(-1))
            logger.error(f"Error applying operator {op}: {e}")
            values.append(NAN)  # Return NaN on math errors

    for i, token in enumerate(tokens):
        token_lower = token.lower()

        if is_number(token):
            values.append(get_number(token))
        elif token_lower in const_map:
            values.append(const_map[token_lower])
        elif token_lower == variable.lower():
            if var_value is None:
                raise ValueError("Variable value not provided for evaluation")
            # Handle case where variable approaches infinity directly
            if var_value == INF:
                values.append(INF)
            elif var_value == NEG_INF:
                values.append(NEG_INF)
            else:
                values.append(var_value)
        elif token_lower in FUNCTIONS:
            ops.append(token_lower)  # Store function name
        elif token == "(":
            ops.append(token)
        elif token == ")":
            while ops and ops[-1] != "(":
                apply_op()
            if not ops or ops[-1] != "(":
                raise ValueError("Mismatched parentheses")
            ops.pop()  # Pop '('
            # If the token before '(' was a function name, apply it
            if ops and ops[-1] in FUNCTIONS:
                apply_op()
        elif token in OPERATORS or token == "unary_minus":
            info = OPERATORS[token]
            while (
                ops
                and ops[-1] != "("
                and (
                    OPERATORS[ops[-1]]["prec"] > info["prec"]
                    or (
                        OPERATORS[ops[-1]]["prec"] == info["prec"]
                        and info["assoc"] == "left"
                    )
                )
            ):
                apply_op()
            ops.append(token)
        else:
            raise ValueError(f"Unknown token: {token}")

    while ops:
        if ops[-1] == "(":
            raise ValueError("Mismatched parentheses")
        apply_op()

    if len(values) != 1:
        # Check for implicit multiplication cases missed, or other errors
        logger.warning(
            f"Evaluation ended with multiple values: {values}. Ops: {ops}. Tokens: {tokens}"
        )
        raise ValueError("Invalid expression: Evaluation resulted in multiple values")

    result = values[0]
    # Clamp extremely large/small values resulting from float precision issues near inf
    if abs(result) > 1e300:
        return INF if result > 0 else NEG_INF
    # Handle very small numbers that should likely be zero
    if abs(result) < 1e-12:
        return 0.0

    return result


# --- Limit Calculation Logic ---


class LimitCalculator:
    def __init__(self, expression: str, variable: str, tending_to_str: str):
        self.original_expression = expression
        self.variable = variable
        self.steps = []
        self.result = "Could not determine limit"
        self.tending_to_str = tending_to_str
        self.limit_point = self._parse_limit_point(tending_to_str)
        self.tokens = []
        self.epsilon = 1e-7  # Small value for numerical checks
        self.large_num = 1e10  # Value for simulating infinity

    def _add_step(self, step: str, explanation: str):
        self.steps.append({"step": step, "explanation": explanation})
        logger.info(f"Step: {step} - {explanation}")

    def _parse_limit_point(self, tending_to_str: str) -> float:
        t_str = tending_to_str.lower().strip()
        if t_str in ["inf", "infinity", "∞"]:
            return INF
        elif t_str in ["-inf", "-infinity", "-∞"]:
            return NEG_INF
        else:
            try:
                return float(t_str)
            except ValueError:
                raise ValueError(
                    f"Invalid limit point: '{tending_to_str}'. Must be a number, 'inf', or '-inf'."
                )

    def _safe_evaluate(self, expr_tokens: list, value: float) -> float:
        """Evaluate expression, returning NAN on error."""
        try:
            # Handle cases where variable is exactly infinity
            if value == INF or value == NEG_INF:
                # Need a way to evaluate limits symbolically for infinity
                # For now, use a large number simulation
                simulated_val = self.large_num if value == INF else -self.large_num
                logger.warning(
                    f"Evaluating at infinity by substituting large number: {simulated_val}"
                )
                eval_result = evaluate_expression(
                    expr_tokens, simulated_val, self.variable
                )
            else:
                eval_result = evaluate_expression(expr_tokens, value, self.variable)

            # Post-evaluation checks for indeterminate forms resulting from evaluation
            if isinstance(eval_result, float) and math.isnan(eval_result):
                return NAN  # Consistent NaN representation
            return eval_result
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.warning(f"Evaluation error at value {value}: {e}")
            # Check if error implies infinity (e.g., division by zero near limit)
            if isinstance(e, ZeroDivisionError):
                # Crude check: evaluate slightly off point
                try:
                    val_plus = evaluate_expression(
                        expr_tokens,
                        value + self.epsilon * (1 if value >= 0 else -1),
                        self.variable,
                    )
                    val_minus = evaluate_expression(
                        expr_tokens,
                        value - self.epsilon * (1 if value >= 0 else -1),
                        self.variable,
                    )
                    if val_plus > 1e10 and val_minus > 1e10:
                        return INF
                    if val_plus < -1e10 and val_minus < -1e10:
                        return NEG_INF
                except:
                    pass  # Ignore errors in this check
            return NAN  # Return NaN if evaluation fails or gives math error

    def _check_standard_limits(
        self, expr_str: str, limit_point: float
    ) -> tuple[float | None, str | None]:
        """Check for common standard limits."""
        expr_simple = expr_str.replace(" ", "").lower()
        var = self.variable.lower()

        # Limits as var -> 0
        if abs(limit_point) < self.epsilon:
            if expr_simple == f"sin({var})/{var}" or expr_simple == f"{var}/sin({var})":
                return 1.0, f"Standard limit: lim (sin({var})/{var}) as {var}->0 = 1"
            if expr_simple == f"(1-cos({var}))/{var}":
                return 0.0, f"Standard limit: lim (1-cos({var}))/{var} as {var}->0 = 0"
            if expr_simple == f"(1-cos({var}))/({var}^2)":
                return (
                    0.5,
                    f"Standard limit: lim (1-cos({var}))/({var}^2) as {var}->0 = 1/2",
                )
            if expr_simple == f"tan({var})/{var}" or expr_simple == f"{var}/tan({var})":
                return 1.0, f"Standard limit: lim (tan({var})/{var}) as {var}->0 = 1"
            if (
                expr_simple == f"(e^{var}-1)/{var}"
                or expr_simple == f"({var})/(e^{var}-1)"
            ):
                return 1.0, f"Standard limit: lim (e^{var}-1)/{var} as {var}->0 = 1"
            if (
                expr_simple == f"ln(1+{var})/{var}"
                or expr_simple == f"{var}/ln(1+{var})"
            ):
                return 1.0, f"Standard limit: lim ln(1+{var})/{var} as {var}->0 = 1"
            # Add more standard limits here (e.g., (a^x-1)/x)

        # Add limits as var -> infinity if needed
        # e.g., (1 + 1/x)^x -> e

        return None, None

    def _analyze_form(self, value: float) -> str:
        """Analyze the result of direct substitution."""
        if isinstance(value, (int, float)):
            if math.isnan(value):
                # Need to determine *why* it's NaN (0/0, inf/inf, 0*inf, inf-inf, 1^inf, 0^0, inf^0)
                # This requires evaluating numerator/denominator or parts separately,
                # which is complex without an AST. We'll make educated guesses based on common patterns.
                # This part is a major limitation without symbolic representation.

                # Simplistic check: if expression has '/', suspect 0/0 or inf/inf
                if "/" in self.original_expression:
                    # Try evaluating numerator and denominator separately (VERY basic)
                    try:
                        parts = self.original_expression.split("/")
                        num_expr = parts[0].strip("() ")
                        den_expr = parts[1].strip("() ")
                        num_val = self._safe_evaluate(
                            tokenize(num_expr, self.variable), self.limit_point
                        )
                        den_val = self._safe_evaluate(
                            tokenize(den_expr, self.variable), self.limit_point
                        )

                        logger.info(
                            f"Indeterminate check: Num val={num_val}, Den val={den_val}"
                        )

                        if abs(num_val) < self.epsilon and abs(den_val) < self.epsilon:
                            return "0/0"
                        if abs(num_val) == INF and abs(den_val) == INF:
                            return "∞/∞"
                    except Exception as e:
                        logger.warning(f"Failed numerator/denominator split check: {e}")
                        pass  # Fallback below

                # Simplistic check for 1^inf
                if "^" in self.original_expression:
                    try:
                        parts = self.original_expression.split("^")
                        base_expr = parts[0].strip("() ")
                        exp_expr = parts[1].strip("() ")
                        base_val = self._safe_evaluate(
                            tokenize(base_expr, self.variable), self.limit_point
                        )
                        exp_val = self._safe_evaluate(
                            tokenize(exp_expr, self.variable), self.limit_point
                        )

                        logger.info(
                            f"Indeterminate check: Base val={base_val}, Exp val={exp_val}"
                        )

                        if abs(base_val - 1) < self.epsilon and abs(exp_val) == INF:
                            return "1^∞"
                        if abs(base_val) < self.epsilon and abs(exp_val) < self.epsilon:
                            return "0^0"
                        if abs(base_val) == INF and abs(exp_val) < self.epsilon:
                            return "∞^0"
                    except Exception as e:
                        logger.warning(f"Failed base/exponent split check: {e}")
                        pass  # Fallback below

                # Other checks (0 * inf, inf - inf) are even harder without structure

                return "Indeterminate (Unknown Type)"  # Default if specific form not identified
            elif math.isinf(value):
                return "∞" if value > 0 else "-∞"
            else:
                return "Finite"
        return "Evaluation Failed"

    # --- Placeholder Symbolic Methods ---
    # These would require a proper CAS implementation
    def _try_factorization(self, current_expr_str: str) -> str | None:
        """
        Attempts simple pattern-based factorization. Returns simplified expression string or None.
        THIS IS HIGHLY SIMPLIFIED AND ILLUSTRATIVE.
        """
        expr_simple = current_expr_str.replace(" ", "").lower()
        var = self.variable.lower()

        # Example: (x^2 - a^2) / (x - a) -> x + a
        if self.limit_point != INF and self.limit_point != NEG_INF:
            a = self.limit_point
            a_sq = a * a

            # Construct target patterns, handling potential float representations
            # Use format to avoid issues with negative 'a' in regex pattern construct
            a_str = str(a)
            a_sq_str = str(a_sq)

            # More robust check comparing simplified strings
            target_pattern1 = f"({var}^2-{a_sq_str})/({var}-{a_str})"
            target_pattern2 = f"(({var}^2)-({a_sq_str}))/(({var})-({a_str}))"  # With extra parens sometimes present

            if expr_simple == target_pattern1 or expr_simple == target_pattern2:
                self._add_step(
                    "Applying factorization",
                    f"Detected form (x² - a²)/(x - a) where a = {a}. Simplifying to x + a.",
                )
                # *** CORRECTED RETURN VALUE ***
                return f"{var} + {a}"  # Return the actual simplified expression string

        # Add more simple patterns here

        return None  # No simple factorization found

    def _try_rationalization(self, current_expr_str: str) -> str | None:
        """
        Attempts simple pattern-based rationalization. Returns modified expression string or None.
        THIS IS HIGHLY SIMPLIFIED AND ILLUSTRATIVE.
        """
        expr = current_expr_str.replace(" ", "").lower()
        var = self.variable.lower()

        # Example: (sqrt(x) - sqrt(a)) / (x - a) as x -> a
        # Multiply by (sqrt(x) + sqrt(a)) / (sqrt(x) + sqrt(a))
        # -> (x - a) / ((x - a) * (sqrt(x) + sqrt(a))) -> 1 / (sqrt(x) + sqrt(a))
        if (
            self.limit_point != INF
            and self.limit_point != NEG_INF
            and self.limit_point >= 0
        ):
            a = self.limit_point
            sqrt_a = math.sqrt(a)
            # Pattern is very specific here due to lack of parsing
            pattern = rf"\(sqrt\({var}\)-({sqrt_a}|{float(sqrt_a)})\)/\(({var}-({a}|{float(a)}))\)"
            if re.search(pattern, expr):
                self._add_step(
                    "Applying rationalization",
                    f"Detected form (√x - √a)/(x - a) where a = {a}. Multiplying by conjugate (√x + √a)/(√x + √a).",
                )
                self._add_step(
                    "Simplification",
                    f"Expression becomes (x - a) / [(x - a)(√x + √a)] = 1 / (√x + √a)",
                )
                return f"1/(sqrt({var})+{sqrt_a})"

        # Add more rationalization patterns (e.g., in numerator)

        return None

    def _handle_limit_at_infinity(
        self, expr_tokens: list
    ) -> tuple[float | None, str | None]:
        """
        Handles limits of rational functions at infinity by comparing highest powers.
        SIMPLIFIED: Assumes a simple rational function structure.
        """
        # This requires identifying numerator and denominator and their degrees.
        # Very hard without an AST. We can try a regex approach for simple cases like P(x)/Q(x).
        expr_str = self.original_expression  # Use original for this crude check

        # Basic check for rational function structure (crude)
        if "/" not in expr_str or "^" not in expr_str:
            return (
                None,
                None,
            )  # Not easily identifiable as rational function for this method

        try:
            num_expr, den_expr = expr_str.split("/", 1)
            num_expr = num_expr.strip("() ")
            den_expr = den_expr.strip("() ")

            # Find highest power of variable in numerator and denominator (very crude regex)
            num_powers = [
                int(p)
                for p in re.findall(rf"{re.escape(self.variable)}\^(\d+)", num_expr)
            ]
            den_powers = [
                int(p)
                for p in re.findall(rf"{re.escape(self.variable)}\^(\d+)", den_expr)
            ]

            # Check for standalone variable term (power 1)
            if re.search(rf"(?<!\^)\b{re.escape(self.variable)}\b", num_expr):
                num_powers.append(1)
            if re.search(rf"(?<!\^)\b{re.escape(self.variable)}\b", den_expr):
                den_powers.append(1)

            num_degree = max(num_powers) if num_powers else 0
            den_degree = max(den_powers) if den_powers else 0

            # Find coefficients of highest power terms (extremely crude, likely fails often)
            num_coeff_match = (
                re.search(
                    rf"([\+\-]?\s*\d*\.?\d*)\*?{re.escape(self.variable)}\^{num_degree}",
                    num_expr,
                )
                if num_degree > 0
                else re.search(r"^([\+\-]?\s*\d+\.?\d*)", num_expr)
            )  # Constant term
            den_coeff_match = (
                re.search(
                    rf"([\+\-]?\s*\d*\.?\d*)\*?{re.escape(self.variable)}\^{den_degree}",
                    den_expr,
                )
                if den_degree > 0
                else re.search(r"^([\+\-]?\s*\d+\.?\d*)", den_expr)
            )  # Constant term

            num_lead_coeff = 1.0
            if num_coeff_match:
                coeff_str = num_coeff_match.group(1).replace(" ", "")
                if coeff_str == "+" or coeff_str == "":
                    num_lead_coeff = 1.0
                elif coeff_str == "-":
                    num_lead_coeff = -1.0
                else:
                    num_lead_coeff = float(coeff_str)
            elif num_degree == 0 and is_number(
                num_expr
            ):  # Check if numerator is just a number
                num_lead_coeff = float(num_expr)

            den_lead_coeff = 1.0
            if den_coeff_match:
                coeff_str = den_coeff_match.group(1).replace(" ", "")
                if coeff_str == "+" or coeff_str == "":
                    den_lead_coeff = 1.0
                elif coeff_str == "-":
                    den_lead_coeff = -1.0
                else:
                    den_lead_coeff = float(coeff_str)
            elif den_degree == 0 and is_number(
                den_expr
            ):  # Check if denominator is just a number
                den_lead_coeff = float(den_expr)

            if (
                den_lead_coeff == 0
            ):  # Avoid division by zero if denominator degree analysis wrong
                return None, "Could not determine leading coefficient of denominator."

            explanation = (
                f"Analyzing limit at {self.tending_to_str}. "
                f"Highest power in numerator ≈ {num_degree} (coeff ≈ {num_lead_coeff}). "
                f"Highest power in denominator ≈ {den_degree} (coeff ≈ {den_lead_coeff})."
            )

            result = None
            if num_degree == den_degree:
                result = num_lead_coeff / den_lead_coeff
                explanation += f" Degrees are equal, limit is ratio of leading coefficients: {result}."
            elif num_degree < den_degree:
                result = 0.0
                explanation += (
                    " Degree of numerator < degree of denominator, limit is 0."
                )
            else:  # num_degree > den_degree
                # Limit is inf or -inf, sign depends on coeffs and whether x -> inf or -inf
                sign = num_lead_coeff / den_lead_coeff
                power_diff_is_even = (num_degree - den_degree) % 2 == 0

                if self.limit_point == INF:
                    result = INF if sign > 0 else NEG_INF
                else:  # limit_point == NEG_INF
                    if power_diff_is_even:
                        result = INF if sign > 0 else NEG_INF  # (-ve)^even is +ve
                    else:
                        result = NEG_INF if sign > 0 else INF  # (-ve)^odd is -ve

                explanation += (
                    f" Degree of numerator > degree of denominator, limit is {result}."
                )

            return result, explanation

        except Exception as e:
            logger.warning(f"Error in limit at infinity analysis: {e}")
            return None, f"Failed to analyze dominant terms: {e}"

    # --- Main Calculation Method ---
    def calculate(self) -> Dict[str, Any]:
        self._add_step(
            "Initial Expression",
            f"lim ({self.original_expression}) as {self.variable} → {self.tending_to_str}",
        )

        try:
            self.tokens = tokenize(self.original_expression, self.variable)
            current_expr_str = (
                self.original_expression
            )  # Keep track of simplified expression string
            current_tokens = self.tokens

            # 1. Check Standard Limits First
            std_limit_val, std_limit_expl = self._check_standard_limits(
                current_expr_str, self.limit_point
            )
            if std_limit_val is not None:
                self._add_step("Standard Limit Found", std_limit_expl)
                self.result = str(std_limit_val)
                return {"steps": self.steps, "result": self.result}

            # 2. Try Direct Substitution
            self._add_step(
                "Attempting Direct Substitution",
                f"Evaluating expression at {self.variable} = {self.tending_to_str}",
            )
            direct_eval_result = self._safe_evaluate(current_tokens, self.limit_point)
            self._add_step(
                "Direct Substitution Result", f"Result: {direct_eval_result}"
            )

            form = self._analyze_form(direct_eval_result)
            self._add_step("Analysis of Result", f"The form is: {form}")

            if form == "Finite":
                self.result = str(direct_eval_result)
                return {"steps": self.steps, "result": self.result}
            elif form == "∞" or form == "-∞":
                self.result = "∞" if direct_eval_result > 0 else "-∞"
                return {"steps": self.steps, "result": self.result}

            # 3. Handle Indeterminate Forms
            while (
                form != "Finite" and form != "∞" and form != "-∞"
            ):  # Loop for potential simplification steps

                previous_expr_str = (
                    current_expr_str  # To detect if simplification occurred
                )

                if form == "0/0" or form == "∞/∞":
                    # Try Factorization (Simplified)
                    simplified_expr = self._try_factorization(current_expr_str)
                    if simplified_expr:
                        current_expr_str = simplified_expr
                        current_tokens = tokenize(current_expr_str, self.variable)
                        self._add_step(
                            "Re-evaluating after Factorization",
                            f"New expression: {current_expr_str}",
                        )
                        direct_eval_result = self._safe_evaluate(
                            current_tokens, self.limit_point
                        )
                        form = self._analyze_form(direct_eval_result)
                        self._add_step(
                            "Re-evaluation Result",
                            f"Result: {direct_eval_result}, Form: {form}",
                        )
                        continue  # Restart analysis with simplified expression

                    # Try Rationalization (Simplified)
                    rationalized_expr = self._try_rationalization(current_expr_str)
                    if rationalized_expr:
                        current_expr_str = rationalized_expr
                        current_tokens = tokenize(current_expr_str, self.variable)
                        self._add_step(
                            "Re-evaluating after Rationalization",
                            f"New expression: {current_expr_str}",
                        )
                        direct_eval_result = self._safe_evaluate(
                            current_tokens, self.limit_point
                        )
                        form = self._analyze_form(direct_eval_result)
                        self._add_step(
                            "Re-evaluation Result",
                            f"Result: {direct_eval_result}, Form: {form}",
                        )
                        continue  # Restart analysis

                    # Try L'Hôpital's Rule (Conceptual - No differentiation implemented)
                    self._add_step(
                        "L'Hôpital's Rule Suggestion",
                        "Form is 0/0 or ∞/∞. L'Hôpital's Rule might apply (requires differentiation).",
                    )
                    self._add_step(
                        "Limitation",
                        "Symbolic differentiation is not implemented in this version.",
                    )
                    # Cannot proceed further with L'Hopital here

                    # Try Limit at Infinity specific method if applicable
                    if self.limit_point == INF or self.limit_point == NEG_INF:
                        inf_lim_val, inf_lim_expl = self._handle_limit_at_infinity(
                            current_tokens
                        )
                        if inf_lim_val is not None:
                            self._add_step("Limit at Infinity Analysis", inf_lim_expl)
                            self.result = str(inf_lim_val)
                            return {"steps": self.steps, "result": self.result}
                        else:
                            self._add_step(
                                "Limit at Infinity Analysis",
                                "Could not determine limit using dominant terms analysis.",
                            )

                    # If no simplification worked, break the loop for 0/0 or inf/inf
                    self._add_step(
                        "Stuck on Indeterminate Form",
                        f"Could not simplify the expression further using available methods for {form}.",
                    )
                    break

                elif form == "1^∞":
                    # Apply lim f(x)^g(x) = exp(lim g(x) * (f(x) - 1))
                    self._add_step(
                        "Handling 1^∞ Form",
                        "Detected form 1^∞. Transforming using lim f(x)^g(x) = exp(lim [g(x) * (f(x) - 1)]).",
                    )
                    try:
                        parts = current_expr_str.split("^")
                        if len(parts) != 2:
                            raise ValueError(
                                "Expression not in base^exponent form for 1^inf"
                            )
                        base_expr = parts[0].strip("() ")
                        exp_expr = parts[1].strip("() ")

                        # Construct the new expression for the limit in the exponent
                        inner_limit_expr = f"({exp_expr}) * (({base_expr}) - 1)"
                        self._add_step(
                            "New Limit Calculation",
                            f"Need to calculate the limit of: {inner_limit_expr} as {self.variable} → {self.tending_to_str}",
                        )

                        # *** RECURSIVE CALL (or iterative approach) NEEDED HERE ***
                        # Create a new LimitCalculator instance for the inner limit
                        # inner_calculator = LimitCalculator(inner_limit_expr, self.variable, self.tending_to_str)
                        # inner_result_data = inner_calculator.calculate()
                        # self.steps.extend(inner_result_data['steps']) # Append steps from inner calculation

                        # For simplicity here, we'll just state the need, not implement recursion fully
                        self._add_step(
                            "Limitation",
                            "Calculating the inner limit recursively is required but not fully implemented here.",
                        )

                        # Placeholder: Assume inner limit 'L' was calculated
                        # inner_limit_result_str = inner_result_data['result']
                        # if inner_limit_result_str not in ["Could not determine limit", "DNE", "Undefined"]:
                        #      try:
                        #          inner_limit_val = float(inner_limit_result_str) # Or handle inf/-inf
                        #          final_result = math.exp(inner_limit_val)
                        #          self._add_step("Final Result (1^∞)", f"Inner limit L = {inner_limit_val}. Final result = e^L = {final_result}")
                        #          self.result = str(final_result)
                        #          return {"steps": self.steps, "result": self.result}
                        #      except ValueError:
                        #           self._add_step("Error", f"Could not convert inner limit result '{inner_limit_result_str}' to number.")
                        # else:
                        #      self._add_step("Result", "Could not determine the inner limit, so the original limit cannot be resolved this way.")

                        break  # Break after attempting 1^inf handling
                    except Exception as e:
                        logger.error(f"Error processing 1^inf form: {e}")
                        self._add_step("Error", f"Failed to process 1^∞ form: {e}")
                        break

                elif form == "0*∞" or form == "∞-∞":
                    # Suggest transformation
                    if form == "0*∞":
                        self._add_step(
                            "Handling 0*∞ Form",
                            "Detected form 0 * ∞. Try rewriting as f/(1/g) (form 0/0) or g/(1/f) (form ∞/∞).",
                        )
                    elif form == "∞-∞":
                        self._add_step(
                            "Handling ∞-∞ Form",
                            "Detected form ∞ - ∞. Try combining terms (e.g., common denominator, rationalization) to get a different form.",
                        )
                    self._add_step(
                        "Limitation",
                        "Symbolic transformation is not implemented in this version.",
                    )
                    break  # Cannot proceed further

                elif form == "0^0" or form == "∞^0":
                    self._add_step(
                        f"Handling {form} Form",
                        f"Detected indeterminate form {form}. Often requires rewriting using exponentials (f^g = exp(g*ln(f))) and evaluating the limit of the exponent.",
                    )
                    self._add_step(
                        "Limitation",
                        "Symbolic transformation is not implemented in this version.",
                    )
                    break  # Cannot proceed further

                else:  # Indeterminate (Unknown Type) or other issues
                    self._add_step(
                        "Indeterminate Form",
                        f"Could not resolve the indeterminate form '{form}' with available methods.",
                    )
                    break  # Cannot resolve

                # Check if simplification happened. If not, break loop to avoid infinite loop.
                if current_expr_str == previous_expr_str:
                    self._add_step(
                        "No Progress",
                        "No simplification method could be applied successfully in this step.",
                    )
                    break

            # 4. Numerical Check (Fallback/Verification) - Only if limit is finite and no result yet
            if (
                self.result == "Could not determine limit"
                and self.limit_point != INF
                and self.limit_point != NEG_INF
            ):
                self._add_step(
                    "Attempting Numerical Check",
                    f"Evaluating function near {self.variable} = {self.limit_point}",
                )
                try:
                    left_val = self._safe_evaluate(
                        self.tokens, self.limit_point - self.epsilon
                    )
                    right_val = self._safe_evaluate(
                        self.tokens, self.limit_point + self.epsilon
                    )
                    self._add_step(
                        "Numerical Evaluation",
                        f"f({self.limit_point - self.epsilon}) ≈ {left_val}, f({self.limit_point + self.epsilon}) ≈ {right_val}",
                    )

                    if (
                        not math.isnan(left_val)
                        and not math.isnan(right_val)
                        and abs(left_val - right_val) < 1e-5
                    ):  # Tolerance for numerical check
                        limit_approx = (left_val + right_val) / 2
                        # Refine approximation slightly
                        if abs(limit_approx) < 1e-9:
                            limit_approx = 0.0

                        # Check against direct eval if it was finite but perhaps failed form analysis initially
                        if (
                            form == "Finite"
                            and abs(direct_eval_result - limit_approx) < 1e-5
                        ):
                            self._add_step(
                                "Numerical Verification",
                                f"Numerical check agrees with direct substitution ({direct_eval_result}).",
                            )
                            self.result = str(direct_eval_result)
                        else:
                            self._add_step(
                                "Numerical Result",
                                f"Left and right evaluations are close. Approximated limit ≈ {limit_approx}",
                            )
                            self.result = str(
                                limit_approx
                            )  # Use numerical approximation
                    elif (
                        math.isinf(left_val)
                        and math.isinf(right_val)
                        and left_val == right_val
                    ):
                        self._add_step(
                            "Numerical Result",
                            f"Left and right evaluations both tend to {'∞' if left_val > 0 else '-∞'}.",
                        )
                        self.result = "∞" if left_val > 0 else "-∞"
                    else:
                        self._add_step(
                            "Numerical Result",
                            "Left and right evaluations differ significantly or failed. Limit likely Does Not Exist (DNE) or calculation failed.",
                        )
                        self.result = "DNE"
                except Exception as e:
                    logger.error(f"Error during numerical check: {e}")
                    self._add_step(
                        "Numerical Check Failed",
                        f"An error occurred during numerical evaluation: {e}",
                    )

        except ValueError as e:
            logger.error(f"Calculation Error: {e}")
            self._add_step("Error", f"Invalid input or calculation error: {e}")
            self.result = "Error"
        except Exception as e:
            logger.exception(
                "Unexpected error during limit calculation"
            )  # Log full traceback
            self._add_step("Error", f"An unexpected error occurred: {e}")
            self.result = "Error"

        # Final fallback if no method yielded a result
        if self.result == "Could not determine limit":
            # Check if numerical DNE was set
            found_dne = any(
                step["step"] == "Numerical Result" and "DNE" in step["explanation"]
                for step in self.steps
            )
            if not found_dne:
                self._add_step(
                    "Conclusion",
                    "Failed to determine the limit using implemented methods.",
                )
            else:
                self.result = "DNE"  # Keep DNE if found numerically

        return {"steps": self.steps, "result": self.result}


# --- API Endpoint ---


class LimitRequest(BaseModel):
    expression: str
    variable: str = "x"  # Default variable
    tending_to: str


class LimitResponse(BaseModel):
    steps: List[Dict[str, str]]
    result: str


@router.post("/calculate_limit", response_model=LimitResponse)
async def calculate_limit_endpoint(request: LimitRequest):
    """
    Calculates the limit of an expression step-by-step.

    - **expression**: The mathematical expression (e.g., `(x^2 - 1)/(x - 1)`, `sin(x)/x`, `(1 + 1/n)^n`). Use standard functions like `sin`, `cos`, `ln`, `log` (base 10), `exp`, `sqrt`. Use `pi` and `e` for constants. Use `inf` or `∞` for infinity. Ensure explicit multiplication with `*`.
    - **variable**: The variable in the expression (default: `x`).
    - **tending_to**: The value the variable approaches (e.g., `1`, `0`, `inf`, `-inf`, `pi/2`).
    """
    try:
        calculator = LimitCalculator(
            expression=request.expression,
            variable=request.variable,
            tending_to_str=request.tending_to,
        )
        result_data = calculator.calculate()
        return LimitResponse(**result_data)
    except ValueError as e:
        # Catch specific errors like invalid limit point
        logger.error(f"Input Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch unexpected errors during calculation
        logger.exception("Unhandled exception in /calculate_limit")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.get("/lim", include_in_schema=False)
async def root():
    return {"message": "Welcome to the Limit Calculator API. POST to /calculate_limit"}


# Example usage (you would run this with uvicorn: uvicorn main:app --reload)
# Example POST request body to /calculate_limit:
# {
#   "expression": "(x^2 - 4) / (x - 2)",
#   "variable": "x",
#   "tending_to": "2"
# }
#
# {
#   "expression": "sin(y)/y",
#   "variable": "y",
#   "tending_to": "0"
# }
#
# {
#   "expression": "(1 + 1/n)^n",
#   "variable": "n",
#   "tending_to": "inf"
# }
#
# {
#   "expression": "(3*x^2 + 2*x - 5) / (2*x^2 - x + 1)",
#   "variable": "x",
#   "tending_to": "inf"
# }
#
# {
#    "expression": "x * sin(1/x)",
#    "variable": "x",
#    "tending_to": "0"
# } # Might fail without Squeeze Theorem or proper 0*inf handling
#
# {
#    "expression": "(1 + x)^(1/x)",
#    "variable": "x",
#    "tending_to": "0"
# } # 1^inf form, needs recursive limit calc
