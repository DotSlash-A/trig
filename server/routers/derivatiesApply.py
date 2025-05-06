import math
from fastapi import FastAPI, HTTPException, Body, APIRouter
from typing import List, Dict, Any, Optional, Union  # Added Union
import sympy
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
from sympy import (
    sympify,  # Use sympify for safer parsing potentially
    symbols,
    limit,
    diff,
    solve,
    Eq,
    N,
    oo,
    zoo,  # Correct import for undefined/complex infinity
    nan,  # Import nan
    Interval as SympyInterval,
    FiniteSet,  # To handle solve results
    Union as SympyUnion,  # To handle solve results
    ConditionSet,  # To handle solve results
    EmptySet,
    solveset,
    Abs,
    atan,
    pi,
    idiff,
    Function,
    Derivative,
    simplify,  # Added simplify
    sqrt,  # Added sqrt if needed directly
    Piecewise,  # Added Piecewise
    Rel,  # For conditions in Piecewise
    And,  # For conditions in Piecewise
    latex,  # For potentially nicer equation output
    singularities,  # For finding undefined points
    Symbol,
)
from sympy.calculus.util import continuous_domain  # For finding domain issues
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanFunction
from sympy.core.relational import Relational

# Assuming your Pydantic models are in 'models/derivatives_model.py'
# Make sure this path is correct relative to where you run main.py
# If running main.py directly, it might be just 'models.py' if in the same folder
try:
    # Adjust this import based on your actual file structure
    from models.derivatives_model import *
except ImportError:
    # Fallback if the structure is different
    print(
        "Warning: Could not import models from 'models.derivatives_model'. Trying 'models'."
    )
    try:
        from models import *
    except ImportError:
        raise ImportError(
            "Could not find Pydantic models. Please ensure 'models.py' or 'models/derivatives_model.py' exists and is importable."
        )


# --- Helper Functions (Updated sympy_to_str, Existing safe_parse_expr, get_relevant_expr) ---


def safe_parse_expr(expr_str: str, local_dict: Optional[Dict[str, Any]] = None):
    """Safely parse a string into a sympy expression."""
    # Consider adding 'pi', 'E', common functions to local_dict by default if needed
    default_dict = {
        "pi": sympy.pi,
        "E": sympy.E,
        "sqrt": sympy.sqrt,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
        "log": sympy.log,
        "ln": sympy.log,
        "exp": sympy.exp,
        "Abs": sympy.Abs,
    }
    if local_dict:
        default_dict.update(local_dict)

    try:
        transformations = standard_transformations + (
            implicit_multiplication_application,
        )
        # Using sympify might be slightly safer for basic expressions but parse_expr allows transformations
        # parsed_expr = sympify(expr_str, locals=default_dict)
        parsed_expr = parse_expr(
            expr_str, local_dict=default_dict, transformations=transformations
        )
        return parsed_expr
    except (SyntaxError, TypeError, Exception) as e:
        # Add more specific error types if possible
        raise HTTPException(
            status_code=400, detail=f"Error parsing expression '{expr_str}': {e}"
        )


# def sympy_to_str(sympy_obj, precision=4):
#     """Convert sympy objects to string for JSON compatibility, with rounding."""
#     if sympy_obj is None:
#         return None
#     # Handle sets returned by solveset/singularities
#     if isinstance(sympy_obj, (FiniteSet, SympyUnion, ConditionSet, EmptySet)):
#         # Convert sets to lists of strings
#         if sympy_obj == EmptySet:
#             return []
#         elif isinstance(sympy_obj, FiniteSet):
#             return sorted(
#                 [sympy_to_str(item, precision) for item in sympy_obj.args],
#                 key=lambda x: float(x) if x.replace(".", "", 1).isdigit() else 0,
#             )  # Basic sort
#         else:  # Union, ConditionSet might be complex - return string representation
#             return str(sympy_obj)  # Or try to iterate args if possible
#     # Handle lists/tuples recursively
#     if isinstance(sympy_obj, (list, tuple)):
#         return [sympy_to_str(item, precision) for item in sympy_obj]
#     if isinstance(sympy_obj, dict):
#         return {str(k): sympy_to_str(v, precision) for k, v in sympy_obj.items()}

#     # Handle specific Sympy constants
#     if sympy_obj == oo:
#         return "oo"
#     if sympy_obj == -oo:
#         return "-oo"
#     if sympy_obj == zoo:
#         return "undefined (complex infinity)"  # More descriptive
#     if sympy_obj == nan:
#         return "NaN"
#     if sympy_obj == pi:
#         return "pi"
#     if sympy_obj == sympy.E:
#         return "E"
#     if sympy_obj == sympy.I:
#         return "I"

#     # Attempt to convert to float for rounding if it's a number type
#     if isinstance(sympy_obj, (sympy.Float, sympy.Integer, sympy.Rational)):
#         try:
#             num_val = float(sympy_obj)
#             # Format to specified precision, removing trailing zeros and decimal if whole
#             formatted = f"{num_val:.{precision}f}".rstrip("0").rstrip(".")
#             return formatted if formatted != "-0" else "0"  # Avoid '-0' representation
#         except (TypeError, ValueError):
#             # If conversion to float fails, return symbolic string
#             return str(sympy_obj)
#     # Handle NumberSymbol like pi, E (already covered above, but as fallback)
#     if isinstance(sympy_obj, sympy.NumberSymbol):
#         return str(sympy_obj)

#     # Handle symbols and expressions
#     if isinstance(sympy_obj, (sympy.Symbol, sympy.Expr)):
#         # Check if it represents a number that can be evaluated
#         try:
#             eval_val = N(
#                 sympy_obj, precision + 1
#             )  # Evaluate with slightly higher precision
#             if isinstance(eval_val, (sympy.Float, float)):
#                 num_val = float(eval_val)
#                 formatted = f"{num_val:.{precision}f}".rstrip("0").rstrip(".")
#                 return formatted if formatted != "-0" else "0"
#             else:
#                 # If evalf doesn't result in a float, return string
#                 return str(sympy_obj)
#         except (TypeError, ValueError, Exception):
#             # If evaluation fails, return the symbolic string representation
#             return str(sympy_obj)

#     # Default fallback for any other types
#     return str(sympy_obj)


# *** TEMPORARY SIMPLIFIED VERSION FOR DEBUGGING ***
def sympy_to_str(sympy_obj, precision=4):  # Keep signature for compatibility
    if sympy_obj is None:
        return None
    # Handle specific Sympy constants directly
    if sympy_obj == oo:
        return "oo"
    if sympy_obj == -oo:
        return "-oo"
    if sympy_obj == zoo:
        return "undefined (complex infinity)"
    if sympy_obj == nan:
        return "NaN"
    if sympy_obj == pi:
        return "pi"
    if sympy_obj == sympy.E:
        return "E"
    if sympy_obj == sympy.I:
        return "I"

    # Handle sets crudely for now
    if isinstance(sympy_obj, (FiniteSet, SympyUnion, ConditionSet, EmptySet)):
        if sympy_obj == EmptySet:
            return []
        # Basic representation for other sets
        return f"Set: {str(sympy_obj)}"  # Or try iterating args if simple FiniteSet

    # Basic list/tuple/dict handling (recursive call)
    if isinstance(sympy_obj, (list, tuple)):
        return [sympy_to_str(item) for item in sympy_obj]  # Recursive call
    if isinstance(sympy_obj, dict):
        return {str(k): sympy_to_str(v) for k, v in sympy_obj.items()}  # Recursive call

    # Fallback to default string conversion for everything else
    try:
        return str(sympy_obj)
    except Exception as e:
        return f"<Error converting object: {e}>"


def get_relevant_expr(
    func_def: FunctionDefinition, point_val: float, target: str = "point"
):
    """Get the correct expression piece for LHL, RHL, or point based on condition (Improved Heuristics)"""
    x = symbols(func_def.variable)
    local_dict = {func_def.variable: x}

    if func_def.type == "single":
        # Make sure expression is not None
        if func_def.expression is None:
            raise HTTPException(
                status_code=400,
                detail="Single function definition requires an 'expression'.",
            )
        return safe_parse_expr(func_def.expression, local_dict)

    if not func_def.pieces:
        raise HTTPException(
            status_code=400, detail="Piecewise function definition requires 'pieces'."
        )

    expr_for_target = None

    # --- Try using Sympy's Piecewise ---
    # This is generally more robust if conditions are well-formed sympy conditions
    pw_args = []
    symbols_in_conditions = {func_def.variable: x}
    has_complex_conditions = False
    for piece in func_def.pieces:
        try:
            expr = safe_parse_expr(piece.expression, local_dict)
            # Attempt to parse condition into a Sympy relational
            cond = sympify(piece.condition, locals=symbols_in_conditions)
            if not isinstance(cond, (BooleanFunction, Relational)):

                # If sympify doesn't return a boolean/relational, parsing might be wrong
                # Or it's a complex condition string like "0 < x < 1" which sympify might handle differently
                # Try parsing simple range conditions
                parts = piece.condition.replace(" ", "").split("<")
                if len(parts) == 3 and parts[1] == func_def.variable:  # e.g., a < x < b
                    cond = And(
                        sympify(parts[0], locals=symbols_in_conditions) < x,
                        x < sympify(parts[2], locals=symbols_in_conditions),
                    )
                elif len(parts) == 2:  # e.g. x < a or a < x
                    if parts[0] == func_def.variable:
                        cond = x < sympify(parts[1], locals=symbols_in_conditions)
                    elif parts[1] == func_def.variable:
                        cond = sympify(parts[0], locals=symbols_in_conditions) < x
                else:  # Fallback for other conditions (>=, <= etc.) - relies on sympify working
                    pass  # Use the result from sympify

            pw_args.append((expr, cond))
        except (SyntaxError, TypeError, Exception) as e:
            # print(f"Warning: Could not parse piece condition '{piece.condition}' reliably: {e}")
            has_complex_conditions = (
                True  # Fallback to string heuristics if parsing fails
            )
            break  # Stop trying Piecewise if one piece fails

    # --- Fallback Heuristics (If Piecewise fails or is complex) ---
    # This part remains less reliable than a perfect Piecewise definition
    if has_complex_conditions or not pw_args:
        # print("Using fallback heuristics for piece selection")
        # Find the most likely candidate based on target
        candidate_expr = None
        match_level = 0  # 0: none, 1: inequality only, 2: inequality+equality

        for piece in func_def.pieces:
            condition_str = piece.condition.replace(" ", "")
            expr = safe_parse_expr(piece.expression, local_dict)

            is_lhl_candidate = "<" in condition_str and str(point_val) in condition_str
            is_rhl_candidate = ">" in condition_str and str(point_val) in condition_str
            includes_equal = (
                "<=" in condition_str or ">=" in condition_str or "==" in condition_str
            )

            if target == "lhl":
                if is_lhl_candidate:
                    if (
                        includes_equal and match_level < 2
                    ):  # Prioritize '<=' over '<' if needed? Usually '<' is correct limit def.
                        candidate_expr = expr
                        match_level = 1
                    elif (
                        not includes_equal
                    ):  # Prefer strict inequality for limit direction
                        candidate_expr = expr
                        match_level = 1  # Keep '<' if found
            elif target == "rhl":
                if is_rhl_candidate:
                    if includes_equal and match_level < 2:
                        candidate_expr = expr
                        match_level = 1
                    elif not includes_equal:
                        candidate_expr = expr
                        match_level = 1
            elif target == "point":
                if includes_equal and str(point_val) in condition_str:
                    # Check if the point *exactly* satisfies the equality part implicitly
                    try_point = False
                    if "<=" in condition_str and point_val <= float(
                        condition_str.split("<=")[-1]
                    ):
                        try_point = True
                    elif ">=" in condition_str and point_val >= float(
                        condition_str.split(">=")[-1]
                    ):
                        try_point = True
                    elif "==" in condition_str and point_val == float(
                        condition_str.split("==")[-1]
                    ):
                        try_point = True

                    if try_point:
                        candidate_expr = expr
                        match_level = 2  # Strongest match
                        break  # Found equality match

        expr_for_target = candidate_expr

    else:
        # --- Use Piecewise to select ---
        # print("Using Piecewise function for selection")
        try:
            # Create the Piecewise function (add a default value? maybe zoo?)
            pw_func = Piecewise(
                *pw_args, (zoo, True)
            )  # Default to undefined if no condition matches

            if target == "point":
                expr_for_target = pw_func.subs(x, point_val)
                # If substitution results in zoo, it means no defined piece at that exact point
                if expr_for_target == zoo:
                    # Revert to heuristic for point if Piecewise doesn't define it explicitly
                    # print("Piecewise undefined at point, reverting to heuristic for 'point'")
                    return get_relevant_expr(
                        func_def, point_val, target="point"
                    )  # Recursive call with heuristic forced

                # Need the original expression *formula*, not the value, for limits/derivatives
                # Find which piece was active
                for expr, cond in pw_args:
                    if simplify(cond.subs(x, point_val)) == True:
                        expr_for_target = expr
                        break
            else:  # LHL / RHL - need the formula associated with the side
                # For limits, sympy's limit function handles Piecewise directly
                expr_for_target = (
                    pw_func  # Return the whole Piecewise for limit calculation
                )

        except Exception as e:
            # print(f"Error creating/using Piecewise function: {e}")
            # If Piecewise creation/use fails, fallback to heuristic
            return get_relevant_expr(
                func_def, point_val, target=target
            )  # Recursive call with heuristic

    if expr_for_target is None:
        # If still no expression found after heuristics
        # Provide a more informative error or a very basic default
        if func_def.pieces:  # Try first/last as absolute fallback
            if target == "lhl":
                expr_for_target = safe_parse_expr(
                    func_def.pieces[0].expression, local_dict
                )
            elif target == "rhl":
                expr_for_target = safe_parse_expr(
                    func_def.pieces[-1].expression, local_dict
                )
            else:
                expr_for_target = safe_parse_expr(
                    func_def.pieces[0].expression, local_dict
                )  # Guess first piece for point?
        if expr_for_target is None:  # If still nothing
            raise HTTPException(
                status_code=400,
                detail=f"Could not determine relevant function piece for {target} at {point_val}",
            )

    return expr_for_target


# --- Routers ---
# (Keep your existing router definitions)
router_continuity = APIRouter(prefix="/calculus/continuity", tags=["Continuity"])
router_differentiability = APIRouter(
    prefix="/calculus/differentiability", tags=["Differentiability"]
)
router_rate_measure = APIRouter(prefix="/calculus/rate-measure", tags=["Rate Measure"])
router_approximations = APIRouter(
    prefix="/calculus/approximations", tags=["Approximations & Errors"]
)
router_tangents_normals = APIRouter(
    prefix="/calculus/tangents-normals", tags=["Tangents & Normals"]
)
router_monotonicity = APIRouter(prefix="/calculus/monotonicity", tags=["Monotonicity"])


# --- Continuity Routes ---


# Existing check_continuity_at_point (Uses updated helpers)
@router_continuity.post("/check-at-point", response_model=ContinuityCheckResponse)
async def check_continuity_at_point(request: ContinuityCheckRequest):
    """Checks if a function is continuous at a given point."""
    x = symbols(request.function_definition.variable)
    point = request.point
    func_def = request.function_definition
    lhl, rhl, f_at_point = None, None, None
    reason = []

    try:
        # Get expressions using the potentially improved helper
        # Limit function handles Piecewise directly if returned by get_relevant_expr
        lhl_expr_base = get_relevant_expr(func_def, point, "lhl")
        rhl_expr_base = get_relevant_expr(func_def, point, "rhl")
        f_expr = get_relevant_expr(
            func_def, point, "point"
        )  # This should return the specific piece's formula

        # Calculate f(a) using the specific piece's formula
        try:
            if isinstance(
                f_expr, Piecewise
            ):  # If point logic returned Piecewise, need value
                f_at_point = f_expr.subs(x, point)
            else:  # It should be a single expression
                f_at_point = f_expr.subs(x, point)

            if f_at_point.has(oo, -oo, zoo, nan):
                f_at_point = zoo  # Standardize undefined/infinite to zoo
                reason.append(f"Function is undefined or infinite at x={point}.")
            # Allow real/complex results otherwise
        except (TypeError, ValueError, Exception) as e:
            f_at_point = zoo
            reason.append(f"Could not evaluate function f({point}): {e}")

        # Calculate LHL (limit handles Piecewise)
        try:
            lhl = limit(lhl_expr_base, x, point, dir="-")
            if lhl.has(oo, -oo, zoo, nan):
                reason.append(f"LHL is infinite or undefined.")
                lhl = zoo  # Standardize
        except Exception as e:
            lhl = zoo
            reason.append(f"Could not compute LHL: {e}")

        # Calculate RHL (limit handles Piecewise)
        try:
            rhl = limit(rhl_expr_base, x, point, dir="+")
            if rhl.has(oo, -oo, zoo, nan):
                reason.append(f"RHL is infinite or undefined.")
                rhl = zoo  # Standardize
        except Exception as e:
            rhl = zoo
            reason.append(f"Could not compute RHL: {e}")

        # Check conditions (using zoo for undefined/infinite states)
        is_cont = False
        limit_exists = False
        limit_value = None

        if lhl == zoo or rhl == zoo:
            reason.append(
                "Limit does not exist (LHL or RHL calculation failed or is infinite/undefined)."
            )
        elif lhl != rhl:
            reason.append(
                f"Limit does not exist (LHL={sympy_to_str(lhl)} != RHL={sympy_to_str(rhl)})."
            )
        else:  # LHL == RHL and finite
            limit_exists = True
            limit_value = lhl  # or rhl
            reason.append(f"Limit exists and equals {sympy_to_str(limit_value)}.")

        if f_at_point == zoo:
            reason.append(f"f({point}) is undefined/infinite.")
        elif not limit_exists:
            pass  # Reason already added above
        elif limit_value != f_at_point:
            reason.append(
                f"Limit ({sympy_to_str(limit_value)}) exists but is not equal to f({point}) ({sympy_to_str(f_at_point)})."
            )
        else:
            # All conditions met: f(a) defined, limit exists, limit = f(a)
            is_cont = True
            # Overwrite reason for success case
            reason = [
                f"LHL ({sympy_to_str(lhl)}) = RHL ({sympy_to_str(rhl)}) = f({point}) ({sympy_to_str(f_at_point)}). Function is continuous."
            ]

        return ContinuityCheckResponse(
            point=point,
            lhl=sympy_to_str(lhl),
            rhl=sympy_to_str(rhl),
            f_at_point=sympy_to_str(f_at_point),
            is_continuous=is_cont,
            reason=" ".join(reason),
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred in continuity check: {e}",
        )


# NEW: find-constants (Continuity)
@router_continuity.post("/find-constants", response_model=ContinuityConstantsResponse)
async def find_continuity_constants(request: ContinuityConstantsRequest):
    """Finds constants that make a piecewise function continuous at a point."""
    x = symbols(request.function_definition.variable)
    point = request.point
    func_def = request.function_definition

    if func_def.type != "piecewise":
        raise HTTPException(
            status_code=400,
            detail="This endpoint requires a piecewise function definition.",
        )
    if not request.constants_to_find:
        raise HTTPException(
            status_code=400, detail="List 'constants_to_find' cannot be empty."
        )

    constants = symbols(request.constants_to_find)
    # Ensure constants are treated as symbols even if input list is single string
    if not isinstance(constants, (tuple, list)):
        constants = (constants,)

    local_dict = {func_def.variable: x}
    local_dict.update({c.name: c for c in constants})

    lhl, rhl, f_at_point = None, None, None
    eqns_derived = []
    solutions = None
    is_possible = False
    reason = []

    try:
        # Get expressions - these might contain the constants
        # Need to pass the local_dict including constants to the parser
        lhl_expr_base = get_relevant_expr(func_def, point, "lhl")
        rhl_expr_base = get_relevant_expr(func_def, point, "rhl")
        f_expr = get_relevant_expr(func_def, point, "point")

        # Calculate limits and f(a) symbolically
        lhl = limit(lhl_expr_base, x, point, dir="-")
        rhl = limit(rhl_expr_base, x, point, dir="+")
        f_at_point = f_expr.subs(x, point)

        # Check for undefined/infinite limits (if they don't involve constants)
        if any(
            val.has(oo, -oo, zoo, nan)
            for val in [lhl, rhl, f_at_point]
            if val.is_constant()
        ):
            raise HTTPException(
                status_code=400,
                detail="Function pieces lead to undefined/infinite values at the point, cannot enforce continuity.",
            )

        # Create equations: LHL = RHL and RHL = f(a)
        # Simplify expressions before creating equations
        eq1 = Eq(simplify(lhl), simplify(rhl))
        eq2 = Eq(simplify(rhl), simplify(f_at_point))

        # Store symbolic equations as strings (maybe use latex for nicer display?)
        eqns_derived_str = [sympy_to_str(eq1), sympy_to_str(eq2)]
        simplified_eqns_str = [latex(eq1), latex(eq2)]  # Example using LaTeX

        # Solve the system
        system = [eq1, eq2]
        try:
            # `solve` is generally good for algebraic systems
            sol = solve(system, constants)

            if isinstance(sol, dict) and sol:
                # Check if all requested constants are in the solution
                if all(c in sol for c in constants):
                    solutions = {
                        sympy_to_str(k): sympy_to_str(v) for k, v in sol.items()
                    }
                    is_possible = True
                    reason.append("Unique solution found.")
                else:
                    is_possible = False
                    reason.append(
                        f"Solver found values for some constants but not all requested: {sympy_to_str(list(sol.keys()))}"
                    )

            elif (
                isinstance(sol, list) and sol
            ):  # Potentially multiple solutions or dependent solutions
                # Handle case where solution might be [{a: 1, b: 1}]
                if isinstance(sol[0], dict):
                    if all(c in sol[0] for c in constants):
                        solutions = {
                            sympy_to_str(k): sympy_to_str(v) for k, v in sol[0].items()
                        }
                        is_possible = True
                        reason.append(
                            "Solution found (potentially one of multiple). Taking the first one."
                        )
                    else:
                        is_possible = False
                        reason.append(
                            f"Solver found solution set, but first solution doesn't contain all constants: {sympy_to_str(sol[0])}"
                        )
                else:  # Solution might be in terms of other constants, e.g. [(k, 1)]
                    # This case is harder to parse into the desired dict format robustly
                    is_possible = False
                    reason.append(
                        f"Solver returned a list format not easily convertible to key-value pairs: {sympy_to_str(sol)}"
                    )

            elif not sol:  # Empty list or dict means no solution
                is_possible = False
                reason.append(
                    "No solution found for the constants satisfying the continuity conditions."
                )
            else:
                is_possible = False
                reason.append(
                    f"Solver returned an unexpected result type: {type(sol)}. Cannot determine solution."
                )

        except NotImplementedError:
            is_possible = False
            reason.append(
                "Sympy solver reported that the system is not solvable with its methods."
            )
        except Exception as solve_e:
            is_possible = False
            reason.append(f"Error during solving the system of equations: {solve_e}")

        return ContinuityConstantsResponse(
            point=point,
            conditions_for_continuity=["LHL == RHL", "RHL == f(point)"],
            equations_derived=eqns_derived_str,
            simplified_equations=simplified_eqns_str,
            solutions=solutions,
            is_possible=is_possible,
            reason=" ".join(reason),
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred in find continuity constants: {e}",
        )


# NEW: check-interval (Continuity) - Basic Implementation
@router_continuity.post("/check-interval", response_model=ContinuityIntervalResponse)
async def check_continuity_interval(request: ContinuityIntervalRequest):
    """
    Checks continuity over an interval.
    NOTE: This is a basic implementation, primarily checking for points where
    denominators are zero or piecewise boundaries within the interval.
    It does NOT perform a full rigorous check for all function types (e.g., log, sqrt domains).
    """
    x = symbols(request.function_definition.variable)
    func_def = request.function_definition
    interval_req = request.interval

    # Convert interval request to Sympy interval
    try:
        start = -oo if interval_req.start == "-oo" else float(interval_req.start)
        end = oo if interval_req.end == "oo" else float(interval_req.end)
        # Sympy Interval uses left_open, right_open
        sympy_interval = SympyInterval(
            start,
            end,
            left_open=not interval_req.start_inclusive,
            right_open=not interval_req.end_inclusive,
        )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid interval start/end values. Use numbers or '-oo', 'oo'.",
        )

    potential_discontinuities = set()
    points_of_discontinuity_found = []
    overall_continuous = True  # Assume true initially

    try:
        if func_def.type == "single":
            if func_def.expression is None:
                raise HTTPException(
                    status_code=400, detail="Single func needs expression"
                )
            expr = safe_parse_expr(func_def.expression, {func_def.variable: x})

            # 1. Find singularities (e.g., division by zero) using sympy
            try:
                # singularities requires the variable and the interval
                singular_points = singularities(expr, x, domain=sympy_interval)
                if singular_points != EmptySet:
                    potential_discontinuities.update(
                        list(
                            singular_points.args
                            if isinstance(singular_points, FiniteSet)
                            else []
                        )
                    )
            except Exception as sing_e:
                # print(f"Could not compute singularities for {expr}: {sing_e}")
                pass  # Ignore if singularity finding fails

            # 2. Check domain for common functions (basic checks)
            # This part is complex for a general solution. Example for rational:
            if expr.is_rational_function():
                den = sympy.denom(expr)
                zeros = solveset(den, x, domain=sympy_interval)
                if zeros != EmptySet:
                    potential_discontinuities.update(
                        list(zeros.args if isinstance(zeros, FiniteSet) else [])
                    )

            # Add more checks here for log(arg), sqrt(arg), etc. if needed
            # e.g., for log(g(x)), find where g(x) <= 0 within the interval
            # e.g., for sqrt(g(x)), find where g(x) < 0 within the interval

        elif func_def.type == "piecewise":
            if not func_def.pieces:
                raise HTTPException(
                    status_code=400, detail="Piecewise func needs pieces"
                )
            # 1. Check boundaries of pieces that fall *within* the interval
            boundary_points = set()
            for piece in func_def.pieces:
                # Extract numbers from conditions (very basic parsing)
                import re

                nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", piece.condition)
                for num_str in nums:
                    try:
                        num = float(num_str)
                        # Check if the boundary point is strictly inside the interval
                        if (
                            sympy_interval.contains(num)
                            and num != sympy_interval.start
                            and num != sympy_interval.end
                        ):
                            boundary_points.add(num)
                    except ValueError:
                        pass  # Ignore non-numeric parts
            potential_discontinuities.update(boundary_points)

            # 2. Check for singularities *within each piece's applicable domain*
            # This requires parsing conditions properly - skipping for this basic version
            # You would need to find singularities of piece.expression within the intersection
            # of sympy_interval and the domain defined by piece.condition.

        # Check each potential discontinuity point
        sorted_potential_points = sorted(list(potential_discontinuities))

        for point in sorted_potential_points:
            # Use the point checker, but handle potential exceptions locally
            try:
                point_check_req = ContinuityCheckRequest(
                    function_definition=func_def, point=float(point)
                )
                point_check_resp = await check_continuity_at_point(point_check_req)
                if not point_check_resp.is_continuous:
                    overall_continuous = False
                    points_of_discontinuity_found.append(
                        DiscontinuityInfo(
                            point=sympy_to_str(point),
                            reason=point_check_resp.reason,
                            lhl=point_check_resp.lhl,
                            rhl=point_check_resp.rhl,
                            f_at_point=point_check_resp.f_at_point,
                        )
                    )
            except Exception as check_e:
                # If checking the point fails, assume discontinuity there
                overall_continuous = False
                points_of_discontinuity_found.append(
                    DiscontinuityInfo(
                        point=sympy_to_str(point),
                        reason=f"Failed to check continuity at this point: {check_e}",
                    )
                )

        # Construct continuity intervals (basic)
        # TODO: This needs refinement based on discontinuity points
        continuity_intervals_list = []
        if overall_continuous:
            continuity_intervals_list.append(interval_req.dict())
        else:
            # Very basic split - assumes discontinuities break interval cleanly
            last_point = interval_req.start
            last_inclusive = interval_req.start_inclusive
            for disc_info in sorted(
                points_of_discontinuity_found, key=lambda p: float(p.point)
            ):
                disc_point = float(disc_info.point)
                if (
                    disc_point > last_point if isinstance(last_point, float) else True
                ):  # Avoid empty interval
                    continuity_intervals_list.append(
                        {
                            "start": last_point,
                            "end": disc_point,
                            "start_inclusive": last_inclusive,
                            "end_inclusive": False,
                        }
                    )
                last_point = disc_point
                last_inclusive = False  # Point of discontinuity is excluded
            # Add final interval
            if (
                interval_req.end > last_point
                if isinstance(interval_req.end, float) and isinstance(last_point, float)
                else True
            ):
                continuity_intervals_list.append(
                    {
                        "start": last_point,
                        "end": interval_req.end,
                        "start_inclusive": False,
                        "end_inclusive": interval_req.end_inclusive,
                    }
                )

        return ContinuityIntervalResponse(
            interval_checked=interval_req.dict(),
            is_continuous_overall=overall_continuous,
            potential_discontinuities_checked=sympy_to_str(sorted_potential_points),
            points_of_discontinuity_found=points_of_discontinuity_found,
            continuity_intervals=continuity_intervals_list,  # Provide the calculated intervals
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred in check continuity interval: {e}",
        )


# --- Differentiability Routes ---


# # Modified check_differentiability_at_point
# @router_differentiability.post(
#     "/check-at-point", response_model=DifferentiabilityCheckResponse
# )
# async def check_differentiability_at_point(request: DifferentiabilityCheckRequest):
#     """Checks if a function is differentiable at a given point."""
#     x = symbols(request.function_definition.variable)
#     h = symbols("h", real=True, positive=True)  # Small positive change for RHD
#     point = request.point
#     func_def = request.function_definition
#     lhd, rhd = None, None
#     is_differentiable = False
#     derivative_value = None
#     reason = []

#     # 1. Check Continuity First
#     continuity_req = ContinuityCheckRequest(function_definition=func_def, point=point)
#     try:
#         continuity_resp = await check_continuity_at_point(continuity_req)
#     except HTTPException as he:
#         # If continuity check itself failed badly
#         # Need to construct a minimal ContinuityCheckResponse for the return type
#         failed_cont_resp = ContinuityCheckResponse(
#             point=point,
#             reason=f"Continuity check failed: {he.detail}",
#             is_continuous=False,
#         )
#         return DifferentiabilityCheckResponse(
#             point=point,
#             continuity_check=failed_cont_resp,
#             is_differentiable=False,
#             reason=f"Cannot check differentiability because continuity check failed: {he.detail}",
#         )

#     if not continuity_resp.is_continuous:
#         return DifferentiabilityCheckResponse(
#             point=point,
#             continuity_check=continuity_resp,
#             lhd=None,
#             rhd=None,
#             is_differentiable=False,
#             derivative_value=None,
#             reason=f"Function is not continuous at x={point}, therefore not differentiable. {continuity_resp.reason}",
#         )

#     # 2. If Continuous, check LHD and RHD
#     try:
#         # Get expressions. Limit approach is more general than differentiating pieces.
#         lhl_expr_base = get_relevant_expr(
#             func_def, point, "lhl"
#         )  # Expression for x < point
#         rhl_expr_base = get_relevant_expr(
#             func_def, point, "rhl"
#         )  # Expression for x > point
#         f_expr = get_relevant_expr(func_def, point, "point")  # Expression at x = point

#         # Ensure f(a) is finite for derivative calculation
#         f_at_point = f_expr.subs(x, point)
#         if f_at_point.has(oo, -oo, zoo, nan):
#             # Should have been caught by continuity check, but double check
#             return DifferentiabilityCheckResponse(
#                 point=point,
#                 continuity_check=continuity_resp,
#                 is_differentiable=False,
#                 reason=f"Function value f({point}) is undefined/infinite, cannot compute derivatives.",
#             )

#         # LHD: limit h->0+ of (f(a) - f(a-h)) / h --- This uses the LHL expression base
#         # Alternative: limit h->0- of (f(a+h)-f(a))/h
#         lhd_limit_expr = (f_expr.subs(x, point) - lhl_expr_base.subs(x, point - h)) / h
#         lhd = limit(lhd_limit_expr, h, 0, dir="+")  # Limit h->0+

#         # RHD: limit h->0+ of (f(a+h) - f(a)) / h --- This uses the RHL expression base
#         rhd_limit_expr = (rhl_expr_base.subs(x, point + h) - f_expr.subs(x, point)) / h
#         rhd = limit(rhd_limit_expr, h, 0, dir="+")  # Limit h->0+

#         # Check LHD/RHD results (standardize to zoo)
#         if lhd.has(oo, -oo, zoo, nan):
#             lhd = zoo
#         if rhd.has(oo, -oo, zoo, nan):
#             rhd = zoo

#         if lhd == zoo or rhd == zoo:
#             reason.append("LHD or RHD calculation failed or is infinite/undefined.")
#             is_differentiable = False
#         elif lhd != rhd:
#             reason.append(
#                 f"Function is continuous, but LHD ({sympy_to_str(lhd)}) != RHD ({sympy_to_str(rhd)})."
#             )
#             is_differentiable = False
#         else:
#             # LHD == RHD and finite
#             is_differentiable = True
#             derivative_value = lhd  # or rhd
#             # Overwrite reason for success
#             reason = [
#                 f"Function is continuous and LHD ({sympy_to_str(lhd)}) = RHD ({sympy_to_str(rhd)}). Differentiable."
#             ]

#         return DifferentiabilityCheckResponse(
#             point=point,
#             continuity_check=continuity_resp,
#             lhd=sympy_to_str(lhd),
#             rhd=sympy_to_str(rhd),
#             is_differentiable=is_differentiable,
#             derivative_value=(
#                 sympy_to_str(derivative_value) if is_differentiable else None
#             ),
#             reason=" ".join(reason),
#         )

#     except HTTPException as he:
#         return DifferentiabilityCheckResponse(
#             point=point,
#             continuity_check=continuity_resp,
#             is_differentiable=False,
#             reason=f"Failed during differentiability check setup: {he.detail}",
#         )
#     except Exception as e:
#         return DifferentiabilityCheckResponse(
#             point=point,
#             continuity_check=continuity_resp,
#             is_differentiable=False,
#             reason=f"An unexpected error occurred during LHD/RHD calculation: {e}",
#         )


@router_continuity.post("/check-at-point", response_model=ContinuityCheckResponse)
async def check_continuity_at_point(request: ContinuityCheckRequest):
    """Checks if a function is continuous at a given point."""
    x = symbols(request.function_definition.variable)
    point = request.point
    func_def = request.function_definition
    lhl, rhl, f_at_point = None, None, None
    lhl_expr, rhl_expr, f_expr = None, None, None
    reason = []

    try:
        # Determine expressions for LHL, RHL, f(a)
        if func_def.type == "single":
            expr = safe_parse_expr(func_def.expression, {func_def.variable: x})
            lhl_expr, rhl_expr, f_expr = expr, expr, expr
        elif func_def.type == "piecewise":
            # This condition parsing is heuristic and may need refinement
            lhl_expr = get_relevant_expr(func_def, point, "lhl")
            rhl_expr = get_relevant_expr(func_def, point, "rhl")
            f_expr = get_relevant_expr(func_def, point, "point")
        else:
            raise HTTPException(
                status_code=400, detail="Invalid function definition type"
            )

        # Calculate f(a)
        try:
            f_at_point = f_expr.subs(x, point)
            # Check for undefined results like 1/0 -> zoo
            if f_at_point.has(oo, -oo, sympy.zoo, sympy.nan):
                f_at_point = sympy.zoo
                reason.append(f"Function is undefined at x={point}.")
            elif not f_at_point.is_real:  # Handle complex results if not expected
                pass  # Allow complex numbers for now

        except (TypeError, ValueError, Exception) as e:
            # Could be undefined (e.g., log(-1)), division by zero handled above
            f_at_point = sympy.zoo
            reason.append(f"Could not evaluate function at x={point}: {e}")

        # Calculate LHL
        try:
            lhl = limit(lhl_expr, x, point, dir="-")
            if not lhl.is_real:  # Allow complex numbers but note if infinite/nan
                if lhl.has(oo, -oo, sympy.zoo, sympy.nan):
                    reason.append(f"LHL is infinite or undefined.")
        except Exception as e:
            lhl = sympy.zoo  # Indicate limit computation failed
            reason.append(f"Could not compute LHL: {e}")

        # Calculate RHL
        try:
            rhl = limit(rhl_expr, x, point, dir="+")
            if not rhl.is_real:
                if rhl.has(oo, -oo, sympy.zoo, sympy.nan):
                    reason.append(f"RHL is infinite or undefined.")
        except Exception as e:
            rhl = sympy.zoo  # Indicate limit computation failed
            reason.append(f"Could not compute RHL: {e}")

        # Check conditions
        is_cont = False
        if f_at_point == sympy.zoo:
            reason.append(f"f({point}) is undefined.")
        elif lhl == sympy.zoo or rhl == sympy.zoo:
            reason.append(
                "Limit does not exist (LHL or RHL calculation failed or is undefined/infinite)."
            )
        elif lhl != rhl:
            reason.append(
                f"Limit does not exist (LHL={sympy_to_str(lhl)} != RHL={sympy_to_str(rhl)})."
            )
        elif lhl != f_at_point:  # If LHL==RHL, check against f(a)
            reason.append(
                f"Limit ({sympy_to_str(lhl)}) exists but is not equal to f({point}) ({sympy_to_str(f_at_point)})."
            )
        else:
            # All conditions met
            is_cont = True
            reason.append(
                f"LHL ({sympy_to_str(lhl)}) = RHL ({sympy_to_str(rhl)}) = f({point}) ({sympy_to_str(f_at_point)}). Function is continuous."
            )

        return ContinuityCheckResponse(
            point=point,
            lhl=sympy_to_str(lhl),
            rhl=sympy_to_str(rhl),
            f_at_point=sympy_to_str(f_at_point),
            is_continuous=is_cont,
            reason=" ".join(reason),
        )

    except HTTPException as he:
        raise he  # Re-raise HTTP exceptions
    except Exception as e:
        # Catch-all for other unexpected errors during setup/parsing
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


# NEW: find-constants (Differentiability) - Placeholder
@router_differentiability.post(
    "/find-constants", response_model=DifferentiabilityConstantsResponse
)
async def find_differentiability_constants(request: DifferentiabilityConstantsRequest):
    """
    Finds constants that make a piecewise function differentiable at a point.
    NOTE: Placeholder implementation. Requires solving continuity AND LHD=RHD equations.
    """
    # TODO: Implement the full logic:
    # 1. Set up and solve continuity equations (LHL=RHL=f(a)) like in the continuity endpoint.
    # 2. Calculate symbolic LHD and RHD (e.g., limit definition or differentiate pieces).
    # 3. Set up LHD = RHD equation.
    # 4. Solve the combined system of equations (from continuity and differentiability).
    raise HTTPException(
        status_code=501,
        detail="Finding constants for differentiability not fully implemented yet.",
    )


# --- Rate Measure Routes ---


# NEW: direct-rate
@router_rate_measure.post("/direct-rate", response_model=DirectRateResponse)
async def calculate_direct_rate(request: DirectRateRequest):
    """Calculates the direct rate of change dy/dx at a point."""
    try:
        # Assume function_str is the expression for the dependent variable
        # e.g., for "A = pi * r**2", function_str should be "pi * r**2"
        # Alternatively, parse the equation to extract expression if needed.
        ind_var = symbols(request.independent_var)
        dep_var = symbols(request.dependent_var)  # Used mainly for output description

        local_dict = {
            request.independent_var: ind_var,
            "pi": pi,
            "E": sympy.E,
        }  # Add common constants

        # Handle potential equation format "A = expr" vs just "expr"
        expr_str = request.function_str
        if "=" in expr_str:
            parts = expr_str.split("=", 1)
            # Basic check if dependent var matches left side
            if parts[0].strip() == request.dependent_var:
                expr_str = parts[1].strip()
            else:
                # Assume the whole string is the expression if '=' is present but doesn't match dep_var
                pass  # Could raise error or just try parsing whole string

        expr = safe_parse_expr(expr_str, local_dict)

        # Calculate derivative
        derivative_expr = diff(expr, ind_var)

        # Evaluate derivative at the point
        point_value = request.point.get(request.independent_var)
        if point_value is None:
            raise HTTPException(
                status_code=400,
                detail=f"Point value for independent variable '{request.independent_var}' not provided.",
            )

        rate_at_point = derivative_expr.subs(ind_var, point_value)

        # Construct derivative expression string (e.g., dA/dr = ...)
        deriv_str = f"d({request.dependent_var})/d({request.independent_var}) = {latex(derivative_expr)}"  # Use LaTeX

        return DirectRateResponse(
            derivative_expression=deriv_str, rate_at_point=sympy_to_str(rate_at_point)
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to calculate direct rate: {e}"
        )


# NEW: related-rates - Placeholder
@router_rate_measure.post("/related-rates", response_model=RelatedRatesResponse)
async def solve_related_rates(request: RelatedRatesRequest):
    """
    Solves a related rates problem.
    NOTE: Placeholder implementation. Requires complex symbolic manipulation.
    """
    # TODO: Implement the full logic:
    # 1. Parse equation_str.
    # 2. Define variables as sympy Functions of time 't'.
    # 3. Substitute functions into equation.
    # 4. Differentiate equation wrt 't'.
    # 5. Create symbols for known/target rates (e.g., dx_dt = symbols('dx_dt')).
    # 6. Substitute rate symbols for Derivative objects (e.g., Derivative(x(t), t) -> dx_dt).
    # 7. Solve the differentiated algebraic equation for the target rate symbol.
    # 8. Substitute known rate values and instance values.
    # 9. Solve original equation for missing instance values if necessary.
    raise HTTPException(
        status_code=501, detail="Related Rates endpoint not fully implemented yet."
    )


# --- Approximations & Errors Routes ---


# NEW: find-differential
@router_approximations.post(
    "/find-differential", response_model=FindDifferentialResponse
)
async def find_differential(request: FindDifferentialRequest):
    """Calculates the differential dy = f'(x) dx."""
    try:
        x = symbols(request.variable)
        dx_sym = symbols("d" + request.variable)  # Symbolic dx, e.g., 'dx'

        local_dict = {request.variable: x, "pi": pi, "E": sympy.E}

        # Handle "y = expr" vs "expr"
        f_expr_str = request.function_str
        if "=" in f_expr_str:
            parts = f_expr_str.split("=", 1)
            # Basic check if dependent var 'y' (or similar) is on left
            if parts[0].strip().lower() == "y":  # Simple check
                f_expr_str = parts[1].strip()

        f_expr = safe_parse_expr(f_expr_str, local_dict)

        # Derivative
        f_prime_expr = diff(f_expr, x)

        # Differential formula dy = f'(x) dx
        dy_formula_expr = f_prime_expr * dx_sym
        dy_formula_str = f"dy = ({latex(f_prime_expr)}) * {latex(dx_sym)}"  # Use latex

        # Value of derivative at x_value
        f_prime_at_x = f_prime_expr.subs(x, request.x_value)

        # Value of differential
        dy_value = f_prime_at_x * request.dx_value  # Use numeric dx_value

        return FindDifferentialResponse(
            derivative_f_prime_x=latex(f_prime_expr),
            derivative_at_x_value=sympy_to_str(f_prime_at_x),
            differential_dy_formula=dy_formula_str,
            differential_dy_value=sympy_to_str(dy_value),
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to find differential: {e}")


# Existing approximate-value (adapted slightly for consistency)
@router_approximations.post(
    "/approximate-value", response_model=ApproximateValueResponse
)
async def approximate_value(request: ApproximateValueRequest):
    """Approximates a function value using tangent line (differentials)."""
    x = symbols(request.variable)
    try:
        local_dict = {request.variable: x, "pi": pi, "E": sympy.E, "sqrt": sqrt}
        f_expr = safe_parse_expr(request.function_str, local_dict)
        f_prime_expr = diff(f_expr, x)

        base_x = request.base_x
        target_x = request.target_x
        delta_x = target_x - base_x  # This is 'dx' in the context of approximation

        f_base_val = f_expr.subs(x, base_x)
        f_prime_base_val = f_prime_expr.subs(x, base_x)

        # Check for undefined values
        if f_base_val.has(oo, -oo, zoo, nan) or f_prime_base_val.has(oo, -oo, zoo, nan):
            raise HTTPException(
                status_code=400,
                detail=f"Function or its derivative is undefined/infinite at base_x={base_x}.",
            )

        dy = f_prime_base_val * delta_x  # Approximate change in y
        approx_val = f_base_val + dy

        formula = f"f({target_x}) ≈ f({base_x}) + f'({base_x}) * ({delta_x})"

        return ApproximateValueResponse(
            base_x=base_x,
            target_x=target_x,
            delta_x=delta_x,
            f_base_x=sympy_to_str(f_base_val),
            derivative_f_prime_x=latex(f_prime_expr),  # Use LaTeX
            f_prime_base_x=sympy_to_str(f_prime_base_val),
            differential_dy=sympy_to_str(dy),
            approximation_formula=formula,
            approximate_value_symbolic=sympy_to_str(
                approx_val
            ),  # Keep symbolic if possible
            approximate_value_numeric=(
                sympy_to_str(N(approx_val))
                if not approx_val.has(oo, -oo, zoo, nan)
                else "undefined"
            ),
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Approximation calculation failed: {e}"
        )


# NEW: calculate-errors
@router_approximations.post("/calculate-errors", response_model=CalculateErrorsResponse)
async def calculate_approximate_errors(request: CalculateErrorsRequest):
    """Calculates approximate absolute, relative, and percentage errors using differentials."""
    try:
        x = symbols(request.variable)  # The variable with measurement error
        dx = request.possible_error_dx  # The error value
        measured_x = request.measured_value

        local_dict = {request.variable: x, "pi": pi, "E": sympy.E}

        # Handle "Y = expr" vs "expr"
        f_expr_str = request.function_str
        dep_var_name = "y"  # Default dependent variable name
        if "=" in f_expr_str:
            parts = f_expr_str.split("=", 1)
            dep_var_name = parts[0].strip()  # Get dependent var name from LHS
            f_expr_str = parts[1].strip()

        f_expr = safe_parse_expr(f_expr_str, local_dict)  # This is y = f(x)

        # Value of function at measured value: y = f(measured_x)
        y_val = f_expr.subs(x, measured_x)

        # Derivative f'(x)
        f_prime_expr = diff(f_expr, x)

        # Derivative value at measured_x: f'(measured_x)
        f_prime_at_measured = f_prime_expr.subs(x, measured_x)

        # Check for undefined/infinite values
        if y_val.has(oo, -oo, zoo, nan) or f_prime_at_measured.has(oo, -oo, zoo, nan):
            raise HTTPException(
                status_code=400,
                detail=f"Function or derivative undefined/infinite at measured value x={measured_x}.",
            )

        # Approximate Absolute Error: dy ≈ f'(measured_x) * dx
        dy = f_prime_at_measured * dx

        # Approximate Relative Error: dy / y
        rel_error = None
        if y_val == 0:
            rel_error_str = (
                "undefined (division by zero)" if dy != 0 else "0"
            )  # Or indeterminate?
        else:
            rel_error = dy / y_val
            rel_error_str = sympy_to_str(rel_error)

        # Approximate Percentage Error: rel_error * 100
        perc_error_str = None
        if rel_error is not None and not rel_error.has(oo, -oo, zoo, nan):
            perc_error = rel_error * 100
            perc_error_str = sympy_to_str(perc_error) + "%"
        elif rel_error_str == "undefined (division by zero)":
            perc_error_str = "undefined"
        else:
            perc_error_str = "0%" if rel_error_str == "0" else "undefined"

        return CalculateErrorsResponse(
            measured_variable_value=measured_x,
            possible_error_dx=dx,
            function_value_y=sympy_to_str(y_val),
            derivative_f_prime_x=latex(f_prime_expr),  # Use LaTeX
            derivative_at_measured_value=sympy_to_str(f_prime_at_measured),
            approximate_absolute_error_dy=sympy_to_str(dy),
            approximate_relative_error_dy_y=rel_error_str,
            approximate_percentage_error=perc_error_str,
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate errors: {e}")


# --- Tangents & Normals Routes ---


def get_derivative_and_point(request: TangentNormalRequest) -> Dict[str, Any]:
    """Helper to calculate dy/dx and validate point based on curve type."""
    curve_def = request.curve_definition
    point_req = request.point
    x = symbols(curve_def.variable)  # Usually 'x'
    y = (
        symbols(curve_def.dependent_variable)
        if curve_def.dependent_variable
        else symbols("y")
    )
    t = symbols("t")  # For parametric

    dy_dx_expr = None
    point_coords = {}  # Will store {x: val, y: val}
    subs_dict = {}  # Dictionary for substituting into dy/dx

    try:
        if curve_def.type == "explicit":
            if not curve_def.equation:
                raise ValueError("Explicit curve needs 'equation'.")
            # Assume equation is "y = f(x)" or just "f(x)"
            expr_str = curve_def.equation
            if "=" in expr_str:
                expr_str = expr_str.split("=", 1)[1].strip()

            f_expr = safe_parse_expr(expr_str, {curve_def.variable: x})
            dy_dx_expr = diff(f_expr, x)

            point_x = point_req.get(curve_def.variable)
            if point_x is None:
                raise ValueError(f"Point needs value for '{curve_def.variable}'.")
            point_y = f_expr.subs(x, point_x)  # Calculate y from x
            if point_y.has(oo, -oo, zoo, nan):
                raise ValueError(
                    f"Function undefined at {curve_def.variable}={point_x}"
                )

            point_coords = {
                curve_def.variable: point_x,
                curve_def.dependent_variable or "y": sympy_to_str(point_y),
            }
            subs_dict = {x: point_x}

        elif curve_def.type == "implicit":
            if not curve_def.equation:
                raise ValueError("Implicit curve needs 'equation'.")
            # Equation should be G(x, y) = 0 or G(x, y) = C
            # We need idiff which assumes G(x,y) = 0 form implicitly
            g_expr = safe_parse_expr(
                curve_def.equation,
                {curve_def.variable: x, curve_def.dependent_variable: y},
            )

            # Check if point values are provided
            point_x = point_req.get(curve_def.variable)
            point_y_val = point_req.get(curve_def.dependent_variable)
            if point_x is None or point_y_val is None:
                raise ValueError(
                    "Implicit curve point needs values for both variables."
                )

            # Optional: Check if point lies on the curve (can be complex)
            # check = simplify(g_expr.subs({x: point_x, y: point_y_val}))
            # Add tolerance for float comparison if check doesn't simplify to 0

            dy_dx_expr = idiff(g_expr, y, x)  # Find dy/dx implicitly

            point_coords = {
                curve_def.variable: point_x,
                curve_def.dependent_variable: point_y_val,
            }
            subs_dict = {x: point_x, y: point_y_val}

        elif curve_def.type == "parametric":
            if not curve_def.x_eq or not curve_def.y_eq:
                raise ValueError("Parametric curve needs 'x_eq' and 'y_eq'.")

            x_t = safe_parse_expr(curve_def.x_eq, {"t": t})
            y_t = safe_parse_expr(curve_def.y_eq, {"t": t})

            point_t = point_req.get("t")
            if point_t is None:
                raise ValueError("Parametric point needs value for 't'.")

            # Calculate dx/dt and dy/dt
            dx_dt = diff(x_t, t)
            dy_dt = diff(y_t, t)

            # dy/dx = (dy/dt) / (dx/dt)
            if dx_dt == 0:
                # Vertical tangent if dy/dt != 0
                dy_dx_expr = (
                    zoo if dy_dt.subs(t, point_t) != 0 else nan
                )  # Or indeterminate if both 0?
            else:
                dy_dx_expr = dy_dt / dx_dt

            # Calculate x, y coordinates at the given t
            point_x = x_t.subs(t, point_t)
            point_y = y_t.subs(t, point_t)

            point_coords = {
                "t": point_t,
                curve_def.variable: sympy_to_str(point_x),
                curve_def.dependent_variable or "y": sympy_to_str(point_y),
            }
            subs_dict = {t: point_t}  # Substitute t into dy/dx expression

        else:
            raise ValueError("Invalid curve type.")

        # Calculate slope value by substituting point into dy/dx expression
        m_tan = zoo  # Default to undefined
        if dy_dx_expr is not None:
            if dy_dx_expr == zoo:  # Already handled vertical tangent case
                m_tan = zoo
            else:
                m_tan = dy_dx_expr.subs(subs_dict)
                # Check result of substitution
                if m_tan.has(oo, -oo, zoo, nan):
                    m_tan = zoo  # If substitution leads to undefined/infinite

        # Calculate normal slope
        m_norm = zoo  # Default to undefined
        if m_tan == 0:
            m_norm = zoo  # Normal is vertical
        elif m_tan == zoo:
            m_norm = 0  # Normal is horizontal
        elif not m_tan.has(oo, -oo, zoo, nan):  # Check m_tan is finite non-zero
            m_norm = -1 / m_tan

        return {
            "point_coords": point_coords,
            "dy_dx_expression": dy_dx_expr,
            "tangent_slope": m_tan,
            "normal_slope": m_norm,
            "subs_dict": subs_dict,  # Pass substitution dict for reuse
            "x_sym": x,  # Pass symbols for equation building
            "y_sym": y,
        }

    except ValueError as ve:  # Catch specific errors for bad input
        raise HTTPException(status_code=400, detail=f"Input Error: {ve}")
    except Exception as e:  # Catch sympy errors etc
        # print(f"Error in get_derivative_and_point: {e}") # Debugging
        raise HTTPException(status_code=500, detail=f"Calculation Error: {e}")


# NEW: find-slopes
@router_tangents_normals.post("/find-slopes", response_model=SlopesResponse)
async def find_tangent_normal_slopes(request: TangentNormalRequest):
    """Finds the slopes of the tangent and normal lines to a curve at a point."""
    try:
        result = get_derivative_and_point(request)

        return SlopesResponse(
            point=result["point_coords"],
            derivative_expression_dy_dx=(
                latex(result["dy_dx_expression"])
                if result["dy_dx_expression"] is not None
                else None
            ),
            tangent_slope=sympy_to_str(result["tangent_slope"]),
            normal_slope=sympy_to_str(result["normal_slope"]),
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        # Generic error if helper raised non-HTTP exception (shouldn't happen often)
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred finding slopes: {e}"
        )


# NEW: find-equations
@router_tangents_normals.post("/find-equations", response_model=EquationsResponse)
async def find_tangent_normal_equations(request: TangentNormalRequest):
    """Finds the equations of the tangent and normal lines."""
    try:
        result = get_derivative_and_point(request)

        m_tan = result["tangent_slope"]
        m_norm = result["normal_slope"]
        x_sym = result["x_sym"]  # Symbol 'x'
        y_sym = result["y_sym"]  # Symbol 'y'

        # Point coordinates (need numerical values for equation)
        # Need to parse back from string representation if needed, or get from original req
        point_x_val = request.point.get(request.curve_definition.variable)
        point_y_val = None
        # Get y value - careful with implicit/parametric where it might be calculated
        dep_var_key = request.curve_definition.dependent_variable or "y"
        if dep_var_key in result["point_coords"]:
            try:
                # Try to convert back to float/sympy number if possible
                point_y_val = sympify(result["point_coords"][dep_var_key])
            except (SyntaxError, TypeError):
                raise HTTPException(
                    status_code=500,
                    detail="Could not parse calculated Y coordinate for equation.",
                )
        elif dep_var_key in request.point:  # If provided directly for implicit
            point_y_val = request.point[dep_var_key]
        else:  # Should have been caught earlier
            raise HTTPException(
                status_code=500, detail="Could not determine Y coordinate for equation."
            )

        if point_x_val is None or point_y_val is None:
            raise HTTPException(
                status_code=500,
                detail="Could not determine point coordinates (x,y) for equation.",
            )

        tan_eq_ps, tan_eq_simp = None, None
        norm_eq_ps, norm_eq_simp = None, None

        # Tangent Equation y - y1 = m(x - x1)
        if m_tan == zoo:  # Vertical tangent: x = x1
            tan_eq_ps = f"{latex(x_sym)} = {sympy_to_str(point_x_val)}"
            tan_eq_simp = tan_eq_ps
        elif not m_tan.has(oo, -oo, zoo, nan):
            eq = Eq(y_sym - point_y_val, m_tan * (x_sym - point_x_val))
            tan_eq_ps = latex(eq)
            # Simplify to Ax + By = C form or y = mx + c form
            try:
                # Expand and collect terms
                simp_eq = sympy.collect(sympy.expand(eq.lhs - eq.rhs), (x_sym, y_sym))
                tan_eq_simp = latex(Eq(simp_eq, 0))
            except Exception:
                tan_eq_simp = tan_eq_ps  # Fallback if simplification fails

        # Normal Equation y - y1 = m_norm(x - x1)
        if m_norm == zoo:  # Vertical normal: x = x1
            norm_eq_ps = f"{latex(x_sym)} = {sympy_to_str(point_x_val)}"
            norm_eq_simp = norm_eq_ps
        elif m_norm == 0:  # Horizontal normal: y = y1
            norm_eq_ps = f"{latex(y_sym)} = {sympy_to_str(point_y_val)}"
            norm_eq_simp = norm_eq_ps
        elif not m_norm.has(oo, -oo, zoo, nan):
            eq = Eq(y_sym - point_y_val, m_norm * (x_sym - point_x_val))
            norm_eq_ps = latex(eq)
            try:
                simp_eq = sympy.collect(sympy.expand(eq.lhs - eq.rhs), (x_sym, y_sym))
                norm_eq_simp = latex(Eq(simp_eq, 0))
            except Exception:
                norm_eq_simp = norm_eq_ps  # Fallback

        return EquationsResponse(
            point=result["point_coords"],
            derivative_expression_dy_dx=(
                latex(result["dy_dx_expression"])
                if result["dy_dx_expression"] is not None
                else None
            ),
            tangent_slope=sympy_to_str(m_tan),
            normal_slope=sympy_to_str(m_norm),
            tangent_equation_point_slope=tan_eq_ps,
            tangent_equation_simplified=tan_eq_simp,
            normal_equation_point_slope=norm_eq_ps,
            normal_equation_simplified=norm_eq_simp,
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        # print(f"Error finding equations: {e}") # Debugging
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred finding equations: {e}",
        )


# NEW: find-point-by-condition - Placeholder
@router_tangents_normals.post(
    "/find-point-by-condition", response_model=FindPointResponse
)
async def find_point_by_tangent_condition(request: FindPointRequest):
    """
    Finds point(s) on a curve where the tangent meets a specific condition.
    NOTE: Placeholder implementation. Requires solving potentially complex systems.
    """
    # TODO: Implement the full logic:
    # 1. Get symbolic dy/dx using logic from get_derivative_and_point.
    # 2. Parse the condition (parallel, perpendicular, slope_is, passes_through).
    # 3. Extract target slope 'm' or external point coordinates.
    # 4. Formulate the condition equation (e.g., dy/dx = m, dy/dx = -1/m, dy/dx = slope_value, dy/dx = (y-y_ext)/(x-x_ext)).
    # 5. Solve the condition equation simultaneously with the curve's equation using `solve`.
    # 6. Format the found points.
    raise HTTPException(
        status_code=501,
        detail="Finding point by tangent condition not fully implemented yet.",
    )


# NEW: angle-between-curves - Placeholder
@router_tangents_normals.post(
    "/angle-between-curves", response_model=AngleBetweenCurvesResponse
)
async def find_angle_between_curves(request: AngleBetweenCurvesRequest):
    """
    Finds the angle between two intersecting curves.
    NOTE: Placeholder implementation. Requires finding intersections and angles.
    """
    # TODO: Implement the full logic:
    # 1. Parse both curve definitions.
    # 2. Find intersection point(s) by solving the system of curve equations (can be hard).
    # 3. For each intersection point:
    #    a. Calculate slope m1 of tangent to curve 1 at the point.
    #    b. Calculate slope m2 of tangent to curve 2 at the point.
    #    c. Use formula tan(theta) = |(m1 - m2) / (1 + m1*m2)|.
    #    d. Handle cases: m1=m2 (tangent), 1+m1*m2=0 (orthogonal), vertical tangents.
    #    e. Calculate angle theta = atan(...).
    raise HTTPException(
        status_code=501,
        detail="Finding angle between curves not fully implemented yet.",
    )


# --- Monotonicity Routes ---


# NEW: find-intervals
@router_monotonicity.post(
    "/find-intervals", response_model=MonotonicityIntervalsResponse
)
async def find_monotonicity_intervals(request: MonotonicityIntervalsRequest):
    """
    Finds intervals where a function is increasing or decreasing.
    NOTE: Basic implementation focusing on f'(x)=0. Doesn't fully handle
    undefined points of f' robustly in all cases.
    """
    x = symbols(request.variable)
    f_expr_str = request.function_str
    if "=" in f_expr_str:
        f_expr_str = f_expr_str.split("=", 1)[1].strip()

    local_dict = {request.variable: x, "pi": pi, "E": sympy.E}
    f_expr = safe_parse_expr(f_expr_str, local_dict)

    increasing_intervals = []
    decreasing_intervals = []
    constant_intervals = []
    sign_analysis_list = []
    error_msg = None

    try:
        # Define the domain for analysis
        domain = SympyInterval(
            -oo if request.domain_start == "-oo" else float(request.domain_start),
            oo if request.domain_end == "oo" else float(request.domain_end),
        )

        # 1. Derivative
        f_prime = diff(f_expr, x)

        # 2. Critical points: where f'(x) = 0
        critical_points_zero = solveset(f_prime, x, domain=domain)

        # 3. Critical points: where f'(x) is undefined (singularities)
        critical_points_undef = singularities(f_prime, x, domain=domain)
        # Also consider singularities of the original function f(x) within the domain
        f_singularities = singularities(f_expr, x, domain=domain)

        # Combine and sort critical points (handle different solveset return types)
        all_critical_points = set()
        for p_set in [critical_points_zero, critical_points_undef, f_singularities]:
            if isinstance(p_set, FiniteSet):
                all_critical_points.update(list(p_set.args))
            # Add handling for Interval results if needed (e.g., if derivative is 0 over an interval)

        # Filter points strictly within the domain's open interval for interval splitting
        # Endpoints of the domain are handled separately or included based on interval def.
        # Need numeric values for sorting
        numeric_critical_points = []
        for p in all_critical_points:
            try:
                # Ensure point is within the analysis domain before adding
                if domain.contains(p):
                    numeric_critical_points.append(float(p))
            except (TypeError, ValueError):
                pass  # Ignore non-numeric points like 'I' or complex infinities

        sorted_points = sorted(list(set(numeric_critical_points)))

        # 4. Define test intervals
        test_intervals_sympy = []
        points_for_intervals = [domain.start] + sorted_points + [domain.end]
        # Remove duplicates that might arise if critical point = domain endpoint
        unique_interval_points = []
        if points_for_intervals:
            unique_interval_points.append(points_for_intervals[0])
            for i in range(1, len(points_for_intervals)):
                # Basic check for near-duplicate floats before adding
                if not (
                    isinstance(points_for_intervals[i], (float, int))
                    and isinstance(points_for_intervals[i - 1], (float, int))
                    and abs(points_for_intervals[i] - points_for_intervals[i - 1])
                    < 1e-9
                ):
                    unique_interval_points.append(points_for_intervals[i])

        for i in range(len(unique_interval_points) - 1):
            # Create open intervals between critical points/domain boundaries
            start_pt = unique_interval_points[i]
            end_pt = unique_interval_points[i + 1]
            # Ensure start < end, skip if points are identical
            if start_pt == end_pt:
                continue
            # Skip if start == -oo and end == -oo etc.
            if (
                start_pt == end_pt
                or (start_pt == -oo and end_pt == -oo)
                or (start_pt == oo and end_pt == oo)
            ):
                continue

            # Ensure the interval has non-zero width
            try:
                if start_pt != -oo and end_pt != oo and end_pt <= start_pt:
                    continue
            except TypeError:  # Handle comparison with oo
                pass

            test_intervals_sympy.append(
                SympyInterval(start_pt, end_pt, left_open=True, right_open=True)
            )

        # 5. Test sign in each interval
        for interval in test_intervals_sympy:
            # Skip empty intervals (e.g., Interval(1,1))
            if interval.measure == 0:
                continue

            # Choose a test point within the interval
            test_point = None
            try:
                if interval.start == -oo and interval.end == oo:
                    test_point = 0
                elif interval.start == -oo:
                    test_point = float(interval.end) - 1
                elif interval.end == oo:
                    test_point = float(interval.start) + 1
                else:
                    test_point = (float(interval.start) + float(interval.end)) / 2

                # Ensure test point is valid number
                if not isinstance(test_point, (int, float)):
                    raise ValueError("Invalid test point")

            except (TypeError, ValueError, OverflowError):
                # Could not find a simple test point, maybe interval involves oo/-oo complexly
                sign_analysis_list.append(
                    MonotonicitySignAnalysis(
                        interval=sympy_to_str([interval.start, interval.end]),
                        behavior="unknown (could not select test point)",
                    )
                )
                continue

            # Evaluate f' sign
            f_prime_sign = "unknown"
            f_prime_val_str = "undefined"
            behavior = "unknown"
            try:
                f_prime_value = f_prime.subs(x, test_point)
                f_prime_val_str = sympy_to_str(f_prime_value)

                if f_prime_value.has(oo, -oo, zoo, nan):
                    f_prime_sign = "undefined"
                    behavior = "undefined derivative"
                elif f_prime_value > 0:
                    f_prime_sign = "+"
                    behavior = "increasing"
                    increasing_intervals.append(
                        sympy_to_str([interval.start, interval.end])
                    )
                elif f_prime_value < 0:
                    f_prime_sign = "-"
                    behavior = "decreasing"
                    decreasing_intervals.append(
                        sympy_to_str([interval.start, interval.end])
                    )
                elif f_prime_value == 0:
                    f_prime_sign = "0"
                    # Check if derivative is zero over the whole interval (unlikely unless f'=0 identically)
                    # If f_prime simplifies to 0, it's constant
                    if simplify(f_prime) == 0:
                        behavior = "constant"
                        constant_intervals.append(
                            sympy_to_str([interval.start, interval.end])
                        )
                    else:  # Derivative is zero only at specific points, not interval
                        behavior = "critical point nearby"  # Or potentially constant if f' is identically 0

            except Exception as eval_e:
                # print(f"Error evaluating f'({test_point}): {eval_e}")
                f_prime_sign = "error"
                behavior = f"error evaluating derivative: {eval_e}"

            sign_analysis_list.append(
                MonotonicitySignAnalysis(
                    interval=sympy_to_str([interval.start, interval.end]),
                    test_point=test_point,
                    f_prime_value_at_test=f_prime_val_str,
                    f_prime_sign=f_prime_sign,
                    behavior=behavior,
                )
            )

        # Handle intervals where derivative might be identically zero
        zero_intervals = solveset(Eq(f_prime, 0), x, domain=domain)
        if isinstance(zero_intervals, SympyInterval):
            constant_intervals.append(
                sympy_to_str([zero_intervals.start, zero_intervals.end])
            )
            # TODO: Need to potentially remove this interval from inc/dec lists if covered

        return MonotonicityIntervalsResponse(
            function_str=request.function_str,
            derivative_f_prime_x=latex(f_prime),
            critical_points=sympy_to_str(
                sorted(
                    list(all_critical_points),
                    key=lambda p: (
                        float(p) if isinstance(p, (float, int, sympy.Number)) else 0
                    ),
                )
            ),  # Sort points for display
            intervals_analyzed=[
                sympy_to_str([i.start, i.end]) for i in test_intervals_sympy
            ],
            sign_analysis=sign_analysis_list,
            strictly_increasing_intervals=increasing_intervals,
            strictly_decreasing_intervals=decreasing_intervals,
            constant_intervals=constant_intervals,
            error=error_msg,
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        # print(f"Monotonicity Error: {e}") # Debugging
        raise HTTPException(
            status_code=500, detail=f"Failed to analyze monotonicity intervals: {e}"
        )


# NEW: check-interval (Monotonicity) - Placeholder
@router_monotonicity.post(
    "/check-interval", response_model=MonotonicityCheckIntervalResponse
)
async def check_monotonicity_on_interval(request: MonotonicityCheckIntervalRequest):
    """
    Checks if a function is strictly increasing/decreasing on a specific interval.
    NOTE: Placeholder implementation. Rigorous check requires careful sign analysis of f'.
    """
    # TODO: Implement the full logic:
    # 1. Calculate f'.
    # 2. Analyze the sign of f' *rigorously* over the entire given interval.
    #    - This might involve finding critical points of f' within the interval.
    #    - Using solveset(f' > 0, x, domain=interval), solveset(f' < 0, ...), solveset(f' == 0, ...)
    # 3. Determine if sign is consistently +, -, 0, or mixed.
    raise HTTPException(
        status_code=501,
        detail="Checking monotonicity on a specific interval not fully implemented yet.",
    )


# --- Register Routers ---
app = FastAPI(
    title="Calculus Solver API",
    description="API endpoints for solving common Class 12 Calculus problems.",
)
