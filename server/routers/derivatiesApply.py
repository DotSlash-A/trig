import math
from fastapi import FastAPI, HTTPException, Body, APIRouter
from typing import List, Dict, Any, Optional
import sympy
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
from sympy import (
    sympify,
    symbols,
    limit,
    diff,
    solve,
    Eq,
    N,
    oo,
    Interval as SympyInterval,
    solveset,
    Abs,
    atan,
    pi,
    idiff,
    Function,
    Derivative,
)


from models.derivatives_model import *


# router = APIRouter(prefix="/applyderivatives", tags=["ApplyDifferentiation"])
def safe_parse_expr(expr_str: str, local_dict: Optional[Dict[str, Any]] = None):
    """Safely parse a string into a sympy expression."""
    try:
        # Adding implicit multiplication and function application transformations
        transformations = standard_transformations + (
            implicit_multiplication_application,
        )
        parsed_expr = parse_expr(
            expr_str, local_dict=local_dict, transformations=transformations
        )
        return parsed_expr
    except (SyntaxError, TypeError, Exception) as e:
        raise HTTPException(
            status_code=400, detail=f"Error parsing expression '{expr_str}': {e}"
        )


def sympy_to_str(sympy_obj):
    """Convert sympy objects to string for JSON compatibility."""
    if sympy_obj is None:
        return None
    if isinstance(sympy_obj, (list, tuple)):
        return [sympy_to_str(item) for item in sympy_obj]
    if isinstance(sympy_obj, dict):
        return {str(k): sympy_to_str(v) for k, v in sympy_obj.items()}
    if isinstance(
        sympy_obj,
        (sympy.Float, sympy.Integer, sympy.Rational, sympy.NumberSymbol, sympy.Symbol),
    ):
        # Attempt numerical evaluation if possible, otherwise string
        try:
            # Use N for approx, but keep precision. Or just str() for exact symbolic
            # return str(N(sympy_obj)) # For numeric approx
            num_val = float(sympy_obj)
            return f"{num_val:.2f}"
            # return str(sympy_obj)  # For symbolic representation

        except Exception:
            return str(sympy_obj)
    if sympy_obj == oo:
        return "oo"
    if sympy_obj == -oo:
        return "-oo"
    if sympy_obj == sympy.zoo:  # Complex infinity
        return "complex_infinity"
    if sympy_obj == sympy.nan:
        return "NaN"
    if sympy_obj == sympy.I:
        return "I"  # Imaginary unit
    # Handle undefined cases specifically if needed
    if isinstance(
        sympy_obj,
        (sympy.zooKind, type(sympy.zoo)),
    ):
        return "undefined"

    return str(sympy_obj)  # D


def get_relevant_expr(
    func_def: FunctionDefinition, point_val: float, target: str = "point"
):
    """Get the correct expression piece for LHL, RHL, or point based on condition"""
    x = symbols(func_def.variable)
    if func_def.type == "single":
        return safe_parse_expr(func_def.expression, {func_def.variable: x})

    expr_for_target = None
    condition_met = False

    if func_def.pieces:
        for piece in func_def.pieces:
            try:
                # Attempt to evaluate the condition symbolically/numerically
                # This is complex and may fail for intricate conditions
                condition_expr = safe_parse_expr(
                    piece.condition, {func_def.variable: x}
                )

                # Check condition based on target (LHL, RHL, Point)
                if target == "lhl":  # x < point_val
                    # Heuristic: if '<' or '<=' is in condition string and involves point
                    if "<" in piece.condition and str(point_val) in piece.condition:
                        expr_for_target = safe_parse_expr(
                            piece.expression, {func_def.variable: x}
                        )
                        break
                elif target == "rhl":  # x > point_val
                    if ">" in piece.condition and str(point_val) in piece.condition:
                        expr_for_target = safe_parse_expr(
                            piece.expression, {func_def.variable: x}
                        )
                        break
                elif target == "point":  # x == point_val
                    # Check for <=, >=, or an explicit == condition (less common)
                    if (
                        "<=" in piece.condition
                        or ">=" in piece.condition
                        or "==" in piece.condition
                    ) and str(point_val) in piece.condition:
                        # More robust check: substitute and see if True
                        try:
                            if condition_expr.subs(x, point_val) == True:
                                expr_for_target = safe_parse_expr(
                                    piece.expression, {func_def.variable: x}
                                )
                                condition_met = True
                                break  # Found the piece for the point value
                        except (TypeError, AttributeError):
                            # Fallback to string check if symbolic fails easily
                            if str(point_val) in piece.condition and (
                                "<=" in piece.condition or ">=" in piece.condition
                            ):
                                expr_for_target = safe_parse_expr(
                                    piece.expression, {func_def.variable: x}
                                )
                                condition_met = True
                                break

            except Exception as e:
                # print(f"Warning: Could not parse or evaluate condition '{piece.condition}': {e}")
                # Fallback heuristic based on string matching (less reliable)
                if target == "lhl" and "<" in piece.condition:
                    expr_for_target = safe_parse_expr(
                        piece.expression, {func_def.variable: x}
                    )
                elif target == "rhl" and ">" in piece.condition:
                    expr_for_target = safe_parse_expr(
                        piece.expression, {func_def.variable: x}
                    )
                elif target == "point" and (
                    "<=" in piece.condition or ">=" in piece.condition
                ):
                    expr_for_target = safe_parse_expr(
                        piece.expression, {func_def.variable: x}
                    )
                    condition_met = True  # Assume this covers the point

        # If no specific piece matched for the point, try LHL/RHL pieces again if they contain '='
        if target == "point" and not condition_met:
            for piece in func_def.pieces:
                if ("<=" in piece.condition or ">=" in piece.condition) and str(
                    point_val
                ) in piece.condition:
                    expr_for_target = safe_parse_expr(
                        piece.expression, {func_def.variable: x}
                    )
                    break

    if expr_for_target is None:
        # Fallback or error if no suitable expression found
        # This logic needs refinement for robustness
        # print(f"Warning: Could not determine expression for {target} at {point_val}")
        # As a last resort for limits, try using the piece definition nearest the point
        if func_def.pieces:
            if target == "lhl":
                return safe_parse_expr(
                    func_def.pieces[0].expression, {func_def.variable: x}
                )  # Simplistic guess
            if target == "rhl":
                return safe_parse_expr(
                    func_def.pieces[-1].expression, {func_def.variable: x}
                )  # Simplistic guess
            if (
                target == "point"
            ):  # Try the piece definition that likely includes the boundary
                for p in func_def.pieces:
                    if ">=" in p.condition or "<=" in p.condition:
                        return safe_parse_expr(p.expression, {func_def.variable: x})
        raise HTTPException(
            status_code=400,
            detail=f"Could not determine relevant function piece for {target} at {point_val}",
        )

    return expr_for_target


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
