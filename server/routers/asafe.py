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

# Import Pydantic models from models.py
from models.derivatives_model import *


# --- Routers ---
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

# --- Helper Functions ---


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
            return str(sympy_obj)  # For symbolic representation
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

    return str(sympy_obj)  # Default fallback


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


# --- Continuity Routes ---


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


# @router_continuity.post("/find-constants", response_model=ContinuityConstantsResponse)
# async def find_continuity_constants(request: ContinuityConstantsRequest):
#     """Finds constants that make a piecewise function continuous at a point."""
#     x = symbols(request.function_definition.variable)
#     point = request.point
#     func_def = request.function_definition
#     constants = symbols(request.constants_to_find)
#     local_dict = {func_def.variable: x}
#     local_dict.update({c.name: c for c in constants})

#     lhl, rhl, f_at_point = None, None, None
#     eqns = []
#     simplified_eqns = []
#     solutions = None
#     is_possible = False
#     reason = []

#     try:
#         if func_def.type != "piecewise":
#             raise HTTPException(
#                 status_code=400,
#                 detail="This endpoint requires a piecewise function definition.",
#             )

#         # Heuristic parsing - needs refinement
#         lhl_expr = get_relevant_expr(func_def, point, "lhl")
#         rhl_expr = get_relevant_expr(func_def, point, "rhl")
#         f_expr = get_relevant_expr(func_def, point, "point")

#         lhl = limit(lhl_expr, x, point, dir="-")
#         rhl = limit(rhl_expr, x, point, dir="+")
#         f_at_point = f_expr.subs(x, point)

#         # Create equations LHL = RHL and RHL = f(a)
#         # Use simplify to handle potential cancellations before creating Eq
#         eq1 = Eq(sympy.simplify(lhl), sympy.simplify(rhl))
#         eq2 = Eq(sympy.simplify(rhl), sympy.simplify(f_at_point))

#         # Store symbolic equations as strings
#         eqns_str = [sympy_to_str(eq1), sympy_to_str(eq2)]
#         # We might only need one unique equation if f(a) definition matches LHL or RHL
#         # Use sympy's equation solving
#         system = [eq1, eq2]
#         try:
#             # Use solveset for potentially more robust solving
#             sol = solve(system, constants)  # Returns dict or list of dicts

#             if isinstance(sol, dict) and sol:  # Check if dict and not empty
#                 solutions = {str(k): sympy_to_str(v) for k, v in sol.items()}
#                 is_possible = True
#                 reason.append("Solution found.")
#             elif (
#                 isinstance(sol, list) and sol
#             ):  # Multiple solutions possible? Typically unique for continuity.
#                 # Take the first solution for simplicity here
#                 solutions = {str(k): sympy_to_str(v) for k, v in sol[0].items()}
#                 is_possible = True
#                 reason.append("Solution found (potentially multiple exist).")
#             elif not sol:  # Empty list or dict means no solution
#                 is_possible = False
#                 reason.append(
#                     "No solution found for the constants satisfying the conditions."
#                 )
#             else:
#                 is_possible = False
#                 reason.append(
#                     f"Solver returned unexpected result: {type(sol)}. Cannot determine solution."
#                 )

#         except NotImplementedError:
#             is_possible = False
#             reason.append("Sympy solver could not handle the system of equations.")
#         except Exception as solve_e:
#             is_possible = False
#             reason.append(f"Error during solving: {solve_e}")

#         return ContinuityConstantsResponse(
#             point=point,
#             conditions_for_continuity=["LHL == RHL", "RHL == f(point)"],
#             equations_derived=eqns_str,
#             simplified_equations=eqns_str,  # Simplification happens before Eq creation mostly
#             solutions=solutions,
#             is_possible=is_possible,
#             reason=" ".join(reason),
#         )

#     except HTTPException as he:
#         raise he
#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"An unexpected error occurred: {e}"
#         )


# --- Differentiability Routes ---




# --- Add other routers and their endpoints here ---
# Placeholder for Rate Measure
@router_rate_measure.post("/related-rates", response_model=RelatedRatesResponse)
async def solve_related_rates(request: RelatedRatesRequest):
    # Implementation requires:
    # 1. Parsing equation string.
    # 2. Defining variables as sympy Functions of time 't'.
    # 3. Substituting functions into equation.
    # 4. Differentiating equation wrt 't'.
    # 5. Creating symbols for known/target rates (e.g., dx_dt = symbols('dx_dt')).
    # 6. Substituting rate symbols for Derivative objects.
    # 7. Solving for the target rate symbol.
    # 8. Substituting known rate values and instance values.
    # 9. Potentially solving original equation for missing instance values.
    raise HTTPException(
        status_code=501, detail="Related Rates endpoint not fully implemented yet."
    )


# Placeholder for Approximations
@router_approximations.post(
    "/approximate-value", response_model=ApproximateValueResponse
)
async def approximate_value(request: ApproximateValueRequest):
    x = symbols(request.variable)
    try:
        f_expr = safe_parse_expr(request.function_str, {request.variable: x})
        f_prime_expr = diff(f_expr, x)

        base_x = request.base_x
        target_x = request.target_x
        delta_x = target_x - base_x

        f_base_val = f_expr.subs(x, base_x)
        f_prime_base_val = f_prime_expr.subs(x, base_x)
        dy = f_prime_base_val * delta_x
        approx_val = f_base_val + dy

        return ApproximateValueResponse(
            base_x=base_x,
            target_x=target_x,
            delta_x=delta_x,
            f_base_x=sympy_to_str(f_base_val),
            derivative_f_prime_x=sympy_to_str(f_prime_expr),
            f_prime_base_x=sympy_to_str(f_prime_base_val),
            differential_dy=sympy_to_str(dy),
            approximation_formula=f"f({target_x}) ≈ f({base_x}) + f'({base_x}) * ({delta_x})",
            approximate_value_symbolic=sympy_to_str(approx_val),
            approximate_value_numeric=(
                sympy_to_str(N(approx_val)) if approx_val.is_real else None
            ),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Approximation calculation failed: {e}"
        )


# Placeholder for Tangents/Normals
@router_tangents_normals.post("/find-equations", response_model=EquationsResponse)
async def find_tangent_normal_equations(request: TangentNormalRequest):
    # Implementation requires handling explicit, implicit, parametric differentiation
    # And substituting point values, then forming line equations.
    raise HTTPException(
        status_code=501,
        detail="Tangent/Normal Equations endpoint not fully implemented yet.",
    )


# Placeholder for Monotonicity
@router_monotonicity.post(
    "/find-intervals", response_model=MonotonicityIntervalsResponse
)
async def find_monotonicity_intervals(request: MonotonicityIntervalsRequest):
    # Implementation requires:
    # 1. Differentiating the function.
    # 2. Finding critical points (where f'=0 or f' undefined) using solveset.
    # 3. Determining intervals based on critical points and domain.
    # 4. Testing the sign of f' in each interval.
    raise HTTPException(
        status_code=501,
        detail="Monotonicity Intervals endpoint not fully implemented yet.",
    )


# # --- Register Routers ---
# app.include_router(router_continuity)
# app.include_router(router_differentiability)
# app.include_router(router_rate_measure)
# app.include_router(router_approximations)
# app.include_router(router_tangents_normals)
# app.include_router(router_monotonicity)


# # --- Main Execution ---
# if __name__ == "__main__":
#     import uvicorn

#     # Example: Run with uvicorn main:app --reload
#     # Access interactive docs at http://127.0.0.1:8000/docs
#     uvicorn.run(app, host="127.0.0.1", port=8000)
