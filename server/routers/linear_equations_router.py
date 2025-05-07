# routers/linear_equations_router.py
from fastapi import APIRouter, Body, HTTPException, Query
from models import linear_equations_models as models
from services import linear_algebra_service as la_service
from typing import Tuple

router = APIRouter(
    prefix="/linear-equations-two-variables",
    tags=["Pair of Linear Equations in Two Variables"]
)

def _get_coeffs_from_request(req: models.PairOfLinearEquationsRequest) -> Tuple[float, float, float, float, float, float]:
    """Helper to extract coefficients a1x+b1y=c1, a2x+b2y=c2"""
    return req.eq1.a, req.eq1.b, req.eq1.c, req.eq2.a, req.eq2.b, req.eq2.c

@router.post("/check-consistency", response_model=models.ConsistencyCheckResponse)
async def check_system_consistency(
    equations_input: models.PairOfLinearEquationsRequest = Body(
        ...,
        examples={
            "unique_solution": {
                "summary": "Unique Solution Example",
                "description": "Coefficients for x + y = 5 and 2x - 3y = 4",
                "value": {
                    "eq1": {"a": 1, "b": 1, "c": 5},
                    "eq2": {"a": 2, "b": -3, "c": 4}
                }
            },
            "infinite_solutions": {
                "summary": "Infinite Solutions Example",
                "description": "Coefficients for 2x + 3y = 9 and 4x + 6y = 18",
                "value": {
                    "eq1": {"a": 2, "b": 3, "c": 9},
                    "eq2": {"a": 4, "b": 6, "c": 18}
                }
            },
            "no_solution": {
                "summary": "No Solution Example",
                "description": "Coefficients for x + 2y = 4 and 2x + 4y = 12",
                "value": {
                    "eq1": {"a": 1, "b": 2, "c": 4},
                    "eq2": {"a": 2, "b": 4, "c": 12}
                }
            }
        }
    )
):
    """
    Checks the consistency of a pair of linear equations:
    a1x + b1y = c1
    a2x + b2y = c2
    (Input c1, c2 are the RHS constants)
    Determines if the system has a unique solution, infinitely many solutions, or no solution.
    """
    a1, b1, c1, a2, b2, c2 = _get_coeffs_from_request(equations_input)
    
    try:
        consistency_type, description, ratios = la_service.check_consistency(a1, b1, c1, a2, b2, c2)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    graphical_interpretation = ""
    if consistency_type == "consistent_unique":
        graphical_interpretation = "Lines intersect at a single point."
    elif consistency_type == "consistent_infinite":
        graphical_interpretation = "Lines are coincident (overlap completely)."
    elif consistency_type == "inconsistent_parallel":
        graphical_interpretation = "Lines are parallel and distinct."

    return models.ConsistencyCheckResponse(
        equations=models.EquationsCoeffs(a1=a1, b1=b1, c1=c1, a2=a2, b2=b2, c2=c2),
        consistency_type=consistency_type,
        description=description,
        ratios=ratios,
        graphical_interpretation=graphical_interpretation
    )

@router.post("/solve/general", response_model=models.SolutionResponse)
async def solve_equations_general(
    equations_input: models.PairOfLinearEquationsRequest = Body(..., examples={
            "example1": {
                "summary": "Solve x+y=5, 2x-3y=4",
                "value": {
                    "eq1": {"a": 1, "b": 1, "c": 5},
                    "eq2": {"a": 2, "b": -3, "c": 4}
                }
            }
        })
):
    """
    Solves a pair of linear equations (a1x + b1y = c1, a2x + b2y = c2) using a general algebraic method
    (typically matrix inversion or equivalent robust method).
    Provides the solution (x, y) if unique, or describes the nature of solutions otherwise.
    """
    a1, b1, c1, a2, b2, c2 = _get_coeffs_from_request(equations_input)
    
    try:
        solution_details = la_service.solve_linear_equations(a1, b1, c1, a2, b2, c2)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return models.SolutionResponse(
        equations=models.EquationsCoeffs(a1=a1, b1=b1, c1=c1, a2=a2, b2=b2, c2=c2),
        **solution_details
    )

@router.post("/solve/substitution", response_model=models.SolutionResponse)
async def solve_equations_substitution(
    equations_input: models.PairOfLinearEquationsRequest = Body(..., examples={
            "example1": {
                "summary": "Solve 7x-15y=2, x+2y=3",
                "value": {
                    "eq1": {"a": 7, "b": -15, "c": 2},
                    "eq2": {"a": 1, "b": 2, "c": 3}
                }
            }
        })
):
    """
    Solves a pair of linear equations (a1x + b1y = c1, a2x + b2y = c2) using the Substitution Method.
    Provides step-by-step details.
    """
    a1, b1, c1, a2, b2, c2 = _get_coeffs_from_request(equations_input)
    try:
        solution_details = la_service.solve_by_substitution(a1, b1, c1, a2, b2, c2)
        # consistency is already in solution_details from the service
        return models.SolutionResponse(
            equations=models.EquationsCoeffs(a1=a1, b1=b1, c1=c1, a2=a2, b2=b2, c2=c2),
            consistency_type=solution_details["consistency"], 
            description=f"Solution by Substitution. Consistency: {solution_details['consistency']}",
            solution_x=solution_details["solution_x"],
            solution_y=solution_details["solution_y"],
            method_used="Substitution",
            steps=solution_details["steps"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in substitution method: {str(e)}")


@router.post("/solve/elimination", response_model=models.SolutionResponse)
async def solve_equations_elimination(
    equations_input: models.PairOfLinearEquationsRequest = Body(..., examples={
            "example1": {
                "summary": "Solve 9x-4y=2000, 7x-3y=2000",
                "value": {
                    "eq1": {"a": 9, "b": -4, "c": 2000},
                    "eq2": {"a": 7, "b": -3, "c": 2000}
                }
            }
        })
):
    """
    Solves a pair of linear equations (a1x + b1y = c1, a2x + b2y = c2) using the Elimination Method.
    Provides step-by-step details.
    """
    a1, b1, c1, a2, b2, c2 = _get_coeffs_from_request(equations_input)
    try:
        solution_details = la_service.solve_by_elimination(a1, b1, c1, a2, b2, c2)
        return models.SolutionResponse(
            equations=models.EquationsCoeffs(a1=a1, b1=b1, c1=c1, a2=a2, b2=b2, c2=c2),
            consistency_type=solution_details["consistency"],
            description=f"Solution by Elimination. Consistency: {solution_details['consistency']}",
            solution_x=solution_details["solution_x"],
            solution_y=solution_details["solution_y"],
            method_used="Elimination",
            steps=solution_details["steps"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in elimination method: {str(e)}")


@router.post("/solve/cross-multiplication", response_model=models.SolutionResponse)
async def solve_equations_cross_multiplication(
    equations_input_axbyc0: models.PairOfLinearEquationsRequest = Body(
        ...,
        description="Input equations in the form a₁x + b₁y + c₁ = 0 and a₂x + b₂y + c₂ = 0. "
                    "The 'c' value in the input should be the c from this form.",
        examples={
            "example1": {
                "summary": "Solve 2x+y-5=0, 3x+2y-8=0",
                "value": {
                    "eq1": {"a": 2, "b": 1, "c": -5}, # c1 is -5 for 2x+y-5=0
                    "eq2": {"a": 3, "b": 2, "c": -8}  # c2 is -8 for 3x+2y-8=0
                }
            }
        }
    )
):
    """
    Solves a pair of linear equations using the Cross-Multiplication Method.
    IMPORTANT: Expects equations in the form a₁x + b₁y + c₁ = 0 and a₂x + b₂y + c₂ = 0.
    The 'c' value provided for each equation in the request body MUST be this c₁, c₂.
    """
    a1, b1, c1_form_axbyc0, a2, b2, c2_form_axbyc0 = _get_coeffs_from_request(equations_input_axbyc0)
    
    # For internal consistency checks and general solver, convert to ax+by=C where C = -c_form_axbyc0
    c1_form_axby_eq_C = -c1_form_axbyc0
    c2_form_axby_eq_C = -c2_form_axbyc0

    try:
        # Pass c1_form_axbyc0, c2_form_axbyc0 directly to cross-multiplication service function
        solution_details = la_service.solve_by_cross_multiplication(
            a1, b1, c1_form_axbyc0, 
            a2, b2, c2_form_axbyc0
        )
        return models.SolutionResponse(
             equations=models.EquationsCoeffs(a1=a1, b1=b1, c1=c1_form_axby_eq_C, a2=a2, b2=b2, c2=c2_form_axby_eq_C),
            consistency_type=solution_details["consistency"],
            description=f"Solution by Cross-Multiplication. Consistency: {solution_details['consistency']}",
            solution_x=solution_details["solution_x"],
            solution_y=solution_details["solution_y"],
            method_used="Cross-Multiplication",
            steps=solution_details["steps"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in cross-multiplication method: {str(e)}")

@router.post("/solve/reducible-equations", response_model=models.ReducibleSolutionResponse)
async def solve_reducible_equations(
    req: models.SubstitutedEquationsRequest = Body(..., examples={
        "example1": {
            "summary": "Example: 2/x + 3/y = 13,  5/x - 4/y = -2",
            "description": "Let u=1/x, v=1/y. Then 2u+3v=13 and 5u-4v=-2",
            "value": {
                "u_coeff1": 2, "v_coeff1": 3, "const1": 13,
                "u_coeff2": 5, "v_coeff2": -4, "const2": -2,
                "original_var_u": "1/x",
                "original_var_v": "1/y"
            }
        }
    })
):
    """
    Solves equations that can be reduced to a pair of linear equations.
    The user must perform the substitution (e.g., u = 1/x, v = 1/y) and provide
    the coefficients for the new linear equations in terms of 'u' and 'v'.
    The API will solve for 'u' and 'v', then substitute back to find 'x' and 'y'.
    """
    a1, b1, c1 = req.u_coeff1, req.v_coeff1, req.const1
    a2, b2, c2 = req.u_coeff2, req.v_coeff2, req.const2

    try:
        # Solve for u and v
        solution_uv_details = la_service.solve_linear_equations(a1, b1, c1, a2, b2, c2)
        
        u_val = solution_uv_details.get("solution_x") # 'x' here is 'u'
        v_val = solution_uv_details.get("solution_y") # 'y' here is 'v'
        
        original_x = None
        original_y = None
        
        substituted_equations_info = {
            "u_eq": f"{a1}u + {b1}v = {c1}",
            "v_eq": f"{a2}u + {b2}v = {c2}",
            "u_represents": req.original_var_u,
            "v_represents": req.original_var_v,
            "solution_u": u_val,
            "solution_v": v_val
        }

        if solution_uv_details["consistency_type"] == "consistent_unique" and u_val is not None and v_val is not None:
            # Attempt to solve for original x and y
            # This assumes u = 1/x and v = 1/y for now. More robust parsing of original_var_u/v would be needed for other forms.
            if req.original_var_u.strip() == "1/x" and abs(float(u_val)) > 1e-9:
                original_x = 1.0 / float(u_val)
            elif req.original_var_u.strip() == "x": # If u directly represented x
                original_x = float(u_val)
            else:
                original_x = f"Cannot determine x from u={u_val} where u={req.original_var_u}"

            if req.original_var_v.strip() == "1/y" and abs(float(v_val)) > 1e-9:
                original_y = 1.0 / float(v_val)
            elif req.original_var_v.strip() == "y": # If v directly represented y
                original_y = float(v_val)
            else:
                original_y = f"Cannot determine y from v={v_val} where v={req.original_var_v}"
        
        elif u_val == "Infinite" or v_val == "Infinite":
            original_x = "Infinite (derived from u)"
            original_y = "Infinite (derived from v)"


        return models.ReducibleSolutionResponse(
            equations=models.EquationsCoeffs(a1=a1,b1=b1,c1=c1,a2=a2,b2=b2,c2=c2), # These are for u,v
            consistency_type=solution_uv_details["consistency_type"],
            description=f"Solution for substituted variables (u,v): {solution_uv_details['description']}",
            solution_x=u_val, # u
            solution_y=v_val, # v
            original_solution_x=original_x, # x
            original_solution_y=original_y, # y
            method_used=solution_uv_details["method_used"],
            substituted_equations_details=substituted_equations_info
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ZeroDivisionError:
        raise HTTPException(status_code=400, detail="Division by zero encountered when trying to find original x or y (u or v was zero).")