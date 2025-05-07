# services/linear_algebra_service.py
from typing import Tuple, Optional, Dict, List, Any, Union
import numpy as np # For solving using matrix method (alternative to cross-multiplication or more robust)

# Standard form: a1x + b1y = c1  OR a1x + b1y + c1 = 0
# For consistency checks and solving, it's often easier to work with a1x + b1y = c1
# We'll assume inputs can be in either form and internally convert to a1x + b1y = c1

def parse_equation_coeffs(
    a1: float, b1: float, c1: float,
    a2: float, b2: float, c2: float,
    equation_form: str = "ax+by=c" # "ax+by=c" or "ax+by+c=0"
) -> Tuple[float, float, float, float, float, float]:
    """
    Ensures coefficients are in the form a1x + b1y = c1 and a2x + b2y = c2.
    """
    if equation_form == "ax+by+c=0":
        # Convert c1 and c2
        c1_adj = -c1
        c2_adj = -c2
    elif equation_form == "ax+by=c":
        c1_adj = c1
        c2_adj = c2
    else:
        raise ValueError("Invalid equation_form. Must be 'ax+by=c' or 'ax+by+c=0'.")
    
    return a1, b1, c1_adj, a2, b2, c2_adj


def check_consistency(
    a1: float, b1: float, c1: float,
    a2: float, b2: float, c2: float
) -> Tuple[str, str, Dict[str, float]]:
    """
    Checks the consistency of a pair of linear equations a1x + b1y = c1 and a2x + b2y = c2.
    Returns:
        - consistency_type: "consistent_unique", "consistent_infinite", "inconsistent_parallel"
        - description: A human-readable explanation.
        - ratios: Calculated ratios a1/a2, b1/b2, c1/c2 (handling division by zero).
    """
    ratios = {}
    description_parts = []

    # Calculate ratios, handling potential division by zero
    # For a1/a2
    if a2 != 0:
        ratio_a1_a2 = a1 / a2
        ratios["a1/a2"] = ratio_a1_a2
        desc_a1_a2 = f"a₁/a₂ = {a1}/{a2} = {ratio_a1_a2:.4f}"
    elif a1 == 0: # 0/0 case implies lines could be x-axis or y-axis or overlapping axes
        ratio_a1_a2 = float('nan') # Or some indicator that it's 0/0
        ratios["a1/a2"] = "0/0 (indeterminate)"
        desc_a1_a2 = "a₁/a₂ = 0/0 (both zero)"
    else: # a1 != 0, a2 == 0
        ratio_a1_a2 = float('inf') if a1 > 0 else float('-inf') # Represents vertical line if b2 is also 0 for a1x=c1
        ratios["a1/a2"] = "undefined (a₁/0, a₁≠0)"
        desc_a1_a2 = f"a₁/a₂ = {a1}/0 (undefined, a₁≠0)"
    description_parts.append(desc_a1_a2)

    # For b1/b2
    if b2 != 0:
        ratio_b1_b2 = b1 / b2
        ratios["b1/b2"] = ratio_b1_b2
        desc_b1_b2 = f"b₁/b₂ = {b1}/{b2} = {ratio_b1_b2:.4f}"
    elif b1 == 0:
        ratio_b1_b2 = float('nan')
        ratios["b1/b2"] = "0/0 (indeterminate)"
        desc_b1_b2 = "b₁/b₂ = 0/0 (both zero)"
    else:
        ratio_b1_b2 = float('inf') if b1 > 0 else float('-inf')
        ratios["b1/b2"] = "undefined (b₁/0, b₁≠0)"
        desc_b1_b2 = f"b₁/b₂ = {b1}/0 (undefined, b₁≠0)"
    description_parts.append(desc_b1_b2)

    # For c1/c2
    if c2 != 0:
        ratio_c1_c2 = c1 / c2
        ratios["c1/c2"] = ratio_c1_c2
        desc_c1_c2 = f"c₁/c₂ = {c1}/{c2} = {ratio_c1_c2:.4f}"
    elif c1 == 0:
        ratio_c1_c2 = float('nan')
        ratios["c1/c2"] = "0/0 (indeterminate)"
        desc_c1_c2 = "c₁/c₂ = 0/0 (both zero)"
    else:
        ratio_c1_c2 = float('inf') if c1 > 0 else float('-inf')
        ratios["c1/c2"] = "undefined (c₁/0, c₁≠0)"
        desc_c1_c2 = f"c₁/c₂ = {c1}/0 (undefined, c₁≠0)"
    description_parts.append(desc_c1_c2)
    
    # Precision for comparison
    epsilon = 1e-9

    # Case 1: Unique solution (intersecting lines)
    # a1/a2 != b1/b2
    # Need to handle a2=0 or b2=0 carefully.
    # If a2=0, a1/a2 is inf or 0/0. If b2=0, b1/b2 is inf or 0/0.
    
    # If (a1*b2 - a2*b1) != 0, then unique solution. This is determinant of coeff matrix.
    determinant = a1 * b2 - a2 * b1

    if abs(determinant) > epsilon:
        consistency_type = "consistent_unique"
        description = f"Intersecting lines (unique solution) because a₁b₂ - a₂b₁ = {determinant:.4f} ≠ 0. Ratios: {', '.join(description_parts)}."
        # More traditional check:
        # Check if a1/a2 != b1/b2. This fails if a2 or b2 is 0.
        # if (a2 == 0 and b2 != 0) or (a2 != 0 and b2 == 0) or \
        #    (a2 != 0 and b2 != 0 and abs(ratio_a1_a2 - ratio_b1_b2) > epsilon):
        # Alternative using cross products to avoid division by zero issues directly in comparison
        # if abs(a1 * b2 - a2 * b1) > epsilon:
        return consistency_type, description, ratios

    # Case 2 & 3: Parallel or Coincident (determinant is close to zero)
    # a1/a2 = b1/b2. Now check c1/c2.
    # Condition for parallel: a1/a2 = b1/b2 != c1/c2
    # Condition for coincident: a1/a2 = b1/b2 = c1/c2
    # Use cross products: a1*b2 = a2*b1 and b1*c2 = b2*c1 and a1*c2 = a2*c1
    
    # Check if a1b2 = a2b1 (already established by determinant ~ 0)
    # Check if b1c2 = b2c1
    if abs(b1 * c2 - b2 * c1) > epsilon: # Means b1/b2 != c1/c2 (or one is defined, other is not)
        consistency_type = "inconsistent_parallel"
        description = f"Parallel lines (no solution) because a₁b₂ - a₂b₁ ≈ 0 AND b₁c₂ - b₂c₁ ≠ 0. Ratios: {', '.join(description_parts)}."
        return consistency_type, description, ratios
    else:
        # Now we have a1b2=a2b1 AND b1c2=b2c1.
        # We also need to check a1c2 = a2c1, or ensure all three ratios are equal.
        # If a2, b2, c2 are non-zero, this means a1/a2 = b1/b2 = c1/c2.
        # Handle cases where some denominators are zero carefully.
        # If a1=b1=c1=0 and a2=b2=c2=0, it's infinite (0x+0y=0)
        if all(abs(val) < epsilon for val in [a1,b1,c1,a2,b2,c2]):
             consistency_type = "consistent_infinite"
             description = f"Coincident lines (infinitely many solutions) because all coefficients are zero (0x + 0y = 0). Ratios: {', '.join(description_parts)}."
             return consistency_type, description, ratios

        # If one equation is 0x+0y=0 and the other is not, it's infinite.
        if (all(abs(val) < epsilon for val in [a1,b1,c1])) or \
           (all(abs(val) < epsilon for val in [a2,b2,c2])):
            consistency_type = "consistent_infinite" # One line is essentially 0=0
            description = f"Coincident lines (infinitely many solutions) - one equation is trivial (0x + 0y = 0). Ratios: {', '.join(description_parts)}."
            return consistency_type, description, ratios

        # Check if one line is a non-zero multiple of the other, e.g. x+y=1 and 2x+2y=2
        # At this point, a1b2-a2b1 ~ 0 and b1c2-b2c1 ~ 0.
        # This implies a1/a2 = b1/b2 and b1/b2 = c1/c2 (if denominators non-zero)
        # So, a1/a2 = b1/b2 = c1/c2
        consistency_type = "consistent_infinite"
        description = f"Coincident lines (infinitely many solutions) because a₁b₂ - a₂b₁ ≈ 0 AND b₁c₂ - b₂c₁ ≈ 0 (implies ratios are equal or lines are dependent). Ratios: {', '.join(description_parts)}."
        return consistency_type, description, ratios


def solve_linear_equations(
    a1: float, b1: float, c1: float,
    a2: float, b2: float, c2: float
) -> Dict[str, Any]:
    """
    Solves a pair of linear equations a1x + b1y = c1 and a2x + b2y = c2.
    Returns a dictionary with solution type, solution (if unique), and explanation.
    Uses numpy.linalg.solve for robustness.
    """
    consistency_type, description, _ = check_consistency(a1, b1, c1, a2, b2, c2)
    
    solution = {
        "consistency_type": consistency_type,
        "description": description,
        "solution_x": None,
        "solution_y": None,
        "method_used": "Analysis of coefficients and numpy.linalg.solve"
    }

    if consistency_type == "consistent_unique":
        # Using numpy.linalg.solve
        # Equations:
        # a1*x + b1*y = c1
        # a2*x + b2*y = c2
        coeff_matrix = np.array([[a1, b1], [a2, b2]])
        constants_vector = np.array([c1, c2])
        try:
            x_val, y_val = np.linalg.solve(coeff_matrix, constants_vector)
            solution["solution_x"] = x_val
            solution["solution_y"] = y_val
        except np.linalg.LinAlgError:
            # This should ideally be caught by consistency check, but as a fallback
            solution["description"] += " (Numpy linalg.solve failed, system might be singular despite initial check)."
            # Attempt cross-multiplication as a fallback or for explicit steps if needed
            # Determinant D = a1*b2 - a2*b1
            D = a1 * b2 - a2 * b1
            # Dx = c1*b2 - c2*b1
            Dx = c1 * b2 - c2 * b1
            # Dy = a1*c2 - a2*c1
            Dy = a1 * c2 - a2 * c1
            if abs(D) > 1e-9:
                solution["solution_x"] = Dx / D
                solution["solution_y"] = Dy / D
                solution["method_used"] += " (Fallback to Cramer's rule/Cross-multiplication)"


    elif consistency_type == "consistent_infinite":
        solution["description"] += " System has infinitely many solutions. One possible parameterization: if b1 != 0, y = (c1 - a1x)/b1. If b1=0 and a1!=0, x = c1/a1, y is free (if consistent). Similar for eq2."
        # (Could provide a parameterized solution if desired)
    elif consistency_type == "inconsistent_parallel":
        solution["description"] += " System has no solution."
        
    return solution


def solve_by_substitution(
    a1: float, b1: float, c1: float,
    a2: float, b2: float, c2: float
) -> Dict[str, Any]:
    """
    Solves by substitution method and provides step-by-step explanation.
    a1x + b1y = c1  (eq1)
    a2x + b2y = c2  (eq2)
    """
    steps = []
    solution_x, solution_y = None, None
    consistency_type, _, _ = check_consistency(a1, b1, c1, a2, b2, c2)

    if consistency_type == "inconsistent_parallel":
        steps.append("The system is inconsistent (parallel lines) and has no solution.")
        return {"method": "Substitution", "steps": steps, "solution_x": None, "solution_y": None, "consistency": consistency_type}
    if consistency_type == "consistent_infinite":
        steps.append("The system is consistent with infinitely many solutions (coincident lines).")
        return {"method": "Substitution", "steps": steps, "solution_x": "Infinite", "solution_y": "Infinite", "consistency": consistency_type}

    # Try to express x from eq1: x = (c1 - b1y) / a1 (if a1 != 0)
    # Or y from eq1: y = (c1 - a1x) / b1 (if b1 != 0)
    # Choose the one with non-zero coefficient
    
    epsilon = 1e-9 # For float comparisons

    if abs(a1) > epsilon:
        steps.append(f"From equation 1 ({a1}x + {b1}y = {c1}):")
        steps.append(f"Express x: {a1}x = {c1} - {b1}y  =>  x = ({c1} - {b1}y) / {a1}")
        # Substitute into eq2: a2 * [ (c1 - b1y) / a1 ] + b2y = c2
        # a2(c1 - b1y) + a1b2y = a1c2
        # a2c1 - a2b1y + a1b2y = a1c2
        # (a1b2 - a2b1)y = a1c2 - a2c1
        lhs_y_coeff = (a1 * b2 - a2 * b1)
        rhs_y_val = (a1 * c2 - a2 * c1)
        steps.append(f"Substitute this x into equation 2 ({a2}x + {b2}y = {c2}):")
        steps.append(f"{a2} * (({c1} - {b1}y) / {a1}) + {b2}y = {c2}")
        steps.append(f"Multiply by {a1} (if a1 != 1): {a2}({c1} - {b1}y) + {a1*b2}y = {a1*c2}")
        steps.append(f"{a2*c1} - {a2*b1}y + {a1*b2}y = {a1*c2}")
        steps.append(f"({a1*b2 - a2*b1})y = {a1*c2 - a2*c1}")
        
        if abs(lhs_y_coeff) > epsilon:
            solution_y = rhs_y_val / lhs_y_coeff
            steps.append(f"{lhs_y_coeff}y = {rhs_y_val}  =>  y = {solution_y}")
            # Substitute y back into expression for x
            solution_x = (c1 - b1 * solution_y) / a1
            steps.append(f"Substitute y = {solution_y} back into x = ({c1} - {b1}y) / {a1}:")
            steps.append(f"x = ({c1} - {b1*solution_y}) / {a1} = {c1 - b1*solution_y} / {a1} = {solution_x}")
        else: # lhs_y_coeff is 0
             # This case should have been caught by consistency check, implies parallel or coincident
            steps.append(f"This simplifies to 0*y = {rhs_y_val}.")
            if abs(rhs_y_val) > epsilon:
                 steps.append("This is a contradiction (0 = non-zero), so no solution (parallel lines).")
            else:
                 steps.append("This is 0 = 0, indicating infinitely many solutions (coincident lines).")

    elif abs(b1) > epsilon: # Try expressing y from eq1
        steps.append(f"From equation 1 ({a1}x + {b1}y = {c1}):")
        steps.append(f"Express y: {b1}y = {c1} - {a1}x  =>  y = ({c1} - {a1}x) / {b1}")
        # Substitute into eq2: a2x + b2 * [ (c1 - a1x) / b1 ] = c2
        # a2b1x + b2(c1 - a1x) = b1c2
        # a2b1x + b2c1 - a1b2x = b1c2
        # (a2b1 - a1b2)x = b1c2 - b2c1
        lhs_x_coeff = (a2 * b1 - a1 * b2)
        rhs_x_val = (b1 * c2 - b2 * c1)
        steps.append(f"Substitute this y into equation 2 ({a2}x + {b2}y = {c2}):")
        steps.append(f"{a2}x + {b2} * (({c1} - {a1}x) / {b1}) = {c2}")
        steps.append(f"Multiply by {b1} (if b1 != 1): {a2*b1}x + {b2}({c1} - {a1}x) = {b1*c2}")
        steps.append(f"{a2*b1}x + {b2*c1} - {a1*b2}x = {b1*c2}")
        steps.append(f"({a2*b1 - a1*b2})x = {b1*c2 - b2*c1}")

        if abs(lhs_x_coeff) > epsilon:
            solution_x = rhs_x_val / lhs_x_coeff
            steps.append(f"{lhs_x_coeff}x = {rhs_x_val}  =>  x = {solution_x}")
            # Substitute x back into expression for y
            solution_y = (c1 - a1 * solution_x) / b1
            steps.append(f"Substitute x = {solution_x} back into y = ({c1} - {a1}x) / {b1}:")
            steps.append(f"y = ({c1} - {a1*solution_x}) / {b1} = {c1 - a1*solution_x} / {b1} = {solution_y}")
        else:
            steps.append(f"This simplifies to 0*x = {rhs_x_val}.")
            if abs(rhs_x_val) > epsilon:
                 steps.append("This is a contradiction (0 = non-zero), so no solution (parallel lines).")
            else:
                 steps.append("This is 0 = 0, indicating infinitely many solutions (coincident lines).")
    
    # If a1=0 and b1=0 in eq1.
    elif abs(a1) < epsilon and abs(b1) < epsilon:
        if abs(c1) < epsilon: # 0x + 0y = 0
            steps.append("Equation 1 is 0x + 0y = 0, which is always true. The solution depends solely on Equation 2.")
            # Solve eq2: a2x + b2y = c2
            if abs(a2) > epsilon :
                steps.append(f"From Equation 2 ({a2}x + {b2}y = {c2}), x = ({c2} - {b2}y)/{a2}. Infinitely many solutions parameterized by y.")
                solution_x, solution_y = "Infinite (parameterized)", "Infinite (parameterized)"
            elif abs(b2) > epsilon:
                steps.append(f"From Equation 2 ({a2}x + {b2}y = {c2}), y = ({c2} - {a2}x)/{b2}. Infinitely many solutions parameterized by x.")
                solution_x, solution_y = "Infinite (parameterized)", "Infinite (parameterized)"
            elif abs(c2) < epsilon: # Eq2 is also 0x+0y=0
                steps.append("Equation 2 is also 0x + 0y = 0. Infinitely many solutions (any x, y).")
                solution_x, solution_y = "Infinite (any x)", "Infinite (any y)"
            else: # Eq2 is 0x+0y = non-zero
                steps.append(f"Equation 2 is 0x + 0y = {c2} (non-zero). This is a contradiction. No solution.")
                solution_x, solution_y = None, None
        else: # 0x + 0y = non_zero_c1
            steps.append(f"Equation 1 is 0x + 0y = {c1} (non-zero). This is a contradiction. No solution.")
            solution_x, solution_y = None, None
            consistency_type = "inconsistent_parallel" # Or specific contradiction state
    else:
        # This case should not be reached if consistency check is done first and handled cases where one eq is trivial.
        # It implies both a1 and b1 are zero, leading to 0 = c1.
        # If c1 is not zero, it's a contradiction. If c1 is zero, then eq1 provides no info.
        steps.append("Equation 1 is trivial (0 = c1). Solution depends on whether c1 is 0 and on Equation 2.")


    return {
        "method": "Substitution", 
        "steps": steps, 
        "solution_x": solution_x, 
        "solution_y": solution_y,
        "consistency": consistency_type
    }

def solve_by_elimination(
    a1: float, b1: float, c1: float,
    a2: float, b2: float, c2: float
) -> Dict[str, Any]:
    """
    Solves by elimination method and provides step-by-step explanation.
    a1x + b1y = c1  (eq1)
    a2x + b2y = c2  (eq2)
    """
    steps = []
    solution_x, solution_y = None, None
    epsilon = 1e-9
    consistency_type, _, _ = check_consistency(a1, b1, c1, a2, b2, c2)

    if consistency_type == "inconsistent_parallel":
        steps.append("The system is inconsistent (parallel lines) and has no solution.")
        return {"method": "Elimination", "steps": steps, "solution_x": None, "solution_y": None, "consistency": consistency_type}
    if consistency_type == "consistent_infinite":
        steps.append("The system is consistent with infinitely many solutions (coincident lines).")
        return {"method": "Elimination", "steps": steps, "solution_x": "Infinite", "solution_y": "Infinite", "consistency": consistency_type}

    steps.append(f"Equation 1: {a1}x + {b1}y = {c1}")
    steps.append(f"Equation 2: {a2}x + {b2}y = {c2}")

    # Try to eliminate x:
    # Multiply eq1 by a2: a1*a2*x + b1*a2*y = c1*a2
    # Multiply eq2 by a1: a2*a1*x + b2*a1*y = c2*a1
    # Subtract: (b1*a2 - b2*a1)y = c1*a2 - c2*a1
    
    # Or try to eliminate y:
    # Multiply eq1 by b2: a1*b2*x + b1*b2*y = c1*b2
    # Multiply eq2 by b1: a2*b1*x + b2*b1*y = c2*b1
    # Subtract: (a1*b2 - a2*b1)x = c1*b2 - c2*b1 (This is Dx from Cramer's rule)

    # Let's try to eliminate y first (common approach)
    if abs(b1) > epsilon and abs(b2) > epsilon : # Both b1 and b2 are non-zero
        m1 = b2
        m2 = b1
        steps.append(f"To eliminate y, multiply Equation 1 by {m1} and Equation 2 by {m2}:")
        
        eq1_new_a = a1 * m1
        eq1_new_b = b1 * m1
        eq1_new_c = c1 * m1
        steps.append(f"  {m1} * (Eq1) => {eq1_new_a}x + {eq1_new_b}y = {eq1_new_c}  (Eq3)")
        
        eq2_new_a = a2 * m2
        eq2_new_b = b2 * m2
        eq2_new_c = c2 * m2
        steps.append(f"  {m2} * (Eq2) => {eq2_new_a}x + {eq2_new_b}y = {eq2_new_c}  (Eq4)")

        # If signs of new b coefficients are same, subtract. If different, add.
        # (a1*b2 - a2*b1)x = c1*b2 - c2*b1
        if abs(eq1_new_b + eq2_new_b) < epsilon : # signs were opposite or one was made negative
             steps.append("Add Eq3 and Eq4 (as y coefficients have opposite signs or one was made negative):")
             final_x_coeff = eq1_new_a + eq2_new_a
             final_c_val = eq1_new_c + eq2_new_c
        else: # signs are same
             steps.append("Subtract Eq4 from Eq3 (or vice-versa to make x coeff positive if possible):")
             final_x_coeff = eq1_new_a - eq2_new_a
             final_c_val = eq1_new_c - eq2_new_c
        
        steps.append(f"  ({eq1_new_a} - {eq2_new_a})x + ({eq1_new_b} - {eq2_new_b})y = {eq1_new_c} - {eq2_new_c}")
        steps.append(f"  {final_x_coeff}x + 0y = {final_c_val}")
        steps.append(f"  {final_x_coeff}x = {final_c_val}")

        if abs(final_x_coeff) > epsilon:
            solution_x = final_c_val / final_x_coeff
            steps.append(f"  x = {solution_x}")
            # Substitute x into original eq1 (or eq2) to find y
            # a1*x_sol + b1*y = c1  => b1*y = c1 - a1*x_sol
            if abs(b1) > epsilon:
                rhs_y = c1 - a1 * solution_x
                solution_y = rhs_y / b1
                steps.append(f"Substitute x = {solution_x} into Equation 1 ({a1}x + {b1}y = {c1}):")
                steps.append(f"  {a1*solution_x} + {b1}y = {c1}")
                steps.append(f"  {b1}y = {c1 - a1*solution_x}")
                steps.append(f"  y = {solution_y}")
            elif abs(b2) > epsilon: # Use eq2 if b1 is 0
                rhs_y = c2 - a2 * solution_x
                solution_y = rhs_y / b2
                steps.append(f"Substitute x = {solution_x} into Equation 2 ({a2}x + {b2}y = {c2}):")
                steps.append(f"  {a2*solution_x} + {b2}y = {c2}")
                steps.append(f"  {b2}y = {c2 - a2*solution_x}")
                steps.append(f"  y = {solution_y}")
            else: # Both b1 and b2 are zero - should have been caught by consistency
                steps.append("Error: Both b1 and b2 are zero, cannot find unique y this way.")
        else: # final_x_coeff is 0
            steps.append(f"This simplifies to 0x = {final_c_val}.")
            if abs(final_c_val) > epsilon: # 0 = non-zero
                steps.append("This is a contradiction (0 = non-zero), so no solution (parallel lines).")
            else: # 0 = 0
                steps.append("This is 0 = 0, indicating infinitely many solutions (coincident lines).")

    elif abs(a1) > epsilon and abs(a2) > epsilon : # Eliminate x, if eliminating y was not straightforward
        m1 = a2
        m2 = a1
        steps.append(f"To eliminate x, multiply Equation 1 by {m1} and Equation 2 by {m2}:")
        # Similar logic as above, but solving for y first
        eq1_new_a = a1 * m1; eq1_new_b = b1 * m1; eq1_new_c = c1 * m1
        steps.append(f"  {m1} * (Eq1) => {eq1_new_a}x + {eq1_new_b}y = {eq1_new_c}  (Eq3)")
        eq2_new_a = a2 * m2; eq2_new_b = b2 * m2; eq2_new_c = c2 * m2
        steps.append(f"  {m2} * (Eq2) => {eq2_new_a}x + {eq2_new_b}y = {eq2_new_c}  (Eq4)")

        steps.append("Subtract Eq4 from Eq3:")
        final_y_coeff = eq1_new_b - eq2_new_b # (b1a2 - b2a1)
        final_c_val_y = eq1_new_c - eq2_new_c # (c1a2 - c2a1)
        steps.append(f"  ({eq1_new_b} - {eq2_new_b})y = {eq1_new_c} - {eq2_new_c}")
        steps.append(f"  {final_y_coeff}y = {final_c_val_y}")

        if abs(final_y_coeff) > epsilon:
            solution_y = final_c_val_y / final_y_coeff
            steps.append(f"  y = {solution_y}")
            if abs(a1) > epsilon:
                solution_x = (c1 - b1 * solution_y) / a1
                steps.append(f"Substitute y = {solution_y} into Equation 1 to find x: x = {solution_x}")
            elif abs(a2) > epsilon:
                solution_x = (c2 - b2 * solution_y) / a2
                steps.append(f"Substitute y = {solution_y} into Equation 2 to find x: x = {solution_x}")
        else: # Coeff is 0
            steps.append(f"This simplifies to 0y = {final_c_val_y}.")
            if abs(final_c_val_y) > epsilon: steps.append("Contradiction, no solution.")
            else: steps.append("0 = 0, infinitely many solutions.")
    else:
        # Handle cases where one equation is very simple e.g. x = 5 or y = 3
        # Or if a1=0, b1=0 etc. (trivial equations)
        # This part might need more refinement for edge cases not covered by initial consistency.
        # For now, rely on the general solver if elimination steps become too complex here.
        steps.append("Simpler case or edge case encountered. Attempting general solution (from numpy or Cramer's rule if unique).")
        general_sol = solve_linear_equations(a1,b1,c1,a2,b2,c2)
        solution_x = general_sol.get("solution_x")
        solution_y = general_sol.get("solution_y")
        if solution_x is not None and solution_y is not None:
            steps.append(f"General solver found x = {solution_x}, y = {solution_y}")
        else:
            steps.append(general_sol.get("description"))

    return {
        "method": "Elimination", 
        "steps": steps, 
        "solution_x": solution_x, 
        "solution_y": solution_y,
        "consistency": consistency_type
    }


def solve_by_cross_multiplication(
    a1: float, b1: float, c1_orig: float, # c1_orig is from ax+by+c=0
    a2: float, b2: float, c2_orig: float  # c2_orig is from ax+by+c=0
) -> Dict[str, Any]:
    """
    Solves by cross-multiplication method: x/(b1c2-b2c1) = y/(c1a2-c2a1) = 1/(a1b2-a2b1)
    Requires equations in form a1x + b1y + c1 = 0 and a2x + b2y + c2 = 0.
    c1_orig and c2_orig are used here for the standard cross-multiplication formula.
    """
    steps = []
    solution_x, solution_y = None, None
    
    # Check consistency first using a1x+b1y = -c1_orig
    consistency_type, _, _ = check_consistency(a1, b1, -c1_orig, a2, b2, -c2_orig)

    if consistency_type == "inconsistent_parallel":
        steps.append("The system is inconsistent (parallel lines) and has no solution.")
        return {"method": "Cross-Multiplication", "steps": steps, "solution_x": None, "solution_y": None, "consistency": consistency_type}
    if consistency_type == "consistent_infinite":
        steps.append("The system is consistent with infinitely many solutions (coincident lines).")
        return {"method": "Cross-Multiplication", "steps": steps, "solution_x": "Infinite", "solution_y": "Infinite", "consistency": consistency_type}

    # Denominators for cross-multiplication:
    # D_xy = a1*b2 - a2*b1  (denominator for 1)
    # D_x = b1*c2_orig - b2*c1_orig (numerator for x, along with D_xy)
    # D_y = c1_orig*a2 - c2_orig*a1 (numerator for y, along with D_xy)

    D_xy = a1 * b2 - a2 * b1
    D_x = b1 * c2_orig - b2 * c1_orig
    D_y = c1_orig * a2 - c2_orig * a1
    
    steps.append(f"Equations: {a1}x + {b1}y + {c1_orig} = 0  AND  {a2}x + {b2}y + {c2_orig} = 0")
    steps.append("Using cross-multiplication formula: x / (b₁c₂ - b₂c₁) = y / (c₁a₂ - c₂a₁) = 1 / (a₁b₂ - a₂b₁)")
    steps.append(f"  b₁c₂ - b₂c₁ = ({b1})*({c2_orig}) - ({b2})*({c1_orig}) = {D_x}")
    steps.append(f"  c₁a₂ - c₂a₁ = ({c1_orig})*({a2}) - ({c2_orig})*({a1}) = {D_y}")
    steps.append(f"  a₁b₂ - a₂b₁ = ({a1})*({b2}) - ({a2})*({b1}) = {D_xy}")
    steps.append(f"So, x / {D_x} = y / {D_y} = 1 / {D_xy}")

    epsilon = 1e-9
    if abs(D_xy) > epsilon: # Unique solution
        solution_x = D_x / D_xy
        solution_y = D_y / D_xy
        steps.append(f"From x / {D_x} = 1 / {D_xy}  => x = {D_x} / {D_xy} = {solution_x}")
        steps.append(f"From y / {D_y} = 1 / {D_xy}  => y = {D_y} / {D_xy} = {solution_y}")
    else: # D_xy is zero, means lines are parallel or coincident
        # This case should have been handled by the initial consistency check.
        # If D_xy = 0 and D_x != 0 (or D_y != 0), then inconsistent (parallel) e.g., x/k = y/m = 1/0
        # If D_xy = 0 and D_x = 0 and D_y = 0, then consistent infinite e.g., x/0 = y/0 = 1/0 (misleading form)
        # The consistency check is more reliable here.
        steps.append(f"Denominator (a₁b₂ - a₂b₁) is {D_xy} (close to zero). System is not uniquely solvable.")
        steps.append(f"Based on consistency check: {consistency_type}")

    return {
        "method": "Cross-Multiplication", 
        "steps": steps, 
        "solution_x": solution_x, 
        "solution_y": solution_y,
        "consistency": consistency_type
    }