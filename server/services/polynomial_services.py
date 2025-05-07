# services/polynomial_services.py
import math
from typing import List, Tuple, Dict, Any, Union, Optional
from sympy import Poly, sympify, SympifyError, symbols, real_roots as sympy_real_roots, nroots as sympy_nroots, expand
import cmath # For complex number arithmetic

# Helper for coefficient representation: List[float], highest power first.
# e.g., 2x^3 - 4x + 5  => [2, 0, -4, 5]

def parse_polynomial_string(poly_str: str, var_symbol: str = 'x') -> List[float]:
    """
    Parses a polynomial string (e.g., "2*x**3 - 4*x + 5") into a list of coefficients.
    Uses sympy for robust parsing.
    Returns coefficients from highest degree to constant term.
    """
    x = symbols(var_symbol)
    try:
        expr = sympify(poly_str, locals={var_symbol: x})
        if not expr.is_polynomial(x):
            # Check if it's a constant
            if expr.is_constant():
                 return [float(expr)]
            raise ValueError("Expression is not a polynomial in the specified variable.")
        
        poly_obj = Poly(expr, x)
        coeffs = [float(c) for c in poly_obj.all_coeffs()]
        return coeffs
    except (SympifyError, TypeError, AttributeError) as e: # TypeError for things like "x + sin(x)"
        raise ValueError(f"Invalid polynomial expression: {str(e)}")


def polynomial_to_string(coeffs: List[float], var_symbol: str = 'x') -> str:
    """Converts a list of coefficients back to a string representation."""
    if not coeffs:
        return "0"
    degree = len(coeffs) - 1
    terms = []
    for i, coeff in enumerate(coeffs):
        if abs(coeff) < 1e-9:  # Skip zero coefficients
            continue
        power = degree - i
        term = ""
        # Coefficient part
        if power < degree: # Not the first term
            if coeff > 0:
                term += " + "
            else:
                term += " - " # Negative sign
            coeff = abs(coeff)

        if abs(coeff - 1.0) > 1e-9 or power == 0: # Print 1 only if it's a constant term
            # term += f"{coeff:.4g}" # .4g for nice formatting
            if coeff == int(coeff):
                term += str(int(coeff))
            else:
                term += f"{coeff:.4f}".rstrip('0').rstrip('.') # More precise, remove trailing zeros

        # Variable part
        if power > 0:
            term += var_symbol
            if power > 1:
                term += f"**{power}"
        terms.append(term.strip())
    
    result = "".join(terms)
    if result.startswith(" + "):
        result = result[3:]
    if result.startswith(" - "): # if first term was negative, sympy might put - coeff, so this is ok
        pass # Keep the leading minus for negative first term
    
    return result if result else "0"


def evaluate_polynomial(coeffs: List[float], x_val: Union[float, complex]) -> Union[float, complex]:
    """Evaluates P(x_val) using Horner's method."""
    result = 0
    for coeff in coeffs:
        result = result * x_val + coeff
    return result

def polynomial_division(dividend_coeffs: List[float], divisor_coeffs: List[float]) -> Tuple[List[float], List[float]]:
    """
    Performs polynomial long division: dividend / divisor.
    Returns (quotient_coeffs, remainder_coeffs).
    Assumes dividend_coeffs and divisor_coeffs are not empty and divisor is not zero polynomial.
    """
    if not divisor_coeffs or all(abs(c) < 1e-9 for c in divisor_coeffs):
        raise ValueError("Divisor polynomial cannot be zero.")
    if not dividend_coeffs: # 0 / P(x)
        return [0.0], [0.0]

    # Normalize divisor if its leading coefficient is not 1 (optional, can simplify, but direct works)
    # For numerical stability, it's often better to work with original coeffs.

    deg_dividend = len(dividend_coeffs) - 1
    deg_divisor = len(divisor_coeffs) - 1

    if deg_dividend < deg_divisor:
        return [0.0], list(dividend_coeffs) # Quotient is 0, remainder is dividend

    quotient_coeffs = [0.0] * (deg_dividend - deg_divisor + 1)
    remainder_coeffs = list(dividend_coeffs) # Start with dividend as remainder

    for i in range(len(quotient_coeffs)):
        # Leading coefficient of current remainder divided by leading coefficient of divisor
        coeff = remainder_coeffs[i] / divisor_coeffs[0]
        quotient_coeffs[i] = coeff
        
        # Subtract coeff * divisor from the current remainder
        for j in range(len(divisor_coeffs)):
            remainder_coeffs[i + j] -= coeff * divisor_coeffs[j]
            
    # The remainder is the last deg_divisor terms of remainder_coeffs (or fewer if it became zero)
    # The first part of remainder_coeffs (up to len(quotient)) should be zeroed out by the process.
    # The actual remainder starts after the part corresponding to quotient.
    # A more careful slicing for remainder:
    num_leading_zeros = deg_dividend - deg_divisor +1
    final_remainder = remainder_coeffs[num_leading_zeros:]
    
    # Clean up leading zeros in remainder if any (e.g., if exact division)
    first_non_zero_rem = 0
    while first_non_zero_rem < len(final_remainder) and abs(final_remainder[first_non_zero_rem]) < 1e-9:
        first_non_zero_rem += 1
    final_remainder = final_remainder[first_non_zero_rem:]
    if not final_remainder:
        final_remainder = [0.0]

    return quotient_coeffs, final_remainder


def synthetic_division(dividend_coeffs: List[float], a_val: Union[float, complex]) -> Tuple[List[float], Union[float, complex]]:
    """
    Performs synthetic division of P(x) by (x - a_val).
    Returns (quotient_coeffs, remainder).
    Remainder is P(a_val).
    """
    if not dividend_coeffs:
        return [0.0], 0.0

    n = len(dividend_coeffs)
    quotient_coeffs = [0.0] * (n - 1)
    
    if not quotient_coeffs: # Dividend was a constant
        return [0.0], dividend_coeffs[0]

    quotient_coeffs[0] = dividend_coeffs[0]
    for i in range(1, n - 1):
        quotient_coeffs[i] = dividend_coeffs[i] + quotient_coeffs[i-1] * a_val
    
    remainder = dividend_coeffs[n-1] + quotient_coeffs[n-2] * a_val
    
    # Handle case where dividend is linear e.g. [a, b] for ax+b
    # n=2, n-1=1. quotient_coeffs = [0.0]
    # quotient_coeffs[0] = dividend_coeffs[0] = a
    # loop range(1,1) doesn't run.
    # remainder = dividend_coeffs[1] + quotient_coeffs[0] * a_val = b + a * a_val
    # This is correct: (ax+b) / (x-a_val) => quotient a, remainder a*a_val + b
    
    return quotient_coeffs, remainder


def solve_quadratic_equation(a: float, b: float, c: float) -> Dict[str, Any]:
    """
    Solves ax^2 + bx + c = 0.
    Returns roots (real or complex), discriminant, and nature of roots.
    """
    if abs(a) < 1e-9: # Not a quadratic equation if a is zero
        if abs(b) < 1e-9: # 0x = -c
            if abs(c) < 1e-9:
                return {"nature": "identity", "description": "0x + 0 = 0 (identity, true for all x)", "roots": ["all_real_numbers"]}
            else:
                return {"nature": "contradiction", "description": f"0 = {-c} (contradiction, no solution)", "roots": []}
        # Linear equation: bx + c = 0 => x = -c/b
        root = -c / b
        return {
            "nature": "linear", 
            "description": f"Linear equation: {b}x + {c} = 0",
            "roots": [root], 
            "discriminant": None
        }

    discriminant = b**2 - 4*a*c
    roots = []
    nature_of_roots = ""

    if discriminant > 1e-9: # Using epsilon for float comparison
        nature_of_roots = "Two distinct real roots"
        sqrt_D = math.sqrt(discriminant)
        roots.append((-b + sqrt_D) / (2*a))
        roots.append((-b - sqrt_D) / (2*a))
    elif abs(discriminant) < 1e-9: # D is effectively zero
        nature_of_roots = "Two equal real roots (repeated root)"
        roots.append(-b / (2*a))
        roots.append(-b / (2*a)) # Or just one entry
    else: # D < 0
        nature_of_roots = "Two complex conjugate roots"
        sqrt_neg_D = cmath.sqrt(-discriminant) # cmath.sqrt handles negative input correctly
        real_part = -b / (2*a)
        imag_part = sqrt_neg_D / (2*a)
        roots.append(complex(real_part, imag_part))
        roots.append(complex(real_part, -imag_part))
        # Convert complex numbers to strings for easier JSON serialization
        roots = [str(r) for r in roots]


    return {
        "coefficients": {"a": a, "b": b, "c": c},
        "discriminant": discriminant,
        "nature_of_roots": nature_of_roots,
        "roots": roots,
        "formula_used": "x = [-b ± sqrt(b²-4ac)] / (2a)"
    }


def relation_roots_coeffs_quadratic(coeffs: List[float]) -> Optional[Dict[str, Any]]:
    """For ax^2+bx+c, finds sum (-b/a) and product (c/a) of roots."""
    if len(coeffs) != 3:
        return None # Not a quadratic
    a, b, c = coeffs[0], coeffs[1], coeffs[2]
    if abs(a) < 1e-9: return None # Not quadratic

    sum_roots = -b / a
    prod_roots = c / a
    return {
        "alpha_plus_beta": sum_roots,
        "alpha_beta": prod_roots,
        "verification_note": "These are theoretical values based on coefficients."
    }

def relation_roots_coeffs_cubic(coeffs: List[float]) -> Optional[Dict[str, Any]]:
    """For ax^3+bx^2+cx+d, finds relations."""
    if len(coeffs) != 4:
        return None
    a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
    if abs(a) < 1e-9: return None

    sum_roots = -b / a  # α+β+γ
    sum_prod_pairs = c / a # αβ+βγ+γα
    prod_roots = -d / a # αβγ
    return {
        "alpha_plus_beta_plus_gamma": sum_roots,
        "alpha_beta_plus_beta_gamma_plus_gamma_alpha": sum_prod_pairs,
        "alpha_beta_gamma": prod_roots,
        "verification_note": "These are theoretical values based on coefficients."
    }


def find_rational_roots(coeffs: List[float]) -> List[float]:
    """
    Finds rational roots of a polynomial with integer coefficients using Rational Root Theorem.
    P(x) = a_n x^n + ... + a_1 x + a_0
    If p/q is a root, p divides a_0 and q divides a_n.
    """
    # Ensure coefficients are integers for RRT to strictly apply.
    # If they are floats that are very close to integers, we can round them.
    # For general float coeffs, this theorem doesn't directly apply without modification.
    # Let's assume for this function that the user intends integer coefficients.
    
    # Check if coeffs are effectively integers
    int_coeffs = []
    for c in coeffs:
        if abs(c - round(c)) > 1e-9:
            # For API, could raise error or try to find common denominator to make them integers.
            # For now, we'll proceed, but RRT is for integer coeffs.
             print(f"Warning: Coefficient {c} is not an integer. Rational Root Theorem strictly applies to integer coefficients.")
        int_coeffs.append(int(round(c)))

    if not int_coeffs or len(int_coeffs) == 1: # Constant polynomial
        return []
    
    a0 = int_coeffs[-1] # Constant term
    an = int_coeffs[0]  # Leading coefficient

    if abs(an) < 1e-9 : return [] # Leading coeff is zero, should not happen if parsed correctly
    if abs(a0) < 1e-9: # Constant term is zero, so x=0 is a root
        # We can factor out x and apply RRT to the rest
        # For simplicity now, just add 0 and let user know.
        # A more robust approach would be to divide by x and recurse.
        # Test P(0) with original (float) coeffs
        if abs(evaluate_polynomial(coeffs, 0)) < 1e-9:
            # Recursively find roots for P(x)/x
            remaining_coeffs = coeffs[:-1]
            rational_roots_remaining = find_rational_roots(remaining_coeffs)
            return [0.0] + rational_roots_remaining # ensure 0.0 is float
        else: # Should not happen if a0 is zero.
            return []


    def get_divisors(n: int) -> List[int]:
        n_abs = abs(n)
        if n_abs == 0: return [] # or [0] if you consider 0 divides 0 for some contexts
        divs = set()
        for i in range(1, int(math.sqrt(n_abs)) + 1):
            if n_abs % i == 0:
                divs.add(i)
                divs.add(-i)
                divs.add(n_abs // i)
                divs.add(-(n_abs // i))
        return sorted(list(divs))

    p_divisors = get_divisors(a0)
    q_divisors = [d for d in get_divisors(an) if d != 0] # q cannot be zero

    if not p_divisors or not q_divisors: # e.g. if a0=0 and an is non-zero
        if abs(a0) < 1e-9 and abs(evaluate_polynomial(coeffs, 0)) < 1e-9: # x=0 is a root
            return [0.0] 
        return []


    potential_rational_roots = set()
    for p_val in p_divisors:
        for q_val in q_divisors:
            potential_rational_roots.add(p_val / q_val)
    
    actual_rational_roots = []
    for root_candidate in sorted(list(potential_rational_roots)):
        # Evaluate polynomial with the candidate (use original float coeffs for precision)
        if abs(evaluate_polynomial(coeffs, root_candidate)) < 1e-9: # Tolerance for float comparison
            actual_rational_roots.append(root_candidate)
            
    return actual_rational_roots

def find_all_roots_numeric(coeffs: List[float], attempts: int = 10) -> List[str]:
    """
    Attempts to find all roots (real and complex) numerically.
    Uses sympy.nroots as a helper for this, as robust numerical root finding is complex.
    """
    if not coeffs or len(coeffs) <= 1: # Constant or empty
        return []
    try:
        # nroots can sometimes be sensitive to float precision.
        # Using a few attempts with slightly perturbed coefficients can sometimes help
        # for ill-conditioned polynomials, but this is advanced.
        # For now, direct call.
        poly_str = polynomial_to_string(coeffs) # Convert back to string for Poly
        x = symbols('x')
        poly_obj = Poly(sympify(poly_str, locals={'x':x}), x)
        
        # sympy.nroots for numerical complex roots
        # For real roots only: sympy.real_roots(poly_obj)
        # roots_found = sympy_nroots(poly_obj, n=15) # n is decimal precision
        
        # sympy's Poly object has .nroots() method
        roots_found = poly_obj.nroots(n=15) # 15 digits of precision
        
        return [str(r) for r in roots_found]
    except Exception as e:
        return [f"Error finding numerical roots: {e}"]

# (Further functions like polynomial differentiation, Descartes' Rule, etc. can be added)