# services/polynomial_services.py
import math
from typing import List, Tuple, Dict, Any, Union, Optional
from sympy import Poly as SympyPoly, sympify, SympifyError, symbols # Renamed Poly to SympyPoly to avoid clash
# from sympy import expand # We will avoid this for polynomial multiplication
import cmath

# Helper for coefficient representation: List[float], highest power first.
# e.g., 2x^3 - 4x + 5  => [2, 0, -4, 5]

def parse_polynomial_string(poly_str: str, var_symbol: str = 'x') -> List[Union[float, complex]]:
    """
    Parses a polynomial string (e.g., "2*x**3 - 4*x + 5") into a list of coefficients.
    Uses sympy for robust parsing.
    Returns coefficients from highest degree to constant term.
    Coefficients can be complex if the input string implies it.
    """
    x = symbols(var_symbol)
    try:
        expr = sympify(poly_str, locals={var_symbol: x})
        # Check if it's a constant, which sympify might make a Sympy number type
        if expr.is_Atom and not expr.is_Symbol: # Catches numbers, I (imaginary unit)
            if expr.is_imaginary or expr.is_complex:
                return [complex(expr.evalf())]
            return [float(expr.evalf())]

        if not expr.is_polynomial(x):
            raise ValueError("Expression is not a polynomial in the specified variable.")
        
        poly_obj = SympyPoly(expr, x)
        # Coefficients can be complex from sympy, convert to Python complex or float
        coeffs = []
        for c in poly_obj.all_coeffs():
            if c.is_complex or I in c.free_symbols: # I is sympy's imaginary unit
                coeffs.append(complex(c.evalf()))
            else:
                coeffs.append(float(c.evalf()))
        return coeffs
    except (SympifyError, TypeError, AttributeError) as e:
        raise ValueError(f"Invalid polynomial expression: {str(e)}")


def polynomial_to_string(coeffs: List[Union[float, complex]], var_symbol: str = 'x') -> str:
    """Converts a list of coefficients back to a string representation. Handles complex coeffs."""
    if not coeffs:
        return "0"
    
    # Remove leading zero coefficients for proper degree calculation
    # (unless it's just [0.0] or [0j])
    first_non_zero_idx = 0
    while first_non_zero_idx < len(coeffs) -1 and abs(coeffs[first_non_zero_idx]) < 1e-9 :
        first_non_zero_idx += 1
    
    coeffs_cleaned = coeffs[first_non_zero_idx:]
    if not coeffs_cleaned: # All were zero
        return "0"


    degree = len(coeffs_cleaned) - 1
    terms = []

    for i, coeff_val in enumerate(coeffs_cleaned):
        if abs(coeff_val) < 1e-9:  # Skip zero coefficients
            continue
        
        power = degree - i
        term_str = ""

        is_complex_coeff = isinstance(coeff_val, complex)
        real_part = coeff_val.real if is_complex_coeff else coeff_val
        imag_part = coeff_val.imag if is_complex_coeff else 0.0

        # Sign handling (for non-first terms)
        if i > 0: # Not the first term displayed
            if real_part > 1e-9 or (abs(real_part) < 1e-9 and imag_part > 1e-9):
                term_str += " + "
            elif real_part < -1e-9 or (abs(real_part) < 1e-9 and imag_part < -1e-9):
                term_str += " - " # The number itself will be positive after this
                real_part = -real_part
                imag_part = -imag_part
            else: # Purely imaginary term that was negative, or real_part is zero
                if imag_part < -1e-9: # and real_part is zero
                     term_str += " - "
                     imag_part = -imag_part
                elif imag_part > 1e-9: # and real_part is zero
                     term_str += " + "
                # If both real and imag are effectively zero, it's already skipped.


        # Coefficient string
        coeff_str = ""
        if is_complex_coeff:
            # (real_val +/- imag_val*j)
            # Handle if purely real part or purely imaginary part
            has_real = abs(real_part) > 1e-9
            has_imag = abs(imag_part) > 1e-9

            if has_real:
                coeff_str += f"{real_part:.4g}".rstrip('0').rstrip('.')
            
            if has_imag:
                if has_real and imag_part > 0: # Add plus if real part exists and imag is positive
                    coeff_str += "+"
                # No extra sign if imag_part is negative, it's handled by its own sign
                # Or if it's the first part of the complex number
                
                if abs(abs(imag_part) - 1.0) < 1e-9 : # if imag_part is 1 or -1
                    coeff_str += "j" if imag_part > 0 else ("-j" if not has_real else "j") # handle -j if no real part
                    if imag_part < 0 and not has_real: coeff_str = "-j" # Ensure -j if purely -1j
                else:
                    coeff_str += f"{imag_part:.4g}".rstrip('0').rstrip('.') + "j"
            
            if has_real or has_imag: # Need parentheses if complex and not just a number
                 # and if it's not the only part of the term (i.e., has a variable)
                if power > 0 and (has_real and has_imag): # e.g. (2+3j)x
                     coeff_str = f"({coeff_str})"
        else: # Real coefficient
            # If it's the first term and negative, the sign is part of the number
            if i == 0 and real_part < 0:
                coeff_str = f"{real_part:.4g}".rstrip('0').rstrip('.')
            else: # Subsequent terms, or first term positive
                # Print 1 only if it's a constant term or the only coeff
                if abs(abs(real_part) - 1.0) > 1e-9 or power == 0:
                    coeff_str = f"{abs(real_part):.4g}".rstrip('0').rstrip('.')
                # else: coeff_str is empty for coeff 1 or -1 (sign already handled)

        term_str += coeff_str

        # Variable part
        if power > 0:
            if coeff_str == "" and abs(abs(real_part)-1.0) < 1e-9 : # Handle "x" not "1x"
                pass # No numerical coefficient string, sign handled
            elif coeff_str and term_str and not term_str.endswith(" ") and not term_str.endswith(")"):
                 # Add space if coeff_str exists and not complex in parens: e.g. "2 x"
                 # This might be too much, usually "2x" is fine.
                 pass


            term_str += var_symbol
            if power > 1:
                term_str += f"**{power}"
        
        terms.append(term_str)
    
    result = "".join(filter(None, terms)) # Filter out empty strings from skipped 1 coefficients
    if not result: return "0"
    
    # Final cleanup of leading " + " if any
    if result.startswith(" + "):
        result = result[3:]
    elif result.startswith(" - "): # First term was negative, sign already included
        pass
    
    return result if result else "0"


def evaluate_polynomial(coeffs: List[Union[float, complex]], x_val: Union[float, complex]) -> Union[float, complex]:
    """Evaluates P(x_val) using Horner's method."""
    result: Union[float, complex] = 0 # Ensure type for result
    for coeff in coeffs:
        result = result * x_val + coeff
    return result

def polynomial_division(dividend_coeffs: List[Union[float, complex]], divisor_coeffs: List[Union[float, complex]]) -> Tuple[List[Union[float, complex]], List[Union[float, complex]]]:
    """
    Performs polynomial long division: dividend / divisor. Handles complex coefficients.
    Returns (quotient_coeffs, remainder_coeffs).
    """
    if not divisor_coeffs or all(abs(c) < 1e-9 for c in divisor_coeffs):
        raise ValueError("Divisor polynomial cannot be zero.")
    if not dividend_coeffs:
        return [0.0], [0.0]

    deg_dividend = len(dividend_coeffs) - 1
    deg_divisor = len(divisor_coeffs) - 1

    if deg_dividend < deg_divisor:
        return [complex(0.0)], list(dividend_coeffs) # Quotient is 0, remainder is dividend

    quotient_coeffs: List[Union[float, complex]] = [complex(0.0)] * (deg_dividend - deg_divisor + 1)
    # Make a mutable copy for remainder that supports complex numbers
    current_remainder_coeffs: List[Union[float, complex]] = [complex(c) for c in dividend_coeffs]


    lead_divisor_coeff = divisor_coeffs[0]
    if abs(lead_divisor_coeff) < 1e-9: # Should be caught by initial check but good to have
        raise ValueError("Leading coefficient of divisor is zero.")

    for i in range(len(quotient_coeffs)):
        coeff = current_remainder_coeffs[i] / lead_divisor_coeff
        quotient_coeffs[i] = coeff
        
        for j in range(len(divisor_coeffs)):
            if (i + j) < len(current_remainder_coeffs): # Boundary check
                current_remainder_coeffs[i + j] -= coeff * divisor_coeffs[j]
            
    # The remainder is the part of current_remainder_coeffs after the quotient part
    num_quotient_terms = deg_dividend - deg_divisor + 1
    final_remainder = current_remainder_coeffs[num_quotient_terms:]
    
    # Clean up leading zeros in remainder
    first_non_zero_rem_idx = 0
    while first_non_zero_rem_idx < len(final_remainder) and abs(final_remainder[first_non_zero_rem_idx]) < 1e-9:
        first_non_zero_rem_idx += 1
    final_remainder = final_remainder[first_non_zero_rem_idx:]

    if not final_remainder:
        final_remainder = [complex(0.0)]

    return quotient_coeffs, final_remainder


def synthetic_division(dividend_coeffs: List[Union[float, complex]], a_val: Union[float, complex]) -> Tuple[List[Union[float, complex]], Union[float, complex]]:
    """
    Performs synthetic division of P(x) by (x - a_val). Handles complex numbers.
    Returns (quotient_coeffs, remainder). Remainder is P(a_val).
    """
    if not dividend_coeffs:
        return [complex(0.0)], complex(0.0)

    n = len(dividend_coeffs)
    if n == 1: # Dividend is a constant
        return [complex(0.0)], dividend_coeffs[0] # Quotient is 0, remainder is the constant

    quotient_coeffs: List[Union[float, complex]] = [complex(0.0)] * (n - 1)
    
    quotient_coeffs[0] = dividend_coeffs[0]
    for i in range(1, n - 1):
        quotient_coeffs[i] = dividend_coeffs[i] + quotient_coeffs[i-1] * a_val
    
    remainder = dividend_coeffs[n-1] + quotient_coeffs[n-2] * a_val
    
    return quotient_coeffs, remainder


def solve_quadratic_equation(a: float, b: float, c: float) -> Dict[str, Any]:
    """
    Solves ax^2 + bx + c = 0.
    Returns roots (real or complex), discriminant, and nature of roots.
    """
    solution_payload = {
        "discriminant": None,
        "nature_of_roots": "", # Ensure this is always set
        "roots": [],
        "formula_used": "x = [-b ± sqrt(b²-4ac)] / (2a)" # Default, may change for linear
    }

    if abs(a) < 1e-9: # Not a quadratic equation if a is zero
        solution_payload["formula_used"] = "Linear equation solver"
        if abs(b) < 1e-9: # 0x = -c
            if abs(c) < 1e-9:
                solution_payload["nature_of_roots"] = "identity (0x + 0 = 0)"
                solution_payload["description"] = "Identity, true for all x."
                solution_payload["roots"] = ["all_real_numbers"]
            else:
                solution_payload["nature_of_roots"] = "contradiction (0 = non-zero)"
                solution_payload["description"] = f"Contradiction (0 = {-c}), no solution."
                solution_payload["roots"] = []
        else: # Linear equation: bx + c = 0 => x = -c/b
            root = -c / b
            solution_payload["nature_of_roots"] = "linear equation"
            solution_payload["description"] = f"Linear equation: {b}x + {c} = 0"
            solution_payload["roots"] = [root]
        return solution_payload # Return early for linear/contradiction/identity

    discriminant = b**2 - 4*a*c
    solution_payload["discriminant"] = discriminant
    roots = []

    if discriminant > 1e-9:
        solution_payload["nature_of_roots"] = "Two distinct real roots"
        sqrt_D = math.sqrt(discriminant)
        roots.append((-b + sqrt_D) / (2*a))
        roots.append((-b - sqrt_D) / (2*a))
    elif abs(discriminant) < 1e-9:
        solution_payload["nature_of_roots"] = "Two equal real roots (repeated root)"
        roots.append(-b / (2*a))
        # roots.append(-b / (2*a)) # Pydantic model expects List, so one is fine if we describe it as repeated
    else: # D < 0
        solution_payload["nature_of_roots"] = "Two complex conjugate roots"
        sqrt_neg_D = cmath.sqrt(-discriminant)
        real_part = -b / (2*a)
        imag_part = sqrt_neg_D / (2*a)
        roots.append(complex(real_part, imag_part))
        roots.append(complex(real_part, -imag_part))
    
    # Store roots appropriately (float or string for complex)
    solution_payload["roots"] = [str(r) if isinstance(r, complex) else r for r in roots]
    return solution_payload


def relation_roots_coeffs_quadratic(coeffs: List[Union[float, complex]]) -> Optional[Dict[str, Any]]:
    """For ax^2+bx+c, finds sum (-b/a) and product (c/a) of roots."""
    if len(coeffs) != 3:
        return None 
    a, b, c = coeffs[0], coeffs[1], coeffs[2]
    if abs(a) < 1e-9: return None 

    sum_roots = -b / a
    prod_roots = c / a
    return {
        "alpha_plus_beta": str(sum_roots) if isinstance(sum_roots, complex) else sum_roots,
        "alpha_beta": str(prod_roots) if isinstance(prod_roots, complex) else prod_roots,
        "verification_note": "These are theoretical values based on coefficients."
    }

def relation_roots_coeffs_cubic(coeffs: List[Union[float, complex]]) -> Optional[Dict[str, Any]]:
    """For ax^3+bx^2+cx+d, finds relations."""
    if len(coeffs) != 4:
        return None
    a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
    if abs(a) < 1e-9: return None

    sum_roots = -b / a
    sum_prod_pairs = c / a
    prod_roots = -d / a
    return {
        "alpha_plus_beta_plus_gamma": str(sum_roots) if isinstance(sum_roots, complex) else sum_roots,
        "alpha_beta_plus_beta_gamma_plus_gamma_alpha": str(sum_prod_pairs) if isinstance(sum_prod_pairs, complex) else sum_prod_pairs,
        "alpha_beta_gamma": str(prod_roots) if isinstance(prod_roots, complex) else prod_roots,
        "verification_note": "These are theoretical values based on coefficients."
    }

def find_rational_roots(coeffs: List[Union[float, complex]]) -> List[float]:
    """
    Finds rational roots of a polynomial.
    If coefficients are complex, this theorem is not directly applicable.
    Assumes real coefficients for Rational Root Theorem logic.
    """
    # Check for complex coefficients, RRT primarily for integer/rational real coeffs.
    if any(isinstance(c, complex) and abs(c.imag) > 1e-9 for c in coeffs):
        print("Warning: Rational Root Theorem is best suited for polynomials with real (ideally integer) coefficients. Results may be unreliable for complex coefficients.")
        # Optionally, could return empty list or raise error.
        # For now, proceed by considering only real parts, which is a heuristic.
        real_coeffs_for_rrt = [c.real if isinstance(c, complex) else c for c in coeffs]
    else:
        real_coeffs_for_rrt = [c.real if isinstance(c, complex) else c for c in coeffs]


    int_coeffs = []
    for c_real in real_coeffs_for_rrt:
        if abs(c_real - round(c_real)) > 1e-9:
             print(f"Warning: Coefficient {c_real} is not an integer. Rational Root Theorem strictly applies to integer coefficients.")
        int_coeffs.append(int(round(c_real)))

    if not int_coeffs or len(int_coeffs) <= 1:
        return []
    
    a0 = int_coeffs[-1]
    an = int_coeffs[0]

    if abs(an) < 1e-9 : return []
    if abs(a0) < 1e-9:
        # Evaluate with original full (potentially complex) coeffs
        if abs(evaluate_polynomial(coeffs, 0)) < 1e-9:
            remaining_coeffs_orig = coeffs[:-1]
            # If remaining_coeffs_orig is empty (was linear ax=0), then 0 is the only root from this step.
            if not remaining_coeffs_orig:
                return [0.0]
            rational_roots_remaining = find_rational_roots(remaining_coeffs_orig)
            # Avoid duplicate zeros if RRT on remainder also finds 0 (e.g., x^2)
            unique_roots = {0.0}
            for r in rational_roots_remaining:
                unique_roots.add(r)
            return sorted(list(unique_roots))
        else:
            return []

    def get_divisors(n: int) -> List[int]:
        n_abs = abs(n)
        if n_abs == 0: return [0] # Special case for a0=0, p can be 0
        divs = set()
        for i in range(1, int(math.sqrt(n_abs)) + 1):
            if n_abs % i == 0:
                divs.add(i); divs.add(-i)
                divs.add(n_abs // i); divs.add(-(n_abs // i))
        return sorted(list(divs)) if divs else ([0] if n==0 else [])


    p_divisors = get_divisors(a0)
    q_divisors = [d for d in get_divisors(an) if d != 0]

    if not p_divisors or not q_divisors:
        return []

    potential_rational_roots = set()
    for p_val in p_divisors:
        for q_val in q_divisors:
            potential_rational_roots.add(p_val / q_val)
    
    actual_rational_roots = []
    for root_candidate in sorted(list(potential_rational_roots)):
        if abs(evaluate_polynomial(coeffs, root_candidate)) < 1e-9:
            actual_rational_roots.append(root_candidate)
            
    return actual_rational_roots


def find_all_roots_numeric(coeffs: List[Union[float, complex]], attempts: int = 10) -> List[str]:
    """
    Attempts to find all roots (real and complex) numerically.
    Uses sympy.nroots for this.
    """
    if not coeffs or len(coeffs) <= 1:
        return []
    try:
        poly_str_for_sympy = polynomial_to_string(coeffs) # Use our formatter
        x = symbols('x')
        # Sympy Poly needs to be created correctly for complex coeffs
        # One way: construct expression from coeffs then Poly(expr, x)
        expr_terms = []
        degree = len(coeffs) - 1
        for i, c in enumerate(coeffs):
            power = degree - i
            if abs(c) > 1e-9: # Only add non-zero terms
                if power > 0:
                    expr_terms.append(c * (x**power))
                else:
                    expr_terms.append(c)
        
        if not expr_terms: # All coeffs were zero
            poly_expr_for_sympy = 0
        else:
            poly_expr_for_sympy = sum(expr_terms)

        poly_obj = SympyPoly(poly_expr_for_sympy, x)
        
        roots_found = poly_obj.nroots(n=15) # 15 digits of precision
        
        return [str(r.evalf(15)) for r in roots_found] # evalf for consistent string format
    except Exception as e:
        return [f"Error finding numerical roots: {e}"]

# --- In-house polynomial multiplication ---
def multiply_polynomials(poly1_coeffs: List[Union[float, complex]], poly2_coeffs: List[Union[float, complex]]) -> List[Union[float, complex]]:
    """Multiplies two polynomials given by their coefficient lists."""
    if not poly1_coeffs or not poly2_coeffs:
        return [0.0] # Or handle as error / return empty
    if (len(poly1_coeffs)==1 and abs(poly1_coeffs[0])<1e-9) or \
       (len(poly2_coeffs)==1 and abs(poly2_coeffs[0])<1e-9):
        return [0.0] # Multiplication by zero polynomial

    deg1 = len(poly1_coeffs) - 1
    deg2 = len(poly2_coeffs) - 1
    result_deg = deg1 + deg2
    
    # Initialize result polynomial with zeros (complex zeros for safety)
    result_coeffs: List[Union[float, complex]] = [complex(0.0)] * (result_deg + 1)

    for i, c1 in enumerate(poly1_coeffs):
        for j, c2 in enumerate(poly2_coeffs):
            # Power of x for c1 is deg1 - i
            # Power of x for c2 is deg2 - j
            # Power of x for c1*c2 is (deg1-i) + (deg2-j) = result_deg - (i+j)
            # Index in result_coeffs is (result_deg - power) = i+j
            result_coeffs[i+j] += c1 * c2
            
    # Optional: convert to float if all imag parts are zero
    final_coeffs = []
    for c in result_coeffs:
        if isinstance(c, complex) and abs(c.imag) < 1e-9:
            final_coeffs.append(c.real)
        else:
            final_coeffs.append(c)
    return final_coeffs