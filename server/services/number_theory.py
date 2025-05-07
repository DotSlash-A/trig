# services/number_theory.py
from math import gcd as math_gcd
from sympy import symbols, Poly, sympify, SympifyError
from sympy.solvers.diophantine.diophantine import diop_DN
from typing import Dict, List, Tuple, Any, Optional

def euclids_division_lemma(dividend: int, divisor: int) -> Tuple[int, int]:
    """
    Applies Euclid's Division Lemma: dividend = divisor * quotient + remainder.
    Returns quotient and remainder.
    """
    if divisor == 0:
        raise ValueError("Divisor cannot be zero.")
    
    # The notebook code used max/min, which is fine for the algorithm's start
    # but for the lemma itself, dividend and divisor are specific.
    # If we want to ensure dividend >= divisor, we can swap, but
    # the modulo operator handles it naturally for positive divisors.
    # For general a = bq + r, b can be larger than a.
    
    quotient = dividend // divisor
    remainder = dividend % divisor
    return quotient, remainder

def euclids_algorithm_hcf(n1: int, n2: int) -> int:
    """
    Calculates the Highest Common Factor (HCF/GCD) of two numbers
    using Euclid's Algorithm.
    """
    if n1 == 0 and n2 == 0:
        raise ValueError("HCF(0, 0) is undefined or sometimes taken as 0. Standard GCD requires at least one non-zero.")
    if n1 == 0: return abs(n2)
    if n2 == 0: return abs(n1)

    # Ensure positive inputs for standard Euclidean algorithm form
    num1 = abs(n1)
    num2 = abs(n2)

    dividend = max(num1, num2)
    divisor = min(num1, num2)
    
    while divisor != 0:
        remainder = dividend % divisor
        dividend = divisor
        divisor = remainder
    return dividend

def get_prime_factorization(n: int) -> Dict[int, int]:
    """
    Computes the prime factorization of a positive integer n.
    Returns a dictionary of {prime: exponent}.
    Example: 12 -> {2: 2, 3: 1}
    """
    if n <= 1:
        return {} 
        # Or raise ValueError("Input must be an integer greater than 1 for prime factorization.")
        # Depending on desired behavior for n=1 or n=0 or negative n.
        # For this API, let's return empty for n <= 1 as they don't have prime factors in the usual sense.

    n_abs = abs(n) # Factorize the absolute value
    factors = {}
    d = 2
    temp_n = n_abs
    
    while d * d <= temp_n:
        while temp_n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            temp_n //= d
        d += 1
    if temp_n > 1: # Remaining temp_n is a prime
        factors[temp_n] = factors.get(temp_n, 0) + 1
    return factors

def hcf_from_prime_factorization(factors1: Dict[int, int], factors2: Dict[int, int]) -> int:
    """
    Calculates HCF from two prime factorization dictionaries.
    """
    hcf_val = 1
    common_primes = set(factors1.keys()) & set(factors2.keys())
    for prime in common_primes:
        power = min(factors1[prime], factors2[prime])
        hcf_val *= prime ** power
    return hcf_val

def lcm_from_prime_factorization(factors1: Dict[int, int], factors2: Dict[int, int]) -> int:
    """
    Calculates LCM from two prime factorization dictionaries.
    """
    lcm_val = 1
    all_primes = set(factors1.keys()) | set(factors2.keys())
    if not all_primes: # Both numbers were 1 or 0, resulting in empty factorizations
        return 1 # LCM(1,1)=1. LCM involving 0 is tricky (often 0 or undefined).
                 # Given we factorize abs(n), LCM(0, k) would become LCM of factors of 0 (empty) and k.
                 # Let's assume inputs to this function are derived from positive integers > 1.

    for prime in all_primes:
        power1 = factors1.get(prime, 0)
        power2 = factors2.get(prime, 0)
        lcm_val *= prime ** max(power1, power2)
    return lcm_val

def calculate_lcm(n1: int, n2: int, hcf: Optional[int] = None) -> int:
    """
    Calculates the Least Common Multiple (LCM) of two numbers.
    Uses the formula LCM(a,b) = |a*b| / HCF(a,b).
    If HCF is not provided, it will be calculated.
    """
    if n1 == 0 or n2 == 0:
        return 0 # Conventionally, LCM(a, 0) = 0
    
    n1_abs = abs(n1)
    n2_abs = abs(n2)

    if hcf is None:
        hcf_val = euclids_algorithm_hcf(n1_abs, n2_abs)
    else:
        hcf_val = hcf
    
    if hcf_val == 0: # Should not happen if n1, n2 are not both zero
        return 0 
        
    return (n1_abs * n2_abs) // hcf_val


def get_decimal_expansion_type(numerator: int, denominator: int) -> Tuple[str, str]:
    """
    Determines if the decimal expansion of a rational number num/den is
    terminating or non-terminating recurring.
    Returns a tuple: (type_of_expansion, simplified_denominator_prime_factors_explanation)
    """
    if denominator == 0:
        raise ValueError("Denominator cannot be zero.")

    common = math_gcd(abs(numerator), abs(denominator))
    simplified_den = abs(denominator) // common

    if simplified_den == 0: # Should not happen if original denominator wasn't 0
        return "undefined", "Denominator became zero after simplification, implies numerator was also zero."
    if simplified_den == 1:
        return "terminating", f"The fraction simplifies to an integer ({numerator//common}). Denominator is 1."

    # Prime factorize the simplified denominator
    temp_den = simplified_den
    only_2_and_5 = True
    factors_str_list = []

    # Check for factor 2
    count_2 = 0
    while temp_den % 2 == 0:
        temp_den //= 2
        count_2 += 1
    if count_2 > 0:
        factors_str_list.append(f"2^{count_2}")

    # Check for factor 5
    count_5 = 0
    while temp_den % 5 == 0:
        temp_den //= 5
        count_5 += 1
    if count_5 > 0:
        factors_str_list.append(f"5^{count_5}")
        
    # Check for other prime factors
    d = 3 # Start with 3, then 7, 9 (skip), 11, 13 ...
    other_factors = []
    while d * d <= temp_den:
        if temp_den % d == 0:
            only_2_and_5 = False
            count_d = 0
            while temp_den % d == 0:
                temp_den //= d
                count_d +=1
            other_factors.append(f"{d}^{count_d}")
        d += 2 # Check odd numbers (can be optimized further, but fine for this)
        if d % 5 == 0 and d > 5 : # skip multiples of 5
            d += 2


    if temp_den > 1: # Remaining temp_den is a prime factor other than 2 or 5
        if temp_den != 2 and temp_den != 5: # Should be true if not already caught
            only_2_and_5 = False
            other_factors.append(f"{temp_den}^1")

    explanation_parts = []
    if factors_str_list:
        explanation_parts.append(" ".join(factors_str_list))
    if other_factors:
        explanation_parts.append(" ".join(other_factors))
    
    denominator_factors_str = " * ".join(explanation_parts) if explanation_parts else "1"


    if only_2_and_5:
        return "terminating", f"Simplified denominator ({simplified_den}) has prime factors: {denominator_factors_str} (only 2s and/or 5s)."
    else:
        return "non-terminating recurring", f"Simplified denominator ({simplified_den}) has prime factors: {denominator_factors_str} (includes primes other than 2 and 5)."


def analyze_polynomial_expression(poly_expr_str: str) -> Dict[str, Any]:
    """
    Analyzes a polynomial expression string.
    Returns degree, coefficients, sum of roots, and product of roots.
    """
    x = symbols('x')
    try:
        poly_expr = sympify(poly_expr_str, locals={'x': x})
        if not poly_expr.is_polynomial(x):
            raise ValueError("Expression is not a polynomial in x.")
        
        poly = Poly(poly_expr, x)
    except (SympifyError, ValueError) as e:
        raise ValueError(f"Invalid polynomial expression: {e}")

    degree = poly.degree()
    coeffs = poly.all_coeffs() # List of Sympy numbers
    
    # Convert Sympy numbers in coeffs to Python floats/ints for easier JSON serialization
    coeffs_py = []
    for c in coeffs:
        if c.is_Integer:
            coeffs_py.append(int(c))
        elif c.is_Rational:
            coeffs_py.append(float(c)) # or str(c) for exact fraction "p/q"
        else:
            coeffs_py.append(float(c)) # Fallback for other Sympy number types

    # Roots (can be complex)
    # sympy.roots returns a dict {root: multiplicity}
    # For sum and product, we need all roots including multiplicity
    all_roots_list = []
    # Note: roots() can be slow or fail for high-degree polynomials symbolically
    # For numerical roots, poly.nroots() might be an option
    try:
        # Attempt to find symbolic roots. This can be computationally intensive.
        # For higher degree polynomials, this might hang or be very slow.
        # roots_dict = poly.all_roots() # This would be ideal but often too slow or fails
        roots_dict = poly.roots() # Finds roots in the domain of coefficients (often rational)

        for r, m in roots_dict.items():
            all_roots_list.extend([r] * m)
        
        # If degree > len(all_roots_list) and degree > 4, symbolic roots might not be found
        # For quadratic, cubic, quartic, Vieta's formulas are direct from coefficients
        sum_of_roots_val = None
        product_of_roots_val = None

        if degree > 0 and coeffs_py[0] != 0:
            # From Vieta's formulas:
            # Sum of roots = - (coeff of x^(n-1)) / (coeff of x^n)
            if degree == 1: # ax + b
                sum_of_roots_val = -coeffs_py[1] / coeffs_py[0] if len(coeffs_py) > 1 else 0
            elif len(coeffs_py) > 1 :
                 sum_of_roots_val = -coeffs_py[1] / coeffs_py[0]
            else: # Monomial like ax^n
                 sum_of_roots_val = 0


            # Product of roots = (-1)^n * (constant term) / (coeff of x^n)
            product_of_roots_val = ((-1)**degree) * coeffs_py[-1] / coeffs_py[0]

            # If roots were found numerically/symbolically, verify
            if all_roots_list:
                calculated_sum = sum(all_roots_list)
                # product can be tricky if 0 is a root
                calculated_product = 1
                for r_val in all_roots_list:
                    calculated_product *= r_val
                
                # Use calculated if available, otherwise Vieta's
                # Be mindful of floating point precision with calculated sum/product
                sum_of_roots_val = float(sum(all_roots_list)) if all_roots_list else sum_of_roots_val
                product_of_roots_val = float(calculated_product) if all_roots_list else product_of_roots_val
        else: # Degree 0 (constant) or leading coeff is zero (should not happen for Poly)
            sum_of_roots_val = 0
            product_of_roots_val = coeffs_py[0] if coeffs_py else 0


    except NotImplementedError: # sympy.roots might raise this for complex cases
        sum_of_roots_val = "Not implemented for this polynomial by Sympy's symbolic root finder"
        product_of_roots_val = "Not implemented for this polynomial by Sympy's symbolic root finder"
        all_roots_list = ["Symbolic roots not found or too complex"]


    # Convert Sympy root objects to string representations for JSON
    roots_str_list = [str(r) for r in all_roots_list]

    return {
        "expression": str(poly_expr),
        "degree": degree,
        "coefficients": coeffs_py, # e.g., [1, -3, 2] for x^2 - 3x + 2
        "roots_found": roots_str_list,
        "sum_of_roots_vieta": str(sum_of_roots_val), # Using Vieta's for clarity
        "product_of_roots_vieta": str(product_of_roots_val) # Using Vieta's for clarity
    }

def is_number_perfect_square(n: int) -> bool:
    if n < 0: return False
    if n == 0: return True
    sqrt_n = int(n**0.5)
    return sqrt_n * sqrt_n == n

def is_prime_basic(n: int) -> bool:
    """Basic primality test."""
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True