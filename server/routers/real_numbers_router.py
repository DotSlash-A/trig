# routers/real_numbers_router.py
from fastapi import APIRouter, Query, Path, HTTPException, Body
from services import number_theory as nt
from models import real_numbers_models as models
from sympy import sqrt as sympy_sqrt # For formatting sqrt string

router = APIRouter(
    prefix="/real-numbers",
    tags=["Real Numbers and Polynomials"]
)

@router.get("/euclids-division-lemma", response_model=models.EuclidLemmaResponse)
async def get_euclids_division_lemma(
    dividend: int = Query(..., example=455, description="The number to be divided (a)"),
    divisor: int = Query(..., example=42, description="The number by which to divide (b, must be non-zero)")
):
    """
    Applies Euclid's Division Lemma: a = bq + r.
    Given dividend 'a' and divisor 'b', finds quotient 'q' and remainder 'r'
    such that a = b*q + r, where 0 <= r < |b|.
    """
    if divisor == 0:
        raise HTTPException(status_code=400, detail="Divisor cannot be zero.")
    try:
        q, r = nt.euclids_division_lemma(dividend, divisor)
        return models.EuclidLemmaResponse(
            dividend=dividend,
            divisor=divisor,
            quotient=q,
            remainder=r,
            equation=f"{dividend} = {divisor} * {q} + {r}"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/hcf/euclids-algorithm", response_model=models.HCFResponse)
async def get_hcf_euclids(
    num1: int = Query(..., example=455, description="First integer"),
    num2: int = Query(..., example=42, description="Second integer")
):
    """Calculates the Highest Common Factor (HCF/GCD) of two integers using Euclid's Algorithm."""
    try:
        hcf = nt.euclids_algorithm_hcf(num1, num2)
        return models.HCFResponse(num1=num1, num2=num2, hcf=hcf)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/prime-factorization/{number}", response_model=models.PrimeFactorizationResponse)
async def get_prime_factors(
    number: int = Path(..., example=3825, description="Integer to factorize (must be > 1 for meaningful factorization)")
):
    """Computes the prime factorization of an integer n > 1."""
    if number <= 1:
        raise HTTPException(status_code=400, detail="Number must be greater than 1 for prime factorization.")
    try:
        factors = nt.get_prime_factorization(number)
        return models.PrimeFactorizationResponse(number=number, factors=factors)
    except ValueError as e: # Should be caught by pre-check, but good practice
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/hcf-lcm/prime-factorization", response_model=models.HCFAndLCMResponse)
async def get_hcf_lcm_prime_factorization(
    num1: int = Query(..., example=96, description="First integer (>1)"),
    num2: int = Query(..., example=404, description="Second integer (>1)")
):
    """
    Calculates HCF and LCM of two integers using their prime factorizations.
    """
    if num1 <= 0 or num2 <= 0: # Factorization is usually for n > 1
        raise HTTPException(status_code=400, detail="Both numbers must be positive integers.")
    if num1 == 1 and num2 == 1:
        factors1 = {}
        factors2 = {}
        hcf_val = 1
        lcm_val = 1
    elif num1 == 1:
        factors1 = {}
        factors2 = nt.get_prime_factorization(num2)
        hcf_val = 1
        lcm_val = num2 # LCM(1, k) = k
    elif num2 == 1:
        factors1 = nt.get_prime_factorization(num1)
        factors2 = {}
        hcf_val = 1
        lcm_val = num1 # LCM(k, 1) = k
    else:
        try:
            factors1 = nt.get_prime_factorization(num1)
            factors2 = nt.get_prime_factorization(num2)
            hcf_val = nt.hcf_from_prime_factorization(factors1, factors2)
            # lcm_val = nt.lcm_from_prime_factorization(factors1, factors2) # Using prime factors
            lcm_val = nt.calculate_lcm(num1, num2, hcf_val) # Using formula (more direct)
        except ValueError as e:
             raise HTTPException(status_code=400, detail=str(e))


    return models.HCFAndLCMResponse(
        num1=num1,
        num2=num2,
        prime_factorization_num1=factors1,
        prime_factorization_num2=factors2,
        hcf=hcf_val,
        lcm=lcm_val
    )

@router.get("/check-irrationality/sqrt/{number}", response_model=models.IrrationalityCheckResponse)
async def check_irrationality_sqrt(
    number: int = Path(..., example=5, ge=0, description="Non-negative integer N to check if sqrt(N) is irrational")
):
    """
    Checks if the square root of a non-negative integer N is irrational.
    Based on the theorem: If p is a prime number, then sqrt(p) is irrational.
    Also, if N is a perfect square, sqrt(N) is rational (an integer).
    For composite N that are not perfect squares, sqrt(N) is also irrational (e.g. sqrt(6)).
    """
    form = f"sqrt({number})"
    if number < 0:
        # This case is handled by ge=0, but as a safeguard
        return models.IrrationalityCheckResponse(
            number_form=form,
            is_irrational=False, # Or True, as it's complex, not real-irrational
            reason=f"sqrt({number}) is a complex number, not a real irrational number."
        )
    
    if nt.is_number_perfect_square(number):
        return models.IrrationalityCheckResponse(
            number_form=form,
            is_irrational=False,
            reason=f"sqrt({number}) = {int(number**0.5)}, which is a rational number (integer)."
        )
    
    # If not a perfect square, sqrt(N) is irrational for integer N > 0.
    # We can optionally add a check if N is prime for specific NCERT style reasoning.
    if nt.is_prime_basic(number):
         reason = f"{number} is a prime number. The square root of a prime number is irrational."
    else:
        reason = f"{number} is not a perfect square. The square root of a non-perfect square integer is irrational."

    return models.IrrationalityCheckResponse(
        number_form=form,
        is_irrational=True,
        reason=reason
    )

@router.get("/decimal-expansion", response_model=models.DecimalExpansionResponse)
async def get_decimal_expansion_properties(
    numerator: int = Query(..., example=13, description="Numerator of the fraction"),
    denominator: int = Query(..., example=3125, description="Denominator of the fraction (must be non-zero)")
):
    """
    Determines if the decimal expansion of a rational number (numerator/denominator)
    is terminating or non-terminating recurring.
    """
    if denominator == 0:
        raise HTTPException(status_code=400, detail="Denominator cannot be zero.")
    try:
        expansion_type, reason = nt.get_decimal_expansion_type(numerator, denominator)
        return models.DecimalExpansionResponse(
            numerator=numerator,
            denominator=denominator,
            fraction=f"{numerator}/{denominator}",
            expansion_type=expansion_type,
            reason=reason
        )
    except ValueError as e: # Should be caught by pre-check
        raise HTTPException(status_code=400, detail=str(e))

# @router.post("/polynomial/analyze", response_model=models.PolynomialAnalysisResponse)
# async def analyze_polynomial(
#     request: models.PolynomialAnalysisRequest = Body(...)
# ):
#     """
#     Analyzes a given polynomial expression string (e.g., "x**2 - 3*x + 2").
#     Provides degree, coefficients, attempts to find roots, and sum/product of roots via Vieta's formulas.
#     Note: Symbolic root finding can be computationally intensive or fail for higher-degree polynomials.
#     """
#     try:
#         analysis = nt.analyze_polynomial_expression(request.expression)
#         return models.PolynomialAnalysisResponse(**analysis)
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))