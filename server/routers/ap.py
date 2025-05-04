from fastapi import FastAPI, Query, APIRouter, HTTPException
from sympy import symbols, Eq, solve, simplify, parse_expr
import math

from typing import Dict, Optional, List, Union, Tuple
from pydantic import BaseModel, Field
from fractions import Fraction


import sympy

router = APIRouter(prefix="/progressions", tags=["progressions"])
# router = APIRouter()


@router.get("/prog/test")
async def test():
    return {"message": "Matrix API is working!"}


class APREquest(BaseModel):
    """
    Request model for APR calculation.
    """

    a: float = Field(..., description="First term of the arithmetic progression")
    d: float = Field(..., description="Common difference of the arithmetic progression")
    n: int = Field(..., description="Number of terms in the arithmetic progression")


@router.post("/arithmetic_progression")
async def arithmetic_progression(request: APREquest):
    """
    Calculate the nth term and sum of the first n terms of an arithmetic progression.
    """
    a = request.a
    d = request.d
    n = request.n

    # Calculate nth term
    nth_term = a + (n - 1) * d

    # Calculate sum of first n terms
    sum_n_terms = (n / 2) * (2 * a + (n - 1) * d)

    return {"nth_term": nth_term, "sum_n_terms": sum_n_terms}


class ap_lastterm(BaseModel):
    """
    Request model for nth term from last term calculation.
    """

    a: float = Field(..., description="First term of the arithmetic progression")
    d: float = Field(..., description="Common difference of the arithmetic progression")
    l: int = Field(..., description="last term in the arithmetic progression")
    n: int = Field(..., description="nth term from end")


@router.post("/nth_term_from_last")
async def nth_term_from_last(request: ap_lastterm):
    """
    Calculate the nth term from the last term of an arithmetic progression.
    """
    a = request.a
    d = request.d
    l = request.l
    n = request.n

    # Calculate nth term from the last term
    nth_term_from_last = l - (n - 1) * d

    return {"nth_term_from_last": nth_term_from_last}


class MiddleTermInput(BaseModel):  # Renamed for clarity
    """Input model to find the middle term(s) given first term, common difference, and last term."""

    a: float = Field(..., description="First term of the arithmetic progression")
    d: float = Field(..., description="Common difference of the arithmetic progression")
    last_term: float = Field(
        ..., description="Value of the last term in the arithmetic progression"
    )  # Renamed n to last_term


@router.post("/middle_term")
async def middle_term(request: MiddleTermInput):  # Use the renamed model
    """
    Calculate the middle term(s) of an arithmetic progression given the first term,
    common difference, and the value of the last term.
    """
    a = request.a
    d = request.d
    an = request.last_term  # Use the renamed field

    if d == 0:
        if a == an:
            # Infinite terms if a == an, or sequence doesn't exist if a != an and d=0
            # Depending on requirements, could return the term 'a' or raise an error.
            # Let's assume if d=0 and a=an, any term is 'a'. Middle term concept is ambiguous.
            # Returning 'a' as the only value. Handle as per specific requirements.
            return {"middle_term(s)": [a], "message": "Progression has constant terms."}
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid progression: d is 0 but first and last terms differ.",
            )

    # Calculate the number of terms n
    if (an - a) % d != 0:
        # Check if last_term is reachable with the given a and d
        raise HTTPException(
            status_code=400,
            detail="The provided last term is not part of the arithmetic progression defined by 'a' and 'd'.",
        )

    n_float = (an - a) / d + 1
    if n_float <= 0 or n_float != int(n_float):
        raise HTTPException(
            status_code=400,
            detail="Invalid progression parameters leading to non-positive or non-integer number of terms.",
        )

    n = int(n_float)  # Number of terms must be an integer

    # Calculate middle term(s)
    middle_terms = []
    if n % 2 == 1:  # Odd number of terms
        # Middle term index is (n+1)/2. Value is a + (((n+1)/2) - 1) * d = a + ((n-1)/2) * d
        # Using integer division: n // 2 gives (n-1)/2 for odd n
        middle_term_value = a + (n // 2) * d
        middle_terms.append(middle_term_value)
    else:  # Even number of terms
        # Middle term indices are n/2 and n/2 + 1
        # Value 1: a + (n/2 - 1) * d
        middle_term1_value = a + (n // 2 - 1) * d
        # Value 2: a + ((n/2 + 1) - 1) * d = a + (n/2) * d
        middle_term2_value = a + (n // 2) * d
        middle_terms.append(middle_term1_value)
        middle_terms.append(middle_term2_value)

    return {"number_of_terms": n, "middle_term(s)": middle_terms}
