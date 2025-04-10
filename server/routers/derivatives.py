from fastapi import FastAPI, Query, APIRouter

from sympy import symbols, Eq, solve, simplify, parse_expr
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication_application,
)


router = APIRouter()


