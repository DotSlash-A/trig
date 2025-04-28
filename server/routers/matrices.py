from fastapi import FastAPI, Query, APIRouter, HTTPException
from sympy import symbols, Eq, solve, simplify, parse_expr
import math

from typing import Dict, Optional
from pydantic import BaseModel, Field
from fractions import Fraction

# from models.matrix_model import MatrixInput
import numpy as np

router = APIRouter()


@router.get("/matrices/test")
async def test():
    return {"message": "Matrix API is working!"}


# @router.post("/construct_matrices")
# def construct_matrices(matrix_a: MatrixInput, matrix_b: MatrixInput):
#     # Validate dimensions
#     if len(matrix_a.numbers) != matrix_a.dimensions[0] * matrix_a.dimensions[1]:
#         raise HTTPException(
#             status_code=400, detail="Matrix A numbers do not match dimensions"
#         )
#     if len(matrix_b.numbers) != matrix_b.dimensions[0] * matrix_b.dimensions[1]:
#         raise HTTPException(
#             status_code=400, detail="Matrix B numbers do not match dimensions"
#         )

#     # Construct matrices
#     a = np.array(matrix_a.numbers).reshape(matrix_a.dimensions)
#     b = np.array(matrix_b.numbers).reshape(matrix_b.dimensions)

#     # Check element-wise equality
#     if a.shape != b.shape:
#         return {"equal": False, "reason": "Matrices have different shapes"}

#     equality_matrix = (a == b).tolist()

#     return {
#         "matrix_a": a.tolist(),
#         "matrix_b": b.tolist(),
#         "element_wise_equality": equality_matrix,
#     }
