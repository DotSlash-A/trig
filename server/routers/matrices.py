from fastapi import FastAPI, Query, APIRouter, HTTPException
from sympy import symbols, Eq, solve, simplify, parse_expr
import math

from typing import Dict, Optional, List, Union, Tuple
from pydantic import BaseModel, Field
from fractions import Fraction
from models.matrix_model import (
    MatrixInputAPI,
    DetInput,
    DeterminantResponseAPI,
    TwoMatrixInput,
    MatrixEqualityResponse,
    MatrixFormulaInput,
    ConstructedMatrixResponse,

)

# from models.matrix_model import MatrixInput
import numpy as np
import sympy

router = APIRouter()


@router.get("/matrices/test")
async def test():
    return {"message": "Matrix API is working!"}


def validate_and_prepare_matrix(
    matrix_list: List[List[Union[float, int]]],
) -> Tuple[List[Union[float, int]], int]:
    """
    Validates if the input list of lists represents a square matrix
    and returns a flat list of its elements and its dimension.

    Args:
        matrix_list: The matrix as a list of lists.

    Returns:
        A tuple containing: (flat list of elements, dimension n)

    Raises:
        ValueError: If the input is invalid (not list, empty, not square, inconsistent rows).
    """
    if not isinstance(matrix_list, list) or not matrix_list:
        raise ValueError("Input must be a non-empty list of lists.")

    rows = len(matrix_list)
    if not isinstance(matrix_list[0], list):
        raise ValueError("Input must be a list of lists.")

    cols = len(matrix_list[0])
    if cols == 0:
        raise ValueError("Matrix rows cannot be empty.")

    if rows != cols:
        raise ValueError(
            f"Matrix must be square, but got {rows} rows and {cols} columns."
        )

    n = rows  # Dimension of the square matrix
    elements_flat = []
    for i, row in enumerate(matrix_list):
        if not isinstance(row, list):
            raise ValueError(f"Item at index {i} is not a list (row).")
        if len(row) != n:
            raise ValueError(
                f"Row {i} has {len(row)} elements, but expected {n} for a square matrix."
            )
        for element in row:
            if not isinstance(element, (int, float)):
                raise ValueError(
                    f"Element '{element}' in row {i} is not a number (int or float)."
                )
            elements_flat.append(element)

    # Double check the total number of elements gathered
    if len(elements_flat) != n * n:
        # This case should ideally be caught by row length check, but good failsafe
        raise ValueError("Internal error: Flattened element count mismatch.")

    return elements_flat, n


def calculate_determinant_core(elements_flat: List[Union[float, int]], n: int) -> float:
    """
    Calculates the determinant from a flat list of elements and the dimension.
    This is the core computational part.

    Args:
        elements_flat: A flat list containing n*n matrix elements in row-major order.
        n: The dimension of the square matrix.

    Returns:
        The calculated determinant as a float.

    Raises:
        ValueError: If input is inconsistent (e.g., len(elements_flat) != n*n).
                    (Should be caught by caller, but good practice).
        np.linalg.LinAlgError: If NumPy encounters a computation error.
    """
    if len(elements_flat) != n * n:
        raise ValueError(
            f"Inconsistent input: {len(elements_flat)} elements provided for dimension {n}."
        )
    if n == 0:
        # Determinant of 0x0 matrix is often defined as 1, but can be ambiguous.
        # For simplicity, let's consider it an invalid input here or handle as needed.
        # Or return 1 if that's the desired convention.
        raise ValueError("Cannot calculate determinant for a 0x0 matrix.")

    try:
        # In Python, use NumPy: reshape the flat list and calculate determinant
        np_matrix = np.array(elements_flat).reshape((n, n))
        determinant = np.linalg.det(np_matrix)
        return float(determinant)  # Ensure it's a standard float

    except np.linalg.LinAlgError as e:
        # Re-raise NumPy's specific error if calculation fails
        raise e
    except Exception as e:
        # Catch other potential errors during reshape or calculation
        raise ValueError(f"Error during determinant calculation: {e}")


@router.post("/determinant", response_model=DeterminantResponseAPI)
async def determinant_endpoint(input_data: DetInput):
    """
    API Endpoint: Takes a matrix as list of lists, validates it,
    calculates the determinant using the core function, and returns the result.
    """
    try:
        # 1. Get the nested list from the API input model
        matrix_list = input_data.matrix

        # 2. Validate and prepare data using the reusable function
        # This function ensures it's square and returns the flat list + dimension
        elements_flat, n = validate_and_prepare_matrix(matrix_list)

        # 3. Call the core calculation function (the part you'd translate)
        determinant_value = calculate_determinant_core(elements_flat, n)

        if abs(determinant_value - round(determinant_value)) < 1e-8:
            determinant_value = int(round(determinant_value))
        else:
            determinant_value = round(determinant_value, 6)

        # 4. Return the successful response
        return DeterminantResponseAPI(
            input_matrix=matrix_list,  # Return the original valid input matrix
            determinant=determinant_value,
        )

    except ValueError as e:
        # Catch validation errors from our functions
        raise HTTPException(status_code=400, detail=str(e))  # 400 Bad Request
    except np.linalg.LinAlgError as e:
        # Catch specific calculation errors from NumPy
        raise HTTPException(
            status_code=500, detail=f"Matrix calculation error: {e}"
        )  # 500 Internal Server Error
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(
            status_code=500, detail=f"An unexpected server error occurred: {e}"
        )


def get_matrix_dimensions(
    matrix: List[List[Union[float, int]]],
) -> Optional[Tuple[int, int]]:
    """Helper function to get dimensions and validate basic structure."""
    if not isinstance(matrix, list) or not matrix:
        return None  # Not a list or empty list
    try:
        rows = len(matrix)
        if not isinstance(matrix[0], list):
            return None  # First element isn't a list (not a list of lists)
        cols = len(matrix[0])
        if cols == 0:
            return None  # Rows are empty lists

        # Check if all rows have the same number of columns
        for row in matrix:
            if not isinstance(row, list) or len(row) != cols:
                return None  # Not a list or inconsistent column count
        return rows, cols
    except (TypeError, IndexError):
        return None  # Handles cases where matrix[0] fails etc.


# def compare_matrices_core(matrix_a: List[List[Union[float, int]]], matrix_b: List[List[Union[float, int]]]) -> bool:
#     """
#     Core function to compare two matrices for element-wise equality.


#     Args:
#         matrix_a: First matrix as a list of lists.
#         matrix_b: Second matrix as a list of lists.

#     Returns:
#         True if matrices are equal, False otherwise.
#     """
#     return np.array_equal(np.array(matrix_a), np.array(matrix_b))


def compare_matrices_core(
    matrix_a: List[List[Union[float, int]]], matrix_b: List[List[Union[float, int]]]
) -> Tuple[bool, str, Optional[str], Optional[str]]:
    """Compares two matrices for equality."""
    dims_a = get_matrix_dimensions(matrix_a)
    dims_b = get_matrix_dimensions(matrix_b)

    dim_a_str = f"{dims_a[0]}x{dims_a[1]}" if dims_a else "Invalid Format"
    dim_b_str = f"{dims_b[0]}x{dims_b[1]}" if dims_b else "Invalid Format"

    if dims_a is None or dims_b is None:
        # Check if *both* are invalid in the same way (e.g., both empty lists)
        # This is debatable, but let's consider two empty lists NOT equal as matrices
        if matrix_a == matrix_b:  # e.g. [] == []
            return (
                False,
                "Inputs are identical but not valid matrices.",
                dim_a_str,
                dim_b_str,
            )
        return False, "One or both inputs are not valid matrices.", dim_a_str, dim_b_str

    rows_a, cols_a = dims_a
    rows_b, cols_b = dims_b

    if rows_a != rows_b or cols_a != cols_b:
        return False, "Matrices have different dimensions.", dim_a_str, dim_b_str

    for i in range(rows_a):
        for j in range(cols_a):
            if matrix_a[i][j] != matrix_b[i][j]:
                reason = f"Element mismatch at row {i}, column {j} ({matrix_a[i][j]} != {matrix_b[i][j]})."
                return False, reason, dim_a_str, dim_b_str

    return True, "Matrices are identical.", dim_a_str, dim_b_str




def construct_matrix_from_formula_conv(m:int, n:int, formula_str:str) -> List[List[float]]:
    """
    Constructs a matrix based on the provided formula.

    Args:
        m: Number of rows.
        n: Number of columns.
        formula_str: Formula string using 'i' and 'j' as indices.

    Returns:
        Constructed matrix as a list of lists.
    """
    i, j = symbols("i j")
    formula = parse_expr(formula_str)

    constructed_matrix = []
    for row in range(1, m + 1):
        constructed_row = []
        for col in range(1, n + 1):
            value = formula.subs({i: row, j: col})
            constructed_row.append(float(value))
        constructed_matrix.append(constructed_row)

    return constructed_matrix

def construct_matrix_from_formula(m:int, n:int, formula_str:str) -> List[List[float]]:
    """
    Constructs a matrix based on the provided formula.

    Args:
        m: Number of rows.
        n: Number of columns.
        formula_str: Formula string using 'i' and 'j' as indices.

    Returns:
        Constructed matrix as a list of lists.
    """
    i,j= symbols("i j")
    try:
        allowed_globals={"__builtins__":None}
        parsed_formula = sympy.parse_expr(
            formula_str,
            local_dict={"i": i, "j": j},
            globals=allowed_globals,
        )
    except (SyntaxError, TypeError, sympy.SympifyError) as e:
        raise ValueError(f"Invalid formula: {e}")
    matrix = [[0.0 for _ in range(n)] for _ in range(m)]
    for row_idx in range(m):
        for col_idx in range(n):
            current_i = row_idx + 1  # 1-based index
            current_j = col_idx + 1
            try:
                value = parsed_formula.subs({i: current_i, j: current_j}).evalf()
                if value.is_complex():
                    raise ValueError("Complex numbers are not supported.")
                matrix[row_idx][col_idx] = float(value)
            except (TypeError, ValueError, ZeroDivisionError) as e:
                raise ValueError(f"Error evaluating formula at ({current_i}, {current_j}): {e}")
            except Exception as e:
                raise ValueError(f"Unexpected error: {e}")
    return matrix

@router.post("/matrix_formula", response_model=ConstructedMatrixResponse)
async def matrix_formula_endpoint(input_data: MatrixFormulaInput):
    """
    API Endpoint: Takes a formula and dimensions, constructs the matrix,
    and returns the constructed matrix.
    """
    try:
        result_matrix= construct_matrix_from_formula(
            m=input_data.m,
            n=input_data.n,
            formula_str=input_data.formula,
        )
        return ConstructedMatrixResponse(
            rows=input_data.m,
            columns=input_data.n,
            formula_used=input_data.formula,
            constructed_matrix=result_matrix
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

















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
