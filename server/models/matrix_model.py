from pydantic import BaseModel, Field, validator
from typing import Optional, Tuple, List
from typing import Dict, Any, Union

# class MatrixInput(BaseModel):
#     numers: List[float]
#     dim=Tuple[int, int]


# CORRECT MODEL FOR YOUR POSTMAN CALL
class MatrixInputAPI(BaseModel):
    matrix: List[List[Union[float, int]]] = Field(...)  # <--- Expects "matrix"


class DetInput(BaseModel):
    # n : int = Field(...,gt=0, description="dimension of the square matrix")
    matrix: List[List[Union[float, int]]] = Field(
        ..., description="The matrix as a list of lists, e.g., [[1, 2], [3, 4]]"
    )


class DeterminantResponseAPI(BaseModel):
    input_matrix: List[List[Union[float, int]]]  # Echo the valid input matrix
    determinant: float

class TwoMatrixInput(BaseModel):
    matrix_a: List[List[Union[float, int]]] = Field(..., description="The first matrix as a list of lists.")
    matrix_b: List[List[Union[float, int]]] = Field(..., description="The second matrix as a list of lists.")

class MatrixEqualityResponse(BaseModel):
    are_equal: bool = Field(..., description="True if the matrices are identical, False otherwise.")
    reason: str = Field(..., description="Explanation of why they are equal or not.")
    dimensions_a: Optional[str] = Field(None, description="Dimensions of matrix A (e.g., '3x4'). None if invalid.")
    dimensions_b: Optional[str] = Field(None, description="Dimensions of matrix B (e.g., '3x4'). None if invalid.")

class MatrixFormulaInput(BaseModel):
    m: int = Field(..., gt=0, description="Number of rows (must be positive).")
    n: int = Field(..., gt=0, description="Number of columns (must be positive).")
    # Use 'formula' instead of 'a_ij_formula' for slight brevity
    formula: str = Field(...,
                         description="Formula for element a_ij using 'i' (row index, 1-based) and 'j' (column index, 1-based). E.g., 'i + j', '(i+2*j)**2 / 2'.")

class ConstructedMatrixResponse(BaseModel):
    rows: int = Field(..., description="Number of rows specified.")
    columns: int = Field(..., description="Number of columns specified.")
    formula_used: str = Field(..., description="The formula provided in the input.")
    constructed_matrix: List[List[float]] = Field(..., description="The resulting matrix calculated using the formula.")