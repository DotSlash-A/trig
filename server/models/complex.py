from pydantic import BaseModel, Field
from typing import Optional, Literal
from typing import Dict, Any


# class lines(BaseModel):
#     slope: float = Field(..., description="Slope of the line")
#     y_intercept: float = Field(..., description="Y-intercept of the line")
#     x_intercept: float = Field(..., description="X-intercept of the line")
#     equation: str = Field(..., description="Equation of the line")

class ComplexNumber(BaseModel):
    """Represents a complex number with real and imaginary parts."""

    real: float = Field(..., description="The real part of the complex number.")
    img: float = Field(..., description="The imaginary part of the complex number.")


class ArithmeticRequest(BaseModel):
    """Request body model for arithmetic operations on complex numbers."""

    z1: ComplexNumber = Field(..., description="The first complex number operand.")
    z2: ComplexNumber = Field(..., description="The second complex number operand.")
    operation: Literal["add", "subtract", "multiply", "divide"] = Field(
        ..., description="Arithmetic operation to perform."
    )


class ArithmeticResponse(BaseModel):
    """Response model for the arithmetic operation result."""

    result: str = Field(..., description="The result of the arithmetic operation.")

class complexStringInput(BaseModel):
    """Request body model for complex number string input."""

    z: str = Field(..., description="Complex number in string format '-3 + sqrt(-7)'.")

class PropertiesResponse(BaseModel):
    """Response model for complex number properties."""
    input_expression: str
    real_part: float
    imaginary_part: float
    modulus:str
    conjugate:str
    standard_form:str

