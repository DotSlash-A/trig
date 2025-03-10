from pydantic import BaseModel, Field
from typing import Optional


class lines(BaseModel):
    slope: float = Field(..., description="Slope of the line")
    y_intercept: float = Field(..., description="Y-intercept of the line")
    x_intercept: float = Field(..., description="X-intercept of the line")
    equation: str = Field(..., description="Equation of the line")


class Original_axes(BaseModel):
    x: float = Field(..., description="original X-coordinate of the point")
    y: float = Field(..., description="origina Y-coordinate of the point")
    h: float = Field(..., description="x-coordinate of the new origin")
    k: float = Field(..., description="y-coordinate of the new origin")


class New_axes(BaseModel):
    X: float = Field(..., description="new X-coordinate of the point")
    Y: float = Field(..., description="new Y-coordinate of the point")
    h: float = Field(..., description="x-coordinate of the new origin")
    k: float = Field(..., description="y-coordinate of the new origin")


class EquationTransform(BaseModel):
    equation: str = Field(..., description="Equation of the line")
    h: float = Field(..., description="x-coordinate of the new origin")
    k: float = Field(..., description="y-coordinate of the new origin")
    equation_transformed: Optional[str] = Field(
        ..., description="Transformed equation of the line"
    )

class SlopeCordiantes(BaseModel):
    x1: float = Field(..., description="x-coordinate of the first point")
    y1: float = Field(..., description="y-coordinate of the first point")
    x2: float = Field(..., description="x-coordinate of the second point")
    y2: float = Field(..., description="y-coordinate of the second point")



