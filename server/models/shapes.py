from pydantic import BaseModel, Field
from typing import Optional
from typing import Dict, Any

class lines(BaseModel):
    slope: float = Field(..., description="Slope of the line")
    y_intercept: float = Field(..., description="Y-intercept of the line")
    x_intercept: float = Field(..., description="X-intercept of the line")
    equation: str = Field(..., description="Equation of the line")


class SlopeInput(BaseModel):
    x1: Optional[float] = None
    y1: Optional[float] = None
    x2: Optional[float] = None
    y2: Optional[float] = None
    m: Optional[float] = None



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

class FindXRequest(BaseModel):
    find_var: str = Field(
        ..., description="The variable to find ('x1', 'y1', 'x2', 'y2', 'm')."
    )
    known_values: Dict[str, float] = Field(
        ..., description="Dictionary containing the known variables and their values."
    )


class coordinates(BaseModel):
    x: float = Field(..., description="X-coordinate of the point")
    y: float = Field(..., description="Y-coordinate of the point")

class SlopeIntercept(BaseModel):

    slope: Optional[float]=Field(None, description="Slope of the line")
    y_intercept: Optional[float] = Field(
        None, description="Y-intercept of the line")
    x_intercept: Optional[float] = Field(
        None, description="X-intercept of the line (optional)"
    )
    equation: Optional[str] = Field(
        None, description="Equation of the line (optional)"
    )
    point1: Optional[coordinates] = Field(
        None, description="Point on the line (optional)"
    )
    point2: Optional[coordinates] = Field(
        None, description="Point on the line (optional)"
    )
