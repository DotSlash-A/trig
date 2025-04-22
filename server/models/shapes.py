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

    slope: Optional[float] = Field(None, description="Slope of the line")
    y_intercept: Optional[float] = Field(None, description="Y-intercept of the line")
    x_intercept: Optional[float] = Field(
        None, description="X-intercept of the line (optional)"
    )
    equation: Optional[str] = Field(None, description="Equation of the line (optional)")
    point1: Optional[coordinates] = Field(
        None, description="Point on the line (optional)"
    )
    point2: Optional[coordinates] = Field(
        None, description="Point on the line (optional)"
    )


class LineInput(BaseModel):
    slope: Optional[float] = Field(None, description="Slope of the line")
    point: Optional[coordinates] = Field(
        None, description="Point on the line (optional)"
    )
    y_intercept: Optional[float] = Field(None, description="Y-intercept of the line")
    x_intercept: Optional[float] = Field(
        None, description="X-intercept of the line (optional)"
    )


class TransformationsLine(BaseModel):
    A: Optional[float] = Field(None, description="X-coeff ")
    B: Optional[float] = Field(None, description="Y-coeff ")
    C: Optional[float] = Field(None, description="Constant term")


class circleGenral(BaseModel):
    r: Optional[float] = Field(None, description="Radius of the circle")
    h: Optional[float] = Field(None, description="X-coordinate of the center")
    k: Optional[float] = Field(None, description="Y-coordinate of the center")
    x: Optional[float] = Field(None, description="X-coordinate of the point")
    y: Optional[float] = Field(None, description="Y-coordinate of the point")


class CircleEqnResponse(BaseModel):
    standard_form: str = Field(..., description="Standard form of the circle equation")
    general_form: str = Field(..., description="General form of the circle equation")
    center_h: float = Field(..., description="X-coordinate of the center of the circle")
    center_k: float = Field(..., description="Y-coordinate of the center of the circle")
    radius: float = Field(..., description="Radius of the circle")
    A: float = Field(..., description="Coefficient A in the general form")
    B: float = Field(..., description="Coefficient B in the general form")
    C: float = Field(..., description="Coefficient C in the general form")
    D: float = Field(..., description="Coefficient D in the general form")
    E: float = Field(..., description="Coefficient E in the general form")


class CircleGeneralFormInput(BaseModel):
    """Input model for the general form of a circle equation."""

    equation: str = Field(
        ..., description="General form equation, e.g., 'x^2 + y^2 - 4*x + 6*y - 12 = 0'"
    )


class CircleDetailsResponse(BaseModel):
    """Response model containing the calculated center and radius."""

    center_h: float
    center_k: float
    radius: float
    input_equation: str
    normalized_equation: Optional[str] = None  # Optional: show the normalized form
