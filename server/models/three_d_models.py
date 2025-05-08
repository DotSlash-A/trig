# models/three_d_models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Union, Dict, Any

# --- Basic Geometric Entities ---
class PointInput(BaseModel):
    x: float
    y: float
    z: float

class VectorInput(BaseModel):
    x: float
    y: float
    z: float

class VectorOutput(BaseModel):
    x: float
    y: float
    z: float
    magnitude: Optional[float] = None

class LinePointDirectionInput(BaseModel):
    point: PointInput = Field(..., description="A point on the line (x0, y0, z0)")
    direction_vector: VectorInput = Field(..., description="Direction vector of the line (a, b, c)")

class LineTwoPointsInput(BaseModel):
    point1: PointInput
    point2: PointInput

class PlaneNormalPointInput(BaseModel):
    normal_vector: VectorInput
    point_on_plane: PointInput

class PlaneNormalDistanceInput(BaseModel):
    normal_vector: VectorInput
    distance_from_origin: float

class PlaneThreePointsInput(BaseModel):
    point1: PointInput
    point2: PointInput
    point3: PointInput

class PlaneEquationCoeffsInput(BaseModel):
    a: float = Field(..., description="Coefficient of x in Ax + By + Cz = D or Ax + By + Cz + D = 0")
    b: float = Field(..., description="Coefficient of y")
    c: float = Field(..., description="Coefficient of z")
    d_rhs: Optional[float] = Field(None, description="Constant D if equation is Ax + By + Cz = D (D on RHS)")
    d_lhs: Optional[float] = Field(None, description="Constant D if equation is Ax + By + Cz + D = 0 (D on LHS)")

    # @validator('d_rhs')
    def check_one_d_must_be_present(cls, v, values):
        if v is None and values.get('d_lhs') is None:
            raise ValueError('Either d_rhs (for Ax+By+Cz=D) or d_lhs (for Ax+By+Cz+D=0) must be provided.')
        if v is not None and values.get('d_lhs') is not None:
            raise ValueError('Provide only one of d_rhs or d_lhs, not both.')
        return v

# --- Vector Algebra Results ---
class VectorOperationResult(VectorOutput):
    pass

class MagnitudeResult(BaseModel):
    vector: VectorInput
    magnitude: float

class UnitVectorResult(VectorOutput):
    original_vector: VectorInput

class SectionFormulaResult(PointInput):
    ratio_m: float
    ratio_n: float
    division_type: str # "internal" or "external"

class CollinearityResult(BaseModel):
    are_collinear: bool
    reason: str
    vectors_or_points: List[Dict[str, Any]]

# --- Scalar and Vector Product Results ---
class DotProductResult(BaseModel):
    vector1: VectorInput
    vector2: VectorInput
    dot_product: float
    angle_degrees: Optional[float] = None # Angle between the two vectors
    angle_radians: Optional[float] = None

class ProjectionResult(BaseModel):
    vector_a: VectorInput # Vector being projected
    vector_b: VectorInput # Vector onto which a is projected
    scalar_projection: float
    vector_projection: VectorOutput

class CrossProductResult(BaseModel):
    vector1: VectorInput
    vector2: VectorInput
    cross_product_vector: VectorOutput
    magnitude_of_cross_product: Optional[float] = None # Useful for area calculations
    area_of_parallelogram: Optional[float] = None # If v1, v2 are adjacent sides

class AreaTriangleResult(BaseModel):
    points_or_vectors: List[Dict[str, Any]]
    area: float

class ScalarTripleProductResult(BaseModel):
    vector_a: VectorInput
    vector_b: VectorInput
    vector_c: VectorInput
    scalar_triple_product: float # Volume of parallelepiped
    are_coplanar: bool

# --- Direction Cosines and Ratios ---
class DirectionRatiosOutput(BaseModel):
    a: float
    b: float
    c: float

class DirectionCosinesOutput(BaseModel):
    l: float
    m: float
    n: float
    is_valid_set: bool = Field(True, description="True if l^2+m^2+n^2 = 1 within tolerance")

# --- Straight Line Results ---
class LineEquationOutput(BaseModel):
    type: str # e.g., "vector_form", "cartesian_symmetric_form", "cartesian_parametric_form"
    equation_str: str
    point_on_line: Optional[PointInput] = None
    direction_vector: Optional[VectorOutput] = None
    direction_ratios: Optional[DirectionRatiosOutput] = None
    direction_cosines: Optional[DirectionCosinesOutput] = None

class AngleBetweenLinesResult(BaseModel):
    line1_definition: Dict[str, Any]
    line2_definition: Dict[str, Any]
    angle_degrees: float
    angle_radians: float

class LinesRelationshipResult(BaseModel):
    line1_definition: Dict[str, Any]
    line2_definition: Dict[str, Any]
    relationship: str # e.g., "parallel", "intersecting", "skew", "perpendicular"
    intersection_point: Optional[PointInput] = None
    shortest_distance: Optional[float] = None
    message: Optional[str] = None

class DistancePointLineResult(BaseModel):
    point: PointInput
    line_definition: Dict[str, Any]
    distance: float
    foot_of_perpendicular: Optional[PointInput] = None

class ImagePointInLineResult(BaseModel):
    point: PointInput
    line_definition: Dict[str, Any]
    image_point: PointInput

# --- Plane Results ---
class PlaneEquationOutput(BaseModel):
    type: str # e.g., "vector_normal_form", "cartesian_form", "intercept_form"
    equation_str: str
    normal_vector: Optional[VectorOutput] = None
    distance_from_origin: Optional[float] = None # For normal form
    point_on_plane: Optional[PointInput] = None # For point-normal form
    coeffs_cartesian: Optional[Dict[str,float]] = None # {a,b,c,d} for Ax+By+Cz+D=0

class CoplanarityLinesResult(BaseModel):
    line1_definition: Dict[str, Any]
    line2_definition: Dict[str, Any]
    are_coplanar: bool
    equation_of_plane_containing_lines: Optional[PlaneEquationOutput] = None # If coplanar and not parallel
    message: Optional[str] = None

class AngleBetweenPlanesResult(BaseModel):
    plane1_definition: Dict[str, Any]
    plane2_definition: Dict[str, Any]
    angle_degrees: float
    angle_radians: float

class AngleLinePlaneResult(BaseModel):
    line_definition: Dict[str, Any]
    plane_definition: Dict[str, Any]
    angle_degrees: float
    angle_radians: float

class RelationshipLinePlaneResult(BaseModel):
    line_definition: Dict[str, Any]
    plane_definition: Dict[str, Any]
    relationship: str # e.g., "line_parallel_to_plane", "line_lies_in_plane", "line_intersects_plane"
    intersection_point: Optional[PointInput] = None
    distance_if_parallel: Optional[float] = None # Distance between line and plane if parallel
    message: Optional[str] = None

class DistancePointPlaneResult(BaseModel):
    point: PointInput
    plane_definition: Dict[str, Any]
    distance: float
    foot_of_perpendicular: Optional[PointInput] = None

class ImagePointInPlaneResult(BaseModel):
    point: PointInput
    plane_definition: Dict[str, Any]
    image_point: PointInput

class IntersectionLinePlaneResult(BaseModel):
    line_definition: Dict[str, Any]
    plane_definition: Dict[str, Any]
    intersects: bool
    intersection_point: Optional[PointInput] = None
    message: str

class IntersectionTwoPlanesResult(BaseModel):
    plane1_definition: Dict[str, Any]
    plane2_definition: Dict[str, Any]
    intersects_in_line: bool
    line_of_intersection: Optional[LineEquationOutput] = None # Vector form
    message: str