# routers/three_d_router.py
from fastapi import APIRouter, Body, HTTPException, Query
from models import three_d_models as td_models
from services import three_d_geometry_service as td_service
from typing import List, Dict, Any
import numpy as np  # For degrees conversion

router = APIRouter(prefix="/three-d-geometry", tags=["3D Geometry"])


# --- 23. ALGEBRA OF VECTORS ---
@router.post("/vectors/add", response_model=td_models.VectorOperationResult)
async def add_two_vectors(v1: td_models.VectorInput, v2: td_models.VectorInput):
    return td_service.add_vectors(v1, v2)


@router.post("/vectors/subtract", response_model=td_models.VectorOperationResult)
async def subtract_two_vectors(v1: td_models.VectorInput, v2: td_models.VectorInput):
    return td_service.subtract_vectors(v1, v2)


@router.post("/vectors/scalar-multiply", response_model=td_models.VectorOperationResult)
async def scalar_multiply(
    scalar: float = Query(..., example=2.5), vector: td_models.VectorInput = Body(...)
):
    return td_service.scalar_multiply_vector(scalar, vector)


@router.post("/vectors/magnitude", response_model=td_models.MagnitudeResult)
async def get_vector_magnitude(vector: td_models.VectorInput):
    mag = td_service.vector_magnitude(vector)
    return td_models.MagnitudeResult(vector=vector, magnitude=mag)


@router.post("/vectors/unit-vector", response_model=td_models.UnitVectorResult)
async def get_unit_vector(vector: td_models.VectorInput):
    try:
        return td_service.unit_vector(vector)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/vectors/from-two-points", response_model=td_models.VectorOperationResult)
async def get_vector_from_points(p1: td_models.PointInput, p2: td_models.PointInput):
    return td_service.vector_from_two_points(p1, p2)


@router.post("/vectors/section-formula", response_model=td_models.SectionFormulaResult)
async def apply_section_formula(
    p1: td_models.PointInput,
    p2: td_models.PointInput,
    m: float = Query(..., example=1, description="Ratio m"),
    n: float = Query(..., example=2, description="Ratio n"),
    internal_division: bool = Query(
        True, description="True for internal, False for external division"
    ),
):
    try:
        return td_service.section_formula(p1, p2, m, n, internal_division)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/vectors/check-collinearity-points", response_model=td_models.CollinearityResult
)
async def check_points_collinearity(
    points: List[td_models.PointInput] = Body(..., min_items=2)
):
    return td_service.check_collinearity_points(points)


# --- 24. SCALAR OR DOT PRODUCT ---
@router.post("/vectors/dot-product", response_model=td_models.DotProductResult)
async def calculate_dot_product(v1: td_models.VectorInput, v2: td_models.VectorInput):
    return td_service.dot_product(v1, v2)


@router.post("/vectors/projection", response_model=td_models.ProjectionResult)
async def calculate_vector_projection(
    vector_to_project: td_models.VectorInput = Body(..., alias="vectorA"),
    vector_onto: td_models.VectorInput = Body(..., alias="vectorB"),
):
    try:
        return td_service.projection_vector_on_vector(vector_to_project, vector_onto)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- 25. VECTOR OR CROSS PRODUCT ---
@router.post("/vectors/cross-product", response_model=td_models.CrossProductResult)
async def calculate_cross_product(v1: td_models.VectorInput, v2: td_models.VectorInput):
    return td_service.cross_product(v1, v2)


@router.post(
    "/vectors/area-triangle-from-vectors", response_model=td_models.AreaTriangleResult
)
async def calculate_area_triangle_adj_sides(
    adjacent_side1: td_models.VectorInput, adjacent_side2: td_models.VectorInput
):
    return td_service.area_triangle_vectors(adjacent_side1, adjacent_side2)


@router.post(
    "/vectors/area-triangle-from-points", response_model=td_models.AreaTriangleResult
)
async def calculate_area_triangle_vertices(
    p1: td_models.PointInput, p2: td_models.PointInput, p3: td_models.PointInput
):
    return td_service.area_triangle_points(p1, p2, p3)


@router.post(
    "/vectors/scalar-triple-product", response_model=td_models.ScalarTripleProductResult
)
async def calculate_scalar_triple_product(
    vector_a: td_models.VectorInput,
    vector_b: td_models.VectorInput,
    vector_c: td_models.VectorInput,
):
    return td_service.scalar_triple_product(vector_a, vector_b, vector_c)


# --- 26. DIRECTION COSINES AND DIRECTION RATIOS ---
@router.post(
    "/lines/direction-ratios/from-vector",
    response_model=td_models.DirectionRatiosOutput,
)
async def dr_from_vector(vector: td_models.VectorInput):
    return td_service.direction_ratios_from_vector(vector)


@router.post(
    "/lines/direction-ratios/from-points",
    response_model=td_models.DirectionRatiosOutput,
)
async def dr_from_points(p1: td_models.PointInput, p2: td_models.PointInput):
    return td_service.direction_ratios_from_points(p1, p2)


@router.post(
    "/lines/direction-cosines/from-ratios",
    response_model=td_models.DirectionCosinesOutput,
)
async def dc_from_ratios(direction_ratios: td_models.DirectionRatiosOutput):
    return td_service.direction_cosines_from_ratios(direction_ratios)


@router.post(
    "/lines/direction-cosines/from-vector",
    response_model=td_models.DirectionCosinesOutput,
)
async def dc_from_vector(vector: td_models.VectorInput):
    return td_service.direction_cosines_from_vector(vector)


# --- 27. STRAIGHT LINE IN SPACE ---
@router.post("/lines/equation/vector-form", response_model=td_models.LineEquationOutput)
async def get_line_eq_vector(line_input: td_models.LinePointDirectionInput):
    return td_service.line_eq_vector_form(line_input.point, line_input.direction_vector)


@router.post(
    "/lines/equation/cartesian-symmetric-form",
    response_model=td_models.LineEquationOutput,
)
async def get_line_eq_cartesian_symm(
    point: td_models.PointInput, direction_ratios: td_models.DirectionRatiosOutput
):
    return td_service.line_eq_cartesian_symmetric(point, direction_ratios)


@router.post(
    "/lines/equation/from-two-points", response_model=List[td_models.LineEquationOutput]
)
async def get_line_eq_two_points(line_input: td_models.LineTwoPointsInput):
    try:
        vec_form, cart_form = td_service.line_eq_from_two_points(
            line_input.point1, line_input.point2
        )
        return [vec_form, cart_form]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/lines/angle-between", response_model=td_models.AngleBetweenLinesResult)
async def get_angle_between_lines(
    line1_dir: td_models.VectorInput = Body(
        ..., description="Direction vector/ratios of Line 1"
    ),
    line2_dir: td_models.VectorInput = Body(
        ..., description="Direction vector/ratios of Line 2"
    ),
):
    try:
        angle_rad = td_service.angle_between_lines(line1_dir, line2_dir)
        angle_deg = np.degrees(angle_rad)
        return td_models.AngleBetweenLinesResult(
            line1_definition={"direction_vector": line1_dir.dict()},
            line2_definition={"direction_vector": line2_dir.dict()},
            angle_degrees=angle_deg,
            angle_radians=angle_rad,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/lines/shortest-distance", response_model=Dict[str, Any]
)  # More specific model later
async def get_shortest_distance_between_lines(
    line1: td_models.LinePointDirectionInput, line2: td_models.LinePointDirectionInput
):
    try:
        distance = td_service.shortest_distance_skew_lines(
            line1.point, line1.direction_vector, line2.point, line2.direction_vector
        )
        # You might want to return more info, like if they are parallel or intersecting
        rel_info = td_service.lines_relationship(
            line1.point, line1.direction_vector, line2.point, line2.direction_vector
        )

        return {
            "line1_definition": {
                "point": line1.point.dict(),
                "direction_vector": line1.direction_vector.dict(),
            },
            "line2_definition": {
                "point": line2.point.dict(),
                "direction_vector": line2.direction_vector.dict(),
            },
            "shortest_distance": distance,
            "relationship_details": rel_info.dict(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/lines/relationship", response_model=td_models.LinesRelationshipResult)
async def get_lines_relationship(
    line1: td_models.LinePointDirectionInput, line2: td_models.LinePointDirectionInput
):
    return td_service.lines_relationship(
        line1.point, line1.direction_vector, line2.point, line2.direction_vector
    )


# --- 28. THE PLANE ---
@router.post(
    "/planes/equation/vector-normal-form", response_model=td_models.PlaneEquationOutput
)
async def get_plane_eq_vector_normal(plane_input: td_models.PlaneNormalDistanceInput):
    try:
        return td_service.plane_eq_normal_form_vector(
            plane_input.normal_vector, plane_input.distance_from_origin
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/planes/equation/cartesian-from-normal-point",
    response_model=td_models.PlaneEquationOutput,
)
async def get_plane_eq_cartesian_normal_point(
    plane_input: td_models.PlaneNormalPointInput,
):
    return td_service.plane_eq_cartesian_from_normal_point(
        plane_input.normal_vector, plane_input.point_on_plane
    )


@router.post(
    "/planes/equation/from-coefficients", response_model=td_models.PlaneEquationOutput
)
async def get_plane_eq_from_coeffs(coeffs_input: td_models.PlaneEquationCoeffsInput):
    try:
        return td_service.plane_eq_from_coeffs(coeffs_input)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/planes/equation/from-three-points", response_model=td_models.PlaneEquationOutput
)
async def get_plane_eq_three_points(plane_input: td_models.PlaneThreePointsInput):
    try:
        return td_service.plane_eq_from_three_points(
            plane_input.point1, plane_input.point2, plane_input.point3
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/planes/angle-between", response_model=td_models.AngleBetweenPlanesResult)
async def get_angle_between_planes(
    plane1_normal: td_models.VectorInput = Body(
        ..., description="Normal vector of Plane 1"
    ),
    plane2_normal: td_models.VectorInput = Body(
        ..., description="Normal vector of Plane 2"
    ),
):
    # This assumes user provides normals. Could also take full plane equations.
    try:
        angle_rad = td_service.angle_between_planes(plane1_normal, plane2_normal)
        angle_deg = np.degrees(angle_rad)
        return td_models.AngleBetweenPlanesResult(
            plane1_definition={"normal_vector": plane1_normal.dict()},
            plane2_definition={"normal_vector": plane2_normal.dict()},
            angle_degrees=angle_deg,
            angle_radians=angle_rad,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/planes/angle-line-plane", response_model=td_models.AngleLinePlaneResult)
async def get_angle_line_and_plane(
    line_direction: td_models.VectorInput = Body(
        ..., description="Direction vector of the line"
    ),
    plane_normal: td_models.VectorInput = Body(
        ..., description="Normal vector of the plane"
    ),
):
    try:
        angle_rad = td_service.angle_line_plane(line_direction, plane_normal)
        angle_deg = np.degrees(angle_rad)
        return td_models.AngleLinePlaneResult(
            line_definition={"direction_vector": line_direction.dict()},
            plane_definition={"normal_vector": plane_normal.dict()},
            angle_degrees=angle_deg,
            angle_radians=angle_rad,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/planes/distance-point-plane", response_model=td_models.DistancePointPlaneResult
)
async def get_distance_point_to_plane(
    point: td_models.PointInput,
    plane_coeffs: td_models.PlaneEquationCoeffsInput = Body(
        ..., description="Plane equation Ax+By+Cz+D=0 or Ax+By+Cz=D"
    ),
):
    try:
        return td_service.distance_point_plane(point, plane_coeffs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/planes/intersection-line-plane",
    response_model=td_models.IntersectionLinePlaneResult,
)
async def get_intersection_of_line_and_plane(
    line: td_models.LinePointDirectionInput,
    plane_coeffs: td_models.PlaneEquationCoeffsInput,
):
    try:
        return td_service.intersection_line_plane(
            line.point, line.direction_vector, plane_coeffs
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/planes/intersection-two-planes",
    response_model=td_models.IntersectionTwoPlanesResult,
)
async def get_intersection_of_two_planes(
    plane1_coeffs: td_models.PlaneEquationCoeffsInput,
    plane2_coeffs: td_models.PlaneEquationCoeffsInput,
):
    try:
        return td_service.intersection_two_planes(plane1_coeffs, plane2_coeffs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Additional endpoints to consider for full Class 12 coverage:
# - Coplanarity of two lines and equation of plane containing them.
# - Image of a point in a line.
# - Image of a point in a plane.
# - Equation of a plane passing through the intersection of two other planes and a given point.
# - Equation of a plane perpendicular to a given vector and at a given distance from origin (already covered by vector-normal form).
# - Equation of a plane in intercept form and conversion.
