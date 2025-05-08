# services/three_d_geometry_service.py
import numpy as np
from typing import Tuple, List, Optional, Dict, Any,Union
from models import three_d_models as models # Assuming your models are in this path

# --- Helper Functions ---
def _to_np_array(p_or_v: Union[models.PointInput, models.VectorInput]) -> np.ndarray:
    """Converts PointInput or VectorInput to a NumPy array."""
    return np.array([p_or_v.x, p_or_v.y, p_or_v.z])

def _format_vector_str(v: np.ndarray) -> str:
    return f"{v[0]:.2f}i + {v[1]:.2f}j + {v[2]:.2f}k"

def _format_point_str(p: np.ndarray) -> str:
    return f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"

EPSILON = 1e-9 # For floating point comparisons

# --- 23. ALGEBRA OF VECTORS ---
def add_vectors(v1_in: models.VectorInput, v2_in: models.VectorInput) -> models.VectorOperationResult:
    v1 = _to_np_array(v1_in)
    v2 = _to_np_array(v2_in)
    result_v = v1 + v2
    return models.VectorOperationResult(x=result_v[0], y=result_v[1], z=result_v[2])

def subtract_vectors(v1_in: models.VectorInput, v2_in: models.VectorInput) -> models.VectorOperationResult:
    v1 = _to_np_array(v1_in)
    v2 = _to_np_array(v2_in)
    result_v = v1 - v2
    return models.VectorOperationResult(x=result_v[0], y=result_v[1], z=result_v[2])

def scalar_multiply_vector(scalar: float, v_in: models.VectorInput) -> models.VectorOperationResult:
    v = _to_np_array(v_in)
    result_v = scalar * v
    return models.VectorOperationResult(x=result_v[0], y=result_v[1], z=result_v[2])

def vector_magnitude(v_in: models.VectorInput) -> float:
    v = _to_np_array(v_in)
    return float(np.linalg.norm(v))

def unit_vector(v_in: models.VectorInput) -> models.UnitVectorResult:
    v = _to_np_array(v_in)
    mag = np.linalg.norm(v)
    if mag < EPSILON:
        raise ValueError("Cannot compute unit vector for a zero vector.")
    unit_v = v / mag
    return models.UnitVectorResult(
        original_vector=v_in,
        x=unit_v[0], y=unit_v[1], z=unit_v[2],
        magnitude=1.0
    )

def vector_from_two_points(p1_in: models.PointInput, p2_in: models.PointInput) -> models.VectorOperationResult:
    """Computes vector P1P2 = P2 - P1."""
    p1 = _to_np_array(p1_in)
    p2 = _to_np_array(p2_in)
    result_v = p2 - p1
    return models.VectorOperationResult(x=result_v[0], y=result_v[1], z=result_v[2])

def section_formula(
    p1_in: models.PointInput, p2_in: models.PointInput,
    m: float, n: float, internal: bool = True
) -> models.SectionFormulaResult:
    p1 = _to_np_array(p1_in)
    p2 = _to_np_array(p2_in)
    if internal:
        r_point = (n * p1 + m * p2) / (m + n)
        div_type = "internal"
    else: # external
        if abs(m - n) < EPSILON:
            raise ValueError("For external division, m cannot be equal to n.")
        r_point = (m * p2 - n * p1) / (m - n)
        div_type = "external"
    return models.SectionFormulaResult(
        x=r_point[0], y=r_point[1], z=r_point[2],
        ratio_m=m, ratio_n=n, division_type=div_type
    )

def check_collinearity_points(points_in: List[models.PointInput]) -> models.CollinearityResult:
    if len(points_in) < 2:
        return models.CollinearityResult(are_collinear=True, reason="Less than 2 points are trivially collinear.", vectors_or_points=[p.dict() for p in points_in])
    if len(points_in) == 2:
        return models.CollinearityResult(are_collinear=True, reason="Two points are always collinear.", vectors_or_points=[p.dict() for p in points_in])

    p_arr = [_to_np_array(p) for p in points_in]
    v1 = p_arr[1] - p_arr[0]
    
    for i in range(2, len(p_arr)):
        v_current = p_arr[i] - p_arr[0]
        # Check if v_current is parallel to v1 (cross product is zero vector)
        cross_prod = np.cross(v1, v_current)
        if np.linalg.norm(cross_prod) > EPSILON:
            return models.CollinearityResult(
                are_collinear=False,
                reason=f"Vector from point 0 to point {i} is not parallel to vector from point 0 to point 1.",
                vectors_or_points=[p.dict() for p in points_in]
            )
    return models.CollinearityResult(
        are_collinear=True,
        reason="All vectors formed from the first point to other points are parallel.",
        vectors_or_points=[p.dict() for p in points_in]
    )


# --- 24. SCALAR OR DOT PRODUCT ---
def dot_product(v1_in: models.VectorInput, v2_in: models.VectorInput) -> models.DotProductResult:
    v1 = _to_np_array(v1_in)
    v2 = _to_np_array(v2_in)
    dp = np.dot(v1, v2)
    
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    angle_rad, angle_deg = None, None
    if mag1 > EPSILON and mag2 > EPSILON:
        cos_theta = dp / (mag1 * mag2)
        # Clamp cos_theta to [-1, 1] to avoid domain errors with acos due to precision
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)
        
    return models.DotProductResult(
        vector1=v1_in, vector2=v2_in, dot_product=float(dp),
        angle_degrees=angle_deg, angle_radians=angle_rad
    )

def projection_vector_on_vector(
    a_in: models.VectorInput,  # Vector to be projected
    b_in: models.VectorInput   # Vector onto which a is projected
) -> models.ProjectionResult:
    a = _to_np_array(a_in)
    b = _to_np_array(b_in)
    mag_b = np.linalg.norm(b)
    if mag_b < EPSILON:
        raise ValueError("Cannot project onto a zero vector.")
    
    scalar_proj = np.dot(a, b) / mag_b
    unit_b = b / mag_b
    vector_proj_np = scalar_proj * unit_b
    
    vector_proj_output = models.VectorOutput(
        x=vector_proj_np[0], y=vector_proj_np[1], z=vector_proj_np[2]
    )
    return models.ProjectionResult(
        vector_a=a_in, vector_b=b_in,
        scalar_projection=scalar_proj,
        vector_projection=vector_proj_output
    )

# --- 25. VECTOR OR CROSS PRODUCT ---
def cross_product(v1_in: models.VectorInput, v2_in: models.VectorInput) -> models.CrossProductResult:
    v1 = _to_np_array(v1_in)
    v2 = _to_np_array(v2_in)
    cp_v = np.cross(v1, v2)
    mag_cp = np.linalg.norm(cp_v)
    
    cp_output = models.VectorOutput(x=cp_v[0], y=cp_v[1], z=cp_v[2])
    return models.CrossProductResult(
        vector1=v1_in, vector2=v2_in,
        cross_product_vector=cp_output,
        magnitude_of_cross_product=mag_cp,
        area_of_parallelogram=mag_cp # Magnitude of cross product is area of parallelogram
    )

def area_triangle_vectors(v1_in: models.VectorInput, v2_in: models.VectorInput) -> models.AreaTriangleResult:
    """Area of triangle with adjacent sides v1, v2."""
    cp_res = cross_product(v1_in, v2_in)
    area = 0.5 * cp_res.magnitude_of_cross_product
    return models.AreaTriangleResult(
        points_or_vectors=[v1_in.dict(), v2_in.dict()],
        area=area
    )

def area_triangle_points(p1_in: models.PointInput, p2_in: models.PointInput, p3_in: models.PointInput) -> models.AreaTriangleResult:
    """Area of triangle with vertices p1, p2, p3."""
    # Form two vectors representing two sides of the triangle, e.g., P1P2 and P1P3
    v_p1p2_in = vector_from_two_points(p1_in, p2_in)
    v_p1p3_in = vector_from_two_points(p1_in, p3_in)
    
    # The vector inputs to area_triangle_vectors must be VectorInput models
    v1_model = models.VectorInput(x=v_p1p2_in.x, y=v_p1p2_in.y, z=v_p1p2_in.z)
    v2_model = models.VectorInput(x=v_p1p3_in.x, y=v_p1p3_in.y, z=v_p1p3_in.z)

    return area_triangle_vectors(v1_model, v2_model)


def scalar_triple_product(a_in: models.VectorInput, b_in: models.VectorInput, c_in: models.VectorInput) -> models.ScalarTripleProductResult:
    """Computes a · (b x c). Represents volume of parallelepiped."""
    a = _to_np_array(a_in)
    b = _to_np_array(b_in)
    c = _to_np_array(c_in)
    
    b_cross_c = np.cross(b, c)
    stp = np.dot(a, b_cross_c)
    
    are_coplanar = abs(stp) < EPSILON
    
    return models.ScalarTripleProductResult(
        vector_a=a_in, vector_b=b_in, vector_c=c_in,
        scalar_triple_product=float(stp),
        are_coplanar=are_coplanar
    )

# --- 26. DIRECTION COSINES AND DIRECTION RATIOS ---
def direction_ratios_from_vector(v_in: models.VectorInput) -> models.DirectionRatiosOutput:
    return models.DirectionRatiosOutput(a=v_in.x, b=v_in.y, c=v_in.z)

def direction_ratios_from_points(p1_in: models.PointInput, p2_in: models.PointInput) -> models.DirectionRatiosOutput:
    v = vector_from_two_points(p1_in, p2_in)
    return models.DirectionRatiosOutput(a=v.x, b=v.y, c=v.z)

def direction_cosines_from_ratios(dr_in: models.DirectionRatiosOutput) -> models.DirectionCosinesOutput:
    a, b, c = dr_in.a, dr_in.b, dr_in.c
    mag_sq = a*a + b*b + c*c
    if mag_sq < EPSILON: # Zero vector has undefined direction cosines
        return models.DirectionCosinesOutput(l=0, m=0, n=0, is_valid_set=False) # Or raise error
        
    mag = np.sqrt(mag_sq)
    l, m, n = a/mag, b/mag, c/mag
    is_valid = abs((l*l + m*m + n*n) - 1.0) < EPSILON
    return models.DirectionCosinesOutput(l=l, m=m, n=n, is_valid_set=is_valid)

def direction_cosines_from_vector(v_in: models.VectorInput) -> models.DirectionCosinesOutput:
    dr = direction_ratios_from_vector(v_in)
    return direction_cosines_from_ratios(dr)


# --- 27. STRAIGHT LINE IN SPACE ---
def line_eq_vector_form(point_in: models.PointInput, dir_vec_in: models.VectorInput) -> models.LineEquationOutput:
    p0_str = _format_point_str(_to_np_array(point_in))
    dir_v_str = _format_vector_str(_to_np_array(dir_vec_in))
    eq_str = f"r = ({p0_str}) + λ({dir_v_str})"
    return models.LineEquationOutput(
        type="vector_form", equation_str=eq_str,
        point_on_line=point_in,
        direction_vector=models.VectorOutput(**dir_vec_in.dict(), magnitude=vector_magnitude(dir_vec_in))
    )

def line_eq_cartesian_symmetric(point_in: models.PointInput, dir_ratios_in: models.DirectionRatiosOutput) -> models.LineEquationOutput:
    x0, y0, z0 = point_in.x, point_in.y, point_in.z
    a, b, c = dir_ratios_in.a, dir_ratios_in.b, dir_ratios_in.c
    
    parts = []
    if abs(a) > EPSILON: parts.append(f"(x - {x0:.2f}) / {a:.2f}")
    else: parts.append(f"x = {x0:.2f}") # Line parallel to yz-plane (or axis if b,c also 0)

    if abs(b) > EPSILON: parts.append(f"(y - {y0:.2f}) / {b:.2f}")
    else: parts.append(f"y = {y0:.2f}")

    if abs(c) > EPSILON: parts.append(f"(z - {z0:.2f}) / {c:.2f}")
    else: parts.append(f"z = {z0:.2f}")
    
    # Handle cases where some direction ratios are zero
    eq_parts = []
    non_zero_denom_parts = [p for p in parts if "/" in p]
    zero_denom_parts = [p for p in parts if "/" not in p]

    if len(non_zero_denom_parts) >= 2:
        eq_str = " = ".join(non_zero_denom_parts)
        if zero_denom_parts:
            eq_str += "; " + ", ".join(zero_denom_parts)
    elif len(non_zero_denom_parts) == 1: # e.g. (x-x0)/a ; y=y0 ; z=z0
        eq_str = non_zero_denom_parts[0] + " (parameterized); " + ", ".join(zero_denom_parts)
    else: # All direction ratios are zero - this is a point, not a line. Or invalid input.
        if abs(a)<EPSILON and abs(b)<EPSILON and abs(c)<EPSILON:
            return models.LineEquationOutput(type="invalid_line", equation_str=f"Point: ({x0:.2f}, {y0:.2f}, {z0:.2f}) - Direction ratios are all zero.")
        eq_str = ", ".join(zero_denom_parts) # e.g. x=x0, y=y0, z=z0 (if a,b,c were 0, error)

    return models.LineEquationOutput(
        type="cartesian_symmetric_form", equation_str=eq_str,
        point_on_line=point_in,
        direction_ratios=dir_ratios_in
    )

def line_eq_from_two_points(p1_in: models.PointInput, p2_in: models.PointInput) -> Tuple[models.LineEquationOutput, models.LineEquationOutput]:
    dir_vec = vector_from_two_points(p1_in, p2_in)
    dir_ratios = models.DirectionRatiosOutput(a=dir_vec.x, b=dir_vec.y, c=dir_vec.z)
    
    if np.linalg.norm(_to_np_array(dir_vec)) < EPSILON: # Points are same
        raise ValueError("The two points are coincident, cannot define a unique line.")

    vec_form = line_eq_vector_form(p1_in, models.VectorInput(**dir_vec.dict())) # Use p1 as the point
    cart_form = line_eq_cartesian_symmetric(p1_in, dir_ratios)
    return vec_form, cart_form

def angle_between_lines(
    dir1_in: models.VectorInput, # Or DirectionRatiosInput
    dir2_in: models.VectorInput  # Or DirectionRatiosInput
) -> float: # Returns angle in radians
    """Angle between two lines given their direction vectors/ratios."""
    d1 = _to_np_array(dir1_in)
    d2 = _to_np_array(dir2_in)
    
    dot_prod = np.dot(d1, d2)
    mag_d1 = np.linalg.norm(d1)
    mag_d2 = np.linalg.norm(d2)
    
    if mag_d1 < EPSILON or mag_d2 < EPSILON:
        raise ValueError("Direction vector(s) cannot be zero vector for angle calculation.")
        
    cos_theta = dot_prod / (mag_d1 * mag_d2)
    cos_theta = max(-1.0, min(1.0, cos_theta)) # Clamp for precision
    return np.arccos(cos_theta) # Angle in radians

def shortest_distance_skew_lines(
    p1_in: models.PointInput, d1_in: models.VectorInput, # Line 1: r = p1 + λd1
    p2_in: models.PointInput, d2_in: models.VectorInput  # Line 2: r = p2 + μd2
) -> float:
    p1 = _to_np_array(p1_in); d1 = _to_np_array(d1_in)
    p2 = _to_np_array(p2_in); d2 = _to_np_array(d2_in)
    
    # Check if lines are parallel first
    cross_d1_d2 = np.cross(d1, d2)
    if np.linalg.norm(cross_d1_d2) < EPSILON: # Lines are parallel
        # Distance between parallel lines = |(p2 - p1) x d1| / |d1|
        p2_minus_p1 = p2 - p1
        num = np.linalg.norm(np.cross(p2_minus_p1, d1))
        den = np.linalg.norm(d1)
        if den < EPSILON: return 0.0 # Should not happen if d1 is valid direction
        return num / den

    # Skew lines: SD = | (p2 - p1) · (d1 x d2) | / |d1 x d2 |
    p2_minus_p1 = p2 - p1
    numerator = abs(np.dot(p2_minus_p1, cross_d1_d2))
    denominator = np.linalg.norm(cross_d1_d2) # Already calculated
    
    if denominator < EPSILON: # This implies d1 x d2 is zero vector -> lines are parallel
        # This case should be caught above, but defensively:
        return 0.0 # Or handle as error, as this means problem with input logic
        
    return numerator / denominator

def lines_relationship(
    p1_in: models.PointInput, d1_in: models.VectorInput,
    p2_in: models.PointInput, d2_in: models.VectorInput
) -> models.LinesRelationshipResult:
    p1 = _to_np_array(p1_in); d1 = _to_np_array(d1_in)
    p2 = _to_np_array(p2_in); d2 = _to_np_array(d2_in)
    
    line1_def = {"point": p1_in.dict(), "direction_vector": d1_in.dict()}
    line2_def = {"point": p2_in.dict(), "direction_vector": d2_in.dict()}

    cross_d1_d2 = np.cross(d1, d2)
    p2_minus_p1 = p2 - p1
    
    shortest_dist = abs(np.dot(p2_minus_p1, cross_d1_d2)) / np.linalg.norm(cross_d1_d2) \
                    if np.linalg.norm(cross_d1_d2) > EPSILON else \
                    np.linalg.norm(np.cross(p2_minus_p1, d1)) / np.linalg.norm(d1) \
                    if np.linalg.norm(d1) > EPSILON else 0.0


    if np.linalg.norm(cross_d1_d2) < EPSILON: # Parallel lines
        # Check if collinear (i.e., same line)
        # If p2-p1 is parallel to d1 (or d2), they are collinear.
        if np.linalg.norm(np.cross(p2_minus_p1, d1)) < EPSILON:
            return models.LinesRelationshipResult(
                line1_definition=line1_def, line2_definition=line2_def,
                relationship="collinear (same line)", shortest_distance=0.0
            )
        else:
            return models.LinesRelationshipResult(
                line1_definition=line1_def, line2_definition=line2_def,
                relationship="parallel_distinct", shortest_distance=shortest_dist
            )
    else: # Not parallel, so either intersecting or skew
        # Check for intersection: (p2-p1, d1, d2) must be coplanar (STP = 0)
        stp = np.dot(p2_minus_p1, cross_d1_d2)
        if abs(stp) < EPSILON: # Intersecting
            # Find intersection point: p1 + t*d1 = p2 + s*d2
            # d1x*t - d2x*s = p2x - p1x
            # d1y*t - d2y*s = p2y - p1y
            # (d1z*t - d2z*s = p2z - p1z) - use first two to solve for t,s then verify with third
            
            # System: A * [t, -s]^T = B
            # A = [[d1x, d2x], [d1y, d2y]]
            # B = [p2x-p1x, p2y-p1y]
            # However, we want to solve for t (from L1) and s (from L2)
            # t*d1_x - s*d2_x = p2_x - p1_x
            # t*d1_y - s*d2_y = p2_y - p1_y
            
            # Matrix for [t, s]:
            # [[d1x, -d2x], [d1y, -d2y]] [t, s]^T = [p2x-p1x, p2y-p1y]
            # If z-components are non-zero and offer better conditioning, use them.
            # Let's try using components with largest magnitudes for d1, d2 to form the 2x2 system
            
            # For simplicity, we can assume a solution exists and find 't' or 's'
            # For example, using Cramer's rule on a 2x2 subsystem or np.linalg.solve
            # (t*d1 - s*d2 = p2 - p1)
            # This is a system of 3 equations with 2 unknowns (t,s).
            # Since we know they intersect, a solution for t,s exists.
            # We can solve for t from: (p2-p1) x d2 = t * (d1 x d2)
            # t = dot( (p2-p1) x d2, d1 x d2 ) / |d1 x d2|^2
            
            if np.linalg.norm(cross_d1_d2)**2 > EPSILON:
                t = np.dot(np.cross(p2_minus_p1, d2), cross_d1_d2) / (np.linalg.norm(cross_d1_d2)**2)
                intersection_pt_np = p1 + t * d1
                intersection_pt = models.PointInput(x=intersection_pt_np[0], y=intersection_pt_np[1], z=intersection_pt_np[2])
                return models.LinesRelationshipResult(
                    line1_definition=line1_def, line2_definition=line2_def,
                    relationship="intersecting", intersection_point=intersection_pt, shortest_distance=0.0
                )
            else: # Fallback if cross_d1_d2 is zero, (should be caught by parallel check)
                 return models.LinesRelationshipResult(
                    line1_definition=line1_def, line2_definition=line2_def,
                    relationship="intersecting (calculation error)", shortest_distance=0.0,
                    message="Lines determined intersecting by STP, but issue calculating intersection point."
                )
        else: # Skew
            return models.LinesRelationshipResult(
                line1_definition=line1_def, line2_definition=line2_def,
                relationship="skew", shortest_distance=shortest_dist
            )

# --- 28. THE PLANE ---
def plane_eq_normal_form_vector(normal_in: models.VectorInput, dist_origin: float) -> models.PlaneEquationOutput:
    n_vec = _to_np_array(normal_in)
    mag_n = np.linalg.norm(n_vec)
    if mag_n < EPSILON: raise ValueError("Normal vector cannot be zero.")
    
    unit_normal = n_vec / mag_n
    # Equation: r · n_unit = d (where d is perpendicular distance from origin)
    # If dist_origin is negative, it implies normal is pointing away from plane relative to origin.
    # Standard form usually has d >= 0.
    # If d < 0, can flip normal and d: r . (-n_unit) = -d
    
    d_eff = dist_origin
    normal_eff_np = unit_normal
    if d_eff < 0:
        d_eff = -d_eff
        normal_eff_np = -normal_eff_np

    normal_str = _format_vector_str(normal_eff_np)
    eq_str = f"r · ({normal_str}) = {d_eff:.2f}"
    
    return models.PlaneEquationOutput(
        type="vector_normal_form", equation_str=eq_str,
        normal_vector=models.VectorOutput(x=normal_eff_np[0], y=normal_eff_np[1], z=normal_eff_np[2], magnitude=1.0),
        distance_from_origin=d_eff
    )

def plane_eq_cartesian_from_normal_point(
    normal_in: models.VectorInput, point_in: models.PointInput
) -> models.PlaneEquationOutput:
    # A(x-x0) + B(y-y0) + C(z-z0) = 0 => Ax + By + Cz = Ax0 + By0 + Cz0
    n_vec = _to_np_array(normal_in)
    p0 = _to_np_array(point_in)
    
    A, B, C = n_vec[0], n_vec[1], n_vec[2]
    D_rhs = np.dot(n_vec, p0) # Ax0 + By0 + Cz0
    
    # Ax + By + Cz = D_rhs OR Ax + By + Cz - D_rhs = 0
    D_lhs = -D_rhs
    eq_str = f"{A:.2f}x + {B:.2f}y + {C:.2f}z + ({D_lhs:.2f}) = 0"
    
    return models.PlaneEquationOutput(
        type="cartesian_form", equation_str=eq_str,
        normal_vector=models.VectorOutput(x=A, y=B, z=C, magnitude=np.linalg.norm(n_vec)),
        point_on_plane=point_in,
        coeffs_cartesian={"a":A, "b":B, "c":C, "d":D_lhs} # for Ax+By+Cz+D=0
    )

def plane_eq_from_coeffs(coeffs_in: models.PlaneEquationCoeffsInput) -> models.PlaneEquationOutput:
    A, B, C = coeffs_in.a, coeffs_in.b, coeffs_in.c
    if coeffs_in.d_rhs is not None:
        D_rhs = coeffs_in.d_rhs
        D_lhs = -D_rhs
    else: # d_lhs must be provided
        D_lhs = coeffs_in.d_lhs
        D_rhs = -D_lhs
    
    eq_str = f"{A:.2f}x + {B:.2f}y + {C:.2f}z + ({D_lhs:.2f}) = 0"
    normal_np = np.array([A,B,C])
    mag_normal = np.linalg.norm(normal_np)

    dist_origin = None
    if mag_normal > EPSILON:
        # Convert to normal form: Ax+By+Cz = -D_lhs
        # Divide by sqrt(A^2+B^2+C^2)
        # lx + my + nz = p where p = -D_lhs / mag_normal
        # Ensure p is positive for standard normal form
        p_candidate = -D_lhs / mag_normal
        if p_candidate >=0:
            dist_origin = p_candidate
            norm_for_normal_form = normal_np / mag_normal
        else:
            dist_origin = -p_candidate
            norm_for_normal_form = -normal_np / mag_normal # flip normal
    
    return models.PlaneEquationOutput(
        type="cartesian_form (from coeffs)", equation_str=eq_str,
        normal_vector=models.VectorOutput(x=A, y=B, z=C, magnitude=mag_normal),
        coeffs_cartesian={"a":A, "b":B, "c":C, "d":D_lhs},
        distance_from_origin=dist_origin
    )


def plane_eq_from_three_points(p1_in: models.PointInput, p2_in: models.PointInput, p3_in: models.PointInput) -> models.PlaneEquationOutput:
    p1 = _to_np_array(p1_in)
    p2 = _to_np_array(p2_in)
    p3 = _to_np_array(p3_in)
    
    v12 = p2 - p1 # Vector P1P2
    v13 = p3 - p1 # Vector P1P3
    
    normal_vec_np = np.cross(v12, v13)
    if np.linalg.norm(normal_vec_np) < EPSILON:
        raise ValueError("The three points are collinear and do not define a unique plane.")
        
    normal_in_model = models.VectorInput(x=normal_vec_np[0], y=normal_vec_np[1], z=normal_vec_np[2])
    return plane_eq_cartesian_from_normal_point(normal_in_model, p1_in) # Use p1 as point on plane

def angle_between_planes(
    n1_in: models.VectorInput, # Normal to plane 1
    n2_in: models.VectorInput  # Normal to plane 2
) -> float: # Returns angle in radians
    """Angle between two planes is the angle between their normals."""
    return angle_between_lines(n1_in, n2_in) # Re-use line angle logic

def angle_line_plane(
    line_dir_in: models.VectorInput,
    plane_normal_in: models.VectorInput
) -> float: # Returns angle in radians
    """Angle between a line and a plane."""
    # If θ is angle between line_dir and plane_normal,
    # then angle between line and plane is α = π/2 - θ (or |π/2 - θ|)
    # sin(α) = cos(θ) = (line_dir · plane_normal) / (|line_dir| * |plane_normal|)
    # Let's calculate θ first.
    
    d = _to_np_array(line_dir_in)
    n = _to_np_array(plane_normal_in)

    mag_d = np.linalg.norm(d)
    mag_n = np.linalg.norm(n)

    if mag_d < EPSILON or mag_n < EPSILON:
        raise ValueError("Direction/normal vector(s) cannot be zero.")

    dot_dn = np.dot(d, n)
    cos_theta = dot_dn / (mag_d * mag_n)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    
    # Angle between line and plane (α) is such that sin(α) = |cos(θ)|
    # So α = arcsin(|cos(θ)|) = arcsin(|d·n / (|d||n|)|)
    sin_alpha = abs(cos_theta) 
    alpha_rad = np.arcsin(sin_alpha)
    return alpha_rad

def distance_point_plane(
    point_in: models.PointInput,
    plane_coeffs_in: models.PlaneEquationCoeffsInput # Ax+By+Cz+D=0 form
) -> models.DistancePointPlaneResult:
    p0 = _to_np_array(point_in) # (x0, y0, z0)
    A, B, C = plane_coeffs_in.a, plane_coeffs_in.b, plane_coeffs_in.c
    
    if plane_coeffs_in.d_lhs is not None:
        D = plane_coeffs_in.d_lhs
    elif plane_coeffs_in.d_rhs is not None:
        D = -plane_coeffs_in.d_rhs
    else:
        raise ValueError("Plane constant D not correctly specified.")

    # Distance = |Ax0 + By0 + Cz0 + D| / sqrt(A^2 + B^2 + C^2)
    numerator = abs(A*p0[0] + B*p0[1] + C*p0[2] + D)
    denominator = np.sqrt(A*A + B*B + C*C)
    
    if denominator < EPSILON:
        raise ValueError("Plane normal vector (A,B,C) is zero, plane is ill-defined.")
        
    dist = numerator / denominator

    # Foot of perpendicular: P_foot = P0 - t * N, where N=(A,B,C)
    # t = (A*x0 + B*y0 + C*z0 + D) / (A^2+B^2+C^2)
    # Note the sign of t depends on which side P0 is.
    # If Ax0+By0+Cz0+D > 0, P0 is on the side the normal points to (from origin for D>0).
    # For P_foot = P0 - t*N, t must have same sign as Ax0+..+D
    t_foot = (A*p0[0] + B*p0[1] + C*p0[2] + D) / (denominator**2)
    normal_np = np.array([A,B,C])
    foot_np = p0 - t_foot * normal_np
    foot_model = models.PointInput(x=foot_np[0], y=foot_np[1], z=foot_np[2])
    
    return models.DistancePointPlaneResult(
        point=point_in,
        plane_definition=plane_coeffs_in.dict(),
        distance=dist,
        foot_of_perpendicular=foot_model
    )

def intersection_line_plane(
    line_point_in: models.PointInput, line_dir_in: models.VectorInput, # r = p + λd
    plane_coeffs_in: models.PlaneEquationCoeffsInput # Ax+By+Cz+D=0
) -> models.IntersectionLinePlaneResult:
    p = _to_np_array(line_point_in) # (px, py, pz)
    d = _to_np_array(line_dir_in)   # (dx, dy, dz)
    A, B, C = plane_coeffs_in.a, plane_coeffs_in.b, plane_coeffs_in.c
    if plane_coeffs_in.d_lhs is not None: D_lhs = plane_coeffs_in.d_lhs
    else: D_lhs = -plane_coeffs_in.d_rhs

    line_def = {"point": line_point_in.dict(), "direction": line_dir_in.dict()}
    plane_def = plane_coeffs_in.dict()

    # Line: x=px+λdx, y=py+λdy, z=pz+λdz
    # Substitute into plane: A(px+λdx) + B(py+λdy) + C(pz+λdz) + D_lhs = 0
    # A*px + A*λdx + B*py + B*λdy + C*pz + C*λdz + D_lhs = 0
    # λ(A*dx + B*dy + C*dz) = -(A*px + B*py + C*pz + D_lhs)
    # λ * (N·d) = -(N·p + D_lhs)
    
    normal_plane = np.array([A, B, C])
    N_dot_d = np.dot(normal_plane, d)
    
    if abs(N_dot_d) < EPSILON: # Line is parallel to plane (or lies in it)
        # Check if point p of line lies on the plane: A*px + B*py + C*pz + D_lhs = 0
        if abs(A*p[0] + B*p[1] + C*p[2] + D_lhs) < EPSILON:
            return models.IntersectionLinePlaneResult(
                line_definition=line_def, plane_definition=plane_def,
                intersects=True, intersection_point=None, # Represents infinitely many points
                message="Line lies in the plane."
            )
        else:
            return models.IntersectionLinePlaneResult(
                line_definition=line_def, plane_definition=plane_def,
                intersects=False, intersection_point=None,
                message="Line is parallel to the plane and does not intersect."
            )
    else: # Unique intersection point
        lambda_val = -(np.dot(normal_plane, p) + D_lhs) / N_dot_d
        intersection_pt_np = p + lambda_val * d
        intersection_pt_model = models.PointInput(
            x=intersection_pt_np[0], y=intersection_pt_np[1], z=intersection_pt_np[2]
        )
        return models.IntersectionLinePlaneResult(
            line_definition=line_def, plane_definition=plane_def,
            intersects=True, intersection_point=intersection_pt_model,
            message="Line intersects the plane at a single point."
        )

# ... more functions for other specific plane operations can be added:
# - Equation of plane passing through intersection of two other planes
# - Coplanarity of two lines
# - Image of a point in a line/plane
# - Intersection of two planes (results in a line)

def intersection_two_planes(
    plane1_coeffs_in: models.PlaneEquationCoeffsInput, # A1x+B1y+C1z+D1=0
    plane2_coeffs_in: models.PlaneEquationCoeffsInput  # A2x+B2y+C2z+D2=0
) -> models.IntersectionTwoPlanesResult:
    
    n1 = np.array([plane1_coeffs_in.a, plane1_coeffs_in.b, plane1_coeffs_in.c])
    D1 = plane1_coeffs_in.d_lhs if plane1_coeffs_in.d_lhs is not None else -plane1_coeffs_in.d_rhs
    
    n2 = np.array([plane2_coeffs_in.a, plane2_coeffs_in.b, plane2_coeffs_in.c])
    D2 = plane2_coeffs_in.d_lhs if plane2_coeffs_in.d_lhs is not None else -plane2_coeffs_in.d_rhs

    plane1_def = plane1_coeffs_in.dict()
    plane2_def = plane2_coeffs_in.dict()

    # Direction vector of line of intersection is n1 x n2
    line_dir_np = np.cross(n1, n2)

    if np.linalg.norm(line_dir_np) < EPSILON: # Normals are parallel -> planes are parallel
        # Check if they are the same plane: n1/D1 = n2/D2 (or if D1,D2 are 0)
        # A simpler check: pick a point on plane1, see if it's on plane2.
        # If n1 and n2 are proportional and D1 and D2 are also in that proportion, they are same.
        # (e.g. A1/A2 = B1/B2 = C1/C2 = D1/D2)
        # This is complex with zero divisions.
        # A robust check: if n1 = k*n2, then planes are parallel. If also D1 = k*D2, then coincident.
        
        # If D1 = D2 = 0, and n1, n2 parallel, then they are the same plane through origin (if normals scaled same)
        # More simply: are the equations multiples of each other?
        # Test ratio for non-zero components
        ratios = []
        if abs(n2[0]) > EPSILON: ratios.append(n1[0]/n2[0])
        if abs(n2[1]) > EPSILON: ratios.append(n1[1]/n2[1])
        if abs(n2[2]) > EPSILON: ratios.append(n1[2]/n2[2])
        
        # Check if all available ratios are consistent
        are_normals_proportional = True
        if len(ratios) > 1:
            for i in range(1, len(ratios)):
                if abs(ratios[i] - ratios[0]) > EPSILON:
                    are_normals_proportional = False # Should not happen if cross product is zero
                    break
        
        # If normals are proportional, check if point from plane1 satisfies plane2
        # (n1 . x + D1 = 0), (k*n1 . x + D2 = 0) => (n1 . x + D2/k = 0)
        # So if D1 = D2/k => k*D1 = D2
        
        same_plane = False
        # A quick check for parallel planes to be the same: n1xD2 == n2xD1 (vector components of n cross D)
        # No, that's not right.
        # If normals are parallel, check if the planes are separated by distance.
        # dist = |D1/|n1| - D2/|n2|| (if both D are distances from origin for normalized normals)
        # Or simpler, are the equations (A1x+B1y+C1z+D1=0) and (A2x+B2y+C2z+D2=0) equivalent?
        # (A1,B1,C1,D1) must be proportional to (A2,B2,C2,D2)
        
        # A simple test: If n1 is parallel to n2, check if origin shifted by -D1/|n1|^2 * n1 (a point on plane 1)
        # also lies on plane 2.
        if are_normals_proportional: # True if cross product is zero
            # Try to find a common point. For Ax+By+Cz+D=0, if A!=0, x=-D/A, y=0, z=0 is a point.
            point_on_plane1 = None
            if abs(n1[0]) > EPSILON: point_on_plane1 = np.array([-D1/n1[0], 0, 0])
            elif abs(n1[1]) > EPSILON: point_on_plane1 = np.array([0, -D1/n1[1], 0])
            elif abs(n1[2]) > EPSILON: point_on_plane1 = np.array([0, 0, -D1/n1[2]])
            elif abs(D1) < EPSILON : # 0x+0y+0z+0=0, a valid plane (all space if no other constraints) - This is ill-defined.
                 pass # This case implies plane1 is trivial
            
            if point_on_plane1 is not None:
                if abs(np.dot(n2, point_on_plane1) + D2) < EPSILON:
                    same_plane = True
            elif abs(D1) < EPSILON and abs(D2) < EPSILON : # Both 0x+0y+0z=0
                 same_plane = True


        if same_plane:
            return models.IntersectionTwoPlanesResult(
                plane1_definition=plane1_def, plane2_definition=plane2_def,
                intersects_in_line=True, # Technically, infinite intersection (same plane)
                line_of_intersection=None, # Not a single line
                message="Planes are coincident (the same plane)."
            )
        else:
            return models.IntersectionTwoPlanesResult(
                plane1_definition=plane1_def, plane_definition=plane2_def,
                intersects_in_line=False, line_of_intersection=None,
                message="Planes are parallel and distinct."
            )

    # Planes intersect in a line. Find a point on the line of intersection.
    # Set one variable to 0 (e.g., z=0) and solve the 2D system:
    # A1x + B1y + D1 = 0
    # A2x + B2y + D2 = 0
    # Solve for x, y:
    # A1x + B1y = -D1
    # A2x + B2y = -D2
    
    # Matrix: [[A1, B1], [A2, B2]] [x,y]^T = [-D1, -D2]^T
    # Determinant = A1*B2 - A2*B1
    
    point_on_line_np = None
    det_xy = n1[0]*n2[1] - n2[0]*n1[1]
    det_yz = n1[1]*n2[2] - n2[1]*n1[2]
    det_zx = n1[2]*n2[0] - n2[2]*n1[0]

    # Try setting z=0
    if abs(det_xy) > EPSILON:
        # x = (-D1*B2 - (-D2)*B1) / det_xy = (D2*B1 - D1*B2) / det_xy
        # y = (A1*(-D2) - A2*(-D1)) / det_xy = (A2*D1 - A1*D2) / det_xy
        x_val = (D2*n1[1] - D1*n2[1]) / det_xy
        y_val = (D1*n2[0] - D2*n1[0]) / det_xy
        point_on_line_np = np.array([x_val, y_val, 0])
    # Try setting x=0
    elif abs(det_yz) > EPSILON:
        # B1y + C1z = -D1
        # B2y + C2z = -D2
        y_val = (D2*n1[2] - D1*n2[2]) / det_yz
        z_val = (D1*n2[1] - D2*n1[1]) / det_yz
        point_on_line_np = np.array([0, y_val, z_val])
    # Try setting y=0
    elif abs(det_zx) > EPSILON:
        # A1x + C1z = -D1
        # A2x + C2z = -D2
        z_val = (D2*n1[0] - D1*n2[0]) / det_zx
        x_val = (D1*n2[2] - D2*n1[2]) / det_zx
        point_on_line_np = np.array([x_val, 0, z_val])
    else:
        # This case implies line_dir_np was not zero, but all 2x2 sub-determinants are zero.
        # This can happen if line of intersection is parallel to an axis.
        # e.g. x=k (planes y=c1, z=c2) -> n1=(0,1,0), n2=(0,0,1), line_dir=(1,0,0)
        # det_xy=0, det_yz=1, det_zx=0.
        # So one of the above should have worked unless problem is ill-conditioned or error in logic.
        # A more robust way is needed if all these fail.
        return models.IntersectionTwoPlanesResult(
            plane1_definition=plane1_def, plane_definition=plane2_def,
            intersects_in_line=True, line_of_intersection=None,
            message="Planes intersect, but failed to find a specific point on the line of intersection (possibly line parallel to axis or edge case)."
        )
        
    point_on_line_model = models.PointInput(x=point_on_line_np[0], y=point_on_line_np[1], z=point_on_line_np[2])
    line_dir_model = models.VectorInput(x=line_dir_np[0], y=line_dir_np[1], z=line_dir_np[2])
    
    line_eq_output = line_eq_vector_form(point_on_line_model, line_dir_model)
    
    return models.IntersectionTwoPlanesResult(
        plane1_definition=plane1_def, plane_definition=plane2_def,
        intersects_in_line=True, line_of_intersection=line_eq_output,
        message="Planes intersect in a line."
    )