# services/geometry_services.py
import math

PI = math.pi
from typing import Dict, Optional


# --- Cuboid ---
def cuboid_surface_areas(
    length: float, breadth: float, height: float
) -> Dict[str, float]:
    if length <= 0 or breadth <= 0 or height <= 0:
        raise ValueError("Dimensions (length, breadth, height) must be positive.")
    lsa = 2 * (length + breadth) * height
    tsa = 2 * ((length * breadth) + (breadth * height) + (height * length))
    return {"lateral_surface_area": lsa, "total_surface_area": tsa}


def cuboid_volume(length: float, breadth: float, height: float) -> float:
    if length <= 0 or breadth <= 0 or height <= 0:
        raise ValueError("Dimensions (length, breadth, height) must be positive.")
    return length * breadth * height


def cuboid_diagonal(length: float, breadth: float, height: float) -> float:
    if length <= 0 or breadth <= 0 or height <= 0:
        raise ValueError("Dimensions (length, breadth, height) must be positive.")
    return math.sqrt(length**2 + breadth**2 + height**2)


# --- Cube ---
def cube_surface_areas(side: float) -> Dict[str, float]:
    if side <= 0:
        raise ValueError("Side length must be positive.")
    lsa = 4 * side**2
    tsa = 6 * side**2
    return {"lateral_surface_area": lsa, "total_surface_area": tsa}


def cube_volume(side: float) -> float:
    if side <= 0:
        raise ValueError("Side length must be positive.")
    return side**3


def cube_diagonal(side: float) -> float:
    if side <= 0:
        raise ValueError("Side length must be positive.")
    return side * math.sqrt(3)


# --- Right Circular Cylinder ---
def cylinder_surface_areas(radius: float, height: float) -> Dict[str, float]:
    if radius <= 0 or height <= 0:
        raise ValueError("Radius and height must be positive.")
    csa = 2 * PI * radius * height
    tsa = 2 * PI * radius * (height + radius)
    return {"curved_surface_area": csa, "total_surface_area": tsa}


def cylinder_volume(radius: float, height: float) -> float:
    if radius <= 0 or height <= 0:
        raise ValueError("Radius and height must be positive.")
    return PI * radius**2 * height


# --- Right Circular Cone ---
def cone_slant_height(radius: float, height: float) -> float:
    if radius <= 0 or height <= 0:
        raise ValueError("Radius and height must be positive.")
    return math.sqrt(radius**2 + height**2)


def cone_surface_areas(
    radius: float, height: float, slant_height: Optional[float] = None
) -> Dict[str, float]:
    if radius <= 0 or height <= 0:
        raise ValueError("Radius and height must be positive.")
    if slant_height is None:
        l = cone_slant_height(radius, height)
    elif (
        slant_height <= 0 or slant_height < radius or slant_height < height
    ):  # Basic sanity checks for l
        raise ValueError("Provided slant height is invalid.")
    else:
        l = slant_height
        # Verify consistency if h is also given: l^2 approx r^2 + h^2
        if (
            abs(l**2 - (radius**2 + height**2)) > 1e-6 and height > 0
        ):  # Allow height=0 if l=r
            raise ValueError(
                f"Provided slant height {l} is inconsistent with radius {radius} and height {height}."
            )

    csa = PI * radius * l
    tsa = PI * radius * (l + radius)
    return {
        "curved_surface_area": csa,
        "total_surface_area": tsa,
        "calculated_slant_height": l,
    }


def cone_volume(radius: float, height: float) -> float:
    if radius <= 0 or height <= 0:
        raise ValueError("Radius and height must be positive.")
    return (1 / 3) * PI * radius**2 * height


# --- Sphere ---
def sphere_surface_area(radius: float) -> float:
    if radius <= 0:
        raise ValueError("Radius must be positive.")
    return 4 * PI * radius**2


def sphere_volume(radius: float) -> float:
    if radius <= 0:
        raise ValueError("Radius must be positive.")
    return (4 / 3) * PI * radius**3


# --- Hemisphere ---
def hemisphere_surface_areas(radius: float) -> Dict[str, float]:
    if radius <= 0:
        raise ValueError("Radius must be positive.")
    csa = 2 * PI * radius**2
    tsa = 3 * PI * radius**2
    return {"curved_surface_area": csa, "total_surface_area": tsa}


def hemisphere_volume(radius: float) -> float:
    if radius <= 0:
        raise ValueError("Radius must be positive.")
    return (2 / 3) * PI * radius**3


# --- Frustum of a Cone ---
def frustum_slant_height(height: float, radius1: float, radius2: float) -> float:
    if height <= 0 or radius1 < 0 or radius2 < 0:  # Allow r=0 for cone case
        raise ValueError("Height must be positive and radii non-negative.")
    if radius1 == radius2:  # It's a cylinder
        return height
    return math.sqrt(height**2 + (radius1 - radius2) ** 2)


def frustum_surface_areas(
    height: float, radius1: float, radius2: float, slant_height: Optional[float] = None
) -> Dict[str, float]:
    # R is typically larger radius, r is smaller. Let's use radius1, radius2
    if height <= 0 or radius1 < 0 or radius2 < 0:
        raise ValueError("Height must be positive and radii non-negative.")
    if (
        radius1 < radius2
    ):  # Ensure R (radius1) is the larger for formula consistency if specific R,r are used
        radius1, radius2 = radius2, radius1  # Swap if r1 is smaller

    R, r = radius1, radius2

    if slant_height is None:
        l = frustum_slant_height(height, R, r)
    elif slant_height <= 0:
        raise ValueError("Provided slant height is invalid.")
    else:
        l = slant_height
        # Verify consistency
        if abs(l**2 - (height**2 + (R - r) ** 2)) > 1e-6 and not (
            R == r and l == height
        ):
            raise ValueError(f"Provided slant height {l} inconsistent with dimensions.")

    csa = PI * l * (R + r)
    base_area1 = PI * R**2
    base_area2 = PI * r**2
    tsa = csa + base_area1 + base_area2
    return {
        "curved_surface_area": csa,
        "total_surface_area": tsa,
        "area_base1": base_area1,
        "area_base2": base_area2,
        "calculated_slant_height": l,
    }


def frustum_volume(height: float, radius1: float, radius2: float) -> float:
    if height <= 0 or radius1 < 0 or radius2 < 0:
        raise ValueError("Height must be positive and radii non-negative.")
    R, r_small = max(radius1, radius2), min(
        radius1, radius2
    )  # Ensure R is larger or equal
    return (1 / 3) * PI * height * (R**2 + r_small**2 + R * r_small)
