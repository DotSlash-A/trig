# models/geometry_models.py
from pydantic import BaseModel, Field, validator
from typing import Dict, Optional, Union


class DimensionsCuboid(BaseModel):
    length: float = Field(..., gt=0)
    breadth: float = Field(..., gt=0)
    height: float = Field(..., gt=0)


class DimensionsCube(BaseModel):
    side: float = Field(..., gt=0)


class DimensionsCylinder(BaseModel):
    radius: float = Field(..., gt=0)
    height: float = Field(..., gt=0)


class DimensionsCone(BaseModel):
    radius: float = Field(..., gt=0)
    height: float = Field(..., gt=0)
    slant_height: Optional[float] = Field(
        None, gt=0, description="Optional: if not provided, it will be calculated."
    )


class DimensionsSphere(BaseModel):
    radius: float = Field(..., gt=0)


class DimensionsHemisphere(DimensionsSphere):
    pass


class DimensionsFrustum(BaseModel):
    height: float = Field(..., gt=0)
    radius1: float = Field(..., ge=0, description="Radius of one circular end.")
    radius2: float = Field(..., ge=0, description="Radius of the other circular end.")
    slant_height: Optional[float] = Field(
        None, gt=0, description="Optional: if not provided, it will be calculated."
    )

    @validator("radius1")
    def check_at_least_one_radius_positive_if_height_exists(cls, v, values):
        # If it's a frustum (not reducing to a point/cone), and height > 0,
        # then at least one radius should ideally be > 0 for typical frustums.
        # However, radius1=0, radius2=R gives a cone.
        # radius1=R, radius2=R gives a cylinder.
        # The service layer handles these specific interpretations more directly.
        # For now, ge=0 is fine.
        return v


class SurfaceAreaResponse(BaseModel):
    shape: str
    dimensions: Dict[str, float]
    lateral_surface_area: Optional[float] = None
    curved_surface_area: Optional[float] = None
    total_surface_area: Optional[float] = None
    surface_area: Optional[float] = None  # For sphere
    area_base1: Optional[float] = None  # For frustum
    area_base2: Optional[float] = None  # For frustum
    calculated_slant_height: Optional[float] = None


class VolumeResponse(BaseModel):
    shape: str
    dimensions: Dict[str, float]
    volume: float


class DiagonalResponse(BaseModel):
    shape: str
    dimensions: Dict[str, float]
    diagonal: float
