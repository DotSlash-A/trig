# routers/geometry_router.py
from fastapi import APIRouter, Body, HTTPException
from models import geometry_models as models
from services import geometry_services as geo_service
from typing import Dict

router = APIRouter(
    prefix="/geometry/class10",
    tags=["Surface Areas and Volumes (Class 10)"]
)

# --- Cuboid Routes ---
@router.post("/cuboid/surface-areas", response_model=models.SurfaceAreaResponse)
async def get_cuboid_sa(dims: models.DimensionsCuboid = Body(...)):
    try:
        sa_data = geo_service.cuboid_surface_areas(dims.length, dims.breadth, dims.height)
        return models.SurfaceAreaResponse(
            shape="Cuboid",
            dimensions=dims.model_dump(),
            lateral_surface_area=sa_data["lateral_surface_area"],
            total_surface_area=sa_data["total_surface_area"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/cuboid/volume", response_model=models.VolumeResponse)
async def get_cuboid_volume(dims: models.DimensionsCuboid = Body(...)):
    try:
        vol = geo_service.cuboid_volume(dims.length, dims.breadth, dims.height)
        return models.VolumeResponse(shape="Cuboid", dimensions=dims.model_dump(), volume=vol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/cuboid/diagonal", response_model=models.DiagonalResponse)
async def get_cuboid_diagonal(dims: models.DimensionsCuboid = Body(...)):
    try:
        diag = geo_service.cuboid_diagonal(dims.length, dims.breadth, dims.height)
        return models.DiagonalResponse(shape="Cuboid", dimensions=dims.model_dump(), diagonal=diag)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Cube Routes ---
@router.post("/cube/surface-areas", response_model=models.SurfaceAreaResponse)
async def get_cube_sa(dims: models.DimensionsCube = Body(...)):
    try:
        sa_data = geo_service.cube_surface_areas(dims.side)
        return models.SurfaceAreaResponse(
            shape="Cube",
            dimensions=dims.model_dump(),
            lateral_surface_area=sa_data["lateral_surface_area"],
            total_surface_area=sa_data["total_surface_area"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/cube/volume", response_model=models.VolumeResponse)
async def get_cube_volume(dims: models.DimensionsCube = Body(...)):
    try:
        vol = geo_service.cube_volume(dims.side)
        return models.VolumeResponse(shape="Cube", dimensions=dims.model_dump(), volume=vol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/cube/diagonal", response_model=models.DiagonalResponse)
async def get_cube_diagonal(dims: models.DimensionsCube = Body(...)):
    try:
        diag = geo_service.cube_diagonal(dims.side)
        return models.DiagonalResponse(shape="Cube", dimensions=dims.model_dump(), diagonal=diag)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Cylinder Routes ---
@router.post("/cylinder/surface-areas", response_model=models.SurfaceAreaResponse)
async def get_cylinder_sa(dims: models.DimensionsCylinder = Body(...)):
    try:
        sa_data = geo_service.cylinder_surface_areas(dims.radius, dims.height)
        return models.SurfaceAreaResponse(
            shape="Cylinder",
            dimensions=dims.model_dump(),
            curved_surface_area=sa_data["curved_surface_area"],
            total_surface_area=sa_data["total_surface_area"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/cylinder/volume", response_model=models.VolumeResponse)
async def get_cylinder_volume(dims: models.DimensionsCylinder = Body(...)):
    try:
        vol = geo_service.cylinder_volume(dims.radius, dims.height)
        return models.VolumeResponse(shape="Cylinder", dimensions=dims.model_dump(), volume=vol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Cone Routes ---
@router.post("/cone/surface-areas", response_model=models.SurfaceAreaResponse)
async def get_cone_sa(dims: models.DimensionsCone = Body(...)):
    try:
        sa_data = geo_service.cone_surface_areas(dims.radius, dims.height, dims.slant_height)
        return models.SurfaceAreaResponse(
            shape="Cone",
            dimensions=dims.model_dump(),
            curved_surface_area=sa_data["curved_surface_area"],
            total_surface_area=sa_data["total_surface_area"],
            calculated_slant_height=sa_data["calculated_slant_height"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/cone/volume", response_model=models.VolumeResponse)
async def get_cone_volume(dims: models.DimensionsCone = Body(...)): # Slant height not needed for volume
    try:
        vol = geo_service.cone_volume(dims.radius, dims.height)
        return models.VolumeResponse(shape="Cone", dimensions={"radius": dims.radius, "height": dims.height}, volume=vol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Sphere Routes ---
@router.post("/sphere/surface-area", response_model=models.SurfaceAreaResponse)
async def get_sphere_sa(dims: models.DimensionsSphere = Body(...)):
    try:
        sa = geo_service.sphere_surface_area(dims.radius)
        return models.SurfaceAreaResponse(
            shape="Sphere",
            dimensions=dims.model_dump(),
            surface_area=sa # Specific field for sphere
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/sphere/volume", response_model=models.VolumeResponse)
async def get_sphere_volume(dims: models.DimensionsSphere = Body(...)):
    try:
        vol = geo_service.sphere_volume(dims.radius)
        return models.VolumeResponse(shape="Sphere", dimensions=dims.model_dump(), volume=vol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Hemisphere Routes ---
@router.post("/hemisphere/surface-areas", response_model=models.SurfaceAreaResponse)
async def get_hemisphere_sa(dims: models.DimensionsHemisphere = Body(...)):
    try:
        sa_data = geo_service.hemisphere_surface_areas(dims.radius)
        return models.SurfaceAreaResponse(
            shape="Hemisphere",
            dimensions=dims.model_dump(),
            curved_surface_area=sa_data["curved_surface_area"],
            total_surface_area=sa_data["total_surface_area"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/hemisphere/volume", response_model=models.VolumeResponse)
async def get_hemisphere_volume(dims: models.DimensionsHemisphere = Body(...)):
    try:
        vol = geo_service.hemisphere_volume(dims.radius)
        return models.VolumeResponse(shape="Hemisphere", dimensions=dims.model_dump(), volume=vol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Frustum Routes ---
@router.post("/frustum/surface-areas", response_model=models.SurfaceAreaResponse)
async def get_frustum_sa(dims: models.DimensionsFrustum = Body(...)):
    try:
        sa_data = geo_service.frustum_surface_areas(dims.height, dims.radius1, dims.radius2, dims.slant_height)
        return models.SurfaceAreaResponse(
            shape="Frustum of a Cone",
            dimensions=dims.model_dump(),
            curved_surface_area=sa_data["curved_surface_area"],
            total_surface_area=sa_data["total_surface_area"],
            area_base1=sa_data["area_base1"],
            area_base2=sa_data["area_base2"],
            calculated_slant_height=sa_data["calculated_slant_height"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/frustum/volume", response_model=models.VolumeResponse)
async def get_frustum_volume(dims: models.DimensionsFrustum = Body(...)):
    try:
        # Ensure consistent naming for dimensions in response
        vol = geo_service.frustum_volume(dims.height, dims.radius1, dims.radius2)
        return models.VolumeResponse(shape="Frustum of a Cone", dimensions=dims.model_dump(), volume=vol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))