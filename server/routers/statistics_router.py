# routers/statistics_router.py
from fastapi import APIRouter, Body, HTTPException
from models import statistics_models as models
from services import statistics_services as stat_service
from typing import List, Dict, Any, Tuple

router = APIRouter(
    prefix="/statistics/class10",
    tags=["Statistics (Class 10)"]
)

# --- Ungrouped Data Routes ---
@router.post("/ungrouped/mean", response_model=models.UngroupedMeanResponse)
async def get_ungrouped_mean(data_input: models.UngroupedDataInput = Body(...)):
    try:
        # Ensure data is numeric for mean
        numeric_data = [float(x) for x in data_input.data if not isinstance(x, str)] # crude filter
        if len(numeric_data) != len(data_input.data):
            raise HTTPException(status_code=400, detail="All data for mean must be numeric.")
        mean = stat_service.calculate_mean_ungrouped(numeric_data)
        return models.UngroupedMeanResponse(data=data_input.data, mean=mean)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post("/ungrouped/median", response_model=models.UngroupedMedianResponse)
async def get_ungrouped_median(data_input: models.UngroupedDataInput = Body(...)):
    try:
        numeric_data = [float(x) for x in data_input.data if not isinstance(x, str)]
        if len(numeric_data) != len(data_input.data):
            raise HTTPException(status_code=400, detail="All data for median must be numeric.")
        median = stat_service.calculate_median_ungrouped(numeric_data)
        return models.UngroupedMedianResponse(
            data=data_input.data,
            sorted_data=sorted(numeric_data),
            median=median
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/ungrouped/mode", response_model=models.UngroupedModeResponse)
async def get_ungrouped_mode(data_input: models.UngroupedDataInput = Body(...)):
    try:
        # Mode can handle mixed types (though typically numbers or categories)
        modes = stat_service.calculate_mode_ungrouped(data_input.data)
        counts = dict(stat_service.Counter(data_input.data))
        return models.UngroupedModeResponse(data=data_input.data, modes=modes, frequency_counts=counts)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Grouped Data Routes ---
def _process_grouped_data_input(raw_data: List[models.GroupedDataItemInput]) -> Tuple[List[stat_service.GroupedDataClass], List[Dict[str, Any]]]:
    """Helper to convert Pydantic input to service layer input and get summary table."""
    # Convert Pydantic models to simple dicts for the service layer's _prepare_grouped_data_table
    dict_data = [item.model_dump(exclude_none=True) for item in raw_data]
    try:
        data_classes, summary_table = stat_service._prepare_grouped_data_table(dict_data)
        return data_classes, summary_table
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Error processing grouped data: {str(e)}")


@router.post("/grouped/mean/direct", response_model=models.GroupedMeanResponse)
async def get_grouped_mean_direct(data_input: models.GroupedDataInput = Body(...)):
    data_classes, summary_table = _process_grouped_data_input(data_input.data)
    try:
        mean, calc_details = stat_service.calculate_mean_grouped_direct(data_classes)
        return models.GroupedMeanResponse(
            mean=mean,
            calculation_details=calc_details,
            full_data_summary_table=summary_table
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/grouped/mean/assumed", response_model=models.GroupedMeanResponse)
async def get_grouped_mean_assumed(data_input: models.GroupedDataInput = Body(...)):
    data_classes, summary_table = _process_grouped_data_input(data_input.data)
    try:
        mean, calc_details = stat_service.calculate_mean_grouped_assumed_mean(
            data_classes,
            assumed_mean_val=data_input.assumed_mean_a
        )
        return models.GroupedMeanResponse(
            mean=mean,
            calculation_details=calc_details,
            full_data_summary_table=summary_table
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/grouped/mean/step-deviation", response_model=models.GroupedMeanResponse)
async def get_grouped_mean_step_deviation(data_input: models.GroupedDataInput = Body(...)):
    data_classes, summary_table = _process_grouped_data_input(data_input.data)
    try:
        mean, calc_details = stat_service.calculate_mean_grouped_step_deviation(
            data_classes,
            assumed_mean_val=data_input.assumed_mean_a,
            class_height_h=data_input.class_height_h
        )
        return models.GroupedMeanResponse(
            mean=mean,
            calculation_details=calc_details,
            full_data_summary_table=summary_table
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/grouped/median", response_model=models.GroupedMedianResponse)
async def get_grouped_median(data_input: models.GroupedDataInput = Body(...)):
    data_classes, summary_table = _process_grouped_data_input(data_input.data)
    try:
        median, calc_details = stat_service.calculate_median_grouped(data_classes, summary_table)
        return models.GroupedMedianResponse(
            median=median,
            calculation_details=calc_details,
            full_data_summary_table=summary_table
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/grouped/mode", response_model=models.GroupedModeResponse)
async def get_grouped_mode(data_input: models.GroupedDataInput = Body(...)):
    data_classes, summary_table = _process_grouped_data_input(data_input.data)
    try:
        mode_val, calc_details = stat_service.calculate_mode_grouped(data_classes)
        return models.GroupedModeResponse(
            mode=mode_val,
            calculation_details=calc_details,
            full_data_summary_table=summary_table
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/grouped/ogive-data", response_model=models.OgiveDataResponse)
async def get_ogive_data_points(data_input: models.GroupedDataInput = Body(...)):
    data_classes, summary_table = _process_grouped_data_input(data_input.data)
    try:
        ogive_points_data = stat_service.generate_ogive_data(summary_table, data_classes)
        
        # Transform to match OgiveDataPoint model
        less_than_transformed = [
            models.OgiveDataPoint(x_limit_value=p["x_upper_limit"], y_cumulative_frequency=p["y_cumulative_frequency"])
            for p in ogive_points_data["less_than_ogive_points"]
        ]
        more_than_transformed = [
            models.OgiveDataPoint(x_limit_value=p["x_lower_limit"], y_cumulative_frequency=p["y_cumulative_frequency_from_here"])
            for p in ogive_points_data["more_than_ogive_points"]
        ]

        return models.OgiveDataResponse(
            less_than_ogive_points=less_than_transformed,
            more_than_ogive_points=more_than_transformed,
            full_data_summary_table=summary_table
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))