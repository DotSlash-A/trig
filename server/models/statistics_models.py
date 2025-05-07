# models/statistics_models.py
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Union, Optional

class UngroupedDataInput(BaseModel):
    data: List[Union[float, int, str]] # Mode can work with strings too
    
    @validator('data')
    def check_data_not_empty(cls, v):
        if not v:
            raise ValueError("Data list cannot be empty")
        return v
    
    @validator('data')
    def ensure_numeric_for_mean_median(cls, v, values):
        # This validation might be better placed in the specific route if mode allows strings
        # For now, let's assume if they call mean/median, data should be numeric
        # This is a basic check; specific routes should handle type errors from services.
        # if any(isinstance(x, str) for x in v):
        #    raise ValueError("Data for mean/median must be numeric.")
        return v


class UngroupedMeanResponse(BaseModel):
    data: List[Union[float, int, str]]
    mean: float

class UngroupedMedianResponse(BaseModel):
    data: List[Union[float, int, str]]
    sorted_data: List[Union[float, int, str]]
    median: float

class UngroupedModeResponse(BaseModel):
    data: List[Union[float, int, str]]
    modes: List[Union[float, int, str]] # Can be multiple modes, or "No mode"
    frequency_counts: Dict[Union[float, int, str], int]

# --- Grouped Data ---
class GroupedDataItemInput(BaseModel):
    # Allow either class_interval string OR lower/upper limits
    class_interval: Optional[str] = Field(None, example="10-20", description="Class interval as 'lower-upper'.")
    lower_limit: Optional[float] = Field(None, example=10.0)
    upper_limit: Optional[float] = Field(None, example=20.0)
    frequency: int = Field(..., ge=0)

    # Pydantic v2 style model validator
    # from pydantic import model_validator
    # @model_validator(mode='before')
    # def check_interval_or_limits(cls, values):
    #     has_interval = values.get('class_interval') is not None
    #     has_limits = values.get('lower_limit') is not None and values.get('upper_limit') is not None
    #     if not (has_interval ^ has_limits): # XOR: one must be true, not both, not neither
    #         if not has_interval and not has_limits:
    #             raise ValueError("Either 'class_interval' or both 'lower_limit' and 'upper_limit' must be provided.")
    #         if has_interval and has_limits:
    #             raise ValueError("Provide 'class_interval' OR 'lower_limit'/'upper_limit', not both.")
    #     if has_interval: # Clear limits if interval is primary
    #         values.pop('lower_limit', None)
    #         values.pop('upper_limit', None)
    #     return values
    class Config:
        validate_assignment = True # For Pydantic V1, less straightforward for XOR

class GroupedDataInput(BaseModel):
    data: List[GroupedDataItemInput]
    # For step-deviation and assumed mean, these can be optional
    assumed_mean_a: Optional[float] = Field(None, description="Optional assumed mean 'a' for mean calculation.")
    class_height_h: Optional[float] = Field(None, description="Optional class height 'h' for step-deviation mean (if uniform).")

class GroupedDataProcessedTableResponse(BaseModel):
    # This could be part of each specific measure's response
    input_data: List[Dict] # The original input items
    processed_table: List[Dict[str, Any]] # includes class_mark, cf, etc.

class GroupedMeanResponse(BaseModel):
    mean: float
    calculation_details: Dict[str, Any] # Includes method, steps table, sums
    full_data_summary_table: List[Dict[str, Any]] # The table with cf, xi etc.

class GroupedMedianResponse(BaseModel):
    median: float
    calculation_details: Dict[str, Any] # Includes N/2, l, f, cf, h etc.
    full_data_summary_table: List[Dict[str, Any]]

class GroupedModeResponse(BaseModel):
    mode: float
    calculation_details: Dict[str, Any] # Includes l, f1, f0, f2, h etc.
    full_data_summary_table: List[Dict[str, Any]]

class OgiveDataPoint(BaseModel):
    x_limit_value: float # Represents upper limit for less-than, lower limit for more-than
    y_cumulative_frequency: float

class OgiveDataResponse(BaseModel):
    less_than_ogive_points: List[OgiveDataPoint] # x is Upper Limit, y is CF
    more_than_ogive_points: List[OgiveDataPoint] # x is Lower Limit, y is N-CF_prev or sum from here
    full_data_summary_table: List[Dict[str, Any]]