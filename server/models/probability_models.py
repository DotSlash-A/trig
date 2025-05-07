# models/probability_models.py
from pydantic import BaseModel, Field, validator
from typing import List, Union, Dict, Any, Optional

class BasicProbabilityInput(BaseModel):
    favorable_outcomes: int = Field(..., ge=0)
    total_outcomes: int = Field(..., gt=0)

class BasicProbabilityResponse(BaseModel):
    favorable_outcomes: int
    total_outcomes: int
    probability_decimal: float
    probability_fraction: str
    probability_percentage: float

class ComplementaryEventInput(BaseModel):
    probability_event_E: float = Field(..., ge=0, le=1)

class ComplementaryEventResponse(BaseModel):
    P_E: float = Field(..., alias="P(E)")
    P_not_E: float = Field(..., alias="P(not E)")
    
    class Config:
        populate_by_name = True # Allows using alias in response

class CoinTossInput(BaseModel):
    num_tosses: int = Field(..., gt=0, le=10) # Limit for practical sample space
    num_heads: Optional[int] = Field(None, ge=0)
    num_tails: Optional[int] = Field(None, ge=0)
    exact_sequence: Optional[str] = Field(None, min_length=1, pattern=r"^[HT]+$")

    @validator('num_heads')
    def check_num_heads(cls, v, values):
        if v is not None and values.get('num_tosses') is not None and v > values['num_tosses']:
            raise ValueError("Number of heads cannot exceed number of tosses.")
        return v

    @validator('num_tails')
    def check_num_tails(cls, v, values):
        if v is not None and values.get('num_tosses') is not None and v > values['num_tosses']:
            raise ValueError("Number of tails cannot exceed number of tosses.")
        return v
    
    @validator('exact_sequence')
    def check_sequence_length(cls, v, values):
        if v is not None and values.get('num_tosses') is not None and len(v) != values['num_tosses']:
            raise ValueError("Length of exact_sequence must match num_tosses.")
        return v
    
    # Add model validator to ensure at least one condition is set
    # from pydantic import model_validator
    # @model_validator(mode='after') # Pydantic v2
    # def check_one_condition(self) -> 'CoinTossInput':
    #     if self.num_heads is None and self.num_tails is None and self.exact_sequence is None:
    #         raise ValueError("Must specify one condition: num_heads, num_tails, or exact_sequence.")
    #     if sum(x is not None for x in [self.num_heads, self.num_tails, self.exact_sequence]) > 1:
    #         raise ValueError("Specify only one condition: num_heads, num_tails, OR exact_sequence.")
    #     return self


class DiceRollInput(BaseModel):
    num_dice: int = Field(..., gt=0, le=4) # Limit for practical sample space
    target_sum: int = Field(..., gt=0)

    @validator('target_sum')
    def check_target_sum(cls, v, values):
        if 'num_dice' in values: # num_dice should already be validated
            min_sum = values['num_dice']
            max_sum = values['num_dice'] * 6
            if not (min_sum <= v <= max_sum):
                raise ValueError(f"Target sum must be between {min_sum} and {max_sum} for {values['num_dice']} dice.")
        return v

class CardDrawingInput(BaseModel):
    rank: Optional[str] = Field(None, examples=["Ace", "King", "7"])
    suit: Optional[str] = Field(None, examples=["Hearts", "Spades"])
    color: Optional[str] = Field(None, pattern=r"^(Red|Black)$")
    is_face_card: Optional[bool] = None
    is_ace: Optional[bool] = None
    is_number_card: Optional[bool] = Field(None, description="Cards from 2 to 10.")
    # event_description_override: Optional[str] = None # For user-defined complex events - advanced

class ProbabilityScenarioResponse(BasicProbabilityResponse):
    scenario: str
    event_description: str
    num_tosses: Optional[int] = None
    num_dice: Optional[int] = None
    target_sum: Optional[int] = None
    favorable_cards_example_count: Optional[int] = None # Specific for cards
    total_cards_in_deck: Optional[int] = None # Specific for cards
    # Note: BasicProbabilityResponse keys (favorable_outcomes, total_outcomes) will be populated by the service.