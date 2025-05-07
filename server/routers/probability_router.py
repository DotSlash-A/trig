# routers/probability_router.py
from fastapi import APIRouter, Body, HTTPException
from models import probability_models as models
from services import probability_services as prob_service
from typing import Dict, Any

router = APIRouter(
    prefix="/probability/class10",
    tags=["Probability (Class 10)"]
)

@router.post("/basic", response_model=models.BasicProbabilityResponse)
async def get_basic_probability(data_input: models.BasicProbabilityInput = Body(...)):
    try:
        result = prob_service.calculate_basic_probability(
            data_input.favorable_outcomes,
            data_input.total_outcomes
        )
        # The service result dict matches BasicProbabilityResponse directly
        return models.BasicProbabilityResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/complementary-event", response_model=models.ComplementaryEventResponse)
async def get_complementary_event_prob(data_input: models.ComplementaryEventInput = Body(...)):
    try:
        result = prob_service.probability_of_complement(data_input.probability_event_E)
        return models.ComplementaryEventResponse(**{"P(E)": result["P(E)"], "P(not E)": result["P(not E)"]})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/coin-tosses", response_model=models.ProbabilityScenarioResponse)
async def get_coin_toss_probability(data_input: models.CoinTossInput = Body(...)):
    # Pydantic v2 model validator would handle this, for v1, check in router
    if sum(x is not None for x in [data_input.num_heads, data_input.num_tails, data_input.exact_sequence]) != 1:
        raise HTTPException(status_code=400, detail="Specify exactly one condition: num_heads, num_tails, OR exact_sequence.")

    try:
        result = prob_service.probability_coin_tosses(
            num_tosses=data_input.num_tosses,
            num_heads=data_input.num_heads,
            num_tails=data_input.num_tails,
            exact_sequence=data_input.exact_sequence
        )
        # Manually map to ProbabilityScenarioResponse, ensuring all fields are covered
        return models.ProbabilityScenarioResponse(
            scenario=result["scenario"],
            num_tosses=result["num_tosses"],
            event_description=result["event_description"],
            favorable_outcomes=result["favorable_outcomes"],
            total_outcomes=result["total_outcomes"],
            probability_decimal=result["probability_decimal"],
            probability_fraction=result["probability_fraction"],
            probability_percentage=result["probability_percentage"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/dice-rolls/sum", response_model=models.ProbabilityScenarioResponse)
async def get_dice_roll_sum_probability(data_input: models.DiceRollInput = Body(...)):
    try:
        result = prob_service.probability_dice_rolls_sum(
            num_dice=data_input.num_dice,
            target_sum=data_input.target_sum
        )
        return models.ProbabilityScenarioResponse(
            scenario=result["scenario"],
            num_dice=result["num_dice"],
            target_sum=result["target_sum"],
            event_description=result["event_description"],
            favorable_outcomes=result["favorable_outcomes"],
            total_outcomes=result["total_outcomes"],
            probability_decimal=result["probability_decimal"],
            probability_fraction=result["probability_fraction"],
            probability_percentage=result["probability_percentage"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/playing-cards/draw", response_model=models.ProbabilityScenarioResponse)
async def get_card_drawing_probability(data_input: models.CardDrawingInput = Body(...)):
    # Ensure at least one condition is specified
    if all(v is None for v in [data_input.rank, data_input.suit, data_input.color, data_input.is_face_card, data_input.is_ace, data_input.is_number_card]):
        raise HTTPException(status_code=400, detail="At least one card characteristic must be specified.")
    
    try:
        result = prob_service.probability_drawing_card(
            rank=data_input.rank,
            suit=data_input.suit,
            color=data_input.color,
            is_face_card=data_input.is_face_card,
            is_ace=data_input.is_ace,
            is_number_card=data_input.is_number_card
        )
        return models.ProbabilityScenarioResponse(
            scenario=result["scenario"],
            event_description=result["event_description"],
            favorable_cards_example_count=result["favorable_cards_example_count"], # Specific key
            total_cards_in_deck=result["total_cards_in_deck"], # Specific key
            favorable_outcomes=result["favorable_cards_example_count"], # Map to general key
            total_outcomes=result["total_cards_in_deck"], # Map to general key
            probability_decimal=result["probability_decimal"],
            probability_fraction=result["probability_fraction"],
            probability_percentage=result["probability_percentage"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))