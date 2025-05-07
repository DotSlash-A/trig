# services/probability_services.py
import math
from fractions import Fraction  # For exact fractional probability
from typing import List, Union, Dict, Any, Tuple, Optional
from itertools import product  # For sample spaces like dice rolls, coin tosses


def calculate_basic_probability(
    favorable_outcomes: int, total_outcomes: int
) -> Dict[str, Union[str, float]]:
    if total_outcomes <= 0:
        raise ValueError("Total number of outcomes must be positive.")
    if favorable_outcomes < 0 or favorable_outcomes > total_outcomes:
        raise ValueError(
            "Number of favorable outcomes must be between 0 and total outcomes (inclusive)."
        )

    if total_outcomes == 0:  # Should be caught by above, but defensive
        prob_decimal = 0.0
        prob_fraction = "0/1"  # Or undefined
    else:
        prob_decimal = favorable_outcomes / total_outcomes
        # Simplify fraction
        common_divisor = math.gcd(favorable_outcomes, total_outcomes)
        num = favorable_outcomes // common_divisor
        den = total_outcomes // common_divisor
        prob_fraction = f"{num}/{den}"

    return {
        "favorable_outcomes": favorable_outcomes,
        "total_outcomes": total_outcomes,
        "probability_decimal": prob_decimal,
        "probability_fraction": prob_fraction,
        "probability_percentage": prob_decimal * 100,
    }


def probability_of_complement(prob_event_E_decimal: float) -> Dict[str, float]:
    if not (0 <= prob_event_E_decimal <= 1):
        raise ValueError("Probability of event E must be between 0 and 1.")
    prob_not_E = 1 - prob_event_E_decimal
    return {"P(E)": prob_event_E_decimal, "P(not E)": prob_not_E}


# --- Coin Toss Scenarios ---
def coin_toss_sample_space(num_tosses: int) -> List[str]:
    if num_tosses <= 0:
        raise ValueError("Number of tosses must be positive.")
    if num_tosses > 10:  # Limit to prevent huge sample spaces
        raise ValueError(
            "Number of tosses too large for explicit sample space generation (max 10)."
        )

    outcomes = ["H", "T"]
    # Generate all combinations for num_tosses
    space = ["".join(p) for p in product(outcomes, repeat=num_tosses)]
    return space


def probability_coin_tosses(
    num_tosses: int,
    num_heads: Optional[int] = None,
    num_tails: Optional[int] = None,
    exact_sequence: Optional[str] = None,
) -> Dict[str, Any]:
    if num_tosses <= 0:
        raise ValueError("Number of tosses must be positive.")

    sample_space = coin_toss_sample_space(
        num_tosses
    )  # Can raise error if too many tosses
    total_outcomes = len(sample_space)  # Should be 2**num_tosses

    favorable_outcomes_count = 0
    favorable_event_description = ""

    if exact_sequence:
        if len(exact_sequence) != num_tosses or not all(
            c in "HT" for c in exact_sequence
        ):
            raise ValueError(
                f"Invalid exact sequence '{exact_sequence}' for {num_tosses} tosses."
            )
        favorable_event_description = f"getting the exact sequence '{exact_sequence}'"
        if exact_sequence in sample_space:  # It will be if valid
            favorable_outcomes_count = 1
    elif num_heads is not None:
        if not (0 <= num_heads <= num_tosses):
            raise ValueError("Number of heads must be between 0 and number of tosses.")
        favorable_event_description = f"getting exactly {num_heads} heads"
        for outcome in sample_space:
            if outcome.count("H") == num_heads:
                favorable_outcomes_count += 1
    elif num_tails is not None:
        if not (0 <= num_tails <= num_tosses):
            raise ValueError("Number of tails must be between 0 and number of tosses.")
        favorable_event_description = f"getting exactly {num_tails} tails"
        for outcome in sample_space:
            if outcome.count("T") == num_tails:
                favorable_outcomes_count += 1
    else:
        raise ValueError(
            "Specify a condition: num_heads, num_tails, or exact_sequence."
        )

    prob_details = calculate_basic_probability(favorable_outcomes_count, total_outcomes)

    return {
        "scenario": "Coin Tosses",
        "num_tosses": num_tosses,
        "event_description": favorable_event_description,
        **prob_details,  # Unpack probability results
    }


# --- Dice Roll Scenarios ---
def dice_roll_sample_space(num_dice: int) -> List[Tuple[int, ...]]:
    if num_dice <= 0:
        raise ValueError("Number of dice must be positive.")
    if num_dice > 4:  # 6^5 = 7776, getting large
        raise ValueError("Number of dice too large for explicit sample space (max 4).")

    outcomes = list(range(1, 7))
    space = list(product(outcomes, repeat=num_dice))
    return space


def probability_dice_rolls_sum(num_dice: int, target_sum: int) -> Dict[str, Any]:
    if num_dice <= 0:
        raise ValueError("Number of dice must be positive.")
    if not (num_dice <= target_sum <= num_dice * 6):
        raise ValueError(f"Target sum {target_sum} is impossible for {num_dice} dice.")

    sample_space = dice_roll_sample_space(num_dice)
    total_outcomes = len(sample_space)  # Should be 6**num_dice

    favorable_outcomes_count = 0
    for roll_outcome in sample_space:
        if sum(roll_outcome) == target_sum:
            favorable_outcomes_count += 1

    prob_details = calculate_basic_probability(favorable_outcomes_count, total_outcomes)

    return {
        "scenario": "Dice Rolls Sum",
        "num_dice": num_dice,
        "target_sum": target_sum,
        "event_description": f"getting a sum of {target_sum}",
        **prob_details,
    }


# --- Playing Cards Scenarios (Standard 52-card deck) ---
TOTAL_CARDS = 52
SUITS = ["Hearts", "Diamonds", "Clubs", "Spades"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
FACE_CARDS = ["Jack", "Queen", "King"]
RED_SUITS = ["Hearts", "Diamonds"]
BLACK_SUITS = ["Clubs", "Spades"]


def probability_drawing_card(
    rank: Optional[str] = None,
    suit: Optional[str] = None,
    color: Optional[str] = None,  # "Red" or "Black"
    is_face_card: Optional[bool] = None,
    is_ace: Optional[bool] = None,
    is_number_card: Optional[bool] = None,  # 2-10
    event_description_override: Optional[str] = None,  # For complex combined events
) -> Dict[str, Any]:

    favorable_outcomes_count = 0
    event_description_parts = []

    # This logic can get complex for combined conditions like "Red King".
    # For Class 10, problems are usually simpler, like "a King" or "a Spade" or "a Red card".
    # Let's handle simple cases. For combined, the user might need to calculate favorable outcomes manually.

    # If an override description is given, assume favorable_outcomes must also be provided by user or calculated externally.
    # For now, this function handles simple, non-overlapping OR conditions if multiple simple flags are true,
    # or specific combined conditions. A truly general function would build the set of favorable cards.

    if (
        event_description_override and "favorable_outcomes_count" in locals()
    ):  # Requires more input
        pass  # Handled by caller

    favorable_cards_set = set()  # To handle overlaps correctly (e.g. "King of Spades")

    # Full deck representation
    full_deck = [(r, s) for r in RANKS for s in SUITS]

    for card_rank, card_suit in full_deck:
        match = True  # Assume it matches until a condition fails
        current_card_conditions = []

        card_color = "Red" if card_suit in RED_SUITS else "Black"

        if rank and card_rank != rank:
            match = False
        if suit and card_suit != suit:
            match = False
        if color and card_color != color:
            match = False
        if is_face_card is not None:
            if is_face_card != (card_rank in FACE_CARDS):
                match = False
        if is_ace is not None:
            if is_ace != (card_rank == "Ace"):
                match = False
        if is_number_card is not None:  # 2-10 (not face, not Ace)
            if is_number_card != (card_rank not in FACE_CARDS and card_rank != "Ace"):
                match = False

        if match:
            favorable_cards_set.add((card_rank, card_suit))

    favorable_outcomes_count = len(favorable_cards_set)

    # Construct description based on what was set and led to favorable outcomes
    if rank:
        event_description_parts.append(f"rank is {rank}")
    if suit:
        event_description_parts.append(f"suit is {suit}")
    if color:
        event_description_parts.append(f"color is {color}")
    if is_face_card is True:
        event_description_parts.append("is a face card")
    if is_face_card is False:
        event_description_parts.append("is NOT a face card")
    if is_ace is True:
        event_description_parts.append("is an Ace")
    if is_ace is False:
        event_description_parts.append("is NOT an Ace")
    if is_number_card is True:
        event_description_parts.append("is a number card (2-10)")
    if is_number_card is False:
        event_description_parts.append("is NOT a number card (2-10)")

    if not event_description_parts:  # e.g. no conditions, means any card
        final_event_description = "drawing any card"
        favorable_outcomes_count = (
            TOTAL_CARDS  # Or handle as error "no event specified"
        )
    else:
        final_event_description = "drawing a card where " + " AND ".join(
            event_description_parts
        )

    prob_details = calculate_basic_probability(favorable_outcomes_count, TOTAL_CARDS)

    return {
        "scenario": "Drawing a Card (Standard 52-card deck)",
        "event_description": final_event_description,
        "favorable_cards_example_count": favorable_outcomes_count,  # Rename from prob_details's key
        "total_cards_in_deck": TOTAL_CARDS,
        "probability_decimal": prob_details["probability_decimal"],
        "probability_fraction": prob_details["probability_fraction"],
        "probability_percentage": prob_details["probability_percentage"],
    }
