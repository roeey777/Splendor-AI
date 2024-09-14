"""
Features extraction from SplendorState

BEWARE: THIS IS UNTESTED!!!!
"""

from dataclasses import dataclass
from itertools import chain, repeat
from numbers import Number
from typing import Dict, Iterable, List, Literal, Optional, ValuesView

import numpy as np

from Engine.Splendor.splendor_model import Card, SplendorState
from Engine.Splendor.splendor_utils import COLOURS


@dataclass
class Noble:
    code: str
    cost: Dict[str, int]


Color = Literal[*COLOURS.values()]
WILDCARD = "yellow"
RESERVED = WILDCARD
NORMAL_COLORS = list(color for color in COLOURS.values() if color != WILDCARD)
MAX_TIER_CARDS = 4
MAX_NOBLES = 5
MAX_RESERVED = 3
MAX_WILDCARDS = 5
MAX_GEMS = 10
WINNING_SCORE_TRESHOLD = 15
MAX_SCORE = 22
MAX_RIVALS = 3
ROUNDS_LIMIT = 100
MAX_STONES = (4, 5, 7)
MAX_CARD_GEMS_DIST_1 = 5
MAX_CARD_GEMS_DIST_2 = 8
MAX_CARD_GEMS_DIST_3 = 14
MAX_CARD_TURNS_DIST_1 = 4
MAX_CARD_TURNS_DIST_2 = 6
MAX_CARD_TURNS_DIST_3 = 7
MAX_NOBLE_CARDS_DISTANCE = 9
MAX_NOBLE_GEMS_DISTANCE = MAX_CARD_GEMS_DIST_3 * MAX_NOBLE_CARDS_DISTANCE
# guesses
MAX_CARDS_PER_COLOR = 8 # normal value is 5
MAX_TOTAL_CARDS = MAX_CARDS_PER_COLOR * 3 # normal value is 20
MAX_VARIANCE = 20
MAX_BUYING_POWER = 10


### RANDOM VALUES (change?) ###
MAX_NOBLE_DISTANCE = 100
MISSING_CARD_GEMS_DEFAULT = 0
MISSING_CARD_TURNS_DEFAULT = 0
MISSING_NOBLE_GEMS_DISTANCE = 0
MISSING_NOBLE_CARDS_DISTANCE = 0
MISSING_RIVAL_DEFAULT = 0


# ********************************
# Utility Functions:
# ********************************
def get_agent(game_state: SplendorState, agent_index: int) -> SplendorState.AgentState:
    """
    Extract the AgentState of a specific agent from the game state.
    """
    if agent_index not in range(len(game_state.agents)):
        raise ValueError("agent index out of range")

    return game_state.agents[agent_index]


def agent_buying_power(agent: SplendorState.AgentState) -> Dict[Color, int]:
    return {
        color: agent.gems[color] + len(agent.cards[color])
        for color in NORMAL_COLORS
    }


def diminish_return(value: Number) -> float:
    return np.log(1 + value)


def agent_won(agent: SplendorState.AgentState) -> bool:
    return agent.score >= WINNING_SCORE_TRESHOLD


def missing_card_to_noble(
    game_state: SplendorState, agent_index: int, noble: Noble
) -> Dict[Color, int]:
    """
    Find which permanent gems (cards) are required by an agent in order to be
    visited by a specific noble.
    """
    cards = permanent_buying_power(game_state, agent_index)
    missing_cards: Dict[Color, int] = {}

    for color, cost in noble.cost.items():
        missing_cards[color] = max(0, cost - cards[color])

    return missing_cards


def turns_made_by_agent(agent: SplendorState.AgentState) -> int:
    """
    Extract the number of turns made by a given agent.
    """
    return len(agent.agent_trace.action_reward)


def missing_gems_to_card(
    card: Card, buying_power: Dict[Color, int]
) -> [Dict[Color, int]]:
    return {
        color: cost - buying_power[color]
        for color, cost in card.cost.items()
        if cost - buying_power[color] > 0
    }


def turns_to_buy_card(missing_gems: ValuesView[int]) -> int:
    if not missing_gems:
        return 0
    return max(np.ceil(sum(missing_gems) / 3), *missing_gems)


def build_array(base_array: np.array, instruction: tuple[int]) -> np.array:
    building_blocks = zip(base_array, instruction, strict=True)
    iters = (repeat(value, times) for value, times in building_blocks)
    match len(base_array.shape):
        case 1:
            return np.hstack(tuple(chain(*iters)), dtype=base_array.dtype)
        case 2:
            return np.vstack(tuple(chain(*iters)), dtype=base_array.dtype)
        case _:
            raise ValueError(f"unsupported array shape '{base_array.shape}'")


# ********************************
# Features Extraction Functions
# ********************************
METRICS_SHAPE : tuple[int] = (
    1,  # constant (hopefully would be used by the manager)
    1,  # agent's turns count
    1,  # agent's score
    1,  # agent has crossed 15 points
    1,  # agent's owned cards count
    1,  # agent's reserved cards count
    1,  # variance of buying power
    1,  # agent's golden gems count
    1,  # agent's total gems count
    len(NORMAL_COLORS),  # agent's buying power (without wild gems)
    len(NORMAL_COLORS),  # agent's diminishing buying power (without wild gems)
    MAX_RIVALS,  # scores of rivals that play before agent
    MAX_RIVALS,  # scores of rivals that play after agent
    MAX_TIER_CARDS,  # distances to cards on tier 1 in gems
    MAX_TIER_CARDS,  # distances to cards on tier 2 in gems
    MAX_TIER_CARDS,  # distances to cards on tier 3 in gems
    MAX_TIER_CARDS,  # distances to cards on tier 1 in turns
    MAX_TIER_CARDS,  # distances to cards on tier 2 in turns
    MAX_TIER_CARDS,  # distances to cards on tier 3 in turns
    MAX_RESERVED,  # distances to reserved cards in gems
    MAX_RESERVED,  # distances to reserved cards in turns
    MAX_NOBLES,  # distances to nobles in gems
    MAX_NOBLES,  # distances to nobles in cards
)

METRIC_NORMALIZATION = np.array(
    [
        1,             # constant (hopefully would be used by the manager)
        ROUNDS_LIMIT,  # agent's turns count
        MAX_SCORE,     # agent's score
        1,             # agent has crossed 15 points
        MAX_TOTAL_CARDS,  # agent's owned cards count
        MAX_RESERVED,   # agent's reserved cards count
        MAX_VARIANCE,   # variance of buying power
        MAX_WILDCARDS,  # agent's golden gems count
        MAX_GEMS,       # agent's total gems count
        MAX_BUYING_POWER,  # agent's buying power (without wild gems)
        diminish_return(MAX_BUYING_POWER),  # agent's diminishing buying power (without wild gems)
        MAX_SCORE,  # scores of rivals that play before agent
        WINNING_SCORE_TRESHOLD - 1,  # scores of rivals that play after agent
        MAX_CARD_GEMS_DIST_1,   # distances to cards on tier 1 in gems
        MAX_CARD_GEMS_DIST_2,   # distances to cards on tier 2 in gems
        MAX_CARD_GEMS_DIST_3,   # distances to cards on tier 3 in gems
        MAX_CARD_TURNS_DIST_1,  # distances to cards on tier 1 in turns
        MAX_CARD_TURNS_DIST_2,  # distances to cards on tier 2 in turns
        MAX_CARD_TURNS_DIST_3,  # distances to cards on tier 3 in turns
        MAX_CARD_GEMS_DIST_3,   # distances to reserved cards in gems
        MAX_CARD_TURNS_DIST_3,  # distances to reserved cards in turns
        MAX_NOBLE_GEMS_DISTANCE,  # distances to nobles in gems
        MAX_NOBLE_CARDS_DISTANCE,  # distances to nobles in cards
    ],
    dtype=float,
)


def extract_metrics(game_state: SplendorState, agent_index: int) -> np.array:
    agent = get_agent(game_state, agent_index)
    buying_power = agent_buying_power(agent)
    owned_cards = sum(len(agent.cards[color]) for color in NORMAL_COLORS)

    card_distance_per_color = {color: list() for color in NORMAL_COLORS}

    cards_distances_in_gems: list[int] = list()
    cards_distances_in_turns: list[int] = list()
    for card in chain(*game_state.board.dealt):
        if card is None:
            cards_distances_in_gems.append(MISSING_CARD_GEMS_DEFAULT)
            cards_distances_in_turns.append(MISSING_CARD_TURNS_DEFAULT)

        else:
            card_missing_gems = missing_gems_to_card(card, buying_power)
            distance_in_gems = sum(card_missing_gems.values())
            cards_distances_in_gems.append(distance_in_gems)
            cards_distances_in_turns.append(
                turns_to_buy_card(card_missing_gems.values())
            )
            card_distance_per_color[card.colour].append(distance_in_gems)

    reserved_distances_in_gems = [MISSING_CARD_GEMS_DEFAULT] * MAX_RESERVED
    reserved_distances_in_turns = [MISSING_CARD_TURNS_DEFAULT] * MAX_RESERVED
    for i, card in enumerate(agent.cards[RESERVED]):
        card_missing_gems = missing_gems_to_card(card, buying_power)
        distance_in_gems = sum(card_missing_gems.values())
        reserved_distances_in_gems[i] = distance_in_gems
        reserved_distances_in_turns[i] = turns_to_buy_card(card_missing_gems.values())
        card_distance_per_color[card.colour].append(distance_in_gems)

    for distances in card_distance_per_color.values():
        distances.sort()

    nobles_distances_in_cards = [MISSING_NOBLE_CARDS_DISTANCE] * MAX_NOBLES
    nobles_distances_in_gems = [MISSING_NOBLE_GEMS_DISTANCE] * MAX_NOBLES
    for i, (_, noble_cost) in enumerate(game_state.board.nobles):
        missing_cards = {
            color: cost - len(agent.cards[color])
            for color, cost in noble_cost.items()
            if cost - len(agent.cards[color]) > 0
        }
        nobles_distances_in_cards[i] = sum(missing_cards.values()) / 9
        distance_in_gems = 0
        for color, count in missing_cards.items():
            distance_in_gems += sum(card_distance_per_color[color][:count])
            more_cards = count - len(card_distance_per_color[color])
            distance_in_gems += MAX_CARD_GEMS_DIST_3 * max(more_cards, 0)
        nobles_distances_in_gems[i] = distance_in_gems

    rivals_scores_1 = [MISSING_RIVAL_DEFAULT] * MAX_RIVALS
    rivals_scores_2 = [MISSING_RIVAL_DEFAULT] * MAX_RIVALS
    for i, rival in enumerate(game_state.agents):
        if i < agent_index:
            rivals_scores_1[i] = rival.score
        elif i > agent_index:
            rivals_scores_2[i - 1] = rival.score

    capped_buying_power = tuple(min(v, MAX_BUYING_POWER)
                                for v in buying_power.values())

    return np.fromiter(
        chain(
            [
                1,
                turns_made_by_agent(agent),
                agent.score,
                1 if agent_won(agent) else 0,
                min(owned_cards, MAX_TOTAL_CARDS),
                len(agent.cards[RESERVED]),
                min(np.var(list(buying_power.values())), MAX_VARIANCE),
                agent.gems[WILDCARD],
                sum(agent.gems.values()),
            ],
            capped_buying_power,
            map(diminish_return, capped_buying_power),
            rivals_scores_1,
            rivals_scores_2,
            cards_distances_in_gems,
            cards_distances_in_turns,
            reserved_distances_in_gems,
            reserved_distances_in_turns,
            nobles_distances_in_gems,
            nobles_distances_in_cards,
        ),
        float,
    )


def normalize_metrics(metrics: np.array) -> np.array:
    normalizer = build_array(METRIC_NORMALIZATION, METRICS_SHAPE)
    return metrics / normalizer
