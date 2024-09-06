"""
Features extraction from SplendorState

BEWARE: THIS IS UNTESTED!!!!
"""

import numpy as np

from dataclasses import dataclass
from typing import List, Literal, Dict
from numbers import Number

from Engine.splendor_model import Card, SplendorState
from Engine.splendor_utils import COLOURS


@dataclass
class Noble:
    code: str
    cost: Dict[str, int]


Color = Literal[*COLOURS.values()]
WILDCARD = "yellow"
RESERVED = WILDCARD
MAX_NOBLE_DISTANCE = 100


# ********************************
# Utility Functions:
# ********************************
def get_agent(game_state: SplendorState, agent_index: int) -> SplendorState.AgentState:
    """
    Extract the AgentState of a specific agent from the game state.
    """
    if agent_index not in range(game_state.agents):
        raise ValueError("agent index out of range")

    return game_state.agents[agent_index]


def find_missing_gems(
    game_state: SplendorState, agent_index: int, card: Card
) -> Dict[Color, int]:
    """
    calculate which gems are required to be obtained by an agent in order
    to purchase a specific card.
    """
    agent = get_agent(game_state, agent_index)
    permanent_gems = permanent_buying_power(game_state, agent_index)

    for color, cost in card.cost.items():
        missing_gems[color] = max(0, cost - agent.gems[color] - permanent_gems[color])

    return missing_gems


def get_wildcard_gems(game_state: SplendorState, agent_index: int) -> int:
    """
    Find the amount of wildcard (yellow) gems a specific agent have in his
    disposel.
    """
    agent = get_agent(game_state, agent_index)
    return agent.gems[WILDCARD]


def diminish_return(value: Number) -> float:
    return np.log(1 + value)


def find_missing_permanent_gems(
    game_state: SplendorState, agent_index: int, noble: Noble
) -> Dict[Color, int]:
    """
    Find which permanent gems (cards) are required by an agent in order to be
    visited by a specific noble.
    """
    permanent_gems = permanent_buying_power(game_state, agent_index)
    missing_permanent_gems: Dict[Color, int] = {}

    for color, cost in noble.cost.items():
        missing_permanent_gems[color] = max(0, cost - permanent_gems[color])

    return missing_permanent_gems


# ********************************
# Features Extraction Functions
# ********************************
def turns_made_by_agent(game_state: SplendorState, agent_index: int) -> int:
    """
    Extract the number of turns made by a given agent.
    """
    return len(get_agent(game_state, agent_index).agent_trace.action_reward)


def score_of_agent(game_state: SplendorState, agent_index: int) -> int:
    """
    Extract the score (number of victory points) of a single player.
    """
    return get_agent(game_state, agent_index).score


def distance_in_gems_to_card(
    game_state: SplendorState, agent_index: int, card: Card
) -> int:
    """
    Calculate the distance (required amount of gems) of an agent from
    purchesing a given card.
    """
    return max(
        0,
        sum(find_missing_gems(game_state, agent_index, card).values())
        - get_wildcard_gems(game_state, agent_index),
    )


def distance_to_card(game_state: SplendorState, agent_index: int, card: Card) -> int:
    """
    Calculate the distance (minimal amount of turns) of an agent from
    purchesing a given card.
    """
    wildcard_gems = get_wildcard_gems(game_state, agent_index)
    missing_gems = find_missing_gems(game_state, agent_index, card)
    return max(0, max(missing_gems.values()) - wildcard_gems)


def buying_power_of_color(
    game_state: SplendorState,
    agent_index: int,
    color: Color,
    diminishing_return: bool = True,
) -> float:
    """
    Calculate the buying power of a specific color.
    Allow the calculation of the buying power with diminishing return.
    """
    agent_buying_power = get_agent(game_state, agent_index).gems[color]

    if diminishing_return:
        return diminish_return(agent_buying_power)
    else:
        return agent_buying_power


def buying_power(
    game_state: SplendorState,
    agent_index: int,
    diminishing_return: bool = True,
) -> Dict[Color, float]:
    """
    Calculate the buying power for all possible colors.
    Allow the calculation of the buying power with diminishing return.
    """
    power: Dict[Color, float] = {}
    permanent_gems = permanent_buying_power(
        game_state, agent_index, diminishing_return=False
    )

    color: Color
    for color in COLOURS.values():
        power[color] = buying_power_of_color(
            game_state, agent_index, color, diminishing_return=False
        )
        power[color] += permanent_gems[color]

        if diminishing_return:
            power[color] = diminish_return(power[color])

    return power


def total_buying_power(
    game_state: SplendorState,
    agent_index: int,
    diminishing_return: bool = True,
) -> float:
    """
    Calculate the total (sum) buying power of a specific agent.
    """
    return sum(buying_power(game_state, agent_index, diminishing_return).values())


def buying_power_variance(
    game_state: SplendorState,
    agent_index: int,
    diminishing_return: bool = True,
) -> float:
    """
    Calculate the variance of buying power among the different gems colors
    held by a specific agent.
    """
    return np.var(
        list(buying_power(game_state, agent_index, diminishing_return).values())
    )


def cards_owned_by_agent(game_state: SplendorState, agent_index: int) -> int:
    """
    Count the amount of cards owned by a specific agent.
    """
    return len(
        filter(
            get_agent(game_state, agent_index).cards,
            lambda card: card.colour != RESERVED,
        )
    )


def cards_reserved_by_agent(game_state: SplendorState, agent_index: int) -> int:
    """
    Count the amount of cards reserved by a specific agent.
    """
    return len(
        filter(
            get_agent(game_state, agent_index).cards,
            lambda card: card.colour == RESERVED,
        )
    )


def permanent_buying_power_of_color(
    game_state: SplendorState,
    agent_index: int,
    color: Color,
    diminishing_return: bool = True,
) -> float:
    """
    Calculate the buying power of the permanent gems of specific color gained
    from the cards owned by a specific agent.
    Allow the calculation of the buying power with diminishing return.
    """
    permanent_gems = sum(
        list(
            filter(
                get_agent(game_state, agent_index).cards,
                lambda card: card.colour == color,
            )
        )
    )

    if diminishing_return:
        return diminish_return(permanent_gems)
    else:
        return permanent_gems


def permanent_buying_power(
    game_state: SplendorState,
    agent_index: int,
    diminishing_return: bool = True,
) -> Dict[Color, float]:
    """
    Calculate the permanent buying power for all possible colors.
    Allow the calculation of the buying power with diminishing return.
    """
    permanent_power: Dict[Color, float] = {}

    color: Color
    for color in filter(COLOURS.values(), lambda color: color != RESERVED):
        permanent_power[color] = permanent_buying_power_of_color(
            game_state, agent_index, color, diminishing_return
        )

    return permanent_power


def total_permanent_buying_power(
    game_state: SplendorState,
    agent_index: int,
    diminishing_return: bool = True,
) -> float:
    """
    Calculate the total (sum) of permanent buying power of a specific agent.
    """
    return sum(
        permanent_buying_power(game_state, agent_index, diminishing_return).values()
    )


def card_score_to_distance_ratio(
    game_state: SplendorState, agent_index: int, card: Card
) -> float:
    """
    calculate the ratio between the card earned victory points to it's distance
    from a specific agent.
    """
    score = card.points
    distance = distance_in_gems_to_card(game_state, agent_index, card)

    return score / distance


def distance_to_noble_in_cards(
    game_state: SplendorState, agent_index: int, noble: Noble
) -> int:
    """
    Calculate the distance (in missing cards) of an agent from being
    visited by a specific noble.
    """
    missing_permanent_gems = find_missing_permanent_gems(game_state, agent_index, noble)

    return sum(missing_permanent_gems.values())


def distance_to_noble_k_minimal(
    game_state: SplendorState, agent_index: int, noble: Noble
) -> float:
    """
    Calculate the distance (sum of k-minimal card's distances) of an agent
    from being visited by a specific noble.
    """
    agent = get_agent(game_state, agent_index)
    missing_permanent_gems = find_missing_permanent_gems(game_state, agent_index, noble)
    reserved_cards = agent.cards[RESERVED]
    potential_cards: Dict[Color, List[Card]] = {
        color: [] for color in filter(COLOURS.values(), lambda color: color != RESERVED)
    }

    for card in reserved_cards + game_state.board.dealt_list():
        potential_cards[card.colour].append(
            distance_to_card(game_state, agent_index, card)
        )

    for l in potential_cards.values():
        l.sort()

    distance = 0.0
    for color, missing in missing_permanent_gems.items():
        if len(potential_cards[color]) < missing:
            return MAX_NOBLE_DISTANCE

        distance += potential_cards[color][missing]

    return max(0, distance)


def distance_to_noble(
    game_state: SplendorState, agent_index: int, noble: Noble
) -> float:
    """
    Calculate the distance of an agent from being visited by a specific noble.
    """
    agent = get_agent(game_state, agent_index)
    missing_permanent_gems = find_missing_permanent_gems(game_state, agent_index, noble)
    reserved_cards = agent.cards[RESERVED]
    potential_cards: Dict[Color, List[Card]] = {
        color: [] for color in filter(COLOURS.values(), lambda color: color != RESERVED)
    }

    for card in reserved_cards + game_state.board.dealt_list():
        potential_cards[card.colour].append(
            distance_to_card(game_state, agent_index, card)
        )

    for l in potential_cards.values():
        l.sort()

    distance = 0.0
    for color, missing in missing_permanent_gems.items():
        if len(potential_cards[color]) < missing:
            return MAX_NOBLE_DISTANCE

        distance += sum(potential_cards[color][: missing + 1])

    return max(0, distance)
