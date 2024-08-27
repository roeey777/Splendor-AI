"""
Features extraction from SplendorState

BEWARE: THIS IS UNTESTED!!!!
"""
import numpy as np

from typing import List, Literal, Dict

from .splendor_model import Card, SplendorState
from .splendor_utils import COLOURS


Color = Literal[*COLOURS.values()]
RESERVED = "yellow"


def get_agent(game_state: SplendorState, agent_index: int) -> SplendorState.AgentState:
    """
    Extract the AgentState of a specific agent from the game state.
    """
    return game_state.agents[agent_index]


def turns_made_by_agent(game_state: SplendorState, agent_index: int) -> int:
    """
    Extract the number of turns made by a given agent.
    """
    if agent_index not in range(game_state.agents):
        raise ValueError("agent index out of range")

    return len(get_agent(game_state, agent_index).agent_trace.action_reward)


def score_of_agent(game_state: SplendorState, agent_index: int) -> int:
    """
    Extract the score (number of victory points) of a single player.
    """
    if agent_index not in range(game_state.agents):
        raise ValueError("agent index out of range")

    return get_agent(game_state, agent_index).score


def distance_to_card(game_state: SplendorState, agent_index: int, card: Card) -> int:
    """
    Calculate the distance (required amount of gems) of an agent from
    purchesing a given card.
    """
    if agent_index not in range(game_state.agents):
        raise ValueError("agent index out of range")

    agent = get_agent(game_state, agent_index)
    missing_gems: List[int] = [
        cost - agent.gems[color] for color, cost in card.cost.items()
    ]
    return sum(missing_gems)


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
    if agent_index not in range(game_state.agents):
        raise ValueError("agent index out of range")

    agent_buying_power = get_agent(game_state, agent_index).gems[color]

    if diminishing_return:
        return np.log(1 + agent_buying_power)
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

    color: Color
    for color in COLOURS.values():
        power[color] = buying_power_of_color(
            game_state, agent_index, color, diminishing_return
        )

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
        return np.log(1 + permanent_gems)
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
    for color in COLOURS.values():
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
    distance = distance_to_card(game_state, agent_index, card)

    return score / distance
