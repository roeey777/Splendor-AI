"""
Features extraction from SplendorState

BEWARE: THIS IS UNTESTED!!!!
"""

from dataclasses import dataclass
from itertools import chain
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
WINNING_SCORE_TRESHOLD = 15
MAX_RIVALS = 3

### RANDOM VALUES (change?) ###
MAX_NOBLE_DISTANCE = 100
MAX_CARD_DISTANCE = 20
MISSING_CARD_GEMS_DEFAULT = 0
MISSING_CARD_TURNS_DEFAULT = 0
MISSING_NOBEL_GEMS_DEFAULT = 0
MISSING_NOBEL_TURNS_DEFAULT = 0
MISSING_RIVAL_DEFAULT = 0
DIMINISHING_MULTIPLIER = 2.5
WINNING_MULTIPLIER = 4


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


def agent_buying_power(agent) -> Dict[Color, int]:
    return {
        color: agent.gems[color] + len(agent.cards[color]) for color in NORMAL_COLORS
    }


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
    return DIMINISHING_MULTIPLIER * np.log(1 + value)


def missing_card_to_nobel(
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
    missing_permanent_gems = missing_card_to_nobel(game_state, agent_index, noble)

    return sum(missing_permanent_gems.values())


def distance_to_noble_k_minimal(
    game_state: SplendorState, agent_index: int, noble: Noble
) -> float:
    """
    Calculate the distance (sum of k-minimal card's distances) of an agent
    from being visited by a specific noble.
    """
    agent = get_agent(game_state, agent_index)
    missing_permanent_gems = missing_card_to_nobel(game_state, agent_index, noble)
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
    missing_permanent_gems = missing_card_to_nobel(game_state, agent_index, noble)
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


def extract_metrics(game_state: SplendorState, agent_index: int) -> np.array:
    agent = get_agent(game_state, agent_index)
    wild_gems = agent.gems[WILDCARD]
    buying_power = agent_buying_power(agent)
    owned_cards = sum(len(agent.cards[color]) for color in NORMAL_COLORS)

    card_distance_per_color = {color: list() for color in NORMAL_COLORS}

    cards_distances_in_gems: list[int] = list()
    cards_distances_in_turns: list[int] = list()
    # rival_cards_turns_diff = list()
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

    nobles_distances_in_cards = [MAX_NOBLE_DISTANCE] * MAX_NOBLES
    nobles_distances_in_gems = [MAX_NOBLE_DISTANCE] * MAX_NOBLES
    for i, (_, noble_cost) in enumerate(game_state.board.nobles):
        missing_cards = {
            color: cost - len(agent.cards[color])
            for color, cost in noble_cost.items()
            if cost - len(agent.cards[color]) > 0
        }
        nobles_distances_in_cards[i] = sum(missing_cards.values())
        distance_in_gems = 0
        for color, count in missing_cards.items():
            distance_in_gems += sum(card_distance_per_color[color][:count])
            more_cards = count - len(card_distance_per_color[color])
            distance_in_gems += MAX_CARD_DISTANCE * max(more_cards, 0)
        nobles_distances_in_gems[i] = distance_in_gems

    rivals_scores_1 = [0] * MAX_RIVALS
    rivals_scores_2 = [0] * MAX_RIVALS
    for i, rival in enumerate(game_state.agents):
        if i < agent_index:
            rivals_scores_1[i] = rival.score
        elif i > agent_index:
            rivals_scores_2[i - 1] = rival.score

    return np.fromiter(
        chain(
            [
                len(agent.agent_trace.action_reward),
                agent.score,
                WINNING_MULTIPLIER if agent.score >= WINNING_SCORE_TRESHOLD else 0,
                owned_cards,
                len(agent.cards[RESERVED]),
                np.var(list(buying_power.values())),
                wild_gems,
            ],
            buying_power.values(),
            map(diminish_return, buying_power.values()),
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


METRICS_SHAPE = (
    1,  # agent's turns count
    1,  # agent's score
    1,  # agent has crossed 15 points
    1,  # agent's owned cards count
    1,  # agent's reserved cards count
    1,  # variance of buying power
    1,  # agent's wild gems count
    len(NORMAL_COLORS),  # agent's buying power (without wild gems)
    len(NORMAL_COLORS),  # agent's dimishing buying power (without wild gems)
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
