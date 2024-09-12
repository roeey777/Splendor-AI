from __future__ import annotations

from collections import Counter
from itertools import combinations, combinations_with_replacement
from typing import Dict, Literal, List, Tuple, Optional
from enum import Enum, auto
from dataclasses import dataclass

import numpy as np
import gymnasium as gym

from Engine.Splendor import splendor_utils


NUMBER_OF_NOBLES = 3
NUMBER_OF_DECKS = 3
MAX_NUMBER_OF_CARDS_FROM_DECK = 4
MAX_NUMBER_OF_RESERVED_CARDS = 3
RESERVED = "yellow"
COLORS = [c for c in splendor_utils.COLOURS.values() if c != RESERVED]
NOBLES_INDICES = list(range(NUMBER_OF_NOBLES))
# ColorType = Literal[*COLORS]

GemsDict = Dict[str, int]


class ActionType(Enum):
    PASS = auto()
    COLLECT_SAME = auto()
    COLLECT_DIFF = auto()
    RESERVE = auto()
    BUY_AVAILABLE = auto()
    BUY_RESERVE = auto()


@dataclass
class CardPosition:
    tier: int
    card_index: int
    reserved_index: int


@dataclass
class Action:
    type: ActionType
    collected_gems: Optional[GemsDict] = None
    returned_gems: Optional[GemsDict] = None
    position: Optional[CardPosition] = None
    noble_index: Optional[int] = None

    @staticmethod
    def to_action_element(
        action: Dict, state: SplendorState, agent_index: int
    ) -> Action:
        """
        Convert an action in SplendorGameRule format to Action.
        """
        action_type = getattr(ActionType, action["type"].upper())
        collected_gems = action.get("collected_gems")
        returned_gems = action.get("returned_gems")
        noble_index = None
        position = None

        if action.get("noble") is not None:
            noble_index = state.board.nobles.index(action["noble"])

        if action.get("card") is not None:
            card = action["card"]
            if (
                action_type == ActionType.RESERVE
                or action_type == ActionType.BUY_AVAILABLE
            ):
                # available card (for buying or reserving).
                tier = card.deck_id
                card_index = state.board.dealt[tier].index(card)
                position = CardPosition(tier, card_index, -1)
            else:
                # buy a reserved card.
                position = CardPosition(
                    -1, -1, state.agents[agent_index].cards[RESERVED].index(card)
                )

        return Action(action_type, collected_gems, returned_gems, position, noble_index)


def card_gen():
    for deck_num in range(NUMBER_OF_DECKS):
        for card_num in range(MAX_NUMBER_OF_CARDS_FROM_DECK):
            yield CardPosition(deck_num, card_num, -1)


def generate_all_reserve_card_actions():
    for position in card_gen():
        for noble_index in [None] + NOBLES_INDICES:
            # reserve a card, without collecting the yellow gem.
            yield Action(
                type=ActionType.RESERVE,
                collected_gems={},
                returned_gems={},
                position=position,
                noble_index=noble_index,
            )
            # reserve a card, collect a yellow gem
            yield Action(
                type=ActionType.RESERVE,
                collected_gems={"yellow": 1},
                returned_gems={},
                position=position,
                noble_index=noble_index,
            )
            for c in COLORS:
                # reserve a card, collect a yellow gem, and return a gem.
                yield Action(
                    type=ActionType.RESERVE,
                    collected_gems={"yellow": 1},
                    returned_gems={c: 1},
                    position=position,
                    noble_index=noble_index,
                )


def generate_all_collect_same_actions():
    for c in COLORS:
        for noble_index in [None] + NOBLES_INDICES:
            # no gems are returned.
            yield Action(
                type=ActionType.COLLECT_SAME,
                noble_index=noble_index,
                collected_gems={c: 2},
                returned_gems={},
            )
            # return 2 gems of different colors.
            for c1, c2 in combinations(filter(lambda x: x != c, COLORS), 2):
                yield Action(
                    type=ActionType.COLLECT_SAME,
                    noble_index=noble_index,
                    collected_gems={c: 2},
                    returned_gems={c1: 1, c2: 1},
                )
            # return gems of the same color.
            for num_gems_to_return in [1, 2]:
                for c_other in filter(lambda x: x != c, COLORS):
                    yield Action(
                        type=ActionType.COLLECT_SAME,
                        noble_index=noble_index,
                        collected_gems={c: 2},
                        returned_gems={c_other: num_gems_to_return},
                    )


def generate_all_collect_different_actions():
    for c1, c2, c3 in combinations(COLORS, 3):
        for noble_index in [None] + NOBLES_INDICES:
            # no gems are returned.
            yield Action(
                type=ActionType.COLLECT_DIFF,
                noble_index=noble_index,
                collected_gems={c1: 1, c2: 1, c3: 1},
                returned_gems={},
            )
            for num_gems_to_return in [1, 2, 3]:
                for to_return in combinations_with_replacement(
                    filter(lambda x: x not in [c1, c2, c3], COLORS),
                    num_gems_to_return,
                ):
                    yield Action(
                        type=ActionType.COLLECT_DIFF,
                        noble_index=noble_index,
                        collected_gems={c1: 1, c2: 1, c3: 1},
                        returned_gems=dict(Counter(to_return)),
                    )


def generate_all_buy_reserve_card_actions():
    for reserved_index in range(MAX_NUMBER_OF_RESERVED_CARDS):
        for noble_index in [None] + NOBLES_INDICES:
            yield Action(
                type=ActionType.BUY_AVAILABLE,
                noble_index=noble_index,
                position=CardPosition(-1, -1, reserved_index),
            )


def generate_all_buy_available_card_actions():
    for position in card_gen():
        for noble_index in [None] + NOBLES_INDICES:
            yield Action(
                type=ActionType.BUY_AVAILABLE,
                noble_index=noble_index,
                position=position,
            )


ALL_ACTIONS = [
    # # Do nothing.
    # Action(type=ActionType.PASS),
    # Only acquire a noble.
    *[Action(type=ActionType.PASS, noble_index=noble) for noble in NOBLES_INDICES],
    # Reserve a card
    *list(generate_all_reserve_card_actions()),
    # Collect 2 stones of the same color.
    *list(generate_all_collect_same_actions()),
    # Collect 3 stones of different colors.
    *list(generate_all_collect_different_actions()),
    # Buy a Reserved Card.
    *list(generate_all_buy_reserve_card_actions()),
    # Buy an Available Card.
    *list(generate_all_buy_available_card_actions()),
]
