"""
Combinatorially define all possible actions in the game.
"""

from collections import Counter
from dataclasses import dataclass
from enum import Enum, auto
from itertools import combinations, combinations_with_replacement
from typing import Self, cast

from splendor.splendor import splendor_utils
from splendor.splendor.constants import (
    MAX_NOBLES,
    MAX_RESERVED,
    MAX_TIER_CARDS,
    NORMAL_COLORS,
    NUMBER_OF_TIERS,
    RESERVED,
)
from splendor.splendor.splendor_model import SplendorState
from splendor.splendor.types import ActionType, BuyAction, GemsCount

ALL_GEMS_COLORS = splendor_utils.COLOURS.values()
NOBLES_INDICES = list(range(MAX_NOBLES))


class ActionEnum(Enum):
    """
    Enum for all action types.
    """

    PASS = auto()
    COLLECT_SAME = auto()
    COLLECT_DIFF = auto()
    RESERVE = auto()
    BUY_AVAILABLE = auto()
    BUY_RESERVE = auto()


@dataclass
class CardPosition:
    """
    dataclass for representing where a card is located on the board.
    """

    tier: int
    card_index: int
    reserved_index: int


@dataclass
class Action:
    """
    Represent an action as a dataclass with a more comfortable API.
    """

    type: ActionEnum
    collected_gems: GemsCount | None = None
    returned_gems: GemsCount | None = None
    position: CardPosition | None = None
    noble_index: int | None = None

    @classmethod
    def to_action_element(
        cls, action: ActionType, state: SplendorState, agent_index: int
    ) -> Self:
        """
        Convert an action in SplendorGameRule format to Action.
        """
        action_type = ActionEnum[action["type"].upper()]
        collected_gems = cast(GemsCount | None, action.get("collected_gems"))
        returned_gems = cast(GemsCount | None, action.get("returned_gems"))
        noble_index = None
        position = None

        if action.get("noble") is not None:
            noble_index = state.board.nobles.index(action["noble"])

        if action.get("card") is not None:
            buy_action = cast(BuyAction, action)
            card = buy_action["card"]
            if action_type == ActionEnum.BUY_RESERVE:
                position = CardPosition(
                    -1, -1, state.agents[agent_index].cards[RESERVED].index(card)
                )
            else:
                # available card (for buying or reserving).
                tier = card.deck_id
                card_index = state.board.dealt[tier].index(card)
                position = CardPosition(tier, card_index, -1)

        if action_type in (ActionEnum.BUY_AVAILABLE, ActionEnum.BUY_RESERVE):
            # ignore cost of buying a card by set returned_gems to None.
            # this minimizes by order of magnitude the length of ALL_ACTIONS.
            collected_gems = None
            returned_gems = None

        return cls(action_type, collected_gems, returned_gems, position, noble_index)


def _card_gen():
    for deck_num in range(NUMBER_OF_TIERS):
        for card_num in range(MAX_TIER_CARDS):
            yield CardPosition(deck_num, card_num, -1)


def _generate_all_reserve_card_actions():
    for position in _card_gen():
        for noble_index in [None] + NOBLES_INDICES:
            # reserve a card, without collecting the yellow gem.
            yield Action(
                type=ActionEnum.RESERVE,
                collected_gems={},
                returned_gems={},
                position=position,
                noble_index=noble_index,
            )
            # reserve a card, collect a yellow gem
            yield Action(
                type=ActionEnum.RESERVE,
                collected_gems={"yellow": 1},
                returned_gems={},
                position=position,
                noble_index=noble_index,
            )
            for c in NORMAL_COLORS:
                # reserve a card, collect a yellow gem, and return a gem.
                yield Action(
                    type=ActionEnum.RESERVE,
                    collected_gems={"yellow": 1},
                    returned_gems={c: 1},
                    position=position,
                    noble_index=noble_index,
                )


def _generate_all_collect_same_actions():
    for c in NORMAL_COLORS:
        for noble_index in [None] + NOBLES_INDICES:
            # no gems are returned.
            yield Action(
                type=ActionEnum.COLLECT_SAME,
                noble_index=noble_index,
                collected_gems={c: 2},
                returned_gems={},
            )
            # return 2 gems of different colors.
            for c1, c2 in combinations(
                filter(lambda x, color=c: x != color, ALL_GEMS_COLORS), 2
            ):
                yield Action(
                    type=ActionEnum.COLLECT_SAME,
                    noble_index=noble_index,
                    collected_gems={c: 2},
                    returned_gems={c1: 1, c2: 1},
                )
            # return gems of the same color.
            for num_gems_to_return in [1, 2]:
                for c_other in filter(lambda x, color=c: x != color, ALL_GEMS_COLORS):
                    yield Action(
                        type=ActionEnum.COLLECT_SAME,
                        noble_index=noble_index,
                        collected_gems={c: 2},
                        returned_gems={c_other: num_gems_to_return},
                    )


def _generate_all_collect_different_actions():
    for combination_length in [1, 2, 3]:
        for combination in combinations(NORMAL_COLORS, combination_length):
            for noble_index in [None] + NOBLES_INDICES:
                # no gems are returned.
                yield Action(
                    type=ActionEnum.COLLECT_DIFF,
                    noble_index=noble_index,
                    collected_gems={color: 1 for color in combination},
                    returned_gems={},
                )
                for num_gems_to_return in range(1, combination_length + 1):
                    for to_return in combinations_with_replacement(
                        filter(
                            lambda x, comb=combination: x not in comb, ALL_GEMS_COLORS
                        ),
                        num_gems_to_return,
                    ):
                        yield Action(
                            type=ActionEnum.COLLECT_DIFF,
                            noble_index=noble_index,
                            collected_gems={color: 1 for color in combination},
                            returned_gems=dict(Counter(to_return)),
                        )


def _generate_all_buy_reserve_card_actions():
    for reserved_index in range(MAX_RESERVED):
        for noble_index in [None] + NOBLES_INDICES:
            yield Action(
                type=ActionEnum.BUY_RESERVE,
                noble_index=noble_index,
                position=CardPosition(-1, -1, reserved_index),
            )


def _generate_all_buy_available_card_actions():
    for position in _card_gen():
        for noble_index in [None] + NOBLES_INDICES:
            yield Action(
                type=ActionEnum.BUY_AVAILABLE,
                noble_index=noble_index,
                position=position,
            )


ALL_ACTIONS = [
    # # Do nothing.
    Action(type=ActionEnum.PASS),
    # Only acquire a noble.
    *[Action(type=ActionEnum.PASS, noble_index=noble) for noble in NOBLES_INDICES],
    # Reserve a card
    *list(_generate_all_reserve_card_actions()),
    # Collect 2 stones of the same color.
    *list(_generate_all_collect_same_actions()),
    # Collect 3 stones of different colors.
    *list(_generate_all_collect_different_actions()),
    # Buy a Reserved Card.
    *list(_generate_all_buy_reserve_card_actions()),
    # Buy an Available Card.
    *list(_generate_all_buy_available_card_actions()),
]
