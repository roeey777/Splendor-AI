"""
define useful type hints.
"""

from typing import Literal, Required, TypedDict

from .constants import Color
from .splendor_model import Card

CollectActionType = Literal[
    "collect_diff",
    "collect_same",
]

ReserveActionType = Literal["reserve"]


BuyActionType = Literal["buy_available", "buy_reserve"]
ActionTypeLiteral = Literal[CollectActionType, ReserveActionType, BuyActionType]

GemsCount = dict[Color, int]
NobleType = tuple[str, GemsCount]


class YellowGemCount(TypedDict):
    """
    TypedDict for the yellow gems count.
    """

    yellow: Required[Literal[1] | None]


class CollectAction(TypedDict):
    """
    TypedDict for collecting gems action.
    """

    type: Required[CollectActionType]
    collected_gems: Required[GemsCount]
    returned_gems: Required[GemsCount]
    noble: Required[NobleType]


class ReserveAction(TypedDict):
    """
    TypedDict for reserving a card action.
    """

    type: Required[ReserveActionType]
    collected_gems: Required[YellowGemCount]
    returned_gems: Required[GemsCount]
    noble: Required[NobleType]


class BuyAction(TypedDict):
    """
    TypedDict for buying action.
    """

    type: Required[BuyActionType]
    card: Required[Card]
    returned_gems: Required[GemsCount]
    noble: Required[NobleType]


ActionType = CollectAction | ReserveAction | BuyAction
