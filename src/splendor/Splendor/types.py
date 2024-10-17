"""
define useful type hints.
"""

from typing import Dict, Literal, Optional, Required, Tuple, TypedDict, Union

from .constants import Color
from .splendor_model import Card

CollectActionType = Literal[
    (
        "collect_diff",
        "collect_same",
    )
]

ReserveActionType = Literal["reserve"]


BuyActionType = Literal["buy_available", "buy_reserve"]
ActionTypeLiteral = Literal[CollectActionType, ReserveActionType, BuyActionType]

GemsCount = Dict[Color, int]
NobleType = Tuple[str, GemsCount]


class YellowGemCount(TypedDict):
    yellow: Required[Optional[Literal[1]]]


class CollectAction(TypedDict):
    type: Required[CollectActionType]
    collected_gems: Required[GemsCount]
    returned_gems: Required[GemsCount]
    noble: Required[NobleType]


class ReserveAction(TypedDict):
    type: Required[ReserveActionType]
    collected_gems: Required[YellowGemCount]
    returned_gems: Required[GemsCount]
    noble: Required[NobleType]


class BuyAction(TypedDict):
    type: Required[BuyActionType]
    card: Required[Card]
    returned_gems: Required[GemsCount]
    noble: Required[NobleType]


ActionType = Union[CollectAction, ReserveAction, BuyAction]
