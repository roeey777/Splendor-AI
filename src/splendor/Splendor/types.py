"""
define useful type hints.
"""

from typing import Dict, Literal, Optional, Tuple, TypedDict, Union

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
    yellow: Optional[Literal[1]]


class CollectAction(TypedDict):
    type: CollectActionType
    collected_gems: GemsCount
    returned_gems: GemsCount
    noble: NobleType


class ReserveAction(TypedDict):
    type: ReserveActionType
    collected_gems: YellowGemCount
    returned_gems: GemsCount
    noble: NobleType


class BuyAction(TypedDict):
    type: BuyActionType
    card: Card
    returned_gems: GemsCount
    noble: NobleType


ActionType = Union[CollectAction, ReserveAction, BuyAction]
