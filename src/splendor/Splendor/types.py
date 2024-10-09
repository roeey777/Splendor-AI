"""
define useful type hints.
"""

from typing import Dict, Literal, Union, TypedDict, Tuple, Optional

from .splendor_model import Card
from .constants import Color


COLLECT_ACTION_TYPES = [
    "collect_diff",
    "collect_same",
]
CollectActionType = Literal[*COLLECT_ACTION_TYPES]

RESERVE_ACTION_TPYES = [
    "reserve",
]
ReserveActionType = Literal[*RESERVE_ACTION_TPYES]

BUY_ACTION_TYPES = [
    "buy_available",
    "buy_reserve",
]
BuyActionType = Literal[*BUY_ACTION_TYPES]
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
