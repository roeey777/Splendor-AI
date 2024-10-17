"""
Useful constants regarding the game.
"""

from typing import Literal, cast

from .splendor_utils import COLOURS

Color = Literal["black", "red", "yellow", "green", "blue", "white"]
WILDCARD = "yellow"
RESERVED = WILDCARD
NORMAL_COLORS = list(
    cast(Color, color) for color in COLOURS.values() if color != WILDCARD
)
NUMBER_OF_TIERS = 3
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