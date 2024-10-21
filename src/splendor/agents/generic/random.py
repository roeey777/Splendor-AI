"""
Implementation of an agent that selects the first legal action.
"""

import random
from typing import override

from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent


class RandomAgent(Agent):
    """
    An agent that selects a random legal action.
    """

    # pylint: disable=too-few-public-methods

    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        return random.choice(actions)


myAgent = RandomAgent  # pylint: disable=invalid-name
