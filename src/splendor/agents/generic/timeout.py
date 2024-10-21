"""
Implementation of an agent that selects the first legal action after 2 seconds sleep.
"""

import time
from typing import override

from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent


class TimeoutAgent(Agent):
    """
    An agent that selects the first legal action, only after a 2 seconds sleep.
    """

    # pylint: disable=too-few-public-methods

    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        time.sleep(2)
        return actions[0]


myAgent = TimeoutAgent  # pylint: disable=invalid-name
