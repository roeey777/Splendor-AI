"""
Implementation of an agent that selects the first legal action.
"""

from typing import override

from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent


class FirstActionAgent(Agent):
    """
    An agent that selects the first legal action.
    """

    # pylint: disable=too-few-public-methods

    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        return actions[0]


myAgent = FirstActionAgent  # pylint: disable=invalid-name
