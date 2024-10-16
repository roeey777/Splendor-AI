"""
Implementation of an agent that selects the first legal action.
"""

from typing import override

from splendor.template import Agent


class FirstActionAgent(Agent):
    """
    An agent that selects the first legal action.
    """

    # pylint: disable=too-few-public-methods

    @override
    def SelectAction(self, actions, game_state, game_rule):
        return actions[0]


myAgent = FirstActionAgent  # pylint: disable=invalid-name
