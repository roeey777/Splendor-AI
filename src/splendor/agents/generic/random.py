"""
Implementation of an agent that selects the first legal action.
"""

import random
from typing import override

from splendor.template import Agent


class RandomAgent(Agent):
    """
    An agent that selects a random legal action.
    """

    # pylint: disable=too-few-public-methods

    @override
    def SelectAction(self, actions, game_state, game_rule):
        return random.choice(actions)


myAgent = RandomAgent  # pylint: disable=invalid-name
