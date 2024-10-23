"""
Useful utilities.
"""

from . import features
from .splendor_model import SplendorGameRule


class LimitRoundsGameRule(SplendorGameRule):
    """
    Wraps `SplendorGameRule`.
    """

    def gameEnds(self) -> bool:
        """
        Limits the game to `ROUNDS_LIMIT` rounds, so random initial agents
        won't get stuck by accident.
        """
        if all(
            len(agent.agent_trace.action_reward) == features.ROUNDS_LIMIT
            for agent in self.current_game_state.agents
        ):
            return True

        return super().gameEnds()
