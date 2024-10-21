"""
Implementation of a PPO agent with MLP neural network
"""

from pathlib import Path
from typing import override

import numpy as np
import torch
from numpy.typing import NDArray

from splendor.splendor.features import extract_metrics_with_cards
from splendor.splendor.gym.envs.utils import (
    create_action_mapping,
    create_legal_actions_mask,
)
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType

from .ppo_agent_base import PPOAgentBase, PPOBase
from .utils import load_saved_ppo

DEFAULT_SAVED_PPO_PATH = Path(__file__).parent / "ppo_model.pth"


class PPOAgent(PPOAgentBase):
    """
    PPO agent with MLP neural network.
    """

    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        """
        select an action to play from the given actions.
        """
        with torch.no_grad():
            state: NDArray = extract_metrics_with_cards(game_state, self.id).astype(
                np.float32
            )
            state_tesnor: torch.Tensor = (
                torch.from_numpy(state).double().to(self.device)
            )

            action_mask = (
                torch.from_numpy(
                    create_legal_actions_mask(actions, game_state, self.id)
                )
                .double()
                .to(self.device)
            )

            # this assertion is only for mypy.
            assert self.net is not None

            action_pred, _ = self.net(state_tesnor, action_mask)
            chosen_action = action_pred.argmax()
            mapping = create_action_mapping(actions, game_state, self.id)

        return mapping[chosen_action.item()]

    @override
    def load(self) -> PPOBase:
        return load_saved_ppo()


myAgent = PPOAgent  # pylint: disable=invalid-name
