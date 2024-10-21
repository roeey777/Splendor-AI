"""
Implementation for a PPO agent which uses GRU in his neural network
"""

from pathlib import Path
from typing import override

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from splendor.agents.our_agents.ppo.ppo_agent_base import PPOAgentBase
from splendor.agents.our_agents.ppo.ppo_base import PPOBase
from splendor.agents.our_agents.ppo.utils import load_saved_model
from splendor.splendor.features import extract_metrics_with_cards
from splendor.splendor.gym.envs.utils import (
    create_action_mapping,
    create_legal_actions_mask,
)
from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType

from .network import PpoGru

DEFAULT_SAVED_PPO_GRU_PATH = Path(__file__).parent / "ppo_gru_model.pth"


class PpoGruAgent(PPOAgentBase):
    """
    PPO agent with GRU.
    """

    @override
    def __init__(self, _id: int, load_net: bool = True):
        super().__init__(_id, load_net)

        if load_net:
            # this assertion is only for mypy
            assert self.net is not None
            self.hidden_state = self.net.init_hidden_state(self.device)

    @override
    def SelectAction(
        self,
        actions: list[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
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

            action_pred, _, next_hidden_state, __ = self.net(
                state_tesnor, action_mask, self.hidden_state
            )
            chosen_action = action_pred.argmax()
            mapping = create_action_mapping(actions, game_state, self.id)
            self.hidden_state = next_hidden_state

        return mapping[chosen_action.item()]

    @override
    def load(self) -> PPOBase:
        """
        load the weights of the network.
        """
        return load_saved_model(DEFAULT_SAVED_PPO_GRU_PATH, PpoGru)

    @override
    def load_policy(self, policy: nn.Module):
        super().load_policy(policy)

        # this assertion is only for mypy
        assert self.net is not None
        self.hidden_state = self.net.init_hidden_state().to(self.device)


myAgent = PpoGruAgent  # pylint: disable=invalid-name
