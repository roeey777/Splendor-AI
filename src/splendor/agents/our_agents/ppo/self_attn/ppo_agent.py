from pathlib import Path
from typing import List, override

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray

from splendor.agents.our_agents.ppo.ppo_agent_base import PPOAgentBase
from splendor.agents.our_agents.ppo.ppo_base import PPOBase
from splendor.agents.our_agents.ppo.utils import load_saved_model
from splendor.Splendor.features import extract_metrics_with_cards
from splendor.Splendor.gym.envs.utils import (
    create_action_mapping,
    create_legal_actions_mask,
)
from splendor.Splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.Splendor.types import ActionType

from .network import PPOSelfAttention

DEFAULT_SAVED_PPO_SELF_ATTENTION_PATH = Path(__file__).parent / "ppo_model.pth"


class PPOSelfAttentionAgent(PPOAgentBase):
    @override
    def SelectAction(
        self,
        actions: List[ActionType],
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
                torch.from_numpy(state).double().unsqueeze(0).to(self.device)
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
        """
        load the weights of the network.
        """
        return load_saved_model(DEFAULT_SAVED_PPO_SELF_ATTENTION_PATH, PPOSelfAttention)


myAgent = PPOSelfAttentionAgent
