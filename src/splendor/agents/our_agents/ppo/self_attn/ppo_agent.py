from pathlib import Path
from typing import List

import numpy as np
import torch
import gymnasium as gym

from splendor.Splendor.features import extract_metrics_with_cards
from splendor.Splendor.gym.envs.utils import (
    create_action_mapping,
    create_legal_actions_mask,
)
from splendor.Splendor.splendor_model import SplendorState, SplendorGameRule
from splendor.Splendor.types import ActionType
from splendor.agents.our_agents.ppo.ppo_agent_base import PPOAgentBase
from splendor.agents.our_agents.ppo.ppo_base import PPOBase
from splendor.agents.our_agents.ppo.utils import load_saved_model

from .network import PPOSelfAttention


DEFAULT_SAVED_PPO_SELF_ATTENTION_PATH = Path(__file__).parent / "ppo_model.pth"


class PPOSelfAttentionAgent(PPOAgentBase):
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
            state: np.array = extract_metrics_with_cards(game_state, self.id).astype(
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

            action_pred, _ = self.net(state_tesnor, action_mask)
            chosen_action = action_pred.argmax()
            mapping = create_action_mapping(actions, game_state, self.id)

        return mapping[chosen_action.item()]

    def load(self) -> PPOBase:
        """
        load the weights of the network.
        """
        return load_saved_model(DEFAULT_SAVED_PPO_SELF_ATTENTION_PATH, PPOSelfAttention)


myAgent = PPOSelfAttentionAgent
