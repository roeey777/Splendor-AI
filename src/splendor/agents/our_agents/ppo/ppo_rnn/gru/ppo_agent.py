from pathlib import Path
from typing import List

import gymnasium as gym
import numpy as np
import torch

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

from .network import PPO_GRU

DEFAULT_SAVED_PPO_GRU_PATH = Path(__file__).parent / "ppo_gru_model.pth"


class PpoGruAgent(PPOAgentBase):
    def __init__(self, _id):
        super().__init__(_id)
        self.hidden_state = self.net.init_hidden_state().to(self.device)

    def SelectAction(
        self,
        actions: List[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        with torch.no_grad():
            state: np.array = extract_metrics_with_cards(game_state, self.id).astype(
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

            action_pred, _, next_hidden_state = self.net(
                state_tesnor, action_mask, self.hidden_state
            )
            chosen_action = action_pred.argmax()
            mapping = create_action_mapping(actions, game_state, self.id)
            self.hidden_state = next_hidden_state

        return mapping[chosen_action.item()]

    def load(self) -> PPOBase:
        """
        load the weights of the network.
        """
        return load_saved_model(DEFAULT_SAVED_PPO_GRU_PATH, PPO_GRU)


myAgent = PpoGruAgent
