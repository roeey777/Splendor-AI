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

from .ppo_agent_base import PPOAgentBase, PPOBase
from .utils import load_saved_ppo


DEFAULT_SAVED_PPO_PATH = Path(__file__).parent / "ppo_model.pth"


class PPOAgent(PPOAgentBase):
    def __init__(self, _id):
        super().__init__(_id)

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
                torch.from_numpy(state).double().to(self.device)
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
        return load_saved_ppo()


myAgent = PPOAgent
