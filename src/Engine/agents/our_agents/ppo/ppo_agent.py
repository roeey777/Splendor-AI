from pathlib import Path

import numpy as np
import torch

from Engine.template import Agent
from Engine.Splendor.features import extract_metrics_with_cards
from Engine.Splendor.gym.envs.utils import (
    create_action_mapping,
    create_legal_actions_mask,
)

from .utils import load_saved_ppo


class PPOAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_saved_ppo().to(self.device)
        self.net.eval()
        self.hidden_state = self.net.init_hidden_state().to(self.device)

    def SelectAction(self, actions, game_state, game_rule):
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


myAgent = PPOAgent
