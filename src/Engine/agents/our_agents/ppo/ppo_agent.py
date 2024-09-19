import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
from Engine.template import Agent
from .ppo import PPO, DROPOUT
from Engine.Splendor.features import extract_metrics_with_cards
from Engine.Splendor.gym.envs.utils import (
    create_action_mapping,
    create_legal_actions_mask,
)
from functools import partial


class PPOAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)

        env = gym.make("splendor-v1", agents=[])

        # load_weights
        self.net = PPO(
            env.observation_space.shape[0], env.action_space.n, dropout=DROPOUT
        ).double()
        checkpoint = torch.load(
            str(Path(__file__).parent / f"ppo_model.pth"),
            weights_only=False,
            map_location="cpu",
        )

        self.net.load_state_dict(checkpoint["model_state_dict"])
        if hasattr(self.net, "input_norm"):
            self.net.input_norm.running_mean = checkpoint["running_mean"]
            self.net.input_norm.running_var = checkpoint["running_var"]
        else:
            self.net.running_mean = checkpoint["running_mean"]
            self.net.running_var = checkpoint["running_var"]

        self.net.eval()

    def SelectAction(self, actions, game_state, game_rule):
        state: np.array = extract_metrics_with_cards(game_state, self.id).astype(
            np.float32
        )
        state_tesnor: torch.Tensor = torch.from_numpy(state).double()

        action_mask = torch.from_numpy(
            create_legal_actions_mask(actions, game_state, self.id)
        ).double()

        action_pred, value_pred = self.net(state_tesnor, action_mask)
        chosen_action = action_pred.argmax()
        mapping = create_action_mapping(actions, game_state, self.id)
        return mapping[chosen_action.item()]


myAgent = PPOAgent
