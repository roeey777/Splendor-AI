import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
from Engine.template import Agent
from .ppo import PPO, DROPOUT
from Engine.Splendor.features import extract_metrics_with_cards
from Engine.Splendor.gym.envs.utils import create_action_mapping
from functools import partial


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        # load_weights
        env = gym.make("splendor-v1", agents=[])
        self.net = PPO(env.observation_space.shape[0], env.action_space.n, DROPOUT)
        # PPO = partial(ppo, input_dim=env.observation_space.shape[0],
        #               output_dim=env.action_space.n, dropout=0)
        torch.serialization.add_safe_globals([PPO])
        self.net = torch.load(str(Path(__file__).parent / f"ppo_model.pth"),
                              weights_only=True,
                              map_location="cpu")
        # self.net.load_state_dict(torch.load(str(Path(__file__).parent /
        #                            f"ppo_model.pth"), weights_only=False))
        self.net.eval()

    def SelectAction(self, actions, game_state, game_rule):
        state = extract_metrics_with_cards(game_state, self.id).astype(
            np.float32)
        state = torch.from_numpy(state)
        action_pred, value_pred = self.net(state)
        chosen_action = action_pred.argmax()
        mapping = create_action_mapping(actions, game_state, self.id)
        return mapping[chosen_action.item()]
