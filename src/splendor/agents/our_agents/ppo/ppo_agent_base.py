from pathlib import Path
from abc import abstractmethod
from typing import List

import numpy as np
import torch
import gymnasium as gym

from splendor.template import Agent
from splendor.Splendor.splendor_model import SplendorState, SplendorGameRule
from splendor.Splendor.types import ActionType

from .ppo_base import PPOBase


class PPOAgentBase(Agent):
    """
    base class for all PPO-based agents.
    """

    def __init__(self, _id):
        super().__init__(_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self.load().to(self.device)
        self.net.eval()

    @abstractmethod
    def SelectAction(
        self,
        actions: List[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule
    ) -> ActionType:
        """
        select an action to play from the given actions.
        """
        raise NotImplementedError()

    @abstractmethod
    def load(self) -> PPOBase:
        """
        load the weights of the network.
        """
        raise NotImplementedError()
