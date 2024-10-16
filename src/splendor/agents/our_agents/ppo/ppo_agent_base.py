"""
Definition for a base class for all PPO-based agents.
"""

from abc import abstractmethod
from typing import List, Optional

import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import

from splendor.splendor.splendor_model import SplendorGameRule, SplendorState
from splendor.splendor.types import ActionType
from splendor.template import Agent

from .ppo_base import PPOBase


class PPOAgentBase(Agent):
    """
    base class for all PPO-based agents.
    """

    def __init__(self, _id: int, load_net: bool = True):
        super().__init__(_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: Optional[nn.Module] = None
        if load_net:
            self.load_policy(self.load())

    @abstractmethod
    def SelectAction(
        self,
        actions: List[ActionType],
        game_state: SplendorState,
        game_rule: SplendorGameRule,
    ) -> ActionType:
        """
        select an action to play from the given actions.
        """
        raise NotImplementedError()

    @abstractmethod
    def load(self) -> PPOBase:
        """
        load and return the weights of the network.
        """
        raise NotImplementedError()

    def load_policy(self, policy: nn.Module):
        """
        Use a given policy as the agent's network policy.
        """
        self.net = policy.to(self.device)
        self.net.eval()
