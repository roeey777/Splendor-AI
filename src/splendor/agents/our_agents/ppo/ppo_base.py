"""
Base class for all neural network that should be used by a PPO agent.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol

import torch
from jaxtyping import Float
from torch import nn


class PPOBase(nn.Module, ABC):
    """
    Base class for all neural network that should be used by a PPO agent.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(
        self,
        x: (
            Float[torch.Tensor, "batch sequence features"]
            | Float[torch.Tensor, "batch features"]
            | Float[torch.Tensor, " features"]
        ),
        action_mask: (
            Float[torch.Tensor, "batch actions"] | Float[torch.Tensor, " actions"]
        ),
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Pass input through the network to gain predictions.

        :param x: the input to the network.
                  expected shape: one of the following:
                  (features,) or (batch_size, features) or (batch_size, sequance_length, features).
        :param action_mask: a binary masking tensor, 1's signals a valid action and 0's signals an
                            invalid action.
                            expected shape: (actions,) or (batch_size, actions).
                            where actions are equal to len(ALL_ACTIONS) which comes
                            from splendor.Splendor.gym.envs.actions
        :return: the actions probabilities and the value estimate.
        """
        raise NotImplementedError()

    def init_hidden_state(self, device: torch.device) -> Any:  # noqa: ANN401
        """
        return the initial hidden state to be used.
        """
        # device is unused.
        _ = device

    def create_hidden_layers(
        self, input_dim: int, hidden_layers_dims: list[int], dropout: float
    ) -> nn.Module:
        """
        Create hidden layers based on given dimensions.
        """
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for next_dim in hidden_layers_dims:
            layers.append(nn.Linear(prev_dim, next_dim))
            layers.append(nn.LayerNorm(next_dim))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            prev_dim = next_dim
        return nn.Sequential(*layers)


class PPOBaseFactory(Protocol):
    """
    factory for PPO models.
    """

    # pylint: disable=too-few-public-methods
    def __call__(self, input_dim: int, output_dim: int, *args, **kwargs) -> PPOBase:
        pass
