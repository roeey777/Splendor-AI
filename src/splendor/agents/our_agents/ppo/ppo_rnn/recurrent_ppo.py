"""
Base class for all PPO which incorporates a recurrent unit in their neural network
architecture.
"""

from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from jaxtyping import Float
from torch import nn

from splendor.agents.our_agents.ppo.ppo_base import PPOBase


class RecurrentPPO(PPOBase):
    """
    Base class for all PPO models with recurrent unit.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        recurrent_unit: nn.GRU | nn.LSTM | nn.RNN,
    ) -> None:
        super().__init__(input_dim, output_dim)
        self.recurrent_unit = recurrent_unit

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """
        Orthogonal initialization of the weights as suggested by (bullet #2):
        https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
        """
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
        elif isinstance(module, nn.GRU | nn.LSTM):
            for name, param in module.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, np.sqrt(2))

    @abstractmethod
    def forward(  # type: ignore
        self,
        x: (
            Float[torch.Tensor, "batch sequence features"]
            | Float[torch.Tensor, "batch features"]
            | Float[torch.Tensor, " features"]
        ),
        action_mask: (
            Float[torch.Tensor, "batch actions"] | Float[torch.Tensor, " actions"]
        ),
        hidden_state: Any,  # noqa: ANN401
        *args,
        **kwargs,
    ) -> tuple[
        Float[torch.Tensor, "batch actions"],
        Float[torch.Tensor, "batch 1"],
        Float[torch.Tensor, "batch hidden_dim"],
    ]:
        # pylint: disable=arguments-differ
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
        :return: the actions probabilities, the value estimate and the next hidden state.
        """
        raise NotImplementedError()

    @abstractmethod
    def init_hidden_state(self, device: torch.device) -> tuple[Any, Any]:
        """
        return the initial hidden state to be used.
        """
        raise NotImplementedError()
