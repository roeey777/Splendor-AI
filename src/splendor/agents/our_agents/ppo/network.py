from typing import List, Tuple, Union

import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from .constants import DROPOUT, HIDDEN_DIMS, HUGE_NEG
from .input_norm import InputNormalization
from .ppo_base import PPOBase


class PPO(PPOBase):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers_dims: List[int] = HIDDEN_DIMS,
        dropout: float = DROPOUT,
    ):
        super().__init__(input_dim, output_dim)

        self.hidden_layers_dims = hidden_layers_dims
        self.dropout = dropout

        self.input_norm = InputNormalization(input_dim)

        layers = []
        prev_dim = input_dim
        for next_dim in hidden_layers_dims:
            layers.append(nn.Linear(prev_dim, next_dim))
            layers.append(nn.LayerNorm(next_dim))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            prev_dim = next_dim
        self.net = nn.Sequential(*layers)

        self.actor = nn.Linear(hidden_layers_dims[-1], output_dim)
        self.critic = nn.Linear(hidden_layers_dims[-1], 1)

        # Initialize weights (recursively)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Orthogonal initialization of the weights as suggested by (bullet #2):
        https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
        """
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()

    def _order_x_shape(
        self,
        x: Union[
            Float[torch.Tensor, "batch features"],
            Float[torch.Tensor, "features"],
        ],
    ) -> Float[torch.Tensor, "batch features"]:
        match len(x.shape):
            case 1:
                # assumes that the batch dimention is missing.
                ordered_x = x.unsqueeze(0)
            case 2:
                ordered_x = x
            case _:
                raise ValueError(
                    f"Got tensor of unexpected shape! shape: {x.shape}. there are just to many dimentions."
                )
        ordered_x: Float[torch.Tensor, "batch features"]
        return ordered_x

    def forward(
        self,
        x: Union[
            Float[torch.Tensor, "batch sequence features"],
            Float[torch.Tensor, "batch features"],
            Float[torch.Tensor, "features"],
        ],
        action_mask: Union[
            Float[torch.Tensor, "batch actions"], Float[torch.Tensor, "actions"]
        ],
        *args,
        **kwargs,
    ) -> Tuple[
        Float[torch.Tensor, "batch actions"],
        Float[torch.Tensor, "batch 1"],
    ]:
        """
        Pass input through the network to gain predictions.

        :param x: the input to the network.
                  expected shape: one of the following:
                  (features,) or (batch_size, features) or (batch_size, sequance_length, features).
        :param action_mask: a binary masking tensor, 1's signals a valid action and 0's signals an
                            invalid action.
                            expected shape: (actions,) or (batch_size, actions).
                            where actions are equal to len(ALL_ACTIONS) which comes
                            from Engine.Splendor.gym.envs.actions
        :param hidden_state: hidden state of the recurrent unit.
                             expected shape: (batch_size, num_layers, hidden_state_dim) or
                             (num_layers, hidden_state_dim).
        :return: the actions probabilities and the value estimate.
        """
        x = self._order_x_shape(x)
        x_normalized = self.input_norm(x)
        x1 = self.net(x_normalized)
        actor_output = self.actor(x1)
        masked_actor_output = torch.where(action_mask == 0, HUGE_NEG, actor_output)
        prob = F.softmax(masked_actor_output, dim=1)
        return prob, self.critic(x1)

    def init_hidden_state(self) -> None:
        """
        return the initial hidden state to be used.
        """
        return None
