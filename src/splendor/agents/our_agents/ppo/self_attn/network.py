"""
Implementation of the neural network, with self-attention, for the PPO.
"""

from typing import override

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import nn

from splendor.agents.our_agents.ppo.input_norm import InputNormalization
from splendor.agents.our_agents.ppo.ppo_base import PPOBase

from .constants import DROPOUT, HIDDEN_DIMS, HUGE_NEG


class PPOSelfAttention(PPOBase):
    """
    PPO neural network with self-attention.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers_dims: list[int] | None = None,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__(input_dim, output_dim)

        self.hidden_layers_dims = (
            hidden_layers_dims if hidden_layers_dims is not None else HIDDEN_DIMS
        )
        self.dropout = dropout

        self.input_norm = InputNormalization(input_dim)

        # In self-attention the embedding dimension is equal to the quary & key dimensions.
        # 1 is for using a single-headed attention.
        self.self_attention = nn.MultiheadAttention(input_dim, 1, dropout=dropout)

        self.net = self.create_hidden_layers(
            input_dim, self.hidden_layers_dims, dropout
        )
        self.actor = nn.Linear(self.hidden_layers_dims[-1], output_dim)
        self.critic = nn.Linear(self.hidden_layers_dims[-1], 1)

        # Initialize weights (recursively)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """
        Orthogonal initialization of the weights as suggested by (bullet #2):
        https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/.
        """
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()

    @override
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
    ) -> tuple[
        Float[torch.Tensor, "batch actions"],
        Float[torch.Tensor, "batch 1"],
        None,
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
        x_normalized = self.input_norm(x)
        x_embeding, _ = self.self_attention(x_normalized, x_normalized, x_normalized)
        x1 = self.net(x_embeding)
        actor_output = self.actor(x1)
        masked_actor_output = torch.where(action_mask == 0, HUGE_NEG, actor_output)
        prob = F.softmax(masked_actor_output, dim=1)
        return prob, self.critic(x1), None
