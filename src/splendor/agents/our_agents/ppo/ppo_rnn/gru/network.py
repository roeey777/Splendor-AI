from typing import List, Tuple, Union, override

import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from splendor.agents.our_agents.ppo.input_norm import InputNormalization
from splendor.agents.our_agents.ppo.ppo_rnn.recurrent_ppo import RecurrentPPO

from .constants import (
    DROPOUT,
    HIDDEN_DIMS,
    HIDDEN_STATE_DIM,
    HUGE_NEG,
    RECURRENT_LAYERS_AMOUNT,
)


class PPO_GRU(RecurrentPPO):
    """
    Implementation of PPO network architecture using a GRU.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers_dims: List[int] = HIDDEN_DIMS,
        dropout: float = DROPOUT,
        hidden_state_dim: int = HIDDEN_STATE_DIM,
        recurrent_layers_num: int = RECURRENT_LAYERS_AMOUNT,
    ):
        super().__init__(
            input_dim,
            output_dim,
            nn.GRU(
                input_dim,
                hidden_state_dim,
                recurrent_layers_num,
                batch_first=True,
            ),
        )

        self.hidden_layers_dims = hidden_layers_dims
        self.dropout = dropout
        self.hidden_state_dim = hidden_state_dim
        self.recurrent_layers_num = recurrent_layers_num

        self.input_norm = InputNormalization(input_dim)

        layers: List[nn.Module] = []
        prev_dim = hidden_state_dim
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
        elif isinstance(module, (nn.GRU, nn.LSTM)):
            for name, param in module.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, np.sqrt(2))

    def _order_x_shape(
        self,
        x: Union[
            Float[torch.Tensor, "batch sequence features"],
            Float[torch.Tensor, "batch features"],
            Float[torch.Tensor, "features"],
        ],
    ) -> Float[torch.Tensor, "batch sequence features"]:
        ordered_x: Float[torch.Tensor, "batch sequence features"]
        match len(x.shape):
            case 1:
                # assumes that both the batch & the sequance dimentions are missing.
                ordered_x = x.unsqueeze(0).unsqueeze(1)
            case 2:
                # assumes that the sequance dimention is missing.
                ordered_x = x.unsqueeze(1)
            case 3:
                ordered_x = x
            case _:
                raise ValueError(
                    f"Got tensor of unexpected shape! shape: {x.shape}. there are just to many dimentions."
                )
        return ordered_x

    def _order_hidden_state_shape(
        self,
        hidden_state: Union[
            Float[torch.Tensor, "batch num_layers hidden_dim"],
            Float[torch.Tensor, "num_layers hidden_dim"],
        ],
    ) -> Float[torch.Tensor, "num_layers batch hidden_dim"]:
        match len(hidden_state.shape):
            case 2:
                # add batch dimention as the second dimention.
                ordered = hidden_state.unsqueeze(1)
            case 3:
                # re-organize the order of dimentions as GRU expects.
                ordered = torch.permute(hidden_state, (1, 0, 2))
            case _:
                raise ValueError(
                    f"Got hidden state tensor of unexpected shape! shape: {hidden_state.shape}"
                )
        return ordered

    @override
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
        hidden_state: Union[
            Float[torch.Tensor, "batch num_layers hidden_dim"],
            Float[torch.Tensor, "num_layers hidden_dim"],
        ],
        *args,
        **kwargs,
    ) -> Tuple[
        Float[torch.Tensor, "batch actions"],
        Float[torch.Tensor, "batch 1"],
        Float[torch.Tensor, "batch hidden_dim"],
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
        :return: the actions probabilities, the value estimate and the next hidden state.
        """
        x = self._order_x_shape(x)
        hidden_state = self._order_hidden_state_shape(hidden_state)

        x_normalized = self.input_norm(x)

        x_rnn, next_hidden_state = self.recurrent_unit(x_normalized, hidden_state)

        # use only the last output of the recurrent unit (GRU\LSTM)
        x1 = self.net(x_rnn[:, -1, :])
        actor_output = self.actor(x1)
        masked_actor_output = torch.where(action_mask == 0, HUGE_NEG, actor_output)
        prob = F.softmax(masked_actor_output, dim=1)
        return prob, self.critic(x1), next_hidden_state

    @override
    def init_hidden_state(
        self, device: torch.device
    ) -> Float[torch.Tensor, "num_layers hidden_dim"]:
        """
        return the initial hidden state to be used.
        """
        return (
            torch.zeros(self.recurrent_layers_num, self.hidden_state_dim)
            .double()
            .unsqueeze(0)
            .to(device)
        )
