"""
PPO with GRU (Gated Recurrent Unit) implementation.
"""

from typing import override

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import nn

from splendor.agents.our_agents.ppo.input_norm import InputNormalization
from splendor.agents.our_agents.ppo.ppo_rnn.recurrent_ppo import RecurrentPPO

from .constants import (
    DROPOUT,
    HIDDEN_DIMS,
    HIDDEN_STATE_DIM,
    HUGE_NEG,
    RECURRENT_LAYERS_AMOUNT,
)


class PpoGru(RecurrentPPO):
    """
    Implementation of PPO network architecture using a GRU.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers_dims: list[int] | None = None,
        dropout: float = DROPOUT,
        hidden_state_dim: int = HIDDEN_STATE_DIM,
        recurrent_layers_num: int = RECURRENT_LAYERS_AMOUNT,
    ) -> None:
        # pylint: disable=too-many-arguments,too-many-positional-arguments
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

        self.hidden_layers_dims = (
            hidden_layers_dims if hidden_layers_dims is not None else HIDDEN_DIMS
        )
        self.dropout = dropout
        self.hidden_state_dim = hidden_state_dim
        self.recurrent_layers_num = recurrent_layers_num

        self.input_norm = InputNormalization(input_dim)
        self.net = self.create_hidden_layers(
            input_dim, self.hidden_layers_dims, dropout
        )
        self.actor = nn.Linear(self.hidden_layers_dims[-1], output_dim)
        self.critic = nn.Linear(self.hidden_layers_dims[-1], 1)

        # Initialize weights (recursively)
        self.apply(self._init_weights)

    def _order_x_shape(
        self,
        x: (
            Float[torch.Tensor, "batch sequence features"]
            | Float[torch.Tensor, "batch features"]
            | Float[torch.Tensor, " features"]
        ),
    ) -> Float[torch.Tensor, "batch sequence features"]:
        ordered_x: Float[torch.Tensor, "batch sequence features"]
        match len(x.shape):
            case 1:
                # assumes that both the batch & the sequance dimensions are missing.
                ordered_x = x.unsqueeze(0).unsqueeze(1)
            case 2:
                # assumes that the sequance dimension is missing.
                ordered_x = x.unsqueeze(1)
            case 3:
                ordered_x = x
            case _:
                raise ValueError(
                    f"Got tensor of unexpected shape! shape: {x.shape}. "
                    "there are just to many dimensions."
                )
        return ordered_x

    def _order_hidden_state_shape(
        self,
        hidden_state: (
            tuple[Float[torch.Tensor, "batch num_layers hidden_dim"], None]
            | tuple[Float[torch.Tensor, "num_layers hidden_dim"], None]
        ),
    ) -> Float[torch.Tensor, "num_layers batch hidden_dim"]:
        hidden, *_ = hidden_state
        match len(hidden.shape):
            case 2:
                # add batch dimension as the second dimension.
                ordered = hidden.unsqueeze(1)
            case 3:
                # re-organize the order of dimensions as GRU expects.
                ordered = torch.permute(hidden, (1, 0, 2))
            case _:
                raise ValueError(
                    f"Got hidden state tensor of unexpected shape! shape: {hidden.shape}"
                )
        return ordered

    @override
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
        hidden_state: (
            tuple[Float[torch.Tensor, "batch num_layers hidden_dim"], None]
            | tuple[Float[torch.Tensor, "num_layers hidden_dim"], None]
        ),
        *args,
        **kwargs,
    ) -> tuple[
        Float[torch.Tensor, "batch actions"],
        Float[torch.Tensor, "batch 1"],
        Float[torch.Tensor, "batch hidden_dim"],
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
        :return: the actions probabilities, the value estimate and the next hidden state.
        """
        x = self._order_x_shape(x)
        hidden = self._order_hidden_state_shape(hidden_state)

        x_normalized = self.input_norm(x)

        x_rnn, next_hidden_state = self.recurrent_unit(x_normalized, hidden)

        # use only the last output of the recurrent unit (GRU\LSTM)
        x1 = self.net(x_rnn[:, -1, :])
        actor_output = self.actor(x1)
        masked_actor_output = torch.where(action_mask == 0, HUGE_NEG, actor_output)
        prob = F.softmax(masked_actor_output, dim=1)
        return prob, self.critic(x1), next_hidden_state, None

    @override
    def init_hidden_state(
        self, device: torch.device
    ) -> tuple[Float[torch.Tensor, "num_layers hidden_dim"], None]:
        """
        return the initial hidden state to be used.
        """
        return (
            torch.zeros(self.recurrent_layers_num, self.hidden_state_dim)
            .double()
            .unsqueeze(0)
            .to(device),
            None,
        )
