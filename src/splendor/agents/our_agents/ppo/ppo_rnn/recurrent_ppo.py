from abc import abstractmethod
from typing import Any, Union, Tuple, Callable

import torch
import torch.nn as nn

from jaxtyping import Float

from splendor.agents.our_agents.ppo.ppo_base import PPOBase


class RecurrentPPO(PPOBase):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        recurrent_unit: Union[nn.GRU, nn.LSTM, nn.RNN],
    ):
        super().__init__(input_dim, output_dim)
        self.recurrent_unit = recurrent_unit

    @abstractmethod
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
                            from splendor.Splendor.gym.envs.actions
        :return: the actions probabilities, the value estimate and the next hidden state.
        """
        raise NotImplementedError()

    @abstractmethod
    def init_hidden_state(self) -> Any:
        """
        return the initial hidden state to be used.
        """
        raise NotImplementedError()


PPOBaseFactory = Callable[[int, int, ...], PPOBase]
