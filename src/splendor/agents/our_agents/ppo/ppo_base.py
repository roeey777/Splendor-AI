from abc import ABC, abstractmethod
from typing import Any, Union, Tuple, Callable

import torch
import torch.nn as nn

from jaxtyping import Float


class PPOBase(nn.Module, ABC):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

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
    ) -> Tuple[torch.Tensor, ...]:
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
        """
        raise NotImplementedError()

    @abstractmethod
    def init_hidden_state(self) -> Any:
        """
        return the initial hidden state to be used.
        """
        raise NotImplementedError()


PPOBaseFactory = Callable[[int, int, ...], PPOBase]
