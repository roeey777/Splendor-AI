from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from .common import calculate_advantages, calculate_returns


@dataclass
class RolloutBuffer:
    size: int
    input_dim: int
    action_dim: int
    is_recurrent: bool = False
    hidden_states_dim: Optional[int] = None
    device: Optional[torch.device] = None
    states: torch.Tensor = field(init=False)
    actions: torch.Tensor = field(init=False)
    action_mask_history: torch.Tensor = field(init=False)
    log_prob_actions: torch.Tensor = field(init=False)
    values: torch.Tensor = field(init=False)
    rewards: torch.Tensor = field(init=False)
    dones: torch.Tensor = field(init=False)
    hidden_states: Optional[torch.Tensor] = field(init=False)
    index: int = field(default=0, init=False)
    full: bool = field(default=False, init=False)

    def __post_init__(self):
        self.index = 0
        self.full = False

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.states = torch.zeros((self.size, self.input_dim), dtype=torch.float64).to(
            self.device
        )
        self.actions = torch.zeros((self.size, 1), dtype=torch.float64).to(self.device)
        self.action_mask_history = torch.zeros(
            (self.size, self.action_dim), dtype=torch.float64
        ).to(self.device)
        self.log_prob_actions = torch.zeros((self.size, 1), dtype=torch.float64).to(
            self.device
        )
        self.values = torch.zeros((self.size, 1), dtype=torch.float64).to(self.device)
        self.rewards = torch.zeros((self.size, 1), dtype=torch.float64).to(self.device)
        self.dones = torch.zeros((self.size, 1), dtype=torch.bool).to(self.device)

        if self.is_recurrent:
            if self.hidden_states_dim is None:
                raise ValueError(
                    "hidden_states_dim must be an int when is_recurrent is set"
                )

            self.hidden_states = torch.zeros(
                (self.size, 1, self.hidden_states_dim), dtype=torch.float64
            ).to(self.device)
        else:
            self.hidden_states = torch.zeros(self.size, dtype=torch.float64)

    def remember(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor,
        log_prob_action: torch.Tensor,
        value: float,
        reward: float,
        done: bool,
        hidden_state: Optional[torch.Tensor] = None,
    ):
        with torch.no_grad():
            if self.full:
                return

            self.states[self.index] = state
            self.actions[self.index] = action
            self.action_mask_history[self.index] = action_mask
            self.log_prob_actions[self.index] = log_prob_action
            self.values[self.index] = value
            self.rewards[self.index] = reward
            self.dones[self.index] = done

            if self.is_recurrent:
                # those assertion is only for mypy.
                assert self.hidden_states is not None
                assert hidden_state is not None
                self.hidden_states[self.index] = hidden_state

            self.index += 1

            if self.index >= self.size:
                self.full = True

    def clear(self):
        self.index = 0
        self.full = False

    def calculate_gae(
        self, discount_factor: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Generalized Advantage Estimation (GAE).
        """
        with torch.no_grad():
            returns = calculate_returns(self.rewards[: self.index], discount_factor).to(
                self.device
            )

            advantages = calculate_advantages(returns, self.values[: self.index]).to(
                self.device
            )

        return advantages, returns

    def unpack(self, discount_factor: float) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.hidden_states is not None:
            hidden_states = self.hidden_states[: self.index]
        else:
            hidden_states = torch.empty(1)

        states = self.states[: self.index]
        actions = self.actions[: self.index]
        action_masks = self.action_mask_history[: self.index]
        log_prob_actions = self.log_prob_actions[: self.index]
        dones = self.dones[: self.index]

        advantages, returns = self.calculate_gae(discount_factor)

        return (
            hidden_states,
            states,
            actions,
            action_masks,
            log_prob_actions,
            advantages,
            returns,
            dones,
        )
