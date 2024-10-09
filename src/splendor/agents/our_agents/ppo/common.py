from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions


from .constants import (
    ENTROPY_COEFFICIENT,
    VALUE_COEFFICIENT,
    VERY_SMALL_EPSILON,
)


def calculate_returns(rewards, discount_factor, normalize=True):
    returns = []
    R = 0

    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)

    returns = torch.tensor(returns)

    if normalize:
        # avoid possible division by 0
        returns = (returns - returns.mean()) / (returns.std() + VERY_SMALL_EPSILON)

    return returns


def calculate_advantages(returns, values, normalize=True):
    advantages = returns - values

    if normalize:
        # avoid possible division by 0
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + VERY_SMALL_EPSILON
        )

    return advantages


def calculate_policy_loss(
    action_prob: torch.Tensor,
    actions: torch.Tensor,
    log_prob_actions: torch.Tensor,
    advantages: torch.Tensor,
    ppo_clip,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dist = distributions.Categorical(action_prob)

    # new log prob using old actions
    new_log_prob_actions = dist.log_prob(actions)
    policy_ratio = (new_log_prob_actions - log_prob_actions).exp()
    policy_loss_1 = policy_ratio * advantages
    policy_loss_2 = (
        torch.clamp(policy_ratio, min=1.0 - ppo_clip, max=1.0 + ppo_clip) * advantages
    )

    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
    kl_divergence_estimate = (
        (log_prob_actions - new_log_prob_actions).mean().detach().cpu()
    )

    # entropy bonus - use to improve exploration.
    # as seen here (bullet #10):
    # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    entropy = dist.entropy().mean()

    return policy_loss, kl_divergence_estimate, entropy


def calculate_loss(
    policy_loss: torch.Tensor,
    value_loss: torch.Tensor,
    entropy_bonus: torch.Tensor
) -> torch.Tensor:
    """
    final loss of clipped objective PPO, as seen here:
    https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L91
    """
    loss = (
        policy_loss
        + VALUE_COEFFICIENT * value_loss
        - ENTROPY_COEFFICIENT * entropy_bonus
    )

    return loss

