"""
Collection of useful calculation functions.
"""

import torch
from torch import distributions

from .constants import ENTROPY_COEFFICIENT, VALUE_COEFFICIENT, VERY_SMALL_EPSILON


def calculate_returns(
    rewards: torch.Tensor, discount_factor: float, normalize: bool = True
) -> torch.Tensor:
    """
    calculate episodes returns (cumulative summation of the rewards).

    :param rewards: the rewards obtained throughout each episode.
    :param discount_factor: by how much rewards decay over time.
    :param normalize: should the returns be normalized (have 0 mean and variance of 1).
    :return: the calculated returns.
    """
    returns_list: list[float] = []
    cumulative_reward: float = 0

    for r in reversed(rewards):
        cumulative_reward = r + cumulative_reward * discount_factor
        returns_list.insert(0, cumulative_reward)

    returns = torch.tensor(returns_list)

    if normalize:
        # avoid possible division by 0
        returns = (returns - returns.mean()) / (returns.std() + VERY_SMALL_EPSILON)

    return returns


def calculate_advantages(
    returns: torch.Tensor, values: torch.Tensor, normalize: bool = True
) -> torch.Tensor:
    """
    Calculate the advantages.

    :param returns: the returns (cumulative summation of rewards).
    :param values: the value estimates for each state.
    :param normalize: should the advantages be normalized, i.e. have 0 mean and variance of 1.
    :return: the calculated advantages.
    """
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
    ppo_clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    calculate the clipped policy loss.

    :param action_prob: the actions probabilities.
    :param actions: the actions taken.
    :param log_prob_actions: the log-probabilities of the actions.
    :param advantages: the advantages.
    :param ppo_clip: the PPO clipped objective clipping epsilon.
    :return: the policy loss, the Kullback-Leibler divergence estimate & the entropy gain.
    """
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
    policy_loss: torch.Tensor, value_loss: torch.Tensor, entropy_bonus: torch.Tensor
) -> torch.Tensor:
    """
    final loss of clipped objective PPO, as seen here:
    https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L91

    :param policy_loss: the calculated policy loss.
    :param value_loss: the calculation value loss.
    :param entropy_bonus: the calculated entropy bonus.
    :return: the PPO objective, i.e. a linear combination of those losses & entropy bonus.
    """
    loss = (
        policy_loss
        + VALUE_COEFFICIENT * value_loss
        - ENTROPY_COEFFICIENT * entropy_bonus
    )

    return loss
