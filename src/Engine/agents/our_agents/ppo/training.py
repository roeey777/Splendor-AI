from typing import Tuple

import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import torch.distributions as distributions
from torch.nn.modules.loss import _Loss as Loss_Fn


ENTROPY_COEFFICIENT = 0.01
VALUE_COEFFICIENT = 0.5
VERY_SMALL_EPSILON = 1e-8

# Global Gradient Norm Clipping as suggested by (bullet #11):
# https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
MAX_GRADIENT_NORM = 1.0


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
        advantages = (advantages - advantages.mean()) / (advantages.std() + VERY_SMALL_EPSILON)

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
    kl_divergence_estimate = (log_prob_actions - new_log_prob_actions).mean().item()

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
    """
    loss = (
        policy_loss
        + VALUE_COEFFICIENT * value_loss
        - ENTROPY_COEFFICIENT * entropy_bonus
    )

    return loss


def train_single_episode(
    env: gym.Env,
    policy: nn.Module,
    optimizer: Optimizer,
    discount_factor: float,
    ppo_steps: int,
    ppo_clip: float,
    loss_fn: Loss_Fn,
    seed: int,
):
    policy.train()

    states = []
    actions = []
    action_mask_history = []
    log_prob_actions = []
    values = []
    rewards = []
    done = False
    episode_reward = 0

    state, info = env.reset(seed=seed)
    while not done:
        state = torch.tensor(state).double().unsqueeze(0)
        # append state here, not after we get the next state from env.step()
        states.append(state)
        action_mask = (
            torch.from_numpy(env.unwrapped.get_legal_actions_mask())
            .double()
            .unsqueeze(0)
        )
        action_prob, value_pred = policy(state, action_mask)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        _, reward, done, __, ___ = env.step(action.item())
        actions.append(action.unsqueeze(0))
        action_mask_history.append(action_mask)
        log_prob_actions.append(log_prob_action.unsqueeze(0))
        values.append(value_pred)
        rewards.append(reward)
        episode_reward += reward

    states = torch.cat(states)
    actions = torch.cat(actions)
    action_mask_history = torch.cat(action_mask_history)
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)

    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)

    policy_loss, value_loss = update_policy(
        policy,
        states,
        actions,
        action_mask_history,
        log_prob_actions,
        advantages,
        returns,
        optimizer,
        ppo_steps,
        ppo_clip,
        loss_fn,
    )

    return policy_loss, value_loss, episode_reward


def update_policy(
    policy: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    action_masks: torch.Tensor,
    log_prob_actions: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    optimizer: Optimizer,
    ppo_steps: int,
    ppo_clip: float,
    loss_fn: Loss_Fn,
):
    total_policy_loss = 0
    total_value_loss = 0

    advantages = advantages.detach()
    log_prob_actions = log_prob_actions.detach()
    actions = actions.detach()
    action_masks = action_masks.detach()

    for _ in range(ppo_steps):
        # get new log prob of actions for all input states
        action_prob, value_pred = policy(states, action_masks)
        value_pred = value_pred.squeeze(-1)

        policy_loss, kl_divergence_estimate, entropy = calculate_policy_loss(
            action_prob, actions, log_prob_actions, advantages, ppo_clip
        )

        value_loss = loss_fn(returns, value_pred).mean()

        loss = calculate_loss(policy_loss, value_loss, entropy)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        # clip gradient norm - limit the amount of change a single step can do.
        torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRADIENT_NORM)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()

    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps
