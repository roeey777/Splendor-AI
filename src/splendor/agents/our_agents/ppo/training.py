from typing import Tuple

import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import torch.distributions as distributions
from torch.nn.modules.loss import _Loss as Loss_Fn

from .constants import MAX_GRADIENT_NORM

from .common import (
    calculate_returns,
    calculate_advantages,
    calculate_policy_loss,
    calculate_loss,
)


def train_single_episode(
    env: gym.Env,
    policy: nn.Module,
    optimizer: Optimizer,
    discount_factor: float,
    ppo_steps: int,
    ppo_clip: float,
    loss_fn: Loss_Fn,
    seed: int,
    device: torch.device,
    is_recurrent: bool,
):
    policy = policy.to(device)
    policy.train()

    hidden_states = []
    states = []
    actions = []
    action_mask_history = []
    log_prob_actions = []
    values = []
    rewards = []
    done = False
    episode_reward = 0

    state, info = env.reset(seed=seed)
    if is_recurrent:
        hidden = policy.init_hidden_state().to(device)

    while not done:
        state = torch.tensor(state, dtype=torch.float64).unsqueeze(0).to(device)
        # append state here, not after we get the next state from env.step()
        states.append(state)

        if is_recurrent:
            hidden_states.append(hidden)

        action_mask = (
            torch.from_numpy(env.unwrapped.get_legal_actions_mask())
            .double()
            .unsqueeze(0)
            .to(device)
        )

        if is_recurrent:
            action_prob, value_pred, next_hidden = policy(state, action_mask, hidden)
        else:
            action_prob, value_pred = policy(state, action_mask)

        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        next_state, reward, done, _, __ = env.step(action.detach().cpu().item())
        actions.append(action.unsqueeze(0))
        action_mask_history.append(action_mask)
        log_prob_actions.append(log_prob_action.unsqueeze(0))
        values.append(value_pred)
        rewards.append(reward)
        episode_reward += reward
        state = next_state
        if is_recurrent:
            hidden = next_hidden

    if is_recurrent:
        hidden_states = torch.cat(hidden_states).to(device)

    states = torch.cat(states).to(device)
    actions = torch.cat(actions).to(device)
    action_mask_history = torch.cat(action_mask_history).to(device)
    log_prob_actions = torch.cat(log_prob_actions).to(device)
    values = torch.cat(values).squeeze(-1).to(device)

    returns = calculate_returns(rewards, discount_factor).to(device)
    advantages = calculate_advantages(returns, values).to(device)

    policy_loss, value_loss = update_policy(
        policy,
        hidden_states,
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
        device,
        is_recurrent,
    )

    return policy_loss, value_loss, episode_reward


def update_policy(
    policy: nn.Module,
    hidden_states: torch.Tensor,
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
    device: torch.device,
    is_recurrent: bool,
):
    total_policy_loss = 0
    total_value_loss = 0

    if is_recurrent:
        hidden_states = hidden_states.detach()

    advantages = advantages.detach()
    log_prob_actions = log_prob_actions.detach()
    actions = actions.detach()
    action_masks = action_masks.detach()

    for _ in range(ppo_steps):
        # get new log prob of actions for all input states
        if is_recurrent:
            action_prob, value_pred, next_hidden_states = policy(
                states, action_masks, hidden_states
            )
            next_hidden_states.detach()
            value_pred = value_pred.squeeze(-1)
        else:
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

        total_policy_loss += policy_loss.detach().cpu().item()
        total_value_loss += value_loss.detach().cpu().item()

    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps
