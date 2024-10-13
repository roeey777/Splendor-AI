from typing import Optional, Tuple

import gymnasium as gym
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces.utils import flatdim
from torch.nn.modules.loss import _Loss as Loss_Fn
from torch.optim.optimizer import Optimizer

from .common import calculate_loss, calculate_policy_loss
from .constants import MAX_GRADIENT_NORM, ROLLOUT_BUFFER_SIZE
from .rollout import RolloutBuffer


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
    hidden_states_dim: Optional[int] = None,
):
    policy = policy.to(device)
    policy.train()

    rollout_buffer = RolloutBuffer(
        ROLLOUT_BUFFER_SIZE,
        flatdim(env.observation_space),
        flatdim(env.action_space),
        is_recurrent,
        hidden_states_dim,
    )

    done = False
    episode_reward = 0

    state, info = env.reset(seed=seed)
    if is_recurrent:
        hidden = policy.init_hidden_state().to(device)

    while not done:
        state = torch.tensor(state, dtype=torch.float64).unsqueeze(0).to(device)

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

        rollout_buffer.remember(
            state,
            action.unsqueeze(0),
            action_mask,
            log_prob_action.unsqueeze(0),
            value_pred,
            reward,
            done,
            hidden if is_recurrent else None,
        )

        state = next_state

        if is_recurrent:
            hidden = next_hidden

    policy_loss, value_loss = update_policy(
        policy,
        rollout_buffer,
        discount_factor,
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
    rollout_buffer: RolloutBuffer,
    discount_factor: float,
    optimizer: Optimizer,
    ppo_steps: int,
    ppo_clip: float,
    loss_fn: Loss_Fn,
    device: torch.device,
    is_recurrent: bool,
):
    total_policy_loss = 0
    total_value_loss = 0

    (
        hidden_states,
        states,
        actions,
        action_masks,
        log_prob_actions,
        advantages,
        returns,
        _,
    ) = rollout_buffer.unpack(discount_factor)

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
