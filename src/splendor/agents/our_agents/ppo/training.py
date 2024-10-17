"""
Implementation of the actual training of the PPO.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, cast

import gymnasium as gym
import torch
import torch.distributions as distributions  # pylint: disable=consider-using-from-import
import torch.nn as nn  # pylint: disable=consider-using-from-import
from gymnasium.spaces.utils import flatdim
from numpy.typing import NDArray
from torch.nn.modules.loss import _Loss as Loss_Fn
from torch.optim.optimizer import Optimizer

from splendor.splendor.gym.envs.splendor_env import SplendorEnv

from .common import calculate_loss, calculate_policy_loss
from .constants import MAX_GRADIENT_NORM, ROLLOUT_BUFFER_SIZE
from .rollout import RolloutBuffer


@dataclass
class LearningParams:
    """
    Placeholder for varius learning parameters.

    discount_factor: by how much the reward decays over environment
    steps (turns in the game).
    optimizer: Which optimizer should be used (Adam, SGD, etc).
    ppo_steps: how many gradient descent steps should be performed.
    ppo_clip: which "epsilon" to use the policy loss clipping.
    loss_fn: Which loss function should be used as the loss of
    he value-regression (L1, L2, Huber, etc).
    device: On which device to execute the calculations.
    is_recurrent: Is the given policy incorporates a recurrent unit in it's architecture.
    This parameter is here to tell if the hidden states should be ignored or not.
    """

    # pylint: disable=too-many-instance-attributes

    optimizer: Optimizer
    discount_factor: float
    ppo_steps: int
    ppo_clip: float
    loss_fn: Loss_Fn
    seed: int
    device: torch.device
    is_recurrent: bool
    hidden_states_dim: Optional[int] = None


def train_single_episode(
    env: gym.Env,
    policy: nn.Module,
    learning_params: LearningParams,
) -> Tuple[float, float, float]:
    # pylint: disable=too-many-locals
    """
    Execute the training procedure for a single episode (game), i.e. record
    a complete episode trajectory (trace of a full game) and then perform multiple
    gradient descent steps on the policy network.

    :param env: The environment that would be used to simulate an episode.
    :param policy: The network of the PPO agent.
    :param learning_params: Varios learning parameters required to define
                            the learning procedure, such as the learning rate.
    :return: the average policy & value losses and the episode reward.
    """
    policy = policy.to(learning_params.device)
    policy.train()

    custom_env = cast(SplendorEnv, env.unwrapped)

    rollout_buffer = RolloutBuffer(
        ROLLOUT_BUFFER_SIZE,
        flatdim(custom_env.observation_space),
        flatdim(custom_env.action_space),
        learning_params.is_recurrent,
        learning_params.hidden_states_dim,
    )

    done = False
    episode_reward: float = 0

    state_vector: NDArray
    state_vector, _ = custom_env.reset(seed=learning_params.seed)
    state = (
        torch.from_numpy(state_vector).double().unsqueeze(0).to(learning_params.device)
    )

    if learning_params.is_recurrent:
        hidden = policy.init_hidden_state().to(learning_params.device)

    while not done:
        action_mask = (
            torch.from_numpy(custom_env.get_legal_actions_mask())
            .double()
            .unsqueeze(0)
            .to(learning_params.device)
        )

        if learning_params.is_recurrent:
            action_prob, value_pred, *next_hidden = policy(state, action_mask, hidden)
        else:
            action_prob, value_pred = policy(state, action_mask)

        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        with torch.no_grad():
            next_state, reward, done, _, __ = custom_env.step(
                action.detach().cpu().item()
            )

        episode_reward += reward

        rollout_buffer.remember(
            state,
            action.unsqueeze(0),
            action_mask,
            log_prob_action.unsqueeze(0),
            value_pred,
            reward,
            done,
            hidden if learning_params.is_recurrent else None,
        )

        state = (
            torch.from_numpy(next_state)
            .double()
            .unsqueeze(0)
            .to(learning_params.device)
        )

        if learning_params.is_recurrent:
            hidden = next_hidden

    policy_loss, value_loss = update_policy(policy, rollout_buffer, learning_params)

    return policy_loss, value_loss, episode_reward


def update_policy(
    policy: nn.Module,
    rollout_buffer: RolloutBuffer,
    learning_params: LearningParams,
) -> Tuple[float, float]:
    # pylint: disable=too-many-locals
    """
    Update the policy using several gradient descent steps (via the given optimizer)
    on the PPO-Clip loss function.

    :param policy: the neutal network to optimize.
    :param rollout_buffer: a record for a complete trajectory of an episode (trace of a full game).
    :param learning_params: all argument required in order to define the learning procedure.
    :return: The average policy loss and the average value loss.
    """
    total_policy_loss: float = 0
    total_value_loss: float = 0

    (
        hidden_states,
        states,
        actions,
        action_masks,
        log_prob_actions,
        advantages,
        returns,
        _,
    ) = rollout_buffer.unpack(learning_params.discount_factor)

    for _ in range(learning_params.ppo_steps):
        # get new log prob of actions for all input states
        if learning_params.is_recurrent:
            action_prob, value_pred, next_hidden_states = policy(
                states, action_masks, hidden_states
            )
            next_hidden_states.detach()
            value_pred = value_pred.squeeze(-1)
        else:
            action_prob, value_pred = policy(states, action_masks)
            value_pred = value_pred.squeeze(-1)

        policy_loss, kl_divergence_estimate, entropy = calculate_policy_loss(
            action_prob, actions, log_prob_actions, advantages, learning_params.ppo_clip
        )

        value_loss = learning_params.loss_fn(returns, value_pred).mean()

        loss = calculate_loss(policy_loss, value_loss, entropy)

        learning_params.optimizer.zero_grad()
        loss.backward(retain_graph=True)

        # clip gradient norm - limit the amount of change a single step can do.
        torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRADIENT_NORM)
        learning_params.optimizer.step()

        total_policy_loss += policy_loss.detach().cpu().item()
        total_value_loss += value_loss.detach().cpu().item()

        # kl_divergence_estimate is unused.
        _ = kl_divergence_estimate

    return (
        total_policy_loss / learning_params.ppo_steps,
        total_value_loss / learning_params.ppo_steps,
    )


def evaluate(
    env: gym.Env, policy: nn.Module, is_recurrent: bool, seed: int, device: torch.device
) -> float:
    # pylint: disable=too-many-locals
    """
    Evaluate the PPO agent (in training) performence against the test opponent.

    :param env: The test environment, configured to simulate a game against the test opponent.
    :param policy: The network of the PPO agent.
    :param is_recurrent: Is the network of the PPO agent incorporates a recurrent unit or not.
                         This signals this functions whether or not the hidden states should be
                         ignored or used.
    :param seed: the seed used by the environment.
    :param device: On which device the mathematical computations should be performed.
    :return: The reward the PPO agent collected during a single episode.
    """
    policy.eval().to(device)

    custom_env = cast(SplendorEnv, env.unwrapped)

    done = False
    episode_reward: float = 0

    state_vector: NDArray
    state_vector, _ = custom_env.reset(seed=seed)
    state = torch.from_numpy(state_vector).double().unsqueeze(0).to(device)

    if is_recurrent:
        hidden = policy.init_hidden_state().to(device)

    with torch.no_grad():
        while not done:
            action_mask = (
                torch.from_numpy(custom_env.get_legal_actions_mask())
                .double()
                .to(device)
            )

            if is_recurrent:
                action_prob, _, hidden = policy(state, action_mask, hidden)
            else:
                action_prob, _ = policy(state, action_mask)

            action = torch.argmax(action_prob, dim=-1)
            next_state, reward, done, _, __ = custom_env.step(int(action.item()))
            episode_reward += reward
            state = torch.from_numpy(next_state).double().unsqueeze(0).to(device)

    return episode_reward
