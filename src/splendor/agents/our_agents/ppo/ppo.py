"""
entry-point for PPO training.
"""

import random
from csv import writer as csv_writer
from datetime import datetime
from importlib import import_module
from itertools import chain
from pathlib import Path
from typing import List, Optional, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import
import torch.optim as optim  # pylint: disable=consider-using-from-import
from gymnasium.spaces.utils import flatdim

# import this would register splendor as one of gym's environments.
import splendor.splendor.gym  # pylint: disable=unused-import
from splendor.splendor.gym.envs.splendor_env import SplendorEnv
from splendor.splendor.splendor_model import SplendorState

from .arguments_parsing import (
    DEFAULT_ARCHITECTURE,
    DEFAULT_OPPONENT,
    DEFAULT_TEST_OPPONENT,
    NN_ARCHITECTURES,
    OPPONENTS_AGENTS_FACTORY,
    WORKING_DIR,
    NeuralNetArch,
    OpponentsFactory,
    parse_args,
)
from .constants import (
    DISCOUNT_FACTOR,
    LEARNING_RATE,
    MAX_EPISODES,
    N_TRIALS,
    PPO_CLIP,
    PPO_STEPS,
    SEED,
    WEIGHT_DECAY,
)
from .ppo_agent_base import PPOAgentBase
from .training import LearningParams, evaluate, train_single_episode
from .utils import load_saved_model

FOLDER_FORMAT = "%y-%m-%d_%H-%M-%S"
STATS_FILE = "stats.csv"
STATS_HEADERS = (
    "episode",
    "players_count",
    "rounds_count",
    "score",
    "nobles_taken",
    "tier1_bought",
    "tier2_bought",
    "tier3_bought",
    "policy_loss",
    "value_loss",
    "train_reward",
    "test_reward",
)


def save_model(model: nn.Module, path: Path):
    """
    Save given model weights into a file at given path.

    :param model: the model whose weights should be stored.
    :param path: Where to store the weights.
    """
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "running_mean": (
                model.cpu().input_norm.running_mean
                if hasattr(model.cpu(), "input_norm")
                else model.cpu().running_mean
            ),
            "running_var": (
                model.cpu().input_norm.running_var
                if hasattr(model.cpu(), "input_norm")
                else model.cpu().running_var
            ),
        },
        str(path),
    )


def extract_game_stats(final_game_state: SplendorState, agent_id: int) -> List[float]:
    """
    Extract game statistics from the final (terminal) game state.

    :param final_game_state: the final, terminal state of the game.
    :param agent_id: the ID (turn) of the PPO agent in training.
    :return: list of the statistics values.
    """
    agent_state = final_game_state.agents[agent_id]
    stats = [
        len(final_game_state.agents),  # players_count
        len(agent_state.agent_trace.action_reward),  # "rounds_count",
        agent_state.score,  # "score",
        len(agent_state.nobles),  # "nobles_taken",
        len(list(filter(lambda c: c.deck_id == 0, chain(*agent_state.cards.values())))),
        len(list(filter(lambda c: c.deck_id == 1, chain(*agent_state.cards.values())))),
        len(list(filter(lambda c: c.deck_id == 2, chain(*agent_state.cards.values())))),
    ]
    return stats


# pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements,too-many-positional-arguments
def train(
    working_dir: Path = WORKING_DIR,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    seed: int = SEED,
    device_name: str = "cpu",
    transfer_learning: bool = False,
    saved_weights: Optional[Path] = None,
    opponent: str = DEFAULT_OPPONENT,
    test_opponent: str = DEFAULT_TEST_OPPONENT,
    architecture: str = DEFAULT_ARCHITECTURE,
) -> nn.Module:
    """
    Train a PPO agent.

    :param working_dir: Where to store the statistics and weights.
    :param learning_rate: The learning rate of the gradient descent based learning.
    :param weight_decay: L2 regularization coefficient.
    :param seed: Which seed to use during training.
    :param device_name: Name of the device used for mathematical computations.
    :param transfer_learning: Whether or not the PPO agent should be initialized from
                              a pre-trained weights.
    :param saved_weights: Path to the weights of a pre-trained PPO agent that would be
                          loaded and used as initialization for this training session.
                          This argument would be ignored if ``transfer_learning`` is ``False``
                          and required if ``transfer_learning`` is ``True``.
    :param opponent: Opponent agent name that the PPO would train against.
    :param test_opponent: Test opponent name that the PPO would be evaluated against.
    :param architecture: PPO network architecture name that should be used.
    :return: The trained model (PPO agent).
    """
    device = torch.device(
        device_name if getattr(torch, device_name).is_available() else "cpu"
    )

    nn_arch: NeuralNetArch = NN_ARCHITECTURES[architecture]

    load_net_later = False
    if opponent in OPPONENTS_AGENTS_FACTORY:
        opponents_factory: OpponentsFactory = OPPONENTS_AGENTS_FACTORY[opponent]
        opponents = opponents_factory(0)
    else:
        # assume that the PPO is meant to train against itself.
        m = import_module(nn_arch.agent_relative_import_path, package=__package__)
        opponents = [m.myAgent(0, load_net=False)]
        load_net_later = True

    load_test_net_later = False
    if test_opponent in OPPONENTS_AGENTS_FACTORY:
        test_opponents_factory: OpponentsFactory = OPPONENTS_AGENTS_FACTORY[
            test_opponent
        ]
        test_opponents = test_opponents_factory(0)
    else:
        # assume that the PPO is meant to be evaluated against itself.
        m = import_module(nn_arch.agent_relative_import_path, package=__package__)
        test_opponents = [m.myAgent(0, load_net=False)]
        load_test_net_later = True

    print(
        f"Training PPO (arch: {nn_arch.name}) against opponent: {opponent}"
        f" and evaluating against test opponent: {test_opponent}"
    )

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    start_time = datetime.now()
    folder = working_dir / f"{start_time.strftime(FOLDER_FORMAT)}__arch_{nn_arch.name}"
    models_folder = folder / "models"
    models_folder.mkdir(parents=True)
    train_env: gym.Env = gym.make("splendor-v1", agents=opponents)
    test_env: gym.Env = gym.make("splendor-v1", agents=test_opponents)
    custom_train_env = cast(SplendorEnv, train_env.unwrapped)

    input_dim = flatdim(train_env.observation_space)
    output_dim = flatdim(train_env.action_space)

    if transfer_learning:
        print("Using pre-trained weights as initialization")
        if saved_weights is not None:
            policy = load_saved_model(saved_weights, nn_arch.ppo_factory)
        else:
            policy = load_saved_model(
                nn_arch.default_saved_weights, nn_arch.ppo_factory
            )
    else:
        policy = nn_arch.ppo_factory(input_dim, output_dim).double().to(device)

    if load_net_later:
        for o in opponents:
            ppo_opponent = cast(PPOAgentBase, o)
            ppo_opponent.load_policy(policy)

    if load_test_net_later:
        for o in test_opponents:
            ppo_test_opponent = cast(PPOAgentBase, o)
            ppo_test_opponent.load_policy(policy)

    optimizer = optim.Adam(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_function = nn.SmoothL1Loss()

    train_rewards = []
    test_rewards = []

    with open(folder / STATS_FILE, "w", newline="\n", encoding="ascii") as stats_file:
        stats_csv = csv_writer(stats_file)
        stats_csv.writerow(STATS_HEADERS)

        learning_params = LearningParams(
            optimizer,
            DISCOUNT_FACTOR,
            PPO_STEPS,
            PPO_CLIP,
            loss_function,
            seed,
            device,
            nn_arch.is_recurrent,
            nn_arch.hidden_state_dim,
        )

        # Main training loop
        for episode in range(MAX_EPISODES):
            print(f"Episode {episode + 1}")
            policy_loss, value_loss, train_reward = train_single_episode(
                train_env,
                policy,
                learning_params,
            )
            test_reward = evaluate(test_env, policy, nn_arch.is_recurrent, seed, device)

            stats = extract_game_stats(custom_train_env.state, custom_train_env.my_turn)
            stats.insert(0, episode)
            stats.extend([policy_loss, value_loss, train_reward, test_reward])
            stats_csv.writerow(stats)

            train_rewards.append(train_reward)
            test_rewards.append(test_reward)

            if (episode + 1) % N_TRIALS == 0:
                mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
                mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
                print(
                    f"| Episode: {episode + 1:3} | Mean Train Rewards: {mean_train_rewards:5.2f} | "
                    f"Mean Test Rewards: {mean_test_rewards:5.2f} |"
                )
                save_model(
                    policy,
                    models_folder / f"ppo_model_{episode + 1 // N_TRIALS}.pth",
                )

    return policy


def main():
    """
    Entry-point for the ``ppo`` console script.
    """
    options = parse_args()
    train(**options)


if __name__ == "__main__":
    main()
