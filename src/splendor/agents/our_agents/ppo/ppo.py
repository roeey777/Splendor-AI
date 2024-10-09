import random

from datetime import datetime
from argparse import ArgumentParser
from itertools import chain
from typing import List, Optional
from pathlib import Path
from csv import writer as csv_writer

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from splendor.Splendor.splendor_model import SplendorState
from splendor.version import get_version

# import this would register splendor as one of gym's environments.
import splendor.Splendor.gym

from splendor.agents.generic.random import myAgent as RandomAgent

from .network import PPO, DROPOUT
from .training import train_single_episode
from .utils import load_saved_ppo
from .constants import (
    SEED,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_EPISODES,
    DISCOUNT_FACTOR,
    N_TRIALS,
    PPO_STEPS,
    PPO_CLIP,
)

OPPONENTS_AGENTS = {
    "random": [RandomAgent(0)],
    "minimax": [MinMaxAgent(0)],
}
DEFAULT_OPPONENT = "random"
DEFAULT_TEST_OPPONENT = "minimax"
OPPONENTS_CHOICES = OPPONENTS_AGENTS.keys()

WORKING_DIR = Path().absolute()
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


def evaluate(env: gym.Env, policy: nn.Module, seed: int, device: torch.device) -> float:
    policy.eval().to(device)

    rewards = []
    done = False
    episode_reward = 0

    state, info = env.reset(seed=seed)
    hidden = policy.init_hidden_state().to(device)

    with torch.no_grad():
        while not done:
            state = torch.tensor(state, dtype=torch.float64).unsqueeze(0).to(device)

            action_mask = (
                torch.from_numpy(env.unwrapped.get_legal_actions_mask())
                .double()
                .to(device)
            )
            action_prob, _, hidden = policy(state, action_mask, hidden)

            action = torch.argmax(action_prob, dim=-1)
            next_state, reward, done, _, __ = env.step(action.item())
            episode_reward += reward
            state = next_state

    return episode_reward


def extract_game_stats(final_game_state: SplendorState, agent_id: int) -> List:
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


def train(
    working_dir: Path = WORKING_DIR,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    seed: int = SEED,
    device: str = "cpu",
    transfer_learning: bool = False,
    saved_weights: Optional[Path] = None,
    opponent: str = DEFAULT_OPPONENT,
    test_opponent: str = DEFAULT_TEST_OPPONENT,
):
    device = torch.device(device if getattr(torch, device).is_available() else "cpu")

    opponents = OPPONENTS_AGENTS[opponent]
    test_opponents = OPPONENTS_AGENTS[test_opponent]

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    start_time = datetime.now()
    folder = working_dir / start_time.strftime(FOLDER_FORMAT)
    models_folder = folder / "models"
    models_folder.mkdir(parents=True)
    train_env = gym.make("splendor-v1", agents=opponents)
    test_env = gym.make("splendor-v1", agents=test_opponents)

    input_dim = train_env.observation_space.shape[0]
    output_dim = train_env.action_space.n

    if transfer_learning:
        policy = load_saved_ppo(saved_weights)
    else:
        policy = PPO(input_dim, output_dim, dropout=DROPOUT).double().to(device)

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

        # Main training loop
        for episode in range(MAX_EPISODES):
            print(f"Episode {episode + 1}")
            policy_loss, value_loss, train_reward = train_single_episode(
                train_env,
                policy,
                optimizer,
                DISCOUNT_FACTOR,
                PPO_STEPS,
                PPO_CLIP,
                loss_function,
                seed,
                device,
            )
            test_reward = evaluate(test_env, policy, seed, device)

            stats = extract_game_stats(
                train_env.unwrapped.state, train_env.unwrapped.my_turn
            )
            stats.insert(0, episode)
            stats.extend([policy_loss, value_loss, train_reward, test_reward])
            stats_csv.writerow(stats)

            train_rewards.append(train_reward)
            test_rewards.append(test_reward)

            if (episode + 1) % N_TRIALS == 0:
                mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
                mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
                # train_rewards_std = np.std(train_rewards[-N_TRIALS:])
                # test_rewards_std = np.std(test_rewards[-N_TRIALS:])
                print(
                    f"| Episode: {episode + 1:3} | Mean Train Rewards: {mean_train_rewards:5.2f} | Mean Test Rewards: {mean_test_rewards:5.2f} |"
                )
                save_model(
                    policy,
                    models_folder / f"ppo_model_{episode + 1 // N_TRIALS}.pth",
                )


def main():
    parser = ArgumentParser(
        prog="ppo",
        description="Train a PPO agent.",
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        default=LEARNING_RATE,
        type=float,
        help="The learning rate to use during training with gradient descent",
    )
    parser.add_argument(
        "-d",
        "--weight-decay",
        default=WEIGHT_DECAY,
        type=float,
        help="The weight decay (L2 regularization) to use during training with gradient descent",
    )
    parser.add_argument(
        "-w",
        "--working-dir",
        default=WORKING_DIR,
        type=Path,
        help="Path to directory to work in (will create a directory with "
        "current timestamp for each run)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=SEED,
        type=int,
        help="Seed to set for numpy's, torch's and random's random number generators.",
    )
    parser.add_argument(
        "-t",
        "--transfer-learning",
        action="store_true",
        help="Learn from previosly learned model, i.e. trasfer learning from previos training sessions",
    )
    parser.add_argument(
        "--saved-weights",
        default=None,
        type=Path,
        help="Path to the weights to start from a new learning session (ignored if not in transfer-learning mode)",
    )
    parser.add_argument(
        "-o",
        "--opponent",
        type=str,
        default=DEFAULT_OPPONENT,
        choices=OPPONENTS_CHOICES,
        help="Against whom the PPO should train",
    )
    parser.add_argument(
        "--test-opponent",
        type=str,
        default=DEFAULT_TEST_OPPONENT,
        choices=OPPONENTS_CHOICES,
        help="Against whom the PPO should be evaluated",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        choices=("cuda", "cpu", "mps"),
        help="On which device to do heavy mathematical computation",
    )
    parser.add_argument("--version", action="version", version=get_version())

    options = parser.parse_args()
    train(**options.__dict__)


if __name__ == "__main__":
    main()
