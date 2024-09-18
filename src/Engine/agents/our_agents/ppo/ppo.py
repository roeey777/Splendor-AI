import random

from datetime import datetime
from argparse import ArgumentParser
from itertools import chain
from typing import List, Dict, Tuple
from pathlib import Path

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from csv import writer as csv_writer

from Engine.Splendor.splendor_model import SplendorState
from Engine.agents.our_agents.minmax import myAgent as MiniMaxAgent

from .network import PPO, DROPOUT
from .training import train_single_episode

# import this would register splendor as one of gym's environments.
import Engine.Splendor.gym

opponents = [MiniMaxAgent(0)]

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

SEED = 1234
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
MAX_EPISODES = 50000

DISCOUNT_FACTOR = 0.99
N_TRIALS = 10
PPO_STEPS = 10
PPO_CLIP = 0.2


def save_model(model: nn.Module, path: Path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "running_mean": (
                model.input_norm.running_mean
                if hasattr(model, "input_norm")
                else model.running_mean
            ),
            "running_var": (
                model.input_norm.running_var
                if hasattr(model, "input_norm")
                else model.running_var
            ),
        },
        str(path),
    )


def evaluate(env, policy, seed):
    policy.eval()

    rewards = []
    done = False
    episode_reward = 0

    state, info = env.reset(seed=seed)

    while not done:
        state = torch.tensor(state, dtype=torch.float64).unsqueeze(0)

        with torch.no_grad():
            action_mask = torch.from_numpy(
                env.unwrapped.get_legal_actions_mask()
            ).double()
            action_prob, _ = policy(state, action_mask)

        action = torch.argmax(action_prob, dim=-1)
        next_state, reward, done, _, __ = env.step(action.item())
        episode_reward += reward
        state = next_state

    return episode_reward


def extract_game_stats(final_game_state: SplendorState, agent_id) -> List:
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
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    start_time = datetime.now()
    folder = working_dir / start_time.strftime(FOLDER_FORMAT)
    models_folder = folder / "models"
    models_folder.mkdir(parents=True)
    train_env = gym.make("splendor-v1", agents=opponents)
    test_env = gym.make("splendor-v1", agents=opponents)

    input_dim = train_env.observation_space.shape[0]
    output_dim = train_env.action_space.n

    policy = PPO(input_dim, output_dim, dropout=DROPOUT).double()

    optimizer = optim.Adam(
        policy.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_function = nn.SmoothL1Loss()

    train_rewards = []
    test_rewards = []

    with open(folder / STATS_FILE, "w", newline="\n") as stats_file:
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
            )
            test_reward = evaluate(test_env, policy, seed)

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
                train_rewards_std = np.std(train_rewards[-N_TRIALS:])
                test_rewards_std = np.std(test_rewards[-N_TRIALS:])
                print(
                    f"| Episode: {episode + 1:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |"
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

    options = parser.parse_args()
    train(**options.__dict__)


if __name__ == "__main__":
    main()
