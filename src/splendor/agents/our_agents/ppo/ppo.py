import random
from csv import writer as csv_writer
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import List, Optional, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces.utils import flatdim
from numpy.typing import NDArray

# import this would register splendor as one of gym's environments.
import splendor.Splendor.gym
from splendor.Splendor.gym.envs.splendor_env import SplendorEnv
from splendor.Splendor.splendor_model import SplendorState

from .arguments_parsing import (
    DEFAULT_ARCHITECTURE,
    DEFAULT_OPPONENT,
    DEFAULT_TEST_OPPONENT,
    NN_ARCHITECTURES,
    OPPONENTS_AGENTS,
    WORKING_DIR,
    NeuralNetArch,
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
from .training import train_single_episode
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


def evaluate(
    env: gym.Env, policy: nn.Module, is_recurrent: bool, seed: int, device: torch.device
) -> float:
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
    device_name: str = "cpu",
    transfer_learning: bool = False,
    saved_weights: Optional[Path] = None,
    opponent: str = DEFAULT_OPPONENT,
    test_opponent: str = DEFAULT_TEST_OPPONENT,
    architecture: str = DEFAULT_ARCHITECTURE,
):
    device = torch.device(
        device_name if getattr(torch, device_name).is_available() else "cpu"
    )

    nn_arch: NeuralNetArch = NN_ARCHITECTURES[architecture]
    opponents = OPPONENTS_AGENTS[opponent]
    test_opponents = OPPONENTS_AGENTS[test_opponent]

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

    if transfer_learning and saved_weights is not None:
        policy = load_saved_model(saved_weights, nn_arch.ppo_factory)
    elif transfer_learning:
        policy = load_saved_model(nn_arch.default_saved_weights, nn_arch.ppo_factory)
    else:
        policy = nn_arch.ppo_factory(input_dim, output_dim).double().to(device)

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
                nn_arch.is_recurrent,
                nn_arch.hidden_state_dim,
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
    options = parse_args()
    train(**options.__dict__)


if __name__ == "__main__":
    main()
