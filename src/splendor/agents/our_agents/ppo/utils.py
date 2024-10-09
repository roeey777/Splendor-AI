from typing import Optional
from pathlib import Path

import torch
import gymnasium as gym

from gymnasium.spaces.utils import flatdim

from .ppo_base import PPOBase, PPOBaseFactory
from .network import PPO, DROPOUT


DEFAULT_SAVED_PPO_PATH = Path(__file__).parent / "ppo_model.pth"


def load_saved_model(
        path: Path,
        ppo_factory: PPOBaseFactory,
        *args,
        **kwargs,
    ) -> PPOBase:
    """
    Load saved weights of a PPO model from a given path, if no path was given
    the installed weights of the PPO agent will be loaded.
    """
    env = gym.make("splendor-v1", agents=[])

    # load_weights
    net = ppo_factory(
        flatdim(env.observation_space), flatdim(env.action_space), *args, **kwargs
    ).double()
    checkpoint = torch.load(
        str(path),
        weights_only=False,
        map_location="cpu",
    )

    net.load_state_dict(checkpoint["model_state_dict"])
    if hasattr(net, "input_norm"):
        # both running_mean & running_var are stored as (1, flatdim(env.observation_space))
        # rather than (flatdim(env.observation_space),)
        net.input_norm.running_mean = checkpoint["running_mean"].squeeze(0)
        net.input_norm.running_var = checkpoint["running_var"].squeeze(0)
    else:
        net.running_mean = checkpoint["running_mean"]
        net.running_var = checkpoint["running_var"]

    return net


def load_saved_ppo(path: Optional[Path] = None) -> PPO:
    """
    Load saved weights of a PPO model from a given path, if no path was given
    the installed weights of the PPO agent will be loaded.
    """
    if path is None:
        path = DEFAULT_SAVED_PPO_PATH

    return load_saved_model(path, PPO, dropout=DROPOUT)
