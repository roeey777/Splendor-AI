"""
All things related to command-line arguments parsing for PPO training.
"""

import argparse
import typing
from argparse import ArgumentParser
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Literal, Required, TypedDict, cast

from splendor.agents.generic.random import myAgent as RandomAgent
from splendor.agents.our_agents.minmax import myAgent as MinMaxAgent
from splendor.template import Agent
from splendor.version import get_version

from .constants import LEARNING_RATE, SEED, WEIGHT_DECAY

# vanilla PPO
from .network import PPO
from .ppo_agent import DEFAULT_SAVED_PPO_PATH
from .ppo_base import PPOBaseFactory

# recurrent PPO with GRU
from .ppo_rnn.gru.constants import HIDDEN_STATE_SHAPE as GRU_HIDDEN_STATE_SHAPE
from .ppo_rnn.gru.network import PpoGru
from .ppo_rnn.gru.ppo_agent import DEFAULT_SAVED_PPO_GRU_PATH
from .ppo_rnn.lstm.constants import HIDDEN_STATE_SHAPE as LSTM_HIDDEN_STATE_SHAPE

# recurrent PPO with LSTM
from .ppo_rnn.lstm.network import PpoLstm
from .ppo_rnn.lstm.ppo_agent import DEFAULT_SAVED_PPO_LSTM_PATH

# PPO with self-attention
from .self_attn.network import PPOSelfAttention
from .self_attn.ppo_agent import DEFAULT_SAVED_PPO_SELF_ATTENTION_PATH


@dataclass
class NeuralNetArch:
    """
    dataclass for storing all essential information regarding a specific
    neural network architecture.
    """

    name: str
    ppo_factory: PPOBaseFactory
    is_recurrent: bool
    default_saved_weights: Path
    agent_relative_import_path: str
    hidden_state_dim: tuple[int, ...] | None = None


OpponentsFactory = Callable[[int], list[Agent]]


NN_ARCHITECTURES = {
    "mlp": NeuralNetArch("ppo_mlp", PPO, False, DEFAULT_SAVED_PPO_PATH, ".ppo_agent"),
    "gru": NeuralNetArch(
        "ppo_gru",
        PpoGru,
        True,
        DEFAULT_SAVED_PPO_GRU_PATH,
        ".ppo_rnn.gru.ppo_agent",
        GRU_HIDDEN_STATE_SHAPE,
    ),
    "self_attn": NeuralNetArch(
        "ppo_self_attn",
        PPOSelfAttention,
        False,
        DEFAULT_SAVED_PPO_SELF_ATTENTION_PATH,
        ".self_attn.ppo_agent",
    ),
    "lstm": NeuralNetArch(
        "ppo_lstm",
        PpoLstm,
        True,
        DEFAULT_SAVED_PPO_LSTM_PATH,
        ".ppo_rnn.lstm.ppo_agent",
        LSTM_HIDDEN_STATE_SHAPE,
    ),
}
NN_ARCHITECTURES_CHOICES = NN_ARCHITECTURES.keys()
DEFAULT_ARCHITECTURE = "mlp"

NN_OPPONENTS_AGENTS_FACTORY: dict[str, OpponentsFactory] = {
    name: partial(
        lambda agent_id, nn_arch: [
            import_module(
                nn_arch.agent_relative_import_path, package=__package__
            ).myAgent(agent_id)
        ],
        nn_arch=arch,
    )
    for name, arch in NN_ARCHITECTURES.items()
}

OPPONENTS_AGENTS_FACTORY: dict[str, OpponentsFactory] = {
    "random": lambda agent_id: [RandomAgent(agent_id)],
    "minimax": lambda agent_id: [MinMaxAgent(agent_id)],
}
OPPONENTS_AGENTS_FACTORY.update(NN_OPPONENTS_AGENTS_FACTORY)
DEFAULT_OPPONENT = "random"
DEFAULT_TEST_OPPONENT = "minimax"
SELF_OPPONENT = "itself"
OPPONENTS_CHOICES = (*OPPONENTS_AGENTS_FACTORY.keys(), SELF_OPPONENT)

WORKING_DIR = Path().absolute()

DeviceName = Literal["cuda", "cpu", "mps"]
DEVICE_NAME_CHOICES = typing.get_args(DeviceName)


class Arguments(TypedDict):
    """
    TypedDict representing the command-line arguments.
    """

    learning_rate: Required[float]
    weight_decay: Required[float]
    working_dir: Required[Path]
    seed: Required[int]
    saved_weights: Required[Path | None]
    opponent: Required[str]
    test_opponent: Required[str]
    device_name: Required[DeviceName]
    architecture: Required[str]


def parse_args() -> Arguments:
    """
    Parse command-line arguments.

    :return: dictionary storing all the required arguments.
    """
    parser = ArgumentParser(
        prog="ppo",
        description="Train a PPO agent.",
    )
    parser.add_argument("--version", action="version", version=get_version())
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
        help="Learn from previously learned model, "
        "i.e. transfer learning from previous training sessions",
    )
    parser.add_argument(
        "--saved-weights",
        default=None,
        type=Path,
        help="Path to the weights to start from a new learning session "
        "(ignored if not in transfer-learning mode)",
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
        choices=DEVICE_NAME_CHOICES,
        dest="device_name",
        help="On which device to do heavy mathematical computation",
    )
    parser.add_argument(
        "-a",
        "--architecture",
        "--arch",
        type=str,
        default=DEFAULT_ARCHITECTURE,
        choices=NN_ARCHITECTURES_CHOICES,
        help="What type of architecture of the neural network should be used",
    )

    options: argparse.Namespace = parser.parse_args()

    return cast(Arguments, vars(options))
