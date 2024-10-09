from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from splendor.version import get_version

from splendor.agents.generic.random import myAgent as RandomAgent
from splendor.agents.our_agents.minmax import myAgent as MinMaxAgent

from .ppo_rnn.gru.network import PPO_GRU
from .ppo_rnn.gru.ppo_agent import DEFAULT_SAVED_PPO_GRU_PATH
from .network import PPO
from .ppo_agent import DEFAULT_SAVED_PPO_PATH
from .ppo_base import PPOBaseFactory
from .constants import (
    SEED,
    LEARNING_RATE,
    WEIGHT_DECAY,
)


@dataclass
class NeuralNetArch:
    name: str
    ppo_factory: PPOBaseFactory
    is_recurrent: bool
    default_saved_weights: Path


OPPONENTS_AGENTS = {
    "random": [RandomAgent(0)],
    "minimax": [MinMaxAgent(0)],
}
DEFAULT_OPPONENT = "random"
DEFAULT_TEST_OPPONENT = "minimax"
OPPONENTS_CHOICES = OPPONENTS_AGENTS.keys()

NN_ARCHITECTURES = {
    "mlp": NeuralNetArch("ppo_mlp", PPO, False, DEFAULT_SAVED_PPO_PATH),
    "gru": NeuralNetArch("ppo_gru", PPO_GRU, True, DEFAULT_SAVED_PPO_GRU_PATH),
}
NN_ARCHITECTURES_CHOICES = NN_ARCHITECTURES.keys()
DEFAULT_ARCHITECTURE = "mlp"

WORKING_DIR = Path().absolute()


def parse_args():
    """
    Parse command-line arguments.
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
    parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        default=DEFAULT_ARCHITECTURE,
        choices=NN_ARCHITECTURES_CHOICES,
        help="What type of architecture of the neural network should be used",
    )

    options = parser.parse_args()

    return options
