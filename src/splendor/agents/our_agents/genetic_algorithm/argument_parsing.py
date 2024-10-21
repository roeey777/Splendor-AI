"""
All things related to command-line arguments for evolving
the genetic algorithm agent.
"""

import argparse
from argparse import ArgumentParser
from pathlib import Path
from typing import Required, TypedDict, cast

from splendor.version import get_version

from .constants import GENERATIONS, MUTATION_RATE, POPULATION_SIZE, WORKING_DIR


class Arguments(TypedDict):
    """
    TypedDict representing the command-line arguments.
    """

    population_size: Required[int]
    generations: Required[int]
    mutation_rate: Required[float]
    working_dir: Required[Path]
    seed: Required[int]
    multiprocess: Required[bool]
    quiet: Required[bool]


def parse_args() -> Arguments:
    """
    parse the command-line arguments.
    """
    parser = ArgumentParser(
        prog="evolve",
        description="Evolves a Splendor agent using genetic algorithm.",
    )
    parser.add_argument(
        "-p",
        "--population-size",
        default=POPULATION_SIZE,
        type=int,
        help="Size of the population (should be multiple of 12)",
    )
    parser.add_argument(
        "-g",
        "--generations",
        default=GENERATIONS,
        type=int,
        help="Amount of generations",
    )
    parser.add_argument(
        "-m",
        "--mutation-rate",
        default=MUTATION_RATE,
        type=float,
        help="Probability to mutate (should be in the range [0,1])",
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
        type=int,
        help="Seed to set for numpy's random number generator",
    )
    parser.add_argument(
        "--multiprocess",
        action="store_true",
        help="Use multiprocessing to evolve faster",
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--version", action="version", version=get_version())

    options: argparse.Namespace = parser.parse_args()
    if options.population_size <= 0 or (options.population_size % 12):
        raise ValueError(
            "To work properly, population size should be a "
            f"multiple of 12 (not {options.population_size})"
        )
    if options.generations <= 0:
        raise ValueError(f"Invalid amount of generations {options.generations}")
    if not 0 <= options.mutation_rate <= 1:
        raise ValueError(f"Invalid mutation rate value {options.mutation_rate}")

    return cast(Arguments, vars(options))
