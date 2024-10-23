"""
Defining how to represent agents as genes (i.e. vectors) so they
could be optimized with a genetic algorithm.
"""

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from splendor.splendor.features import METRICS_SHAPE, build_array

GeneTypeVar = TypeVar("GeneTypeVar", bound="Gene")
Mutator = Callable[[float], float]


class Gene:
    """
    A generic gene representation class.
    """

    LOWER_BOUND = -20
    UPPER_BOUND = 20
    SHAPE: tuple[int, ...] | None = None

    def __init__(self, dna: NDArray) -> None:
        assert dna.shape == self.SHAPE, "Given DNA has the wrong shape"
        self._dna = dna
        self._prepared_dna: NDArray | None = None

    @property
    def dna(self) -> NDArray:
        """
        We want some metrics to have the same weight (in places the is no
        meaning to the order). This methods returns a dna which matches in
        dimensions to the metrics we get in practice by repeating some value
        multiple time (according to the instructions of `METRICS_SHAPE`).
        """
        if self._prepared_dna is None:
            self._prepared_dna = build_array(self._dna, METRICS_SHAPE)

        return self._prepared_dna

    @property
    def raw_dna(self) -> NDArray:
        """
        Return a direct access to the private _dna attribute.
        This is useful when implementing the crossover functionality.
        """
        return self._dna

    @classmethod
    def random(cls: type[GeneTypeVar]) -> GeneTypeVar:
        """
        Initiate a gene with random DNA.
        """
        return cls(np.random.uniform(cls.LOWER_BOUND, cls.UPPER_BOUND, cls.SHAPE))

    @classmethod
    def load(cls: type[GeneTypeVar], path_or_file: Path | str) -> GeneTypeVar:
        """
        Initiate a gene with DNA from a saved file.
        """
        return cls(np.load(path_or_file))

    def save(self, path_or_file: Path | str) -> None:
        """
        Saves a gene's DNA to a file.
        """
        np.save(path_or_file, self._dna)

    def mutate(self, mutate_rate: float, mutator: Mutator) -> None:
        """
        Mutates a gene's DNA (in place).
        Should be the only thing that edits the DNA of an existing gene.
        """
        # this assertion is only for mypy.
        assert self.SHAPE is not None

        for pos in np.ndindex(self.SHAPE):
            if np.random.rand() < mutate_rate:
                self._dna[pos] = mutator(self._dna[pos])

        self._prepared_dna = None


class StrategyGene(Gene):
    """
    Represent a gene that helps to choose an action each turn.
    """

    SHAPE = (len(METRICS_SHAPE),)

    def evaluate_state(self, state_metrics: NDArray) -> float:
        """
        Evaluates a state's value according to its metrics.
        """
        return np.matmul(state_metrics, self.dna)


class ManagerGene(Gene):
    """
    Represent a gene that helps to choose a strategy to follow (from 3 options).
    """

    SHAPE = (len(METRICS_SHAPE), 3)

    def select_strategy(
        self, state_metrics: NDArray, strategies: tuple[StrategyGene, ...]
    ) -> StrategyGene:
        """
        Select which strategy should be used based on given state.

        :param state_metrics: the feature vector extracted from the state.
        :param strategies: all the available strategies.
        :return: the selected strategy.
        """
        output = np.matmul(state_metrics, self.dna)
        assert len(output) == len(strategies), f"Mismatching lengths ({output.shape})"
        # might replace this with output.argmax()
        index = max(range(len(output)), key=lambda i: output[i])

        return strategies[index]
