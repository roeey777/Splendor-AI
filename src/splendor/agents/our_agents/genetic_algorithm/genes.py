from splendor.Splendor.features import METRICS_SHAPE, build_array
import numpy as np


class Gene:
    """
    A generic gene representation class.
    """

    LOWER_BOUND = -20
    UPPER_BOUND = 20
    SHAPE = None

    def __init__(self, dna: np.array):
        assert dna.shape == self.SHAPE, "Given DNA has the wrong shape"
        self._dna = dna
        self._prepared_dna = None

    @property
    def dna(self):
        """
        We want some metrics to have the same weight (in places the is no
        meaning to the order). This methods returns a dna which matches in
        dimentions to the metrics we get in practice by repeting some value
        multiple time (according to the instructions of `METRICS_SHAPE`).
        """
        if self._prepared_dna is None:
            self._prepared_dna = build_array(self._dna, METRICS_SHAPE)

        return self._prepared_dna

    @classmethod
    def random(cls):
        """
        Initiate a gene with random DNA.
        """
        return cls(np.random.uniform(cls.LOWER_BOUND, cls.UPPER_BOUND, cls.SHAPE))

    @classmethod
    def load(cls, path_or_file):
        """
        Initiate a gene with DNA from a saved file.
        """
        return cls(np.load(path_or_file))

    def save(self, path_or_file):
        """
        Saves a gene's DNA to a file.
        """
        np.save(path_or_file, self._dna)

    def mutate(self, mutate_rate: float, mutator):
        """
        Mutates a gene's DNA (in place).
        Should be the only thing that edits the DNA of an existing gene.
        """
        for pos in np.ndindex(self.SHAPE):
            if np.random.rand() < mutate_rate:
                self._dna[pos] = mutator(self._dna[pos])

        self._prepared_dna = None


class StrategyGene(Gene):
    """
    Represent a gene that helps to choose an action each turn.
    """

    SHAPE = (len(METRICS_SHAPE),)

    def evaluate_state(self, state_metrics):
        """
        Evaluates a state's value according to its metrics
        """
        return np.matmul(state_metrics, self.dna)


class ManagerGene(Gene):
    """
    Represent a gene that helps to choose a strategy to follow (from 3 options)
    """

    SHAPE = (len(METRICS_SHAPE), 3)

    def select_strategy(
        self, state_metrics, strategies: tuple[StrategyGene]
    ) -> StrategyGene:
        output = np.matmul(state_metrics, self.dna)
        assert len(output) == len(strategies), f"Mismatching lengths ({output.shape})"
        index = max(range(len(output)), key=lambda i: output[i])
        # with open("shit.txt", 'a') as f:
        #     print(index, file=f)

        return strategies[index]