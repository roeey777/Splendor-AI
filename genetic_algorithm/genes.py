import numpy as np

from Engine.Splendor.features import METRICS_SHAPE


RANDOM = np.random.default_rng(1234)


class Gene:
    MIN = -10
    MAX = 10
    SHAPE = (len(METRICS_SHAPE),)

    def __init__(self, dna: np.array):
        assert dna.shape == self.SHAPE, "Given DNA has the wrong shape"
        self._dna = dna
        self._prepared_dna = None

    @property
    def dna(self):
        if self._prepared_dna is None:
            arr = np.empty((0, self.SHAPE[1] if len(self.SHAPE) == 2 else 1))
            for repetition, value in zip(METRICS_SHAPE, self._dna, strict=True):
                stack = [value for _ in range(repetition)]
                arr = np.vstack([arr] + stack)
            self._prepared_dna = arr

        return self._prepared_dna


    @classmethod
    def random(cls):
        return cls(RANDOM.uniform(cls.MIN, cls.MAX, cls.SHAPE))


    @classmethod
    def load(cls, path_or_file):
        return cls(np.load(path_or_file))


    def save(self, path_or_file):
        np.save(path_or_file, self._dna)


    def mutate(self, mutate_rate: float, mutator):
        for pos in np.ndindex(self.SHAPE):
            if RANDOM.random() < mutate_rate:
                self._dna[pos] = mutator(self._dna[pos])

        self._prepared_dna = None



class StrategyGene(Gene):
    def evaluate_state(self, state_metrics):
        return np.matmul(state_metrics, self.dna)



class ManagerGene(Gene):
    SHAPE = (len(METRICS_SHAPE), 3)


    def select_strategy(self, state_metrics, strategies: tuple[StrategyGene]) -> StrategyGene:
        # output = np.exp(np.matmul(state_metrics, self.dna))
        # softmax_out = output / np.sum(output)
        # assert len(output) == len(strategies), "Mismatching lengths"
        # index = max(range(len(output)), key = lambda i: softmax_out[i])
        output = np.matmul(state_metrics, self.dna)
        assert len(output) == len(strategies), "Mismatching lengths"
        index = max(range(len(output)), key = lambda i: output[i])

        return strategies[index]
