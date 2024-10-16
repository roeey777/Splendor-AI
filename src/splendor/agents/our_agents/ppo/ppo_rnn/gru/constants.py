"""
Constants regarding PPO with GRU.
"""

from typing import List

DROPOUT = 0.2
HUGE_NEG = -1e8
HIDDEN_DIMS: List[int] = [128, 128, 128, 128]
HIDDEN_STATE_DIM = 64
RECURRENT_LAYERS_AMOUNT = 1
