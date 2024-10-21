"""
Constants relevant for MLP-based PPO and Gradient-Descent based learning.
"""

DROPOUT = 0.2
HUGE_NEG = -1e8
HIDDEN_DIMS: list[int] = [128, 128, 128, 128]
HIDDEN_STATE_DIM = 64
RECURRENT_LAYERS_AMOUNT = 1
VERY_SMALL_EPSILON = 1e-8

SEED = 1234
LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
MAX_EPISODES = 50000

DISCOUNT_FACTOR = 0.99
N_TRIALS = 10
PPO_STEPS = 10
PPO_CLIP = 0.2

ENTROPY_COEFFICIENT = 0.005
VALUE_COEFFICIENT = 0.5
VERY_SMALL_EPSILON = 1e-8

# Global Gradient Norm Clipping as suggested by (bullet #11):
# https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
MAX_GRADIENT_NORM = 1.0

ROLLOUT_BUFFER_SIZE = 1000

FIRST_DECK = 0
SECOND_DECK = 1
THIRD_DECK = 2
