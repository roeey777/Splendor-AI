"""
Implementation of an input normalization layer.
"""

import torch
from torch import nn

from .constants import VERY_SMALL_EPSILON


class InputNormalization(nn.Module):
    """
    Input normalization layer - using a running average for calibrating the mean & variance.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, num_features: int, epsilon: float = VERY_SMALL_EPSILON):
        """
        Create a new input normalization layer.

        :param num_features: how many features to expect in the inputs.
        :param epsilon: which epsilon value should be add to the denominator during the
                        normalization in order to avoid division by 0.
        """

        super().__init__()
        self.num_features = num_features
        self.epsilon = epsilon

        self.register_buffer("running_mean", torch.zeros(1, num_features))
        self.register_buffer("running_var", torch.ones(1, num_features))

        self.running_mean: torch.Tensor
        self.running_var: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pylint: disable=attribute-defined-outside-init
        """
        Normalize the input using a running mean & variance estimators.
        The output should have 0 mean and variance of 1.

        :param x: the un-normalized input.
        :return: a normalized x.
        """

        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            with torch.no_grad():
                self.running_mean = self.running_mean * 0.9 + mean * 0.1
                self.running_var = self.running_var * 0.9 + var * 0.1
        else:
            mean = self.running_mean
            var = self.running_var

        x_normalized = (x - mean) / torch.sqrt(var + self.epsilon)
        return x_normalized
