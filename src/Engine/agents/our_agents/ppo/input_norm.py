import torch
import torch.nn as nn

from .constants import VERY_SMALL_EPSILON


class InputNormalization(nn.Module):
    def __init__(self, num_features, epsilon=VERY_SMALL_EPSILON):
        super().__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(1, num_features))
        self.register_buffer("running_var", torch.ones(1, num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            n = x.numel() / x.size(1)
            with torch.no_grad():
                self.running_mean = self.running_mean * 0.9 + mean * 0.1
                self.running_var = self.running_var * 0.9 + var * 0.1
        else:
            mean = self.running_mean
            var = self.running_var

        x_normalized = (x - mean) / torch.sqrt(var + self.epsilon)
        return x_normalized
