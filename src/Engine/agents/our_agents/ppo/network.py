from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np


DROPOUT = 0.2
HUGE_NEG = -1e8


class InputNormalization(nn.Module):
    def __init__(self, num_features, epsilon=1e-10):
        super().__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        # self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            n = x.numel() / x.size(1)
            with torch.no_grad():
                self.running_mean = self.running_mean * 0.9 + mean * 0.1
                self.running_var = self.running_var * 0.9 + var * 0.1
                # self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            var = self.running_var

        x_normalized = (x - mean) / torch.sqrt(var + self.epsilon)
        return x_normalized


class PPO(nn.Module):
    HIDDEN_DIMS: List[int] = [128, 128]

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = HIDDEN_DIMS,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.input_norm = InputNormalization(input_dim)

        layers = []
        prev_dim = input_dim
        for next_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, next_dim))
            layers.append(nn.LayerNorm(next_dim))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            prev_dim = next_dim
        self.net = nn.Sequential(*layers)

        self.actor = nn.Linear(hidden_dims[-1], output_dim)
        self.critic = nn.Linear(hidden_dims[-1], 1)

        # Initialize weights (recursively)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Orthogonal initialization of the weights as suggested by (bullet #2):
        https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
        """
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()

    def forward(
        self, x: torch.Tensor, action_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x_normalized = self.input_norm(x)
        x1 = self.net(x_normalized)
        actor_output = self.actor(x1)
        masked_actor_output = torch.where(action_mask == 0, HUGE_NEG, actor_output)
        prob = F.softmax(masked_actor_output, dim=1)
        return prob, self.critic(x1)
