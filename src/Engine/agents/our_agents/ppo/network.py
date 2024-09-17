from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np


DROPOUT = 0.4
HUGE_NEG = -1e8


class PPO(nn.Module):
    HIDDEN_DIM = 128

    def __init__(self, input_dim, output_dim, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, PPO.HIDDEN_DIM),
            nn.LayerNorm(PPO.HIDDEN_DIM),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        self.actor = nn.Linear(PPO.HIDDEN_DIM, output_dim)
        self.critic = nn.Linear(PPO.HIDDEN_DIM, 1)

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
        x1 = self.net(x)
        actor_output = self.actor(x1)
        masked_actor_output = torch.where(action_mask == 0, HUGE_NEG, actor_output)
        prob = F.softmax(masked_actor_output, dim=1)
        return prob, self.critic(x1)
