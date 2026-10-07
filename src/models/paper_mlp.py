"""
Paper Baseline MLP Model.

Faithfully implements the paper's best performing architecture (ID 3 in Table 1):
Hidden layers: relu_128_256
Input: Joint rotation quaternions
Output: PCA coefficients -> fixed PC layer reconstruction
"""

import torch
import torch.nn as nn
from .pc_layer import PCLayer


class PaperMLP(nn.Module):
    """
    MLP architecture from Xue & Wu (2021):
    Linear(in, 128) -> ReLU -> Linear(128, 256) -> ReLU -> Linear(256, k_pca) -> PC Layer.
    """

    def __init__(
        self,
        input_dim: int,
        k_pca: int,
        pc_layer: PCLayer,
        hidden1: int = 128,
        hidden2: int = 256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.k_pca = k_pca

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, k_pca),
        )
        self.pc_layer = pc_layer

    def forward(self, x: torch.Tensor, return_coeffs: bool = False):
        """
        x: (batch_size, input_dim)
        """
        c = self.net(x)
        offsets = self.pc_layer(c)
        if return_coeffs:
            return offsets, c
        return offsets
