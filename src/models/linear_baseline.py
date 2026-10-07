"""
Linear Regression Baseline Model.

Mirrors paper Model ID 0 (Section 3.3):
'linear regression is equivalent to a neural network with zero hidden layers...
Our model takes in joint rotation quaternions and applies a linear transformation.'
"""

import torch
import torch.nn as nn
from .pc_layer import PCLayer


class LinearBaseline(nn.Module):
    """
    Linear model mapping pose representation directly to PCA coefficients,
    then reconstructing vertex offsets using the fixed PC layer.
    """

    def __init__(self, input_dim: int, k_pca: int, pc_layer: PCLayer):
        super().__init__()
        self.input_dim = input_dim
        self.k_pca = k_pca
        self.fc = nn.Linear(input_dim, k_pca)
        self.pc_layer = pc_layer

    def forward(self, x: torch.Tensor, return_coeffs: bool = False):
        """
        x: (batch_size, input_dim)
        """
        c = self.fc(x)
        offsets = self.pc_layer(c)
        if return_coeffs:
            return offsets, c
        return offsets
