"""
Improved Model: Residual MLP with LayerNorm, Dropout, and GELU.

Our architectural improvement beyond the paper's plain feedforward baseline:
1. Residual skip connections allow gradients to flow cleanly through hidden layers.
2. Layer Normalization stabilizes latent activations across poses.
3. GELU non-linearities provide smooth activation gating without dead neurons.
4. Dropout regularizes high-dimensional pose features against overfitting.
"""

import torch
import torch.nn as nn
from .pc_layer import PCLayer


class ResidualBlock(nn.Module):
    """
    Residual feed-forward block:
    x -> LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout -> (+x)
    """

    def __init__(self, dim: int, dropout_p: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout_p)
        self.ln2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop2 = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.ln1(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop1(h)
        h = self.ln2(h)
        h = self.fc2(h)
        h = self.drop2(h)
        return residual + h


class ImprovedResMLP(nn.Module):
    """
    Residual MLP with Layer Normalization, GELU, and Dropout regularization.
    """

    def __init__(
        self,
        input_dim: int,
        k_pca: int,
        pc_layer: PCLayer,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.k_pca = k_pca

        # Input projection
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Residual backbone
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_p=dropout_p)
            for _ in range(num_blocks)
        ])

        # Output head
        self.out_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, k_pca),
        )

        self.pc_layer = pc_layer

    def forward(self, x: torch.Tensor, return_coeffs: bool = False):
        """
        x: (batch_size, input_dim)
        """
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        c = self.out_head(h)
        offsets = self.pc_layer(c)
        if return_coeffs:
            return offsets, c
        return offsets
