"""
PC Reconstruction Layer.

As described in Section 3.2 of the paper:
'When the network is configured to predict PCA coefficients, we append a "PC layer"
that reconstructs vertex offsets and whose weights and biases (principal components u_l,
sample mean mu) are fixed.'
"""

import torch
import torch.nn as nn


class PCLayer(nn.Module):
    """
    Fixed linear layer that maps predicted PCA coefficients back to vertex offsets.
    Formula: x_hat = c_hat @ components + mean
    """

    def __init__(self, components: torch.Tensor, mean: torch.Tensor):
        super().__init__()
        # components: shape (k_pca, output_dim), mean: shape (output_dim,)
        self.register_buffer("components", components.clone().detach())
        self.register_buffer("mean", mean.clone().detach())

    @property
    def k_pca(self) -> int:
        return self.components.shape[0]

    @property
    def output_dim(self) -> int:
        return self.components.shape[1]

    def forward(self, pca_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pca_coeffs: (batch_size, k_pca)
        Returns:
            reconstructed_offsets: (batch_size, output_dim)
        """
        return torch.matmul(pca_coeffs, self.components) + self.mean
