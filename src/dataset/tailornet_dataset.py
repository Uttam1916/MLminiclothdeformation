"""
PyTorch Dataset and DataLoader for TailorNet Pose-to-Deformation Prediction.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def normalize_quaternions(quats: torch.Tensor) -> torch.Tensor:
    """
    Normalizes joint quaternions to unit sphere.
    quats: shape (..., 24, 4) or (..., 96)
    """
    original_shape = quats.shape
    q = quats.reshape(-1, 24, 4)
    norms = torch.norm(q, p=2, dim=-1, keepdim=True) + 1e-8
    q_norm = q / norms
    return q_norm.reshape(original_shape)


class TailorNetDataset(Dataset):
    """
    Dataset yielding (pose_quaternions, vertex_offsets, pca_coefficients).
    """

    def __init__(
        self,
        poses: np.ndarray,
        offsets: np.ndarray,
        pca_coeffs: np.ndarray,
        augment: bool = False,
        noise_std: float = 0.01,
    ):
        self.poses = torch.tensor(poses, dtype=torch.float32)
        self.offsets = torch.tensor(offsets, dtype=torch.float32)
        self.pca_coeffs = torch.tensor(pca_coeffs, dtype=torch.float32)
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.poses)

    def __getitem__(self, idx: int):
        pose = self.poses[idx]
        offset = self.offsets[idx]
        pca_coeff = self.pca_coeffs[idx]

        if self.augment:
            # Pose perturbation augmentation: add slight Gaussian noise to quaternions
            noise = torch.randn_like(pose) * self.noise_std
            pose = normalize_quaternions(pose + noise)

        return pose, offset, pca_coeff


def get_dataloaders(
    cache_path: str = "data/processed/dataset_cache.npz",
    batch_size: int = 32,
    augment_train: bool = False,
    noise_std: float = 0.01,
):
    """
    Creates train, val, test PyTorch DataLoaders from cached dataset file.
    """
    data = np.load(cache_path)
    train_dataset = TailorNetDataset(
        data["x_train"],
        data["y_train"],
        data["train_pca_coeffs"],
        augment=augment_train,
        noise_std=noise_std,
    )
    val_dataset = TailorNetDataset(
        data["x_val"],
        data["y_val"],
        data["val_pca_coeffs"],
        augment=False,
    )
    test_dataset = TailorNetDataset(
        data["x_test"],
        data["y_test"],
        data["test_pca_coeffs"],
        augment=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    metadata = {
        "pca_components": torch.tensor(data["pca_components"], dtype=torch.float32),
        "pca_mean": torch.tensor(data["pca_mean"], dtype=torch.float32),
        "canonical_vertices": data["canonical_vertices"],
        "faces": data["faces"],
        "input_dim": data["x_train"].shape[1],
        "output_dim": data["y_train"].shape[1],
        "k_pca": data["pca_components"].shape[0],
    }

    return train_loader, val_loader, test_loader, metadata
