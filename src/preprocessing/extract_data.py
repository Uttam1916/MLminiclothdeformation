"""
TailorNet Data Preprocessing & Feature Extraction.

Extracts SMPL pose parameters and unposed cloth vertex offsets from the TailorNet dataset,
converts joint rotations to unit quaternions, fits PCA for output dimensionality reduction,
and creates reproducible train/val/test splits.
"""

import os
import argparse
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA


def load_tailornet_raw(extracted_dir: str):
    """
    Loads raw TailorNet files from extracted directory.
    """
    pose_npz_path = os.path.join(extracted_dir, "t-shirt_female", "pose", "000_023", "poses_000.npz")
    unposed_path = os.path.join(extracted_dir, "t-shirt_female", "pose", "000_023", "unposed_000.npy")
    style_path = os.path.join(extracted_dir, "t-shirt_female", "style_model.npz")
    gamma_path = os.path.join(extracted_dir, "t-shirt_female", "style", "gamma_023.npy")
    info_path = os.path.join(extracted_dir, "garment_class_info.pkl")

    if not os.path.exists(pose_npz_path) or not os.path.exists(unposed_path):
        raise FileNotFoundError(f"Missing required dataset files in {extracted_dir}")

    pose_data = np.load(pose_npz_path)
    thetas = pose_data["thetas"]  # shape (N, 72)
    unposed_offsets = np.load(unposed_path)  # shape (N, 7702, 3)

    # Garment mesh topology
    with open(info_path, "rb") as f:
        garment_info = pickle.load(f)
    faces = garment_info["t-shirt"]["f"]  # shape (15180, 3)

    # Style model canonical garment template
    style_data = np.load(style_path)
    mean_gar = style_data["mean"]
    pca_w = style_data["pca_w"]
    coeff_mean = style_data["coeff_mean"]
    gamma = np.load(gamma_path)
    canonical_v = (pca_w.T.dot(gamma + coeff_mean) + mean_gar).reshape(-1, 3)  # shape (7702, 3)

    return thetas, unposed_offsets, canonical_v, faces


def axis_angle_to_quaternions(thetas: np.ndarray) -> np.ndarray:
    """
    Converts SMPL axis-angle parameters (N, 72) into joint rotation unit quaternions (N, 96).
    SMPL has 24 joints, each with 3 axis-angle components.
    Output: 24 * 4 = 96 dimensions (w, x, y, z format for each joint).
    """
    n_samples = thetas.shape[0]
    thetas_joints = thetas.reshape(-1, 3)
    rotations = R.from_rotvec(thetas_joints)
    # Scipy returns (x, y, z, w); reorder to (w, x, y, z) to match paper convention
    quats_xyzw = rotations.as_quat().reshape(n_samples, 24, 4)
    quats_wxyz = np.concatenate([quats_xyzw[:, :, 3:4], quats_xyzw[:, :, 0:3]], axis=-1)
    return quats_wxyz.reshape(n_samples, -1)


def preprocess_and_split(
    thetas: np.ndarray,
    unposed_offsets: np.ndarray,
    canonical_v: np.ndarray,
    faces: np.ndarray,
    n_train: int = 70,
    n_val: int = 20,
    n_test: int = 20,
    k_pca: int = 32,
    seed: int = 42,
    output_dir: str = "data/processed",
):
    """
    Processes pose to quaternions, splits data reproducibly, computes PCA basis on train split,
    and saves processed artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_samples = len(thetas)
    assert n_train + n_val + n_test <= n_samples, f"Total split ({n_train}+{n_val}+{n_test}) > total ({n_samples})"

    # Convert pose to quaternions
    quats = axis_angle_to_quaternions(thetas)  # (N, 96)
    offsets_flat = unposed_offsets.reshape(n_samples, -1)  # (N, 23106)

    # Reproducible shuffle and split
    np.random.seed(seed)
    perm = np.random.permutation(n_samples)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val : n_train + n_val + n_test]

    x_train, y_train = quats[train_idx], offsets_flat[train_idx]
    x_val, y_val = quats[val_idx], offsets_flat[val_idx]
    x_test, y_test = quats[test_idx], offsets_flat[test_idx]

    # Fit PCA ONLY on training split to avoid data leakage
    pca = PCA(n_components=k_pca)
    pca.fit(y_train)

    train_pca_coeffs = pca.transform(y_train)
    val_pca_coeffs = pca.transform(y_val)
    test_pca_coeffs = pca.transform(y_test)

    # Verification: Reconstruction MSE on test split using fitted PCA
    test_recon = pca.inverse_transform(test_pca_coeffs)
    pca_test_mse = float(np.mean((y_test - test_recon) ** 2))
    cum_variance = float(pca.explained_variance_ratio_.sum())

    print(f"Dataset split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    print(f"PCA components (k={k_pca}): Cumulative Explained Variance = {cum_variance * 100:.2f}%")
    print(f"Test Set Direct PCA Reconstruction MSE = {pca_test_mse:.6e}")

    # Save processed dataset
    cache_path = os.path.join(output_dir, "dataset_cache.npz")
    np.savez_compressed(
        cache_path,
        x_train=x_train,
        y_train=y_train,
        train_pca_coeffs=train_pca_coeffs,
        x_val=x_val,
        y_val=y_val,
        val_pca_coeffs=val_pca_coeffs,
        x_test=x_test,
        y_test=y_test,
        test_pca_coeffs=test_pca_coeffs,
        pca_components=pca.components_,  # (k, 23106)
        pca_mean=pca.mean_,  # (23106,)
        pca_explained_var=pca.explained_variance_ratio_,
        canonical_vertices=canonical_v,  # (7702, 3)
        faces=faces,  # (15180, 3)
        raw_thetas=thetas,
        raw_offsets=unposed_offsets,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    print(f"Saved processed dataset to {cache_path}")

    # Also save PCA model pickle
    pca_path = os.path.join(output_dir, "pca_model.pkl")
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
    print(f"Saved PCA model to {pca_path}")

    return cache_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess TailorNet dataset")
    parser.add_argument("--extracted-dir", type=str, default="data/extracted")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--n-train", type=int, default=70)
    parser.add_argument("--n-val", type=int, default=20)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--k-pca", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    thetas, unposed, canonical_v, faces = load_tailornet_raw(args.extracted_dir)
    preprocess_and_split(
        thetas,
        unposed,
        canonical_v,
        faces,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        k_pca=args.k_pca,
        seed=args.seed,
        output_dir=args.output_dir,
    )
