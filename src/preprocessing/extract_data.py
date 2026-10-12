"""
TailorNet Data Preprocessing & Feature Extraction.

Extracts SMPL pose parameters and unposed cloth vertex offsets from the TailorNet dataset,
loads all available motion sequences (up to 2,122 frames across 18 sequences),
converts joint rotations to unit quaternions, fits PCA for output dimensionality reduction,
and creates reproducible train/val/test splits.
"""

import os
import glob
import argparse
import pickle
from typing import Tuple, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA


def load_tailornet_raw(extracted_dir: str = "data/extracted") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads raw TailorNet files from extracted directory across all available sequences.
    Concatenates motion sequences into a unified dataset.
    """
    pose_dir = os.path.join(extracted_dir, "t-shirt_female", "pose", "000_023")
    style_path = os.path.join(extracted_dir, "t-shirt_female", "style_model.npz")
    gamma_path = os.path.join(extracted_dir, "t-shirt_female", "style", "gamma_023.npy")
    info_path = os.path.join(extracted_dir, "garment_class_info.pkl")

    if not os.path.exists(pose_dir):
        raise FileNotFoundError(f"Missing sequence directory: {pose_dir}")

    # Discover all matching pairs of poses and unposed files
    pose_files = sorted(glob.glob(os.path.join(pose_dir, "poses_*.npz")))
    unposed_files = sorted(glob.glob(os.path.join(pose_dir, "unposed_*.npy")))

    if not pose_files or not unposed_files:
        raise FileNotFoundError(f"No pose or unposed files found in {pose_dir}")

    thetas_list = []
    unposed_list = []
    loaded_seqs = 0

    for pf in pose_files:
        seq_id = os.path.basename(pf).replace("poses_", "").replace(".npz", "")
        uf = os.path.join(pose_dir, f"unposed_{seq_id}.npy")
        if os.path.exists(uf):
            p_data = np.load(pf)
            u_data = np.load(uf)
            thetas_list.append(p_data["thetas"])
            unposed_list.append(u_data)
            loaded_seqs += 1

    thetas = np.concatenate(thetas_list, axis=0)  # (N, 72)
    unposed_offsets = np.concatenate(unposed_list, axis=0)  # (N, 7702, 3)
    print(f"Loaded {loaded_seqs} motion sequences ({len(thetas):,} frames total).")

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
    quats_xyzw = rotations.as_quat().reshape(n_samples, 24, 4)
    quats_wxyz = np.concatenate([quats_xyzw[:, :, 3:4], quats_xyzw[:, :, 0:3]], axis=-1)
    return quats_wxyz.reshape(n_samples, -1)


def preprocess_and_split(
    thetas: np.ndarray,
    unposed_offsets: np.ndarray,
    canonical_v: np.ndarray,
    faces: np.ndarray,
    n_train: Optional[int] = None,
    n_val: Optional[int] = None,
    n_test: Optional[int] = None,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    k_pca: int = 128,
    seed: int = 42,
    output_dir: str = "data/processed",
):
    """
    Processes pose to quaternions, splits data reproducibly, computes PCA basis on train split,
    and saves processed dataset cache.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_samples = len(thetas)

    if n_train is None or n_val is None or n_test is None:
        r_tr, r_va, r_te = split_ratios
        n_train = int(n_samples * r_tr)
        n_val = int(n_samples * r_va)
        n_test = n_samples - n_train - n_val

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
    print(f"Fitting PCA with k={k_pca} components on {len(y_train)} training frames...")
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
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--k-pca", type=int, default=128)
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
