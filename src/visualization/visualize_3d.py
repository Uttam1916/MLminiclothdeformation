"""
Visualization and Plotting Module.

Generates academic figures closely mirroring Xue & Wu (2021):
- Figure 1: PCA cumulative explained variance and reconstruction error (mirroring paper Figure 3)
- Figure 2: Training & Validation loss curves comparison (Baseline vs Improved)
- Figure 3: Squared per-vertex error heatmaps - front & back view (mirroring paper Figure 4)
- Figure 4: Qualitative mesh comparison - Ground Truth vs Predicted best/worst (mirroring paper Figure 5)
- Figure 5: Model comparison bar chart (Loss, error, parameter count)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def plot_pca_analysis(
    explained_variance_ratio: np.ndarray,
    reconstruction_errors: dict,
    save_path: str = "figures/pca_analysis.png",
):
    """
    Plots cumulative explained variance and reconstruction error vs PCA components.
    Reproduces paper Figure 3(a) and 3(b).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    # (a) Cumulative explained variance
    k_range = np.arange(1, len(explained_variance_ratio) + 1)
    cum_var = np.cumsum(explained_variance_ratio)

    ax1.plot(k_range, cum_var, color="#d9534f", lw=2, label="Cumulative Explained Variance")
    # Annotate key thresholds (e.g., 90% and 99%)
    k_90 = np.argmax(cum_var >= 0.90) + 1
    k_99 = np.argmax(cum_var >= 0.99) + 1
    ax1.axvline(k_90, color="gray", linestyle="--", alpha=0.7, label=f"90% var (k={k_90})")
    ax1.axvline(k_99, color="blue", linestyle=":", alpha=0.7, label=f"99% var (k={k_99})")
    ax1.set_xlabel("Number of Principal Components ($k$)", fontsize=11)
    ax1.set_ylabel("Cumulative Explained Variance", fontsize=11)
    ax1.set_title("(a) Cumulative Explained Variance", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")
    ax1.set_ylim(0.5, 1.02)

    # (b) Reconstruction error vs k
    ks = sorted(reconstruction_errors.keys())
    errs = [reconstruction_errors[k] for k in ks]
    ax2.plot(ks, errs, marker="o", color="#0275d8", lw=2, label="Test Reconstruction MSE")
    ax2.set_xlabel("Number of Principal Components ($k$)", fontsize=11)
    ax2.set_ylabel("Reconstruction MSE", fontsize=11)
    ax2.set_title("(b) Reconstruction Error vs $k$", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA analysis figure to {save_path}")


def plot_loss_curves(
    histories: dict,
    save_path: str = "figures/loss_curves.png",
):
    """
    Plots training and validation loss curves across models.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300)

    colors = {
        "Linear Baseline": "#6c757d",
        "Paper MLP": "#d9534f",
        "Improved ResMLP": "#0275d8",
        "Improved ResMLP + Aug": "#28a745",
    }

    for name, hist in histories.items():
        c = colors.get(name, None)
        train_loss = hist["train_loss"]
        val_loss = hist["val_loss"]
        epochs = np.arange(1, len(train_loss) + 1)

        ax1.plot(epochs, train_loss, label=f"{name}", color=c, lw=1.8, alpha=0.9)
        ax2.plot(epochs, val_loss, label=f"{name}", color=c, lw=1.8, alpha=0.9)

    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("MSE Loss (log scale)", fontsize=11)
    ax1.set_title("Training Loss Convergence", fontsize=12, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    ax1.legend(fontsize=9)

    ax2.set_yscale("log")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("MSE Loss (log scale)", fontsize=11)
    ax2.set_title("Validation Loss Convergence", fontsize=12, fontweight="bold")
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curves figure to {save_path}")


def plot_vertex_error_heatmap(
    canonical_v: np.ndarray,
    squared_vertex_error: np.ndarray,
    save_path: str = "figures/vertex_error_heatmap.png",
):
    """
    Plots squared per-vertex error heatmap (Front and Back views).
    Directly mirrors Figure 4 of Xue & Wu (2021).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Take square root of error to reduce variance, normalize between 0.0 and 1.0
    rmse = np.sqrt(squared_vertex_error)
    norm_err = (rmse - rmse.min()) / (rmse.max() - rmse.min() + 1e-8)

    fig = plt.figure(figsize=(12, 5.5), dpi=300)

    # Front view
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    sc1 = ax1.scatter(
        canonical_v[:, 0],
        canonical_v[:, 2],  # depth
        canonical_v[:, 1],  # height
        c=norm_err,
        cmap="coolwarm",
        s=1.2,
        alpha=0.8,
    )
    ax1.view_init(elev=10, azim=-90)
    ax1.set_axis_off()
    ax1.set_title("(a) Front View - Per-Vertex Error", fontsize=11, fontweight="bold")

    # Back view
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    sc2 = ax2.scatter(
        canonical_v[:, 0],
        canonical_v[:, 2],
        canonical_v[:, 1],
        c=norm_err,
        cmap="coolwarm",
        s=1.2,
        alpha=0.8,
    )
    ax2.view_init(elev=10, azim=90)
    ax2.set_axis_off()
    ax2.set_title("(b) Back View - Per-Vertex Error", fontsize=11, fontweight="bold")

    cbar = fig.colorbar(sc1, ax=[ax1, ax2], shrink=0.6, pad=0.03, aspect=20)
    cbar.set_label("Normalized Per-Vertex Error ($0.0 - 1.0$)", fontsize=10)

    plt.suptitle("Average Squared Per-Vertex Error Heatmap", fontsize=13, fontweight="bold", y=0.92)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved vertex error heatmap to {save_path}")


def plot_mesh_comparison(
    canonical_v: np.ndarray,
    ground_truth_offsets: np.ndarray,
    pred_offsets: np.ndarray,
    sample_label: str = "Lowest Error Sample",
    save_path: str = "figures/mesh_comparison.png",
):
    """
    Plots Ground Truth vs Predicted deformed mesh side-by-side.
    Mirroring Figure 5 of Xue & Wu (2021).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    gt_v = canonical_v + ground_truth_offsets
    pred_v = canonical_v + pred_offsets

    # Local per-vertex error
    v_err = np.linalg.norm(pred_v - gt_v, axis=-1) * 1000  # mm

    fig = plt.figure(figsize=(14, 5.5), dpi=300)

    # 1. Ground Truth Mesh
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.scatter(gt_v[:, 0], gt_v[:, 2], gt_v[:, 1], c="#2b5c8f", s=1.0, alpha=0.7)
    ax1.view_init(elev=15, azim=-75)
    ax1.set_axis_off()
    ax1.set_title("Ground Truth Simulation", fontsize=11, fontweight="bold")

    # 2. Predicted Mesh
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.scatter(pred_v[:, 0], pred_v[:, 2], pred_v[:, 1], c="#28a745", s=1.0, alpha=0.7)
    ax2.view_init(elev=15, azim=-75)
    ax2.set_axis_off()
    ax2.set_title("Predicted Deformation", fontsize=11, fontweight="bold")

    # 3. Error Overlay
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    sc3 = ax3.scatter(pred_v[:, 0], pred_v[:, 2], pred_v[:, 1], c=v_err, cmap="magma", s=1.0, alpha=0.85)
    ax3.view_init(elev=15, azim=-75)
    ax3.set_axis_off()
    ax3.set_title("Deformation Error (mm)", fontsize=11, fontweight="bold")
    cbar = fig.colorbar(sc3, ax=ax3, shrink=0.6, pad=0.03, aspect=20)
    cbar.set_label("Vertex Error (mm)", fontsize=10)

    plt.suptitle(f"3D Cloth Deformation: {sample_label}", fontsize=13, fontweight="bold", y=0.92)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved mesh comparison figure to {save_path}")


def plot_metrics_summary(
    results_list: list,
    save_path: str = "figures/metrics_summary.png",
):
    """
    Plots a multi-panel comparison of Test MSE, Mean Error, and Latency.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    names = [r["model_name"] for r in results_list]
    mses = [r["test_mse"] * 1e4 for r in results_list]  # in 1e-4 units matching paper
    mean_errs = [r["mean_vertex_error_mm"] for r in results_list]
    latencies = [r["mean_latency_ms"] for r in results_list]

    x = np.arange(len(names))
    width = 0.45

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2), dpi=300)

    bars1 = ax1.bar(x, mses, width, color=["#6c757d", "#d9534f", "#0275d8", "#28a745"][:len(names)])
    ax1.set_ylabel("Test MSE Loss ($\\times 10^{-4}$)", fontsize=10)
    ax1.set_title("Test MSE Loss", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    bars2 = ax2.bar(x, mean_errs, width, color=["#6c757d", "#d9534f", "#0275d8", "#28a745"][:len(names)])
    ax2.set_ylabel("Mean Vertex Error (mm)", fontsize=10)
    ax2.set_title("Mean 3D Geometric Error", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    bars3 = ax3.bar(x, latencies, width, color=["#6c757d", "#d9534f", "#0275d8", "#28a745"][:len(names)])
    ax3.set_ylabel("Inference Latency (ms)", fontsize=10)
    ax3.set_title("Real-Time Inference Speed", fontsize=11, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved metrics summary figure to {save_path}")
