"""
Evaluation and Benchmarking Module.

Computes:
1. MSE Loss (per-vertex and full vector, matching Table 1 of Xue & Wu)
2. Mean and Maximum Per-Vertex Euclidean Distance (in mm and cm)
3. Inference Latency per sample (matching Table 2 of Xue & Wu)
4. Model Parameter Counts
5. Generates comparison tables across models.
"""

import time
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Counts trainable and non-trainable parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "buffer_params": buffers,
    }


def benchmark_inference(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
    n_warmup: int = 20,
    n_runs: int = 200,
) -> Dict[str, float]:
    """
    Measures average inference latency per sample.
    """
    model.eval()
    x = sample_input.to(device)
    if x.ndim == 1:
        x = x.unsqueeze(0)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

    latencies = np.array(latencies)
    return {
        "mean_latency_sec": float(np.mean(latencies)),
        "std_latency_sec": float(np.std(latencies)),
        "mean_latency_ms": float(np.mean(latencies) * 1000),
    }


def evaluate_model(
    model: nn.Module,
    test_loader,
    device: torch.device,
    model_name: str = "model",
) -> Dict[str, Any]:
    """
    Evaluates model on test loader and calculates all metrics.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_poses = []

    with torch.no_grad():
        for poses, offsets, _ in test_loader:
            poses_dev = poses.to(device)
            preds = model(poses_dev).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(offsets.numpy())
            all_poses.append(poses.numpy())

    preds = np.concatenate(all_preds, axis=0)  # (N_test, 23106)
    targets = np.concatenate(all_targets, axis=0)  # (N_test, 23106)
    poses = np.concatenate(all_poses, axis=0)

    n_samples = preds.shape[0]
    n_vertices = preds.shape[1] // 3

    # Mean Squared Error across all elements (matching paper definition)
    mse_total = float(np.mean((preds - targets) ** 2))

    # Reshape to (N, V, 3) for 3D Euclidean per-vertex calculations
    preds_3d = preds.reshape(n_samples, n_vertices, 3)
    targets_3d = targets.reshape(n_samples, n_vertices, 3)

    # Per-vertex Euclidean error for each sample: (N, V)
    vertex_errors = np.linalg.norm(preds_3d - targets_3d, axis=-1)  # in meters
    mean_vertex_err_m = float(np.mean(vertex_errors))
    mean_vertex_err_mm = float(mean_vertex_err_m * 1000)
    max_vertex_err_m = float(np.max(vertex_errors))
    max_vertex_err_mm = float(max_vertex_err_m * 1000)

    # Squared per-vertex error averaged across test samples (for heatmaps)
    squared_per_vertex_error = np.mean(np.sum((preds_3d - targets_3d) ** 2, axis=-1), axis=0)  # (V,)

    # Per-sample MSE for finding best and worst examples
    sample_mse = np.mean((preds - targets) ** 2, axis=1)
    best_sample_idx = int(np.argmin(sample_mse))
    worst_sample_idx = int(np.argmax(sample_mse))

    # Model complexity & latency
    param_counts = count_parameters(model)
    first_pose = torch.tensor(poses[0:1], dtype=torch.float32)
    latency_info = benchmark_inference(model, first_pose, device)

    results = {
        "model_name": model_name,
        "test_mse": mse_total,
        "mean_vertex_error_mm": mean_vertex_err_mm,
        "max_vertex_error_mm": max_vertex_err_mm,
        "mean_vertex_error_m": mean_vertex_err_m,
        "max_vertex_error_m": max_vertex_err_m,
        "squared_per_vertex_error": squared_per_vertex_error,
        "best_sample_idx": best_sample_idx,
        "worst_sample_idx": worst_sample_idx,
        "best_sample_mse": float(sample_mse[best_sample_idx]),
        "worst_sample_mse": float(sample_mse[worst_sample_idx]),
        "predictions": preds_3d,
        "ground_truth": targets_3d,
        "test_poses": poses,
        **param_counts,
        **latency_info,
    }

    return results


def format_comparison_table(results_list: List[Dict[str, Any]]) -> str:
    """
    Formats markdown comparison table.
    """
    header = (
        "| Model | Trainable Params | Test MSE Loss | Mean Vertex Err (mm) | "
        "Max Vertex Err (mm) | Latency (ms) |\n"
        "|---|---:|---:|---:|---:|---:|"
    )
    rows = []
    for r in results_list:
        rows.append(
            f"| {r['model_name']} | {r['trainable_params']:,} | {r['test_mse']:.6e} | "
            f"{r['mean_vertex_error_mm']:.2f} mm | {r['max_vertex_error_mm']:.2f} mm | "
            f"{r['mean_latency_ms']:.3f} ms |"
        )
    return "\n".join([header] + rows)
