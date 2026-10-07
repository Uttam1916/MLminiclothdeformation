"""
Live Inference and 3D Garment Demonstration Script.

Loads the trained model, performs real-time inference on a given test pose,
measures latency, renders the predicted 3D garment mesh, and exports wavefront .obj files.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import PCLayer, ImprovedResMLP, PaperMLP
from src.training import get_device
from src.visualization import plot_mesh_comparison


def run_demo(
    sample_idx: int = 0,
    model_type: str = "improved",
    checkpoint_path: str = "checkpoints/improved_resmlp_best.pt",
    cache_path: str = "data/processed/dataset_cache.npz",
    output_dir: str = "outputs",
    export_obj: bool = True,
):
    print("=" * 70)
    print("3D Garment Deformation Live Inference Demo")
    print("=" * 70)

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Processed cache not found at {cache_path}. Run scripts/run_pipeline.py first.")

    data = np.load(cache_path)
    x_test = data["x_test"]
    y_test = data["y_test"]
    canonical_v = data["canonical_vertices"]
    faces = data["faces"]
    pca_comp = torch.tensor(data["pca_components"], dtype=torch.float32)
    pca_mean = torch.tensor(data["pca_mean"], dtype=torch.float32)

    device = get_device()
    print(f"Device: {device}")

    # Initialize model
    pc_layer = PCLayer(pca_comp, pca_mean)
    input_dim = x_test.shape[1]
    k_pca = pca_comp.shape[0]

    if "paper" in model_type:
        model = PaperMLP(input_dim, k_pca, pc_layer)
    else:
        model = ImprovedResMLP(input_dim, k_pca, pc_layer, hidden_dim=256, num_blocks=2, dropout_p=0.1)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Running with initialized weights.")

    model.to(device)
    model.eval()

    sample_idx = min(max(0, sample_idx), len(x_test) - 1)
    pose = torch.tensor(x_test[sample_idx : sample_idx + 1], dtype=torch.float32).to(device)
    gt_offset = y_test[sample_idx].reshape(-1, 3)

    # Benchmark single inference
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred_offset = model(pose).cpu().numpy().reshape(-1, 3)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_time_ms = (time.perf_counter() - t0) * 1000

    # Calculate error
    vertex_errors_mm = np.linalg.norm(pred_offset - gt_offset, axis=-1) * 1000
    mean_err_mm = np.mean(vertex_errors_mm)
    max_err_mm = np.max(vertex_errors_mm)

    print(f"\n--- Inference Results for Test Sample #{sample_idx} ---")
    print(f"Input Pose Dimensions: {pose.shape[-1]} (24 joint quaternions)")
    print(f"Output Mesh Vertices:  {len(pred_offset):,} (3D offsets: {pred_offset.size:,} floats)")
    print(f"Inference Latency:     {inference_time_ms:.3f} ms")
    print(f"Mean Vertex Error:     {mean_err_mm:.2f} mm")
    print(f"Max Vertex Error:      {max_err_mm:.2f} mm")

    os.makedirs(output_dir, exist_ok=True)

    # Export 3D Wavefront .obj meshes
    if export_obj:
        pred_mesh = trimesh.Trimesh(vertices=canonical_v + pred_offset, faces=faces)
        gt_mesh = trimesh.Trimesh(vertices=canonical_v + gt_offset, faces=faces)

        pred_obj_path = os.path.join(output_dir, f"demo_sample_{sample_idx}_pred.obj")
        gt_obj_path = os.path.join(output_dir, f"demo_sample_{sample_idx}_gt.obj")

        pred_mesh.export(pred_obj_path)
        gt_mesh.export(gt_obj_path)
        print(f"Exported predicted 3D mesh:   {pred_obj_path}")
        print(f"Exported ground truth mesh:   {gt_obj_path}")

    # Generate demo visualization
    demo_fig_path = os.path.join(output_dir, f"demo_sample_{sample_idx}_render.png")
    plot_mesh_comparison(
        canonical_v,
        gt_offset,
        pred_offset,
        sample_label=f"Demo Test Sample #{sample_idx}",
        save_path=demo_fig_path,
    )
    print(f"Saved visualization render:   {demo_fig_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--model-type", type=str, default="improved", choices=["improved", "paper"])
    parser.add_argument("--checkpoint", type=str, default="checkpoints/improved_resmlp_best.pt")
    args = parser.parse_args()

    run_demo(
        sample_idx=args.sample_idx,
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
    )
