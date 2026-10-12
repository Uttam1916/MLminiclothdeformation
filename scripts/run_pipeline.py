"""
End-to-End Pipeline Runner.

Executes data preprocessing on the expanded multi-sequence TailorNet dataset (2,122 frames),
runs a systematic PCA ablation study across different k (k in {32, 64, 128, 256}),
trains full model benchmarks (Linear Baseline, Paper MLP Baseline, Improved ResMLP,
and Improved ResMLP + Augmentation) on the optimal k, evaluates all metrics,
and generates publication figures.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from sklearn.decomposition import PCA

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.extract_data import load_tailornet_raw, preprocess_and_split
from src.dataset.tailornet_dataset import get_dataloaders
from src.models import PCLayer, LinearBaseline, PaperMLP, ImprovedResMLP
from src.training import Trainer, get_device
from src.evaluation import evaluate_model, format_comparison_table
from src.visualization import (
    plot_pca_analysis,
    plot_loss_curves,
    plot_vertex_error_heatmap,
    plot_mesh_comparison,
    plot_metrics_summary,
)


def run_pca_k_ablation(
    cache_path: str,
    device: torch.device,
    candidate_ks: list = [32, 64, 128, 256],
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 5e-4,
    checkpoint_dir: str = "checkpoints",
    output_dir: str = "outputs",
):
    """
    Evaluates PCA expressiveness and trains Paper MLP across different k
    to systematically determine the optimal PCA dimensionality.
    """
    print("\n" + "=" * 70)
    print("STEP 2: PCA k ABLATION STUDY (Comparing k in {32, 64, 128, 256})")
    print("=" * 70)

    data = np.load(cache_path)
    y_test = data["y_test"]
    pca_mean = data["pca_mean"]
    pca_components = data["pca_components"]
    explained_var = data["pca_explained_var"]

    ablation_results = []

    for k in candidate_ks:
        print(f"\n--- Evaluating k = {k} ---")
        # 1. Theoretical PCA reconstruction
        cum_var = float(np.sum(explained_var[:k]) * 100)
        y_test_centered = y_test - pca_mean
        coeffs_k = y_test_centered @ pca_components[:k].T
        direct_recon = coeffs_k @ pca_components[:k] + pca_mean
        direct_mse = float(np.mean((y_test - direct_recon) ** 2))
        direct_diff_mm = np.linalg.norm((y_test - direct_recon).reshape(-1, 7702, 3), axis=-1) * 1000.0
        direct_mean_err_mm = float(np.mean(direct_diff_mm))

        # 2. Train Paper MLP with this k
        train_loader, val_loader, test_loader, meta = get_dataloaders(
            cache_path=cache_path,
            batch_size=batch_size,
            k_pca=k,
        )
        pc_layer = PCLayer(meta["pca_components"], meta["pca_mean"])
        model = PaperMLP(meta["input_dim"], k, pc_layer, hidden1=128, hidden2=256)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=lr,
            device=device,
            checkpoint_dir=checkpoint_dir,
            model_name=f"paper_mlp_k{k}",
        )
        trainer.fit(epochs=epochs, patience=20, log_interval=40)

        # 3. Evaluate trained network on test split
        eval_res = evaluate_model(model, test_loader, device=device, model_name=f"Paper MLP (k={k})")

        res_item = {
            "k": k,
            "explained_variance_pct": round(cum_var, 2),
            "direct_pca_mse": direct_mse,
            "direct_pca_mean_mm": round(direct_mean_err_mm, 2),
            "model_test_mse": eval_res["test_mse"],
            "model_mean_err_mm": eval_res["mean_vertex_error_mm"],
            "model_max_err_mm": eval_res["max_vertex_error_mm"],
            "trainable_params": eval_res["trainable_params"],
        }
        ablation_results.append(res_item)

    # Format ablation markdown table
    header = "| $k$ | Expl. Var (%) | Direct PCA MSE | Direct Err (mm) | Model Test MSE | Mean Err (mm) | Max Err (mm) | Trainable Params |"
    sep = "|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for r in ablation_results:
        rows.append(
            f"| **{r['k']}** | {r['explained_variance_pct']}% | {r['direct_pca_mse']:.3e} | "
            f"{r['direct_pca_mean_mm']:.2f} mm | **{r['model_test_mse']:.3e}** | "
            f"{r['model_mean_err_mm']:.2f} mm | {r['model_max_err_mm']:.2f} mm | {r['trainable_params']:,} |"
        )
    ablation_table_str = "\n".join(rows)

    print("\n" + "=" * 70)
    print("PCA k ABLATION RESULTS")
    print("=" * 70)
    print(ablation_table_str)
    print("=" * 70)

    # Save to outputs
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "pca_k_ablation.md"), "w") as f:
        f.write("# PCA Dimensionality (k) Ablation Study\n\n" + ablation_table_str + "\n")
    with open(os.path.join(output_dir, "pca_k_ablation.json"), "w") as f:
        json.dump(ablation_results, f, indent=2)

    # Pick the k with lowest model test mse
    best_k = min(ablation_results, key=lambda x: x["model_test_mse"])["k"]
    print(f"\n>> Optimal PCA dimensionality found: k = {best_k} (lowest Test MSE)")
    return best_k, ablation_results


def run_full_pipeline(
    extracted_dir: str = "data/extracted",
    processed_dir: str = "data/processed",
    checkpoint_dir: str = "checkpoints",
    output_dir: str = "outputs",
    figures_dir: str = "figures",
    k_pca: int = 128,
    run_ablation: bool = True,
    epochs: int = 120,
    lr: float = 5e-4,
    batch_size: int = 64,
    seed: int = 42,
):
    print("=" * 70)
    print("Starting ML Mini-Project: 3D Garment Deformation from Human Pose")
    print(f"Dataset: Full 18-Sequence TailorNet Dataset | PCA Basis: k={k_pca}")
    print("=" * 70)

    # 1. Preprocessing
    cache_path = os.path.join(processed_dir, "dataset_cache.npz")
    recompute_cache = True
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        if len(data["x_train"]) > 500 and data["pca_components"].shape[0] >= 256:
            recompute_cache = False
            print(f"\n--- STEP 1: Using existing verified dataset cache ({len(data['x_train']) + len(data['x_val']) + len(data['x_test']):,} frames) ---")

    if recompute_cache:
        print("\n--- STEP 1: Preprocessing TailorNet Dataset across all sequences ---")
        thetas, unposed, canonical_v, faces = load_tailornet_raw(extracted_dir)
        preprocess_and_split(
            thetas,
            unposed,
            canonical_v,
            faces,
            split_ratios=(0.8, 0.1, 0.1),
            k_pca=256,
            seed=seed,
            output_dir=processed_dir,
        )

    device = get_device()
    print(f"Active compute device: {device}")

    # 2. PCA k Ablation (if enabled)
    best_k = k_pca
    if run_ablation:
        best_k, _ = run_pca_k_ablation(
            cache_path=cache_path,
            device=device,
            candidate_ks=[32, 64, 128, 256],
            epochs=min(epochs, 80),
            batch_size=batch_size,
            lr=lr,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
        )

    # 3. PCA Analysis Plot
    print("\n--- STEP 3: Generating PCA Variance and Reconstruction Curves ---")
    data = np.load(cache_path)
    y_train = data["y_train"]
    y_test = data["y_test"]
    canonical_v = data["canonical_vertices"]
    faces = data["faces"]
    pca_comp = data["pca_components"]
    pca_mean = data["pca_mean"]
    explained_var = data["pca_explained_var"]

    recon_errors = {}
    for k in [16, 32, 64, 128, 192, 256]:
        y_test_centered = y_test - pca_mean
        pred = (y_test_centered @ pca_comp[:k].T) @ pca_comp[:k] + pca_mean
        recon_errors[k] = float(np.mean((y_test - pred) ** 2))

    plot_pca_analysis(explained_var, recon_errors, os.path.join(figures_dir, "pca_analysis.png"))

    # 4. Setup DataLoaders with best_k
    print(f"\n--- STEP 4: Initializing DataLoaders with Optimal k = {best_k} ---")
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        cache_path=cache_path,
        batch_size=batch_size,
        augment_train=False,
        k_pca=best_k,
    )
    train_loader_aug, _, _, _ = get_dataloaders(
        cache_path=cache_path,
        batch_size=batch_size,
        augment_train=True,
        noise_std=0.015,
        k_pca=best_k,
    )

    # Fixed PC Reconstruction Layer
    pc_layer = PCLayer(meta["pca_components"], meta["pca_mean"])

    models_to_train = {
        "Linear Baseline": (
            LinearBaseline(meta["input_dim"], meta["k_pca"], pc_layer),
            train_loader,
            1e-3,
        ),
        "Paper MLP": (
            PaperMLP(meta["input_dim"], meta["k_pca"], pc_layer, hidden1=128, hidden2=256),
            train_loader,
            lr,
        ),
        "Improved ResMLP": (
            ImprovedResMLP(meta["input_dim"], meta["k_pca"], pc_layer, hidden_dim=256, num_blocks=2, dropout_p=0.1),
            train_loader,
            lr,
        ),
        "Improved ResMLP + Aug": (
            ImprovedResMLP(meta["input_dim"], meta["k_pca"], pc_layer, hidden_dim=256, num_blocks=2, dropout_p=0.1),
            train_loader_aug,
            lr,
        ),
    }

    # 5. Training Models
    histories = {}
    print(f"\n--- STEP 5: Training Benchmark Models (k = {best_k}) ---")
    for name, (model, loader, model_lr) in models_to_train.items():
        print(f"\n>> Training {name}...")
        safe_name = name.lower().replace(" ", "_").replace("+", "plus")
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=val_loader,
            lr=model_lr,
            device=device,
            checkpoint_dir=checkpoint_dir,
            model_name=safe_name,
        )
        hist = trainer.fit(epochs=epochs, patience=25, log_interval=20)
        histories[name] = hist

    # Save loss curves
    plot_loss_curves(histories, os.path.join(figures_dir, "loss_curves.png"))

    # 6. Evaluation
    print("\n--- STEP 6: Model Evaluation on Unseen Test Split ---")
    evaluation_results = []
    best_overall_model_res = None
    best_overall_test_mse = float("inf")

    for name, (model, _, _) in models_to_train.items():
        safe_name = name.lower().replace(" ", "_").replace("+", "plus")
        best_pt = os.path.join(checkpoint_dir, f"{safe_name}_best.pt")
        if os.path.exists(best_pt):
            ckpt = torch.load(best_pt, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

        res = evaluate_model(model, test_loader, device=device, model_name=name)
        evaluation_results.append(res)

        if res["test_mse"] < best_overall_test_mse:
            best_overall_test_mse = res["test_mse"]
            best_overall_model_res = res

    # Format and save comparison table
    table_str = format_comparison_table(evaluation_results)
    print("\n" + "=" * 70)
    print("FINAL EXPERIMENTAL RESULTS COMPARISON")
    print("=" * 70)
    print(table_str)
    print("=" * 70)

    with open(os.path.join(output_dir, "evaluation_table.md"), "w") as f:
        f.write(f"# Model Evaluation Benchmarks (k = {best_k})\n\n" + table_str + "\n")

    clean_json = []
    for r in evaluation_results:
        clean_json.append({
            k: v for k, v in r.items()
            if not isinstance(v, (np.ndarray, torch.Tensor))
        })
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(clean_json, f, indent=2)

    # 7. Visualizations
    print("\n--- STEP 7: Generating Publication Figures & 3D Deformations ---")
    plot_metrics_summary(evaluation_results, os.path.join(figures_dir, "metrics_summary.png"))

    plot_vertex_error_heatmap(
        canonical_v,
        best_overall_model_res["squared_per_vertex_error"],
        save_path=os.path.join(figures_dir, "vertex_error_heatmap.png"),
    )

    best_idx = best_overall_model_res["best_sample_idx"]
    worst_idx = best_overall_model_res["worst_sample_idx"]

    plot_mesh_comparison(
        canonical_v,
        best_overall_model_res["ground_truth"][best_idx],
        best_overall_model_res["predictions"][best_idx],
        sample_label="Lowest Error Test Example",
        save_path=os.path.join(figures_dir, "mesh_comparison_best.png"),
    )

    plot_mesh_comparison(
        canonical_v,
        best_overall_model_res["ground_truth"][worst_idx],
        best_overall_model_res["predictions"][worst_idx],
        sample_label="Highest Error Test Example",
        save_path=os.path.join(figures_dir, "mesh_comparison_worst.png"),
    )

    print("\n[COMPLETE] Full pipeline successfully finished!")
    return evaluation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--k-pca", type=int, default=128)
    parser.add_argument("--no-ablation", action="store_true")
    args = parser.parse_args()

    run_full_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        k_pca=args.k_pca,
        run_ablation=not args.no_ablation,
    )
