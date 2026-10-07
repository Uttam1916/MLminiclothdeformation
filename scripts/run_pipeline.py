"""
End-to-End Pipeline Runner.

Executes data preprocessing, model training (Linear Baseline, Paper MLP Baseline,
Improved ResMLP, and Improved ResMLP + Augmentation), evaluation, benchmarking,
and publication figure generation.
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


def run_full_pipeline(
    extracted_dir: str = "data/extracted",
    processed_dir: str = "data/processed",
    checkpoint_dir: str = "checkpoints",
    output_dir: str = "outputs",
    figures_dir: str = "figures",
    k_pca: int = 32,
    epochs: int = 1000,
    lr: float = 1e-4,
    batch_size: int = 32,
    seed: int = 42,
):
    print("=" * 70)
    print("Starting ML Mini-Project: 3D Garment Deformation from Human Pose")
    print("=" * 70)

    # 1. Preprocessing
    cache_path = os.path.join(processed_dir, "dataset_cache.npz")
    if not os.path.exists(cache_path):
        print("\n--- STEP 1: Preprocessing TailorNet Dataset ---")
        thetas, unposed, canonical_v, faces = load_tailornet_raw(extracted_dir)
        preprocess_and_split(
            thetas,
            unposed,
            canonical_v,
            faces,
            n_train=70,
            n_val=20,
            n_test=20,
            k_pca=k_pca,
            seed=seed,
            output_dir=processed_dir,
        )
    else:
        print(f"\n--- STEP 1: Using existing cached preprocessed dataset at {cache_path} ---")

    # 2. PCA variance analysis across k
    print("\n--- STEP 2: Computing Multi-k PCA Reconstruction Trade-off ---")
    data = np.load(cache_path)
    y_train = data["y_train"]
    y_test = data["y_test"]
    canonical_v = data["canonical_vertices"]
    faces = data["faces"]

    recon_errors = {}
    for k in [4, 8, 16, 24, 32, 48, 64]:
        if k <= len(y_train):
            p = PCA(n_components=k)
            p.fit(y_train)
            pred = p.inverse_transform(p.transform(y_test))
            recon_errors[k] = float(np.mean((y_test - pred) ** 2))

    full_pca = PCA()
    full_pca.fit(y_train)
    plot_pca_analysis(full_pca.explained_variance_ratio_, recon_errors, os.path.join(figures_dir, "pca_analysis.png"))

    # 3. Setup DataLoaders & Models
    print("\n--- STEP 3: Initializing PyTorch DataLoaders & Models ---")
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        cache_path=cache_path,
        batch_size=batch_size,
        augment_train=False,
    )
    # Augmented train loader
    train_loader_aug, _, _, _ = get_dataloaders(
        cache_path=cache_path,
        batch_size=batch_size,
        augment_train=True,
        noise_std=0.015,
    )

    device = get_device()
    print(f"Active compute device: {device}")

    # Fixed PC Reconstruction Layer
    pc_layer = PCLayer(meta["pca_components"], meta["pca_mean"])

    models_to_train = {
        "Linear Baseline": (
            LinearBaseline(meta["input_dim"], meta["k_pca"], pc_layer),
            train_loader,
            1e-3,  # linear models benefit from slightly higher lr
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

    # 4. Training
    histories = {}
    print("\n--- STEP 4: Training Models ---")
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
        hist = trainer.fit(epochs=epochs, patience=250, log_interval=200)
        histories[name] = hist

    # Save loss curves
    plot_loss_curves(histories, os.path.join(figures_dir, "loss_curves.png"))

    # 5. Evaluation
    print("\n--- STEP 5: Model Evaluation on Test Split ---")
    evaluation_results = []
    best_overall_model_res = None
    best_overall_test_mse = float("inf")

    for name, (model, _, _) in models_to_train.items():
        safe_name = name.lower().replace(" ", "_").replace("+", "plus")
        # Load best weights
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

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "evaluation_table.md"), "w") as f:
        f.write("# Model Evaluation Benchmarks\n\n" + table_str + "\n")

    # Serialize results to json (exclude numpy arrays)
    clean_json = []
    for r in evaluation_results:
        clean_json.append({
            k: v for k, v in r.items()
            if not isinstance(v, (np.ndarray, torch.Tensor))
        })
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(clean_json, f, indent=2)

    # 6. Visualizations
    print("\n--- STEP 6: Generating Figures & 3D Deformations ---")
    plot_metrics_summary(evaluation_results, os.path.join(figures_dir, "metrics_summary.png"))

    # Vertex error heatmaps using best model
    plot_vertex_error_heatmap(
        canonical_v,
        best_overall_model_res["squared_per_vertex_error"],
        save_path=os.path.join(figures_dir, "vertex_error_heatmap.png"),
    )

    # Mesh comparisons: best and worst test samples
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
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--k-pca", type=int, default=32)
    args = parser.parse_args()

    run_full_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        k_pca=args.k_pca,
    )
