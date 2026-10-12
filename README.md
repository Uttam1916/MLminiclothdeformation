# Data-Driven 3D Garment Deformation from Human Pose
# SRNs: PES1UG24CS697, PES1UG24CS698
**Course:** UE24CS352A - Machine Learning Mini-Project  
**Topic:** Reproduction and Extension of *Data-Driven Clothing for Interactive Applications* (Xue & Wu, Stanford CS 229)  
**Dataset:** Multi-Sequence TailorNet Benchmark (CVPR 2020, Max Planck Institute for Informatics)  

---

##  Project Overview

Simulating cloth physics using classical finite-element numerical solvers (such as PhysBAM) takes minutes per frame (~180 seconds in the reference paper). This high computational cost prevents procedural solvers from being used in interactive real-time applications like video games, virtual reality, and virtual try-on.

This project implements a data-driven machine learning pipeline that predicts 3D garment mesh deformation directly from human body skeletal poses in real time (< 0.55 ms per frame, over 1,800 FPS).

<<<<<<< HEAD
=======
### Key Work Done
- **Methodological Reproduction:** Reimplemented the core architecture of Xue & Wu (2021) using joint rotation quaternions as inputs, PCA dimensionality reduction on vertex displacements, and a fixed Principal Component (PC) reconstruction layer.
- **Dataset Expansion:** Scaled up from a single 110-frame sample to all **18 motion sequences** (2,122 simulation frames, 7,702 vertices, 15,180 triangular faces) from the official public TailorNet benchmark.
- **Systematic PCA ($k$) Ablation:** Evaluated PCA subspace dimensions across $k \in \{32, 64, 128, 256\}$. Proved that **$k=128$** yields the best model performance on multi-motion data, directly verifying the findings of Xue & Wu (2021).
- **Proposed ResMLP Architecture:** Designed an **Improved ResMLP** with residual skip connections, Layer Normalization, GELU activations, Dropout, and pose jitter data augmentation. Achieved a **42.5% reduction in Test MSE** over the paper baseline, bringing mean geometric error down to **sub-centimeter accuracy (9.66 mm)**.
- **Interactive 3D Demo & OBJ Export:** Built a live inference demo that runs on any test pose, outputs latency and per-vertex error metrics, and exports predicted and ground-truth 3D meshes as Wavefront `.obj` files.
>>>>>>> f50b985 (update: bigger dataset and k value optimization)

---

## Relationship to the Reference Paper

| Feature | Original Paper (Xue & Wu, 2021) | Our Implementation | Rationale / Difference |
|---|---|---|---|
| **Garment Type** | Loose Coat (PhysBAM procedural simulation) | T-Shirt (`t-shirt_female` physics simulation) | Reference coat dataset is proprietary; TailorNet is an open, authoritative 3D benchmark. |
| **Dataset Size** | 9,813 poses (6,000 train / 2,000 val / 1,813 test) | **2,122 frames** (1,697 train / 212 val / 213 test across 18 motions) | Multi-motion dataset covering running, jumping, bending, and stretching. |
| **Input Representation** | 14 joints &times; 4 quaternions = 56 dimensions | 24 SMPL joints &times; 4 quaternions = 96 dimensions | Converted TailorNet's 72-dim Rodrigues axis-angle to unit quaternions to prevent gimbal lock. |
| **Output Representation** | 2,170 vertices &times; 3 = 6,510 displacement dims | 7,702 vertices &times; 3 = 23,106 displacement dims | Full high-resolution garment mesh (15,180 triangular faces). |
| **PCA Dimensionality** | $k \in \{64, 128, 256\}$ ($k=128$ was optimal) | **$k \in \{32, 64, 128, 256\}$ ($k=128$ verified best)** | Fitted strictly on training split to eliminate data leakage. |
| **PC Reconstruction Layer** | Fixed linear layer: $\hat{x} = \hat{c} U + \mu$ | Fixed linear layer: $\hat{x} = \hat{c} U + \mu$ | Mathematically identical fixed PyTorch buffer layer. |
| **Baseline Architecture** | ID 0 (Linear) & ID 3 (`relu_128_256`) | Linear Baseline & Paper MLP (`relu_128_256`) | Exact reproduction of paper's feedforward MLP. |
| **Our Improvement** | None (paper used plain feedforward MLPs) | **ResMLP + LayerNorm + GELU + Dropout + Pose Aug** | Mitigates internal covariate shift, dead ReLUs, and overfitting. |

---

## 3. Dataset Information & Acquisition

We use the official [TailorNet Dataset](https://github.com/zycliao/TailorNet_dataset) hosted on [Hugging Face](https://huggingface.co/datasets/zycliao/TailorNet_dataset).

- **Data Assets:**
  - `data/extracted/t-shirt_female/pose/000_023/`: Contains 18 motion sequences (`000` to `017`), totaling **2,122 frames** of ground-truth cloth simulation.
  - `dataset_meta.zip`: Garment topology dictionary (`garment_class_info.pkl`), canonical garment bounds, and A-pose templates.
- **Sequence Streaming:**
  Because the full dataset zip is over 7 GB, `src/preprocessing/download_sequences.py` uses HTTP range requests to download only the 18 sequences for style/shape `000_023` (~160 MB total) directly on demand.

```bash
# Verify or download the 18 motion sequences if needed:
python src/preprocessing/download_sequences.py 18
```

---

##  Environment Setup & Dependencies

Python 3.10+ or 3.11 is recommended.

```bash
# 1. Clone repository
git clone <repo-url>
cd MLmini

# 2. Activate virtual environment
source .venv/bin/activate
# Or create a new one:
# python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

##  Running the Pipeline

<<<<<<< HEAD
###  Preprocessing & Dimensionality Reduction
Preprocesses poses to unit quaternions, splits data into 70 train / 20 val / 20 test, and computes PCA on train offsets:
=======
```
.
├── checkpoints/                   # Saved model weights & training history logs
│   ├── linear_baseline_best.pt
│   ├── paper_mlp_best.pt
│   ├── improved_resmlp_best.pt
│   └── improved_resmlp_plus_aug_best.pt
├── data/
│   ├── raw/                       # Raw archive files
│   ├── extracted/                 # Extracted TailorNet files (18 sequences)
│   └── processed/                 # Cached preprocessed dataset (dataset_cache.npz)
├── figures/                       # High-resolution evaluation figures
│   ├── loss_curves.png            # Training & validation convergence
│   ├── mesh_comparison_best.png   # Best predicted 3D mesh vs ground truth
│   ├── mesh_comparison_worst.png  # Worst predicted 3D mesh vs ground truth
│   ├── metrics_summary.png        # Benchmark comparison bar charts
│   ├── pca_analysis.png           # Cumulative variance & reconstruction error vs k
│   └── vertex_error_heatmap.png   # Spatial heatmap of vertex displacement error
├── outputs/                       # Benchmark outputs and demo exports
│   ├── demo_sample_0_gt.obj       # Ground truth 3D mesh (Wavefront OBJ)
│   ├── demo_sample_0_pred.obj     # Predicted 3D mesh (Wavefront OBJ)
│   ├── demo_sample_0_render.png   # 3-panel comparison render
│   ├── evaluation_results.json    # Detailed evaluation metrics
│   ├── evaluation_table.md        # Formatted benchmark markdown table
│   ├── pca_k_ablation.json        # Numerical PCA ablation results
│   └── pca_k_ablation.md          # Formatted PCA ablation markdown table
├── scripts/
│   ├── demo.py                    # Live inference & 3D OBJ export demo
│   ├── generate_report_pdf.py     # Script to generate the 2-page report PDF
│   └── run_pipeline.py            # Complete end-to-end training & evaluation pipeline
├── src/
│   ├── dataset/                   # PyTorch dataset & data loaders with augmentation
│   ├── evaluation/                # Evaluation metrics, latency benchmarks, tables
│   ├── models/                    # PCLayer, LinearBaseline, PaperMLP, ImprovedResMLP
│   ├── preprocessing/             # Pose quaternion conversion, PCA fitting & splitting
│   ├── training/                  # Training loop, early stopping, checkpointing
│   └── visualization/             # 3D mesh rendering and heatmap generation
├── report.pdf                     # Final 2-page academic project report (PDF)
├── requirements.txt               # Python package dependencies
└── README.md
```

---

## 6. Running the Pipeline

### Step 1: Preprocessing & Feature Extraction
Loads all 18 motion sequences (2,122 frames), converts SMPL axis-angle poses into 96-dimensional unit quaternions, splits into 1,697 train / 212 val / 213 test frames (80/10/10), and fits PCA on the training displacements:
>>>>>>> f50b985 (update: bigger dataset and k value optimization)

```bash
python src/preprocessing/extract_data.py --k-pca 256 --seed 42
```

<<<<<<< HEAD
###  Full End-to-End Pipeline
Trains all 4 models (Linear Baseline, Paper MLP, Improved ResMLP, Improved ResMLP + Aug), benchmarks inference latency, evaluates error metrics, and outputs all plots:
=======
### Step 2: Training & Benchmarking Pipeline
Runs the PCA dimensionality ablation across $k \in \{32, 64, 128, 256\}$, identifies $k=128$ as optimal, trains all 4 models (Linear Baseline, Paper MLP, Improved ResMLP, Improved ResMLP + Aug), evaluates metrics on the test set, and saves all plots:
>>>>>>> f50b985 (update: bigger dataset and k value optimization)

```bash
python scripts/run_pipeline.py --epochs 100 --batch-size 64
```

<<<<<<< HEAD
### Live Inference & 3D Garment Demo
Runs interactive inference on any test sample, measures latency in milliseconds, exports deformed 3D garment meshes to `.obj` format, and renders a 3D comparison plot:
=======
### Step 3: Interactive Live Inference Demo
Performs real-time inference on a selected test sample, measures latency, calculates mean/max vertex errors, exports deformed 3D garment meshes to `.obj` format, and renders a 3-panel comparison:
>>>>>>> f50b985 (update: bigger dataset and k value optimization)

```bash
python scripts/demo.py --sample-idx 0 --model-type improved
```

**Outputs generated:**
- `outputs/demo_sample_0_pred.obj`: Predicted deformed 3D garment mesh.
- `outputs/demo_sample_0_gt.obj`: Ground-truth simulated garment mesh.
- `outputs/demo_sample_0_render.png`: Render comparing Ground Truth, Predicted Mesh, and Displacement Difference.

The exported `.obj` files can be opened in Blender, MeshLab, or any standard 3D viewer.

<<<<<<< HEAD

##  Experimental Results & Benchmarks
=======
### Step 4: Compiling the 2-Page Report PDF
Generates the publication-ready 2-page project report:

```bash
python scripts/generate_report_pdf.py
```
This produces `report.pdf` in the root directory.

---

## 7. Experimental Results & Analysis
>>>>>>> f50b985 (update: bigger dataset and k value optimization)

### A. PCA Dimensionality ($k$) Ablation Study

**Why was $k=32$ used in initial prototypes?**  
In early testing with only 1 sequence (110 frames total, 70 training frames), the empirical data covariance matrix had rank at most $\min(N, D) = 70$. Attempting $k=128$ or $k=256$ on 70 frames caused rank deficiency and severe overfitting.

With the expanded 18-sequence dataset (**1,697 training frames**), we evaluated $k \in \{32, 64, 128, 256\}$ using the Paper MLP architecture:

| $k$ | Expl. Var (%) | Direct PCA MSE | Direct Err (mm) | Model Test MSE | Mean Err (mm) | Max Err (mm) | Trainable Params |
|---|---|---|---|---|---|---|---|
| **32** | 95.85% | $1.758 \times 10^{-5}$ | 6.00 mm | $8.366 \times 10^{-5}$ | 13.50 mm | 103.85 mm | 53,664 |
| **64** | 97.70% | $1.044 \times 10^{-5}$ | 4.54 mm | $8.120 \times 10^{-5}$ | 13.31 mm | 103.55 mm | 61,888 |
| **128** | **98.84%** | $5.877 \times 10^{-6}$ | 3.39 mm | **$8.112 \times 10^{-5}$** | **13.24 mm** | 108.38 mm | 78,336 |
| **256** | 99.52% | $2.934 \times 10^{-6}$ | 2.40 mm | $8.258 \times 10^{-5}$ | 13.42 mm | 107.90 mm | 111,232 |

**Key Finding:**  
$k=128$ achieves the lowest Model Test MSE ($8.112 \times 10^{-5}$) and lowest mean vertex error (13.24 mm). While $k=256$ slightly improves mathematical direct reconstruction, the extra degrees of freedom lead to overfitting in the neural network. This directly replicates the conclusion of Xue & Wu (2021).

---

### B. Core Model Evaluation Benchmarks ($k=128$)

Evaluated on the held-out test split of 213 unseen poses:

| Model Architecture | Trainable Params | Test MSE Loss | Mean Vertex Err (mm) | Max Vertex Err (mm) | Latency (GPU batched) |
|---|---:|---:|---:|---:|---:|
| **Linear Baseline (Paper ID 0)** | 12,416 | $1.188 \times 10^{-4}$ | 15.81 mm | 101.13 mm | 0.261 ms |
| **Paper MLP Baseline (Paper ID 3)** | 78,336 | $7.822 \times 10^{-5}$ | 13.02 mm | 98.55 mm | 0.339 ms |
| **Improved ResMLP (Ours)** | 323,968 | **$4.500 \times 10^{-5}$** | **9.66 mm** | 81.05 mm | 0.482 ms |
| **Improved ResMLP + Aug (Ours)** | 323,968 | **$4.543 \times 10^{-5}$** | **9.83 mm** | **75.43 mm** | 0.549 ms |

### Key Observations
1. **Significant Error Reduction:** Improved ResMLP cuts Test MSE by **42.5%** compared to the Paper MLP baseline ($7.822 \times 10^{-5} \to 4.500 \times 10^{-5}$) and brings mean geometric error down to **sub-centimeter accuracy (9.66 mm)** (a **25.8% reduction in geometric error**).
2. **Outlier Suppression via Pose Augmentation:** Injecting spherical pose jitter during training reduces the maximum peak vertex error from 98.55 mm down to **75.43 mm** (a **23.5% reduction in peak error**).
3. **Physical Error Distribution:** Per-vertex heatmaps reveal that shoulder and collar areas track the skeleton with minimal error (< 4 mm), while the unconstrained bottom hem has higher variance (up to 75 mm), mirroring the coattail dynamics reported in Xue & Wu.
4. **Real-Time Efficiency:** All models run in under **0.55 ms per frame** on GPU (> 1,800 FPS) and ~16.8 ms on CPU, comfortably satisfying real-time 60 FPS interactive requirements.


<<<<<<< HEAD
##  Limitations & Future Work
=======
## 8. Deliverables Summary

- **Source Code:** Complete modular package under `src/` and runners under `scripts/`.
- **Pretrained Checkpoints:** Saved under `checkpoints/`.
- **Evaluation Visualizations:** Loss curves, heatmaps, and mesh comparisons in `figures/`.
- **Benchmark Tables:** Stored in `outputs/evaluation_table.md` and `outputs/pca_k_ablation.md`.
- **3D Assets:** Ground truth and predicted `.obj` files in `outputs/`.
- **Project Report:** 2-page academic report in `report.pdf`.

---

## 9. Teacher Demonstration Quick Guide
>>>>>>> f50b985 (update: bigger dataset and k value optimization)

When demonstrating the project during review:

1. **Activate the environment and run the demo:**
   ```bash
   source .venv/bin/activate
   python scripts/demo.py --sample-idx 0 --model-type improved
   ```
2. **What to point out on screen:**
   - Input: 24 joint rotation quaternions (96 floats).
   - Output: 7,702 vertices deformed in 3D (23,106 floats).
   - Fast inference time (< 20 ms CPU / < 0.5 ms GPU).
   - Sub-centimeter mean vertex error (~8.95 mm on Sample 0).
3. **What figures to show:**
   - `outputs/demo_sample_0_render.png` for visual fidelity.
   - `figures/loss_curves.png` showing rapid, stable ResMLP convergence.
   - `figures/pca_analysis.png` explaining why $k=128$ was chosen.
   - `figures/vertex_error_heatmap.png` explaining why the hem has higher error than the shoulders.
4. **Report:**
   - Present `report.pdf` as the 2-page project summary.
