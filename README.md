# Data-Driven 3D Garment Deformation from Human Pose

**Course:** UE24CS352A - Machine Learning Mini-Project  
**Topic:** Reproduction and Extension of *Data-Driven Clothing for Interactive Applications* (Xue & Wu, Stanford CS 229)  
**Alternative Dataset:** TailorNet Dataset (CVPR 2020, Max Planck Institute for Informatics)  

---

## 1. Project Overview

Simulating cloth physics using classical partial differential equation solvers (e.g., PhysBAM finite element or mass-spring systems) takes multiple minutes per frame (~180 seconds in the reference paper), making interactive 3D applications (VR, gaming, character animation, and virtual try-on) intractable.

This project implements a data-driven neural network pipeline to predict 3D cloth deformation directly from human body skeletal poses in real-time (< 0.3 ms per frame, over 3,300 FPS).

### Key Highlights
- **Faithful Methodological Reproduction:** Reproduces the core architecture of Xue & Wu (2021) featuring joint rotation quaternions as inputs, PCA dimensionality reduction on vertex displacements, and a fixed Principal Component (PC) reconstruction layer.
- **Public Dataset Adaptation:** Because the original paper's proprietary PhysBAM coat dataset is not publicly available, we adapt the official public TailorNet dataset (`t-shirt_female` simulation sequence).
- **Novel Architectural Improvement:** Proposes an **Improved ResMLP** with Residual skip connections, Layer Normalization, GELU activations, Dropout, and physical pose jitter data augmentation, achieving a **35.4% reduction in test MSE** and a **27.0% reduction in 3D geometric error** compared to the paper baseline.
- **End-to-End Reproducibility:** Includes full pipeline scripts, live 3D inference demo, Wavefront `.obj` mesh exports, a two-page academic report (PDF & Markdown), and a presentation slide deck (PDF & Markdown).

---

## 2. Relationship to the Reference Paper

| Feature | Original Paper (Xue & Wu, 2021) | Our Implementation | Rationale / Note |
|---|---|---|---|
| **Garment Type** | Loose Coat (PhysBAM procedural simulation) | T-Shirt (`t-shirt_female` physics simulation) | Paper dataset is proprietary; TailorNet is an open, authoritative benchmark. |
| **Input Representation** | 14 joints × 4 quaternions = 56 dimensions | 24 SMPL joints × 4 quaternions = 96 dimensions | Converted TailorNet's 72-dim Rodrigues axis-angle to unit quaternions to prevent gimbal lock. |
| **Output Representation** | 2,170 vertices × 3 = 6,510 displacement dims | 7,702 vertices × 3 = 23,106 displacement dims | Full high-resolution garment mesh (15,180 triangular faces). |
| **PCA Reduction** | $k \in \{64, 128, 256\}$ ($k=128$ was best) | $k=32$ (captures 99.49% of variance, MSE = $1.15 \times 10^{-5}$) | Fitted strictly on training split to eliminate data leakage. |
| **PC Reconstruction Layer** | Fixed linear layer: $\hat{x} = \hat{c} U + \mu$ | Fixed linear layer: $\hat{x} = \hat{c} U + \mu$ | Mathematically identical fixed PyTorch buffer layer. |
| **Baseline Architecture** | ID 0 (Linear) & ID 3 (`relu_128_256`) | Linear Baseline & Paper MLP (`relu_128_256`) | Exact reproduction of paper's feedforward MLP. |
| **Our Improvement** | None (paper used plain feedforward MLPs) | **ResMLP + LayerNorm + GELU + Dropout + Pose Aug** | Addresses internal covariate shift, dead ReLUs, and overfitting. |

---

## 3. Dataset Information & Setup

We use the official [TailorNet Dataset](https://github.com/zycliao/TailorNet_dataset) hosted on [Hugging Face](https://huggingface.co/datasets/zycliao/TailorNet_dataset).

- **Files Used:**
  - `t-shirt_female_sample.zip` (18.9 MB): Curated sample containing 110 simulation frames, body shape $\beta$, garment style $\gamma$, style model, and unposed vertex displacements.
  - `dataset_meta.zip` (5.4 MB): Topology dictionary (`garment_class_info.pkl`), A-pose definition, and canonical bounds.
- **License:** Non-commercial scientific research license (Max-Planck-Gesellschaft).

### Dataset Download & Extraction
The project downloads and extracts the data automatically if missing, or you can run:

```bash
mkdir -p data/raw data/extracted
# Download official sample files (total ~24 MB)
curl -L "https://huggingface.co/datasets/zycliao/TailorNet_dataset/resolve/main/dataset_meta.zip" -o data/raw/dataset_meta.zip
curl -L "https://huggingface.co/datasets/zycliao/TailorNet_dataset/resolve/main/t-shirt_female_sample.zip" -o data/raw/t-shirt_female_sample.zip

# Extract
unzip -q data/raw/dataset_meta.zip -d data/extracted/
unzip -q data/raw/t-shirt_female_sample.zip -d data/extracted/
```

---

## 4. Environment Setup & Dependencies

Python 3.10+ or 3.11 is recommended.

```bash
# 1. Clone repository
git clone <repo-url>
cd MLmini

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 5. Project Structure

```
.
├── checkpoints/               # Saved model weights & training history logs
│   ├── linear_baseline_best.pt
│   ├── paper_mlp_best.pt
│   ├── improved_resmlp_best.pt
│   └── improved_resmlp_plus_aug_best.pt
├── data/
│   ├── raw/                   # Downloaded raw zip archives (ignored by git)
│   ├── extracted/             # Extracted TailorNet files (ignored by git)
│   └── processed/             # Cached preprocessed dataset (dataset_cache.npz)
├── figures/                   # Generated high-resolution publication figures
│   ├── loss_curves.png
│   ├── mesh_comparison_best.png
│   ├── mesh_comparison_worst.png
│   ├── metrics_summary.png
│   ├── pca_analysis.png
│   └── vertex_error_heatmap.png
├── outputs/                   # Benchmark results and demo 3D OBJ exports
│   ├── demo_sample_0_gt.obj
│   ├── demo_sample_0_pred.obj
│   ├── demo_sample_0_render.png
│   ├── evaluation_results.json
│   └── evaluation_table.md
├── presentation/              # Final review slide deck
│   ├── generate_slides_pdf.py # ReportLab slide compiler
│   ├── presentation.pdf       # 10-slide landscape presentation PDF
│   └── slides.md              # Markdown presentation slides
├── report/                    # Final two-page project write-up
│   ├── generate_pdf.py        # ReportLab report compiler
│   ├── report.pdf             # Two-page academic report PDF
│   └── report.md              # Markdown source report
├── requirements.txt           # Project dependencies
├── scripts/
│   ├── demo.py                # Real-time inference & 3D OBJ export demo
│   └── run_pipeline.py        # Complete end-to-end training & benchmarking runner
└── src/
    ├── dataset/               # PyTorch dataset & data loaders with augmentation
    ├── evaluation/            # Evaluation metrics, latency benchmarks & tables
    ├── models/                # PCLayer, LinearBaseline, PaperMLP, ImprovedResMLP
    ├── preprocessing/         # Pose quaternion conversion, PCA fitting & splitting
    ├── training/              # Training loop, early stopping, and checkpointing
    └── visualization/         # 3D mesh rendering and heatmap generation
```

---

## 6. Running the Pipeline

### Step 1: Preprocessing & Dimensionality Reduction
Preprocesses poses to unit quaternions, splits data into 70 train / 20 val / 20 test, and computes PCA on train offsets:

```bash
python src/preprocessing/extract_data.py --k-pca 32 --seed 42
```

### Step 2: Full End-to-End Pipeline
Trains all 4 models (Linear Baseline, Paper MLP, Improved ResMLP, Improved ResMLP + Aug), benchmarks inference latency, evaluates error metrics, and outputs all plots:

```bash
python scripts/run_pipeline.py --epochs 800 --k-pca 32
```

### Step 3: Live Inference & 3D Garment Demo
Runs interactive inference on any test sample, measures latency in milliseconds, exports deformed 3D garment meshes to `.obj` format, and renders a 3D comparison plot:

```bash
python scripts/demo.py --sample-idx 0 --model-type improved
```

Generated 3D assets:
- `outputs/demo_sample_0_pred.obj`: Predicted 3D garment mesh.
- `outputs/demo_sample_0_gt.obj`: Ground truth physics simulated mesh.
- `outputs/demo_sample_0_render.png`: 3-panel visual render.

These `.obj` files can be opened in Blender, MeshLab, or any standard 3D viewer.

### Step 4: Compiling Deliverables
Generate the two-page write-up and presentation slide deck PDFs:

```bash
# Two-page project report PDF
python report/generate_pdf.py

# Landscape presentation slide deck PDF
python presentation/generate_slides_pdf.py
```

---

## 7. Experimental Results & Benchmarks

All models were evaluated on the held-out test split of 20 unseen poses:

| Model Architecture | Trainable Params | Test MSE Loss | Mean Vertex Err (mm) | Max Vertex Err (mm) | Inference Latency (ms) |
|---|---:|---:|---:|---:|---:|
| **Linear Baseline (Paper ID 0)** | 3,104 | $1.543 \times 10^{-4}$ | 17.57 mm | 72.56 mm | 0.078 ms |
| **Paper MLP Baseline (Paper ID 3)** | 53,664 | $1.242 \times 10^{-4}$ | 15.95 mm | 83.90 mm | 0.096 ms |
| **Improved ResMLP (Ours)** | 299,296 | **$0.851 \times 10^{-4}$** | **11.99 mm** | **72.15 mm** | 0.299 ms |
| **Improved ResMLP + Aug (Ours)** | 299,296 | **$0.802 \times 10^{-4}$** | **11.64 mm** | **69.12 mm** | 0.236 ms |

### Key Findings
1. **Accuracy Gain:** Our proposed ResMLP with LayerNorm, GELU, and Dropout reduces test MSE by **31.5%** over the paper baseline. Adding spherical pose jitter data augmentation brings total MSE reduction to **35.4%** ($1.242 \times 10^{-4} \to 0.802 \times 10^{-4}$), lowering mean 3D geometric error from 15.95 mm to 11.64 mm (**27.0% error reduction**).
2. **Physical Error Heatmap:** Errors are concentrated along the lower hem of the shirt (up to 69 mm) where fabric hangs and folds freely under gravity, whereas shoulders and collar have minimal error (< 4 mm) due to tight coupling with skeletal joints. This replicates the exact physical phenomenon reported by Xue & Wu on loose coattails.
3. **Real-Time Speed:** Inference latency across all models is **< 0.3 ms** (> 3,300 FPS), delivering a **~600,000× speedup** over procedural simulation.

---

## 8. Deliverables Summary

- **Code:** Hosted in this repository with clean, modular architecture.
- **Write-Up:** [report/report.pdf](report/report.pdf) (concise 2-page academic report) and [report/report.md](report/report.md).
- **Presentation:** [presentation/presentation.pdf](presentation/presentation.pdf) (10-slide landscape presentation) and [presentation/slides.md](presentation/slides.md).
- **Checkpoints:** Stored in `checkpoints/`.
- **Figures:** Stored in `figures/`.

---

## 9. Limitations & Future Work

- **Static Equilibrium:** The current dataset models equilibrium deformation per pose, omitting dynamic velocity and cloth inertia over continuous movement sequences.
- **Future Directions:** Integrating temporal sequence models (GRUs, temporal convolutions) to predict frame-to-frame cloth dynamics, and incorporating physics-informed Laplacian surface loss functions to regularize inter-vertex curvature.
