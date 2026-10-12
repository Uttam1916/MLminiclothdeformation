# Data-Driven 3D Garment Deformation from Human Pose

**SRNs:** PES1UG24CS697, PES1UG24CS698  
**Course:** UE24CS352A - Machine Learning Mini-Project  
**Topic:** Reproduction and Extension of *Data-Driven Clothing for Interactive Applications* (Xue & Wu, Stanford CS 229)  
**Dataset:** TailorNet Benchmark (CVPR 2020, Max Planck Institute for Informatics)  

---

## 1. Overview

Physics-based cloth simulations (e.g., PhysBAM) take several minutes per frame (~180 s), making them too slow for interactive applications like games and VR. This project implements a machine learning pipeline that predicts 3D garment mesh deformation directly from human body skeletal poses in real time (< 0.55 ms per frame).

We reproduce the core approach from Xue & Wu (2021) and extend it:
- Replaced the initial 110-frame subset with all **18 motion sequences (2,122 frames)** from TailorNet.
- Ran a systematic PCA ablation across $k \in \{32, 64, 128, 256\}$, confirming that **$k=128$** is optimal.
- Designed an **Improved ResMLP** (Residual blocks + LayerNorm + GELU + Dropout + Pose Jitter Augmentation), achieving a **42.5% reduction in Test MSE** and **sub-centimeter mean vertex error (9.66 mm)**.

---

## 2. Comparison with Reference Paper

| Feature | Reference Paper (Xue & Wu, 2021) | Our Implementation | Notes |
|---|---|---|---|
| **Garment** | Loose Coat (PhysBAM simulation) | T-Shirt (`t-shirt_female`) | Paper dataset is proprietary; TailorNet is an open benchmark. |
| **Data Size** | 9,813 frames (synthetic coat) | 2,122 frames across 18 motions | 1,697 train / 212 val / 213 test frames (80/10/10 split). |
| **Input** | 14 joints &times; 4 quaternions = 56 dims | 24 SMPL joints &times; 4 quaternions = 96 dims | Unit quaternions prevent gimbal lock. |
| **Output** | 2,170 vertices &times; 3 = 6,510 dims | 7,702 vertices &times; 3 = 23,106 dims | Predicts 3D displacement from canonical template. |
| **PCA Components** | $k=128$ (chosen out of 64, 128, 256) | $k=128$ (evaluated 32, 64, 128, 256) | Fitted strictly on training split. |
| **Baseline Model** | ID 0 (Linear) & ID 3 (`relu_128_256`) | Linear Baseline & Paper MLP | Exact reproduction of paper's architectures. |
| **Our Model** | None | ResMLP + LayerNorm + GELU + Aug | Mitigates vanishing gradients and dead ReLUs. |

---

## 3. Setup

Requires Python 3.10 or 3.11.

```bash
# Clone and enter repo
git clone <repo-url>
cd MLmini

# Activate virtual environment
source .venv/bin/activate
# or: python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Running the Code

### Step 1: Preprocessing & PCA Extraction
Processes the 18 TailorNet sequences, converts SMPL poses to unit quaternions, splits the dataset (80/10/10), and computes PCA on the training displacements:
```bash
python src/preprocessing/extract_data.py --k-pca 256 --seed 42
```

### Step 2: Training & Benchmarking Pipeline
Runs the PCA dimensionality ablation ($k \in \{32, 64, 128, 256\}$), trains all 4 models at $k=128$, evaluates metrics on the test set, and generates all figures:
```bash
python scripts/run_pipeline.py --epochs 100 --batch-size 64
```

### Step 3: Interactive 3D Demo & OBJ Export
Runs real-time inference on a test pose, measures latency, calculates vertex error, and exports `.obj` 3D meshes:
```bash
python scripts/demo.py --sample-idx 0 --model-type improved
```

Outputs generated in `outputs/`:
- `demo_sample_0_pred.obj`: Predicted 3D garment mesh.
- `demo_sample_0_gt.obj`: Ground-truth simulated garment mesh.
- `demo_sample_0_render.png`: 3-panel comparison render.

### Step 4: Generate the 2-Page Report PDF
Compiles the clean, academic black-and-white 2-page report:
```bash
python scripts/generate_report_pdf.py
```
Output: `report.pdf`.

---

## 5. Experimental Results

### A. PCA Subspace ($k$) Ablation

Earlier code used $k=32$ because 70 training frames caused empirical rank deficiency ($\text{rank} \le 70$). With 1,697 training frames, we tested $k \in \{32, 64, 128, 256\}$ using the Paper MLP:

| $k$ | Expl. Var (%) | Direct PCA MSE | Direct Err (mm) | Model Test MSE | Mean Err (mm) | Max Err (mm) |
|---|---|---|---|---|---|---|
| **32** | 95.85% | $1.758 \times 10^{-5}$ | 6.00 mm | $8.366 \times 10^{-5}$ | 13.50 mm | 103.85 mm |
| **64** | 97.70% | $1.044 \times 10^{-5}$ | 4.54 mm | $8.120 \times 10^{-5}$ | 13.31 mm | 103.55 mm |
| **128** | **98.84%** | $5.877 \times 10^{-6}$ | 3.39 mm | **$8.112 \times 10^{-5}$** | **13.24 mm** | 108.38 mm |
| **256** | 99.52% | $2.934 \times 10^{-6}$ | 2.40 mm | $8.258 \times 10^{-5}$ | 13.42 mm | 107.90 mm |

**Takeaway:** $k=128$ gives the lowest Model Test MSE ($8.112 \times 10^{-5}$) and lowest mean error (13.24 mm). $k=256$ has slightly better linear reconstruction but adds extra degrees of freedom that cause the MLP to overfit.

---

### B. Model Benchmarks ($k=128$)

Evaluated on the held-out test split of 213 unseen multi-motion poses:

| Model Architecture | Trainable Params | Test MSE | Mean Vertex Err | Max Vertex Err | Latency (GPU) | Latency (CPU) |
|---|---:|---:|---:|---:|---:|---:|
| **Linear Baseline (Paper ID 0)** | 12,416 | $1.188 \times 10^{-4}$ | 15.81 mm | 101.13 mm | 0.261 ms | ~14.2 ms |
| **Paper MLP Baseline (Paper ID 3)** | 78,336 | $7.822 \times 10^{-5}$ | 13.02 mm | 98.55 mm | 0.339 ms | ~15.1 ms |
| **Improved ResMLP (Ours)** | 323,968 | **$4.500 \times 10^{-5}$** | **9.66 mm** | 81.05 mm | 0.482 ms | ~16.8 ms |
| **Improved ResMLP + Aug (Ours)** | 323,968 | **$4.543 \times 10^{-5}$** | **9.83 mm** | **75.43 mm** | 0.549 ms | ~17.0 ms |

### Key Findings
1. **Accuracy:** Improved ResMLP reduces Test MSE by **42.5%** over the paper baseline ($7.822 \times 10^{-5} \to 4.500 \times 10^{-5}$) and brings mean geometric error to **sub-centimeter accuracy (9.66 mm)** (a **25.8% reduction**).
2. **Robustness:** Pose Jitter Augmentation ($\sigma=0.015$ with unit normalization) reduces worst-case peak vertex error from 98.55 mm to **75.43 mm** (a **23.5% reduction**).
3. **Physical Error Distribution:** Per-vertex error heatmaps show shoulder and neck areas have minimal error (< 4 mm) because they track the skeleton closely, while the unconstrained hem has higher error (up to 75 mm), matching the coattail findings in Xue & Wu.
4. **Latency:** All models run in under **0.55 ms** on GPU (> 1,800 FPS) and under **17 ms** on CPU (> 58 FPS), well within real-time requirements.

---

## 6. Project Structure

```
.
├── figures/                       # Evaluation plots (loss curves, heatmaps, PCA analysis)
├── outputs/                       # Benchmark tables (.md, .json) and demo OBJ meshes
├── scripts/
│   ├── demo.py                    # Interactive inference demo & OBJ exporter
│   ├── generate_report_pdf.py     # Script that builds the 2-page report.pdf
│   └── run_pipeline.py            # End-to-end training and evaluation runner
├── src/
│   ├── dataset/                   # Dataset loader and pose jitter augmentation
│   ├── evaluation/                # Evaluation metrics and benchmark tables
│   ├── models/                    # PCLayer, LinearBaseline, PaperMLP, ImprovedResMLP
│   ├── preprocessing/             # Pose quaternion conversion and PCA fitting
│   ├── training/                  # Training loop with early stopping
│   └── visualization/             # 3D mesh rendering and heatmap generation
├── report.pdf                     # 2-page academic project report
├── requirements.txt
└── README.md
```
