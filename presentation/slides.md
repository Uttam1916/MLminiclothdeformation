# Data-Driven 3D Garment Deformation from Human Pose
## UE24CS352A - Machine Learning Mini-Project Final Review
**Based on:** *Data-Driven Clothing for Interactive Applications* (Xue & Wu, Stanford CS 229)  
**Alternative Dataset:** TailorNet (CVPR 2020, Max Planck Institute)  
**Team:** 2-Member Student Team  

---

## Slide 1: Problem & Motivation
- **The Challenge:** Simulating realistic clothing in real-time is crucial for gaming, VR, animation, and virtual try-on.
- **The Bottleneck:** Traditional physics-based cloth simulation (PDE solvers like PhysBAM) takes **~180 seconds per frame**.
- **The Goal:** Build a data-driven neural network pipeline that predicts 3D garment mesh deformation directly from human body pose in **sub-millisecond time (< 1 ms)** while retaining visual fidelity.

---

## Slide 2: Original Research Paper (Xue & Wu, 2021)
- **Reference:** *Data-Driven Clothing for Interactive Applications* by Kangrui Xue and Jane Wu.
- **Approach:**
  - Procedural simulation of a loose coat across 10,000 poses (9,813 after filtering).
  - Joint rotation quaternions as input (14 joints × 4 = 56 dimensions).
  - MLP (`relu_128_256`) predicting PCA coefficients.
  - Fixed "PC Layer" ($x = cU + \mu$) reconstructing 6,510 vertex offset dimensions (2,170 vertices × 3).
  - Achieved test MSE of $1.498 \times 10^{-4}$ and **~100,000× speedup** over PhysBAM.

---

## Slide 3: Dataset Unavailability & Alternative Selection
- **The Constraint:** The original paper's synthetic coat simulation dataset is proprietary and not publicly available.
- **Our Solution:** We adopted the official public **TailorNet dataset** (CVPR 2020, Max Planck Institute for Informatics).
- **TailorNet Dataset Details:**
  - Curated female t-shirt physics simulation sequence.
  - 110 simulation frames under diverse SMPL poses.
  - High-resolution cloth mesh: **7,702 vertices** and **15,180 triangular faces** (23,106 displacement dimensions).
  - Exact match with paper's formulation: predicting 3D vertex displacements from human joint rotations.

---

## Slide 4: Data Adaptation & Preprocessing Pipeline
1. **Pose Representation:** SMPL axis-angle vectors (72 dims) converted to unit quaternions (24 joints × 4 = **96 dimensions**), preserving orientation continuity and preventing gimbal lock.
2. **Target Representation:** 3D vertex displacements from canonical template ($7,702 \times 3 = 23,106$ dimensions).
3. **Reproducible Split:** 70 Train (~64%), 20 Validation (~18%), 20 Test (~18%) with random seed 42.
4. **PCA Output Bottleneck:**
   - Fitted strictly on the 70 training samples to prevent data leakage.
   - $k=32$ principal components captures **99.49% of total explained variance** with test reconstruction MSE of $1.15 \times 10^{-5}$.

---

## Slide 5: Model Architectures
- **Linear Baseline (Paper ID 0):**
  - $\text{Input}(96) \to \text{Linear}(96, 32) \to \text{Fixed PC Layer} \to \hat{x} \in \mathbb{R}^{23,106}$.
- **Paper MLP Baseline (Paper ID 3):**
  - $\text{Linear}(96, 128) \to \text{ReLU} \to \text{Linear}(128, 256) \to \text{ReLU} \to \text{Linear}(256, 32) \to \text{PC Layer}$.
- **Our Proposed Improvement: ResMLP with Normalization & Regularization:**
  - **Layer Normalization:** Prevents internal covariate shift and stabilizes hidden states.
  - **GELU Activations:** Smooth non-linearities prevent dead neurons on quaternion inputs.
  - **Residual Skip Connections:** $h_{l+1} = h_l + \mathcal{F}(h_l)$ ensures direct gradient flow across blocks.
  - **Dropout ($p=0.1$):** Prevents co-adaptation and overfitting on pose features.
  - **Pose Jitter Augmentation:** Training with spherical Gaussian quaternion perturbations ($\sigma = 0.015$).

---

## Slide 6: Quantitative Experimental Results

| Model Architecture | Trainable Params | Test MSE Loss | Mean Vertex Err | Max Vertex Err | Latency |
|---|---:|---:|---:|---:|---:|
| Linear Baseline | 3,104 | $1.543 \times 10^{-4}$ | 17.57 mm | 72.56 mm | 0.078 ms |
| Paper MLP Baseline | 53,664 | $1.242 \times 10^{-4}$ | 15.95 mm | 83.90 mm | 0.096 ms |
| **Improved ResMLP (Ours)** | 299,296 | **$0.851 \times 10^{-4}$** | **11.99 mm** | **72.15 mm** | 0.299 ms |
| **Improved ResMLP + Aug (Ours)** | 299,296 | **$0.802 \times 10^{-4}$** | **11.64 mm** | **69.12 mm** | 0.236 ms |

- **Key Finding:** ResMLP + Aug reduces test MSE by **35.4%** and average geometric error from 15.95 mm to 11.64 mm (**27.0% error reduction**) over the paper baseline.

---

## Slide 7: Visual Analysis & Error Heatmaps
- **Heatmap Observations:**
  - Tight areas (shoulders, neckline) have low error (< 4 mm) as they follow skeletal skinning.
  - Loose regions (bottom hem, waist) show higher deformation variance (up to 69 mm).
  - This exactly mirrors Xue & Wu's finding where loose coattails exhibited higher variance than the shoulders.
- **Inference Speed:** All models execute in **< 0.3 ms per frame** (> 3,300 FPS), meeting interactive standards.

---

## Slide 8: Live Demonstration Workflow
1. Run `python scripts/demo.py --sample-idx 0`
2. Real-time inference executed on test pose quaternion vector.
3. Deformed 3D garment exported as Wavefront `.obj` files:
   - `outputs/demo_sample_0_pred.obj` (predicted 3D garment)
   - `outputs/demo_sample_0_gt.obj` (ground truth simulated garment)
4. Visual comparison plot rendered to `outputs/demo_sample_0_render.png`.

---

## Slide 9: Conclusion & Limitations
- **Conclusions:**
  - Successfully reproduced and validated the paper's pose-to-cloth PCA framework on TailorNet.
  - Demonstrated that Residual blocks, LayerNorm, GELU, and pose jittering significantly improve generalization over plain MLPs.
  - Achieved real-time inference latency of 0.24 ms per mesh.
- **Limitations:**
  - Evaluated on static equilibrium poses (no temporal velocity / momentum).
  - Future Work: Temporal modeling (GRUs / temporal convolutions) and physics-informed Laplacian edge losses.
