# Data-Driven 3D Garment Deformation from Human Pose
**Course:** UE24CS352A - Machine Learning Mini-Project  
**Topic:** Reproduction and Extension of Xue & Wu (2021) using the TailorNet Dataset  

---

### 1. Problem Statement
Simulating realistic cloth dynamics in interactive settings such as 3D animation, gaming, and virtual try-on is computationally challenging. Traditional physics-based cloth simulation algorithms (such as PhysBAM using mass-spring or finite element equations) take dozens of seconds to several minutes per frame (~180 seconds in the reference paper). This high computational cost makes procedural physics solvers impractical for real-time applications. Data-driven machine learning models offer a promising alternative by learning a direct non-linear mapping from human body pose to 3D garment mesh deformation, achieving orders-of-magnitude speedups while preserving visually plausible cloth wrinkles and drape.

### 2. Original Paper Overview
In the reference study *"Data-Driven Clothing for Interactive Applications"* (Xue & Wu, Stanford CS 229, 2021), the authors addressed real-time prediction of loose coat meshes from skeletal poses. They employed a two-stage pipeline:
1. Procedural physics simulation to generate ground truth deformed coat meshes across diverse poses.
2. Supervised learning using a fully connected multilayer perceptron (MLP) with a fixed Principal Component Analysis (PCA) reconstruction layer to predict 3D vertex offsets from input joint quaternions.

Their model took 14 joint rotation quaternions (56 dimensions) as input and predicted 128 PCA coefficients, which were passed through a fixed linear "PC layer" to reconstruct 6,510 vertex offset dimensions (2,170 vertices × 3 coordinates). They evaluated linear regression against various MLP depths, finding that a two-layer network (`relu_128_256`) with 128 PCA components achieved the lowest test MSE ($1.498 \times 10^{-4}$) and ran in approximately 0.107 ms on GPU—over 100,000× faster than procedural simulation.

### 3. Original Dataset & Unavailability
The original paper relied on a proprietary synthetic coat dataset created specifically by the authors using PhysBAM and Blender Linear Blend Skinning (LBS), comprising 9,813 filtered poses. Because this coat dataset and its procedural generation pipeline are not publicly released, we could not train or evaluate on the original coat data.

### 4. Alternative Dataset: TailorNet
To faithfully evaluate the paper's core hypothesis without fabricating data, we adopted the publicly available TailorNet dataset (CVPR 2020, Max Planck Institute for Informatics). TailorNet provides physically simulated 3D garment deformations across SMPL body poses. We utilized the official curated `t-shirt_female` sample (19 MB) along with `dataset_meta.zip`, which contains:
- 110 simulation frames of human body poses and simulated 3D clothing.
- Pose representation: 24 SMPL joint rotation vectors (72 dimensions).
- Garment representation: 3D vertex displacements for a female t-shirt consisting of 7,702 vertices and 15,180 triangular faces (23,106 displacement dimensions).
- Canonical garment template derived from the TailorNet style PCA model.

### 5. Dataset Adaptation & Preprocessing
To align TailorNet with the methodology of Xue & Wu, we implemented the following adaptations:
1. **Pose Representation:** TailorNet provides SMPL poses in axis-angle Rodrigues vectors ($24 \times 3 = 72$ dimensions). Following the paper's design principle that unit quaternions provide a smooth, physically meaningful parameterization without gimbal lock, we converted each joint's rotation into unit quaternions ($w, x, y, z$), yielding a 96-dimensional input vector.
2. **Output Representation:** The target deformation is the 3D vertex displacement vector from the canonical garment mesh to the simulated shape ($7,702 \times 3 = 23,106$ dimensions).
3. **Reproducible Splitting:** The 110 simulation frames were split into 70 training samples (~64%), 20 validation samples (~18%), and 20 test samples (~18%) using random seed 42.
4. **PCA Dimensionality Reduction:** We fitted PCA strictly on the 70 training displacement vectors to avoid data leakage. Capturing $k=32$ principal components retains 99.49% of total explained variance with an intrinsic reconstruction MSE of $1.15 \times 10^{-5}$ on unseen test data.

### 6. Baseline and Proposed Improvements
We implemented two baselines matching the paper, followed by our architectural improvements:
- **Linear Baseline (Paper ID 0):** A single linear transformation mapping the 96-dimensional quaternion input directly to 32 PCA coefficients, followed by the fixed PC reconstruction layer.
- **Paper MLP Baseline (Paper ID 3):** The primary neural architecture from Xue & Wu: `Linear(96, 128) -> ReLU -> Linear(128, 256) -> ReLU -> Linear(256, 32)` connected to the fixed PC layer ($x = cU + \mu$).
- **Improved Model (ResMLP with LayerNorm, GELU, and Dropout):** The baseline MLP suffers from two issues: standard ReLU neurons can become inactive for negative quaternion coordinates, and feedforward networks without normalization can overfit on moderate datasets. Our improved model introduces:
  1. Input projection to 256 dimensions with Layer Normalization and GELU activation.
  2. Two residual blocks with pre-LayerNorm, GELU non-linearities, and Dropout ($p=0.1$) to prevent co-adaptation.
  3. Residual skip connections ($h_{l+1} = h_l + \mathcal{F}(h_l)$) that preserve direct gradient propagation.
- **Improved Model with Pose Augmentation:** We also evaluated our ResMLP trained with physical pose jittering—injecting Gaussian perturbation ($\sigma = 0.015$) into joint quaternions followed by spherical re-normalization during training to improve robustness against pose variations.

### 7. Experimental Setup
All models were trained using the Adam optimizer with mean squared error (MSE) loss calculated directly on the reconstructed 23,106-dimensional vertex offsets. Training ran for 800 epochs with a batch size of 32, a learning rate of $1 \times 10^{-4}$ ($1 \times 10^{-3}$ for the linear model), and checkpointing based on validation loss. We measured test MSE loss, mean and maximum per-vertex 3D Euclidean error (in millimeters), inference latency per sample (averaged over 200 runs), and parameter counts.

### 8. Results and Comparison

| Model | Trainable Parameters | Test MSE Loss | Mean Vertex Error (mm) | Max Vertex Error (mm) | Inference Latency (ms) |
|---|---:|---:|---:|---:|---:|
| Linear Baseline | 3,104 | $1.543 \times 10^{-4}$ | 17.57 mm | 72.56 mm | 0.078 ms |
| Paper MLP Baseline | 53,664 | $1.242 \times 10^{-4}$ | 15.95 mm | 83.90 mm | 0.096 ms |
| **Improved ResMLP** | 299,296 | **$0.851 \times 10^{-4}$** | **11.99 mm** | **72.15 mm** | 0.299 ms |
| **Improved ResMLP + Aug** | 299,296 | **$0.802 \times 10^{-4}$** | **11.64 mm** | **69.12 mm** | 0.236 ms |

### 9. Discussion & Error Analysis
1. **Geometric Fidelity:** The Paper MLP baseline improved upon the linear baseline by 19.5% in MSE loss (from $1.543 \times 10^{-4}$ to $1.242 \times 10^{-4}$) and reduced average vertex error to 15.95 mm. However, our proposed ResMLP architecture achieved a much larger gain: test MSE dropped by 31.5% compared to the paper MLP (down to $0.851 \times 10^{-4}$), and adding pose augmentation further reduced it to $0.802 \times 10^{-4}$ (a total 35.4% improvement over the paper baseline). Mean Euclidean vertex error was reduced from 15.95 mm to 11.64 mm.
2. **Physical Error Distribution:** Per-vertex error heatmaps reveal that errors are non-uniformly distributed across garment geometry. Vertices around the neck, upper chest, and shoulders exhibit very small errors (< 4 mm) because these regions closely follow skeletal joints. In contrast, the lower hem and waist of the t-shirt exhibit larger errors (up to 69 mm) because loose fabric hangs and folds freely under gravity. This physical behavior directly corroborates the findings of Xue & Wu, who observed that coattails showed substantially higher error than the upper coat.
3. **Runtime Performance:** All neural models run in under 0.3 ms per sample (> 3,300 frames per second), confirming that data-driven neural cloth deformation provides real-time performance suitable for interactive VR and gaming environments.

### 10. Conclusion & Limitations
We successfully reproduced the core methodology of Xue & Wu (2021) using the open TailorNet dataset. By predicting PCA coefficients and reconstructing mesh offsets with a fixed PC layer, neural networks capture realistic garment deformations with sub-millisecond execution times. Our architectural extension—incorporating residual blocks, LayerNorm, GELU, and pose jitter augmentation—significantly outperforms the paper's plain feedforward baseline, reducing mean vertex error by 27.0%.

**Limitations:** The dataset focuses on quasi-static simulations of isolated poses, omitting temporal velocity and dynamic cloth inertia across continuous animations. Future extensions should incorporate sequence modeling (e.g., temporal convolutions or recurrent networks) and physics-inspired Laplacian mesh loss terms to enforce inter-vertex surface smoothness.
