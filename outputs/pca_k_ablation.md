# PCA Dimensionality (k) Ablation Study

| $k$ | Expl. Var (%) | Direct PCA MSE | Direct Err (mm) | Model Test MSE | Mean Err (mm) | Max Err (mm) | Trainable Params |
|---|---|---|---|---|---|---|---|
| **32** | 95.85% | 1.758e-05 | 6.00 mm | **8.366e-05** | 13.50 mm | 103.85 mm | 53,664 |
| **64** | 97.7% | 1.044e-05 | 4.54 mm | **8.120e-05** | 13.31 mm | 103.55 mm | 61,888 |
| **128** | 98.84% | 5.877e-06 | 3.39 mm | **8.112e-05** | 13.24 mm | 108.38 mm | 78,336 |
| **256** | 99.52% | 2.934e-06 | 2.40 mm | **8.258e-05** | 13.42 mm | 107.90 mm | 111,232 |
