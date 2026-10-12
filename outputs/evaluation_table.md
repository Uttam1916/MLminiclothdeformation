# Model Evaluation Benchmarks (k = 128)

| Model | Trainable Params | Test MSE Loss | Mean Vertex Err (mm) | Max Vertex Err (mm) | Latency (CPU) |
|---|---:|---:|---:|---:|---:|
| Linear Baseline | 12,416 | 1.188448e-04 | 15.81 mm | 101.13 mm | 0.261 ms |
| Paper MLP | 78,336 | 7.822229e-05 | 13.02 mm | 98.55 mm | 0.339 ms |
| Improved ResMLP | 323,968 | 4.499596e-05 | 9.66 mm | 81.05 mm | 0.482 ms |
| Improved ResMLP + Aug | 323,968 | 4.542750e-05 | 9.83 mm | 75.43 mm | 0.549 ms |
