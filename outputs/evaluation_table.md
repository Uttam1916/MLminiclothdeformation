# Model Evaluation Benchmarks

| Model | Trainable Params | Test MSE Loss | Mean Vertex Err (mm) | Max Vertex Err (mm) | Latency (ms) |
|---|---:|---:|---:|---:|---:|
| Linear Baseline | 3,104 | 1.543115e-04 | 17.57 mm | 72.56 mm | 0.078 ms |
| Paper MLP | 53,664 | 1.241655e-04 | 15.95 mm | 83.90 mm | 0.096 ms |
| Improved ResMLP | 299,296 | 8.506049e-05 | 11.99 mm | 72.15 mm | 0.299 ms |
| Improved ResMLP + Aug | 299,296 | 8.018791e-05 | 11.64 mm | 69.12 mm | 0.236 ms |
