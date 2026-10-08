# Model Evaluation Benchmarks

| Model | Trainable Params | Test MSE Loss | Mean Vertex Err (mm) | Max Vertex Err (mm) | Latency (ms) |
|---|---:|---:|---:|---:|---:|
| Linear Baseline | 3,104 | 1.517141e-04 | 17.47 mm | 71.79 mm | 0.052 ms |
| Paper MLP | 53,664 | 1.265825e-04 | 16.25 mm | 77.19 mm | 0.067 ms |
| Improved ResMLP | 299,296 | 8.224905e-05 | 12.14 mm | 76.04 mm | 0.163 ms |
| Improved ResMLP + Aug | 299,296 | 7.634860e-05 | 11.50 mm | 66.78 mm | 0.162 ms |
