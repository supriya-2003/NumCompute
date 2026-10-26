# Benchmark results

Baseline (speed-up = 1.0×) is the first implementation in each task.
Inputs: default sizes; best of 5 runs after 1 warm-up.

| Task | Implementation | Best (ms) | Mean ± std (ms) | Speed-up |
|---|---|---:|---:|---:|
| StandardScaler (n×8) | python loop | 23.149 | 25.128 ± 1.391 | 1.0× |
| StandardScaler (n×8) | numcompute | 1.686 | 3.110 ± 0.791 | 13.7× |
| Pairwise euclidean | python loop | 109.084 | 109.883 ± 0.955 | 1.0× |
| Pairwise euclidean | numcompute | 0.151 | 0.169 ± 0.021 | 720.6× |
| Row softmax | python loop | 17.608 | 18.228 ± 0.469 | 1.0× |
| Row softmax | numcompute | 0.675 | 0.685 ± 0.015 | 26.1× |
| Mean & variance | python loop (Welford) | 11.557 | 11.698 ± 0.080 | 1.0× |
| Mean & variance | numcompute RunningStats | 0.393 | 0.408 ± 0.011 | 29.4× |
| Histogram (30 bins) | python loop | 24.270 | 24.704 ± 0.297 | 1.0× |
| Histogram (30 bins) | numcompute | 1.036 | 1.067 ± 0.020 | 23.4× |
| Rank (average, ties) | python loop | 4.150 | 4.248 ± 0.058 | 1.0× |
| Rank (average, ties) | numcompute | 0.803 | 0.844 ± 0.028 | 5.2× |
| Percentile (q=90) | python sorted() | 16.483 | 16.644 ± 0.122 | 1.0× |
| Percentile (q=90) | numcompute | 4.382 | 4.446 ± 0.037 | 3.8× |
| Top-10 | python sorted() | 16.653 | 16.725 ± 0.057 | 1.0× |
| Top-10 | full np.sort | 4.350 | 4.424 ± 0.046 | 3.8× |
| Top-10 | numcompute topk (argpartition) | 0.820 | 0.869 ± 0.040 | 20.3× |
| Confusion matrix (5 classes) | python loop | 3.523 | 3.643 ± 0.085 | 1.0× |
| Confusion matrix (5 classes) | numcompute (bincount) | 0.215 | 0.217 ± 0.002 | 16.4× |
| One-hot encode (20 cats) | python loop | 7.471 | 11.821 ± 5.027 | 1.0× |
| One-hot encode (20 cats) | numcompute | 0.928 | 1.040 ± 0.116 | 8.1× |
| Finite-diff gradient | per-point f calls | 2.604 | 2.679 ± 0.064 | 1.0× |
| Finite-diff gradient | batched f (vectorized=True) | 1.170 | 1.194 ± 0.025 | 2.2× |

## Environment

- **python**: 3.9.6
- **numpy**: 2.0.2
- **platform**: macOS-26.6-arm64-arm-64bit
- **machine**: arm64
- **processor**: arm
- **cpu_count**: 10
