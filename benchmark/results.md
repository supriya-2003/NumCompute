# Benchmark results

Baseline (speed-up = 1.0×) is the first implementation in each task.
Inputs: default sizes; best of 5 runs after 1 warm-up.

| Task | Implementation | Best (ms) | Mean ± std (ms) | Speed-up |
|---|---|---:|---:|---:|
| StandardScaler (n×8) | python loop | 36.389 | 39.014 ± 2.981 | 1.0× |
| StandardScaler (n×8) | numcompute | 1.440 | 1.479 ± 0.051 | 25.3× |
| Pairwise euclidean | python loop | 219.379 | 220.043 ± 0.626 | 1.0× |
| Pairwise euclidean | numcompute | 0.247 | 0.303 ± 0.042 | 888.0× |
| Row softmax | python loop | 21.369 | 23.523 ± 3.124 | 1.0× |
| Row softmax | numcompute | 1.254 | 1.273 ± 0.027 | 17.0× |
| Mean & variance | python loop (Welford) | 11.565 | 11.705 ± 0.165 | 1.0× |
| Mean & variance | numcompute RunningStats | 0.935 | 0.990 ± 0.081 | 12.4× |
| Histogram (30 bins) | python loop | 30.814 | 31.098 ± 0.233 | 1.0× |
| Histogram (30 bins) | numcompute | 2.343 | 2.379 ± 0.027 | 13.1× |
| Rank (average, ties) | python loop | 5.750 | 5.900 ± 0.171 | 1.0× |
| Rank (average, ties) | numcompute | 1.220 | 1.316 ± 0.131 | 4.7× |
| Percentile (q=90) | python sorted() | 23.060 | 23.257 ± 0.124 | 1.0× |
| Percentile (q=90) | numcompute | 6.484 | 6.541 ± 0.050 | 3.6× |
| Top-10 | python sorted() | 23.180 | 23.454 ± 0.189 | 1.0× |
| Top-10 | full np.sort | 6.432 | 6.516 ± 0.080 | 3.6× |
| Top-10 | numcompute topk (argpartition) | 1.317 | 1.339 ± 0.017 | 17.6× |
| Confusion matrix (5 classes) | python loop | 3.718 | 3.774 ± 0.051 | 1.0× |
| Confusion matrix (5 classes) | numcompute (bincount) | 0.324 | 0.342 ± 0.015 | 11.5× |
| One-hot encode (20 cats) | python loop | 8.957 | 13.253 ± 5.068 | 1.0× |
| One-hot encode (20 cats) | numcompute | 1.373 | 1.494 ± 0.090 | 6.5× |
| Finite-diff gradient | per-point f calls | 3.853 | 4.108 ± 0.282 | 1.0× |
| Finite-diff gradient | batched f (vectorized=True) | 2.248 | 2.461 ± 0.166 | 1.7× |

## Environment

- **python**: 3.12.4
- **numpy**: 2.5.4
- **platform**: macOS-14.2-arm64-arm-64bit
- **machine**: arm64
- **processor**: arm
- **cpu_count**: 8
