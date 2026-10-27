# NumCompute

A modular, end-to-end machine-learning toolkit written from scratch in **pure Python + NumPy**.
NumCompute re-implements the core building blocks of libraries such as scikit-learn:
data loading, preprocessing, sorting/searching, ranking, statistics, metrics,
numerical differentiation and a pipeline abstraction. No scikit-learn, pandas,
SciPy or PyTorch is used.

Design goals:

* **Vectorised**: no Python loops in core computations, except where they can't be avoided (per-feature category sets, the sequential iterations of an optimiser).
* **Numerically stable**: max-shifted `softmax`/`logsumexp`, overflow-safe `sigmoid`, Welford/Chan streaming variance, relative finite-difference steps.
* **Consistent API**: explicit `axis` semantics, documented shapes, informative errors for type and shape mismatches.
* **Documented and tested**: every public function has a docstring listing parameters, returns, exceptions and time/space complexity. The suite has 99 unit tests.

---

## Repository layout

```
NumCompute/
├── numcompute/
│   ├── __init__.py
│   ├── io.py            # CSV reader (streaming/chunking), dtype + missing-value handling
│   ├── preprocessing.py # StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder
│   ├── sort_search.py   # stable & multi-key sort, top-k (argpartition), quickselect, binary search
│   ├── rank.py          # ranking with ties; percentiles (5 interpolation modes)
│   ├── stats.py         # descriptive & axis-wise stats, histogram, quantiles, streaming Welford
│   ├── metrics.py       # accuracy/precision/recall/F1, confusion matrix, MSE, ROC/AUC
│   ├── optim.py         # finite-difference gradient & Jacobian, line search
│   ├── pipeline.py      # Transformer/Estimator API, Pipeline, FeatureUnion, ColumnTransformer
│   ├── utils.py         # distances, activations, logsumexp, top-k mask, batching
│   └── benchmarking.py  # micro-benchmark harness
├── tests/               # 99 pytest unit tests (edge cases: empty, ties, NaN, extreme k, strides)
├── benchmark/           # loop vs vectorised benchmarks + results
├── demo/
│   ├── quickstart.ipynb # end-to-end demo (executed, outputs saved)
│   ├── build_notebook.py
│   └── data/customers.csv
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Installation

Requires Python ≥ 3.9 and NumPy ≥ 1.23.

```bash
git clone <your-repo-url> NumCompute
cd NumCompute
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"          # installs numcompute + pytest, jupyter, matplotlib
```

To install the package alone, without the dev tools, run `pip install -e .`, which pulls in only NumPy.

## Running tests, demo and benchmarks

```bash
pytest                                   # 99 tests
jupyter notebook demo/quickstart.ipynb   # end-to-end demo
python benchmark/run_benchmarks.py       # writes benchmark/results.md and results.csv
python benchmark/run_benchmarks.py --quick
```

## Quick example

```python
import numpy as np
from numcompute import io, preprocessing, pipeline, metrics, sort_search, rank, stats, optim

# 1. Load data (missing cells -> NaN)
X, cols = io.read_csv("demo/data/customers.csv", usecols=["age", "income", "hours_per_week"])

# 2. Preprocess with a pipeline
pipe = pipeline.Pipeline([
    ("impute", preprocessing.SimpleImputer(strategy="median")),
    ("scale",  preprocessing.StandardScaler()),
])
X_tr = pipe.fit_transform(X)

# The spec's example also works:
pipe2 = pipeline.Pipeline([("scale", preprocessing.StandardScaler()),
                           ("encode", preprocessing.OneHotEncoder())])

# 3. Top-k, ranks, percentiles, stats
vals, idx = sort_search.topk(X_tr[:, 1], k=5)               # O(n) via argpartition
r = rank.rank([10, 20, 20, 30], method="average")           # [1. 2.5 2.5 4.]
p90 = rank.percentile(X, 90, axis=0, nan_policy="omit")
counts, edges = stats.histogram(X[:, 1], bins=10)

# 4. Gradients
g = optim.grad(lambda v: (v ** 2).sum(), np.array([1.0, 2.0]))   # [2. 4.]

# 5. Metrics
metrics.f1([1, 0, 1, 1], [1, 0, 0, 1])                           # 0.8
```

## API overview

All transformers implement `fit(X) -> self`, `transform(X) -> X_out` and `fit_transform(X)`. Models implement `fit(X, y)` and `predict(X)`. Inputs are `(n_samples, n_features)` arrays. A 1-D input to a transformer raises `ValueError` with a hint to `reshape(-1, 1)`.

| Module | Main API | Notes |
|---|---|---|
| `io` | `read_csv(path, delimiter=',', dtype=float, fill_value=nan, drop_missing=False, usecols=None)` → `(array, columns)`; `read_csv_columns`; `iter_csv_chunks(path, chunksize)`; `count_missing` | Built on `np.genfromtxt`. Missing tokens: `'' NA N/A NaN null None ?`. Chunked streaming keeps memory constant. |
| `preprocessing` | `StandardScaler(with_mean, with_std, ddof)`, `MinMaxScaler(feature_range, clip)`, `SimpleImputer(strategy, fill_value)`, `OneHotEncoder(categories, handle_unknown, drop_first)` | Constant features don't divide by zero. NaNs are ignored in `fit`. `inverse_transform` and `get_feature_names` are available. |
| `sort_search` | `stable_sort(a, axis, descending)`, `multikey_sort(X, keys, descending)`, `topk(values, k, largest=True, return_indices=True, axis=-1)`, `quickselect(arr, k)`, `binary_search(sorted, x)` → `(index, found)` | Descending sort stays stable. `topk` breaks ties deterministically (lowest index first). `quickselect` uses a 3-way partition, so duplicates are safe. |
| `rank` | `rank(data, method='average'\|'min'\|'max'\|'dense'\|'ordinal', ascending, axis, nan_policy)`, `percentile(data, q, interpolation='linear'\|'lower'\|'higher'\|'midpoint'\|'nearest', axis, nan_policy)` | Implemented from scratch on `argsort`/`sort` and checked against `np.percentile`. |
| `stats` | `mean/median/var/std/minimum/maximum(x, axis, skipna)`, `quantile`, `describe`, `histogram(x, bins, range, density)`, `RunningStats`, `StreamingHistogram` | `axis=0` gives per column, `axis=1` gives per row, `None` gives a scalar. Streaming uses Welford/Chan merging. |
| `metrics` | `accuracy`, `precision`, `recall`, `f1` (`average='binary'\|'macro'\|'micro'\|'weighted'\|None`), `confusion_matrix(labels, normalize)`, `mse/rmse/mae/r2_score`, `roc_curve`, `auc`, `roc_auc_score` | Confusion matrix built with a single `bincount`. `zero_division` controls 0/0. |
| `optim` | `grad(f, x, h=1e-5, method='central'\|'forward'\|'backward', vectorized=False)`, `jacobian(F, x, ...)`, `check_grad`, `line_search` (Armijo), `gradient_descent` | Relative step `h·max(1,\|x\|)`. `vectorized=True` evaluates all perturbations in a single call. |
| `pipeline` | `Pipeline`, `make_pipeline`, `FeatureUnion`, `ColumnTransformer`, `FunctionTransformer`, `BaseEstimator` (`get_params/set_params`) | Includes reference estimators: `MeanRegressor`, `MajorityClassifier`, `LeastSquaresRegressor`, `LogisticRegressionGD`. |
| `utils` | `pairwise_distances(A, B, metric)`, `logsumexp`, `softmax`, `log_softmax`, `sigmoid`, `relu`, `topk_mask`, `batch_iterator`, `check_array` | Max-shift trick for `softmax` and `logsumexp`. |
| `benchmarking` | `timeit`, `compare`, `format_table`, `environment_info` | Takes the best of N repeats after a warm-up, and verifies all implementations return identical results. |

## Performance (vectorised vs Python loops)

Generated by `python benchmark/run_benchmarks.py`. Each figure is the best of 5 runs after a warm-up, and every implementation's output was verified equal. Speed-up is measured against the first row of each task.

| Task | Implementation | Best (ms) | Speed-up |
|---|---|---:|---:|
| StandardScaler (20k × 8) | python loop | 39.21 | 1.0× |
| | numcompute | 3.20 | **12.3×** |
| Pairwise euclidean (400 × 300 × 16) | python loop | 178.55 | 1.0× |
| | numcompute | 0.57 | **311×** |
| Row softmax (5k × 50) | python loop | 24.18 | 1.0× |
| | numcompute | 1.43 | **17.0×** |
| Mean & variance (200k) | python Welford loop | 11.10 | 1.0× |
| | numcompute RunningStats | 1.45 | **7.7×** |
| Histogram (200k, 30 bins) | python loop | 17.74 | 1.0× |
| | numcompute | 2.79 | **6.4×** |
| Rank, average ties (20k) | python loop | 7.18 | 1.0× |
| | numcompute | 1.51 | **4.7×** |
| Percentile q=90 (200k) | python `sorted()` | 33.66 | 1.0× |
| | numcompute | 1.62 | **20.8×** |
| Top-10 of 200k | python `sorted()` | 33.86 | 1.0× |
| | full `np.sort` | 1.15 | 29.6× |
| | numcompute `topk` (argpartition) | 0.51 | **65.9×** |
| Confusion matrix (100k, 5 classes) | python loop | 3.57 | 1.0× |
| | numcompute (bincount) | 0.42 | **8.6×** |
| One-hot encode (40k, 20 cats) | python loop | 9.02 | 1.0× |
| | numcompute | 2.06 | **4.4×** |
| Finite-diff gradient (n=500) | per-point `f` calls | 4.91 | 1.0× |
| | batched `f` (`vectorized=True`) | 3.21 | **1.5×** |

Environment: Python 3.13, NumPy 2.5.3, Linux x86_64, 2 vCPUs. The full table with mean ± std is in `benchmark/results.md`. Absolute timings vary by machine, so re-run the script on your own hardware before quoting numbers.

## Edge cases covered by tests

* Empty arrays raise informative `ValueError`s (`stats`, `percentile`, `quickselect`, `metrics`, scalers).
* All-equal values: constant features in the scalers, histograms of a constant array, `quickselect` and `rank` on identical values.
* Duplicates and ties: every `rank` method, stable descending sort, `topk` boundary ties, ROC with tied scores.
* Extreme `k`: `topk` with `k=0` and `k=n`, `quickselect` with `k=0`, `k=n-1` and `k=-1`.
* NaN handling: `skipna` and `nan_policy` options, NaN as a one-hot category, streaming stats with NaN, `logsumexp` with ±inf and NaN.
* Non-contiguous (strided) inputs to the scalers and sorting.
* Numerical stability: `softmax` and `logsumexp` at ±1000, `RunningStats` with a 1e6 offset, cancellation in pairwise distances.

## Known limitations

* `read_csv` does not parse quoted fields that contain the delimiter, because `genfromtxt` doesn't support it.
* `rank(..., axis=k)` on N-D input runs the vectorised 1-D kernel once per slice through `np.apply_along_axis`.
* The reference estimators in `pipeline.py` demonstrate the fit/predict protocol and are not tuned ML models.

## Authors and contributions

| Member | Contribution focus | Main work |
|---|---|---|
| Sai Datta Kamal Ashwin | Numerical implementation and vectorisation | Vectorised NumPy implementations using array operations, broadcasting and matrix calculations to remove unnecessary loops, e.g. pairwise distances, preprocessing and top-k selection. |
| Supriya Gopal | Module integration, API consistency and review | Bringing the modules together into one package; keeping inputs, outputs and error handling consistent across modules; reviewing the package as a whole. |
| Harsh Shail Joshi | Testing, edge cases and correctness checks | Unit tests for ordinary and edge cases (empty inputs, repeated values, mismatched shapes, constant-valued features, invalid parameters) and checks of error behaviour. |
| Siri Tummala | Benchmarking, documentation and performance evidence | Loop-versus-vectorised benchmarks, performance results and documentation. |
