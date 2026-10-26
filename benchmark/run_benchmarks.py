import argparse
import csv
import os
import sys

import numpy as np

# Set up paths so the benchmark can import the project modules.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   
sys.path.insert(0, HERE)

import loop_impls as L  
from numcompute import (benchmarking, metrics, optim, preprocessing, rank,  
                        sort_search, stats, utils)


def build_cases(quick: bool):
    # Use a fixed seed so the same random data can be generated again.
    rng = np.random.default_rng(0)
    n = 2_000 if quick else 20_000
    X = rng.normal(size=(n, 8))
    Xl = X.tolist()
    A, B = rng.normal(size=(150 if quick else 400, 16)), rng.normal(size=(100 if quick else 300, 16))
    Al, Bl = A.tolist(), B.tolist()
    S = rng.normal(size=(n // 4, 50)) * 50
    Sl = S.tolist()
    x = rng.normal(size=n * 10)
    xl = x.tolist()
    xi = rng.integers(0, n // 10, size=n)
    xil = xi.tolist()
    yt, yp = rng.integers(0, 5, n * 5), rng.integers(0, 5, n * 5)
    ytl, ypl = yt.tolist(), yp.tolist()
    cats = rng.integers(0, 20, n * 2)
    catsl = cats.tolist()
    k = 10

    # Compare the gradient calculation with individual and batched function calls.
    def grad_loop(x0):
        f = lambda v: float(np.sum(np.sin(v) * v))
        return optim.grad(f, x0)

    def grad_vec(x0):
        fb = lambda V: np.sum(np.sin(V) * V, axis=1)
        return optim.grad(fb, x0, vectorized=True)

    x0 = rng.normal(size=200 if quick else 500)

    # Each case compares a Python-loop implementation with a NumPy-based version.
    return [
        ("StandardScaler (n×8)", {
            "python loop": lambda: L.standardize_loop(Xl),
            "numcompute": lambda: preprocessing.StandardScaler().fit_transform(X)}),
        ("Pairwise euclidean", {
            "python loop": lambda: L.pairwise_euclidean_loop(Al, Bl),
            "numcompute": lambda: utils.pairwise_distances(A, B)}),
        ("Row softmax", {
            "python loop": lambda: L.softmax_rows_loop(Sl),
            "numcompute": lambda: utils.softmax(S, axis=1)}),
        ("Mean & variance", {
            "python loop (Welford)": lambda: L.welford_loop(xl),
            "numcompute RunningStats": lambda: (lambda r: (r.mean[0], r.var()[0]))(stats.RunningStats().update(x))}),
        ("Histogram (30 bins)", {
            "python loop": lambda: L.histogram_loop(xl, 30),
            "numcompute": lambda: stats.histogram(x, 30)[0]}),
        ("Rank (average, ties)", {
            "python loop": lambda: L.rank_average_loop(xil),
            "numcompute": lambda: rank.rank(xi, "average")}),
        ("Percentile (q=90)", {
            "python sorted()": lambda: L.percentile_loop(xl, 90),
            "numcompute": lambda: rank.percentile(x, 90)}),
        (f"Top-{k}", {
            "python sorted()": lambda: L.topk_loop(xl, k),
            "full np.sort": lambda: np.sort(x)[::-1][:k],
            "numcompute topk (argpartition)": lambda: sort_search.topk(x, k, return_indices=False)}),
        ("Confusion matrix (5 classes)", {
            "python loop": lambda: L.confusion_matrix_loop(ytl, ypl, 5),
            "numcompute (bincount)": lambda: metrics.confusion_matrix(yt, yp)}),
        ("One-hot encode (20 cats)", {
            "python loop": lambda: L.one_hot_loop(catsl, list(range(20))),
            "numcompute": lambda: preprocessing.OneHotEncoder().fit_transform(cats.reshape(-1, 1))}),
        ("Finite-diff gradient", {
            "per-point f calls": lambda: grad_loop(x0),
            "batched f (vectorized=True)": lambda: grad_vec(x0)}),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller inputs")
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    rows = []
    # Run every benchmark and check that the implementations produce matching results.
    for task, impls in build_cases(args.quick):
        r = benchmarking.compare(impls, task=task, repeat=args.repeat, rtol=1e-6, atol=1e-8)
        rows.extend(r)
        best = max(r, key=lambda z: z["speedup"])
        print(f"{task:32s} {best['name']:32s} {best['speedup']:8.1f}x faster than baseline")

    # Collect system details and format the results for reporting.
    env = benchmarking.environment_info()
    table = benchmarking.format_table(rows)
    md = ["# Benchmark results", "",
          "Baseline (speed-up = 1.0×) is the first implementation in each task.",
          f"Inputs: {'quick' if args.quick else 'default'} sizes; best of {args.repeat} runs after 1 warm-up.", "",
          table, "", "## Environment", ""]
    md += [f"- **{k}**: {v}" for k, v in env.items()]

    # Save the benchmark results in Markdown and CSV formats.
    with open(os.path.join(HERE, "results.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(HERE, "results.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nWrote benchmark/results.md and benchmark/results.csv")


if __name__ == "__main__":
    main()