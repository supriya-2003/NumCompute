
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["stable_sort", "multikey_argsort", "multikey_sort", "topk", "quickselect", "binary_search"]


def _codes(col: np.ndarray) -> np.ndarray:
    return np.unique(col, return_inverse=True)[1].reshape(col.shape)


def stable_sort(
    a, axis: int = -1, descending: bool = False, return_indices: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    
    a = np.asarray(a)
    if axis is None:
        a, axis = a.ravel(), -1
    if a.ndim == 0:
        raise ValueError("Cannot sort a 0-d array.")
    if not descending:
        idx = np.argsort(a, axis=axis, kind="stable")
    else:
        n = a.shape[axis]
        idx = np.argsort(np.flip(a, axis=axis), axis=axis, kind="stable")
        idx = (n - 1) - np.flip(idx, axis=axis)
    out = np.take_along_axis(a, idx, axis=axis)
    return (out, idx) if return_indices else out


def multikey_argsort(
    X, keys: Sequence[int], descending: Union[bool, Sequence[bool]] = False
) -> np.ndarray:
    
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}.")
    keys = list(keys)
    if not keys:
        raise ValueError("At least one key column is required.")
    if isinstance(descending, (bool, np.bool_)):
        descending = [bool(descending)] * len(keys)
    if len(descending) != len(keys):
        raise ValueError("descending must have one entry per key.")
    for k in keys:
        if not -X.shape[1] <= k < X.shape[1]:
            raise IndexError(f"Key column {k} out of range for {X.shape[1]} columns.")
    cols = []
    for k, desc in zip(keys, descending):
        c = _codes(X[:, k])
        cols.append(-c if desc else c)
    return np.lexsort(cols[::-1])


def multikey_sort(X, keys: Sequence[int], descending: Union[bool, Sequence[bool]] = False) -> np.ndarray:
    X = np.asarray(X)
    return X[multikey_argsort(X, keys, descending)]


def topk(
    values,
    k: int,
    largest: bool = True,
    return_indices: bool = True,
    axis: int = -1,
    sorted: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    
    a = np.asarray(values)
    if a.ndim == 0:
        raise ValueError("topk requires at least a 1-D array.")
    n = a.shape[axis]
    if not isinstance(k, (int, np.integer)) or k < 0 or k > n:
        raise ValueError(f"k must be an integer in [0, {n}], got {k!r}.")
    if k == 0:
        shape = list(a.shape)
        shape[axis] = 0
        empty_v, empty_i = np.empty(shape, dtype=a.dtype), np.empty(shape, dtype=np.intp)
        return (empty_v, empty_i) if return_indices else empty_v

    has_nan = np.issubdtype(a.dtype, np.floating) and np.isnan(a).any()
    if has_nan:
        kth = n - k if largest else k - 1
        part = np.argpartition(a, kth, axis=axis)
        sel = np.arange(n - k, n) if largest else np.arange(k)
        idx = np.sort(np.take(part, sel, axis=axis), axis=axis)
    else:
        moved = np.moveaxis(a, axis, -1)
        flat = moved.reshape(-1, n)
        kth = n - k if largest else k - 1
        part = np.argpartition(flat, kth, axis=1)
        sel = part[:, n - k:] if largest else part[:, :k]
        t = np.take_along_axis(flat, part[:, kth:kth + 1], axis=1)      
        eq = flat == t
        if np.array_equal(eq.sum(axis=1), (np.take_along_axis(flat, sel, axis=1) == t).sum(axis=1)):
            idx_flat = np.sort(sel, axis=1)
        else:
            strict = flat > t if largest else flat < t
            need = k - strict.sum(axis=1, keepdims=True)
            mask = strict | (eq & (np.cumsum(eq, axis=1) <= need))
            idx_flat = np.nonzero(mask)[1].reshape(-1, k)               
        idx = np.moveaxis(idx_flat.reshape(moved.shape[:-1] + (k,)), -1, axis)
    if sorted:
        vals = np.take_along_axis(a, idx, axis=axis)
        order = stable_sort(vals, axis=axis, descending=largest, return_indices=True)[1]
        idx = np.take_along_axis(idx, order, axis=axis)
    vals = np.take_along_axis(a, idx, axis=axis)
    return (vals, idx) if return_indices else vals


def quickselect(arr, k: int, seed: Optional[int] = 0):
    
    a = np.asarray(arr).ravel()
    n = a.size
    if n == 0:
        raise ValueError("quickselect on an empty array.")
    if np.issubdtype(a.dtype, np.floating) and np.isnan(a).any():
        raise ValueError("quickselect does not support NaN values.")
    if not -n <= k < n:
        raise ValueError(f"k must be in [{-n}, {n - 1}], got {k}.")
    k = k % n
    rng = np.random.default_rng(seed)
    while True:
        if a.size == 1:
            return a[0]
        pivot = a[rng.integers(a.size)]
        lt = a[a < pivot]
        n_lt = lt.size
        n_eq = int(np.count_nonzero(a == pivot))
        if k < n_lt:
            a = lt
        elif k < n_lt + n_eq:
            return pivot
        else:
            a = a[a > pivot]
            k -= n_lt + n_eq


def binary_search(
    sorted_array, x, side: str = "left", check_sorted: bool = False, method: str = "numpy"
) -> Tuple[Union[int, np.ndarray], Union[bool, np.ndarray]]:
    
    a = np.asarray(sorted_array)
    if a.ndim != 1:
        raise ValueError(f"sorted_array must be 1-D, got shape {a.shape}.")
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'.")
    if check_sorted and a.size > 1 and np.any(a[1:] < a[:-1]):
        raise ValueError("sorted_array is not sorted in ascending order.")

    if method == "iterative":
        if np.ndim(x) != 0:
            raise ValueError("method='iterative' only supports scalar x.")
        lo, hi = 0, a.size
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] < x or (side == "right" and a[mid] == x):
                lo = mid + 1
            else:
                hi = mid
        idx = lo
        if side == "left":
            found = idx < a.size and a[idx] == x
        else:
            found = idx > 0 and a[idx - 1] == x
        return int(idx), bool(found)
    if method != "numpy":
        raise ValueError("method must be 'numpy' or 'iterative'.")

    idx = np.searchsorted(a, x, side=side)
    if a.size == 0:
        found = np.zeros(np.shape(idx), dtype=bool)
    elif side == "left":
        found = (idx < a.size) & (a[np.minimum(idx, a.size - 1)] == x)
    else:
        found = (idx > 0) & (a[np.maximum(idx - 1, 0)] == x)
    if np.ndim(idx) == 0:
        return int(idx), bool(found)
    return idx, found
