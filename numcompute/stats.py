
from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .rank import percentile

__all__ = [
    "mean", "median", "var", "std", "minimum", "maximum", "quantile", "describe",
    "histogram", "RunningStats", "StreamingHistogram",
]


def _prep(x, axis) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        raise ValueError("Statistic of an empty array is undefined.")
    if axis is not None and not -a.ndim <= axis < a.ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {a.ndim}.")
    return a


def _reduce(fn_nan, fn, x, axis, skipna, keepdims, **kw):
    a = _prep(x, axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  
        r = (fn_nan if skipna else fn)(a, axis=axis, keepdims=keepdims, **kw)
    return r[()] if np.ndim(r) == 0 else r


def mean(x, axis: Optional[int] = None, skipna: bool = False, keepdims: bool = False):
    return _reduce(np.nanmean, np.mean, x, axis, skipna, keepdims)


def median(x, axis: Optional[int] = None, skipna: bool = False, keepdims: bool = False):
    return _reduce(np.nanmedian, np.median, x, axis, skipna, keepdims)


def var(x, axis: Optional[int] = None, ddof: int = 0, skipna: bool = False, keepdims: bool = False):
    return _reduce(np.nanvar, np.var, x, axis, skipna, keepdims, ddof=ddof)


def std(x, axis: Optional[int] = None, ddof: int = 0, skipna: bool = False, keepdims: bool = False):
    return _reduce(np.nanstd, np.std, x, axis, skipna, keepdims, ddof=ddof)


def minimum(x, axis: Optional[int] = None, skipna: bool = False, keepdims: bool = False):
    return _reduce(np.nanmin, np.min, x, axis, skipna, keepdims)


def maximum(x, axis: Optional[int] = None, skipna: bool = False, keepdims: bool = False):
    return _reduce(np.nanmax, np.max, x, axis, skipna, keepdims)


def quantile(x, q, axis: Optional[int] = None, method: str = "linear", skipna: bool = False,
             keepdims: bool = False):
    q = np.asarray(q, dtype=float)
    if np.any((q < 0) | (q > 1)):
        raise ValueError(f"Quantiles must be in [0, 1], got {q.tolist()}.")
    a = _prep(x, axis)
    return percentile(a, q * 100.0, interpolation=method, axis=axis,
                      nan_policy="omit" if skipna else "propagate", keepdims=keepdims)


def describe(X, axis: Optional[int] = 0, skipna: bool = True) -> Dict[str, Union[float, np.ndarray]]:
    a = _prep(X, axis)
    nan = np.isnan(a)
    count = (~nan).sum(axis=axis) if skipna else np.ones_like(a).sum(axis=axis).astype(int)
    q = quantile(a, [0.25, 0.5, 0.75], axis=axis, skipna=skipna)
    return {
        "count": count,
        "mean": mean(a, axis, skipna),
        "std": std(a, axis, ddof=1, skipna=skipna),
        "min": minimum(a, axis, skipna),
        "25%": q[0],
        "50%": q[1],
        "75%": q[2],
        "max": maximum(a, axis, skipna),
        "n_missing": nan.sum(axis=axis),
    }



def histogram(
    x, bins: Union[int, Sequence[float]] = 10, range: Optional[Tuple[float, float]] = None,
    density: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    
    a = np.asarray(x, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if np.ndim(bins) == 0:
        nb = int(bins)
        if nb < 1:
            raise ValueError(f"bins must be a positive integer, got {bins}.")
        if range is not None:
            lo, hi = float(range[0]), float(range[1])
            if lo > hi:
                raise ValueError(f"range lower bound must be <= upper bound, got {range}.")
        elif a.size:
            lo, hi = float(a.min()), float(a.max())
        else:
            lo, hi = 0.0, 1.0
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        edges = np.linspace(lo, hi, nb + 1)
        a = a[(a >= lo) & (a <= hi)]
        idx = ((a - lo) * (nb / (hi - lo))).astype(np.intp)
        idx = np.minimum(idx, nb - 1) 
        idx -= (a < edges[idx]).astype(np.intp)
        idx += ((idx < nb - 1) & (a >= edges[np.minimum(idx + 1, nb)])).astype(np.intp)
    else:
        edges = np.asarray(bins, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("bins edges must be a 1-D array with at least 2 entries.")
        if np.any(np.diff(edges) < 0):
            raise ValueError("bins edges must increase monotonically.")
        nb = edges.size - 1
        a = a[(a >= edges[0]) & (a <= edges[-1])]
        idx = np.searchsorted(edges, a, side="right") - 1
        idx = np.minimum(idx, nb - 1)
    counts = np.bincount(idx, minlength=nb)[:nb]
    if density:
        total = counts.sum()
        widths = np.diff(edges)
        with np.errstate(invalid="ignore", divide="ignore"):
            return counts / (total * widths) if total else np.zeros(nb), edges
    return counts, edges



class RunningStats:
    
    def __init__(self, n_features: Optional[int] = None):
        self.n_features = n_features
        self._initialised = False
        if n_features is not None:
            self._init(n_features)

    def _init(self, d: int) -> None:
        self.n_features = d
        self.count = np.zeros(d)
        self.mean = np.zeros(d)
        self._m2 = np.zeros(d)
        self.min = np.full(d, np.inf)
        self.max = np.full(d, -np.inf)
        self._initialised = True

    def update(self, batch) -> "RunningStats":
        
        b = np.asarray(batch, dtype=float)
        if b.ndim == 0:
            b = b.reshape(1, 1)
        elif b.ndim == 1:
            b = b.reshape(-1, 1) if (self.n_features in (None, 1)) else b.reshape(1, -1)
        elif b.ndim != 2:
            raise ValueError(f"batch must be at most 2-D, got shape {b.shape}.")
        if not self._initialised:
            self._init(b.shape[1])
        if b.shape[1] != self.n_features:
            raise ValueError(f"batch has {b.shape[1]} features, expected {self.n_features}.")
        if b.shape[0] == 0:
            return self
        valid = ~np.isnan(b)
        n_b = valid.sum(0).astype(float)
        bz = np.where(valid, b, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_b = np.where(n_b > 0, bz.sum(0) / np.maximum(n_b, 1), 0.0)
            m2_b = (np.where(valid, b - mean_b, 0.0) ** 2).sum(0)
        n_a = self.count
        n = n_a + n_b
        delta = mean_b - self.mean
        safe_n = np.where(n > 0, n, 1.0)
        self.mean = self.mean + delta * n_b / safe_n
        self._m2 = self._m2 + m2_b + delta ** 2 * n_a * n_b / safe_n
        self.count = n
        self.min = np.minimum(self.min, np.where(valid, b, np.inf).min(0))
        self.max = np.maximum(self.max, np.where(valid, b, -np.inf).max(0))
        return self

    def merge(self, other: "RunningStats") -> "RunningStats":

        if not other._initialised:
            return self
        if not self._initialised:
            self._init(other.n_features)
        n = self.count + other.count
        safe_n = np.where(n > 0, n, 1.0)
        delta = other.mean - self.mean
        self.mean = self.mean + delta * other.count / safe_n
        self._m2 = self._m2 + other._m2 + delta ** 2 * self.count * other.count / safe_n
        self.count = n
        self.min = np.minimum(self.min, other.min)
        self.max = np.maximum(self.max, other.max)
        return self

    def var(self, ddof: int = 0) -> np.ndarray:

        self._require()
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.count > ddof, self._m2 / (self.count - ddof), np.nan)

    def std(self, ddof: int = 0) -> np.ndarray:
        return np.sqrt(self.var(ddof))

    def _require(self):
        if not self._initialised:
            raise RuntimeError("RunningStats has not received any data yet.")

    def __repr__(self) -> str:
        if not self._initialised:
            return "RunningStats(empty)"
        return f"RunningStats(count={self.count.tolist()}, mean={np.round(self.mean, 4).tolist()})"


class StreamingHistogram:
    

    def __init__(self, edges: Sequence[float]):
        self.edges = np.asarray(edges, dtype=float)
        if self.edges.ndim != 1 or self.edges.size < 2 or np.any(np.diff(self.edges) <= 0):
            raise ValueError("edges must be a strictly increasing 1-D array with >= 2 entries.")
        self.counts = np.zeros(self.edges.size - 1, dtype=np.int64)
        self.n_out_of_range = 0

    def update(self, batch) -> "StreamingHistogram":
        b = np.asarray(batch, dtype=float).ravel()
        b = b[np.isfinite(b)]
        inside = (b >= self.edges[0]) & (b <= self.edges[-1])
        self.n_out_of_range += int((~inside).sum())
        self.counts += histogram(b[inside], bins=self.edges)[0]
        return self
