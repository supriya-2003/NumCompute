
from __future__ import annotations

from typing import Optional, Union

import numpy as np

__all__ = ["rank", "percentile"]

_RANK_METHODS = ("average", "min", "max", "dense", "ordinal")
_INTERP = ("linear", "lower", "higher", "midpoint", "nearest")


def _rank_1d(a: np.ndarray, method: str) -> np.ndarray:
    n = a.size
    order = np.argsort(a, kind="stable")
    s = a[order]
    inv = np.empty(n, dtype=np.intp)
    inv[order] = np.arange(n)
    if method == "ordinal":
        return inv + 1
    is_start = np.empty(n, dtype=bool)
    is_start[:1] = True
    is_start[1:] = s[1:] != s[:-1]
    group = np.cumsum(is_start) - 1                
    if method == "dense":
        return group[inv] + 1
    starts = np.flatnonzero(is_start)               
    ends = np.append(starts[1:], n)                 
    if method == "min":
        return (starts + 1)[group][inv]
    if method == "max":
        return ends[group][inv]
    return ((starts + 1 + ends) / 2.0)[group][inv]  


def rank(
    data,
    method: str = "average",
    ascending: bool = True,
    axis: Optional[int] = None,
    nan_policy: str = "keep",
) -> np.ndarray:
    
    if method not in _RANK_METHODS:
        raise ValueError(f"method must be one of {_RANK_METHODS}, got {method!r}.")
    if nan_policy not in ("keep", "raise"):
        raise ValueError("nan_policy must be 'keep' or 'raise'.")
    a = np.asarray(data)
    if a.size == 0:
        return np.empty(a.shape, dtype=float)

    def kernel(v: np.ndarray) -> np.ndarray:
        nan_mask = np.isnan(v) if np.issubdtype(v.dtype, np.floating) else np.zeros(v.shape, bool)
        if nan_mask.any() and nan_policy == "raise":
            raise ValueError("Input contains NaN (nan_policy='raise').")
        valid = v[~nan_mask]
        if not ascending:
            valid = -_codes(valid)
        r = _rank_1d(valid, method)
        if nan_mask.any():
            out = np.full(v.shape, np.nan)
            out[~nan_mask] = r
            return out
        return r.astype(float) if method == "average" else r

    if axis is None:
        return kernel(a.ravel()).reshape(a.shape)
    if a.ndim == 1:
        return kernel(a)
    has_nan = np.issubdtype(a.dtype, np.floating) and np.isnan(a).any()
    res = np.apply_along_axis(lambda v: kernel(v).astype(float), axis, a)
    return res if (method == "average" or has_nan) else res.astype(np.intp)


def _codes(v: np.ndarray) -> np.ndarray:
    return np.unique(v, return_inverse=True)[1].reshape(v.shape)


def percentile(
    data,
    q: Union[float, np.ndarray],
    interpolation: str = "linear",
    axis: Optional[int] = None,
    nan_policy: str = "propagate",
    keepdims: bool = False,
) -> Union[float, np.ndarray]:
    
    if interpolation not in _INTERP:
        raise ValueError(f"interpolation must be one of {_INTERP}, got {interpolation!r}.")
    if nan_policy not in ("propagate", "omit", "raise"):
        raise ValueError("nan_policy must be 'propagate', 'omit' or 'raise'.")
    a = np.asarray(data, dtype=float)
    if a.size == 0:
        raise ValueError("percentile of an empty array is undefined.")
    q_arr = np.asarray(q, dtype=float)
    if np.any(np.isnan(q_arr)) or np.any((q_arr < 0) | (q_arr > 100)):
        raise ValueError(f"Percentiles must be in [0, 100], got {q!r}.")

    if axis is None:
        work = a.reshape(-1)
        red_axis = 0
    else:
        if not -a.ndim <= axis < a.ndim:
            raise ValueError(f"axis {axis} is out of bounds for array of dimension {a.ndim}.")
        red_axis = axis % a.ndim
        work = np.moveaxis(a, red_axis, -1)
    nan_mask = np.isnan(work)
    if nan_policy == "raise" and nan_mask.any():
        raise ValueError("Input contains NaN (nan_policy='raise').")

    s = np.sort(work, axis=-1)                      
    N = s.shape[-1]
    batch = s.shape[:-1]
    if nan_policy == "omit":
        n = (~nan_mask).sum(axis=-1)                
    else:
        n = np.full(batch, N)

    qf = q_arr.reshape(q_arr.shape + (1,) * len(batch))  
    pos = qf / 100.0 * (n - 1)                       
    pos = np.where(n > 0, pos, 0.0)
    lo = np.floor(pos).astype(np.intp)
    hi = np.ceil(pos).astype(np.intp)
    frac = pos - lo

    full_shape = pos.shape
    s_b = np.broadcast_to(s, q_arr.shape + s.shape) if q_arr.ndim else s

    def gather(idx):
        idx = np.broadcast_to(idx, full_shape)[..., None]
        return np.take_along_axis(s_b, idx, axis=-1)[..., 0]

    v_lo, v_hi = gather(lo), gather(hi)
    if interpolation == "linear":
        res = v_lo + (v_hi - v_lo) * frac
        res = np.where(frac == 0, v_lo, res)
    elif interpolation == "lower":
        res = v_lo
    elif interpolation == "higher":
        res = v_hi
    elif interpolation == "midpoint":
        res = (v_lo + v_hi) / 2.0
    else:  # nearest
        res = gather(np.around(pos).astype(np.intp))

    res = np.where(np.broadcast_to(n == 0, full_shape), np.nan, res)
    if nan_policy == "propagate":
        res = np.where(np.broadcast_to(nan_mask.any(axis=-1), full_shape), np.nan, res)

    if keepdims:
        if axis is None:
            res = res.reshape(q_arr.shape + (1,) * a.ndim)
        else:
            res = np.expand_dims(res, q_arr.ndim + red_axis)
    return res[()] if np.ndim(res) == 0 else res
