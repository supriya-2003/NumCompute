
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "accuracy", "confusion_matrix", "precision", "recall", "f1", "precision_recall_f1",
    "mse", "rmse", "mae", "r2_score", "roc_curve", "auc", "roc_auc_score",
]


def _check_targets(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray]:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    if yt.ndim != 1 or yp.ndim != 1:
        raise ValueError(f"y_true and y_pred must be 1-D, got shapes {yt.shape} and {yp.shape}.")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"y_true and y_pred have different lengths: {yt.shape[0]} vs {yp.shape[0]}.")
    if yt.size == 0:
        raise ValueError("y_true and y_pred are empty.")
    return yt, yp


def accuracy(y_true, y_pred, normalize: bool = True) -> float:
    
    yt, yp = _check_targets(y_true, y_pred)
    correct = np.count_nonzero(yt == yp)
    return correct / yt.size if normalize else int(correct)


def confusion_matrix(
    y_true, y_pred, labels: Optional[Sequence] = None, normalize: Optional[str] = None
) -> np.ndarray:
    
    yt, yp = _check_targets(y_true, y_pred)
    cm = None
    if labels is None and yt.dtype.kind in "iub" and yp.dtype.kind in "iub":
        
        lo = int(min(yt.min(), yp.min()))
        R = int(max(yt.max(), yp.max())) - lo + 1
        if R <= 4096:
            full = np.bincount((yt.astype(np.int64) - lo) * R + (yp.astype(np.int64) - lo),
                               minlength=R * R).reshape(R, R)
            present = (full.sum(0) + full.sum(1)) > 0
            cm = full[np.ix_(present, present)]
    if cm is None:
        labels = np.unique(np.concatenate([yt, yp])) if labels is None else np.asarray(labels)
        if labels.size == 0:
            raise ValueError("labels is empty.")
        order = np.argsort(labels, kind="stable")
        sl = labels[order]
        L = labels.size

        def encode(v):
            pos = np.clip(np.searchsorted(sl, v), 0, L - 1)
            ok = sl[pos] == v
            return order[pos], ok

        ti, tok = encode(yt)
        pi, pok = encode(yp)
        keep = tok & pok
        cm = np.bincount(ti[keep] * L + pi[keep], minlength=L * L).reshape(L, L)
    if normalize is None:
        return cm
    with np.errstate(invalid="ignore", divide="ignore"):
        if normalize == "true":
            out = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            out = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == "all":
            out = cm / cm.sum()
        else:
            raise ValueError("normalize must be None, 'true', 'pred' or 'all'.")
    return np.nan_to_num(out)


def _safe_div(num, den, zero_division):
    num, den = np.asarray(num, float), np.asarray(den, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), zero_division)


def precision_recall_f1(
    y_true, y_pred, average: Optional[str] = "binary", pos_label=1,
    labels: Optional[Sequence] = None, zero_division: float = 0.0,
):
    
    yt, yp = _check_targets(y_true, y_pred)
    if average not in ("binary", "macro", "micro", "weighted", None):
        raise ValueError("average must be 'binary', 'macro', 'micro', 'weighted' or None.")
    lab = np.unique(np.concatenate([yt, yp])) if labels is None else np.asarray(labels)
    if average == "binary":
        if lab.size > 2:
            raise ValueError(
                f"Target is multiclass ({lab.size} labels) but average='binary'. "
                "Use average='macro', 'micro' or 'weighted'."
            )
        if pos_label not in lab:
            lab = np.unique(np.append(lab, pos_label))
    cm = confusion_matrix(yt, yp, labels=lab)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)

    if average == "micro":
        tp, fp, fn = tp.sum(), fp.sum(), fn.sum()
    p = _safe_div(tp, tp + fp, zero_division)
    r = _safe_div(tp, tp + fn, zero_division)
    f = _safe_div(2 * p * r, p + r, zero_division)
    
    f = np.where((2 * tp + fp + fn) > 0, _safe_div(2 * tp, 2 * tp + fp + fn, zero_division), f)

    if average == "binary":
        i = int(np.flatnonzero(lab == pos_label)[0])
        return float(p[i]), float(r[i]), float(f[i])
    if average == "micro":
        return float(p), float(r), float(f)
    if average == "macro":
        return float(p.mean()), float(r.mean()), float(f.mean())
    if average == "weighted":
        w = support / support.sum() if support.sum() else np.zeros_like(support, float)
        return float(p @ w), float(r @ w), float(f @ w)
    return p, r, f


def precision(y_true, y_pred, average="binary", pos_label=1, labels=None, zero_division=0.0):
    
    return precision_recall_f1(y_true, y_pred, average, pos_label, labels, zero_division)[0]


def recall(y_true, y_pred, average="binary", pos_label=1, labels=None, zero_division=0.0):
    
    return precision_recall_f1(y_true, y_pred, average, pos_label, labels, zero_division)[1]


def f1(y_true, y_pred, average="binary", pos_label=1, labels=None, zero_division=0.0):
    
    return precision_recall_f1(y_true, y_pred, average, pos_label, labels, zero_division)[2]



def _check_reg(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"y_true and y_pred shapes differ: {yt.shape} vs {yp.shape}.")
    if yt.size == 0:
        raise ValueError("y_true and y_pred are empty.")
    return yt, yp


def mse(y_true, y_pred, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    
    yt, yp = _check_reg(y_true, y_pred)
    r = np.mean((yt - yp) ** 2, axis=axis)
    return float(r) if np.ndim(r) == 0 else r


def rmse(y_true, y_pred, axis: Optional[int] = None):
    
    return np.sqrt(mse(y_true, y_pred, axis))


def mae(y_true, y_pred, axis: Optional[int] = None):
    
    yt, yp = _check_reg(y_true, y_pred)
    r = np.mean(np.abs(yt - yp), axis=axis)
    return float(r) if np.ndim(r) == 0 else r


def r2_score(y_true, y_pred) -> float:
    
    yt, yp = _check_reg(y_true, y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)



def roc_curve(y_true, y_score, pos_label=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    yt, ys = _check_targets(y_true, y_score)
    ys = ys.astype(float)
    if not np.all(np.isfinite(ys)):
        raise ValueError("y_score contains NaN or infinity.")
    pos = (yt == pos_label)
    n_pos = int(pos.sum())
    n_neg = pos.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_curve requires both positive and negative samples.")
    order = np.argsort(-ys, kind="stable")
    s, p = ys[order], pos[order].astype(float)
    distinct = np.flatnonzero(np.diff(s)) if s.size > 1 else np.empty(0, dtype=np.intp)
    ends = np.append(distinct, s.size - 1)      # last index of each tie group
    tps = np.cumsum(p)[ends]
    fps = (ends + 1) - tps
    tpr = np.concatenate([[0.0], tps / n_pos])
    fpr = np.concatenate([[0.0], fps / n_neg])
    thr = np.concatenate([[np.inf], s[ends]])
    return fpr, tpr, thr


def auc(x, y) -> float:
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of equal length.")
    if x.size < 2:
        raise ValueError("At least 2 points are needed to compute AUC.")
    dx = np.diff(x)
    if np.any(dx < 0) and np.any(dx > 0):
        raise ValueError("x must be monotonic.")
    area = float(np.sum(dx * (y[1:] + y[:-1]) / 2.0))
    return abs(area)


def roc_auc_score(y_true, y_score, pos_label=1) -> float:
    
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label)
    return auc(fpr, tpr)
