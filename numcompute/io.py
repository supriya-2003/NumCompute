
from __future__ import annotations

import io as _io
import itertools
import os
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["read_csv", "read_csv_columns", "iter_csv_chunks", "count_missing", "DEFAULT_MISSING"]

DEFAULT_MISSING: Tuple[str, ...] = ("", "NA", "N/A", "NaN", "nan", "null", "NULL", "None", "?")

PathLike = Union[str, os.PathLike]


def _check_path(path: PathLike) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV file not found: {path!r}")


def _read_header(path: PathLike, delimiter: str, skip_rows: int, encoding: str) -> List[str]:
    with open(path, "r", encoding=encoding) as fh:
        for _ in range(skip_rows):
            fh.readline()
        line = fh.readline()
    if not line:
        return []
    return [c.strip() for c in line.rstrip("\r\n").split(delimiter)]


def _resolve_usecols(usecols, header: List[str]) -> Optional[List[int]]:
    if usecols is None:
        return None
    out = []
    for c in usecols:
        if isinstance(c, str):
            if c not in header:
                raise KeyError(f"Column {c!r} not found in header {header}.")
            out.append(header.index(c))
        else:
            out.append(int(c))
    return out


def _is_missing(arr: np.ndarray, missing_values: Sequence[str]) -> np.ndarray:
    
    stripped = np.char.strip(arr.astype(str))
    return np.isin(stripped, np.asarray(list(missing_values), dtype=str))


def read_csv(
    path: PathLike,
    delimiter: str = ",",
    header: bool = True,
    dtype: type = float,
    missing_values: Sequence[str] = DEFAULT_MISSING,
    fill_value: Union[float, str] = np.nan,
    drop_missing: bool = False,
    usecols: Optional[Sequence[Union[int, str]]] = None,
    skip_rows: int = 0,
    encoding: str = "utf-8",
) -> Tuple[np.ndarray, List[str]]:
    """Load a delimited text file into a 2-D NumPy array.

    Parameters
    ----------
    path : str or path-like
        File to read.
    delimiter : str, default=','
        Field separator, e.g. ``','``, ``'\\t'`` or ``';'``.
    header : bool, default=True
        Whether the first (non-skipped) line contains column names.
    dtype : type, default=float
        ``float`` parses numbers (non-numeric fields become missing);
        ``str`` keeps every field as text.
    missing_values : sequence of str
        Tokens treated as missing.
    fill_value : float or str, default=np.nan
        Placeholder written into missing cells (when ``drop_missing=False``).
    drop_missing : bool, default=False
        If True, rows that contain any missing value are removed instead.
    usecols : sequence of int or str, optional
        Subset of columns to read (indices or header names).
    skip_rows : int, default=0
        Lines to skip before the header / data.
    encoding : str, default='utf-8'

    Returns
    -------
    data : np.ndarray, shape (n_rows, n_cols)
        Always 2-D, even for a single row or column. ``float64`` or ``<U``.
    columns : list of str
        Column names (``['col0', 'col1', ...]`` when ``header=False``).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file has no data rows or rows have inconsistent lengths.

    Complexity
    ----------
    O(n_rows * n_cols) time and memory.

    Examples
    --------
    >>> X, cols = read_csv("data.csv")                       # doctest: +SKIP
    >>> X, cols = read_csv("data.tsv", delimiter="\\t", drop_missing=True)  # doctest: +SKIP
    """
    _check_path(path)
    names = _read_header(path, delimiter, skip_rows, encoding) if header else []
    cols = _resolve_usecols(usecols, names)
    skip = skip_rows + (1 if header else 0)

    with open(path, "r", encoding=encoding) as fh:
        n_lines = sum(1 for ln in fh if ln.strip())
    if n_lines <= skip:
        raise ValueError(f"{path!r} contains no data rows.")

    try:
        if dtype is str:
            data = np.genfromtxt(
                path, delimiter=delimiter, skip_header=skip, dtype=str, usecols=cols,
                encoding=encoding, ndmin=2, autostrip=True, comments=None,
            )
            missing = _is_missing(data, missing_values)
        else:
            data = np.genfromtxt(
                path, delimiter=delimiter, skip_header=skip, dtype=float, usecols=cols,
                missing_values=list(missing_values), filling_values=np.nan,
                encoding=encoding, ndmin=2, comments=None,
            )
            missing = np.isnan(data)
    except ValueError as exc:  # inconsistent row lengths etc.
        raise ValueError(f"Could not parse {path!r}: {exc}") from exc

    if data.size == 0:
        raise ValueError(f"{path!r} contains no data rows.")

    if drop_missing:
        data = data[~missing.any(axis=1)]
    elif dtype is str:
        data = np.where(missing, str(fill_value), data)
    elif not (isinstance(fill_value, float) and np.isnan(fill_value)):
        data = np.where(missing, float(fill_value), data)

    if not names:
        names = [f"col{i}" for i in range(data.shape[1] if cols is None else len(cols))]
    elif cols is not None:
        names = [names[i] for i in cols]
    return data, names


def read_csv_columns(
    path: PathLike,
    delimiter: str = ",",
    missing_values: Sequence[str] = DEFAULT_MISSING,
    encoding: str = "utf-8",
) -> Dict[str, np.ndarray]:
    
    raw, names = read_csv(path, delimiter=delimiter, header=True, dtype=str,
                          missing_values=missing_values, fill_value="", encoding=encoding)
    out: Dict[str, np.ndarray] = {}
    for j, name in enumerate(names):  # loop over columns (types differ per column)
        col = raw[:, j]
        miss = _is_missing(col, missing_values)
        try:
            vals = np.where(miss, "nan", col).astype(float)
            out[name] = vals
        except ValueError:
            out[name] = np.where(miss, "", col)
    return out


def iter_csv_chunks(
    path: PathLike,
    chunksize: int = 10_000,
    delimiter: str = ",",
    header: bool = True,
    dtype: type = float,
    missing_values: Sequence[str] = DEFAULT_MISSING,
    fill_value: Union[float, str] = np.nan,
    encoding: str = "utf-8",
) -> Iterator[np.ndarray]:
    """Stream a CSV file in chunks of at most ``chunksize`` rows.

    Only one chunk is held in memory at a time, which makes it possible to
    combine with :class:`numcompute.stats.RunningStats` for out-of-core
    statistics.

    Yields
    ------
    np.ndarray, shape (<= chunksize, n_cols)

    Complexity: O(n_rows * n_cols) time overall, O(chunksize * n_cols) memory.
    """
    if chunksize < 1:
        raise ValueError(f"chunksize must be >= 1, got {chunksize}.")
    _check_path(path)
    with open(path, "r", encoding=encoding) as fh:
        if header:
            fh.readline()
        while True:
            lines = [ln for ln in itertools.islice(fh, chunksize) if ln.strip()]
            if not lines:
                break
            buf = _io.StringIO("".join(lines))
            if dtype is str:
                chunk = np.genfromtxt(buf, delimiter=delimiter, dtype=str, ndmin=2,
                                      autostrip=True, comments=None)
                chunk = np.where(_is_missing(chunk, missing_values), str(fill_value), chunk)
            else:
                chunk = np.genfromtxt(buf, delimiter=delimiter, dtype=float, ndmin=2,
                                      missing_values=list(missing_values),
                                      filling_values=fill_value, comments=None)
            yield chunk


def count_missing(X: np.ndarray) -> np.ndarray:
    """Number of missing entries per column (NaN for floats, '' for strings)."""
    X = np.asarray(X)
    if np.issubdtype(X.dtype, np.floating):
        return np.isnan(X).sum(axis=0)
    return (np.char.strip(X.astype(str)) == "").sum(axis=0)
