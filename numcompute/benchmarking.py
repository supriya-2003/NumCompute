
from __future__ import annotations

import os
import platform
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["timeit", "compare", "format_table", "environment_info"]


def timeit(func: Callable, *args, repeat: int = 5, number: int = 1, warmup: int = 1,
           **kwargs) -> Dict[str, Any]:
    
    if repeat < 1 or number < 1:
        raise ValueError("repeat and number must be >= 1.")
    result = None
    for _ in range(warmup):
        result = func(*args, **kwargs)
    times: List[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            result = func(*args, **kwargs)
        times.append((time.perf_counter() - t0) / number)
    return {
        "best": min(times),
        "mean": statistics.fmean(times),
        "std": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "times": times,
        "result": result,
    }


def compare(
    implementations: Dict[str, Callable], args: Sequence = (), kwargs: Optional[Dict] = None,
    baseline: Optional[str] = None, repeat: int = 5, number: int = 1, check: bool = True,
    rtol: float = 1e-6, atol: float = 1e-8, task: str = "",
) -> List[Dict[str, Any]]:
    
    kwargs = kwargs or {}
    names = list(implementations)
    if not names:
        raise ValueError("No implementations given.")
    baseline = baseline or names[0]
    res = {n: timeit(fn, *args, repeat=repeat, number=number, **kwargs) for n, fn in implementations.items()}
    if check:
        ref = res[baseline]["result"]
        for n in names:
            out = res[n]["result"]
            if isinstance(ref, tuple):
                for a, b in zip(ref, out):
                    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=rtol, atol=atol,
                                               err_msg=f"{n} disagrees with {baseline}")
            else:
                np.testing.assert_allclose(np.asarray(ref, float), np.asarray(out, float), rtol=rtol, atol=atol,
                                           err_msg=f"{n} disagrees with {baseline}")
    base = res[baseline]["best"]
    return [{
        "task": task, "name": n,
        "best_ms": res[n]["best"] * 1e3, "mean_ms": res[n]["mean"] * 1e3, "std_ms": res[n]["std"] * 1e3,
        "speedup": base / res[n]["best"] if res[n]["best"] > 0 else float("inf"),
    } for n in names]


def format_table(rows: List[Dict[str, Any]]) -> str:
    """Render :func:`compare` rows as a Markdown table."""
    lines = ["| Task | Implementation | Best (ms) | Mean ± std (ms) | Speed-up |",
             "|---|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['task']} | {r['name']} | {r['best_ms']:.3f} | "
                     f"{r['mean_ms']:.3f} ± {r['std_ms']:.3f} | {r['speedup']:.1f}× |")
    return "\n".join(lines)


def environment_info() -> Dict[str, str]:
    
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": str(os.cpu_count()),
    }
