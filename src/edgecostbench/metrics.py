"""Small dependency-free statistical helpers for recorded inference durations.

Regression examples (also runnable with ``python -m doctest -v``):

>>> import math
>>> percentile([1.0, 2.0, 3.0, 4.0], 0.5)
2.5
>>> summarize_latencies([1.0, 2.0], 0.5, 1)["calls_per_second"]
4.0
>>> for_bad = [math.nan, math.inf, -1.0]
>>> all(_rejects_bad_latency(x) for x in for_bad)
True
>>> all(_rejects_bad_elapsed(x) for x in [math.nan, math.inf, 0.0, -1.0])
True
"""

from __future__ import annotations

from math import isfinite
from statistics import mean


def percentile(values: list[float], fraction: float) -> float:
    """Linearly interpolated empirical percentile, with endpoints included."""
    if not values or not 0 <= fraction <= 1 or not isfinite(fraction):
        raise ValueError("percentile requires values and finite fraction in [0, 1]")
    if any(not isfinite(value) for value in values):
        raise ValueError("percentile values must be finite")
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def summarize_latencies(latencies_ms: list[float], elapsed_seconds: float, items_per_call: int) -> dict:
    if not latencies_ms or not isfinite(elapsed_seconds) or elapsed_seconds <= 0 or items_per_call < 1:
        raise ValueError("non-empty timings, finite positive elapsed time and items_per_call required")
    if any(not isfinite(value) or value < 0 for value in latencies_ms):
        raise ValueError("latencies must be finite and nonnegative")
    rate = len(latencies_ms) / elapsed_seconds
    return {
        "runs": len(latencies_ms),
        "latency_ms": {
            "min": min(latencies_ms),
            "mean": mean(latencies_ms),
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
            "max": max(latencies_ms),
        },
        "wall_seconds": elapsed_seconds,
        "calls_per_second": rate,
        "items_per_call": items_per_call,
        "items_per_second": rate * items_per_call,
    }


def _rejects_bad_latency(value: float) -> bool:
    """Internal regression helper for the module's executable documentation."""
    try:
        summarize_latencies([value], 1.0, 1)
        return False
    except ValueError:
        return True


def _rejects_bad_elapsed(value: float) -> bool:
    """Internal regression helper for the module's executable documentation."""
    try:
        summarize_latencies([1.0], value, 1)
        return False
    except ValueError:
        return True
