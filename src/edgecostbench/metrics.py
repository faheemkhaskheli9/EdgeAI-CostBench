"""Small dependency-free statistical helpers for recorded inference durations."""

from __future__ import annotations

from statistics import mean


def percentile(values: list[float], fraction: float) -> float:
    """Linearly interpolated empirical percentile, with endpoints included."""
    if not values or not 0 <= fraction <= 1:
        raise ValueError("percentile requires values and fraction in [0, 1]")
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def summarize_latencies(latencies_ms: list[float], elapsed_seconds: float, items_per_call: int) -> dict:
    if not latencies_ms or elapsed_seconds <= 0 or items_per_call < 1:
        raise ValueError("non-empty timings, positive elapsed time and items_per_call required")
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
