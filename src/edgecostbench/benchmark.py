"""Warm ONNX Runtime inference benchmark (sequential, synchronous, batch-aware)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter, perf_counter_ns
from datetime import datetime, timezone

from . import __version__
from .cost import estimate_cost
from .hardware import ProcessMemorySampler, system_info
from .inputs import build_inputs
from .identity import fingerprint_feeds
from .metrics import summarize_latencies


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_benchmark(*, model_path: str, provider: str = "CPUExecutionProvider",
                  shape_overrides: dict[str, tuple[int, ...]] | None = None,
                  inputs_npz: str | None = None, warmup: int = 10, runs: int = 100,
                  seed: int = 42, items_per_call: int = 1,
                  power_watts: float | None = None,
                  electricity_price_per_kwh: float | None = None,
                  currency: str = "USD", intra_op_threads: int | None = None,
                  session_factory=None) -> dict:
    # Reject invalid cost inputs before expensive model initialization/inference.
    estimate_cost(0, power_watts, electricity_price_per_kwh, currency)
    if not Path(model_path).is_file():
        raise ValueError(f"model file not found: {model_path}")
    if warmup < 0 or runs < 1 or items_per_call < 1:
        raise ValueError("warmup must be >= 0; runs and items_per_call must be positive")
    if intra_op_threads is not None and intra_op_threads < 1:
        raise ValueError("intra_op_threads must be positive")
    if session_factory is None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX Runtime required: pip install '.[cpu]' or '.[gpu]'") from exc
        if provider not in ort.get_available_providers():
            raise ValueError(f"provider {provider!r} not available; installed: {ort.get_available_providers()}")
        opts = ort.SessionOptions()
        if intra_op_threads is not None:
            opts.intra_op_num_threads = intra_op_threads
        session = ort.InferenceSession(model_path, sess_options=opts, providers=[provider])
        runtime_version = ort.__version__
    else:
        # Test seam for simulating ONNX runtime independently of hardware.
        session = session_factory(model_path, provider)
        runtime_version = "test-double"
    actual_providers = session.get_providers()
    if provider not in actual_providers:
        raise RuntimeError(f"requested provider {provider} not active; actual: {actual_providers}")
    feeds = build_inputs(session.get_inputs(), shape_overrides or {}, seed=seed, npz_path=inputs_npz)
    for _ in range(warmup):
        session.run(None, feeds)
    durations: list[float] = []
    with ProcessMemorySampler() as memory:
        wall_start = perf_counter()
        for _ in range(runs):
            started = perf_counter_ns()
            session.run(None, feeds)  # ONNX Runtime run is synchronous.
            durations.append((perf_counter_ns() - started) / 1_000_000)
        elapsed = perf_counter() - wall_start
    summary = summarize_latencies(durations, elapsed, items_per_call)
    summary["sampled_peak_process_rss_bytes"] = memory.peak_rss_bytes
    return {
        "schema_version": 1,
        "tool": {"name": "edgeai-costbench", "version": __version__},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
        "runtime": {
            "name": "onnxruntime", "version": runtime_version,
            "requested_provider": provider, "active_providers": actual_providers,
            "intra_op_threads": intra_op_threads,
        },
        "hardware": system_info(),
        "input": {
            "source": "npz" if inputs_npz else "synthetic",
            "sha256": fingerprint_feeds(feeds),
            "npz_sha256": sha256_file(inputs_npz) if inputs_npz else None,
            "npz_path": inputs_npz,
            "seed": seed if inputs_npz is None else None,
            "tensors": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in feeds.items()},
        },
        "measurement": {
            "method": "sequential synchronous calls, warmup excluded, no model load or input generation in timed region",
            "warmup_runs": warmup, **summary,
            "memory_notes": "Best-effort sampled process RSS; may miss peaks and excludes GPU memory. Null if psutil unavailable.",
        },
        "cost": estimate_cost(summary["latency_ms"]["mean"], power_watts,
                              electricity_price_per_kwh, currency),
    }
