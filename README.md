# EdgeAI-CostBench

Reproducible **edge AI inference benchmarks** for comparing affordable CPU/GPU devices, ONNX models, and assumed electricity costs. Record actual inference latency/throughput, process memory, system metadata, input configuration, model SHA-256, and explicitly labeled **estimated** energy costs.

> **Status: v0.1 MVP.** ONNX Runtime CPU and installed execution providers are supported. No energy sensor telemetry, accuracy validation, GPU memory profiling, or capital/hosting cost model yet. Results are local experiments, **not** a hardware leaderboard.

## Install

Python >= 3.10:

```bash
python -m pip install -e '.[cpu]'
# For CUDA environments, use .[gpu] instead (ensure compatible CUDA/cuDNN drivers).
```

Benchmark an existing ONNX model with **real inference**. Inputs are deterministic synthetic tensors unless `--inputs-npz` is supplied. They are not suitable for evaluating model accuracy.

```bash
edgecostbench run --model model.onnx --warmup 10 --runs 100 \
  --power-watts 25 --electricity-price 0.20 --currency USD \
  --out results/cpu.json

# For dynamic input sizes, specify each dynamic input's entire shape:
edgecostbench run --model model.onnx --shape images=1x3x224x224 --runs 100 \
  --out results/cpu.json

# To run with known valid inference inputs, create an NPZ whose keys match
# ONNX input names and whose tensors match the exact model shape and dtype.
edgecostbench run --model model.onnx --inputs-npz representative_inputs.npz \
  --provider CPUExecutionProvider --out results/representative.json

# Compare results. Only matching workloads receive a valid within-group rank:
edgecostbench compare results/cpu.json results/representative.json
# Explicitly force a cross-group latency sort (NOT a valid model ranking):
edgecostbench compare results/cpu.json results/representative.json --allow-incomparable
edgecostbench compare results/cpu.json --format csv --out results/summary.csv
```

You can also run without installation using `PYTHONPATH=src python -m edgecostbench --help`. `edgecostbench run` requires NumPy and ONNX Runtime. `psutil` is optional for process RSS sampling.

## Metrics and methodology

- ONNX session creation and input generation happen **before** timing. Warmup calls are excluded.
- Sequential synchronous `session.run()` calls produce min/mean/p50/p95/max **per-call** latency. Wall-clock rate includes Python loop overhead; `items/s = calls/s * --items-per-call`.
- Synthetic random inputs are seeded, and shape/type metadata is recorded. For models that require semantically valid IDs/masks, provide a valid NPZ; invalid inputs may fail or change execution paths.
- Peak RSS is **sampled process resident memory**, not accelerator memory or a guaranteed allocation peak. Missing `psutil` means null RSS.
- `--power-watts` is a **user assumption**, not a wattmeter reading. The energy estimate is `watts * mean_latency_ms / 3,600,000,000` kWh/call. Electricity cost per million calls multiplies by user-supplied price/kWh. No power assumption means no energy estimate, and no price means no cost estimate.
- ONNX Runtime may assign individual operators to different execution providers. Requested/active providers are recorded; per-op placement isn't measured. GPU run times assume synchronous `session.run()` completion.
- Performance results **do not measure accuracy, thermal throttling over extended periods, concurrent capacity, setup time, amortized hardware prices, cloud prices, or idle power**. Compare only runs with equivalent input tensors, batching, and machine/environment.

Results use stable `schema_version: 1` JSON; `compare` supports table, JSON, and CSV. Each new run fingerprints the *actual tensor feed contents* (names, shapes, dtypes, and values), in addition to its model SHA-256. Comparisons group results only when model hash, input fingerprint, tensor layout, and `items_per_call` match. A CPU versus GPU comparison is possible inside one group. Legacy reports without input fingerprints are isolated, even when shapes look identical. JSON/CSV include group IDs and within-group ranks. `--allow-incomparable` forces a cross-group sort **only for exploratory viewing** and must never be used as an evidence-based leaderboard ranking. The stored model SHA-256 and runtime/provider/hardware metadata help catch comparisons between incompatible experiments. No inference input contents are stored; NPZ paths may include private local paths, so review reports before publishing.

## Development

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

For an optional end-to-end ONNX test, install ONNX and ONNX Runtime with `python -m pip install '.[cpu]' onnx` and run the same test suite.

## Next milestones

1. **Measured energy:** opt-in NVML and external-power-meter telemetry with uncertainty/coverage and provenance.
2. **Hardware-cost modeling:** measured watts, utilization, amortized device price and duty cycle; compute cost per useful inference.
3. **Accuracy + workloads:** fixed public model/task suites with representative inputs and correctness metrics; repeated trials and environmental controls.
4. **Publishing:** opt-in standardized, auditable benchmark manifests and reproducible hardware comparison dashboard.

Community submissions should include complete model/runtime/hardware metadata, dataset and measurement protocol; **do not publish synthetic-input throughput as real-world accuracy or efficiency**.