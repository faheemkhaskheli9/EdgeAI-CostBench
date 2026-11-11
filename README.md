# EdgeAI-CostBench

Reproducible **edge AI inference benchmarks** for comparing affordable CPU/GPU devices, ONNX models, and assumed electricity costs. Record actual inference latency/throughput, process memory, system metadata, input configuration, model SHA-256, and explicitly labeled **estimated** energy costs.

> **Status: v0.1 MVP.** Includes versioned input/reference workload checks and an assumption-based ownership cost model. No real-world accuracy evaluation, measured power telemetry, GPU memory profiling, or standardized hardware leaderboard yet. Real-hardware ONNX tests are still pending.

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

## Versioned workloads and correctness checks

Generate two reproducible demonstration suites (224x224 preprocessed RGB image with
average pooling; 128-token sequence with an attention mask and masked sum):

```bash
python -m pip install -e '.[cpu,test]' onnx
python examples/make_workload_fixtures.py --out-dir /tmp/edgebench-workloads
edgecostbench run --model /tmp/edgebench-workloads/vision_pool.onnx \
  --manifest /tmp/edgebench-workloads/vision-pool-v1.json --runs 50 \
  --out results/vision.json
edgecostbench run --model /tmp/edgebench-workloads/nlp_masked_sum.onnx \
  --manifest /tmp/edgebench-workloads/nlp-masked-sum-v1.json --runs 50 \
  --out results/nlp.json
edgecostbench compare results/vision.json results/nlp.json
```

The version-1 JSON manifest contains the workload ID, task, unit of work,
exact model SHA-256, named input tensor shapes/dtypes, input and reference NPZ
SHA-256, number of samples per call, and reference-check tolerances. Relative
artifact paths resolve relative to the manifest, not the terminal's directory.
Reference outputs are checked **before** warmup or timing; failed checks block
benchmark results. Reports store correctness and throughput separately.
Comparison cohorts now include manifest identity; identical tensor shapes with
different reference contracts are not comparable. Unmanifested runs retain
their historical throughput behavior but report correctness as not evaluated.

**Important:** These tiny fixture models verify data/runner correctness only.
Their tensor shapes resemble vision/NLP preprocessing, but their simple
operators are not a substitute for actual ResNet, YOLO, or transformer workloads.
Passing a known-output reference check is not classification accuracy or
semantic task quality. Do not treat benchmark outputs as model-accuracy claims.
See docs/plans/02-standardized-workloads.md for the next phases.

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

For an optional **real CPU ONNX Runtime integration test**, install extras and generate a
deterministic Add model with a dynamic batch dimension and valid named NPZ feeds:

```bash
python -m pip install -e '.[cpu,test]' onnx
python examples/make_smoke_fixture.py --out-dir /tmp/edgebench-smoke
edgecostbench run --model /tmp/edgebench-smoke/add.onnx \
  --inputs-npz /tmp/edgebench-smoke/add-inputs.npz \
  --provider CPUExecutionProvider --warmup 10 --runs 100 \
  --out /tmp/edgebench-smoke/report.json
python -m pytest -q tests/test_onnx_integration.py
python --version
python -m pip show numpy onnx onnxruntime
```

The test checks expected numeric output and requested provider activation, NPZ
shape/dtype failures, symbolic dimensions shared across inputs, and dynamic-shape
overrides. The integration module **skips** when ONNX authoring or runtime packages
are missing; a skipped test is not proof of hardware compatibility. The toy Add
fixture is for pipeline correctness, not representative model accuracy or performance.
Capture device, runtime, OS, and environment metadata alongside a real-device
benchmark before publishing hardware conclusions.

## Total ownership cost and break-even

Use the separately modeled `economics` command with a saved inference run.
It uses that run's *measured items/s* but requires **assumptions** for every
financial, uptime, power and task-success input. Prices are in a single
user-selected currency (no FX conversion). Example values are illustrative:

```bash
edgecostbench economics --result results/cpu.json \
  --device-price 300 --lifespan-years 3 --maintenance-per-year 25 \
  --idle-watts 8 --load-watts 35 --electricity-price 0.20 \
  --utilization 0.5 --useful-fraction 0.95 \
  --cloud-price-per-million 2 --currency USD \
  --out results/ownership.json
# Override the default 10/50/90% utilization and 0.5x/1x/1.5x tariff sweeps:
edgecostbench economics --result results/cpu.json --device-price 300 \
  --lifespan-years 3 --idle-watts 8 --load-watts 35 \
  --electricity-price 0.20 --utilization 0.5 --useful-fraction 0.95 \
  --sensitivity-utilizations 0.2,0.5,0.8 --sensitivity-prices 0.10,0.20,0.40
```

The model assumes the device is powered on 8,760 hours/year and runs the
benchmark workload at the reported items/second during the specified
utilization fraction. Annualized costs = device price / service life + annual
maintenance + 8,760h × (idle W × (1-u) + load W × u) / 1,000 × tariff.
Annual **useful** items = measured items/s × 8,760 × 3,600 × u ×
user-assumed useful fraction. At zero utilization there is no cost per useful
item (null), although owning/idle costs remain. The cloud comparator uses
price per *million calls*, normalized by an explicit cloud useful fraction;
if omitted, the cloud useful fraction is **assumed equal** to edge's, not
measured. Break-even utilization is null if none exists in 0–100% load.
This annual model excludes cooling, networking, taxes, financing, equipment
replacement, concurrent load, and real-world validation. Never treat assumed
useful fractions or watts as sensor measurements or accuracy scores.

## Next milestones

1. **Measured energy:** opt-in NVML and external-power-meter telemetry with uncertainty/coverage and provenance.
2. **Hardware-cost modeling:** measured watts, utilization, amortized device price and duty cycle; compute cost per useful inference.
3. **Accuracy + workloads:** add real public model/task suites with dataset-backed quality metrics, repeated trials and environmental controls (demonstration reference checks implemented).
4. **Publishing:** opt-in standardized, auditable benchmark manifests and reproducible hardware comparison dashboard.

Community submissions should include complete model/runtime/hardware metadata, dataset and measurement protocol; **do not publish synthetic-input throughput as real-world accuracy or efficiency**.