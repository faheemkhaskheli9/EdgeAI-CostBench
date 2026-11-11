# Plan 02: Standardized workload contracts and correctness

## Objective

Prevent incomparable model/hardware results from entering the same performance
cohort. Record deterministic inputs and untimed reference checks, independently
of latency/throughput measurements. All runs remain reproducible and offline.

## Phase 1 — Versioned manifests and correctness gate (implemented)

- Format v1: id, task, unit_of_work, model SHA-256, items_per_call,
  input NPZ SHA-256, reference NPZ SHA-256, named input tensor shapes and dtypes,
  and finite nonnegative absolute and relative tolerances.
- Resolve artifacts relative to the manifest. Reject tampering, missing input
  names, conflicting CLI NPZ paths, mismatched shapes/dtypes and wrong model.
- Compare model outputs to reference tensors *outside* timing. If validation
  fails, emit no success/benchmark report. Record reference-check status and
  per-output maximum absolute error separately from performance.
- Include manifest SHA-256 in comparison grouping. Runs without manifests are
  reported as not evaluated and may be compared for throughput only.

## Phase 2 — Reproducible vision/NLP fixture examples (implemented)

- Generate fixed 1x3x224x224 float32 image input and a simple pooling ONNX
  model with expected output NPZ.
- Generate fixed 1x128 int64 token IDs and attention mask, with a masked-sum
  ONNX model and reference NPZ.
- Keep fixtures generated rather than committing opaque binary assets. Verify
  hashes during the run and document exact commands.
- Include simulated session checks that work without ONNX Runtime and optional
  real ONNX Runtime tests that skip if the libraries are unavailable.

## Phase 3 — Real representative models and task accuracy (open)

- Select small public models with pinned versions/licenses: e.g., MobileNet
  classification, a lightweight object detector, and a compact text encoder.
- Pin preprocessing pipelines and licensed evaluation datasets with dataset
  versions, label mappings, sample counts and immutable hashes.
- Measure task quality (e.g., top-1 / mAP / retrieval metrics) independently of
  reference-output compatibility. Check accuracy before published rankings.
- Record repeated trials, CPU affinity/thermal state, runtime version and
  device power conditions; never equate toy reference correctness with accuracy.

## Phase 4 — Public comparisons (open)

- Publish only clearly comparable task cohorts; separate accuracy, inference
  throughput, device cost and measured-vs-assumed energy. Do not rank runs
  across different manifest hashes unless a future explicit cross-model task
  standard allows it.
- Preserve the existing --allow-incomparable exploratory option but label its
  ranking unverified.

## Evidence / limitations

- Unit tests cover hash/model/batch mismatch, invalid tolerances, output
  correctness failures, and manifest grouping. Optional ONNX Runtime smoke
  tests require local ONNX packages and actual CPU execution.
- Synthetic demonstration models are **not** representative accuracy
  measurements. No physical-device validation or power telemetry is claimed.
- Free/local testing only; no paid GitHub CI dependency.
