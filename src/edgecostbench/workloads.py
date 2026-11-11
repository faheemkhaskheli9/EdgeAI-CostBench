"""Versioned workload manifests and reference-output checks (not task accuracy)."""

from __future__ import annotations

import hashlib
import json
from math import isfinite
from pathlib import Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _artifact(manifest_path: Path, spec: object, label: str) -> tuple[Path, str]:
    if not isinstance(spec, dict) or not isinstance(spec.get("path"), str) or not spec["path"]:
        raise ValueError(f"{label} must contain a non-empty path")
    expected = _digest(spec.get("sha256"), f"{label}.sha256")
    file_path = Path(spec["path"])
    if not file_path.is_absolute():
        file_path = manifest_path.parent / file_path
    if not file_path.is_file():
        raise ValueError(f"{label} file missing: {file_path}")
    if _sha256_file(file_path) != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {file_path}")
    return file_path, expected


def load_manifest(manifest_path: str, model_path: str, inputs_npz: str | None,
                  items_per_call: int) -> tuple[dict, str, str]:
    """Validate an immutable workload contract before creating the session."""
    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read workload manifest: {exc}") from exc
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("workload manifest must have schema_version=1")
    for key in ("id", "task", "unit_of_work"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"workload manifest requires non-empty {key}")
    if _sha256_file(Path(model_path)) != _digest(manifest.get("model_sha256"), "model_sha256"):
        raise ValueError("model SHA-256 mismatch with workload manifest")
    if isinstance(manifest.get("items_per_call"), bool) or manifest.get("items_per_call") != items_per_call:
        raise ValueError("items_per_call differs from workload manifest")
    input_path, input_sha = _artifact(path, manifest.get("inputs"), "inputs")
    reference_path, reference_sha = _artifact(path, manifest.get("reference"), "reference")
    if inputs_npz is not None and Path(inputs_npz).resolve() != input_path.resolve():
        raise ValueError("--inputs-npz conflicts with workload manifest")
    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("workload manifest requires named tensor specs")
    for name, spec in tensors.items():
        if (not isinstance(name, str) or not name or not isinstance(spec, dict) or
                not isinstance(spec.get("dtype"), str) or not spec["dtype"] or
                not isinstance(spec.get("shape"), list) or
                any(type(dim) is not int or dim <= 0 for dim in spec["shape"])):
            raise ValueError(f"invalid tensor specification for {name}")
    reference = manifest["reference"]
    for key in ("atol", "rtol"):
        value = reference.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
            raise ValueError(f"reference.{key} must be finite and nonnegative")
    return ({"id": manifest["id"], "task": manifest["task"], "unit_of_work": manifest["unit_of_work"],
             "sha256": _sha256_file(path), "model_sha256": manifest["model_sha256"],
             "input_npz_sha256": input_sha, "reference_npz_sha256": reference_sha,
             "tensors": tensors, "atol": float(reference["atol"]), "rtol": float(reference["rtol"])},
            str(input_path), str(reference_path))


def check_tensors(feeds: dict, workload: dict) -> None:
    actual = {name: {"shape": list(array.shape), "dtype": str(array.dtype)} for name, array in feeds.items()}
    if actual != workload["tensors"]:
        raise ValueError(f"workload tensor specification mismatch: expected {workload['tensors']}, got {actual}")


def check_reference(session, feeds: dict, reference_npz: str, workload: dict) -> dict:
    """Run untimed reference check. Failure prevents producing performance numbers."""
    import numpy as np

    output_names = [meta.name for meta in session.get_outputs()]
    if len(output_names) != len(set(output_names)):
        raise ValueError("duplicate model output names")
    with np.load(reference_npz, allow_pickle=False) as archive:
        if set(archive.files) != set(output_names):
            raise ValueError("reference NPZ keys must match model outputs")
        expected = {name: archive[name] for name in output_names}
    observed = session.run(output_names, feeds)
    if len(observed) != len(output_names):
        raise ValueError("model returned wrong number of outputs")
    output_details = {}
    for name, actual in zip(output_names, observed):
        reference = expected[name]
        actual = np.asarray(actual)
        if actual.shape != reference.shape or actual.dtype != reference.dtype:
            raise ValueError(f"reference output {name} shape/dtype mismatch")
        if np.issubdtype(actual.dtype, np.floating):
            if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(reference)):
                raise ValueError(f"reference output {name} has non-finite values")
            good = np.isclose(actual, reference, rtol=workload["rtol"], atol=workload["atol"], equal_nan=False)
            max_diff = float(np.max(np.abs(actual.astype(np.float64) - reference.astype(np.float64)))) if actual.size else 0.0
        else:
            good = actual == reference
            max_diff = None
        if not bool(np.all(good)):
            raise ValueError(f"reference output mismatch for {name}: {int(np.size(good) - np.count_nonzero(good))} elements differ")
        output_details[name] = {"shape": list(actual.shape), "dtype": str(actual.dtype), "max_abs_error": max_diff}
    return {"status": "passed", "method": "reference_npz_allclose_or_exact",
            "note": "Matching known reference tensors is not a model accuracy estimate.",
            "outputs": output_details}
