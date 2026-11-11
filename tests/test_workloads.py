import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from edgecostbench.workloads import check_reference, check_tensors, load_manifest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_manifest(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"deterministic model")
    inputs = tmp_path / "inputs.npz"
    reference = tmp_path / "expected.npz"
    np.savez(inputs, x=np.ones((2, 4), dtype=np.float32))
    np.savez(reference, y=np.full((2, 4), 2, dtype=np.float32))
    manifest = {"schema_version": 1, "id": "test-v1", "task": "unit_test", "unit_of_work": "sample",
                "model_sha256": sha(model), "items_per_call": 2,
                "inputs": {"path": inputs.name, "sha256": sha(inputs)},
                "reference": {"path": reference.name, "sha256": sha(reference), "atol": 1e-5, "rtol": 1e-5},
                "tensors": {"x": {"shape": [2, 4], "dtype": "float32"}}}
    path = tmp_path / "workload.json"
    path.write_text(json.dumps(manifest))
    return model, inputs, reference, path, manifest


class FakeSession:
    def get_outputs(self):
        return [SimpleNamespace(name="y")]

    def run(self, names, feeds):
        assert names == ["y"]
        return [feeds["x"] * 2]


def test_manifest_and_reference_pass(tmp_path):
    model, inputs, reference, path, _ = make_manifest(tmp_path)
    meta, in_path, ref_path = load_manifest(str(path), str(model), None, 2)
    assert Path(in_path) == inputs
    assert Path(ref_path) == reference
    assert meta["sha256"] == sha(path)
    feeds = {"x": np.ones((2, 4), dtype=np.float32)}
    check_tensors(feeds, meta)
    check = check_reference(FakeSession(), feeds, ref_path, meta)
    assert check["status"] == "passed"
    assert check["outputs"]["y"]["max_abs_error"] == 0


@pytest.mark.parametrize("field,expected", [
    ("model_sha256", "model SHA-256"), ("items_per_call", "items_per_call"),
    ("schema_version", "schema_version"), ("tensors", "tensor specs"),
])
def test_manifest_rejects_modified_contract(tmp_path, field, expected):
    model, _, _, path, manifest = make_manifest(tmp_path)
    manifest[field] = {"model_sha256": "0" * 64, "items_per_call": 1,
                       "schema_version": 2, "tensors": {}}[field]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=expected):
        load_manifest(str(path), str(model), None, 2)


def test_hashes_and_override_reject_mismatches(tmp_path):
    model, inputs, reference, path, manifest = make_manifest(tmp_path)
    other = tmp_path / "other.npz"
    other.write_bytes(inputs.read_bytes())
    with pytest.raises(ValueError, match="conflicts"):
        load_manifest(str(path), str(model), str(other), 2)
    inputs.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="inputs SHA-256"):
        load_manifest(str(path), str(model), None, 2)
    manifest["inputs"]["sha256"] = sha(inputs)
    path.write_text(json.dumps(manifest))
    reference.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="reference SHA-256"):
        load_manifest(str(path), str(model), None, 2)


def test_fail_closed_on_output_difference_and_tensor_specs(tmp_path):
    model, _, reference, path, _ = make_manifest(tmp_path)
    workload, _, ref_path = load_manifest(str(path), str(model), None, 2)
    with pytest.raises(ValueError, match="tensor specification"):
        check_tensors({"x": np.zeros((1, 4), dtype=np.float32)}, workload)
    with pytest.raises(ValueError, match="output mismatch"):
        check_reference(FakeSession(), {"x": np.zeros((2, 4), dtype=np.float32)}, ref_path, workload)
    np.savez(reference, y=np.full((2, 4), 2, dtype=np.float64))
    with pytest.raises(ValueError, match="shape/dtype"):
        check_reference(FakeSession(), {"x": np.ones((2, 4), dtype=np.float32)}, str(reference), workload)


def test_no_nonfinite_tolerances_and_bad_npz_keys(tmp_path):
    model, inputs, reference, path, manifest = make_manifest(tmp_path)
    manifest["reference"]["atol"] = float("nan")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="atol"):
        load_manifest(str(path), str(model), None, 2)
    manifest["reference"]["atol"] = 0.0
    path.write_text(json.dumps(manifest))
    workload, _, ref_path = load_manifest(str(path), str(model), None, 2)
    np.savez(reference, bad=np.ones((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="keys must match"):
        check_reference(FakeSession(), {"x": np.ones((2, 4), dtype=np.float32)}, str(reference), workload)
