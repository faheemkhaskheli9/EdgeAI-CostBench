"""Optional real CPU ONNX Runtime integration tests (skip if extras unavailable)."""

import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from edgecostbench.benchmark import run_benchmark
from edgecostbench.cli import main
from edgecostbench.inputs import build_inputs


def _fixture_generator():
    path = Path(__file__).resolve().parents[1] / "examples" / "make_smoke_fixture.py"
    spec = importlib.util.spec_from_file_location("make_smoke_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_fixture


def test_real_cpu_dynamic_npz_and_correct_output(tmp_path):
    assert "CPUExecutionProvider" in ort.get_available_providers()
    model_file, inputs_file = _fixture_generator()(tmp_path)
    session = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
    assert session.get_providers() == ["CPUExecutionProvider"]
    feeds = build_inputs(session.get_inputs(), {}, npz_path=str(inputs_file))
    np.testing.assert_array_equal(session.run(None, feeds)[0], np.full((2, 4), 9, dtype=np.float32))

    result = run_benchmark(model_path=str(model_file), inputs_npz=str(inputs_file), warmup=1, runs=3)
    assert result["runtime"]["requested_provider"] == "CPUExecutionProvider"
    assert result["runtime"]["active_providers"] == ["CPUExecutionProvider"]
    assert result["runtime"]["version"] == ort.__version__
    assert result["input"]["tensors"]["x"]["shape"] == [2, 4]
    assert result["input"]["sha256"] and result["input"]["npz_sha256"]
    assert result["measurement"]["runs"] == 3
    assert result["measurement"]["items_per_second"] > 0
    assert result["cost"]["energy_measurement"] is None

    output = tmp_path / "benchmark.json"
    assert main(["run", "--model", str(model_file), "--inputs-npz", str(inputs_file),
                 "--runs", "2", "--warmup", "1", "--out", str(output)]) == 0
    assert json.loads(output.read_text())["measurement"]["runs"] == 2


def test_real_cpu_invalid_npz_and_provider(tmp_path):
    model_file, inputs_file = _fixture_generator()(tmp_path)
    with np.load(inputs_file) as archived:
        x, y = archived["x"], archived["y"]
    invalid = tmp_path / "invalid.npz"
    np.savez(invalid, x=x, y=y.astype(np.float64))
    with pytest.raises(ValueError, match="expected dtype"):
        run_benchmark(model_path=str(model_file), inputs_npz=str(invalid), warmup=0, runs=1)
    np.savez(invalid, x=x, y=y[:1])
    with pytest.raises(ValueError, match="symbolic dimension"):
        run_benchmark(model_path=str(model_file), inputs_npz=str(invalid), warmup=0, runs=1)
    with pytest.raises(ValueError, match="not available"):
        run_benchmark(model_path=str(model_file), provider="UnobtainableProvider", warmup=0, runs=1)
    with pytest.raises(ValueError, match="dynamic shape"):
        run_benchmark(model_path=str(model_file), warmup=0, runs=1)
    result = run_benchmark(model_path=str(model_file),
                           shape_overrides={"x": (2, 4), "y": (2, 4)}, warmup=0, runs=1)
    assert result["input"]["source"] == "synthetic"
    assert result["input"]["tensors"]["y"]["shape"] == [2, 4]
