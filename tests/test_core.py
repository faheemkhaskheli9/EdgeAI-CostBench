import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from edgecostbench.benchmark import run_benchmark
from edgecostbench.cost import estimate_cost
from edgecostbench.inputs import build_inputs, parse_shapes
from edgecostbench.metrics import percentile, summarize_latencies
from edgecostbench.report import load_results, render_comparison
from edgecostbench.cli import main


def test_statistics_and_throughput():
    assert percentile([40, 10, 20, 30], 0.5) == 25
    assert percentile([40, 10, 20, 30], 0.95) == pytest.approx(38.5)
    m = summarize_latencies([10, 20], 0.1, 4)
    assert m["calls_per_second"] == 20
    assert m["items_per_second"] == 80
    assert m["latency_ms"]["mean"] == 15
    with pytest.raises(ValueError):
        percentile([], 0.5)
    with pytest.raises(ValueError):
        summarize_latencies([10], 0.0, 1)


def test_cost_is_estimate_not_measurement():
    cost = estimate_cost(10, 100, 0.20, "USD")
    assert cost["energy_measurement"] is None
    assert cost["estimated_kwh_per_call"] == pytest.approx(100 * 10 / 3_600_000_000)
    assert cost["estimated_electricity_cost_per_million_calls"] == pytest.approx(0.0555555555)
    assert estimate_cost(10, None, None, "PKR")["estimated_kwh_per_call"] is None
    assert estimate_cost(10, 10, None, "PKR")["estimated_electricity_cost_per_million_calls"] is None
    with pytest.raises(ValueError):
        estimate_cost(10, math.nan, 1, "USD")
    with pytest.raises(ValueError):
        estimate_cost(10, 20, math.inf, "USD")
    with pytest.raises(ValueError, match="requires"):
        estimate_cost(10, None, 20, "PKR")
    with pytest.raises(ValueError):
        estimate_cost(10, -5, 2, "USD")


def test_shape_parsing_errors():
    assert parse_shapes(["images=1x3x224x224", "token=1x16"]) == {"images": (1, 3, 224, 224), "token": (1, 16)}
    for bad in ["images", "images=1x0", "=3", "input=2xa", "input=-1"]:
        with pytest.raises(ValueError):
            parse_shapes([bad])
    with pytest.raises(ValueError, match="duplicate"):
        parse_shapes(["input=1", "input=2"])


def test_synthetic_inputs_validate_types_and_dynamic_shape(tmp_path):
    np = pytest.importorskip("numpy")
    metas = [SimpleNamespace(name="images", type="tensor(float)", shape=["batch", 3]),
             SimpleNamespace(name="mask", type="tensor(bool)", shape=[1, 2])]
    overrides = {"images": (2, 3)}
    feeds = build_inputs(metas, overrides, seed=7)
    assert feeds["images"].shape == (2, 3)
    assert feeds["images"].dtype == np.float32
    assert feeds["mask"].dtype == np.bool_
    assert np.array_equal(feeds["images"], build_inputs(metas, overrides, seed=7)["images"])
    with pytest.raises(ValueError, match="dynamic"):
        build_inputs(metas, {}, seed=7)
    with pytest.raises(ValueError, match="not in model"):
        build_inputs(metas, {"unknown": (1,)})
    with pytest.raises(ValueError, match="expects dimension"):
        build_inputs(metas, {"images": (2, 4)})
    saved = tmp_path / "inputs.npz"
    np.savez(saved, images=np.zeros((2, 3), dtype=np.float32), mask=np.ones((1, 2), dtype=np.bool_))
    from_npz = build_inputs(metas, {}, npz_path=str(saved))
    assert np.array_equal(from_npz["mask"], np.ones((1, 2), dtype=np.bool_))
    with pytest.raises(ValueError, match="differs"):
        build_inputs(metas, {"images": (1, 3)}, npz_path=str(saved))
    wrong = tmp_path / "wrong.npz"
    np.savez(wrong, images=np.zeros((2, 3), dtype=np.float64), mask=np.ones((1, 2), dtype=np.bool_))
    with pytest.raises(ValueError, match="expected dtype"):
        build_inputs(metas, {}, npz_path=str(wrong))
    np.savez(wrong, images=np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="missing"):
        build_inputs(metas, {}, npz_path=str(wrong))


def test_benchmark_mock_runtime_and_compare(tmp_path, capsys):
    np = pytest.importorskip("numpy")
    model = tmp_path / "model.onnx"
    model.write_bytes(b"dummy-model-for-injected-session")

    class FakeSession:
        def __init__(self):
            self.count = 0

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return [SimpleNamespace(name="x", type="tensor(float)", shape=[1, 4])]

        def run(self, outputs, feeds):
            self.count += 1
            assert outputs is None
            assert feeds["x"].dtype == np.float32
            return [feeds["x"] * 2]

    fake = FakeSession()
    result = run_benchmark(model_path=str(model), warmup=3, runs=5, items_per_call=1,
                           power_watts=10, electricity_price_per_kwh=0.5,
                           session_factory=lambda path, provider: fake)
    assert fake.count == 8
    assert result["schema_version"] == 1
    assert result["model"]["sha256"]
    assert result["measurement"]["runs"] == 5
    assert result["measurement"]["latency_ms"]["p95"] >= result["measurement"]["latency_ms"]["p50"]
    assert result["cost"]["energy_measurement"] is None
    saved = tmp_path / "result.json"
    saved.write_text(json.dumps(result))
    rows = load_results([str(saved)])
    assert rows[0]["provider"] == "CPUExecutionProvider"
    assert "p50 ms" in render_comparison(rows)
    assert "result_file" in render_comparison(rows, "csv")
    assert json.loads(render_comparison(rows, "json"))[0]["model_sha256"] == result["model"]["sha256"]
    assert main(["compare", str(saved), "--format", "csv"]) == 0
    assert "result_file" in capsys.readouterr().out
    assert main(["run", "--model", "non-existent.onnx"]) == 2
    assert "model file not found" in capsys.readouterr().err
    saved.write_text('{"schema_version": 100}')
    with pytest.raises(ValueError, match="unsupported"):
        load_results([str(saved)])


def test_validation_prevents_invalid_runs(tmp_path):
    model = tmp_path / "some.onnx"
    model.write_bytes(b"fake")
    with pytest.raises(ValueError, match="runs"):
        run_benchmark(model_path=str(model), runs=0)
    with pytest.raises(ValueError, match="warmup"):
        run_benchmark(model_path=str(model), warmup=-1)


def test_optional_real_onnx_runtime_smoke(tmp_path):
    np = pytest.importorskip("numpy")
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "identity", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)],
                              ir_version=8)
    path = tmp_path / "identity.onnx"
    onnx.save_model(model, path)
    out = tmp_path / "output.json"
    assert main(["run", "--model", str(path), "--warmup", "1", "--runs", "3", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["measurement"]["runs"] == 3
    assert report["runtime"]["requested_provider"] == "CPUExecutionProvider"
    assert report["cost"]["estimated_electricity_cost_per_million_calls"] is None
