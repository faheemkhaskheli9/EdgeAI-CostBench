"""Optional real ONNX Runtime CPU fixture tests; skipping is NOT hardware validation."""
from pathlib import Path
import importlib.util

import pytest

pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from edgecostbench.benchmark import run_benchmark


def fixture_generator():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("make_workload_fixtures", root / "examples" / "make_workload_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_fixtures


@pytest.mark.parametrize("fixture_name", ["vision-pool-v1", "nlp-masked-sum-v1"])
def test_real_cpu_workload_fixtures(tmp_path, fixture_name):
    fixture_generator()(tmp_path)
    manifest = tmp_path / f"{fixture_name}.json"
    model = tmp_path / ("vision_pool.onnx" if fixture_name.startswith("vision") else "nlp_masked_sum.onnx")
    result = run_benchmark(model_path=str(model), manifest_path=str(manifest), warmup=1, runs=3)
    assert result["correctness"]["status"] == "passed"
    assert result["workload"]["id"] == fixture_name
    assert result["runtime"]["requested_provider"] == "CPUExecutionProvider"
    assert result["runtime"]["active_providers"] == ["CPUExecutionProvider"]
    assert result["measurement"]["runs"] == 3
