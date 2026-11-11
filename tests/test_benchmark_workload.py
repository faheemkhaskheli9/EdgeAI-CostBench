from types import SimpleNamespace
import pytest

np = pytest.importorskip("numpy")

from test_workloads import make_manifest
from edgecostbench.benchmark import run_benchmark


class FakeSession:
    def __init__(self):
        self.calls = []

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_inputs(self):
        return [SimpleNamespace(name="x", type="tensor(float)", shape=["batch", 4])]

    def get_outputs(self):
        return [SimpleNamespace(name="y")]

    def run(self, names, feeds):
        self.calls.append(names)
        return [feeds["x"] * 2]


def test_reference_checked_before_timing_and_metadata_present(tmp_path):
    model, inputs, expected, path, data = make_manifest(tmp_path)
    session = FakeSession()
    result = run_benchmark(model_path=str(model), manifest_path=str(path), items_per_call=2,
                           session_factory=lambda *args: session, runs=3, warmup=2)
    assert result["correctness"]["status"] == "passed"
    assert result["workload"]["id"] == data["id"]
    assert result["input"]["source"] == "npz"
    assert result["measurement"]["runs"] == 3
    assert session.calls == [["y"], None, None, None, None, None]


def test_reference_failure_stops_before_warmup_and_timing(tmp_path):
    model, inputs, expected, path, _ = make_manifest(tmp_path)
    session = FakeSession()
    session.run = lambda names, feeds: [np.zeros((2, 4), dtype=np.float32)]
    with pytest.raises(ValueError, match="reference output mismatch"):
        run_benchmark(model_path=str(model), manifest_path=str(path), items_per_call=2,
                      session_factory=lambda *args: session, runs=10)


def test_no_manifest_never_claims_accuracy(tmp_path):
    model, inputs, expected, path, _ = make_manifest(tmp_path)
    result = run_benchmark(model_path=str(model), inputs_npz=str(inputs),
                           session_factory=lambda *args: FakeSession(), runs=2, warmup=0)
    assert result["correctness"]["status"] == "not_evaluated"
    assert result["workload"] is None
