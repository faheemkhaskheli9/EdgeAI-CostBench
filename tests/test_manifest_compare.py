import json
import pytest
from edgecostbench.report import load_results, render_comparison


def result(tmp_path, name, manifest_hash, status="passed", p50=1.0):
    data = {"schema_version": 1, "model": {"sha256": "abc"},
            "input": {"sha256": "f" * 64, "tensors": {"x": {"shape": [2, 4], "dtype": "float32"}}, "source": "npz"},
            "measurement": {"latency_ms": {"p50": p50, "p95": p50}, "items_per_second": 5,
                            "items_per_call": 2, "sampled_peak_process_rss_bytes": None},
            "runtime": {"requested_provider": "CPUExecutionProvider"}, "hardware": {"machine": "x86"},
            "cost": {"estimated_electricity_cost_per_million_calls": None, "currency": "USD"},
            "workload": {"id": name, "task": "vision", "unit_of_work": "image", "sha256": manifest_hash}
                        if manifest_hash is not None else None,
            "correctness": {"status": status}}
    path = tmp_path / (name + ".json")
    path.write_text(json.dumps(data))
    return str(path)


def test_manifest_identity_disallows_mixed_comparisons(tmp_path):
    a = result(tmp_path, "a", "a" * 64, p50=4)
    b = result(tmp_path, "b", "b" * 64, p50=1)
    c = result(tmp_path, "c", "a" * 64, p50=2)
    rows = load_results([a, b, c])
    assert [row["cohort"] for row in rows] == ["G1", "G2", "G1"]
    ranked = json.loads(render_comparison(rows, "json"))
    assert [r["result_file"] for r in ranked] == [c, a, b]
    assert [r["reference_check"] for r in ranked] == ["passed", "passed", "passed"]
    assert "Checked" in render_comparison(rows)


def test_manifest_without_passing_reference_is_rejected(tmp_path):
    path = result(tmp_path, "failed", "a" * 64, status="not_evaluated")
    with pytest.raises(ValueError, match="reference check"):
        load_results([path])


def test_checked_and_unchecked_same_tensors_not_one_group(tmp_path):
    checked = result(tmp_path, "checked", "a" * 64)
    unchecked = result(tmp_path, "unchecked", None, status="not_evaluated")
    rows = load_results([checked, unchecked])
    assert rows[0]["cohort"] != rows[1]["cohort"]
