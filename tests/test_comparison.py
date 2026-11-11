import json

import pytest

np = pytest.importorskip("numpy")

from edgecostbench.identity import fingerprint_feeds
from edgecostbench.report import load_results, render_comparison, comparison_warnings


def make_result(tmp_path, name, model="model1", input_hash="1" * 64, p50=3.0,
                items=1, tensors=None):
    data = {"schema_version": 1, "model": {"sha256": model},
            "input": {"source": "synthetic", "tensors": tensors or {"x": {"shape": [1, 4], "dtype": "float32"}}},
            "runtime": {"requested_provider": "CPUExecutionProvider"},
            "hardware": {"machine": "x86_64"},
            "measurement": {"latency_ms": {"p50": p50, "p95": p50 + 1},
                            "items_per_call": items, "items_per_second": 12.3,
                            "sampled_peak_process_rss_bytes": None},
            "cost": {"estimated_electricity_cost_per_million_calls": None, "currency": "USD"}}
    if input_hash is not None:
        data["input"]["sha256"] = input_hash
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_feed_hash_discriminates_values_types_shapes_and_names():
    a = np.array([[1, 2]], dtype=np.int32)
    hash1 = fingerprint_feeds({"x": a})
    assert hash1 == fingerprint_feeds({"x": a.copy()})
    assert hash1 != fingerprint_feeds({"x": np.array([[1, 3]], dtype=np.int32)})
    assert hash1 != fingerprint_feeds({"x": np.array([1, 2], dtype=np.int32)})
    assert hash1 != fingerprint_feeds({"x": a.astype(np.int64)})
    assert hash1 != fingerprint_feeds({"other": a})
    assert fingerprint_feeds({"x": a, "y": a}) == fingerprint_feeds({"y": a, "x": a})
    assert len(hash1) == 64


def test_incompatible_results_are_grouped_not_globally_ranked(tmp_path):
    a = make_result(tmp_path, "slow", p50=9)
    b = make_result(tmp_path, "different-model", model="model2", p50=1)
    c = make_result(tmp_path, "same", p50=3)
    d = make_result(tmp_path, "different-input", input_hash="2" * 64, p50=0.5)
    e = make_result(tmp_path, "different-batch", items=2, p50=0.2)
    rows = load_results([a, b, c, d, e])
    assert [r["cohort"] for r in rows] == ["G1", "G2", "G1", "G3", "G4"]
    ordered = json.loads(render_comparison(rows, "json"))
    assert [r["result_file"] for r in ordered] == [c, a, b, d, e]
    assert [r["rank_in_cohort"] for r in ordered] == [1, 2, 1, 1, 1]
    assert all(r["global_rank_unverified"] is None for r in ordered)
    assert "incomparable" in comparison_warnings(rows)[0]
    force = json.loads(render_comparison(rows, "json", allow_incomparable=True))
    assert force[0]["result_file"] == e
    assert force[0]["global_rank_unverified"] == 1
    assert "WARNING" in render_comparison(rows)
    assert "cohort" in render_comparison(rows, "csv")


def test_legacy_results_cannot_prove_same_inputs(tmp_path):
    a = make_result(tmp_path, "oldA", input_hash=None)
    b = make_result(tmp_path, "oldB", input_hash=None)
    rows = load_results([a, b])
    assert rows[0]["cohort"] != rows[1]["cohort"]
    assert len(comparison_warnings(rows)) == 2


def test_bad_digest_and_nonfinite_metrics_are_rejected(tmp_path):
    a = make_result(tmp_path, "bad", input_hash="bad")
    with pytest.raises(ValueError, match="fingerprint"):
        load_results([a])
    b = make_result(tmp_path, "inf", p50=float("inf"))
    with pytest.raises(ValueError, match="measurement"):
        load_results([b])
    with pytest.raises(ValueError, match="no benchmark"):
        render_comparison([])
