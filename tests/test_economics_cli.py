import json

from edgecostbench.cli import main


def args(path):
    return ["economics", "--result", str(path), "--device-price", "1000",
            "--lifespan-years", "3", "--idle-watts", "10", "--load-watts", "50",
            "--electricity-price", "0.2", "--utilization", "0.5", "--useful-fraction", "0.9"]


def test_economics_cli_reads_benchmark_and_exports_json(tmp_path, capsys):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"schema_version": 1, "measurement": {"items_per_second": 20},
                                "model": {"sha256": "sha"}}))
    dest = tmp_path / "total.json"
    assert main(args(path) + ["--out", str(dest), "--cloud-price-per-million", "3"]) == 0
    doc = json.loads(dest.read_text())
    assert doc["report_type"] == "economics_tco"
    assert doc["model_sha256"] == "sha"
    assert doc["economics"]["assumptions"]["items_per_second"] == 20
    assert len(doc["sensitivity"]) == 9
    assert doc["economics"]["annual"]["total_cost"] > 0
    assert capsys.readouterr().err == ""


def test_economics_cli_catches_invalid_source_and_ranges(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 1}))
    assert main(args(bad)) == 2
    assert "throughput" in capsys.readouterr().err
    bad.write_text(json.dumps({"schema_version": 1, "measurement": {"items_per_second": 30},
                               "model": {"sha256": "sha"}}))
    assert main(args(bad) + ["--sensitivity-utilizations", "wrong"] ) == 2
    assert "comma-separated" in capsys.readouterr().err
    assert main(args(bad) + ["--utilization", "0"]) == 0
    assert json.loads(capsys.readouterr().out)["economics"]["edge_cost_per_million_useful_items"] is None
