"""Validate benchmark result files and render lightweight comparisons."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path


def load_results(paths: list[str]) -> list[dict]:
    if not paths:
        raise ValueError("provide at least one results JSON file")
    rows = []
    for path in paths:
        try:
            result = json.loads(Path(path).read_text(encoding="utf-8"))
            if result["schema_version"] != 1:
                raise ValueError(f"unsupported result schema in {path}")
            measurement = result["measurement"]
            latency = measurement["latency_ms"]
            rows.append({
                "result_file": path,
                "model_sha256": result["model"]["sha256"],
                "provider": result["runtime"]["requested_provider"],
                "hardware": result["hardware"]["machine"],
                "input_source": result["input"]["source"],
                "tensors": json.dumps(result["input"]["tensors"], sort_keys=True),
                "items_per_call": measurement["items_per_call"],
                "latency_p50_ms": latency["p50"],
                "latency_p95_ms": latency["p95"],
                "items_per_second": measurement["items_per_second"],
                "peak_rss_mb": (measurement["sampled_peak_process_rss_bytes"] / 1024**2
                                if measurement["sampled_peak_process_rss_bytes"] is not None else None),
                "estimated_electricity_cost_per_million_calls": result["cost"]["estimated_electricity_cost_per_million_calls"],
                "currency": result["cost"]["currency"],
            })
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid benchmark result in {path}: {exc}") from exc
    return sorted(rows, key=lambda row: row["latency_p50_ms"])


def render_comparison(rows: list[dict], format_name: str = "table") -> str:
    if format_name == "json":
        return json.dumps(rows, indent=2, sort_keys=True)
    if format_name == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue().rstrip("\n")
    if format_name != "table":
        raise ValueError(f"unknown format {format_name}")
    headers = ("Result", "Provider", "p50 ms", "p95 ms", "items/s", "RSS MB", "Cost/1M calls")
    display = []
    for row in rows:
        price = row["estimated_electricity_cost_per_million_calls"]
        display.append((Path(row["result_file"]).name, row["provider"],
                        f"{row['latency_p50_ms']:.3f}", f"{row['latency_p95_ms']:.3f}",
                        f"{row['items_per_second']:.2f}",
                        "n/a" if row["peak_rss_mb"] is None else f"{row['peak_rss_mb']:.1f}",
                        "n/a" if price is None else f"{price:.5g} {row['currency']}"))
    widths = [max(len(headers[i]), *(len(row[i]) for row in display)) for i in range(len(headers))]
    lines = ["  ".join(value.ljust(widths[i]) for i, value in enumerate(headers)),
             "  ".join("-" * width for width in widths)]
    lines.extend("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)) for row in display)
    lines.append("Note: compare like-for-like model inputs, batch sizes, and hardware. Energy/cost values use assumed power, not measurements.")
    return "\n".join(lines)
