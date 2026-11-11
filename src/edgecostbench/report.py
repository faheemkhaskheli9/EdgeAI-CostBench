"""Validate benchmark result files and avoid ranking incomparable workloads."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path


def load_results(paths: list[str]) -> list[dict]:
    if not paths:
        raise ValueError("provide at least one results JSON file")
    rows = []
    cohorts: dict[tuple, str] = {}
    for index, path in enumerate(paths):
        try:
            result = json.loads(Path(path).read_text(encoding="utf-8"))
            if result["schema_version"] != 1:
                raise ValueError(f"unsupported result schema in {path}")
            measurement = result["measurement"]
            latency = measurement["latency_ms"]
            input_meta = result["input"]
            fingerprint = input_meta.get("sha256")
            if fingerprint is not None and (not isinstance(fingerprint, str) or len(fingerprint) != 64
                                            or any(c not in "0123456789abcdef" for c in fingerprint)):
                raise ValueError(f"invalid input fingerprint in {path}")
            tensors = json.dumps(input_meta["tensors"], sort_keys=True)
            items_per_call = measurement["items_per_call"]
            p50, p95, throughput = latency["p50"], latency["p95"], measurement["items_per_second"]
            if any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0
                   for n in (p50, p95, throughput)):
                raise ValueError(f"invalid performance measurement in {path}")
            if not isinstance(items_per_call, int) or isinstance(items_per_call, bool) or items_per_call < 1:
                raise ValueError(f"invalid items_per_call in {path}")
            # Legacy schema-v1 reports lack feed identity. Keep each separate: even
            # matching shape/seed is not proof of matching actual input values.
            key = ((result["model"]["sha256"], fingerprint, tensors, items_per_call)
                   if fingerprint is not None else ("unverified", index))
            if key not in cohorts:
                cohorts[key] = f"G{len(cohorts) + 1}"
            rss = measurement["sampled_peak_process_rss_bytes"]
            rows.append({
                "result_file": path,
                "cohort": cohorts[key],
                "input_verified": fingerprint is not None,
                "model_sha256": result["model"]["sha256"],
                "input_sha256": fingerprint,
                "provider": result["runtime"]["requested_provider"],
                "hardware": result["hardware"]["machine"],
                "input_source": input_meta["source"],
                "tensors": tensors,
                "items_per_call": items_per_call,
                "latency_p50_ms": p50,
                "latency_p95_ms": p95,
                "items_per_second": throughput,
                "peak_rss_mb": rss / 1024**2 if rss is not None else None,
                "estimated_electricity_cost_per_million_calls": result["cost"]["estimated_electricity_cost_per_million_calls"],
                "currency": result["cost"]["currency"],
            })
        except (KeyError, TypeError, json.JSONDecodeError, ZeroDivisionError) as exc:
            raise ValueError(f"invalid benchmark result in {path}: {exc}") from exc
    return rows


def comparison_warnings(rows: list[dict]) -> list[str]:
    warnings = []
    if any(not row["input_verified"] for row in rows):
        warnings.append("legacy result(s) lack input SHA-256; their workloads cannot be verified")
    if len({row["cohort"] for row in rows}) > 1:
        warnings.append("incomparable workloads detected; results are ranked only within each cohort, not globally")
    return warnings


def render_comparison(rows: list[dict], format_name: str = "table",
                      allow_incomparable: bool = False) -> str:
    if not rows:
        raise ValueError("no benchmark rows to compare")
    # Default ordering is by cohort then p50; global sort requires explicit consent.
    ordered = (sorted(rows, key=lambda row: row["latency_p50_ms"])
               if allow_incomparable else
               sorted(rows, key=lambda row: (int(row["cohort"][1:]), row["latency_p50_ms"])))
    ranked = []
    for global_index, row in enumerate(ordered, 1):
        cohort_rank = 1 + sum(other["cohort"] == row["cohort"] and
                              other["latency_p50_ms"] < row["latency_p50_ms"] for other in rows)
        ranked.append({**row, "rank_in_cohort": cohort_rank,
                       "global_rank_unverified": global_index if allow_incomparable and
                       len({r["cohort"] for r in rows}) > 1 else None})
    if format_name == "json":
        return json.dumps(ranked, indent=2, sort_keys=True)
    if format_name == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(ranked[0]))
        writer.writeheader()
        writer.writerows(ranked)
        return output.getvalue().rstrip("\n")
    if format_name != "table":
        raise ValueError(f"unknown format {format_name}")
    headers = ("Group", "Rank", "Result", "Provider", "p50 ms", "p95 ms", "items/s", "RSS MB", "Cost/1M calls")
    display = []
    for row in ranked:
        price = row["estimated_electricity_cost_per_million_calls"]
        display.append((row["cohort"], str(row["rank_in_cohort"]), Path(row["result_file"]).name,
                        row["provider"], f"{row['latency_p50_ms']:.3f}", f"{row['latency_p95_ms']:.3f}",
                        f"{row['items_per_second']:.2f}",
                        "n/a" if row["peak_rss_mb"] is None else f"{row['peak_rss_mb']:.1f}",
                        "n/a" if price is None else f"{price:.5g} {row['currency']}"))
    widths = [max(len(headers[i]), *(len(row[i]) for row in display)) for i in range(len(headers))]
    lines = ["  ".join(value.ljust(widths[i]) for i, value in enumerate(headers)),
             "  ".join("-" * width for width in widths)]
    lines.extend("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)) for row in display)
    lines.append("Groups require identical model, actual input content, tensor layout and items/call. Different providers/devices can be compared within a group.")
    if len({row["cohort"] for row in ranked}) > 1:
        lines.append("WARNING: Cross-group comparison is not valid; displayed ranks are within each group."
                     if not allow_incomparable else
                     "WARNING: Forced cross-group latency ordering is not a scientifically valid ranking.")
    if any(not row["input_verified"] for row in ranked):
        lines.append("WARNING: Legacy result(s) without input hashes are unverified and isolated.")
    lines.append("Energy/cost values use assumed power, not measurements.")
    return "\n".join(lines)
