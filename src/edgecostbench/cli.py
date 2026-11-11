"""Command-line entrypoint for benchmarking and comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edgecostbench", description="Measure real ONNX inference; estimate electricity costs")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a warm inference benchmark")
    run.add_argument("--model", required=True, help="path to an ONNX model")
    run.add_argument("--provider", default="CPUExecutionProvider", help="ONNX Runtime execution provider")
    run.add_argument("--shape", action="append", default=[], help="override dynamic input shape: NAME=1x3x224x224; repeatable")
    run.add_argument("--inputs-npz", help="NPZ containing matching named input tensors")
    run.add_argument("--manifest", help="versioned workload JSON: pins model, input NPZ and reference outputs")
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--runs", type=int, default=100)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--items-per-call", type=int, default=1, help="number of independent samples in one call; affects items/s only")
    run.add_argument("--intra-op-threads", type=int, help="ONNX Runtime intra-op thread count")
    run.add_argument("--power-watts", type=float, help="ASSUMED average device watts during infer-only calls (not measured)")
    run.add_argument("--electricity-price", type=float, help="price per kWh; requires --power-watts")
    run.add_argument("--currency", default="USD", help="label for electricity-price currency, e.g. PKR")
    run.add_argument("--out", help="write JSON to this path in addition to stdout")
    compare = sub.add_parser("compare", help="compare saved JSON results")
    compare.add_argument("results", nargs="+", help="run output JSON paths")
    compare.add_argument("--format", choices=("table", "json", "csv"), default="table")
    compare.add_argument("--out", help="write rendered comparison to this path")
    compare.add_argument("--allow-incomparable", action="store_true",
                         help="force global latency sorting across incompatible workloads (invalid for benchmarking)")
    economics = sub.add_parser("economics", help="estimate annualized edge TCO from a saved benchmark")
    economics.add_argument("--result", required=True, help="saved run JSON with actual measured items/s")
    economics.add_argument("--device-price", type=float, required=True)
    economics.add_argument("--lifespan-years", type=float, required=True)
    economics.add_argument("--maintenance-per-year", type=float, default=0.0)
    economics.add_argument("--idle-watts", type=float, required=True, help="assumed watts when not inferring")
    economics.add_argument("--load-watts", type=float, required=True, help="assumed watts during inference")
    economics.add_argument("--electricity-price", type=float, required=True, help="tariff per kWh")
    economics.add_argument("--utilization", type=float, required=True, help="fraction of year inferring, 0..1")
    economics.add_argument("--useful-fraction", type=float, required=True, help="assumed correct/useful fraction, (0,1]")
    economics.add_argument("--cloud-price-per-million", type=float, help="cloud cost per 1M API calls")
    economics.add_argument("--cloud-useful-fraction", type=float, help="assumed cloud useful fraction, else equal to edge")
    economics.add_argument("--sensitivity-utilizations", default="0.1,0.5,0.9")
    economics.add_argument("--sensitivity-prices", help="tariffs per kWh separated by commas; defaults to 0.5x/1x/1.5x base")
    economics.add_argument("--currency", default="USD")
    economics.add_argument("--out", help="write JSON to this path in addition to stdout")
    return parser


def _float_values(raw: str) -> list[float]:
    try:
        values = [float(value.strip()) for value in raw.split(",")]
    except ValueError as exc:
        raise ValueError("sensitivity ranges must be comma-separated numbers") from exc
    if not values:
        raise ValueError("sensitivity ranges cannot be empty")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            from .benchmark import run_benchmark
            from .inputs import parse_shapes
            report = run_benchmark(
                model_path=args.model, provider=args.provider, shape_overrides=parse_shapes(args.shape),
                inputs_npz=args.inputs_npz, warmup=args.warmup, runs=args.runs, seed=args.seed,
                items_per_call=args.items_per_call, power_watts=args.power_watts,
                electricity_price_per_kwh=args.electricity_price, currency=args.currency,
                intra_op_threads=args.intra_op_threads, manifest_path=args.manifest,
            )
            rendered = json.dumps(report, indent=2, sort_keys=True)
        elif args.command == "compare":
            from .report import comparison_warnings, load_results, render_comparison
            rows = load_results(args.results)
            for warning in comparison_warnings(rows):
                print(f"warning: {warning}", file=sys.stderr)
            rendered = render_comparison(rows, args.format, allow_incomparable=args.allow_incomparable)
        else:
            from .economics import estimate_total_cost, sensitivity_grid
            source = json.loads(Path(args.result).read_text(encoding="utf-8"))
            if not isinstance(source, dict) or source.get("schema_version") != 1:
                raise ValueError("economics requires a saved schema_version=1 benchmark result")
            try:
                rate = source["measurement"]["items_per_second"]
                model_digest = source["model"]["sha256"]
            except (KeyError, TypeError) as exc:
                raise ValueError("benchmark result lacks throughput or model digest") from exc
            assumptions = dict(
                items_per_second=rate, device_price=args.device_price, lifespan_years=args.lifespan_years,
                maintenance_per_year=args.maintenance_per_year, idle_watts=args.idle_watts,
                load_watts=args.load_watts, electricity_price_per_kwh=args.electricity_price,
                utilization=args.utilization, useful_fraction=args.useful_fraction,
                currency=args.currency, cloud_price_per_million_calls=args.cloud_price_per_million,
                cloud_useful_fraction=args.cloud_useful_fraction,
            )
            calculation = estimate_total_cost(**assumptions)
            utilization_values = _float_values(args.sensitivity_utilizations)
            prices = (_float_values(args.sensitivity_prices) if args.sensitivity_prices is not None
                      else [args.electricity_price * mult for mult in (0.5, 1.0, 1.5)])
            report = {"schema_version": 1, "report_type": "economics_tco",
                      "source_result": args.result, "model_sha256": model_digest,
                      "economics": calculation, "sensitivity": sensitivity_grid(assumptions, utilization_values, prices)}
            rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.out:
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
