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
    return parser


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
                intra_op_threads=args.intra_op_threads,
            )
            rendered = json.dumps(report, indent=2, sort_keys=True)
        else:
            from .report import load_results, render_comparison
            rendered = render_comparison(load_results(args.results), args.format)
        if args.out:
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
