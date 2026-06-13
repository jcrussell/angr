#!/usr/bin/env python3
"""Measure bimodal Z3 variance for known-bimodal benchmarks.

Drives ``run_single.py --engine rust`` as a subprocess N times per
benchmark and prints a histogram of the elapsed wall-clock timings.

Usage:
    python tests/benchmarks/bimodal_variance.py
    python tests/benchmarks/bimodal_variance.py --runs 20
    python tests/benchmarks/bimodal_variance.py --benchmarks google2016_unbreakable_1
    python tests/benchmarks/bimodal_variance.py --runs 5 --json out.json

By default it runs the three bimodal benchmarks tracked in
``invariant-bimodal-variance-benchmarks``:
``ekopartyctf2016_sokohashv2``, ``securityfest_fairlight``,
``google2016_unbreakable_1``.

run_single.py is invoked as a subprocess so each run inherits its
4 GB RLIMIT_AS — safe on 8 GB / 0-swap hosts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

DEFAULT_BENCHMARKS = [
    "ekopartyctf2016_sokohashv2",
    "securityfest_fairlight",
    "google2016_unbreakable_1",
]

RUN_SINGLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_single.py")
# Match either OK or FAIL — we want timing data even when the example's
# in-script assertion blows due to Z3 picking a different valid model.
ELAPSED_RE = re.compile(r"^(OK|FAIL) rust \S+ ([0-9.]+)s", re.MULTILINE)
FAIL_LINE_RE = re.compile(r"^FAIL rust", re.MULTILINE)


def run_once(example: str, timeout: int = 120) -> tuple[float, str] | None:
    """Return (elapsed_seconds, status) where status is 'ok' or 'fail',
    or None on TIMEOUT/subprocess crash."""
    cmd = [sys.executable, RUN_SINGLE, example, "--engine", "rust", "--timeout", str(timeout)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30)
    except subprocess.TimeoutExpired:
        return None
    m = ELAPSED_RE.search(proc.stdout)
    if not m:
        return None
    return float(m.group(2)), m.group(1).lower()


def histogram(values: list[float], bin_width: float = 1.0) -> list[tuple[float, float, int]]:
    """Return [(lo, hi, count)] bins. Bins are half-open [lo, hi)."""
    if not values:
        return []
    lo = int(min(values))
    hi = int(max(values)) + 1
    bins: dict[int, int] = {}
    for v in values:
        idx = int((v - lo) / bin_width)
        bins[idx] = bins.get(idx, 0) + 1
    out = []
    for idx in sorted(bins):
        bin_lo = lo + idx * bin_width
        out.append((bin_lo, bin_lo + bin_width, bins[idx]))
    return out


def format_histogram(values: list[float], bin_width: float) -> str:
    bins = histogram(values, bin_width)
    if not bins:
        return "(no data)"
    max_count = max(c for _, _, c in bins)
    lines = []
    for lo, hi, c in bins:
        bar = "#" * int(40 * c / max_count) if max_count else ""
        lines.append(f"  {lo:5.1f}–{hi:5.1f}s | {c:3d} {bar}")
    return "\n".join(lines)


def summarise(label: str, values: list[float]) -> dict:
    if not values:
        return {"benchmark": label, "n": 0}
    return {
        "benchmark": label,
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=20, help="Runs per benchmark (default 20)")
    ap.add_argument(
        "--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS, help="Benchmark names (default: 3 bimodal benches)"
    )
    ap.add_argument("--timeout", type=int, default=120, help="Per-run timeout in seconds")
    ap.add_argument("--bin-width", type=float, default=1.0, help="Histogram bin width in seconds")
    ap.add_argument("--json", dest="json_path", default=None, help="Write raw JSON output to PATH")
    args = ap.parse_args()

    print(f"=== bimodal variance: {len(args.benchmarks)} benchmark(s) × {args.runs} runs ===")
    all_summaries = []
    overall_start = time.perf_counter()
    for bench in args.benchmarks:
        print(f"\n--- {bench} ---")
        values: list[float] = []
        fail_count = 0
        for i in range(args.runs):
            res = run_once(bench, timeout=args.timeout)
            if res is None:
                print(f"  run {i + 1}/{args.runs}: TIMEOUT")
                continue
            t, status = res
            values.append(t)
            if status == "fail":
                fail_count += 1
            tag = "OK  " if status == "ok" else "FAIL"
            print(f"  run {i + 1}/{args.runs}: {tag} {t:.2f}s")
        summary = summarise(bench, values)
        summary["fail_count"] = fail_count
        all_summaries.append(summary)
        if values:
            print(
                f"\n  summary: n={summary['n']} (fail={fail_count}) "
                f"min={summary['min']:.2f}s median={summary['median']:.2f}s "
                f"max={summary['max']:.2f}s mean={summary['mean']:.2f}s "
                f"stdev={summary['stdev']:.2f}s"
            )
            print("\n  histogram:")
            print(format_histogram(values, args.bin_width))

    elapsed = time.perf_counter() - overall_start
    print(f"\n=== total wall-clock: {elapsed:.1f}s ===")

    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump({"runs": args.runs, "summaries": all_summaries}, f, indent=2)
        print(f"wrote {args.json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
