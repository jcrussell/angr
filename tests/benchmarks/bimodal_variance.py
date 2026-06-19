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


def classify(values: list[float]) -> dict:
    """Label a timing sample as STABLE / BIMODAL / OUTLIER.

    Heuristic: sort the samples and find the largest gap between
    consecutive values. That gap splits the sample into a low and a
    high cluster.

    - BIMODAL: the gap is wide (>= ``GAP_FRAC`` of the median) AND the
      two clusters are well separated (high_min / low_max >= ``SEP``)
      AND each cluster has >= 2 members (two genuine modes, not a lone
      flier).
    - OUTLIER: the split is wide/separated but one side holds a single
      sample — noisy, not a second mode.
    - STABLE: everything else.

    Returns a dict with the label plus the supporting numbers so the
    JSON consumer can re-derive the decision.
    """
    GAP_FRAC = 0.20  # largest gap must exceed 20% of the median
    SEP = 1.30  # high cluster must be >= 1.3x the low cluster boundary
    n = len(values)
    if n < 3:
        return {"label": "STABLE", "reason": "n<3", "cv": 0.0}
    s = sorted(values)
    median = statistics.median(s)
    mean = statistics.fmean(s)
    stdev = statistics.stdev(s)
    cv = stdev / mean if mean else 0.0
    # largest consecutive gap
    gaps = [(s[i + 1] - s[i], i) for i in range(n - 1)]
    gap, idx = max(gaps)
    low = s[: idx + 1]
    high = s[idx + 1 :]
    sep = (high[0] / low[-1]) if low[-1] else float("inf")
    wide = gap >= GAP_FRAC * median and median > 0
    separated = sep >= SEP
    label = "STABLE"
    if wide and separated:
        label = "BIMODAL" if len(low) >= 2 and len(high) >= 2 else "OUTLIER"
    return {
        "label": label,
        "cv": cv,
        "gap": gap,
        "gap_frac": (gap / median) if median else 0.0,
        "sep": sep,
        "low_n": len(low),
        "high_n": len(high),
        "low_max": low[-1],
        "high_min": high[0],
    }


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
        "classification": classify(values),
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
            cls = summary["classification"]
            print(
                f"\n  summary: n={summary['n']} (fail={fail_count}) "
                f"min={summary['min']:.2f}s median={summary['median']:.2f}s "
                f"max={summary['max']:.2f}s mean={summary['mean']:.2f}s "
                f"stdev={summary['stdev']:.2f}s"
            )
            print(
                f"  classification: {cls['label']} "
                f"(cv={cls['cv']:.2f} gap_frac={cls.get('gap_frac', 0):.2f} sep={cls.get('sep', 0):.2f})"
            )
            print("\n  histogram:")
            print(format_histogram(values, args.bin_width))

    elapsed = time.perf_counter() - overall_start

    print("\n=== classification table ===")
    print(f"  {'benchmark':40s} {'label':8s} {'median':>8s} {'cv':>5s} {'gap':>5s} {'sep':>5s}")
    for s in all_summaries:
        if not s.get("n"):
            print(f"  {s['benchmark']:40s} {'NODATA':8s}")
            continue
        c = s["classification"]
        print(
            f"  {s['benchmark']:40s} {c['label']:8s} {s['median']:7.2f}s "
            f"{c['cv']:5.2f} {c.get('gap_frac', 0):5.2f} {c.get('sep', 0):5.2f}"
        )

    print(f"\n=== total wall-clock: {elapsed:.1f}s ===")

    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump({"runs": args.runs, "summaries": all_summaries}, f, indent=2)
        print(f"wrote {args.json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
