#!/usr/bin/env python3
"""Drive ``run_fleet.py`` across (engine, count, concurrency) points.

Writes a CSV row per (engine, count, concurrency) configuration to
``results.csv`` alongside this script. Each row carries the metrics
emitted by ``run_fleet.py --json``.

Default sweep:
    concurrency in {1, 2, 4, 8, 16}, count = concurrency * 2
    engines: rust, python

Adjust with --concurrencies and --count-multiplier as needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RUN_FLEET = os.path.join(HERE, "run_fleet.py")
CSV_PATH = os.path.join(HERE, "results.csv")

FIELDS = [
    "engine",
    "count",
    "concurrency",
    "example",
    "wall_s",
    "per_proc_mean_s",
    "per_proc_max_s",
    "peak_aggregate_rss_mb",
    "peak_per_proc_rss_mb",
    "sum_per_proc_peak_rss_mb",
    "peak_concurrency_observed",
    "failures",
]


def run_one(engine: str, count: int, concurrency: int, mem_limit_mb: int) -> dict:
    cmd = [
        sys.executable,
        RUN_FLEET,
        "--engine",
        engine,
        "--count",
        str(count),
        "--concurrency",
        str(concurrency),
        "--mem-limit-mb",
        str(mem_limit_mb),
        "--json",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    if proc.returncode != 0:
        print(
            f"[warn] run_fleet failed (engine={engine}, count={count}, "
            f"concurrency={concurrency}): rc={proc.returncode}",
            file=sys.stderr,
        )
        if proc.stderr:
            print(proc.stderr[-1000:], file=sys.stderr)
    # run_fleet writes the JSON blob on the last non-empty stdout line
    last = ""
    for line in proc.stdout.splitlines():
        if line.strip():
            last = line
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return {"engine": engine, "count": count, "concurrency": concurrency, "failures": -1, "error": "json-decode"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--concurrencies",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[1, 2, 4, 8, 16],
        help="Comma-separated concurrency levels (default: 1,2,4,8,16)",
    )
    p.add_argument(
        "--count-multiplier",
        type=int,
        default=2,
        help="For each concurrency K, count = K * multiplier (default: 2)",
    )
    p.add_argument(
        "--engines",
        nargs="+",
        choices=["rust", "python"],
        default=["rust", "python"],
    )
    p.add_argument("--mem-limit-mb", type=int, default=1024)
    p.add_argument("--csv", default=CSV_PATH)
    args = p.parse_args()

    rows = []
    for concurrency in args.concurrencies:
        count = concurrency * args.count_multiplier
        for engine in args.engines:
            print(f"[sweep] engine={engine} count={count} concurrency={concurrency} ...", flush=True)
            metrics = run_one(engine, count, concurrency, args.mem_limit_mb)
            rows.append(metrics)
            keep = {k: metrics.get(k) for k in FIELDS}
            print(f"  -> {keep}", flush=True)

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in FIELDS})
    print(f"[sweep] wrote {args.csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
