#!/usr/bin/env python3
"""Run a curated bench list with the Rust engine and aggregate the
`simprocedure_fallback_by_name` counter across runs.

Used by the angr-otjw spike to rank which Python SimProcedures dominate
Rust->Python fallback frequency. Each bench runs in its own subprocess
(via run_single.run_example) under a 4GB RLIMIT_AS — safe on 8GB/no-swap.

Usage:
    python tests/benchmarks/collect_simproc_fallbacks.py
    python tests/benchmarks/collect_simproc_fallbacks.py --benches fauxware ais3_crackme
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from run_single import DEFAULT_MEM_LIMIT_MB, run_example

# Benches with callback_count > 0 from baseline_timings.json — these are the
# only ones that contribute to the fallback counter. Slow benches (sokohash,
# fairlight, baby-re) included since they may be dominated by hot Python
# procs; sokohash skipped because of bimodal Z3 variance > 15s/run.
CANDIDATE_BENCHES = [
    "fauxware",
    "ais3_crackme",
    "csaw_wyvern",
    "csgames2018",
    "defcon2016quals_baby-re",
    "ekopartyctf2016_rev250",
    "flareon2015_5",
    "flareon2015_10",
    "google2016_unbreakable_0",
    "mma_howtouse",
    "strcpy_find",
    "sym-write",
    "unmapped_analysis",
    "whitehatvn2015_re400",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benches", nargs="+", default=CANDIDATE_BENCHES)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--out", default=None, help="Write the aggregated JSON to this file (default: stdout only)")
    args = parser.parse_args()

    per_bench: dict[str, dict[str, int]] = {}
    per_bench_total: dict[str, int] = {}

    for name in args.benches:
        print(f"\n=== {name} ===", flush=True)
        result = run_example(name, "rust", timeout=args.timeout, mem_limit_mb=DEFAULT_MEM_LIMIT_MB)
        if not result or not result.get("ok"):
            print(f"  (skipped: {result.get('error') if result else 'no result'})")
            per_bench[name] = {}
            per_bench_total[name] = -1
            continue
        stats = result.get("stats") or {}
        by_name = stats.get("simprocedure_fallback_by_name") or {}
        total = stats.get("simprocedure_python_fallback_count", 0)
        per_bench[name] = dict(by_name)
        per_bench_total[name] = total
        if by_name:
            top = sorted(by_name.items(), key=lambda kv: -kv[1])
            print(f"  fallback_total={total}")
            for pname, pcount in top:
                print(f"    {pname}: {pcount}")
        else:
            print(f"  fallback_total={total} (no per-proc breakdown)")

    # Aggregate
    agg: dict[str, int] = collections.Counter()
    bench_appearances: dict[str, set] = collections.defaultdict(set)
    for name, by_name in per_bench.items():
        for pname, pcount in by_name.items():
            agg[pname] += pcount
            bench_appearances[pname].add(name)

    print("\n=== AGGREGATED RANKING ===")
    print(f"{'rank':>4}  {'count':>8}  {'benches':>8}  procedure")
    for rank, (pname, total) in enumerate(sorted(agg.items(), key=lambda kv: -kv[1]), 1):
        nbench = len(bench_appearances[pname])
        print(f"{rank:>4}  {total:>8}  {nbench:>8}  {pname}")

    out_payload = {
        "per_bench": per_bench,
        "per_bench_total": per_bench_total,
        "aggregated": dict(agg),
        "bench_appearances": {p: sorted(list(v)) for p, v in bench_appearances.items()},
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out_payload, f, indent=2, sort_keys=True)
        print(f"\nWrote JSON to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
