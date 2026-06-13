#!/usr/bin/env python3
"""One-off backfill script for python_time entries in baseline_timings.json.

For each baseline entry with python_time == null, runs the Python engine
in an isolated subprocess (4GB RLIMIT_AS) with a generous timeout, then
writes back the elapsed time. Entries that timeout, OOM, or crash get
python_time=null with a python_skip_reason field documenting why.

Usage:
    python tests/benchmarks/backfill_python_time.py
    python tests/benchmarks/backfill_python_time.py --timeout 120
    python tests/benchmarks/backfill_python_time.py --only fauxware
    python tests/benchmarks/backfill_python_time.py --force          # re-run even if already populated
    python tests/benchmarks/backfill_python_time.py --dry-run        # don't write the JSON
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from run_regression import load_baseline, run_one, save_baseline

# Default per-entry timeout (seconds). Some Python runs take 30+s.
DEFAULT_TIMEOUT = 90

# Per-entry timeout overrides for examples where 90s is too short.
TIMEOUT_OVERRIDES = {
    "flareon2015_5": 120,
    "ekopartyctf2016_rev250": 120,
    "csaw_wyvern": 120,
    "ekopartyctf2016_sokohashv2": 120,
    "hackcon2016_angry-reverser": 180,
    "securityfest_fairlight": 120,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"Default per-entry timeout (default: {DEFAULT_TIMEOUT}s)"
    )
    parser.add_argument("--mem-limit", type=int, default=4096)
    parser.add_argument("--only", help="Only run a single named benchmark")
    parser.add_argument("--force", action="store_true", help="Re-run even if python_time is already populated")
    parser.add_argument("--dry-run", action="store_true", help="Don't save updates to baseline_timings.json")
    args = parser.parse_args()

    baseline = load_baseline()
    targets = []
    for key in sorted(baseline.keys()):
        entry = baseline[key]
        if args.only and key != args.only:
            continue
        if entry.get("python_time") is not None and not args.force:
            continue
        targets.append(key)

    if not targets:
        print("Nothing to do.")
        return 0

    print(f"Backfilling {len(targets)} entries: {', '.join(targets)}")
    print()

    updated = 0
    skipped = 0
    for key in targets:
        # Strategy is encoded in the key for __dfs variants; the angr-example
        # name is the part before "__dfs" if present.
        if key.endswith("__dfs"):
            example_name = key[: -len("__dfs")]
            strategy = "dfs"
        else:
            example_name = key
            strategy = "bfs"

        timeout = TIMEOUT_OVERRIDES.get(example_name, args.timeout)
        print(f"--- {key} (timeout={timeout}s, strategy={strategy}) ---")
        result = run_one(example_name, "python", timeout, args.mem_limit, strategy)

        if result.get("ok"):
            elapsed = result["elapsed"]
            baseline[key]["python_time"] = round(elapsed, 3)
            # Clear any prior skip reason if the run now succeeds.
            baseline[key].pop("python_skip_reason", None)
            print(f"  python_time = {elapsed:.3f}s")
            updated += 1
        else:
            err = result.get("error", "unknown error")
            baseline[key]["python_time"] = None
            baseline[key]["python_skip_reason"] = err
            print(f"  SKIP: {err}")
            skipped += 1

        # Save incrementally so a crash doesn't lose prior progress.
        if not args.dry_run:
            save_baseline(baseline)

    print()
    print(f"Done. Updated: {updated}, skipped: {skipped}, total: {len(targets)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
