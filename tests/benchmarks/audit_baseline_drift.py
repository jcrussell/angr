#!/usr/bin/env python3
"""Audit ``baseline_timings.json`` for drift in BOTH directions.

A baseline that is too **tight** announces itself — the gate fails and someone
fixes it the same day. A baseline that is too **loose** is silent: the bench
just passes, and the slack hides real regressions.  ``whitehatvn2015_re400``
sat at 2.2x loose for ~150 commits (angr-018m9) because the speedup that
earned it (``68d7ad34a``) never tightened the number.

This script runs the regression suite N times, takes the **max** observed
``rust_time`` per baseline key, and reports ``max / baseline``:

* ``< --loose-threshold`` (default 0.85) — LOOSE, the gate is effectively off.
  ``--apply`` lowers ``rust_time`` to the max-of-N for these.
* ``> --tight-threshold`` (default 1.05) — TIGHT, little headroom left below
  the 15% gate; reported only, never auto-edited (a tight baseline may be a
  genuine regression, and silently raising it would paper over the bug).
* in between — noise, left alone.

max-of-N rather than median is deliberate: it leaves the gate's threshold as
headroom above the *slowest* observed run, so the gate stays sensitive without
flapping.

Usage::

    python tests/benchmarks/audit_baseline_drift.py                 # report only
    python tests/benchmarks/audit_baseline_drift.py --reps 7 --apply
    python tests/benchmarks/audit_baseline_drift.py --json

Notes
-----
Each rep is a **subprocess** running ``run_regression.py``.  Do not import and
call ``run_regression.run_one`` from an ad-hoc harness: it uses a
multiprocessing *spawn* pool, so a module without an ``if __name__ ==
"__main__"`` guard fork-bombs the box (bd memory
``avoid-unguarded-run-one-harness-forkbomb``).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
BASELINE_PATH = BENCH_DIR / "baseline_timings.json"
RUN_REGRESSION = BENCH_DIR / "run_regression.py"

# run_regression prints one header per bench and one "  Rust:   X.XXs" line
# under it.  The DFS twin's header is "--- <name> (DFS) ---" and maps to the
# "<name>__dfs" baseline key -- a naive r'^--- (\S+) ---' parse silently drops
# it and mis-attributes its timing to the previous bench (bd memory
# ``bench-baseline-dfs-variant-key``).
_HEADER_RE = re.compile(r"^--- (.+?) ---$")
_RUST_RE = re.compile(r"^\s*Rust:\s+([0-9.]+)s")


def parse_run(text):
    """Map baseline key -> rust_time for one ``run_regression.py`` run."""
    timings = {}
    current = None
    for line in text.splitlines():
        header = _HEADER_RE.match(line.rstrip())
        if header:
            name = header.group(1)
            if name.endswith(" (DFS)"):
                name = name[: -len(" (DFS)")] + "__dfs"
            current = name
            continue
        rust = _RUST_RE.match(line.rstrip())
        if rust and current is not None:
            timings[current] = float(rust.group(1))
            current = None
    return timings


def run_suite(full, extra_args, verbose):
    """Run the suite once in a subprocess; return its parsed timings."""
    cmd = [sys.executable, str(RUN_REGRESSION), "--rust-only", "--skip-bimodal"]
    if full:
        cmd.append("--full")
    cmd.extend(extra_args)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if verbose:
        print(f"  (exit {proc.returncode})", file=sys.stderr)
    # A non-zero exit means some bench tripped the gate.  Its timing line is
    # still printed and still valid evidence, so parse the output either way.
    return parse_run(proc.stdout + proc.stderr)


def classify(observations, baseline, loose_threshold, tight_threshold):
    """Build one report row per measured key, sorted by ratio ascending."""
    rows = []
    for key, values in observations.items():
        entry = baseline.get(key)
        if entry is None or entry.get("rust_time") is None:
            rows.append({"key": key, "verdict": "unbaselined", "observed": values})
            continue
        base = entry["rust_time"]
        peak = max(values)
        ratio = peak / base
        if ratio < loose_threshold:
            verdict = "loose"
        elif ratio > tight_threshold:
            verdict = "tight"
        else:
            verdict = "ok"
        rows.append(
            {
                "key": key,
                "verdict": verdict,
                "baseline": base,
                "max": peak,
                "min": min(values),
                "ratio": ratio,
                "reps": len(values),
                "observed": values,
            }
        )
    rows.sort(key=lambda r: r.get("ratio", float("inf")))
    return rows


def apply_updates(rows, dry_run=False):
    """Lower rust_time to max-of-N for every LOOSE row. Returns updated keys."""
    loose = [r for r in rows if r["verdict"] == "loose"]
    if not loose or dry_run:
        return [r["key"] for r in loose]
    # sort_keys=False: the file on disk is insertion-ordered, and re-sorting it
    # would bury a one-line change in a whole-file reorder diff (bd memory
    # ``bench-add-baseline-files-insertion-order``).
    baseline = json.loads(BASELINE_PATH.read_text())
    for row in loose:
        baseline[row["key"]]["rust_time"] = row["max"]
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2, sort_keys=False) + "\n")
    return [r["key"] for r in loose]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=6, help="suite runs to sample (default 6; >=6 recommended)")
    parser.add_argument("--full", action="store_true", default=True, help="run fast+medium tier (default)")
    parser.add_argument("--fast-only", dest="full", action="store_false", help="fast tier only")
    parser.add_argument("--loose-threshold", type=float, default=0.85)
    parser.add_argument("--tight-threshold", type=float, default=1.05)
    parser.add_argument("--apply", action="store_true", help="lower rust_time for LOOSE entries")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        dest="extra_args",
        help="extra flag forwarded to run_regression.py (repeatable)",
    )
    args = parser.parse_args()

    observations = defaultdict(list)
    for rep in range(1, args.reps + 1):
        if not args.json:
            print(f"rep {rep}/{args.reps} ...", file=sys.stderr)
        for key, value in run_suite(args.full, args.extra_args, verbose=not args.json).items():
            observations[key].append(value)

    if not observations:
        print("ERROR: no timings parsed -- did run_regression.py fail to start?", file=sys.stderr)
        return 2

    baseline = json.loads(BASELINE_PATH.read_text())
    rows = classify(observations, baseline, args.loose_threshold, args.tight_threshold)
    updated = apply_updates(rows, dry_run=not args.apply)

    if args.json:
        print(json.dumps({"reps": args.reps, "rows": rows, "updated": updated}, indent=2))
        return 0

    print(f"\n{'ratio':>6} {'verdict':<12} {'bench':<40} {'baseline':>9} {'max':>7} {'min':>7}")
    for row in rows:
        if row["verdict"] == "unbaselined":
            print(f"{'--':>6} {'UNBASELINED':<12} {row['key']:<40}")
            continue
        print(
            f"{row['ratio']:6.2f} {row['verdict'].upper():<12} {row['key']:<40} "
            f"{row['baseline']:9.3f} {row['max']:7.2f} {row['min']:7.2f}"
        )

    loose = [r for r in rows if r["verdict"] == "loose"]
    tight = [r for r in rows if r["verdict"] == "tight"]
    print(f"\n{len(rows)} benches over {args.reps} reps: {len(loose)} loose, {len(tight)} tight")
    if loose and not args.apply:
        print(f"re-run with --apply to lower: {', '.join(r['key'] for r in loose)}")
    elif updated:
        print(f"updated {BASELINE_PATH.name}: {', '.join(updated)}")
    if tight:
        print("TIGHT entries are reported only -- investigate for a real regression before raising a baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
