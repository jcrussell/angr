#!/usr/bin/env python3
"""Number gate for the Rust-engine showcase (angr-4n26m.3).

Every number destined for the showcase blog post must come from a FRESH
measurement with a reported median + range — never quoted verbatim from
``baseline_timings.json`` (stale by design). This harness runs
``run_single.py --both`` N>=5 times per target, parses the ``OK <engine>``
timing lines, and reports per-engine median + [min, max] plus the median
speedup. Bimodal-Z3 benches are flagged so their numbers carry the
mandatory caveat (their wall-clock is multi-modal and a median understates
the spread — see run_regression.BIMODAL_BENCHMARKS and
docs/advanced-topics/rust_bimodal_variance.rst).

Usage::

    # Default curated showcase set, N=5
    python tests/benchmarks/measure_showcase.py

    # Custom targets / repeat count, machine-readable JSON
    python tests/benchmarks/measure_showcase.py --targets fauxware ais3_crackme -n 7
    python tests/benchmarks/measure_showcase.py --json > /tmp/numbers.json

Each ``--both`` invocation is a subprocess with its own RLIMIT_AS (run_single
caps memory), so this harness is OOM-safe to run in the ralph loop on the
curated fast set. Heavy targets (python_time > ~10s) are intentionally NOT in
the default set; pass them explicitly with a smaller -n when needed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys

# Bimodal-Z3 benches whose wall-clock is multi-modal: a median is reported but
# MUST carry a caveat in the post. Mirrors run_regression.BIMODAL_BENCHMARKS
# (plus CADET_00001_partial, which the showcase brief lists explicitly).
BIMODAL = {
    "sokohashv2",
    "fairlight",
    "angry-reverser",
    "hackcon2016_angry-reverser",
    "unbreakable_1",
    "google2016_unbreakable_1",
    "CADET_00001_partial",
}

# Curated default set: bounded targets (python_time < ~5s each) that are
# candidates for the raw-speed / breadth demos. None are bimodal. Heavy
# winners (ekopartyctf2016_rev250 ~15x but py~32s, flareon2015_5 ~10x but
# py~57s) are excluded from the default to keep the gate OOM- and time-safe;
# measure them on demand with an explicit --targets and small -n.
DEFAULT_TARGETS = [
    "fauxware",
    "defcamp_r100",
    "ais3_crackme",
    "sharif7_rev50",
    "busybox_static",
]

_OK_RE = re.compile(r"^OK\s+(python|rust)\s+\S+\s+([0-9]+\.[0-9]+)s")


def _run_once(target: str, timeout: int) -> dict[str, float] | None:
    """Run ``run_single.py --both`` once; return {'python': t, 'rust': t}.

    Returns None (and prints a diagnostic) if either engine line is missing
    or the subprocess fails — a partial run must not silently skew the median.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable,
        os.path.join(here, "run_single.py"),
        target,
        "--both",
        "--timeout",
        str(timeout),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 60,
        )
    except subprocess.TimeoutExpired:
        print(f"  ! {target}: harness timeout", file=sys.stderr)
        return None

    times: dict[str, float] = {}
    for line in proc.stdout.splitlines():
        m = _OK_RE.match(line)
        if m:
            times[m.group(1)] = float(m.group(2))
    if "python" not in times or "rust" not in times:
        print(
            f"  ! {target}: missing engine line (got {sorted(times)}); exit={proc.returncode}",
            file=sys.stderr,
        )
        return None
    return times


def measure(target: str, n: int, timeout: int) -> dict | None:
    """Collect n samples for one target; return a stats dict or None."""
    py: list[float] = []
    rs: list[float] = []
    for i in range(n):
        res = _run_once(target, timeout)
        if res is None:
            continue
        py.append(res["python"])
        rs.append(res["rust"])
        print(
            f"  {target} run {i + 1}/{n}: py={res['python']:.2f}s rust={res['rust']:.2f}s",
            file=sys.stderr,
        )
    if not py or not rs:
        return None
    py_med = statistics.median(py)
    rs_med = statistics.median(rs)
    return {
        "target": target,
        "bimodal": target in BIMODAL,
        "samples": len(py),
        "python": {"median": py_med, "min": min(py), "max": max(py), "raw": py},
        "rust": {"median": rs_med, "min": min(rs), "max": max(rs), "raw": rs},
        # Median-of-medians speedup; >1 means Rust is faster.
        "speedup_median": (py_med / rs_med) if rs_med else None,
    }


def _fmt_range(s: dict) -> str:
    return f"{s['median']:.2f}s [{s['min']:.2f}-{s['max']:.2f}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("-n", "--repeats", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--json", action="store_true", help="Emit JSON only")
    args = ap.parse_args()

    if args.repeats < 5 and not args.json:
        print(
            f"WARNING: n={args.repeats} < 5 — the number gate requires N>=5 for any value quoted in the post.",
            file=sys.stderr,
        )

    results = []
    for t in args.targets:
        print(f"=== measuring {t} (n={args.repeats}) ===", file=sys.stderr)
        r = measure(t, args.repeats, args.timeout)
        if r is not None:
            results.append(r)

    if args.json:
        json.dump({"repeats": args.repeats, "results": results}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0 if results else 1

    print()
    print(f"{'target':<22} {'python (med [min-max])':<26} {'rust':<26} {'speedup':<8} note")
    print("-" * 96)
    for r in results:
        note = "BIMODAL — caveat required" if r["bimodal"] else ""
        sp = f"{r['speedup_median']:.2f}x" if r["speedup_median"] else "n/a"
        print(f"{r['target']:<22} {_fmt_range(r['python']):<26} {_fmt_range(r['rust']):<26} {sp:<8} {note}")
    print()
    print(
        "Numbers are fresh medians over N>="
        f"{args.repeats} --both runs; ranges are [min, max]. Do NOT quote "
        "baseline_timings.json. Bimodal benches: report the range, not a point."
    )
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
