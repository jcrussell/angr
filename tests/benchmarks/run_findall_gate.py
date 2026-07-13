#!/usr/bin/env python3
"""Exhaustive find-all bench driver with per-worker structural counters
(bd angr-op0dn.13.2 — pre-GO measurement harness for spike angr-op0dn.7 / M5).

Sibling of ``run_steal_fraction_gate.py`` / ``run_width_audit.py``: it drives the
EXISTING ``run_single.py`` child harness (4GB ``RLIMIT_AS``, Rust-manager
monkeypatch, ``--counters-json``) rather than standing up a second runner.

What it measures. Each bench is a find-all workload — ``explore(find=[addr],
num_find=k_large)``, i.e. run-to-exhaustion, which is the ONLY shape the parallel
loops can pay off on (a ``num_find=1`` first-find on a wide frontier is
anti-parallel; bd ``parallel-numfind1-speculative-waste``). Every bench is swept
over worker counts (default {1,2,4}) x reps (default 3 — bimodal discipline per
bd ``benchmark-bimodal-variance-rules`` / ``parallel-cliff-was-bimodal-not-superlinear``:
one run of a Z3-bound bench is noise, so every wall number below is a MEDIAN and
carries a min/max spread). The steady-state loop is engaged through the existing
plumbing (``RUST_PARALLEL_STEADY=1`` + >=2 workers => ``rust_manager.py``
auto-calls ``_set_frontier_residency(True)`` in its run path).

The GATE is structural, not wall-clock (wall-clock acceptance lives on the spike,
angr-op0dn.7): the found-set fingerprint — the multiset of found pcs, exported by
``run_single.py`` under ``ANGR_BENCH_FOUND_FINGERPRINT=1`` — must be IDENTICAL
across every worker count and rep. That is the same projection the 13.1 parity
test uses; the vh834 AST content fingerprint is deliberately NOT used because a
Rust found state exports a near-empty Python-side constraint list, so set equality
over it passes vacuously (bd ``avoid-content-fingerprint-on-found-states``).

Reported per (bench, workers): parallel_tasks, parallel_migrations, steal fraction
(migrations/tasks), parallel_width_hist + parallel_max_active_width,
parallel_reattaches, parallel_residual_drains, parallel_bounce_roundtrips,
gil_work_time_ns / run_wall_time_ns, and the found-set fingerprint.

Known gap: there is no per-worker dispatch counter in the Rust engine today
(``stats_api.rs`` exposes only aggregate ``parallel_tasks`` / ``parallel_migrations``),
so the "states/worker balance (max/min dispatched per worker)" column is reported
as ``n/a`` rather than faked. Adding it is bd angr-op0dn.13.9.

Usage:
    python tests/benchmarks/run_findall_gate.py                        # default corpus
    python tests/benchmarks/run_findall_gate.py --only fauxware --reps 1
    python tests/benchmarks/run_findall_gate.py --workers 1 2 --json findall.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_SINGLE = HERE / "run_single.py"

OK_RE = re.compile(r"^OK rust \S+ ([0-9.]+)s", re.MULTILINE)

# Default find-all corpus. Each entry pins the num_find override (via the bench's
# own FORK_SOLVE_NUM_FIND env knob) and the leaf count the drain must reach.
#
# fork_solve_pbounce_W6_S8_M12_B2 — partial-bounce wide synthetic: 2^6 = 64
# leaves, all reachable, num_find=64 => a true exhaustive drain with a Python
# bounce at 4 interleaved levels (the steady loop's demonstrator workload).
# fork_solve_trap_W5_S8_M12 — full-bounce sibling: 2^5 = 32 leaves, every leaf
# bounces, so it stresses the migration/reattach path instead.
#
# There is deliberately no real-binary leg yet: the only cheap real find-all
# candidate (fauxware) cannot be driven through run_single.py under
# RUST_PARALLEL_WORKERS>=2 — its solve.py indexes found[] assuming a serial
# result shape and dies with IndexError (harness bug, NOT a scheduler bug; bd
# memory avoid-run-single-parallel-fauxware-false-alarm). Add the real leg when
# angr-75mc lands its bench.
DEFAULT_BENCHES = ["fork_solve_pbounce_W6_S8_M12_B2", "fork_solve_trap_W5_S8_M12"]

# Per-bench env overrides applied on top of the sweep's own vars. Each pins the
# bench's own num_find knob to its exhaustive value (2^W leaves).
BENCH_ENV = {
    "fork_solve_pbounce_W6_S8_M12_B2": {"FORK_SOLVE_NUM_FIND": "64"},
    "fork_solve_trap_W5_S8_M12": {"FORK_SOLVE_NUM_FIND": "32"},
    "fork_solve_W6_S8": {"FORK_SOLVE_NUM_FIND": "64"},
}


def _trailing_json(out: str) -> dict | None:
    brace = out.rfind("\n{")
    if brace == -1:
        if out.startswith("{"):
            brace = 0
        else:
            return None
    try:
        return json.loads(out[brace:])
    except json.JSONDecodeError:
        return None


def run_one(bench: str, workers: int, timeout: int, steady: bool) -> dict | None:
    """One find-all run through run_single.py. Returns None on timeout/crash."""
    env = dict(os.environ)
    env.setdefault("ANGR_EXAMPLES_DIR", os.path.expanduser("~/repos/angr-examples/examples"))
    env["ANGR_BENCH_FOUND_FINGERPRINT"] = "1"
    env["RUST_PARALLEL_WORKERS"] = str(workers)
    if steady and workers > 1:
        env["RUST_PARALLEL_STEADY"] = "1"
    else:
        env.pop("RUST_PARALLEL_STEADY", None)
    env.update(BENCH_ENV.get(bench, {}))

    cmd = [
        sys.executable,
        str(RUN_SINGLE),
        bench,
        "--engine",
        "rust",
        "--counters-json",
        "--timeout",
        str(timeout),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60, env=env)
    except subprocess.TimeoutExpired:
        return None
    stats = _trailing_json(proc.stdout)
    if stats is None:
        return None
    m = OK_RE.search(proc.stdout)
    if m is None:
        # run_single prints `FAIL rust <bench> ...` and still emits partial
        # counters when the bench raises. A crashed run must not silently
        # contribute an empty found-set to the gate.
        return None
    tasks = stats.get("parallel_tasks", 0)
    run_wall_ns = stats.get("run_wall_time_ns", 0)
    return {
        "wall_s": float(m.group(1)),
        "found": stats.get("found", 0),
        "found_pcs": stats.get("found_pcs", ""),
        "steps": stats.get("steps", 0),
        "tasks": tasks,
        "migrations": stats.get("parallel_migrations", 0),
        "steal_fraction": (stats.get("parallel_migrations", 0) / tasks) if tasks else 0.0,
        "reattaches": stats.get("parallel_reattaches", 0),
        "residual_drains": stats.get("parallel_residual_drains", 0),
        "bounce_roundtrips": stats.get("parallel_bounce_roundtrips", 0),
        "resume_reinjects": stats.get("parallel_resume_reinjects", 0),
        "width_hist": stats.get("parallel_width_hist", []),
        "max_active_width": stats.get("parallel_max_active_width", 0),
        "real_workers": stats.get("parallel_real_workers", 1),
        "gil_work_time_ns": stats.get("gil_work_time_ns", 0),
        "run_wall_time_ns": run_wall_ns,
        "gil_fraction": (stats.get("gil_work_time_ns", 0) / run_wall_ns) if run_wall_ns else 0.0,
        # No per-worker dispatch counter exists yet (angr-op0dn.13.9).
        "per_worker_dispatch": None,
    }


def _median_i(reps: list[dict], key: str) -> int:
    return int(statistics.median([r[key] for r in reps]))


def aggregate(bench: str, workers: int, reps: list[dict]) -> dict:
    walls = [r["wall_s"] for r in reps if r["wall_s"] is not None]
    fingerprints = sorted({r["found_pcs"] for r in reps})
    return {
        "bench": bench,
        "workers": workers,
        "reps": len(reps),
        "real_workers": reps[0]["real_workers"],
        "wall_median_s": statistics.median(walls) if walls else None,
        "wall_min_s": min(walls) if walls else None,
        "wall_max_s": max(walls) if walls else None,
        "found": _median_i(reps, "found"),
        "found_pcs": fingerprints[0] if len(fingerprints) == 1 else fingerprints,
        "fingerprint_stable_across_reps": len(fingerprints) == 1,
        "steps": _median_i(reps, "steps"),
        "tasks": _median_i(reps, "tasks"),
        "migrations": _median_i(reps, "migrations"),
        "steal_fraction": statistics.median([r["steal_fraction"] for r in reps]),
        "reattaches": _median_i(reps, "reattaches"),
        "residual_drains": _median_i(reps, "residual_drains"),
        "bounce_roundtrips": _median_i(reps, "bounce_roundtrips"),
        "resume_reinjects": _median_i(reps, "resume_reinjects"),
        "width_hist": reps[0]["width_hist"],
        "max_active_width": max(r["max_active_width"] for r in reps),
        "gil_fraction": statistics.median([r["gil_fraction"] for r in reps]),
        "per_worker_dispatch": None,
        "raw_reps": reps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help=f"Benches to drive (default: {' '.join(DEFAULT_BENCHES)})")
    ap.add_argument("--workers", nargs="*", type=int, default=[1, 2, 4], help="Worker-count sweep (default 1 2 4)")
    ap.add_argument("--reps", type=int, default=3, help="Reps per (bench, workers) — median + spread (default 3)")
    ap.add_argument("--timeout", type=int, default=300, help="Per-run timeout seconds (default 300)")
    ap.add_argument("--no-steady", action="store_true", help="Drive the wave loop instead of the steady loop")
    ap.add_argument("--json", type=str, help="Write the full per-(bench,workers) artifact here")
    args = ap.parse_args()

    benches = args.only or DEFAULT_BENCHES
    steady = not args.no_steady
    rows: list[dict] = []
    failures: list[str] = []

    for bench in benches:
        for workers in args.workers:
            reps: list[dict] = []
            for rep in range(args.reps):
                print(f"[run] {bench} workers={workers} rep={rep + 1}/{args.reps} ...", file=sys.stderr, flush=True)
                r = run_one(bench, workers, args.timeout, steady)
                if r is None:
                    failures.append(f"{bench} workers={workers} rep={rep + 1}: timeout/crash/no-json")
                    continue
                reps.append(r)
            if reps:
                rows.append(aggregate(bench, workers, reps))

    hdr = (
        f"{'bench':<34} {'w':>2} {'wall_med':>9} {'spread':>13} {'found':>5} {'tasks':>6} "
        f"{'migr':>5} {'steal%':>7} {'reatt':>6} {'resid':>6} {'gil%':>6} {'peak_w':>6} {'bal':>4}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        wall = f"{r['wall_median_s']:.2f}s" if r["wall_median_s"] is not None else "-"
        spread = (
            f"{r['wall_min_s']:.2f}-{r['wall_max_s']:.2f}"
            if r["wall_min_s"] is not None and r["wall_max_s"] is not None
            else "-"
        )
        print(
            f"{r['bench']:<34} {r['workers']:>2} {wall:>9} {spread:>13} {r['found']:>5} {r['tasks']:>6} "
            f"{r['migrations']:>5} {100 * r['steal_fraction']:>6.2f}% {r['reattaches']:>6} "
            f"{r['residual_drains']:>6} {100 * r['gil_fraction']:>5.2f}% {r['max_active_width']:>6} {'n/a':>4}"
        )

    # --- the gate: found-set fingerprint identical across worker counts + reps ---
    print()
    verdict_ok = True
    for bench in benches:
        bench_rows = [r for r in rows if r["bench"] == bench]
        if not bench_rows:
            print(f"FAIL {bench}: no successful runs")
            verdict_ok = False
            continue
        empty = [r for r in bench_rows if not r["found_pcs"]]
        if empty:
            # A find-all bench that found nothing makes the fingerprint compare
            # pass vacuously — that is a harness failure, not a PASS.
            verdict_ok = False
            print(f"FAIL {bench}: empty found-set at workers {[r['workers'] for r in empty]} (not a find-all run?)")
            continue
        unstable = [r for r in bench_rows if not r["fingerprint_stable_across_reps"]]
        prints = {r["workers"]: r["found_pcs"] for r in bench_rows}
        distinct = {json.dumps(p, sort_keys=True) for p in prints.values()}
        if unstable or len(distinct) != 1:
            verdict_ok = False
            print(f"FAIL {bench}: found-set fingerprint differs across workers/reps:")
            for w, p in sorted(prints.items()):
                print(f"    workers={w}: {p}")
        else:
            only = next(iter(prints.values()))
            print(f"PASS {bench}: found-set identical across workers {sorted(prints)} x {args.reps} reps -> {only}")

    for f in failures:
        print(f"[fail] {f}", file=sys.stderr)
    if failures:
        verdict_ok = False

    print()
    print(
        "VERDICT: PASS — find-all is worker-count-invariant on this corpus."
        if verdict_ok
        else "VERDICT: FAIL — see the mismatches above (structural gate; wall-clock lives on angr-op0dn.7)."
    )

    if args.json:
        Path(args.json).write_text(json.dumps({"steady": steady, "reps": args.reps, "rows": rows}, indent=2))
        print(f"[wrote] {args.json}", file=sys.stderr)
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
