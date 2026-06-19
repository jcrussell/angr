#!/usr/bin/env python3
"""Solver-instrumentation showcase demo for the Rust engine (angr-4n26m.9).

The "look inside the solver" angle. While an exploration runs, the Rust engine
maintains process-wide atomic counters around every ``solver.check()`` call
(``native/angr/src/symbolic/context.rs``), broken down **per call-site** and
including **lazy-fork materialization** counts. Stock Python angr has no
equivalent: you can time the whole ``explore()`` but you cannot cheaply ask
"how much of that was Z3, and *which* query site drove it?".

This script runs a real CTF crackme search under ``RustExplorationManager``,
brackets it with ``reset_solver_stats()`` / ``get_solver_stats()``, and prints:

* the headline Z3 budget — total ``check()`` calls and wall-time, and what
  fraction of the whole search that was (the sokohashv2 writeup attributes
  ~8.7s of a 10s run to Z3; this is the same attribution, live);
* a **per-call-site** table (``satisfiable`` / ``branch_true`` /
  ``branch_false`` / ``eval`` / ``min_*`` / ``max_*`` …) sorted by time, so you
  can see whether branch feasibility, model extraction, or range search
  dominated;
* **lazy-fork** materialization counts (``z3_materialize_count`` /
  ``_time_ns``) — solvers cloned on demand instead of eagerly per fork;
* the branch fast-path counters (concrete-assume short-circuits and cached
  branch-model hits) that keep ``check()`` off the hot path.

Every number is measured live in this process; nothing is quoted from
``baseline_timings.json`` (stale by design). The search result is verified
(``Code_Talkers…``) so the attribution is for a *correct* run, not a misfire.

Usage::

    # Default target (defcamp_r100), human-readable report.
    python tests/benchmarks/show_solver_instrumentation.py

    # Machine-readable; refresh the durable artifact.
    python tests/benchmarks/show_solver_instrumentation.py --json > \
        tests/benchmarks/solver_instrumentation_numbers.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import resource
import sys
import time

EXAMPLES_DIR = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser("~/repos/angr-examples/examples")

# Target registry. Each entry is a self-contained find/avoid search whose
# solution we can verify, so the solver attribution is for a correct run.
# defcamp_r100 is a fast (sub-5s under Python) crackme with real Z3 activity:
# a symbolic stdin password threaded through a comparison loop.
TARGETS = {
    "defcamp_r100": {
        "binary": os.path.join(EXAMPLES_DIR, "defcamp_r100", "r100"),
        "find": 0x400844,
        "avoid": 0x400855,
        "expect_prefix": b"Code_Talkers",
    },
}

# Per-call-site labels, in the order they're worth reading. Sites absent from a
# given run (count 0) are dropped from the table.
SITE_ORDER = [
    "satisfiable",
    "branch_true",
    "branch_false",
    "eval",
    "eval_upto",
    "min_init",
    "min_search",
    "max_init",
    "max_search",
]


def run_instrumented(target_name: str) -> dict:
    """Run the search under the Rust engine, bracketed by solver-stat resets.

    Returns a dict with the verified solution plus the full post-run stats
    snapshot (overall, per-site, lazy-fork, branch fast-path).
    """
    import angr
    from angr.exploration import RustExplorationManager

    spec = TARGETS[target_name]
    project = angr.Project(spec["binary"], load_options={"auto_load_libs": False})
    state = project.factory.full_init_state()
    mgr = RustExplorationManager(project, [state])

    # Zero the global counters immediately before the measured window so the
    # attribution covers only this exploration, not setup/CFG work.
    mgr.reset_solver_stats()
    t0 = time.time()
    mgr.explore(find=spec["find"], avoid=spec["avoid"])
    elapsed = time.time() - t0
    stats = mgr.get_solver_stats()

    found = mgr.found
    solution = None
    correct = False
    if found:
        raw = found[0].posix.dumps(0).strip(b"\x00\n")
        solution = raw
        correct = raw.startswith(spec["expect_prefix"])

    return {
        "target": target_name,
        "binary": spec["binary"],
        "found": bool(found),
        "solution": solution.decode("latin-1") if solution is not None else None,
        "correct": correct,
        "elapsed": round(elapsed, 3),
        "stats": stats,
    }


def _sites_from_stats(stats: dict) -> list[dict]:
    """Extract the per-call-site breakdown as a list of dicts, time desc."""
    rows = []
    for name in SITE_ORDER:
        count = stats.get(f"z3_site_{name}_count", 0)
        if not count:
            continue
        time_ns = stats.get(f"z3_site_{name}_time_ns", 0)
        rows.append({"site": name, "count": count, "time_ns": time_ns})
    rows.sort(key=lambda r: r["time_ns"], reverse=True)
    return rows


def _fmt_ms(ns: int) -> str:
    return f"{ns / 1e6:.1f}ms"


def _print_report(res: dict) -> None:
    stats = res["stats"]
    if not stats:
        print("error: Rust engine reported no solver stats (built without Z3?).")
        return

    check_count = stats.get("z3_check_count", 0)
    check_ns = stats.get("z3_check_time_ns", 0)
    elapsed_ns = res["elapsed"] * 1e9
    z3_frac = (check_ns / elapsed_ns) if elapsed_ns else 0.0

    print("=" * 72)
    print("Rust-engine solver instrumentation — look inside the Z3 budget")
    print("=" * 72)
    print(f"  target        : {res['target']}  ({res['binary']})")
    status = "FOUND" if res["found"] else "no path"
    verdict = "verified" if res["correct"] else "UNVERIFIED"
    print(f"  search result : {status}  ({verdict})")
    if res["solution"] is not None:
        print(f"  solution      : {res['solution']!r}")
    print(f"  search time   : {res['elapsed']:.3f}s")
    print("-" * 72)
    print("  Z3 budget (this is what stock Python angr can't cheaply see):")
    print(f"    solver.check() calls : {check_count}")
    print(f"    time in check()      : {_fmt_ms(check_ns)}  ({z3_frac * 100:.0f}% of the whole search)")
    print(
        f"    sat / unsat / timeout: "
        f"{stats.get('z3_sat_count', 0)} / "
        f"{stats.get('z3_unsat_count', 0)} / "
        f"{stats.get('z3_timeout_count', 0)}"
    )
    print(f"    AST nodes built      : {stats.get('z3_ast_build', 0)}")
    print("-" * 72)

    sites = _sites_from_stats(stats)
    if sites:
        print("  Per-call-site attribution (where the solver time went):")
        print(f"    {'site':<14}{'calls':>8}{'time':>12}{'% of check':>13}")
        for row in sites:
            pct = (row["time_ns"] / check_ns * 100) if check_ns else 0.0
            print(f"    {row['site']:<14}{row['count']:>8}{_fmt_ms(row['time_ns']):>12}{pct:>12.0f}%")
        print("-" * 72)

    mat_count = stats.get("z3_materialize_count", 0)
    mat_ns = stats.get("z3_materialize_time_ns", 0)
    print("  Lazy-fork materialization (solvers cloned on demand, not per-fork):")
    print(f"    materializations     : {mat_count}  ({_fmt_ms(mat_ns)})")
    print("-" * 72)
    print("  Branch fast-paths (queries the engine avoided sending to Z3):")
    print(f"    concrete-assume      : {stats.get('z3_assume_concrete', 0)}")
    print(f"    symbolic-assume      : {stats.get('z3_assume_symbolic', 0)}")
    print(f"    branch model hit/miss: {stats.get('z3_branch_model_hit', 0)} / {stats.get('z3_branch_model_miss', 0)}")
    print("=" * 72)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--target",
        default="defcamp_r100",
        choices=sorted(TARGETS),
        help="which instrumented search to run (default: defcamp_r100)",
    )
    args = parser.parse_args(argv)

    # Self-cap address space at 3 GB: this runs angr in-process and the loop
    # box has no swap. The target is a small crackme, so this is a backstop.
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))

    spec = TARGETS[args.target]
    if not os.path.exists(spec["binary"]):
        sys.stderr.write(
            f"error: target binary not found at {spec['binary']}\n"
            "set ANGR_EXAMPLES_DIR to your angr-examples/examples checkout.\n"
        )
        return 2

    res = run_instrumented(args.target)

    if args.json:
        json.dump(res, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        _print_report(res)

    return 0 if (res["found"] and res["correct"]) else 1


if __name__ == "__main__":
    sys.exit(main())
