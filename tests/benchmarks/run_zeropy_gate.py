#!/usr/bin/env python3
"""Zero-Python (ZeroPy) milestone gate — bd angr-gorvf.4.

Measures, per corpus bench, whether the Rust run loop ever touches Python after
setup, and — when it does — *which* callback class is responsible.

The metric is ``gil_work_time_ns`` (``native/angr/src/gil_profile.rs``): a
thread-local accumulator that only banks time while a profiled ``RunLoopWallGuard``
is live, i.e. strictly inside ``run()``.  Setup (project load, state creation,
hook installation) happens outside that window and is *by construction* excluded,
so "post-setup" needs no extra bookkeeping — ``gil_work_time_ns == 0`` **is** the
zero-bounce condition.

**A zero is only a pass when the run loop was actually measured.**  A bench whose
counter dict has no ``run_wall_time_ns`` (or reports 0) never entered a profiled
run loop on the stats-reading thread, so its ``gil_work_time_ns`` reads 0 for the
uninteresting reason.  Those are reported ``UNMEASURED``, never ``PASS`` — reading
a missing key as 0 is exactly the trap that made a stale ``baseline_counters.json``
look like it had 7 passing benches when it had none.

Residual-bounce attribution.  For every non-passing bench the gate prints the
``callback_<class>_count`` breakdown, which names the Python surface still being
crossed.  The classes and what retires them:

* ``lift_block``  — pyvex VEX lifting.  Only the ``libvex-ffi`` cargo feature
  (native cold-block lifting, bd ``libvex-stage3-default-on``) removes these, and
  only on AMD64.  A stock build cannot reach zero on any bench that lifts.
* ``posix``       — the Python SimPosix plugin.  Retired by RustPosixState
  ownership of fds/env/cwd (the scope-trimmed part of bd angr-6zxx).
* ``simprocedure``— a Python SimProcedure declined by the native registry.
* ``find_predicate`` / ``avoid_predicate`` — user Python callables passed to
  ``explore()``; irreducible by design (bd ``rust-python-boundary-audit``).
* ``memory_load`` / ``fetch_page`` — Python-backed page fetches.
* ``symbolic_branch`` / ``vex_fallback`` — CADET-class hard fallbacks.

Usage:
    python tests/benchmarks/run_zeropy_gate.py                   # whole corpus
    python tests/benchmarks/run_zeropy_gate.py --only fauxware ais3_crackme
    python tests/benchmarks/run_zeropy_gate.py --json zeropy.json --min-pass 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_SINGLE = HERE / "run_single.py"
BASELINE_TIMINGS = HERE / "baseline_timings.json"

OK_RE = re.compile(r"^OK rust \S+ ([0-9.]+)s", re.MULTILINE)

# Bounce classes, in the order they are reported. Anything not listed here still
# shows up in the attribution dict — the list only pins the column order for the
# classes we have a retirement story for.
CLASS_ORDER = [
    "lift_block",
    "posix",
    "simprocedure",
    "syscall",
    "memory_load",
    "fetch_page",
    "find_predicate",
    "avoid_predicate",
    "symbolic_branch",
    "vex_fallback",
]


# gil_profile::GilClass, spelled out rather than prefix-matched: the per-site
# keys (`gil_work_ns_callback_*`) share the `gil_work_ns_` prefix, and folding
# them into the class split would double-count the callback class.
GIL_CLASSES = ["callback", "claripy_export", "claripy_import", "fork_metadata"]


def corpus() -> list[str]:
    with BASELINE_TIMINGS.open() as fh:
        return sorted(json.load(fh))


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


def attribution(stats: dict) -> dict[str, dict[str, int]]:
    """Per-class ``{count, ns}`` for every callback class the bench crossed.

    Ranked by ``ns`` descending, because **count and GIL cost do not agree**.
    The ``callback_*`` counters are process-wide: they also tick for callbacks
    made during setup, state export, and predicate evaluation *outside* the
    profiled run-loop window, none of which ``gil_work_time_ns`` sees.  fauxware
    on a libvex-ffi build reports ``posix=5`` and still scores a hard zero — those
    five calls all land outside the loop.  Conversely a single in-loop ``posix``
    callback can cost hundreds of milliseconds.  Read the ns column to rank
    levers; the count column only says the surface was touched at all.
    """
    found: dict[str, dict[str, int]] = {}
    for key, val in stats.items():
        # `callback_count` is the aggregate, not a class — it would otherwise
        # match the prefix/suffix test with an empty class name.
        if key == "callback_count" or not val:
            continue
        if key.startswith("callback_") and key.endswith("_count"):
            found.setdefault(key[len("callback_") : -len("_count")], {})["count"] = val
        elif key.startswith("callback_") and key.endswith("_total_ns"):
            found.setdefault(key[len("callback_") : -len("_total_ns")], {})["ns"] = val
    for slot in found.values():
        slot.setdefault("count", 0)
        slot.setdefault("ns", 0)
    return dict(sorted(found.items(), key=lambda kv: (-kv[1]["ns"], kv[0])))


def run_one(bench: str, timeout: int) -> dict:
    env = dict(os.environ)
    env.setdefault("ANGR_EXAMPLES_DIR", os.path.expanduser("~/repos/angr-examples/examples"))
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
        return {"bench": bench, "status": "ERROR", "reason": "timeout"}
    stats = _trailing_json(proc.stdout)
    if stats is None or OK_RE.search(proc.stdout) is None:
        return {"bench": bench, "status": "ERROR", "reason": "run failed or emitted no counters"}

    # A missing key is NOT a zero: it means the profiled run loop never ran on
    # the thread stats() was read from. Distinguish that from a real zero.
    wall_ns = stats.get("run_wall_time_ns")
    gil_ns = stats.get("gil_work_time_ns")
    row = {
        "bench": bench,
        "wall_s": float(OK_RE.search(proc.stdout).group(1)),
        "found": stats.get("found", 0),
        "gil_work_time_ns": gil_ns,
        "run_wall_time_ns": wall_ns,
        # The partition of `gil_work_time_ns` by *why* the GIL was taken
        # (gil_profile::GilClass). This is the lever ranking; `gil_by_site`
        # below sub-partitions the `callback` class.
        "gil_by_class": {c: stats[f"gil_work_ns_{c}"] for c in GIL_CLASSES if stats.get(f"gil_work_ns_{c}")},
        # The `callback` class again, split by which dispatch entry point took
        # the GIL (gil_profile::CallbackSite). Sums exactly to the `callback`
        # class, so a residual here is a real unattributed site, not a counter
        # artifact — unlike `attribution`.
        "gil_by_site": {
            k[len("gil_work_ns_callback_") :]: v
            for k, v in stats.items()
            if k.startswith("gil_work_ns_callback_") and v
        },
        "attribution": attribution(stats),
    }
    if not wall_ns:
        row["status"] = "UNMEASURED"
        row["reason"] = "no profiled run-loop window (run_wall_time_ns missing/zero)"
    elif gil_ns:
        row["status"] = "FAIL"
        row["gil_fraction"] = gil_ns / wall_ns
    else:
        row["status"] = "PASS"
        row["gil_fraction"] = 0.0
    return row


def render(rows: list[dict], min_pass: int) -> bool:
    order = {"PASS": 0, "FAIL": 1, "UNMEASURED": 2, "ERROR": 3}
    rows = sorted(rows, key=lambda r: (order[r["status"]], -(r.get("gil_work_time_ns") or 0)))
    print(f"\n{'bench':<32} {'status':<11} {'gil_ns':>13} {'gil%':>7}  residual bounces")
    print("-" * 110)
    for r in rows:
        if r["status"] == "ERROR":
            print(f"{r['bench']:<32} {'ERROR':<11} {'-':>13} {'-':>7}  {r['reason']}")
            continue
        gil = r["gil_work_time_ns"] or 0
        frac = f"{r.get('gil_fraction', 0):.1%}" if r["status"] != "UNMEASURED" else "-"
        split = ", ".join(f"{k}={v / 1e6:.0f}ms" for k, v in sorted(r["gil_by_class"].items(), key=lambda kv: -kv[1]))
        print(f"{r['bench']:<32} {r['status']:<11} {gil:>13,} {frac:>7}  {split or '(none)'}")

    passes = [r for r in rows if r["status"] == "PASS"]

    # Lever ranking. `ns` is the GIL time each class actually accounts for across
    # the FAIL set — retiring the top class buys the most. `sole` counts benches
    # where it is the ONLY class holding the GIL, i.e. benches that retiring it
    # alone flips to PASS.
    levers: dict[str, list[int]] = {}
    for r in rows:
        if r["status"] != "FAIL":
            continue
        for cls, ns in r["gil_by_class"].items():
            acc = levers.setdefault(cls, [0, 0, 0])
            acc[1] += 1
            acc[2] += ns
            if len(r["gil_by_class"]) == 1:
                acc[0] += 1
    print("\nGIL classes holding the loop (FAIL benches), ranked by GIL ns — the lever ranking:")
    for cls, (sole, appears, ns) in sorted(levers.items(), key=lambda kv: -kv[1][2]):
        print(f"  {cls:<18} sole={sole:<3} appears={appears:<3} {ns / 1e6:>9.0f}ms")

    # Sub-partition of the `callback` class by dispatch site (angr-gorvf.4.1).
    # These are thread-local and run-loop-gated, so they sum to the class and a
    # nonzero `other` is a genuinely unattributed Python-touching callback.
    sites: dict[str, list[int]] = {}
    unexplained: list[tuple[str, float]] = []
    for r in rows:
        if r["status"] != "FAIL":
            continue
        cb = r["gil_by_class"].get("callback", 0)
        if not cb:
            continue
        for site, ns in r["gil_by_site"].items():
            acc = sites.setdefault(site, [0, 0])
            acc[0] += 1
            acc[1] += ns
        named = sum(ns for s, ns in r["gil_by_site"].items() if s != "other")
        if named < 0.95 * cb:
            unexplained.append((r["bench"], 1 - named / cb))
    print("\ncallback GIL by dispatch site (FAIL benches) — sums to the callback class:")
    for site, (appears, ns) in sorted(sites.items(), key=lambda kv: -kv[1][1]):
        print(f"  {site:<28} appears={appears:<3} {ns / 1e6:>9.0f}ms")
    if unexplained:
        print("\n  UNEXPLAINED callback residual (>5% in `other`):")
        for bench, frac in sorted(unexplained, key=lambda kv: -kv[1]):
            print(f"    {bench:<32} {frac:.1%} unattributed")
    else:
        print("\n  every FAIL bench's callback class is >=95% attributed to a named site.")

    # Refinement of the `callback` slice only: which callback surfaces were
    # crossed. Counts here include out-of-loop calls (see `attribution`), so they
    # explain the callback class rather than rank it.
    surfaces: dict[str, list[int]] = {}
    for r in rows:
        if r["status"] != "FAIL":
            continue
        for cls, slot in r["attribution"].items():
            acc = surfaces.setdefault(cls, [0, 0])
            acc[0] += slot["count"]
            acc[1] += slot["ns"]
    print("\ncallback surfaces crossed (FAIL benches; counts include out-of-loop calls):")
    for cls, (count, ns) in sorted(surfaces.items(), key=lambda kv: -kv[1][1]):
        print(f"  {cls:<18} count={count:<5} {ns / 1e6:>9.0f}ms")

    ok = len(passes) >= min_pass
    print(f"\nZeroPy gate: {len(passes)}/{len(rows)} benches at gil_work_time_ns == 0 (need >= {min_pass})")
    print("PASS" if ok else "FAIL")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="BENCH", help="restrict to these benches")
    ap.add_argument("--min-pass", type=int, default=8, help="benches that must reach gil==0 (default 8)")
    ap.add_argument("--timeout", type=int, default=300, help="per-bench timeout in seconds")
    ap.add_argument("--json", metavar="PATH", help="write the full per-bench rows here")
    args = ap.parse_args()

    benches = args.only or corpus()
    rows = []
    for i, bench in enumerate(benches, 1):
        print(f"[{i}/{len(benches)}] {bench} ...", flush=True)
        rows.append(run_one(bench, args.timeout))

    ok = render(rows, args.min_pass)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
