#!/usr/bin/env python3
"""Attribute the SimProcedure park-and-bounce on the ZeroPy bounce-sole benches.

angr-gorvf.12.  ``tests/benchmarks/zeropy_attribution.json`` says seven FAIL
benches have ``bounce`` as their *only* nonzero ``gil_by_class`` entry, so
retiring their SimProcedure bounce flips each to ``gil_work_time_ns == 0`` (a
ZeroPy PASS).  Before any of that can be funded we need to know *which* proc
bounces, at what PC, and why — instrumented, not inferred from the binary.

This harness runs each bench in its own subprocess (``run_single.py``, 4 GB
``RLIMIT_AS``) with ``ANGR_BOUNCE_TRACE=1``, which makes
``rust_callback_dispatch._handle_simprocedure_callback`` emit one
``[bounce] name=... addr=... cls=... module=...`` line per crossing.  It joins
those lines against the Rust-side ``native_proc_*_fallbacks_by_name`` counters
from the same run and classifies every distinct (name, PC) into one bucket:

  A — no native proc: nothing in the Rust registry claims this symbol.  A
      native SimProcedure has to be written.  Fundable.
  B — native proc declined: the registry *has* a handler but it bailed to
      Python (symbolic args / unimplemented / other).  Extending the arg policy
      is the fix.  Fundable, and cheaper than A.
  C — irreducible: a user Python hook (the bench's own ``proj.hook``, a CTF
      ``my_scanf``/``get_flag``), which per ``rust-python-boundary-audit`` must
      stay in Python.  Such a bench can never reach ``gil == 0`` and has to
      leave the gate denominator rather than be "fixed".

The bounce set does not depend on the lift path, so this runs on the *stock*
build — no libvex-ffi ``.so`` needed (unlike ``run_zeropy_gate.py``).

Usage:
    python tests/benchmarks/zeropy_bounce_census.py
    python tests/benchmarks/zeropy_bounce_census.py --benches fauxware
    python tests/benchmarks/zeropy_bounce_census.py --out /tmp/census.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from run_zeropy_gate import _trailing_json, split_baseline_key

RUN_SINGLE = os.path.join(_HERE, "run_single.py")
OUT_JSON = os.path.join(_HERE, "zeropy_bounce_census.json")

#: The bounce-sole FAIL benches from zeropy_attribution.json (iter142): `bounce`
#: is the only nonzero gil_by_class entry, so the SimProcedure crossing is the
#: sole thing standing between each of these and a ZeroPy PASS.
BOUNCE_SOLE_BENCHES = [
    "cmu_binary_bomb_partial",
    "defcon2016quals_baby-re",
    "ekopartyctf2016_rev250",
    "fauxware",
    "google2016_unbreakable_0",
    "securityfest_fairlight",
    "unmapped_analysis",
]

BOUNCE_RE = re.compile(r"^\[bounce\] name=(?P<name>\S+) addr=(?P<addr>\S+) cls=(?P<cls>\S+) module=(?P<module>\S+)")

BUCKET_DESC = {
    "A": "no native proc — write one",
    "B": "native proc exists but did not serve the call — extend arg policy / dispatch",
    "C": "irreducible user Python hook — bench leaves the gate denominator",
}

_PROCS_DIR = os.path.join(os.path.dirname(_HERE), os.pardir, "native", "angr", "src", "procedures")
_NAME_ATTR_RE = re.compile(r'^\s*name = "([^"]+)"')


def native_registry_names() -> set[str]:
    """Symbols the Rust native-procedure registry claims.

    Read out of the ``name = "..."`` attribute on each ``#[native_procedure]``
    in ``native/angr/src/procedures/*.rs`` (test modules excluded). Needed
    because a bounce whose proc IS natively implemented is a *dispatch* gap
    (bucket B), not a missing-proc gap (bucket A) — and the runtime
    ``native_proc_*_fallbacks_by_name`` counters cannot tell them apart: a proc
    hooked at a binary-internal address takes the hook path, where the native
    registry is never consulted at all, so no decline counter ever ticks.
    """
    names: set[str] = set()
    for fname in sorted(os.listdir(_PROCS_DIR)):
        if not fname.endswith(".rs") or fname.endswith("_tests.rs"):
            continue
        with open(os.path.join(_PROCS_DIR, fname)) as f:
            for line in f:
                m = _NAME_ATTR_RE.match(line)
                if m:
                    names.add(m.group(1))
    return names


def _bucket(site: dict, declines: dict[str, str], native: set[str], registry_consulted: bool) -> tuple[str, str]:
    """Classify one bounce site into A/B/C plus a human-readable reason."""
    name, module, cls = site["name"], site["module"], site["cls"]
    if name == "__internal_passthrough__":
        # Not a SimProcedure at all — Rust handing back internal binary code.
        return "B", "internal passthrough (no proc; Rust resumes at addr)"
    if not module.startswith("angr."):
        return "C", f"user Python hook defined in {module}"
    if name in declines:
        return "B", f"native proc exists; declined at runtime ({declines[name]})"
    if name in native:
        why = "hook path bypassed the native registry" if not registry_consulted else "native proc never dispatched"
        return "B", f"native proc exists; {why}"
    if cls == "ReturnUnconstrained":
        # A SimLibrary stub for an unresolved symbol — the "proc" is just
        # "return a fresh symbol". One native stub handler retires every one of
        # these, so it is much cheaper than a real bucket-A proc.
        return "A", "angr stub (ReturnUnconstrained) — needs a native stub handler, not a real proc"
    return "A", "no native handler claimed this symbol"


def run_one(bench: str, timeout: int, native: set[str]) -> dict:
    example, strategy = split_baseline_key(bench)
    env = dict(os.environ)
    env.setdefault("ANGR_EXAMPLES_DIR", os.path.expanduser("~/repos/angr-examples/examples"))
    env["ANGR_BOUNCE_TRACE"] = "1"
    cmd = [
        sys.executable,
        RUN_SINGLE,
        example,
        "--engine",
        "rust",
        "--counters-json",
        "--timeout",
        str(timeout),
        "--strategy",
        strategy,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60, env=env)
    except subprocess.TimeoutExpired:
        return {"bench": bench, "status": "ERROR", "reason": f"timeout (>{timeout + 60}s)", "bounces": []}

    stats = _trailing_json(proc.stdout) or {}

    # Why each native handler bailed, keyed by proc name. A name can show up in
    # more than one counter (different call sites decline differently) — keep
    # every reason so the table does not lie by picking one.
    declines: dict[str, str] = {}
    for reason, key in (
        ("symbolic args", "native_proc_symbolic_fallbacks_by_name"),
        ("not implemented", "native_proc_not_implemented_fallbacks_by_name"),
        ("other", "native_proc_other_fallbacks_by_name"),
    ):
        for name, count in (stats.get(key) or {}).items():
            prior = declines.get(name)
            entry = f"{reason} x{count}"
            declines[name] = f"{prior}, {entry}" if prior else entry

    bounces: list[dict] = []
    for line in proc.stderr.splitlines():
        m = BOUNCE_RE.match(line)
        if m:
            bounces.append(m.groupdict())

    # Collapse to distinct (name, PC) — the gate cares that a crossing site is
    # retired, not how many times it fired.
    sites: dict[tuple[str, str], dict] = {}
    for b in bounces:
        key = (b["name"], b["addr"])
        site = sites.setdefault(
            key,
            {
                "name": b["name"],
                "addr": b["addr"],
                "cls": b["cls"],
                "module": b["module"],
                "count": 0,
            },
        )
        site["count"] += 1
    native_proc_calls = stats.get("native_proc_calls", 0)
    for site in sites.values():
        site["bucket"], site["reason"] = _bucket(site, declines, native, native_proc_calls > 0)

    return {
        "bench": bench,
        "status": "OK" if stats else "ERROR",
        "reason": "" if stats else "run failed or emitted no counters",
        "bounce_count": len(bounces),
        "native_proc_calls": native_proc_calls,
        "simprocedure_callback_count": stats.get("callback_simprocedure_count", 0),
        "simprocedure_total_ns": stats.get("callback_simprocedure_total_ns", 0),
        "sites": sorted(sites.values(), key=lambda s: (s["bucket"], s["name"])),
    }


def render(rows: list[dict]) -> None:
    print()
    print("| bench | proc | PC | bucket | why |")
    print("|---|---|---|---|---|")
    counts: dict[str, int] = {"A": 0, "B": 0, "C": 0}
    for r in rows:
        if r["status"] != "OK":
            print(f"| {r['bench']} | — | — | ERROR | {r['reason']} |")
            continue
        if not r["sites"]:
            print(f"| {r['bench']} | (none) | — | — | no bounce observed |")
            continue
        for s in r["sites"]:
            counts[s["bucket"]] += 1
            print(f"| {r['bench']} | {s['name']} (x{s['count']}) | {s['addr']} | {s['bucket']} | {s['reason']} |")
    print()
    for b in ("A", "B", "C"):
        print(f"  {b} = {counts[b]:2d} sites — {BUCKET_DESC[b]}")
    bench_bucket = {}
    for r in rows:
        if r["status"] == "OK" and r["sites"]:
            buckets = {s["bucket"] for s in r["sites"]}
            # A bench is only reducible if EVERY one of its sites is reducible.
            bench_bucket[r["bench"]] = "C" if "C" in buckets else ("A" if "A" in buckets else "B")
    reducible = [b for b, k in bench_bucket.items() if k != "C"]
    irreducible = [b for b, k in bench_bucket.items() if k == "C"]
    print()
    print(f"Benches whose bounces are ALL reducible (A/B): {len(reducible)} — {', '.join(reducible) or '—'}")
    print(f"Benches with an irreducible (C) bounce:        {len(irreducible)} — {', '.join(irreducible) or '—'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", nargs="*", default=BOUNCE_SOLE_BENCHES)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()

    native = native_registry_names()
    print(f"native registry: {len(native)} symbols")
    rows = []
    for bench in args.benches:
        print(f"--- {bench}", flush=True)
        row = run_one(bench, args.timeout, native)
        print(f"    {row['status']}: {row.get('bounce_count', 0)} bounces, {len(row.get('sites', []))} sites")
        rows.append(row)

    render(rows)
    with open(args.out, "w") as f:
        json.dump({"benches": rows}, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
