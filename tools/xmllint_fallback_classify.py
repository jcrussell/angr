#!/usr/bin/env python
"""gorvf.3.1 measure-first: per-PC A/B classification of the 27 xmllint_getenv
SimProcedure->Python fallbacks.

Bucket A = is_in_binary-gate MISROUTE: a native Rust proc EXISTS in the
registry but the hook PC sits inside a loaded object's executable region
(libc .text), so run_loop/stepping route it to Python (native never consulted).
Bucket B = genuinely ADDS_EXITS-dependent: the SimProc adds successor exits
(sub-calls) and needs the native dispatcher sub-call machinery.

RESULT (gorvf.3.1, 2026-07-05): 27/27 fallbacks are Bucket A. Every one of the
8 distinct hook PCs (strcmp/pthread_once/pthread_mutex_{lock,unlock}/malloc/
calloc/getenv/time) resolves inside libc.so.6's executable region, and a native
proc exists for all 8 -> the is_in_binary gate is the sole root cause. ZERO are
Bucket B: none of the hooked angr SimProcedures set ADDS_EXITS (pthread_once is
modelled as a leaf). Implication: .4c (native dispatcher ADDS_EXITS/sub-call
wiring) is UNFUNDED by this bench; all 27 are governed by .4b (the gate flip),
whose net win was already measured <0.1% (a8epx-gate-measured-defer: the
dominant strcmp x10 runs on symbolic stdin -> native strcmp bails symbolic
anyway; only the 13 concrete mutex/malloc/calloc/time/getenv calls could
succeed natively, the ~3ms upside already recorded).

Runs under a 4 GB RLIMIT_AS in this subprocess (safe on the 6G loop box).
Reproduce: ANGR_EXAMPLES_DIR=... python tools/xmllint_fallback_classify.py
"""

from __future__ import annotations

import os
import resource
import sys

resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))


import angr

# The 8 distinct fallback symbols + counts from
# collect_simproc_fallbacks.py --benches xmllint_getenv (total 27).
FALLBACKS = {
    "strcmp": 10,
    "pthread_once": 4,
    "pthread_mutex_lock": 4,
    "pthread_mutex_unlock": 3,
    "malloc": 3,
    "time": 1,
    "getenv": 1,
    "calloc": 1,
}

# Native procs confirmed present in native/angr/src/procedures/ registry.
NATIVE_PROCS = {
    "strcmp",
    "pthread_once",
    "pthread_mutex_lock",
    "pthread_mutex_unlock",
    "malloc",
    "calloc",
    "getenv",
    "time",
}


def _binary_path():
    base = os.environ.get("ANGR_EXAMPLES_DIR") or os.path.expanduser("~/repos/angr-examples/examples")
    return os.path.join(base, "xmllint", "xmllint_bin")


def main():
    target = _binary_path()
    proj = angr.Project(target, auto_load_libs=True, use_sim_procedures=True)

    # Reproduce _load_binary_regions membership: executable sections (fallback
    # segments) of every non-cle loaded object.
    main_object = proj.loader.main_object
    regions = []  # (min, max_inclusive, owner_binary)
    for obj in proj.loader.all_objects:
        if obj.binary is None and obj is not main_object:
            continue
        if isinstance(obj.binary, str) and obj.binary.startswith("cle##"):
            continue
        exec_ranges = [s for s in obj.sections if s.is_executable]
        if not exec_ranges:
            exec_ranges = [s for s in obj.segments if s.is_executable]
        for r in exec_ranges:
            regions.append((r.min_addr, r.max_addr, os.path.basename(str(obj.binary))))

    def region_of(addr):
        for lo, hi, name in regions:
            if lo <= addr <= hi:
                return name
        return None

    # Map each symbol -> the address angr will actually execute the hook at.
    # With use_sim_procedures the libc symbol is hooked at its rebased addr.
    print(f"{'symbol':>22} {'count':>5} {'hook_addr':>12} {'in_binary':>9} {'native':>6} {'ADDS_EXITS':>10} {'owner'}")
    total_a = total_b = total_other = 0
    rows = []
    for name, count in FALLBACKS.items():
        sym = proj.loader.find_symbol(name)
        addr = sym.rebased_addr if sym is not None else None
        owner = region_of(addr) if addr is not None else None
        in_binary = owner is not None
        has_native = name in NATIVE_PROCS

        adds_exits = None
        hooked_proc = None
        if addr is not None and proj.is_hooked(addr):
            hooked_proc = proj.hooked_by(addr)
            adds_exits = bool(getattr(hooked_proc, "ADDS_EXITS", False))

        # Classification.
        if adds_exits:
            bucket = "B"
            total_b += count
        elif has_native and in_binary:
            bucket = "A"
            total_a += count
        else:
            bucket = "?"
            total_other += count

        rows.append(
            (
                name,
                count,
                addr,
                in_binary,
                has_native,
                adds_exits,
                owner,
                bucket,
                type(hooked_proc).__name__ if hooked_proc else None,
            )
        )
        print(
            f"{name:>22} {count:>5} {addr if addr is None else hex(addr):>12} "
            f"{in_binary!s:>9} {has_native!s:>6} {adds_exits!s:>10} {owner}"
        )

    print()
    print("=== PER-PC A/B TABLE (each libc symbol = one hook PC, hit `count` times) ===")
    print(f"{'bucket':>6} {'symbol':>22} {'count':>5} {'hook_addr':>12} {'simproc':>28}")
    for name, count, addr, _in_binary, _has_native, _adds_exits, _owner, bucket, procname in rows:
        print(f"{bucket:>6} {name:>22} {count:>5} {addr if addr is None else hex(addr):>12} {procname!s:>28}")

    print()
    print(f"Bucket A (is_in_binary MISROUTE, native exists): {total_a} fallbacks")
    print(f"Bucket B (ADDS_EXITS-dependent):                 {total_b} fallbacks")
    print(f"Bucket ? (neither):                              {total_other} fallbacks")
    print(f"Total:                                           {total_a + total_b + total_other}")


if __name__ == "__main__":
    sys.exit(main())
