#!/usr/bin/env python3
"""Run a curated bench list with the Rust engine and aggregate every
Rust->Python *fallback* counter across runs, bucketed by name/number and
tagged intentional-vs-closable.

Originally the angr-otjw spike (rank which Python SimProcedures dominate
`simprocedure_fallback_by_name`). Extended for angr-aq26d to also bucket:

  * native_proc_{not_implemented,symbolic,other}_fallbacks_by_name
    (a native proc exists in the registry but bailed to Python — these are
    the *closable* native gaps, distinct from intentional Python procs)
  * syscall_python_fallback_by_num    (syscall gaps)
  * the scalar VEX-op fallback counters (rust_python_vex_*_fallback_count,
    vex_fallback_count, vex_fallback_unique_addrs)

Each fallback name is tagged `closable` (a libc-ish proc we could implement
or extend natively) vs `intentional` (angr-internal stub, C++ stdlib, or a
user CTF hook that will always run in Python). The tag drives the
re-decision in angr-aq26d: close confirmed recurring native gaps, or
conclude native coverage is not the bottleneck.

Each bench runs in its own subprocess (via run_single.run_example) under a
4GB RLIMIT_AS — safe on 8GB/no-swap.

Usage:
    python tests/benchmarks/collect_simproc_fallbacks.py
    python tests/benchmarks/collect_simproc_fallbacks.py --benches fauxware ais3_crackme
    python tests/benchmarks/collect_simproc_fallbacks.py --out /tmp/fallbacks.json
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
    # === CTF-heavy x86_64 corpus (original otjw/aq26d round-1 set) ===
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
    # === angr-hzd9e corpus-add: libc-I/O-heavy workloads ===
    # Stress the stdio write path (fwrite/fputc) and ctype/fprintf surfaces
    # that the CTF crackmes never exercise. write_stream_heavy + cli_ctype_fprintf
    # are in-repo synthetic fixtures (always present); busybox_static + xmllint_getenv
    # need external fixtures (/usr/bin/busybox, ANGR_EXAMPLES_DIR) and skip
    # gracefully when absent.
    "write_stream_heavy",
    "cli_ctype_fprintf",
    "busybox_static",
    "xmllint_getenv",
    # === angr-hzd9e corpus-add: non-x86 (ARM/AArch64/MIPS) workloads ===
    # Arch smoke fixtures — single-branch binaries that drive the VEX
    # interpreter on big-endian / non-x86 guests, surfacing arch-specific
    # VEX-op fallbacks the x86_64 corpus cannot reach.
    "arm_le_branch",
    "aarch64_le_branch",
    "mips32_le_branch",
    "mips64_le_branch",
    "mips64_be_branch",
    # === angr-amtxu corpus-add: FP/SIMD-heavy workload ===
    # The CTF crackmes + other synthetic fixtures are all integer/string-
    # bound, so vex_fallback_count / vecret_gsptr_fallback_count read ZERO
    # everywhere and the native VEX FP/SIMD interpreter path has no measured
    # coverage. fp_simd_kernel is an in-repo synthetic (vendored gcc -O3
    # binary, always present) whose concrete path drives SSE scalar FP
    # (sqrtsd/divsd/mulsd) + auto-vectorized packed SIMD (mulps/addps). A
    # non-zero VEX-fallback reading here re-opens native FP/SIMD handler work.
    "fp_simd_kernel",
]

# angr-internal SimProcedure stubs that are intentionally Python and have no
# native equivalent by design (control-flow stubs, unresolvable targets,
# returns-unconstrained shims). A fallback to one of these is NOT a closable
# native gap.
_INTENTIONAL_STUBS = {
    "ReturnUnconstrained",
    "UnresolvableJumpTarget",
    "UnresolvableCallTarget",
    "PathTerminator",
    "CallReturn",
    "UserHook",
    "Redirect",
    "Nop",
    "Unsupported",
    "stub",
    "abort",
    "__assert_fail",
    "__stack_chk_fail",
    "_exit",
    "exit",
    # C runtime init/fini shims angr models in Python on purpose.
    "__libc_start_main",
    "__libc_csu_init",
    "__libc_csu_fini",
    "_start",
}


def _is_cpp_mangled(name: str) -> bool:
    """Heuristic: C++ stdlib / mangled symbols are intentional (no native
    libc proc). Demangled forms carry `::`/`operator`; mangled forms start
    with `_Z`."""
    return name.startswith(("_Z", "std")) or "::" in name or "operator" in name


def classify_fallback(name: str) -> str:
    """Tag a SimProcedure-fallback name as 'intentional' or 'closable'.

    'closable' = a libc-ish proc that has (or could have) a native Rust
    implementation, so the fallback represents a coverage gap worth closing.
    'intentional' = an angr-internal stub, C++ stdlib symbol, or otherwise a
    proc that will always run in Python by design.
    """
    if name in _INTENTIONAL_STUBS:
        return "intentional"
    if _is_cpp_mangled(name):
        return "intentional"
    return "closable"


# Scalar VEX-op fallback counters — a non-zero value here means a VEX op ran
# in Python (closable VEX gap). Reported as a flat sum across benches.
_VEX_SCALAR_COUNTERS = [
    "rust_python_vex_op_fallback_count",
    "rust_python_vex_unop_fallback_count",
    "rust_python_vex_binop_fallback_count",
    "rust_python_vex_triop_fallback_count",
    "rust_python_vex_qop_fallback_count",
    "vex_fallback_count",
    "vex_fallback_unique_addrs",
]

# Native-proc fallback buckets: a native proc IS registered but deferred to
# Python. 'not_implemented' = registry miss (definitely closable: add the
# proc). 'symbolic' = native proc bailed on symbolic args (closable only by
# extending the native proc to handle symbolic). 'other' = misc deferral.
_NATIVE_BUCKETS = {
    "not_implemented": "native_proc_not_implemented_fallbacks_by_name",
    "symbolic": "native_proc_symbolic_fallbacks_by_name",
    "other": "native_proc_other_fallbacks_by_name",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benches", nargs="+", default=CANDIDATE_BENCHES)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--out", default=None, help="Write the aggregated JSON to this file (default: stdout only)")
    args = parser.parse_args()

    per_bench: dict[str, dict[str, int]] = {}
    per_bench_total: dict[str, int] = {}
    per_bench_native: dict[str, dict[str, dict[str, int]]] = {}
    per_bench_syscall: dict[str, dict[str, int]] = {}
    per_bench_vex: dict[str, dict[str, int]] = {}

    for name in args.benches:
        print(f"\n=== {name} ===", flush=True)
        result = run_example(name, "rust", timeout=args.timeout, mem_limit_mb=DEFAULT_MEM_LIMIT_MB)
        if not result or not result.get("ok"):
            print(f"  (skipped: {result.get('error') if result else 'no result'})")
            per_bench[name] = {}
            per_bench_total[name] = -1
            continue
        stats = result.get("stats") or {}

        # SimProcedure python fallbacks (the original signal).
        by_name = stats.get("simprocedure_fallback_by_name") or {}
        total = stats.get("simprocedure_python_fallback_count", 0)
        per_bench[name] = dict(by_name)
        per_bench_total[name] = total

        # Native-proc deferral buckets (registered proc bailed to Python).
        native = {bucket: dict(stats.get(key) or {}) for bucket, key in _NATIVE_BUCKETS.items()}
        per_bench_native[name] = native

        # Syscall fallbacks, keyed by syscall number (stringified for JSON).
        per_bench_syscall[name] = {str(k): v for k, v in (stats.get("syscall_python_fallback_by_num") or {}).items()}

        # VEX-op scalar fallback counters.
        per_bench_vex[name] = {k: stats.get(k, 0) for k in _VEX_SCALAR_COUNTERS}

        if by_name:
            top = sorted(by_name.items(), key=lambda kv: -kv[1])
            print(f"  simproc_fallback_total={total}")
            for pname, pcount in top:
                print(f"    [{classify_fallback(pname)[:4]}] {pname}: {pcount}")
        else:
            print(f"  simproc_fallback_total={total} (no per-proc breakdown)")

        native_hits = {b: d for b, d in native.items() if d}
        if native_hits:
            print("  native-proc deferrals:")
            for bucket, d in native_hits.items():
                for pname, pcount in sorted(d.items(), key=lambda kv: -kv[1]):
                    print(f"    [{bucket}] {pname}: {pcount}")
        sysd = per_bench_syscall[name]
        if sysd:
            print(f"  syscall fallbacks: {sysd}")
        vex_nonzero = {k: v for k, v in per_bench_vex[name].items() if v}
        if vex_nonzero:
            print(f"  vex fallbacks: {vex_nonzero}")

    # Aggregate simproc fallbacks, tagged.
    agg: dict[str, int] = collections.Counter()
    bench_appearances: dict[str, set] = collections.defaultdict(set)
    for name, by_name in per_bench.items():
        for pname, pcount in by_name.items():
            agg[pname] += pcount
            bench_appearances[pname].add(name)

    # Aggregate native-proc deferrals across buckets (always closable signal).
    native_agg: dict[str, collections.Counter] = {b: collections.Counter() for b in _NATIVE_BUCKETS}
    for native in per_bench_native.values():
        for bucket, d in native.items():
            for pname, pcount in d.items():
                native_agg[bucket][pname] += pcount

    syscall_agg: dict[str, int] = collections.Counter()
    for sysd in per_bench_syscall.values():
        for num, cnt in sysd.items():
            syscall_agg[num] += cnt

    vex_agg: dict[str, int] = collections.Counter()
    for vexd in per_bench_vex.values():
        for k, v in vexd.items():
            vex_agg[k] += v

    print("\n=== SIMPROC FALLBACK RANKING (tagged) ===")
    print(f"{'rank':>4}  {'count':>8}  {'benches':>8}  {'tag':>11}  procedure")
    closable_total = 0
    for rank, (pname, total) in enumerate(sorted(agg.items(), key=lambda kv: -kv[1]), 1):
        nbench = len(bench_appearances[pname])
        tag = classify_fallback(pname)
        if tag == "closable":
            closable_total += total
        print(f"{rank:>4}  {total:>8}  {nbench:>8}  {tag:>11}  {pname}")
    print(f"  -> closable simproc-fallback total: {closable_total}")

    print("\n=== NATIVE-PROC DEFERRALS (all closable native gaps) ===")
    any_native = False
    for bucket, counter in native_agg.items():
        for pname, total in sorted(counter.items(), key=lambda kv: -kv[1]):
            any_native = True
            print(f"  [{bucket}] {pname}: {total}")
    if not any_native:
        print("  (none — every registered native proc handled its calls)")

    print("\n=== SYSCALL FALLBACKS (by number) ===")
    if syscall_agg:
        for num, total in sorted(syscall_agg.items(), key=lambda kv: -kv[1]):
            print(f"  syscall {num}: {total}")
    else:
        print("  (none)")

    print("\n=== VEX-OP FALLBACKS ===")
    vex_nonzero = {k: v for k, v in vex_agg.items() if v}
    if vex_nonzero:
        for k, v in sorted(vex_nonzero.items(), key=lambda kv: -kv[1]):
            print(f"  {k}: {v}")
    else:
        print("  (none — all VEX ops ran natively)")

    out_payload = {
        "per_bench": per_bench,
        "per_bench_total": per_bench_total,
        "per_bench_native": per_bench_native,
        "per_bench_syscall": per_bench_syscall,
        "per_bench_vex": per_bench_vex,
        "aggregated": dict(agg),
        "aggregated_tags": {p: classify_fallback(p) for p in agg},
        "aggregated_native": {b: dict(c) for b, c in native_agg.items()},
        "aggregated_syscall": dict(syscall_agg),
        "aggregated_vex": dict(vex_agg),
        "bench_appearances": {p: sorted(v) for p, v in bench_appearances.items()},
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out_payload, f, indent=2, sort_keys=True)
        print(f"\nWrote JSON to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
