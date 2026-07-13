#!/usr/bin/env python3
"""angr-op0dn.14.1.1 (M6.5a) — Rust->Python fallback census.

Two inventories, one artifact:

1. **Observed fallbacks.** Consumes one or more raw sweeps produced by
   ``collect_simproc_fallbacks.py --out <json>`` (which runs each bench in a
   subprocess under a 4GB RLIMIT_AS and dumps every fallback counter) and
   classifies every SimProcedure / syscall name that was observed *nonzero*
   across the corpus.

2. **Tagged degradation sites.** Scans ``angr/exploration/*.py`` for the
   ``cat-(a)`` / ``cat-(b) FALLBACK WITH LOSS`` / ``cat-(c) WRONG-ANSWER RISK``
   comments that mark every bridge-side degradation path, and classifies each
   site on the same axis.

The classification axis (shared by both inventories) is *observational*, not
"is this a closable native gap" (that axis lives in
``collect_simproc_fallbacks.classify_fallback``):

* ``silent-equivalent`` — the fallback produces the same observable result as
  the pure-Python engine. Only wall-clock differs.
* ``logged-degraded`` — the fallback continues with reduced fidelity and says
  so (a log record is emitted, or a counter is bumped). Observable, bounded.
* ``flip-blocking`` — the fallback can change the answer (wrong model, lost
  constraint, dropped side effect) with no signal. These gate the M6 default
  flip and are what S10 scores.

Usage::

    python tests/benchmarks/collect_simproc_fallbacks.py --out /tmp/raw1.json
    python tests/benchmarks/fallback_census.py --raw /tmp/raw1.json [/tmp/raw2.json]
    python tests/benchmarks/fallback_census.py --raw /tmp/raw*.json --out tests/benchmarks/fallback_census.json

Exits nonzero when any observed fallback name is unclassified (the bead's
acceptance criterion: ``unclassified == 0``).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from collect_simproc_fallbacks import _is_cpp_mangled, classify_fallback

SILENT = "silent-equivalent"
DEGRADED = "logged-degraded"
BLOCKING = "flip-blocking"

# ---------------------------------------------------------------------------
# Inventory 1 — observed fallback names.
#
# A SimProcedure fallback bounces into Python and runs *angr's own* proc
# against a SimState built by `rust_callback_dispatch._create_state_for_callback`,
# so the proc's own logic is identical to the Python engine's by construction.
# What can differ is the ROUND-TRIP: the resume FFI
# (`exploration/manager_methods.rs::resume_after_simprocedure`) carries exactly
# four payloads — new_pc, register changes, memory changes, new constraints —
# plus the `import_symbolic_to_state` / `set_pending_register_symbolic_ast`
# side channels. Anything a proc mutates outside that set has no wire to travel
# on and is silently dropped. So a name's class is decided by WHICH STATE
# SURFACE the proc touches, not by what it computes:
#
#   regs / memory / constraints / new BVS  -> carried by the resume FFI
#       (`rust_state_sync._extract_register_changes` / `_extract_memory_changes`
#       / `_extract_new_constraints`)                       -> silent-equivalent.
#   heap plugin (malloc/calloc bump)  -> `_ensure_critical_plugins` copies only
#       `posix` and `libc` from a template state, and the heap/brk push
#       (`rust_state_export._sync_rust_heap_brk_to_state`) runs only on the
#       EXPORT path, never from `_create_state_for_callback`. A bounced malloc
#       therefore reads a stale `heap_location` and its bump never reaches Rust
#       -> a later native alloc can hand out an overlapping address.
#   posix fd buffers (fwrite/fputc/fprintf/open)  -> `_inject_rust_stdout` is
#       likewise export-path-only, and there is NO outbound posix channel at all
#       (`_capture_stdin_from_successors` stashes stdin into a manager-level
#       dict, not into the Rust state) -> stream writes made by a bounced proc
#       never land in the Rust fd buffer that `posix.dumps(1)` reads back.
#
# A name whose round-trip fidelity is not established by inspection or by a
# witnessing bench is classed flip-blocking until .14.1.2's differential test
# proves otherwise. That is the conservative direction: a false flip-blocking
# costs a test; a false silent-equivalent ships a wrong answer.
# ---------------------------------------------------------------------------

# Explicit per-name classification. Rationale strings land in the artifact.
NAME_SEMANTICS: dict[str, tuple[str, str]] = {
    # --- pure compute over regs/memory. Reads args from synced registers,
    # writes its result through `state.memory.store` (which CallbackMemoryTracker
    # wraps) and/or a return register. Both cross the resume FFI.
    "strcmp": (SILENT, "reads synced memory, returns a BV in a synced register; no plugin state touched"),
    "strncmp": (SILENT, "reads synced memory, returns a BV in a synced register; no plugin state touched"),
    "strncpy": (SILENT, "memory-only; on rust_manager._MEMORY_WRITING_PROCS so it gets a full copy+diff"),
    "memmove": (SILENT, "memory-only; on rust_manager._MEMORY_WRITING_PROCS so it gets a full copy+diff"),
    "__ctype_b_loc": (
        SILENT,
        "writes its table through state.memory.store (tracked by CallbackMemoryTracker) and returns a "
        "pointer in a synced register",
    ),
    # --- heap: the bump round-trips since angr-op0dn.14.1.3 (the callback path
    # now reads state.heap_brk in and pushes the bounced bump back out).
    "malloc": (
        SILENT,
        "the bounced heap bump now round-trips (angr-op0dn.14.1.3): _create_state_for_callback seeds "
        "heap_location from Rust's brk and the resume path pushes it back, so a later native alloc "
        "cannot overlap. Differential parity: test_fallback_parity.py::test_parity[malloc]",
    ),
    "calloc": (
        SILENT,
        "same channel as malloc (angr-op0dn.14.1.3). Differential parity, including the zeroed block: "
        "test_fallback_parity.py::test_parity[calloc]",
    ),
    # C++ new/delete land on angr's malloc/free SimProcedures, so they inherit
    # malloc's heap round-trip; parity is proven transitively via that proc.
    "operator new(unsigned long)": (
        SILENT,
        "routes to angr's malloc proc, whose bounced bump round-trips (angr-op0dn.14.1.3); parity is the malloc row's",
    ),
    "operator delete(void*)": (
        SILENT,
        "routes to angr's free proc; angr's free is a no-op on the heap bump, so nothing is lost",
    ),
    # --- posix stream writes: the outbound posix channel landed in
    # angr-op0dn.14.1.4 (bounced writes are replayed into Rust's fd buffer).
    "fwrite": (
        SILENT,
        "the bounced stream write now reaches Rust's fd buffer through the outbound posix channel "
        "(angr-op0dn.14.1.4), so a later posix.dumps(1) sees it. Differential parity: "
        "test_fallback_parity.py::test_parity[fwrite]. The known ~2.8ms/call bounce cost (csyy9) "
        "remains a perf, not a correctness, issue",
    ),
    "fputc": (SILENT, "same outbound posix channel as fwrite. Parity: test_parity[fputc]"),
    "fprintf": (SILENT, "same outbound posix channel as fwrite. Parity: test_parity[fprintf]"),
    # --- fd path: both halves of the fd-table channel landed.
    "open": (
        SILENT,
        "a bounced open's fd is adopted into Rust's fd table (angr-op0dn.14.1.5) and Rust's own fds "
        "are seeded into the callback state (angr-op0dn.14.1.6), so the two sides cannot disagree on "
        "an fd number. Differential parity: test_fallback_parity.py::test_parity[open]",
    ),
    "read": (
        SILENT,
        "witnessed by the file_read_kernel bench (angr-w5llj): the pre-seeded file IS readable "
        "through the Python fallback and the bench reaches found. read is on _MEMORY_WRITING_PROCS, "
        "so the buffer it fills crosses the resume FFI",
    ),
    # --- env/time: read-only against angr's own plugin defaults, which the
    # Rust side has no independent copy of.
    "getenv": (SILENT, "reads the posix environ, which is fixed at project init; no Rust-side environ to desync"),
    "time": (SILENT, "returns a fresh BV in a synced register; angr's Python proc is the reference"),
    # --- pthread stubs: angr models these as no-ops / unconstrained returns.
    "pthread_once": (SILENT, "angr models it as a call-once stub; no Rust-side pthread state exists"),
    "pthread_mutex_lock": (SILENT, "angr models it as a no-op returning 0; no Rust-side lock state"),
    "pthread_mutex_unlock": (SILENT, "angr models it as a no-op returning 0; no Rust-side lock state"),
    # --- user hooks: a CTF solve.py hook. Always Python by design, and
    # _snapshot_orig_state gives any UserHook a full state.copy(), so its
    # memory writes are diffed back in full.
    "UserHook": (SILENT, "user-supplied Python hook; gets a full state.copy() snapshot by design"),
    "my_scanf": (SILENT, "solve.py UserHook (defcon2016quals_baby-re) that injects symbolic stdin"),
    "get_flag": (SILENT, "solve.py UserHook (whitehatvn2015_re400)"),
    "readline_hook": (SILENT, "solve.py UserHook (cmu_binary_bomb_partial) that injects a symbolic line"),
    "strtol_hook": (SILENT, "solve.py UserHook (cmu_binary_bomb_partial)"),
}

# Syscall numbers observed nonzero, keyed by the number the counter reports.
#
# Every syscall fallback used to be flip-blocking: `_handle_syscall_callback_inner`
# passed the 2-tuple returned by `_extract_memory_changes` straight to
# `resume_after_syscall` (every other caller unpacks it), so the PyO3 conversion
# raised, the enclosing `except Exception` swallowed it, and the state resumed at
# `state.addr + 1` with ALL register and memory changes discarded. Fixed in
# angr-89w70; regression test
# tests/engines/rust/test_syscalls.py::TestPythonSyscallFallbackSyncsChanges.
SYSCALL_SEMANTICS: dict[str, tuple[str, str]] = {
    "334": (
        SILENT,
        "faccessat: native syscall registered (syscalls/mod.rs) but bailed to Python. The Python "
        "syscall resume path now carries the bounce's register and memory changes back into Rust "
        "(angr-89w70); proven by test_syscalls.py::TestPythonSyscallFallbackSyncsChanges",
    ),
}

# ---------------------------------------------------------------------------
# Inventory 2 — tagged degradation sites in the Python bridge.
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"cat-\((?P<cat>[abc])\)")
_DEF_RE = re.compile(r"^(?P<indent>\s*)def\s+(?P<name>\w+)")
_LOG_RE = re.compile(r"\bl\.(debug|info|warning|error|exception)\(|\blog(ger)?\.(debug|info|warning|error)\(")

_CAT_CLASS = {"a": SILENT, "b": DEGRADED, "c": BLOCKING}

# How far past the tag comment to look for the log call that makes a
# degradation observable. Handlers in this codebase log within a few lines of
# the tag or not at all.
_LOG_WINDOW = 10

_BRIDGE_DIR = os.path.join(_REPO, "angr", "exploration")


def _enclosing_def(lines: list[str], idx: int) -> str:
    """Nearest preceding `def` at a lower indent than the tagged line."""
    tag_indent = len(lines[idx]) - len(lines[idx].lstrip())
    for j in range(idx, -1, -1):
        m = _DEF_RE.match(lines[j])
        if m and len(m.group("indent")) < tag_indent:
            return m.group("name")
    return "<module>"


def scan_tag_sites() -> list[dict]:
    sites: list[dict] = []
    for fname in sorted(os.listdir(_BRIDGE_DIR)):
        if not fname.endswith(".py"):
            continue
        path = os.path.join(_BRIDGE_DIR, fname)
        with open(path) as f:
            lines = f.read().splitlines()
        for i, line in enumerate(lines):
            m = _TAG_RE.search(line)
            if not m:
                continue
            cat = m.group("cat")
            window = "\n".join(lines[i : i + _LOG_WINDOW])
            logs = bool(_LOG_RE.search(window))
            sites.append(
                {
                    "file": f"angr/exploration/{fname}",
                    "line": i + 1,
                    "function": _enclosing_def(lines, i),
                    "category": f"cat-{cat}",
                    "semantics": _CAT_CLASS[cat],
                    "logs": logs,
                    "comment": line.strip().lstrip("# ").strip(),
                }
            )
    return sites


# ---------------------------------------------------------------------------


def classify_name(name: str) -> tuple[str, str]:
    """(semantics, rationale) for an observed SimProcedure-fallback name."""
    if name in NAME_SEMANTICS:
        return NAME_SEMANTICS[name]
    if _is_cpp_mangled(name):
        return (
            SILENT,
            "C++ stdlib symbol: angr's Python proc (or ReturnUnconstrained) is the reference; "
            "no native equivalent exists to diverge from",
        )
    if classify_fallback(name) == "intentional":
        return (SILENT, "angr-internal stub, Python by design; no Rust-side implementation to diverge from")
    return ("unclassified", "")


def _merge_raw(paths: list[str]) -> dict:
    per_bench: dict[str, dict[str, int]] = {}
    per_bench_total: dict[str, int] = {}
    per_bench_syscall: dict[str, dict[str, int]] = {}
    per_bench_native: dict[str, dict[str, dict[str, int]]] = {}
    per_bench_vex: dict[str, dict[str, int]] = {}
    for p in paths:
        with open(p) as f:
            raw = json.load(f)
        per_bench.update(raw.get("per_bench") or {})
        per_bench_total.update(raw.get("per_bench_total") or {})
        per_bench_syscall.update(raw.get("per_bench_syscall") or {})
        per_bench_native.update(raw.get("per_bench_native") or {})
        per_bench_vex.update(raw.get("per_bench_vex") or {})
    return {
        "per_bench": per_bench,
        "per_bench_total": per_bench_total,
        "per_bench_syscall": per_bench_syscall,
        "per_bench_native": per_bench_native,
        "per_bench_vex": per_bench_vex,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True, help="raw JSON sweeps from collect_simproc_fallbacks.py --out")
    ap.add_argument("--out", default=None, help="write the census JSON artifact here")
    args = ap.parse_args()

    merged = _merge_raw(args.raw)

    # --- aggregate observed SimProcedure fallbacks
    agg: collections.Counter = collections.Counter()
    appearances: dict[str, set] = collections.defaultdict(set)
    for bench, by_name in merged["per_bench"].items():
        for name, count in by_name.items():
            agg[name] += count
            appearances[name].add(bench)

    names_out = {}
    for name, count in agg.items():
        semantics, why = classify_name(name)
        names_out[name] = {
            "count": count,
            "benches": sorted(appearances[name]),
            "closability": classify_fallback(name),
            "semantics": semantics,
            "rationale": why,
        }

    syscalls_out = {}
    syscall_agg: collections.Counter = collections.Counter()
    for sysd in merged["per_bench_syscall"].values():
        for num, cnt in sysd.items():
            syscall_agg[num] += cnt
    for num, count in syscall_agg.items():
        semantics, why = SYSCALL_SEMANTICS.get(num, ("unclassified", ""))
        syscalls_out[num] = {"count": count, "semantics": semantics, "rationale": why}

    sites = scan_tag_sites()

    unclassified = sorted([n for n, d in names_out.items() if d["semantics"] == "unclassified"]) + sorted(
        [f"syscall {n}" for n, d in syscalls_out.items() if d["semantics"] == "unclassified"]
    )

    # --- report
    benches = merged["per_bench_total"]
    print(
        f"=== CENSUS: {len(benches)} benches swept, "
        f"{sum(1 for v in benches.values() if v > 0)} with nonzero SimProcedure fallbacks"
    )
    print(f"\n=== OBSERVED FALLBACK NAMES ({len(names_out)}) ===")
    print(f"{'count':>6}  {'semantics':>17}  {'closability':>11}  name")
    for name, d in sorted(names_out.items(), key=lambda kv: -kv[1]["count"]):
        short = name if len(name) <= 60 else name[:57] + "..."
        print(f"{d['count']:>6}  {d['semantics']:>17}  {d['closability']:>11}  {short}")
    for num, d in sorted(syscalls_out.items()):
        print(f"{d['count']:>6}  {d['semantics']:>17}  {'-':>11}  syscall {num}")

    by_class: collections.Counter = collections.Counter(d["semantics"] for d in names_out.values())
    print(f"\n  name classes: {dict(by_class)}")

    site_class: collections.Counter = collections.Counter(s["semantics"] for s in sites)
    unlogged = [s for s in sites if s["semantics"] == DEGRADED and not s["logs"]]
    print(f"\n=== TAGGED DEGRADATION SITES ({len(sites)}) ===")
    print(f"  classes: {dict(site_class)}")
    print(f"  cat-(b) sites that degrade WITHOUT a log record: {len(unlogged)}")
    for s in unlogged:
        print(f"    {s['file']}:{s['line']} {s['function']}()")
    print("\n=== flip-blocking sites (cat-(c)) by file ===")
    per_file: collections.Counter = collections.Counter(s["file"] for s in sites if s["semantics"] == BLOCKING)
    for f, n in per_file.most_common():
        print(f"  {n:>3}  {f}")

    payload = {
        "bead": "angr-op0dn.14.1.1",
        "benches_swept": sorted(benches),
        "bench_fallback_totals": benches,
        "names": names_out,
        "syscalls": syscalls_out,
        "native_proc_deferrals": merged["per_bench_native"],
        "tag_sites": sites,
        "summary": {
            "name_classes": dict(by_class),
            "site_classes": dict(site_class),
            "unlogged_degraded_sites": [f"{s['file']}:{s['line']}" for s in unlogged],
            "unclassified": unclassified,
        },
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"\nWrote census artifact to {args.out}")

    if unclassified:
        print(f"\nFAIL: {len(unclassified)} unclassified fallback(s): {unclassified}")
        return 1
    print("\nOK: every observed fallback name and every tagged site is classified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
