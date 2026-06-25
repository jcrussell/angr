#!/usr/bin/env python
"""Synthetic ctype/fprintf micro-bench: __ctype_b_loc + fprintf (angr-11djq.20).

The CTF-heavy baseline corpus barely touches the libc helpers that real
command-line utilities lean on. Before this fixture the bench gate reached
``fprintf`` only incidentally (sharif7_rev50) and never exercised the locale
ctype classifier (``__ctype_b_loc``) at all — so the native-coverage beads
(angr-tx7ec.4 ctype, angr-884yn fprintf) shipped with unit tests but no
end-to-end gate.

This driver runs a tiny checked-in ELF (``cli_ctype_fprintf``, gcc ``-O0
-no-pie`` x86-64) that runs four symbolic stdin bytes through an ``isdigit``
short-circuit chain, fanning out to five leaf states (four early not-a-digit
returns, each printing via ``fprintf(stderr)``, plus one all-digits ``win``
printing via ``fprintf(stdout)``). The ``win`` symbol is the explore target.

Coverage notes:
  * ``isdigit`` resolves the classifier table through ``__ctype_b_loc`` (native
    proc, angr-tx7ec.4) and the formatted output goes through ``fprintf``
    (native proc, angr-884yn). Inspect the fallback surface with
    ``run_single.py cli_ctype_fprintf --dump-counters``.
  * Both procs currently fall back to Python in this *end-to-end* harness, and
    that is expected: the native __ctype_b_loc reads a classifier-table pointer
    that ``__libc_start_main`` only builds *during* exploration (after the Rust
    seed state is captured), so the pointer is null at native dispatch time and
    the proc returns Err "table not initialized" (ctype.rs ctype_loc_ptr);
    fprintf to stderr/stdout falls back pending write-side fileno resolution
    (angr-csyy9). The native execution itself is unit-tested (the tables are
    set up manually there). What this bench adds is end-to-end gate coverage of
    the ctype/fprintf *dispatch + fallback* path on a symbolic CLI-shaped path —
    nothing else in the corpus drives __ctype_b_loc at all — so a regression in
    that machinery now trips a gate, not just a unit test.
  * getopt is deliberately absent: as of angr-ae54t.6 it has a Python
    SimProcedure but no *native* proc, so it would fall back to Python for zero
    native coverage (the point of this fixture); before that it had no
    SimProcedure at all and ran real glibc through the VEX interpreter (~15s).

The binary is small and self-contained, so it is checked in alongside its
source (``cli_ctype_fprintf.c``) and needs no compiler at test time.

``rust_only=True`` in ``run_regression.FAST_SUITE``: the point is native-proc
gate coverage / fallback measurement, not a Rust-vs-Python speed claim.

Run it:
    python tests/benchmarks/run_single.py cli_ctype_fprintf --engine rust
    python tests/benchmarks/run_single.py cli_ctype_fprintf --both
    python tests/benchmarks/run_single.py cli_ctype_fprintf --dump-counters
"""

from __future__ import annotations

import os

import claripy

import angr

# The all-digits path forks at most five leaf states; this cap is a backstop
# against any unexpected fork storm.
STEP_BUDGET = 100


def _binary_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "cli_ctype_fprintf")


def solve():
    target = _binary_path()
    # auto_load_libs=False so the libc hooks land in angr's extern region
    # rather than a loaded-libc binary region: the Rust engine skips its
    # native fast path for addresses inside a binary region (run_loop.rs
    # is_in_binary check), so loading real libc would force every
    # fprintf/__ctype_b_loc call back to the Python SimProcedure and defeat
    # the native-coverage purpose of this fixture.
    proj = angr.Project(target, auto_load_libs=False, use_sim_procedures=True)

    win = proj.loader.find_symbol("win")
    assert win is not None, "win symbol not found"

    # entry_state (not blank_state) so __libc_start_main runs and populates the
    # locale ctype table — the native __ctype_b_loc proc returns Err ("table
    # not initialized") and falls back to Python without it (ctype.rs
    # ctype_loc_ptr). Four fully-symbolic stdin bytes drive the isdigit chain;
    # read() consumes exactly four, so the fork count is five leaf states.
    stdin = claripy.BVS("stdin", 4 * 8)
    state = proj.factory.entry_state(stdin=stdin)

    sm = proj.factory.simulation_manager(state)
    sm.explore(find=win.rebased_addr, num_find=1, n=STEP_BUDGET)
    return sm


def test():
    sm = solve()
    assert sm.found, "did not reach the all-digits win path within the step budget"


if __name__ == "__main__":
    sm = solve()
    print(f"cli_ctype_fprintf: reached win -> {len(sm.found)} found state(s)")
