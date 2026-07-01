#!/usr/bin/env python
"""Wide-AND-slow fork/solve synthetic benchmark (parallel-symex GO gate).

The parallel wave loop needs a workload that is *wide* (many independent active
states at the same time) AND *slow* (each state carries a hard solver check).
The rest of the corpus is one or the other, never both (see
``run_width_audit.py``). This bench manufactures the shape:

  * **Width** — ``W`` independent ``if (b[i] != 0)`` branches over a fully
    symbolic 32-byte stdin fork into up to ``2^W`` leaves that BFS keeps active
    concurrently (sustained frontier width -> ``frac_ge3`` well above 0.20).
  * **Slow**  — after the branch region every leaf runs ``S`` rounds of nonlinear
    mixing and a closing ``if ((acc & MASK) == PATTERN) reach_target();``. The
    partial-mask match over the mixed accumulator is a real but bounded Z3 check
    (a full-word equality instead forces pathological UNSAT proofs, seconds per
    leaf). BFS materializes the whole 2^W frontier before any leaf reaches the
    find address, so the first find-address ``satisfiable()`` fires with the
    frontier already wide -> ``ms/task`` well above 50.

The checked-in binary is ``fork_solve_W6_S8`` (W=6 -> 64 leaves, S=8 mixing
rounds, M=20-bit match mask), built ``gcc -O0 -no-pie`` x86-64 by ``build.py`` —
small enough to stay well under the 4 GB regression cap (memory is bounded by
``2^W`` leaves; cost lives in constraint *depth*, not state count).

Rust-engine only in spirit (the point is the Rust parallel frontier), but the
driver builds its simulation manager through ``proj.factory.simulation_manager``
so ``run_single.py --engine rust`` swaps in ``RustExplorationManager``
transparently. Symbolic stdin, ``auto_load_libs=False``, no libc callbacks ->
clean parallel signal.
"""

from __future__ import annotations

import os

import claripy

import angr

# Baked into the checked-in fork_solve_W6_S8 binary (M=20 match-mask width).
# To regenerate: python build.py --w 6 --s 8 --m 20
W = 6  # branch count -> 2^W (64) leaf states
S = 8  # nonlinear mixing rounds (per-state solve cost)
# FORK_SOLVE_VARIANT overrides the binary during tuning only; the committed
# bench always loads the checked-in fork_solve_W{W}_S{S}.
VARIANT = os.environ.get("FORK_SOLVE_VARIANT") or f"fork_solve_W{W}_S{S}"

# Hard backstop: 2^W leaves each take ~W + a handful of mixing blocks to reach
# reach_target, so a few hundred steps total. The cap only guards against an
# unexpected fork storm.
STEP_BUDGET = 4096

# num_find: how many find-hits to collect before stopping. BFS explores
# breadth-first, so the full 2^W-leaf frontier is materialized and active *at the
# same time* well before any leaf reaches reach_target (peak width ~2^W, high
# frac_ge3). Stopping at the FIRST find (num_find=1) bounds the very expensive
# per-leaf satisfiable() work to the first wave so the whole bench finishes in
# ~1 min under the audit timeout, while still exhibiting the wide-AND-slow shape.
# A full sweep (num_find=2^W) instead runs every leaf's hard check and blows past
# the timeout. FORK_SOLVE_NUM_FIND overrides for tuning.
NUM_FIND = int(os.environ.get("FORK_SOLVE_NUM_FIND", "1"))


def solve():
    bin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), VARIANT)

    proj = angr.Project(bin_path, auto_load_libs=False)
    main_sym = proj.loader.find_symbol("main")
    assert main_sym is not None, "main symbol not found"
    target_sym = proj.loader.find_symbol("reach_target")
    assert target_sym is not None, "reach_target symbol not found"

    state = proj.factory.blank_state(addr=main_sym.rebased_addr)

    # Pre-stock stdin with 32 fully-symbolic bytes so the modelled
    # ``read(0, b, 32)`` returns symbolic content without an extra fork.
    sym_bytes = claripy.BVS("stdin", 32 * 8)
    state.posix.stdin.content.append((sym_bytes, claripy.BVV(32, state.arch.bits)))

    sm = proj.factory.simulation_manager(state)
    # BFS builds the full 2^W-leaf frontier (all leaves active at once) and stops
    # at the first find; the expensive per-leaf satisfiable() check at the find
    # address is the slow work a worker pool would parallelize.
    sm.explore(find=target_sym.rebased_addr, num_find=NUM_FIND, n=STEP_BUDGET)
    return sm


def test():
    sm = solve()
    assert sm.found, "no state reached reach_target within the step budget"


if __name__ == "__main__":
    sm = solve()
    print(f"{VARIANT}: found {len(sm.found)} state(s) reaching reach_target")
