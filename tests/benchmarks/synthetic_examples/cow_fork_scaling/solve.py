#!/usr/bin/env python
"""CoW fork-scaling synthetic benchmark (angr-vx8p.2, from angr-uf0g).

A parameterised fork-tree binary that reads 16 fully-symbolic bytes from
stdin and runs them through ``N`` independent ``if (b[i] != 0)`` branches,
so a fully symbolic input produces ``2^N`` reachable leaf states. The
checked-in binary is built with ``N=8`` (256 leaf states) — large enough
to exercise the engine's fork machinery at scale, small enough to finish
in seconds well under the 4 GB regression cap.

This is the synthetic regression gate for the Rust engine's core O(1)
CoW-fork claim (``im::OrdMap`` structural sharing). Unlike the rest of the
baseline corpus — 28/30 entries have ``state_creations == 0`` — this bench
deliberately forks hundreds of states, so ``state_creations`` / ``steps``
carry real signal here.

The binary ships prebuilt (``fork_tree_8``, gcc ``-O0 -no-pie`` x86_64) so
the bench needs no cross-compiler at test time; ``fork_tree_8.c`` records
the exact source. The full sweep tooling and characterisation writeup live
under ``tests/benchmarks/characterization/cow_fork_scaling/``.

Configured ``rust_only=True`` in ``run_regression.FAST_SUITE``: the point
is Rust fork scaling, and the Python engine OOMs past N=8 in the uf0g
sweep, so no Python comparison is recorded.
"""

from __future__ import annotations

import os

import claripy

import angr

N = 8  # branch count baked into the checked-in fork_tree_8 binary
EXPECTED_LEAVES = 1 << N  # 256 fully-deadended leaf states


def solve():
    bin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"fork_tree_{N}")

    proj = angr.Project(bin_path, auto_load_libs=False)
    main_sym = proj.loader.find_symbol("main")
    assert main_sym is not None, "main symbol not found"

    state = proj.factory.blank_state(addr=main_sym.rebased_addr)

    # Pre-stock stdin with 16 fully-symbolic bytes so the modelled
    # ``read(0, b, 16)`` returns symbolic content without an extra fork.
    sym_bytes = claripy.BVS("stdin", 16 * 8)
    state.posix.stdin.content.append((sym_bytes, claripy.BVV(16, state.arch.bits)))

    sm = proj.factory.simulation_manager(state)
    sm.run()

    return len(sm.deadended)


def test():
    leaves = solve()
    assert leaves == EXPECTED_LEAVES, f"expected {EXPECTED_LEAVES} leaf states, got {leaves}"


if __name__ == "__main__":
    print(f"deadended leaf states = {solve()}")
