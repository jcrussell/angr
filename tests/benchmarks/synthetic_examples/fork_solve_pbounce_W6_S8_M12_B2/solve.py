#!/usr/bin/env python
"""Wide-AND-slow fork/solve synthetic with a PARTIAL Python-bounce (steady-state
demonstrator, angr-nkoct).

Sibling of ``fork_solve_trap`` (the FULL-bounce migration stress bench). Here
only a fraction ``B = 2^-k`` of the ``2^W`` leaves bounces at each of ``T``
interleaved levels; the rest keep stepping worker-locally. That partial overlap
is what the steady-state parallel loop converts into a wall-clock win — the
majority frontier does real mixing work IN PARALLEL while the minority is
serviced through the Python callback, and a bounce costs one materialize +
re-inject instead of a full-frontier re-seed.

Run EXHAUSTIVE (``num_find == 2^W``) so every leaf runs the whole chain; steady
mode helps only no-early-exit workloads (a num_find=1 first-find on a wide
frontier is anti-parallel — see bd parallel-numfind1-speculative-waste).

Drive with the steady loop via the environment:
    RUST_PARALLEL_WORKERS=2 RUST_PARALLEL_STEADY=1 python .../run_single.py \
        fork_solve_pbounce_W6_S8_M12_B2 --engine rust
"""

from __future__ import annotations

import os

import claripy

import angr

# Baked into the checked-in binary. Regenerate: python build_pbounce.py --w 6 --s 8 --m 12 --t 4 --b 2
W = 6  # branch count -> 2^W (64) leaf states
S = 8  # total nonlinear mixing rounds (spread across T levels)
M = 12  # find-gate mask width in bits
T = 4  # interleaved gated trap levels
B = 2  # bounce-fraction exponent k: 2^-k (1/4) of leaves bounce per level

VARIANT = os.environ.get("FORK_SOLVE_VARIANT") or f"fork_solve_pbounce_W{W}_S{S}_M{M}_B{B}"

STEP_BUDGET = 16384

# Exact-drain the full 2^W-leaf frontier so the run is exhaustive (no early
# exit). 64 for W=6.
NUM_FIND = int(os.environ.get("FORK_SOLVE_NUM_FIND", str(1 << W)))


def _expected_bounce_leaves_per_level() -> int:
    """Leaves whose concrete index makes the level gate fire.

    The gate is ``((s >> shift) & ((1<<B)-1)) == 0`` over a W-bit leaf index.
    With ``shift + B <= W`` (no wrap past the top bit — true for the committed
    W=6,B=2,T=4: shifts 0..3, top bit 4 < 6) exactly ``2^(W-B)`` of the ``2^W``
    indices have that field zero.
    """
    return 1 << (W - B)


class TrapPoint(angr.SimProcedure):
    """Identity hook for ``trap_point(x) -> x`` — a pure Python bounce."""

    def run(self, x):
        return x


def solve():
    bin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), VARIANT)

    proj = angr.Project(bin_path, auto_load_libs=False)
    main_sym = proj.loader.find_symbol("main")
    assert main_sym is not None, "main symbol not found"
    target_sym = proj.loader.find_symbol("reach_target")
    assert target_sym is not None, "reach_target symbol not found"
    trap_sym = proj.loader.find_symbol("trap_point")
    assert trap_sym is not None, "trap_point symbol not found"

    proj.hook(trap_sym.rebased_addr, TrapPoint())

    state = proj.factory.blank_state(addr=main_sym.rebased_addr)

    # 32 fully-symbolic stdin bytes so the modelled read(0, b, 32) returns
    # symbolic content without an extra fork.
    sym_bytes = claripy.BVS("stdin", 32 * 8)
    state.posix.stdin.content.append((sym_bytes, claripy.BVV(32, state.arch.bits)))

    sm = proj.factory.simulation_manager(state)
    sm.explore(find=target_sym.rebased_addr, num_find=NUM_FIND, n=STEP_BUDGET)
    return sm


def test():
    sm = solve()
    # Exhaustive drain: every one of the 2^W leaves reaches reach_target (the
    # partial-mask gate is essentially always satisfiable), pinning the
    # no-extra-fork property (a forking gate would inflate the leaf count).
    assert len(sm.found) == (1 << W), f"expected {1 << W} leaves, got {len(sm.found)}"


if __name__ == "__main__":
    sm = solve()
    print(
        f"{VARIANT}: found {len(sm.found)}/{1 << W} leaves; "
        f"~{_expected_bounce_leaves_per_level()} leaves bounce per level x {T} levels"
    )
