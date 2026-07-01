#!/usr/bin/env python
"""Wide-AND-slow fork/solve synthetic WITH a Python-bounce trap (migration gate).

Reconstruction of a lost benchmark that is the acceptance gate for angr-nkoct:
the level-synchronous parallel wave loop re-migrates the whole active frontier
across every Python-callback/wave boundary. This bench manufactures that
boundary REPEATEDLY at sequential program points so migration dominates.

Shape (see fork_solve_trap_template.c / README.md):

  * **Width** — ``W`` independent ``if (b[i] != 0)`` branches over a fully
    symbolic 32-byte stdin fork into ``2^W`` leaves BFS keeps active
    concurrently. W=5 -> exactly 32 leaves (the exact-drain value; found=32).
  * **Slow**  — ``S`` rounds of nonlinear mixing + a partial-mask find gate make
    every leaf's find-address ``satisfiable()`` a real but bounded Z3 solve.
  * **Trap**  — ``T`` sequential ``trap_point(acc)`` call sites AFTER the mixing
    region, hooked here by an identity ``SimProcedure``. Every call bounces all
    32 leaves to Python (Hook/SimProcedurePython -> NeedCallback), forcing run()
    re-entry and, at workers>1, a full-frontier re-migration (Z3-AST serde
    round-trip via StateMigrationPayload). T call sites => ~T re-migrations of
    the 32-wide frontier per exploration.

The identity SimProcedure returns the symbolic accumulator unchanged, so the
find-gate Z3 check stays exactly as hard as the un-trapped base — the trap adds
migration cost, not solver cost.
"""

from __future__ import annotations

import os

import claripy

import angr

# Baked into the checked-in fork_solve_trap_W5_S8_M12 binary.
# To regenerate: python build_trap.py --w 5 --s 8 --m 12 --t 4
W = 5  # branch count -> 2^W (32) leaf states (exact-drain: found == 32)
S = 8  # nonlinear mixing rounds (per-state solve cost)
M = 12  # match-mask width in bits (find-gate Z3 hardness)
T = 4  # trap_point() Python-bounce call sites (migration re-drain count)
# FORK_SOLVE_VARIANT overrides the binary during tuning only; the committed
# bench always loads the checked-in fork_solve_trap_W{W}_S{S}_M{M}.
VARIANT = os.environ.get("FORK_SOLVE_VARIANT") or f"fork_solve_trap_W{W}_S{S}_M{M}"

# Hard backstop against an unexpected fork storm; the real path is a few
# hundred steps (2^W leaves x (W branch + S mixing + T trap + gate) blocks).
STEP_BUDGET = 8192

# num_find: exact-drain the full 2^W-leaf frontier (32 for W=5) so every leaf
# runs the whole trap chain and the find check. This is what makes migration
# dominate: the entire frontier must survive every trap-level re-drain.
NUM_FIND = int(os.environ.get("FORK_SOLVE_NUM_FIND", "32"))


class TrapPoint(angr.SimProcedure):
    """Identity hook for ``trap_point(x) -> x``.

    Returning the (symbolic) argument unchanged preserves the accumulator so the
    downstream find-gate solve is unaffected; the only effect is the Python
    bounce that repopulates the active stash and drives re-migration.
    """

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

    # Hook the trap symbol with the identity SimProcedure -> every call site is a
    # Python bounce. cc=None lets angr infer the default cdecl/SysV convention.
    proj.hook(trap_sym.rebased_addr, TrapPoint())

    state = proj.factory.blank_state(addr=main_sym.rebased_addr)

    # Pre-stock stdin with 32 fully-symbolic bytes so the modelled
    # ``read(0, b, 32)`` returns symbolic content without an extra fork.
    sym_bytes = claripy.BVS("stdin", 32 * 8)
    state.posix.stdin.content.append((sym_bytes, claripy.BVV(32, state.arch.bits)))

    sm = proj.factory.simulation_manager(state)
    sm.explore(find=target_sym.rebased_addr, num_find=NUM_FIND, n=STEP_BUDGET)
    return sm


def test():
    sm = solve()
    assert sm.found, "no state reached reach_target within the step budget"


if __name__ == "__main__":
    sm = solve()
    print(f"{VARIANT}: found {len(sm.found)} state(s) reaching reach_target")
