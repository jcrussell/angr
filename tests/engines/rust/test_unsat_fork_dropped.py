"""An UNSAT fork is dropped, not stashed in ``pruned`` (angr-vmi2l.326).

Python's SimulationManager drops unsat successors unless ``save_unsat`` is
set. Under eager forks (``use_deferred_forks=False``, which the phase-2 retry
sets) the Rust engine used to file the infeasible side of a branch in
``pruned``, the stash that should hold only ``max_active_states`` prunes.
"""

from __future__ import annotations

import pytest

import angr
from angr import claripy
from angr.exploration.rust_manager import RustExplorationManager

LOAD = 0x400000
# cmp rsi, 10 ; ja +2 ; jmp $ (avoid, LOAD+6) ; jmp $ (find, LOAD+8)
CODE = bytes.fromhex("4883fe0a") + b"\x77\x02" + b"\xeb\xfe" + b"\xeb\xfe"


def _explore(engine: str, deferred: bool = True) -> dict[str, int]:
    proj = angr.load_shellcode(CODE, arch="amd64", load_address=LOAD)
    state = proj.factory.blank_state(addr=LOAD, add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS})
    x = claripy.BVS("x", 64)
    state.solver.add(x.ULT(5))
    state.regs.rsi = x
    if engine == "rust":
        mgr = RustExplorationManager(proj, [state])
        mgr._rust_mgr.set_use_deferred_forks(deferred)
    else:
        mgr = proj.factory.simulation_manager(state)
    mgr.explore(find=LOAD + 8, avoid=LOAD + 6)
    return {k: len(v) for k, v in mgr.stashes.items() if v}


@pytest.mark.parametrize("deferred", [True, False], ids=["deferred", "eager"])
def test_unsat_branch_side_matches_python(deferred):
    py = _explore("py")
    assert py == {"avoid": 1}
    assert _explore("rust", deferred) == py


def test_eager_unsat_fork_still_counted():
    proj = angr.load_shellcode(CODE, arch="amd64", load_address=LOAD)
    state = proj.factory.blank_state(addr=LOAD, add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS})
    x = claripy.BVS("x", 64)
    state.solver.add(x.ULT(5))
    state.regs.rsi = x
    mgr = RustExplorationManager(proj, [state])
    mgr._rust_mgr.set_use_deferred_forks(False)
    mgr.explore(find=LOAD + 8, avoid=LOAD + 6)
    assert len(mgr.pruned) == 0
    assert mgr._rust_mgr.stats()["pruned_count"] > 0
