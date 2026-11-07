"""``RustExplorationManager.merge`` keys, prunes and is called like ``SimulationManager.merge`` (angr-vmi2l.240).

The Rust merge grouped on the pc alone, ignored ``prune`` and took ``stash``
as its first positional. Upstream's default key also holds the call stack's
function addresses and the open fds, and ``prune=True`` moves unsat states
to ``pruned`` before merging. The shellcode reads one stdin byte and reaches
``helper`` either directly or through ``f1``, so both arms stop at one pc
under different call stacks.
"""

from __future__ import annotations

import struct

import pytest

import angr
from angr import claripy
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")

LOAD = 0x400000
BUF = 0x600000


def _call(at: int, target: int) -> bytes:
    return b"\xe8" + struct.pack("<i", target - (at + 5))


def _two_callers() -> tuple[bytes, int]:
    """read(0, BUF, 1); 'A' calls f1, which calls helper; else call helper. Returns (code, helper)."""
    movabs = (b"\x48\xb8", b"\x48\xbf", b"\x48\xbe", b"\x48\xba")
    read = b"".join(op + struct.pack("<Q", v) for op, v in zip(movabs, (0, 0, BUF, 1))) + b"\x0f\x05"
    cmp_a = b"\x80\x3c\x25" + struct.pack("<I", BUF) + b"\x41"
    arm_b_at = LOAD + len(read) + len(cmp_a) + 2
    arm_a_at = arm_b_at + 7
    f1 = arm_a_at + 7
    helper = f1 + 6
    code = (
        read
        + cmp_a
        + b"\x74\x07"  # je arm_a
        + _call(arm_b_at, helper)
        + b"\xeb\xfe"
        + _call(arm_a_at, f1)
        + b"\xeb\xfe"
        + _call(f1, helper)
        + b"\xc3"
        + b"\xeb\xfe"  # helper: spin
    )
    assert LOAD + len(code) - 2 == helper
    return code, helper


def _project():
    code, helper = _two_callers()
    return angr.load_shellcode(code, arch="amd64", simos="linux", load_address=LOAD), helper


def _entry(proj):
    st = proj.factory.entry_state(
        addr=LOAD,
        add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS, angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY},
    )
    st.memory.store(BUF, b"\0" * 8)
    return st


def _found_call_stacks(engine: str, merge_key=None):
    proj, helper = _project()
    st = _entry(proj)
    mgr = RustExplorationManager(proj, [st]) if engine == "rust" else proj.factory.simulation_manager(st)
    mgr.explore(find=helper, num_find=2)
    assert len(list(mgr.found)) == 2
    if engine == "rust" and merge_key is None:
        # The native path keys them itself, rather than handing them to Python.
        keys = [mgr._native_merge_key(sid) for sid in mgr._rust_mgr.get_state_ids("found")]
        assert None not in keys
        assert keys[0] != keys[1]
    mgr.merge(merge_key=merge_key, stash="found")
    return sorted(len(list(s.callstack)) for s in mgr.found)


@pytest.mark.parametrize("engine", ["py", "rust"])
def test_same_pc_under_different_callers_stays_unmerged(engine):
    # Two found states, each keeping its own call stack (helper<-f1<-root, helper<-root).
    stacks = _found_call_stacks(engine)
    assert len(stacks) == 2
    assert stacks[0] != stacks[1]


@pytest.mark.parametrize("engine", ["py", "rust"])
def test_pc_only_key_still_merges_them(engine):
    assert len(_found_call_stacks(engine, merge_key=lambda s: s.addr)) == 1


@pytest.mark.parametrize("engine", ["py", "rust"])
def test_prune_moves_an_unsat_state_out_before_merging(engine):
    proj, _helper = _project()
    x = claripy.BVS("x", 32)
    sat, unsat = _entry(proj), _entry(proj)
    sat.solver.add(x == 1)
    unsat.solver.add(x == 1)
    unsat.solver.add(x == 2)
    mgr = (
        RustExplorationManager(proj, [sat, unsat])
        if engine == "rust"
        else proj.factory.simulation_manager([sat, unsat])
    )
    mgr.merge()
    assert (len(list(mgr.active)), len(list(mgr.pruned))) == (1, 1)
    assert next(iter(mgr.active)).solver.satisfiable()


def test_merge_takes_merge_func_first_like_upstream():
    proj, _helper = _project()
    a, b = _entry(proj), _entry(proj)
    seen = []

    def merge_func(*states):
        seen.append(len(states))
        return states[0]

    mgr = RustExplorationManager(proj, [a, b])
    mgr.merge(merge_func)
    assert seen == [2]
    assert len(list(mgr.active)) == 1
