"""Merging states a RustBackend exported (angr-mx00p.132.68).

An export's proxy plugins read a Rust state and cannot merge, so
``SimState.merge`` merges a copy of each export with plugins of its own, as
pickling and a Python step do. The exports themselves stay bound to their Rust
states.
"""

from __future__ import annotations

import os

import pytest

import angr
from angr import claripy
from angr.exploration.rust_backend import RustBackend
from angr.exploration.rust_state_binding import binding_of
from angr.exploration.rust_state_export import _stream_packet_ends
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, TEST_BINARIES_DIR

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust extension not built")


def _fauxware(arch):
    return angr.Project(os.path.join(TEST_BINARIES_DIR, "..", arch, "fauxware"), auto_load_libs=False, engine="rust")


@pytest.mark.parametrize("arch", ["i386", "x86_64"])
def test_a_factory_built_rust_manager_merges_its_deadended_stash(arch):
    """Upstream test_fauxware's _run_merge on the factory route: the merged
    state's merge conditions separate the SOSNEAKY input from the rest."""
    simgr = _fauxware(arch).factory.simulation_manager()
    assert isinstance(simgr.backend, RustBackend)
    simgr.explore()
    for s in simgr.deadended:
        s.release_plugin("fs")
        if 3 in s.posix.fd:
            s.posix.close(3)

    simgr.merge(stash="deadended", merge_key=lambda s: s.addr)

    path = next(s for s in simgr.deadended if b"Welcome" in s.posix.dumps(1))
    conditions = path.history.merge_conditions
    inp = path.posix.stdin.content[2][0]
    sneaky = [b"SOSNEAKY" in path.solver.eval(inp, cast_to=bytes, extra_constraints=(c,)) for c in conditions]
    assert sorted(sneaky) == [False, True]


def test_merging_two_exports_leaves_them_bound():
    simgr = _fauxware("i386").factory.simulation_manager()
    simgr.run(until=lambda sm: len(sm.active) > 1)
    a, b = simgr.active[:2]
    assert binding_of(a) is not None and binding_of(b) is not None
    # Native writes and reads keep Python's packet layout, so the posix
    # plugins merge too (angr-mx00p.132.72).
    for s in (a, b):
        assert [p[1].concrete_value for p in s.posix.stdout.content] == [10, 1, 10, 1]
        assert [p[1].concrete_value for p in s.posix.stdin.content] == [8, 1, 8, 1]
        assert (s.posix.stdout.pos, s.posix.stdin.pos) == (4, 4)

    merged, _, merging_occurred = a.merge(b)

    assert merging_occurred
    assert binding_of(merged) is None
    assert sorted(merged.solver.eval_upto(merged.regs.ip, 3)) == sorted([a.addr, b.addr])
    assert binding_of(a) is not None and binding_of(b) is not None
    assert {a.addr, b.addr} == {s.addr for s in simgr.active[:2]}


def test_stream_packet_ends_follow_the_python_packets():
    # The write-back appends one Rust write per packet at these ends (angr-mx00p.132.72).
    state = _fauxware("x86_64").factory.blank_state()
    stdout = state.posix.stdout
    stdout.write(None, claripy.BVV(b"Username: "))
    stdout.write(None, claripy.BVV(b"\n"))
    assert _stream_packet_ends(stdout) == [10, 11]
    assert _stream_packet_ends(state.posix.stdin) == []
