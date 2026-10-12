# pylint: disable=missing-class-docstring,no-self-use
"""Regression tests for angr-hc1yj: callable find/avoid on stdout.

Under RustExplorationManager with callable predicates, (a) the sweep handed
predicates stale cached Python frames whose posix stdout had another state's
output appended, (b) ``found`` listed every predicate match twice, and (c) the
find-callback proxy carried no stdout at all. fauxware's accepting paths print
"Welcome ..." and the rejecting path prints "Go away!", so each found/avoid
state's own stdout pins the classification. The Python engine finds 2
accepting paths and avoids 1.
"""

from __future__ import annotations

import pytest

from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")

WELCOME = b"Welcome"
GO_AWAY = b"Go away"


def _explore(project, num_find=1, record=None):
    def find(s):
        out = s.posix.dumps(1)
        if record is not None:
            record.append(out)
        return WELCOME in out

    mgr = RustExplorationManager(project, [project.factory.entry_state()])
    mgr.explore(find=find, avoid=lambda s: GO_AWAY in s.posix.dumps(1), num_find=num_find)
    return mgr


class TestPredicateStdout:
    def test_found_lists_each_match_once(self, fauxware_project):
        mgr = _explore(fauxware_project)
        rust_found = mgr._rust_mgr.get_state_ids("found")
        assert rust_found, "no accepting path found"
        assert len(mgr.found) == len(rust_found), "found double-lists predicate matches"

    def test_found_and_avoid_carry_their_own_stdout(self, fauxware_project):
        mgr = _explore(fauxware_project, num_find=5)
        assert mgr.found and mgr.avoid
        for s in mgr.found:
            out = s.posix.dumps(1)
            assert WELCOME in out and GO_AWAY not in out, out
        for s in mgr.avoid:
            out = s.posix.dumps(1)
            assert GO_AWAY in out and WELCOME not in out, out

    def test_num_find_counts_distinct_states(self, fauxware_project):
        mgr = _explore(fauxware_project, num_find=2)
        assert len(set(mgr._rust_mgr.get_state_ids("found"))) == 2, "a single find was counted twice"

    def test_predicate_sees_stdout(self, fauxware_project):
        seen = []
        _explore(fauxware_project, record=seen)
        assert any(b"Username" in out for out in seen), "find predicate only ever saw empty stdout"


class _FdStub:
    """Stands in for the Rust manager's fd-buffer FFI in the injection test."""

    def __init__(self, out):
        self._out = out

    def has_state_fd_output(self, _sid, fd):
        return fd == 1

    def get_state_fd_output(self, _sid, fd):
        return self._out if fd == 1 else b""


class TestInjectRustFdOutput:
    def test_non_prefix_stream_is_replaced(self, fauxware_project):
        """A Python stream that is not a prefix of Rust's buffer (a frame copied
        from a root carrying another branch's output) is replaced, not spliced."""
        mgr = RustExplorationManager(fauxware_project, [fauxware_project.factory.entry_state()])
        state = fauxware_project.factory.entry_state()
        state.posix.stdout.write(None, b"Go away!\n")
        mgr._rust_mgr = _FdStub(b"Username: \nWelcome\n")
        mgr._inject_rust_fd_output(state, 1, 1)
        assert state.posix.dumps(1) == b"Username: \nWelcome\n"

    def test_prefix_stream_gets_only_the_suffix(self, fauxware_project):
        mgr = RustExplorationManager(fauxware_project, [fauxware_project.factory.entry_state()])
        state = fauxware_project.factory.entry_state()
        state.posix.stdout.write(None, b"Username: \n")
        mgr._rust_mgr = _FdStub(b"Username: \nWelcome\n")
        mgr._inject_rust_fd_output(state, 1, 1)
        assert state.posix.dumps(1) == b"Username: \nWelcome\n"


def test_sym_write_single_win_state():
    """sym-write (angr-acg99's bench): one found state, carrying its own
    "you win!" — it used to be listed twice, once with "you lose!"."""
    import os

    import angr
    from angr import claripy

    binary = os.path.expanduser("~/repos/angr-examples/examples/sym-write/issue")
    if not os.path.exists(binary):
        pytest.skip("angr-examples sym-write binary not available")
    proj = angr.Project(binary, load_options={"auto_load_libs": False})
    state = proj.factory.entry_state(add_options={angr.options.SYMBOLIC_WRITE_ADDRESSES})
    u = claripy.BVS("u", 8)
    state.memory.store(0x804A021, u)
    mgr = RustExplorationManager(proj, [state])
    mgr.explore(find=lambda s: b"win" in s.posix.dumps(1), avoid=lambda s: b"lose" in s.posix.dumps(1))
    assert len(mgr.found) == 1
    assert b"you win!" in mgr.found[0].posix.dumps(1)
    assert sorted(mgr.found[0].solver.eval_upto(u, 256)) == sorted(v for v in range(256) if v.bit_count() == 4)
