"""Tests for the out-of-band ``step_state()`` pymethod (E1.a, angr-op0dn.1.1).

``step_state(state_id, extra_stop_points)`` steps ONE state and hands its
successors back bucketed by category, parked in the ``_step_out`` quarantine
stash — nothing is auto-stashed into ``active``/``deadended``/..., because the
Python caller (E1.b's ``SimulationManager`` successor-dict proxy) owns
placement.
"""

from __future__ import annotations

import claripy
import pytest

from tests.engines.conftest import (  # noqa: F401
    RUST_EXPLORATION_AVAILABLE,
    RustExplorationManager,
    fauxware_project,
)

pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)

# fauxware/amd64 landmarks.
MAIN = 0x40071D
# Block boundary inside main's first block flow: the PLT entry main calls.
PLT_PUTS = 0x400510
# Where main's first step lands with no stop point (past the PLT call).
MAIN_NEXT = 0x40073E
# `test eax, eax; je 0x4007c9` — forks when EAX is symbolic.
AUTH_CHECK = 0x4007B3
AUTH_TAKEN = 0x4007C9
AUTH_FALLTHROUGH = 0x4006FB


def _rust_mgr_at(project, addr, symbolic_rax=False):
    """Seed a Rust manager with one blank state at ``addr``; return (mgr, state_id)."""
    state = project.factory.blank_state(addr=addr)
    if symbolic_rax:
        state.regs.rax = claripy.BVS("authed", 64)
    mgr = RustExplorationManager(project, [state])
    rust_mgr = mgr._rust_mgr
    (state_id,) = rust_mgr.get_state_ids("active")
    return mgr, rust_mgr, state_id


class TestStepState:
    """Per-call stop points, successor bucketing, and the no-auto-stash contract."""

    def test_step_returns_flat_successor_without_stashing(self, fauxware_project):
        _mgr, rust_mgr, state_id = _rust_mgr_at(fauxware_project, MAIN)

        buckets = rust_mgr.step_state(state_id, None)

        assert sorted(buckets) == ["flat"]
        assert [s.pc for s in buckets["flat"]] == [MAIN_NEXT]
        # The successor is parked, not placed: active is empty and the state
        # lives in the quarantine stash until the caller moves it.
        counts = rust_mgr.stash_counts()
        assert counts["active"] == 0
        assert counts["_step_out"] == 1

    def test_extra_stop_point_stops_at_that_address(self, fauxware_project):
        _mgr, rust_mgr, state_id = _rust_mgr_at(fauxware_project, MAIN)

        buckets = rust_mgr.step_state(state_id, [PLT_PUTS])

        # Without the stop point this step runs through the PLT call to
        # MAIN_NEXT (see the test above); with it, execution breaks AT the
        # stop address.
        assert [s.pc for s in buckets["flat"]] == [PLT_PUTS]

    def test_extra_stop_point_is_per_call_only(self, fauxware_project):
        # Two identical states in one manager: stepping the first WITH the stop
        # point must not leak it into the manager-level stop set the second one
        # steps under.
        states = [fauxware_project.factory.blank_state(addr=MAIN) for _ in range(2)]
        mgr = RustExplorationManager(fauxware_project, states)
        rust_mgr = mgr._rust_mgr
        first, second = rust_mgr.get_state_ids("active")

        stopped = rust_mgr.step_state(first, [PLT_PUTS])
        unstopped = rust_mgr.step_state(second, None)

        assert [s.pc for s in stopped["flat"]] == [PLT_PUTS]
        assert [s.pc for s in unstopped["flat"]] == [MAIN_NEXT]

    def test_symbolic_branch_returns_two_flat_successors(self, fauxware_project):
        _mgr, rust_mgr, state_id = _rust_mgr_at(fauxware_project, AUTH_CHECK, symbolic_rax=True)

        buckets = rust_mgr.step_state(state_id, None)

        assert sorted(s.pc for s in buckets["flat"]) == sorted([AUTH_FALLTHROUGH, AUTH_TAKEN])
        assert rust_mgr.stash_counts()["_step_out"] == 2

    def test_caller_places_successors_with_move_state(self, fauxware_project):
        _mgr, rust_mgr, state_id = _rust_mgr_at(fauxware_project, AUTH_CHECK, symbolic_rax=True)

        buckets = rust_mgr.step_state(state_id, None)
        for snapshot in buckets["flat"]:
            assert rust_mgr.move_state(snapshot.state_id, "_step_out", "active")

        counts = rust_mgr.stash_counts()
        assert counts["active"] == 2
        assert counts["_step_out"] == 0

    def test_unknown_state_id_raises(self, fauxware_project):
        _mgr, rust_mgr, _state_id = _rust_mgr_at(fauxware_project, MAIN)

        with pytest.raises(ValueError, match="not found"):
            rust_mgr.step_state(0xDEADBEEF, None)

    def test_callback_bounce_raises_not_implemented(self, fauxware_project):
        # main's `read` SimProcedure bounces to Python; driving that protocol
        # from step_state() is E1.b's job, so the MVP refuses it loudly rather
        # than silently dropping the callback.
        _mgr, rust_mgr, state_id = _rust_mgr_at(fauxware_project, MAIN_NEXT)

        with pytest.raises(NotImplementedError, match="callback"):
            rust_mgr.step_state(state_id, None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
