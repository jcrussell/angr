# pylint: disable=missing-class-docstring,no-self-use
"""Regression test for angr-hv4lt.9: RustSolverFallback must refresh on re-sync.

``RustSolverFallback`` caches a forked Rust Z3 context the first time
eval/min/max/satisfiable is called. ``attach()`` was idempotent via the
``_rust_fallback_attached`` flag: once *some* fallback was patched onto
``state.solver`` it no-opped every subsequent attach. But a re-attach happens
routinely — ``_invalidate_state_export_cache`` (called by step()/run()) clears
``rust_fully_synced`` on every cached state, and the next materialization
re-runs ``_sync_cached_state`` on the SAME state_id / SAME Python object, which
unconditionally calls ``_attach_rust_solver_fallback``. Because the no-op left
the ORIGINAL fallback (with its stale ``_cached_rust_ctx``, forked before the
additional Rust-side stepping) bound, eval() returned progressively staler
answers for any state whose id survived without forking.

The fix makes ``attach()`` refresh the already-bound instance
(``_rebind_for_resync``): drop the cached fork and rewind the replay cursor so
the next solve re-forks from the CURRENT Rust state.
"""

from __future__ import annotations

import claripy
import pytest

import angr
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")


class TestSolverFallbackResync:
    def test_resync_refreshes_stale_cached_fork(self):
        """After a Rust-side constraint on the same state_id, eval must not be stale."""
        proj = angr.load_shellcode(b"\x90\xc3", arch="amd64")
        state = proj.factory.blank_state(addr=proj.entry)
        x = claripy.BVS("x", 32)
        state.regs.eax = x

        mgr = RustExplorationManager(proj, [state])
        sid = mgr._rust_mgr.get_state_ids("active")[0]

        # First eval forks + caches the Rust ctx while x is unconstrained -> 0.
        s1 = mgr.active[0]
        v1 = s1.solver.eval(x)
        assert v1 == 0

        # Simulate a further Rust-side step narrowing the SAME state_id. This
        # constraint is invisible to the Python mirror's constraint list.
        mgr._rust_mgr.add_constraints_to_state(sid, [x > 1_000_000])

        # What step()/run() call before the next materialization.
        mgr._invalidate_state_export_cache()

        # s2 is the same cached Python object bound to the same state_id.
        s2 = mgr.active[0]
        v2 = s2.solver.eval(x)

        # Pre-fix: the stale cached fork still returns 0, violating x > 1_000_000.
        assert v2 > 1_000_000, f"eval returned stale value {v2} that violates the re-synced Rust constraint"

    def test_resync_satisfiable_reflects_new_constraint(self):
        """satisfiable() must also re-fork so it sees a fresh unsat context."""
        proj = angr.load_shellcode(b"\x90\xc3", arch="amd64")
        state = proj.factory.blank_state(addr=proj.entry)
        x = claripy.BVS("x", 32)
        state.regs.eax = x

        mgr = RustExplorationManager(proj, [state])
        sid = mgr._rust_mgr.get_state_ids("active")[0]

        s1 = mgr.active[0]
        assert s1.solver.satisfiable() is True

        # Add a pair of mutually-exclusive constraints on the Rust state -> unsat.
        mgr._rust_mgr.add_constraints_to_state(sid, [x > 100, x < 10])
        mgr._invalidate_state_export_cache()

        s2 = mgr.active[0]
        assert s2.solver.satisfiable() is False, "satisfiable() served a stale (still-sat) cached fork after re-sync"
