# pylint: disable=missing-class-docstring,no-self-use
"""Regression tests for angr-87e56: owner-thread teardown of solver forks.

An owned ``RustSolverContext`` (Owned ``SymContext`` + cloned Z3 solver) is
``#[pyclass(unsendable)]``. Before this fix, under ``RUST_PARALLEL_WORKERS>1`` a
dead fork whose last Python reference was collected on a scheduler worker thread
took its final ``tp_dealloc`` off the owning (coordinator) thread, where pyo3
refuses to drop it — writing an unraisable ``RuntimeError`` (``... is
unsendable, but is being dropped on another thread``) and permanently leaking
the Z3 solver clone.

The fix gives ``RustSolverContext`` an owner-thread-guarded ``close()`` plus a
Python graveyard (``rust_state_proxy._release_owned_ctx`` /
``drain_solver_graveyard``) that proxy finalizers hand contexts to and the
coordinator drains on its own thread. These tests pin the primitive's contract
and assert a workers=4 explore raises no ``unsendable``-drop unraisables.
"""

from __future__ import annotations

import sys

import pytest

import angr  # noqa: F401  (ensures the extension + logging are initialized)
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")

FAUXWARE_ACCEPTED_ADDR = 0x4006ED


class TestSolverContextClose:
    def test_close_empties_owned_fork(self, fauxware_project):
        """``close()`` on the owning thread frees the payload; idempotent."""
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        sid = mgr._rust_mgr.get_state_ids("active")[0]
        ctx = mgr._rust_mgr.fork_state_solver(sid)

        assert ctx.is_closed() is False
        ctx.close()
        assert ctx.is_closed() is True
        # Idempotent: a second close() must not double-free / panic.
        ctx.close()
        assert ctx.is_closed() is True

    def test_graveyard_bury_and_drain(self, fauxware_project):
        """A buried owned context is closed by the coordinator drain."""
        from angr.exploration import rust_state_proxy as rsp

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        sid = mgr._rust_mgr.get_state_ids("active")[0]
        ctx = mgr._rust_mgr.fork_state_solver(sid)

        # Simulate an off-owner finalizer that could not close inline: bury it.
        with rsp._SOLVER_GRAVEYARD_LOCK:
            rsp._SOLVER_GRAVEYARD.append(ctx)
        assert ctx.is_closed() is False

        drained = rsp.drain_solver_graveyard()
        assert drained >= 1
        # Drained on this (owning) thread, so the payload is freed.
        assert ctx.is_closed() is True
        # Graveyard is now empty; a second drain is a no-op.
        assert rsp.drain_solver_graveyard() == 0


class TestParallelNoUnsendableUnraisable:
    def test_workers_explore_no_unsendable_drop(self, fauxware_project, monkeypatch):
        """A workers=4 explore raises no ``unsendable``-drop unraisable.

        Counts unraisable exceptions whose message names the cross-thread
        ``unsendable`` drop. The graveyard keeps dead forks alive until the
        coordinator closes them on the owning thread, so none should fire.
        """
        monkeypatch.setenv("RUST_PARALLEL_WORKERS", "4")

        unsendable_hits = []
        prev_hook = sys.unraisablehook

        def _hook(unraisable):
            exc = unraisable.exc_value
            if exc is not None and "unsendable" in str(exc):
                unsendable_hits.append(str(exc))
            # Defer to the previous hook for visibility / other diagnostics.
            prev_hook(unraisable)

        sys.unraisablehook = _hook
        try:
            state = fauxware_project.factory.entry_state()
            mgr = RustExplorationManager(fauxware_project, [state])
            mgr.explore(find=FAUXWARE_ACCEPTED_ADDR, num_find=2)
            # Force any deferred finalizers to run on this (coordinator) thread.
            import gc

            for _ in range(3):
                gc.collect()
            mgr._drain_solver_graveyard()
        finally:
            sys.unraisablehook = prev_hook

        assert not unsendable_hits, f"unsendable cross-thread drops leaked: {unsendable_hits}"
