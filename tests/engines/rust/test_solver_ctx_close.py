# pylint: disable=missing-class-docstring,no-self-use
"""Regression tests for angr-87e56 / angr-pxq0i: owner-thread teardown of
solver forks.

An owned ``RustSolverContext`` (Owned ``SymContext`` + cloned Z3 solver) is
``#[pyclass(unsendable)]``. Before angr-87e56, under ``RUST_PARALLEL_WORKERS>1``
a dead fork whose last Python reference was collected on a scheduler worker
thread took its final ``tp_dealloc`` off the owning (coordinator) thread, where
pyo3 refuses to drop it — writing an unraisable ``RuntimeError`` (``... is
unsendable, but is being dropped on another thread``) and permanently leaking
the Z3 solver clone.

angr-87e56 gave ``RustSolverContext`` an owner-thread-guarded ``close()`` plus
a Python graveyard (``rust_state_proxy._release_owned_ctx`` /
``drain_solver_graveyard``) that proxy finalizers hand contexts to and the
coordinator drains on its own thread.

angr-pxq0i: that fix was itself broken. PyO3 0.27.2's codegen for ANY
receiver method on an ``unsendable`` pyclass — including ``close()`` and
``is_closed()`` — runs a cross-thread guard (a hard ``assert_eq!`` panic)
BEFORE the method body, i.e. before ``close()``'s own internal owner check
ever gets a chance to no-op. Since the crate builds with ``panic = "abort"``,
calling ``close()``/``is_closed()`` from any thread but the true owner aborts
the WHOLE PROCESS — not a catchable Python exception. The fix tracks the
owner thread id in plain Python (captured on the thread that actually forks
the context) and checks it via ``threading.get_ident()`` *before* ever
calling a pymethod on an off-thread context, routing straight to the
graveyard instead. ``TestGraveyardOwnerCheck`` pins that contract; the
``test_workers_explore_no_unsendable_drop`` test below still guards the
angr-87e56 leak behavior end-to-end.
"""

from __future__ import annotations

import sys
import threading

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
        owner_tid = threading.get_ident()

        # Simulate an off-owner finalizer that could not close inline: bury it.
        # angr-pxq0i: graveyard entries are (ctx, owner_tid) pairs so a later
        # drain can tell whether it's safe to touch each entry.
        with rsp._SOLVER_GRAVEYARD_LOCK:
            rsp._SOLVER_GRAVEYARD.append((ctx, owner_tid))
        assert ctx.is_closed() is False

        # This thread IS the owner, so the drain closes it for real.
        drained = rsp.drain_solver_graveyard()
        assert drained >= 1
        # Drained on this (owning) thread, so the payload is freed.
        assert ctx.is_closed() is True
        # Graveyard is now empty; a second drain is a no-op.
        assert rsp.drain_solver_graveyard() == 0


class TestGraveyardOwnerCheck:
    """angr-pxq0i: off-owner calls must never touch the Rust object.

    Before this fix, ``_release_owned_ctx``/``drain_solver_graveyard`` called
    ``ctx.close()``/``ctx.is_closed()`` unconditionally and relied on the
    Rust-side owner check inside ``close()`` to gracefully no-op off-thread.
    That assumption was wrong: PyO3's unsendable-pyclass thread guard panics
    (and, under ``panic = "abort"``, aborts the process) before ``close()``'s
    body ever runs. The fix checks ``threading.get_ident()`` against the
    recorded owner BEFORE calling any pymethod, so these in-process checks can
    safely use a deliberately WRONG owner id (simulating "called from another
    thread") without ever reaching the panicking codepath — no real
    cross-thread call is needed to prove the guard fires first.
    """

    def test_release_with_wrong_owner_never_calls_close(self, fauxware_project):
        """A mismatched owner_tid buries the context without calling close()/is_closed().

        ``RustSolverContext`` is a PyO3 pyclass with read-only method slots
        (monkeypatching ``close``/``is_closed`` on an instance raises
        ``AttributeError: ... attribute 'close' is read-only``), so this
        can't spy on the calls directly. Instead it checks the observable
        contract: after an off-owner ``_release_owned_ctx``, the context must
        still report ``is_closed() is False`` (proving ``close()`` never ran)
        and must be sitting in the graveyard.
        """
        from angr.exploration import rust_state_proxy as rsp

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        sid = mgr._rust_mgr.get_state_ids("active")[0]
        ctx = mgr._rust_mgr.fork_state_solver(sid)

        fake_owner_tid = threading.get_ident() + 1  # guaranteed not to match
        rsp._release_owned_ctx(ctx, fake_owner_tid)

        # close() was never reached -- the off-owner branch short-circuits
        # before touching the Rust object at all.
        assert ctx.is_closed() is False
        with rsp._SOLVER_GRAVEYARD_LOCK:
            assert (ctx, fake_owner_tid) in rsp._SOLVER_GRAVEYARD
            rsp._SOLVER_GRAVEYARD.remove((ctx, fake_owner_tid))
        # Clean up for real now that the thread check is done (this thread IS
        # the true owner of ctx, so a direct close() here is safe).
        ctx.close()

    def test_drain_skips_entries_owned_by_another_thread(self, fauxware_project):
        """The drain only closes graveyard entries owned by the current thread."""
        from angr.exploration import rust_state_proxy as rsp

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        sid = mgr._rust_mgr.get_state_ids("active")[0]
        ctx = mgr._rust_mgr.fork_state_solver(sid)

        fake_owner_tid = threading.get_ident() + 1
        with rsp._SOLVER_GRAVEYARD_LOCK:
            rsp._SOLVER_GRAVEYARD.append((ctx, fake_owner_tid))

        # Nothing owned by this thread, so nothing is drained -- and the
        # foreign entry is put back rather than being touched.
        drained = rsp.drain_solver_graveyard()
        assert drained == 0
        assert ctx.is_closed() is False
        with rsp._SOLVER_GRAVEYARD_LOCK:
            assert (ctx, fake_owner_tid) in rsp._SOLVER_GRAVEYARD
            rsp._SOLVER_GRAVEYARD.remove((ctx, fake_owner_tid))
        # Clean up for real now that the thread checks are done.
        ctx.close()

    def test_off_owner_release_does_not_abort_process(self):
        """A real background thread calling the fixed ``_release_owned_ctx``
        on a main-thread-owned context must NOT abort the process.

        This is the direct repro from the bead: before the fix, the exact call
        path here (``_release_owned_ctx`` invoked from a non-owner thread) hit
        PyO3's cross-thread thread-guard panic, which ``panic = "abort"`` turns
        into a hard process abort ("Fatal Python error: Aborted") — not a
        Python exception, so no ``try/except`` in this test process could ever
        catch it or turn it into a normal assertion failure. A wrong fix here
        kills the whole pytest run, not just this test, so the repro runs in
        an isolated subprocess and the parent only checks the subprocess exited
        cleanly and printed the expected marker.
        """
        import os
        import subprocess
        import textwrap

        from tests.engines.conftest import TEST_BINARIES_DIR

        # Mirrors the fauxware_project fixture's own path resolution exactly.
        binary_path = os.path.join(TEST_BINARIES_DIR, "fauxware")
        if not os.path.exists(binary_path):
            pytest.skip("fauxware binary not found")

        script = textwrap.dedent(
            f"""
            import threading
            import angr
            from angr.exploration import RustExplorationManager
            from angr.exploration import rust_state_proxy as rsp

            project = angr.Project({binary_path!r}, auto_load_libs=False)
            state = project.factory.entry_state()
            mgr = RustExplorationManager(project, [state])
            sid = mgr._rust_mgr.get_state_ids("active")[0]
            ctx = mgr._rust_mgr.fork_state_solver(sid)
            owner_tid = threading.get_ident()

            errors = []

            def worker():
                try:
                    # Off-owner call from a real background thread. Before
                    # angr-pxq0i this aborted the process inside close().
                    rsp._release_owned_ctx(ctx, owner_tid)
                except Exception as e:  # pragma: no cover - defensive
                    errors.append(repr(e))

            t = threading.Thread(target=worker)
            t.start()
            t.join(timeout=30)
            assert not t.is_alive(), "worker thread hung"
            assert not errors, f"worker thread raised: {{errors}}"

            # The off-owner call must not have touched the Rust object at all.
            assert ctx.is_closed() is False, "off-owner call touched close()/is_closed()"

            # Back on the owning (main) thread: the drain closes it for real,
            # proving the graveyard-append path worked correctly off-thread.
            drained = rsp.drain_solver_graveyard()
            assert drained == 1, f"expected 1 drained, got {{drained}}"
            assert ctx.is_closed() is True

            print("PASS", flush=True)
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, (
            f"subprocess aborted or failed (returncode={proc.returncode}):\n"
            f"stdout: {proc.stdout!r}\nstderr: {proc.stderr[-4000:]!r}"
        )
        assert "PASS" in proc.stdout, f"subprocess did not report success:\nstdout: {proc.stdout!r}"


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
