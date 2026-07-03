# pylint: disable=missing-class-docstring,no-self-use
"""Live parallel wave-loop integration tests (angr-vh834 Phase 5).

These drive the high-level :class:`RustExplorationManager` on the fauxware
multi-path binary once single-threaded (``RUST_PARALLEL_WORKERS`` unset → the
verbatim single-threaded loop) and once with two real worker threads (the
work-stealing wave loop in ``native/.../exploration/run_loop.rs``), and assert
that the parallel path finds the SAME SET OF FEASIBLE PATHS while exposing real
scheduler counters.

Why the comparison is over the *stable* projection (pc + the deterministic
backdoor solution) rather than the raw content fingerprint
(``tests/benchmarks/content_fingerprint.py``):

* fauxware's "correct password" accepting path equates symbolic stdin to a
  symbolic ``passwd`` FILE (``ALL_FILES_EXIST`` creates it symbolic). That file
  is unconstrained, so the accepting path's concrete stdin is picked
  arbitrarily by Z3 — and it varies **even between two single-threaded runs**
  (verified: run-to-run the value changes). Only the backdoor path
  (``strcmp(pw, "SOSNEAKY")``) pins stdin deterministically.
* state migration (``StateMigrationPayload``) faithfully rebuilds the Z3 solver
  (``satisfiable`` / ``eval`` / ``posix.dumps`` are correct) but does NOT
  preserve the Python-exported *assume-class constraint log* — a recovered
  found state shows a different ``state.solver.constraints`` list than its
  single-threaded twin. This is the same pre-existing snapshot round-trip
  fidelity gap tracked by the known-failing
  ``test_dump_load_fauxware_round_trip_preserves_stash_shape``.

Both divergence sources are orthogonal to the wave loop. The order-independent,
migration-robust content guarantee the scheduler itself provides is covered by
the Rust unit test ``exploration::scheduler::tests::test_determinism_result_set``
(pinned-witness multiset equality across detach/steal/reattach). Here we assert
the integration-level invariant: the wave loop discovers the same feasible-path
SET and reproduces the deterministic solution.
"""

from __future__ import annotations

import pytest

# fauxware_project fixture lives in tests/engines/conftest.py (angr-7gdp).
from tests.engines.conftest import RUST_EXPLORATION_AVAILABLE, RustExplorationManager

pytestmark = pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration engine not built")

# Address of `accepted` in the in-repo / angr-examples fauxware (amd64). Reached
# by exactly two feasible paths: the backdoor ("SOSNEAKY") and the computed
# password.
FAUXWARE_ACCEPTED_ADDR = 0x4006ED

# Address of the `read` SimProcedure stub (amd64 fauxware, auto_load_libs=False).
# `read` has no native Rust handler, so reaching it produces a Python-fallback
# SimProcedure *bounce* (`RunResult::SimProcedure` → `NeedsPython`), NOT the
# deferred-fork pre-step route that `accepted` is found through. Used by the
# Bug-C1 regression below: a find target reached via the bounce path.
FAUXWARE_READ_SIMPROC_ADDR = 0x700010


def _path_set(mgr):
    """Stable, migration-robust projection of the found stash.

    Each found state maps to ``(pc, is_backdoor)`` where ``is_backdoor`` is
    whether its (deterministically pinned) stdin carries the ``SOSNEAKY``
    backdoor token. This collapses the nondeterministic computed-password input
    to a single ``(pc, False)`` class while keeping the path COUNT and the
    deterministic backdoor solution observable.
    """
    out = set()
    for st in mgr.found:
        is_backdoor = b"SOSNEAKY" in st.posix.dumps(0)
        out.add((st.addr, is_backdoor))
    return out


def _explore(project, workers, monkeypatch):
    if workers > 1:
        monkeypatch.setenv("RUST_PARALLEL_WORKERS", str(workers))
    else:
        monkeypatch.delenv("RUST_PARALLEL_WORKERS", raising=False)
    mgr = RustExplorationManager(project, [project.factory.entry_state()])
    assert mgr.stats["parallel_real_workers"] == workers
    # num_find == the exact number of feasible accepting paths, so exploration
    # collects the whole found set and terminates (an unbounded num_find would
    # explore the symbolic-file frontier forever).
    mgr.explore(find=FAUXWARE_ACCEPTED_ADDR, num_find=2)
    return mgr


class TestParallelWaveFoundSet:
    def test_found_path_set_matches_single_threaded(self, fauxware_project, monkeypatch):
        """workers=2 finds the same feasible-path SET as workers=1."""
        baseline = _explore(fauxware_project, 1, monkeypatch)
        base_set = _path_set(baseline)
        base_count = len(list(baseline.found))

        parallel = _explore(fauxware_project, 2, monkeypatch)
        par_set = _path_set(parallel)
        par_count = len(list(parallel.found))

        # Both paths to `accepted` are discovered, at the same pc.
        assert base_count == 2, f"single-threaded baseline found {base_count}, expected 2"
        assert par_count == 2, f"parallel found {par_count}, expected 2"
        assert par_set == base_set, f"parallel feasible-path set diverged: parallel={par_set} baseline={base_set}"
        # The deterministic backdoor solution is reproduced by both.
        assert (FAUXWARE_ACCEPTED_ADDR, True) in par_set
        assert (FAUXWARE_ACCEPTED_ADDR, False) in par_set

    def test_parallel_counters_nonzero(self, fauxware_project, monkeypatch):
        """workers=2 exercises real scheduler dispatch + migration."""
        mgr = _explore(fauxware_project, 2, monkeypatch)
        stats = mgr.stats
        assert stats["parallel_real_workers"] == 2
        assert stats["parallel_tasks"] > 0, "wave loop dispatched no tasks"
        # parallel_migrations == surplus_offloaded + materialized_terminals; the
        # two materialized found states alone make this > 0.
        assert stats["parallel_migrations"] > 0, "no states crossed the worker join"

    def test_parallel_path_set_is_stable_across_runs(self, fauxware_project, monkeypatch):
        """The parallel feasible-path SET is itself deterministic run-to-run."""
        run_a = _path_set(_explore(fauxware_project, 2, monkeypatch))
        run_b = _path_set(_explore(fauxware_project, 2, monkeypatch))
        assert (
            run_a
            == run_b
            == {
                (FAUXWARE_ACCEPTED_ADDR, True),
                (FAUXWARE_ACCEPTED_ADDR, False),
            }
        )


def _explore_find(project, find_addr, workers, monkeypatch, num_find=1):
    if workers > 1:
        monkeypatch.setenv("RUST_PARALLEL_WORKERS", str(workers))
    else:
        monkeypatch.delenv("RUST_PARALLEL_WORKERS", raising=False)
    mgr = RustExplorationManager(project, [project.factory.entry_state()])
    assert mgr.stats["parallel_real_workers"] == workers
    mgr.explore(find=find_addr, num_find=num_find)
    return mgr


class TestParallelWaveBounceFind:
    """Bug C1: a find target reached via the SimProcedure/Hook *bounce* path.

    Unlike ``accepted`` (found through the deferred-fork pre-step route — a
    materialized fork whose PC lands on the target, caught by the worker's
    pre-step find check), the ``read`` SimProcedure stub is reached mid-step via
    a CALL whose target is a hooked, non-native address. The interpreter returns
    ``RunResult::SimProcedure`` and ``run_post_step_core`` turns it into a
    ``NeedsPython`` bounce. Single-threaded ``step_one`` short-circuits such a
    bounce to FOUND/AVOID when its address is a find/avoid target (the
    ``NeedCallback`` special case). The parallel coordinator must do the same in
    its bounce routing; without the C1 fix it instead surfaces a spurious Python
    SimProcedure callback and the found state is LOST (``found == []``).
    """

    def test_bounce_find_matches_single_threaded(self, fauxware_project, monkeypatch):
        """workers=2 finds the bounce-reached target exactly like workers=1."""
        baseline = _explore_find(fauxware_project, FAUXWARE_READ_SIMPROC_ADDR, 1, monkeypatch)
        base_addrs = sorted(s.addr for s in baseline.found)

        parallel = _explore_find(fauxware_project, FAUXWARE_READ_SIMPROC_ADDR, 2, monkeypatch)
        par_addrs = sorted(s.addr for s in parallel.found)

        # Single-threaded reaches the bounce target and routes it to FOUND.
        assert base_addrs == [FAUXWARE_READ_SIMPROC_ADDR], (
            f"single-threaded baseline did not find the bounce target: {[hex(a) for a in base_addrs]}"
        )
        # Parallel must match. Pre-C1-fix this is [] (the found state is lost to a
        # spurious Python bounce), so this assertion fails without the fix.
        assert par_addrs == base_addrs, (
            f"parallel bounce-find diverged: parallel={[hex(a) for a in par_addrs]} "
            f"baseline={[hex(a) for a in base_addrs]}"
        )


def _explore_steady(project, monkeypatch, num_find=2):
    """Explore fauxware under the steady-state loop (angr-nkoct).

    Only sets ``RUST_PARALLEL_STEADY`` + workers; the DRIVER engages the loop by
    setting the frontier-residency flag for this address-based, no-``until``
    exploration (Phase C2). No manual ``set_parallel_frontier_residency`` — so
    these tests exercise the real driver wiring end to end.
    """
    monkeypatch.setenv("RUST_PARALLEL_WORKERS", "2")
    monkeypatch.setenv("RUST_PARALLEL_STEADY", "1")
    mgr = RustExplorationManager(project, [project.factory.entry_state()])
    mgr.explore(find=FAUXWARE_ACCEPTED_ADDR, num_find=num_find)
    return mgr


class TestParallelSteady:
    """Steady-state loop (angr-nkoct): frontiers resident across the
    Python-callback boundary, engaged automatically by the driver for
    address-based exploration under RUST_PARALLEL_STEADY=1 (Phase C1 + C2)."""

    def test_steady_found_set_matches_single_threaded(self, fauxware_project, monkeypatch):
        base = _path_set(_explore(fauxware_project, 1, monkeypatch))
        steady = _path_set(_explore_steady(fauxware_project, monkeypatch))
        assert steady == base, f"steady found-set diverged: steady={steady} baseline={base}"
        assert steady == {
            (FAUXWARE_ACCEPTED_ADDR, True),
            (FAUXWARE_ACCEPTED_ADDR, False),
        }

    def test_steady_bounce_and_resume_counters(self, fauxware_project, monkeypatch):
        """The steady bounce/resume protocol is exercised: both previously
        always-zero counters go positive when a Python SimProcedure bounce is
        serviced and its successors re-injected."""
        mgr = _explore_steady(fauxware_project, monkeypatch)
        stats = mgr.stats
        assert stats["parallel_real_workers"] == 2
        assert stats["parallel_bounce_roundtrips"] > 0, "no bounce round-tripped in steady mode"
        assert stats["parallel_resume_reinjects"] > 0, "no resumed state re-injected in steady mode"

    def test_steady_path_set_stable_across_runs(self, fauxware_project, monkeypatch):
        run_a = _path_set(_explore_steady(fauxware_project, monkeypatch))
        run_b = _path_set(_explore_steady(fauxware_project, monkeypatch))
        assert (
            run_a
            == run_b
            == {
                (FAUXWARE_ACCEPTED_ADDR, True),
                (FAUXWARE_ACCEPTED_ADDR, False),
            }
        )
