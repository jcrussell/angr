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

The same two reasons are why the exhaustive find-all gate added for
angr-op0dn.13.1 (``TestParallelExhaustiveSynthetic``) compares drain COUNTS and
pcs rather than vh834 AST content fingerprints -- see that class's docstring.

Both divergence sources are orthogonal to the wave loop. The order-independent,
migration-robust content guarantee the scheduler itself provides is covered by
the Rust unit test ``exploration::scheduler::tests::test_determinism_result_set``
(pinned-witness multiset equality across detach/steal/reattach). Here we assert
the integration-level invariant: the wave loop discovers the same feasible-path
SET and reproduces the deterministic solution.
"""

from __future__ import annotations

import os

import claripy
import pytest

import angr

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


class TestParallelFindAllParity:
    """angr-op0dn.13.1: the find-ALL (exhaustive) found-set is worker-count
    invariant.

    ``TestParallelWaveFoundSet`` above pins workers=1 vs 2. This widens the
    gate to {1, 2, 4}: a 4-worker pool has strictly more steal/migration
    interleavings than 2 (more workers than fauxware has frontier width, so
    speculation and idle-steal both engage), and a lost or duplicated path
    under a wider pool is the failure mode S7 (angr-op0dn.7) would ship on top
    of.
    """

    @pytest.mark.parametrize("workers", [1, 2, 4])
    def test_found_set_invariant_across_worker_counts(self, fauxware_project, workers, monkeypatch):
        mgr = _explore(fauxware_project, workers, monkeypatch)
        assert len(list(mgr.errored)) == 0, f"workers={workers} produced errored states: {list(mgr.errored)}"
        assert _path_set(mgr) == {
            (FAUXWARE_ACCEPTED_ADDR, True),
            (FAUXWARE_ACCEPTED_ADDR, False),
        }, f"workers={workers} feasible-path set diverged"


# --- Synthetic exhaustive find-all ---------------------------------------
# A W=3 partial-bounce fork/solve variant (8 leaves, 2 mixing rounds, 2 gated
# trap levels, half the leaves bouncing per level) generated by the bench
# generator `synthetic_examples/.../build_pbounce.py --w 3 --s 2 --m 8 --t 2
# --b 1`. The committed W=6 bench is a ~60s measurement bench; this shrunken
# sibling runs in well under a second per worker count, which is what makes an
# exhaustive 3-rep x 3-worker-count sweep affordable in the unit suite.
#
# Why it is the right exhaustive shape: after the width region the leaf index
# `s` is CONCRETE per path, so the gated trap partitions the 8 existing leaves
# with no extra forking, and every leaf's find gate ((acc & 0xff) == 0xee) is
# independently satisfiable. num_find == 8 therefore drains the whole frontier
# with no early exit -- exactly the find-all regime S7 targets, and the one
# where a dropped path (too few) or cancel-token over-collection (too many) is
# observable as a COUNT.
_SYNTH_NAME = "fork_solve_pbounce_W3_S2_M8_B1"
_SYNTH_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "benchmarks",
    "synthetic_examples",
    _SYNTH_NAME,
    _SYNTH_NAME,
)
_SYNTH_LEAVES = 8  # 2^W, W=3


class _TrapPoint(angr.SimProcedure):
    """Identity hook for ``trap_point(x) -> x`` -- a pure Python bounce."""

    def run(self, x):  # pylint: disable=arguments-differ
        return x


@pytest.fixture(scope="module")
def pbounce_project():
    if not os.path.exists(_SYNTH_PATH):
        pytest.skip(f"synthetic bench binary not found: {_SYNTH_PATH}")
    proj = angr.Project(_SYNTH_PATH, auto_load_libs=False)
    trap = proj.loader.find_symbol("trap_point")
    assert trap is not None, "trap_point symbol not found"
    proj.hook(trap.rebased_addr, _TrapPoint())
    return proj


def _explore_pbounce(project, workers, monkeypatch):
    if workers > 1:
        monkeypatch.setenv("RUST_PARALLEL_WORKERS", str(workers))
    else:
        monkeypatch.delenv("RUST_PARALLEL_WORKERS", raising=False)

    main_sym = project.loader.find_symbol("main")
    target = project.loader.find_symbol("reach_target")
    state = project.factory.blank_state(addr=main_sym.rebased_addr)
    # 32 fully-symbolic stdin bytes so the modelled read(0, b, 32) returns
    # symbolic content without an extra fork (same seeding as the bench's
    # solve.py).
    state.posix.stdin.content.append((claripy.BVS("stdin", 32 * 8), claripy.BVV(32, state.arch.bits)))

    mgr = RustExplorationManager(project, [state])
    assert mgr.stats["parallel_real_workers"] == workers
    mgr.explore(find=target.rebased_addr, num_find=_SYNTH_LEAVES, n=4096)
    return mgr, target.rebased_addr


class TestParallelExhaustiveSynthetic:
    """angr-op0dn.13.1: exhaustive find-all on the synthetic, {1,2,4} x 3 reps.

    The projection is the found *multiset of pcs* plus the exact drain count,
    NOT the vh834 AST content fingerprint
    (``tests/benchmarks/content_fingerprint.py``). That gate reads
    ``state.constraints``, and a ``RustExplorationManager`` found state exports
    a nearly-empty Python-side constraint log -- the path condition lives in
    the Rust solver and is reached only through ``eval`` / ``posix.dumps``
    (fauxware found states export exactly ONE constraint; these export zero).
    Fingerprinting that list collapses all 8 leaves to a single fingerprint, so
    set equality across worker counts would pass vacuously. The content the
    Rust engine really does guarantee here is the exhaustive drain: 8 distinct
    leaves reach the gate, each satisfiable independently, so a lost path shows
    up as ``< 8`` and cancel-token over-collection as ``> 8``. The
    order-independent *witness*-multiset guarantee is pinned separately by the
    Rust unit test ``scheduler::tests::test_determinism_result_set``.
    """

    @pytest.mark.parametrize("rep", [0, 1, 2])
    @pytest.mark.parametrize("workers", [1, 2, 4])
    def test_exhaustive_drain_invariant(self, pbounce_project, workers, rep, monkeypatch):
        mgr, target_addr = _explore_pbounce(pbounce_project, workers, monkeypatch)
        found = list(mgr.found)
        assert len(list(mgr.errored)) == 0, f"workers={workers} rep={rep} errored: {list(mgr.errored)}"
        assert len(found) == _SYNTH_LEAVES, (
            f"workers={workers} rep={rep}: exhaustive find-all drained {len(found)} leaves, expected {_SYNTH_LEAVES}"
        )
        assert {s.addr for s in found} == {target_addr}


def _pbounce_state(project):
    """Blank state at ``main`` with 32 symbolic stdin bytes (the 8-leaf synthetic)."""
    main_sym = project.loader.find_symbol("main")
    state = project.factory.blank_state(addr=main_sym.rebased_addr)
    state.posix.stdin.content.append((claripy.BVS("stdin", 32 * 8), claripy.BVV(32, state.arch.bits)))
    return state


def _fingerprint(mgr):
    """Structural search-frontier fingerprint (mirrors ``show_checkpoint_resume.py``).

    Per-stash population plus the sorted PCs of the active stash — neither
    depends on Z3 model latitude, so an exact match across a snapshot round-trip
    proves the search frontier was *restored*, not re-derived.

    PCs are read from Rust (``get_state_pc_by_id``), not from the ``mgr.active``
    proxies: a proxy's ``.addr`` can be served from a stale ``_state_cache``
    entry and lag the real pc by a block (angr-ibx8j), which would make this
    fingerprint report a divergence that does not exist in the snapshot.
    """
    counts = {k: v for k, v in mgr.stash_counts().items() if v}
    active_ids = mgr._rust_mgr.get_state_ids("active")
    return {
        "stash_counts": counts,
        "active_addrs": sorted(hex(mgr._rust_mgr.get_state_pc_by_id(sid)) for sid in active_ids),
    }


def _explore_pbounce_find_k(project, workers, monkeypatch, num_find):
    """``_explore_pbounce`` with a caller-chosen ``num_find`` (< 8 => early cancel)."""
    if workers > 1:
        monkeypatch.setenv("RUST_PARALLEL_WORKERS", str(workers))
    else:
        monkeypatch.delenv("RUST_PARALLEL_WORKERS", raising=False)

    target = project.loader.find_symbol("reach_target")
    mgr = RustExplorationManager(project, [_pbounce_state(project)])
    assert mgr.stats["parallel_real_workers"] == workers
    mgr.explore(find=target.rebased_addr, num_find=num_find, n=4096)
    return mgr, target.rebased_addr


class TestActiveProxyAddrMatchesRust:
    """angr-ibx8j: ``mgr.active[i].addr`` must agree with the Rust-authoritative pc.

    A mid-run callback caches ONE Python frame under several Rust state ids
    (successive callbacks on a lineage reuse the frame; the symbolic-branch fork
    path caches the parent frame under each child id). Before the fix, the first
    materialization synced that shared object in place — binding its
    ``RustRegisterProxy`` to its own id — so every sibling id read back the same,
    wrong pc. A ``num_find`` early-exit leaves exactly such a frontier behind.
    """

    def test_active_addrs_match_rust_pcs(self, pbounce_project, monkeypatch):
        mgr, _ = _explore_pbounce_find_k(pbounce_project, 1, monkeypatch, num_find=3)
        active_ids = list(mgr._rust_mgr.get_state_ids("active"))
        assert active_ids, "num_find=3 should leave an un-explored active frontier"
        rust_pcs = sorted(hex(mgr._rust_mgr.get_state_pc_by_id(sid)) for sid in active_ids)
        proxy_pcs = sorted(hex(s.addr) for s in mgr.active)
        assert proxy_pcs == rust_pcs, f"proxy addrs {proxy_pcs} diverge from Rust pcs {rust_pcs}"


class TestParallelCancelFrontier:
    """angr-op0dn.13.8 (Bug M1): a ``num_find`` early-exit must NOT eat the
    un-explored frontier.

    Reaching ``num_find`` trips the scheduler's ``CancelToken``. Every worker
    stops at its next task boundary; before the fix it dropped its un-dispatched
    worker-local states (and the never-stolen injector surplus) on the floor, so
    the post-explore active stash was a strict subset of the single-threaded
    loop's and the exploration was not resumable. Now both halves are drained
    back to ``STASH_ACTIVE`` (counted by ``parallel_residual_drains``).

    Driven on the 8-leaf synthetic, not fauxware: a find-1 cancel needs a WIDE
    live frontier at the cancel boundary to have anything to lose. Fauxware
    reaches ``accepted`` with an empty worker-local queue, so it cannot observe
    the bug at all.
    """

    @pytest.mark.parametrize("workers", [2, 4])
    def test_frontier_survives_num_find_cancel(self, pbounce_project, workers, monkeypatch):
        """A find-1 early exit still leaves a live active frontier behind.

        This is the end-to-end smoke check; the EXACT conservation proof (every
        child comes back as a processed terminal, a worker-local residual, or an
        injector residual — nothing lost, ``pending`` balanced) is the Rust unit
        test ``scheduler::tests::test_wave_cancel_drains_residual_frontier``,
        which can pin a 40-wide frontier at the cancel boundary. No cheap real
        binary reliably does: on this synthetic the DFS has already dispatched
        most of the tree by the time a leaf is found, so the residual is small
        and its size is timing-dependent — hence no counter assertion here.
        """
        mgr, _ = _explore_pbounce_find_k(pbounce_project, workers, monkeypatch, num_find=1)

        assert len(list(mgr.found)) >= 1
        active = len(list(mgr.active))
        assert active > 0, f"workers={workers} lost the whole frontier on cancel (active={active})"

    @pytest.mark.parametrize("workers", [2, 4])
    def test_cancelled_frontier_is_resumable(self, pbounce_project, workers, monkeypatch):
        """The drained frontier is LIVE: after a find-1 early cancel, resuming the
        same manager to ``num_find=8`` still reaches all 8 leaves. Pre-fix the
        residual frontier is gone, so the resumed explore can never make up the
        difference and the drain count comes up short."""
        mgr, target_addr = _explore_pbounce_find_k(pbounce_project, workers, monkeypatch, num_find=1)
        found_first = len(list(mgr.found))
        assert found_first < _SYNTH_LEAVES, "the first explore must have exited early to test resumability"

        mgr.explore(find=target_addr, num_find=_SYNTH_LEAVES, n=4096)
        found = list(mgr.found)
        assert len(list(mgr.errored)) == 0, f"workers={workers} errored: {list(mgr.errored)}"
        assert len(found) == _SYNTH_LEAVES, (
            f"workers={workers}: resumed find-all drained {len(found)} leaves, expected {_SYNTH_LEAVES} "
            f"(first pass found {found_first})"
        )
        assert {s.addr for s in found} == {target_addr}


class TestParallelCheckpointFrontier:
    """angr-op0dn.13.6 — the drained frontier is CHECKPOINTABLE.

    ``dump_snapshot`` finalizes any live steady session before capturing (Python
    ``_finalize_parallel_session`` + the Rust ``steady_config_guard`` inside
    ``dump_snapshot_bytes``), so a snapshot can never silently omit states that
    are resident in worker Z3 contexts rather than in a stash. Here the dump is
    taken right after a parallel find-1 early cancel — the point where the
    residual frontier is largest and a truncation would be invisible.
    """

    @pytest.mark.parametrize("workers", [1, 4])
    def test_snapshot_after_cancel_round_trips_and_resumes(self, pbounce_project, workers, monkeypatch, tmp_path):
        mgr, target_addr = _explore_pbounce_find_k(pbounce_project, workers, monkeypatch, num_find=1)
        assert not mgr._rust_mgr.parallel_session_active(), "dump-point must have no live session"

        assert len(list(mgr.found)) < _SYNTH_LEAVES, "the explore must have exited early to test resumability"

        snap = tmp_path / "pbounce.snap"
        mgr.dump_snapshot(str(snap))

        # Fingerprint AFTER the dump: dumping is a finalize (angr-op0dn.13.6
        # drains a live steady session; .13.10 flushes the parked parallel
        # bounce queue into `active`), so the pre-dump stashes are not yet the
        # ones the snapshot captures. The contract under test is
        # dump-point == restore-point.
        pre = _fingerprint(mgr)
        assert pre["stash_counts"].get("active", 0) > 0, f"workers={workers}: nothing to checkpoint"

        resumed = RustExplorationManager(pbounce_project, [_pbounce_state(pbounce_project)])
        resumed.load_snapshot(str(snap))

        assert _fingerprint(resumed) == pre, f"workers={workers}: frontier truncated across the snapshot"

        resumed.explore(find=target_addr, num_find=_SYNTH_LEAVES, n=4096)
        found = list(resumed.found)
        assert len(list(resumed.errored)) == 0, f"workers={workers} errored: {list(resumed.errored)}"
        assert {s.addr for s in found} == {target_addr}

        # angr-op0dn.13.12 recalibrated this gate. It used to demand all 8 leaves
        # on the serial arm, and it passed — but for the wrong reason: the resume
        # exhausted the restored frontier, went active_empty, and the angr-027h
        # phase-2 eager retry re-seeded `_initial_seed_states`, i.e. the *pre-load
        # placeholder* the constructor was handed above. That placeholder happens
        # to be a valid pbounce entry state, so phase 2 silently re-ran the whole
        # exploration from scratch and back-filled the leaves the snapshot had
        # actually lost. With phase 2 retired on a resumed manager, the true
        # restored-frontier yield shows through: 4/8 serial. The remaining loss is
        # deferred-fork bookkeeping... or so it looked. angr-op0dn.13.14 then
        # proved the restore is faithful and the deferred frontier complete: 4-6
        # IS the phase-1 ceiling for this dump point, and the 2 leaves it never
        # reaches are UNSAT arrivals the live run prunes too.
        #
        # The range gate stays, and angr-op0dn.13.13 says why it must: this arm
        # dumps after a *parallel* find-1 cancel, and which residual frontier that
        # captures is not deterministic (the CancelToken fires in whichever worker
        # reaches the target first; the others stop at their own task boundaries).
        # A wider residual resumes into more leaves, so the count moves with the
        # dump, not with the resume. `test_resume_from_fixed_snapshot_is_worker_
        # count_invariant` below holds the snapshot bytes fixed and pins the exact
        # invariant the resume side really owes: same leaves, any worker count.
        # The upper bound is still load-bearing — it is the .13.12 fix's own
        # observable, i.e. that a resume can no longer overshoot by replaying the
        # constructor's placeholder seed (the old 9-of-8).
        #
        # angr-vplge: the lower bound applies only to the SERIAL arm. On the
        # parallel arm the residual width at the dump point is set by which worker
        # wins the race to the target and where its peers happened to be — under
        # full-suite load that residual can be narrower than the one this bound was
        # calibrated on, which drops the resumed yield below 4 and fails a test
        # that passes in isolation. Half-the-leaves is a claim about the *dump*, so
        # it can only be asserted where the dump point is deterministic. What the
        # resume side itself owes (a restored frontier that is live, and identical
        # leaves at any worker count) is pinned by the >= 1 floor here plus
        # ``test_resume_from_fixed_snapshot_is_worker_count_invariant``, which holds
        # the snapshot bytes fixed and so has no dump-width variance at all.
        lower = _SYNTH_LEAVES // 2 if workers == 1 else 1
        assert lower <= len(found) <= _SYNTH_LEAVES, (
            f"workers={workers}: resume-from-snapshot drained {len(found)} leaves, "
            f"expected {lower}-{_SYNTH_LEAVES} from dump-point frontier {pre}"
        )

    def test_resume_from_fixed_snapshot_is_worker_count_invariant(self, pbounce_project, monkeypatch, tmp_path):
        """angr-op0dn.13.13 — the resume side is deterministic; the *dump* side is not.

        The bead titled this "leaf count is nondeterministic under CPU contention
        (6-9 of 8)" and read it as the resumed exploration *losing* states to
        worker scheduling. Measurement says otherwise, in two parts.

        First, most of the spread is dump-side, not resume-side. A parallel
        ``num_find=1`` explore trips the CancelToken in whichever worker reaches
        the target first and the others stop at their own task boundaries, so the
        residual frontier drained back to ``active`` — the thing the snapshot
        captures — differs run to run. A wider residual resumes into more leaves.
        That is early-cancel semantics, and it is why the round-trip test above
        keeps a range gate: its dump point is itself nondeterministic.

        Second, holding the snapshot bytes fixed (dumped serially here, so every
        resume below sees identical input), the *serial* resume is bit-stable and
        the parallel one is not — but it varies UPWARD (4 leaves serially, 4-6
        under workers=4), never down. So the parallel arm is not dropping restored
        subtrees; the two arms disagree on how far a deferred frontier drains, and
        angr-op0dn.13.15 owns deciding which is right. What this test pins is the
        invariant that survives either answer: no worker count may resume FEWER
        leaves than the serial drain, and every leaf it does report must be a real,
        independently-satisfiable path.

        The projection is the leaf count plus witness distinctness, not the witness
        bytes: the gate ``(acc & 0xff) == 0xee`` leaves the stdin bytes plenty of
        model latitude, so two solvers walking the same path legitimately report
        different satisfying prefixes.
        """
        # Dump once, serially, so every resume below sees byte-identical input.
        mgr, target_addr = _explore_pbounce_find_k(pbounce_project, 1, monkeypatch, num_find=1)
        snap = tmp_path / "pbounce.snap"
        mgr.dump_snapshot(str(snap))

        def _resume(workers):
            if workers > 1:
                monkeypatch.setenv("RUST_PARALLEL_WORKERS", str(workers))
            else:
                monkeypatch.delenv("RUST_PARALLEL_WORKERS", raising=False)
            resumed = RustExplorationManager(pbounce_project, [_pbounce_state(pbounce_project)])
            resumed.load_snapshot(str(snap))
            resumed.explore(find=target_addr, num_find=_SYNTH_LEAVES, n=4096)
            found = list(resumed.found)
            assert len(list(resumed.errored)) == 0, f"workers={workers} errored: {list(resumed.errored)}"
            assert {s.addr for s in found} == {target_addr}
            # The synthetic branches on the first 3 stdin bytes (W=3), so distinct
            # leaves must carry distinct 3-byte prefixes -- a run that silently
            # re-collects one leaf twice shows up here, not in the count.
            prefixes = {s.posix.dumps(0)[:3] for s in found}
            assert len(prefixes) == len(found), (
                f"workers={workers}: {len(found)} leaves collapsed to {len(prefixes)} stdin witnesses"
            )
            return len(found)

        serial = _resume(1)
        assert serial, "resume drained no leaves; the round-trip test owns that failure"

        for rep in range(2):
            assert _resume(1) == serial, f"rep={rep}: serial resume from a fixed snapshot is not repeatable"
            parallel = _resume(4)
            assert serial <= parallel <= _SYNTH_LEAVES, (
                f"rep={rep}: workers=4 resume drained {parallel} leaves from the same snapshot the "
                f"serial resume drained {serial} from (bound: {serial}..{_SYNTH_LEAVES})"
            )

    def test_resumed_found_states_solve_to_distinct_stdin(self, pbounce_project, monkeypatch, tmp_path):
        """angr-op0dn.13.14: a resumed find must still yield usable inputs.

        The snapshot restores each state's constraints, but those constraints are
        phrased over the *original* manager's harness-seeded stdin BVS. A restored
        state materializes from a bare Rust export, so its posix is angr's default
        empty one and that symbol exists nowhere on the Python side: every found
        state used to dump b"" (or, given some other seed's BVS, the same all-zero
        bytes). The frontier resumed correctly and the answers were still useless.

        The envelope now carries the seed's byte ASTs, so the restored content and
        the restored constraints name the same claripy symbols again.
        """
        mgr, target_addr = _explore_pbounce_find_k(pbounce_project, 1, monkeypatch, num_find=1)
        snap = tmp_path / "pbounce.snap"
        mgr.dump_snapshot(str(snap))

        resumed = RustExplorationManager(pbounce_project, [_pbounce_state(pbounce_project)])
        resumed.load_snapshot(str(snap))
        resumed.explore(find=target_addr, num_find=_SYNTH_LEAVES, n=4096)

        found = list(resumed.found)
        assert found, "resume drained no leaves; the round-trip test owns that failure"

        # The synthetic branches on the first 3 stdin bytes, one per level (W=3),
        # so each distinct leaf must pin a distinct 3-byte prefix. Anything past
        # byte 3 is unconstrained latitude the solver may fill however it likes.
        prefixes = {s.posix.dumps(0)[:3] for s in found}
        assert len(prefixes) == len(found), (
            f"resumed found states collapsed to {len(prefixes)} distinct stdin prefixes "
            f"across {len(found)} leaves: {sorted(prefixes)}"
        )
        assert prefixes != {b"\x00\x00\x00"}, "restored stdin solved unconstrained (the pre-fix symptom)"

    def test_snapshot_round_trips_bucket_d_overlays(self, pbounce_project, monkeypatch, tmp_path):
        """angr-op0dn.13.14: the per-state Python-AST overlays must survive a dump.

        The Rust ``StashManager`` codec cannot serialize the ``Py<PyAny>``
        overlays (``symbolic_pages`` / ``hook_symbolic_memory`` /
        ``addr_to_ast``) and restores them empty. Every pbounce state carries
        them — the values a *Python* SimProcedure handed back live nowhere else
        — so a bare-Rust envelope silently resumed with those addresses
        unconstrained. ``dump_snapshot`` now pickles them alongside the Rust
        bytes; this pins the round-trip.
        """
        mgr, _ = _explore_pbounce_find_k(pbounce_project, 1, monkeypatch, num_find=1)

        def overlays(m):
            core = m._rust_mgr
            return {
                sid: (
                    sorted(core.get_state_symbolic_pages(sid)),
                    sorted(core.get_state_hook_symbolic_memory(sid)),
                    sorted(core.get_state_addr_to_ast(sid)),
                )
                for sid in core.get_state_ids("active")
            }

        snap = tmp_path / "bucket_d.snap"
        mgr.dump_snapshot(str(snap))
        pre = overlays(mgr)
        assert any(any(m) for m in pre.values()), "pre-condition: pbounce states carry Python-AST overlays"

        resumed = RustExplorationManager(pbounce_project, [_pbounce_state(pbounce_project)])
        resumed.load_snapshot(str(snap))
        assert overlays(resumed) == pre, "bucket-D overlays lost across the snapshot round-trip"


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


def _drain_pbounce(project, workers, monkeypatch):
    """Run the 8-leaf synthetic to quiescence (no ``find``, so no cancel).

    Without a find target every path runs to a terminal, so the terminal
    *counters* are a property of the search tree rather than of where a cancel
    token happened to land — the only regime in which the serial and parallel
    arms are comparable at all (see ``snapshot-resume-spread-is-dump-side``).
    """
    if workers > 1:
        monkeypatch.setenv("RUST_PARALLEL_WORKERS", str(workers))
    else:
        monkeypatch.delenv("RUST_PARALLEL_WORKERS", raising=False)
    mgr = RustExplorationManager(project, [_pbounce_state(project)])
    mgr.run(n=4096)
    return mgr


class TestParallelTerminalAccounting:
    """angr-op0dn.13.15 (1): dead paths are *counted* even though they are dropped.

    A parallel worker never serializes a dead path back to the coordinator — it
    records a ``TerminalSummary`` and drops the state in its own Z3 context. That
    is the design (dead paths must not pay the serde tax), but before this the
    summaries died in a scheduler-local atomic, so ``stats()["deadended_count"]``
    read **0** on a parallel run where the serial loop reported 14. The counts now
    ride back with ``SchedulerStats`` and land in the manager's terminal counters.

    The *content* caveat stands and is asserted here: ``stash_counts()`` still
    shows no ``deadended`` stash under parallel, because the states themselves are
    gone. Callers that need the terminal states must run single-threaded.
    """

    @pytest.mark.parametrize("workers", [2, 4])
    def test_deadended_count_survives_the_worker_boundary(self, pbounce_project, workers, monkeypatch):
        serial = _drain_pbounce(pbounce_project, 1, monkeypatch).stats["deadended_count"]
        assert serial > 0, "the serial drain must produce dead paths for this test to mean anything"

        mgr = _drain_pbounce(pbounce_project, workers, monkeypatch)
        assert mgr.stats["parallel_real_workers"] == workers
        assert mgr.stats["deadended_count"] == serial, (
            f"workers={workers}: parallel run counted {mgr.stats['deadended_count']} dead paths, "
            f"serial counted {serial} — summarized terminals are not reaching the manager"
        )

    @pytest.mark.parametrize("workers", [2, 4])
    def test_deadended_states_are_not_recoverable_under_parallel(self, pbounce_project, workers, monkeypatch):
        """The documented content caveat: counted, not stashed."""
        mgr = _drain_pbounce(pbounce_project, workers, monkeypatch)
        assert mgr.stash_counts().get("deadended", 0) == 0


# Ground truth for the 8-leaf synthetic drained to quiescence with no ``find``:
# the vanilla Python `SimulationManager` ends with 14 terminal states, and so
# does the FIRST Rust drain in a fresh process (at every worker count). See
# `parallel-drain-depth-ground-truth`.
_SYNTH_TERMINALS = 14


class TestSecondExplorationInAProcessDiverges:
    """angr-op0dn.13.16: a drain's leaf count must not depend on process history.

    It used to: the first `RustExplorationManager` drain of the synthetic in a
    process ended with 14 dead paths (matching the Python engine) and every
    later drain in the same process ended with 16. Root cause was the symbol-id
    allocator — a per-`SymContext` counter restarting at 0 for each exploration,
    while the registry that resolves an id back to its claripy AST is
    process-global, so a second exploration's symbols aliased the first's. The
    counter is now process-global (`symbolic::bv_id_ops::NEXT_SYMBOL_ID`).
    """

    def test_a_warm_process_drains_the_same_leaf_count_as_a_cold_one(self, pbounce_project, monkeypatch):
        _drain_pbounce(pbounce_project, 1, monkeypatch)  # warm the process
        warm = _drain_pbounce(pbounce_project, 1, monkeypatch).stats["deadended_count"]
        assert warm == _SYNTH_TERMINALS, (
            f"a warm-process serial drain collected {warm} dead paths, but the Python engine "
            f"and a cold-process Rust drain both collect {_SYNTH_TERMINALS}"
        )
