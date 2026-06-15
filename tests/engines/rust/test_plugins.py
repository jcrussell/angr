"""Tests for RustExplorationManager.

This module tests the Rust-native exploration manager for symbolic execution.
"""

from __future__ import annotations

import pytest

# Rust availability guard, binary-path resolution, and the module-scoped
# fauxware_project fixture all live in tests/engines/conftest.py (angr-7gdp).
from tests.engines.conftest import (  # noqa: F401
    RUST_EXPLORATION_AVAILABLE,
    TEST_BINARIES_DIR,
    ExplorationEvent,
    PythonCallbacks,
    RustExplorationManager,
    RustSimState,
    _RustExplorationManager,
)

# All tests in this module require the Rust extension; skip the whole module
# when it is unavailable (matches tests/engines/test_rust_public_api.py).
pytestmark = pytest.mark.skipif(
    not RUST_EXPLORATION_AVAILABLE,
    reason="Rust exploration not available",
)


class TestPluginMutationAcrossCallbacks:
    """Regressions for angr-qm7w lazy ``_state_cache`` populate/evict.

    A SimProcedure / Python hook callback may mutate a plugin (most commonly
    ``state.globals`` / ``state.posix``); a second callback that fires on the
    *same* state must observe the mutation. Pre-qm7w this was implicitly
    guaranteed because the callback exit wrote ``_state_cache[sid] = succ_state``
    and the next callback re-used the same SimState instance. The lazy-cache
    refactor evicts entries between callbacks to bound cache size, so the
    plugin-mutation chain has to survive the eviction explicitly.
    """

    def _hook_two_addrs(self, proj, addr_a, addr_b, hook_a, hook_b):
        """Install hook_a and hook_b at the given addresses, returning an
        unhook callback the caller invokes from a ``finally:`` block."""
        proj.hook(addr_a, hook=hook_a, length=0)
        proj.hook(addr_b, hook=hook_b, length=0)

        def _unhook():
            proj.unhook(addr_a)
            proj.unhook(addr_b)

        return _unhook

    def test_globals_mutation_visible_in_second_callback(self, fauxware_project):
        """Callback A writes ``state.globals['qm7w_marker']``; callback B at a
        later address on the same state path must read the same value back.

        Hooks at fauxware main prologue (mov rbp, rsp at 0x40071e and
        mov dword ptr [rbp-0x34], edi at 0x400725) are guaranteed to fire
        sequentially on the same state because no symbolic branching happens
        between them.
        """

        proj = fauxware_project
        main_sym = proj.loader.find_symbol("main")
        assert main_sym is not None
        addr_a = main_sym.rebased_addr + 1  # 0x40071e: mov rbp, rsp
        addr_b = main_sym.rebased_addr + 8  # 0x400725: mov dword ptr [rbp-0x34], edi

        observations = []

        def hook_a(state):
            state.globals["qm7w_marker"] = "written_in_A"

        def hook_b(state):
            observations.append(state.globals.get("qm7w_marker", "<MISSING>"))

        unhook = self._hook_two_addrs(proj, addr_a, addr_b, hook_a, hook_b)
        try:
            state = proj.factory.entry_state()
            mgr = RustExplorationManager(proj, [state])
            mgr.run(max_steps=200)
        finally:
            unhook()

        assert observations, "hook B never fired — could not exercise the chain"
        assert all(obs == "written_in_A" for obs in observations), (
            f"Plugin mutation written by hook A was not visible in hook B. "
            f"observations={observations!r} — the second callback fell back to "
            f"a state without A's globals mutation, which means lazy "
            f"_state_cache eviction lost the plugin chain."
        )

    def test_posix_set_fd_visible_in_second_callback(self, fauxware_project):
        """Callback A calls ``state.posix.set_fd(99, ...)``; callback B reads
        ``state.posix.get_fd(99)`` and must see the same SimFile back.

        Mirrors the bd description's specific example
        (``state.posix.set_fd``). Globals is a thin dict-plugin; posix carries
        nested SimPacketsStream weakrefs and is the more dangerous failure
        mode if the plugin chain breaks.
        """
        from angr.storage import SimFile

        proj = fauxware_project
        main_sym = proj.loader.find_symbol("main")
        addr_a = main_sym.rebased_addr + 1
        addr_b = main_sym.rebased_addr + 8

        from angr.storage.file import SimFileDescriptor

        sentinel_name = "qm7w_sentinel_fd"
        observations = []

        def hook_a(state):
            simfile = SimFile(sentinel_name, content=b"qm7w_data")
            state.fs.insert(sentinel_name, simfile)
            simfd = SimFileDescriptor(simfile, 0)
            simfd.set_state(state)
            state.posix.fd[99] = simfd

        def hook_b(state):
            fd_obj = state.posix.fd.get(99)
            observations.append(getattr(getattr(fd_obj, "file", None), "name", None))

        unhook = self._hook_two_addrs(proj, addr_a, addr_b, hook_a, hook_b)
        try:
            state = proj.factory.entry_state()
            mgr = RustExplorationManager(proj, [state])
            mgr.run(max_steps=200)
        finally:
            unhook()

        assert observations, "hook B never fired — could not exercise the chain"
        assert all(obs == sentinel_name for obs in observations), (
            f"posix.set_fd(99, ...) written by hook A was not visible to "
            f"hook B (observations={observations!r}). Lazy _state_cache "
            f"eviction lost the posix-plugin chain."
        )


class TestStateCacheSizeBound:
    """Regressions for angr-qm7w lazy ``_state_cache`` populate/evict.

    The acceptance criterion says the cache size at any moment during a
    1000-state exploration is bounded by ``concurrent_callbacks`` (typically
    ≤ 2) plus a small fixed overhead for root states. We approximate this with
    a heavy fork-rate run and assert the cache stays well below the old
    ``_max_state_cache_size = 500`` cap.
    """

    def test_cache_size_stays_bounded_under_forking(self, fauxware_project):
        """Run fauxware with no callbacks and confirm ``_state_cache`` does
        not balloon past a tight bound. Pre-qm7w the cache grew on every
        symbolic-branch fork (one full state.copy() per new fork id) and only
        trimmed back down at the 500-entry cap. The refactor removes the
        per-fork copy, so the cache size should track ``# root states +
        # in-flight callbacks``.
        """

        proj = fauxware_project
        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])

        peak_cache = 0
        for _ in range(20):
            mgr.run(max_steps=15)
            peak_cache = max(peak_cache, len(mgr._state_cache))
            if not mgr._rust_mgr.has_active_states():
                break

        # Bound: ``_max_state_cache_size`` is now 8; the cleanup pass evicts
        # non-pinned entries past that.  Pre-fix the cache could grow to 500.
        # We allow a small slop (cap + 4) for transient post-callback writes
        # that happen between cleanups in a single run() batch.
        cap = mgr._max_state_cache_size
        assert peak_cache <= cap + 4, (
            f"_state_cache grew to {peak_cache} entries during fauxware "
            f"exploration (cap={cap}) — lazy populate/evict regressed."
        )


class TestPluginTemplateSelection:
    """Regressions for angr-2k64: ``_restore_plugins_to_state`` must never
    pick a plugin template from an arbitrary cached state.

    Before the fix, the function fell back to ``next(iter(_state_cache.values()))``
    when the tracked root for a given ``state_id`` was not cached. That arbitrary
    pick could land on a forked descendant whose plugin state held mutations
    (open fds, heap allocations, fs entries) belonging to an unrelated path —
    silently cross-pollinating exploration branches. The fix walks
    ``state_id → snapshot.parent_id → tracked root → any cached root`` and
    refuses to fall through to a non-root descendant.
    """

    def _make_state_with_marker_fd(self, proj, fd_num, name):
        """Build an entry state and stamp a distinctive SimFile on ``posix.fd``.

        The fd entry is the divergence marker — restoration that uses a
        wrong template will leak this entry into an unrelated state.
        """
        from angr.storage import SimFile
        from angr.storage.file import SimFileDescriptor

        state = proj.factory.entry_state()
        simfile = SimFile(name, content=b"marker_data")
        state.fs.insert(name, simfile)
        simfd = SimFileDescriptor(simfile, 0)
        simfd.set_state(state)
        state.posix.fd[fd_num] = simfd
        return state

    def test_returns_state_itself_when_cached(self, fauxware_project):
        """If ``state_id`` is in ``_state_cache``, that state must be the
        template — its own plugins are by definition correct."""

        proj = fauxware_project
        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])

        # The entry state's id is its own root.
        root_id = next(iter(mgr._state_roots))
        cached = mgr._state_cache[root_id]

        template = mgr._find_plugin_template_state(root_id)
        assert template is cached, (
            "Expected _find_plugin_template_state to return the cached "
            f"state itself for state_id={root_id}, got {template!r}"
        )

    def test_falls_back_to_parent_id_when_state_not_cached(self, fauxware_project):
        """When ``state_id`` is uncached but its snapshot parent IS cached,
        the parent must be the template."""

        proj = fauxware_project
        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])
        parent_id = next(iter(mgr._state_roots))
        parent_state = mgr._state_cache[parent_id]

        # Synthesize an uncached descendant id.
        uncached_id = parent_id + 99999
        template = mgr._find_plugin_template_state(uncached_id, snapshot_parent_id=parent_id)
        assert template is parent_state, (
            "Expected parent state as template when state_id is uncached "
            f"and snapshot_parent_id points at the cached parent; got {template!r}"
        )

    def test_never_picks_non_root_descendant_as_fallback(self, fauxware_project):
        """The old fallback was ``next(iter(_state_cache.values()))``, which
        could land on a forked descendant with mutated plugins (e.g.,
        ``posix.fd[99]`` from another branch). The fix limits the
        last-resort fallback to *root* states only — non-root descendants
        in the cache must never be returned as a template.
        """

        proj = fauxware_project
        root_state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [root_state])
        root_id = next(iter(mgr._state_roots))

        # Inject a fake forked-descendant SimState into _state_cache without
        # registering it as a root. Stamp it with a marker fd so we can detect
        # cross-pollination if the fallback ever picks it.
        descendant_id = root_id + 12345
        descendant = self._make_state_with_marker_fd(proj, 99, "2k64_descendant")
        mgr._state_cache[descendant_id] = descendant
        # Critically: _state_roots does NOT contain descendant_id (it's a
        # forked descendant, not a root).

        # Now temporarily evict the root so the only cached state is the
        # mutated descendant. Pre-fix this would force "first cached" to
        # return `descendant`.
        original_root_state = mgr._state_cache.pop(root_id)
        try:
            # Synthesize a third state_id with no known root in the cache.
            orphan_id = descendant_id + 54321
            template = mgr._find_plugin_template_state(orphan_id)
            assert template is None or template is not descendant, (
                "Plugin template fallback returned a non-root descendant "
                f"({descendant!r}) whose posix.fd[99] would leak into the "
                "unrelated state being restored. Old 'first cached' fallback "
                "regressed."
            )
        finally:
            mgr._state_cache[root_id] = original_root_state

    def test_falls_back_to_root_when_descendant_id_is_unknown(self, fauxware_project):
        """If the orphan state's tracked root is cached, return it; the
        empty-baseline root plugins are the safe default."""

        proj = fauxware_project
        root_state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [root_state])
        root_id = next(iter(mgr._state_roots))
        root_cached = mgr._state_cache[root_id]

        # Synthesize an uncached orphan id and register its root.
        orphan_id = root_id + 4242
        mgr._state_roots[orphan_id] = root_id

        template = mgr._find_plugin_template_state(orphan_id)
        assert template is root_cached, f"Expected the tracked-root state as the template fallback, got {template!r}"

    def test_returns_none_when_no_root_cached(self, fauxware_project):
        """No cached ancestor and no cached root → return None (skip plugin
        restore rather than guess). Better to leave the state with angr's
        default plugins than to leak mutations from an unrelated branch.
        """

        proj = fauxware_project
        root_state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [root_state])
        # Drain everything that could match.
        mgr._state_cache.clear()
        mgr._state_roots.clear()

        # Use a state_id Rust definitely doesn't know about so get_state_root
        # returns None.
        template = mgr._find_plugin_template_state(0xDEAD_BEEF_CAFE)
        assert template is None, f"Expected None when nothing is cached, got {template!r}"


class TestAdversarial:
    """Adversarial tests: edge cases, API misuse, resource bounds."""

    @classmethod
    def setup_class(cls):
        """Ensure the shared Z3 context is initialized for solver tests."""
        from angr.exploration.rust_manager import _setup_shared_z3_context

        _setup_shared_z3_context()

    # --- API misuse ---

    def test_create_state_invalid_stash(self):
        """Creating state in nonexistent stash should work (dynamic stash)."""
        mgr = _RustExplorationManager("amd64")
        sid = mgr.create_state("nonexistent_stash_42")
        counts = mgr.stash_counts()
        assert counts.get("nonexistent_stash_42", 0) == 1

    def test_set_find_empty_list(self):
        """Setting empty find/avoid lists should not crash."""
        mgr = _RustExplorationManager("amd64")
        mgr.set_find_addrs([])
        mgr.set_avoid_addrs([])

    def test_set_find_duplicate_addresses(self):
        """Duplicate find/avoid addresses should be handled."""
        mgr = _RustExplorationManager("amd64")
        mgr.set_find_addrs([0x1000, 0x1000, 0x1000])
        mgr.set_avoid_addrs([0x2000, 0x2000])

    def test_run_with_no_callbacks(self):
        """Running without callbacks set should raise RuntimeError."""
        mgr = _RustExplorationManager("amd64")
        with pytest.raises(RuntimeError, match="callbacks not set"):
            mgr.run(10)

    # --- State operations ---

    def test_state_register_unknown(self):
        """Getting unknown register should raise ValueError."""
        state = RustSimState("amd64")
        with pytest.raises((ValueError, RuntimeError)):
            state.get_register("nonexistent_register_xyz")

    def test_state_register_zero_value(self):
        """Zero is a valid register value."""
        state = RustSimState("amd64")
        state.set_register("rax", 0)
        assert state.get_register("rax") == 0

    def test_state_register_max_value(self):
        """Maximum 64-bit value should be preserved."""
        state = RustSimState("amd64")
        state.set_register("rax", 0xFFFFFFFFFFFFFFFF)
        assert state.get_register("rax") == 0xFFFFFFFFFFFFFFFF

    def test_state_pc_zero(self):
        """PC=0 is valid (common null-pointer case)."""
        state = RustSimState("amd64")
        state.pc = 0
        assert state.pc == 0

    def test_state_pc_max(self):
        """Maximum address should work."""
        state = RustSimState("amd64")
        state.pc = 0xFFFFFFFFFFFFFFFF
        assert state.pc == 0xFFFFFFFFFFFFFFFF

    def test_state_double_fork(self):
        """Fork of a fork should work."""
        s1 = RustSimState("amd64")
        s1.set_register("rax", 1)
        s2 = s1.fork()
        s2.set_register("rax", 2)
        s3 = s2.fork()
        s3.set_register("rax", 3)
        assert s1.get_register("rax") == 1
        assert s2.get_register("rax") == 2
        assert s3.get_register("rax") == 3

    def test_state_many_forks(self):
        """Many forks should not crash (tests CoW efficiency)."""
        state = RustSimState("amd64")
        state.set_register("rax", 42)
        forks = [state.fork() for _ in range(100)]
        for i, f in enumerate(forks):
            f.set_register("rax", i)
        assert state.get_register("rax") == 42
        for i, f in enumerate(forks):
            assert f.get_register("rax") == i

    # --- Solver edge cases ---

    def test_solver_empty_constraints(self):
        """Solver with no constraints should be satisfiable."""
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        assert ctx.satisfiable()

    def test_solver_contradictory_constraints(self):
        """UNSAT detection: x==5 && x==10 is correctly reported unsatisfiable.

        Characterization (see bd memory characterization-vs-fix-pattern): the
        Rust solver routes equality constraints through Z3, which detects the
        contradiction, so ``satisfiable()`` returns False. An earlier docstring
        claimed add_constraint_ast might NOT preserve == semantics — that is no
        longer the observed behavior. This assertion pins the current contract
        so a regression back to "spuriously SAT" is caught.
        """
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 5)
        ctx.add_constraint_ast(x == 10)
        assert ctx.satisfiable() is False

    def test_solver_wide_bitvector(self):
        """Wide bitvector (256-bit) should work."""
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        x = claripy.BVS("wide", 256)
        ctx.add_constraint_ast(claripy.Extract(7, 0, x) == 0x42)
        assert ctx.satisfiable()

    # --- PyO3 boundary: None / empty / very-wide RustBV (angr-my1w) ---

    def test_eval_handle_invalid_id_returns_none(self):
        """eval_handle for an unknown handle id maps Option::None → Python None."""
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        # Pick an id that no handle was ever issued for.
        assert ctx.eval_handle(99999) is None

    def test_create_concrete_zero_width_round_trip(self):
        """Width-0 RustBV survives the PyO3 round-trip via create_concrete.

        The concrete-value mask collapses to 0 for width=0, so the value
        is normalized to 0; the handle still reports width=0 and is_concrete.
        """
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        h = ctx.create_concrete(0, 0)
        assert h.width == 0
        assert h.is_concrete is True
        assert h.concrete() == 0

    def test_solver_very_wide_bitvector_round_trip(self):
        """1024-bit BV survives add_constraint + eval round-trip across PyO3.

        Wide values (>128 bits) take the eval_wide path that returns a
        Python int reconstructed from big-endian bytes. The low byte
        constraint must be reflected in the evaluated value.
        """
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        x = claripy.BVS("wide1k", 1024)
        ctx.add_constraint_ast(claripy.Extract(7, 0, x) == 0x42)
        assert ctx.satisfiable()
        v = ctx.eval(x)
        assert v is not None
        assert v.bit_length() <= 1024
        assert (v & 0xFF) == 0x42

    def test_wide_concrete_bvv_import_rejected(self):
        """angr-cxw7: a concrete BVV whose value needs >128 bits must error
        loudly, not silently truncate to its low 128 bits.

        ``RustBV::Concrete`` stores its value in a u128, so importing
        ``BVV(1 << 200, 256)`` used to wrap to 0. extract_int_value now
        rejects it with a RuntimeError (BridgeError::InvalidArgs). The import
        path is exercised here via ``min`` (which propagates the conversion
        error; ``eval``/``add_constraint_ast`` take Z3-export fast paths that
        bypass the u128 import).
        """
        import claripy
        import pytest
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        with pytest.raises(RuntimeError):
            ctx.min(claripy.BVV(1 << 200, 256), signed=False)

        # A 256-bit literal whose magnitude fits in 128 bits imports fine.
        assert ctx.min(claripy.BVV(1 << 100, 256), signed=False) == (1 << 100)

    def test_wide_bv_extrema_returns_none(self):
        """angr-cxw7: min/max on a >128-bit symbolic BV returns None rather
        than a value truncated to the u128 binary-search range.

        128-bit BVs still return a real extremum (range is exactly
        [0, u128::MAX]); only widths strictly above 128 are unknown.
        """
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        x = claripy.BVS("xwide", 256)
        ctx.add_constraint_ast(claripy.Extract(7, 0, x) == 0x42)
        assert ctx.satisfiable()
        assert ctx.min(x, signed=False) is None
        assert ctx.max(x, signed=False) is None

        # A 128-bit BV is still handled (fits in u128).
        ctx2 = RustSolverContext()
        y = claripy.BVS("y128", 128)
        ctx2.add_constraint_ast(claripy.UGE(y, 5))
        ctx2.add_constraint_ast(claripy.ULE(y, 9))
        assert ctx2.min(y, signed=False) == 5
        assert ctx2.max(y, signed=False) == 9

    # --- Exploration manager with states ---

    def test_many_states_in_stash(self):
        """Many states in a stash should work."""
        mgr = _RustExplorationManager("amd64")
        for i in range(50):
            sid = mgr.create_state("active")
        assert mgr.active_count() == 50

    def test_stash_counts_empty(self):
        """Empty manager should report zero counts."""
        mgr = _RustExplorationManager("amd64")
        counts = mgr.stash_counts()
        assert counts.get("active", 0) == 0
        assert counts.get("found", 0) == 0

    def test_move_state_nonexistent(self):
        """Moving from an empty source stash is a no-op: counts stay zero, no raise."""
        mgr = _RustExplorationManager("amd64")
        before = mgr.stash_counts()
        # Contract: moving from an empty source stash does nothing and must NOT
        # raise. No try/except — a raise here is now a genuine failure, and the
        # stash counts must be unchanged (both stay at zero).
        mgr.move_states("active", "found", None)
        after = mgr.stash_counts()
        assert before.get("active", 0) == after.get("active", 0) == 0
        assert before.get("found", 0) == after.get("found", 0) == 0

    # --- Integration: RustExplorationManager Python wrapper ---

    def test_python_wrapper_no_states(self, fauxware_project):
        """Python wrapper with empty state list."""
        mgr = RustExplorationManager(fauxware_project, [])
        assert len(mgr.active) == 0

    def test_python_wrapper_explore_no_find(self, fauxware_project):
        """Explore with no find addresses should terminate on active_empty."""
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(max_steps=10)
        # Should not crash, should have run some steps

    def test_python_wrapper_double_explore(self, fauxware_project):
        """Calling explore() twice should not crash."""
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ED, max_steps=5)
        # Run again — should continue from where it left off
        mgr.explore(find=0x4006ED, max_steps=5)
