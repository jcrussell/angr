"""Tests for RustExplorationManager.

This module tests the Rust-native exploration manager for symbolic execution.
"""

from __future__ import annotations

import warnings

import pytest

import angr

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


class TestClaripyAnnotationRoundtrip:
    """Annotations attached to claripy ASTs must survive a Rust→Python
    roundtrip (constraint export, memory load, eval). See angr-ykdq."""

    def test_uninitialized_annotation_on_bvs_in_memory(self, fauxware_project):
        """A UninitializedAnnotation on a BVS stored in symbolic memory
        must still be present after a full RustExplorationManager run."""
        import claripy
        from claripy.annotation import UninitializedAnnotation

        # SYMBOL_FILL_UNCONSTRAINED_REGISTERS is raised under Rust
        # (angr-apre) — RegisterFile always returns zero so symbolic-fill
        # cannot be honored. This test exercises annotation roundtrip
        # through memory, not registers, so the memory variant is enough.
        state = fauxware_project.factory.entry_state(
            add_options={angr.options.SYMBOL_FILL_UNCONSTRAINED_MEMORY},
        )

        x = claripy.BVS("uninit_x", 32).annotate(UninitializedAnnotation())
        addr = 0x500000
        state.memory.store(addr, x, endness=state.arch.memory_endness)
        state.solver.add(x == 42)

        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.run(max_steps=200)

        all_states = list(mgr.found) + list(mgr.active) + list(mgr.deadended)
        assert all_states, "expected at least one state after run"
        s = all_states[0]
        loaded = s.memory.load(addr, 4)
        assert loaded.has_annotation_type(UninitializedAnnotation), (
            f"UninitializedAnnotation lost on roundtripped BVS in memory; "
            f"loaded={loaded!r}, annotations={loaded.annotations}"
        )

    def test_annotation_on_expression_via_export_constraints(self, fauxware_project):
        """An annotation attached to an Expression node (not on a leaf BVS)
        must survive when constraints are exported from the Rust solver back
        to Python via export_state_constraints. Pre-fix, rustbv_to_claripy
        rebuilt the Expression from BVOp+operands, dropping annotations
        attached at the Expression level. The Arc-keyed expression cache in
        claripy_bridge.rs preserves them."""
        import claripy

        class _ExprTaint(claripy.Annotation):
            def __init__(self, tag):
                self.tag = tag

            @property
            def relocatable(self):
                return True

            @property
            def eliminatable(self):
                return False

            def __hash__(self):
                return hash(("_ExprTaint", self.tag))

            def __eq__(self, other):
                return isinstance(other, _ExprTaint) and self.tag == other.tag

        state = fauxware_project.factory.entry_state()
        # Use a unique BVS name so the assertion can filter to constraints
        # produced by this test (the Rust thread-local cache and global
        # registry persist across tests in the module-scoped fixture).
        bvs_name = "ann_expr_test_x"
        x = claripy.BVS(bvs_name, 32)
        # Annotate the (x + 100) Expression node specifically. Leaves carry
        # no annotations; only the inner Expression does.
        expr_ann = (x + 100).annotate(_ExprTaint("expr-level"))
        state.solver.add(expr_ann > 200)
        state.solver.add(expr_ann < 1000)

        mgr = RustExplorationManager(fauxware_project, [state])
        active_ids = mgr._rust_mgr.get_state_ids("active")
        assert active_ids, "expected active state after manager init"
        sid = active_ids[0]

        exported = mgr._rust_mgr.export_state_constraints(sid)
        exported = [c for c in exported if c is not None]
        # Filter to constraints produced by this test. The fauxware_project
        # fixture is module-scoped and the Rust thread-local Z3 context may
        # carry leaked constraints from earlier tests. Match on op (only this
        # test introduces __gt__/__lt__ on a __add__ subexpression) and on
        # the unique BVS name.
        our_constraints = [
            c
            for c in exported
            if bvs_name in str(c)
            and getattr(c, "op", None) in ("__gt__", "__lt__", "ULT", "ULE", "UGT", "UGE", "SLT", "SGT", "SLE", "SGE")
            and any(getattr(a, "op", None) == "__add__" for a in getattr(c, "args", ()))
        ]
        assert our_constraints, f"no exported constraints match this test's pattern: got {[str(c) for c in exported]}"

        for c in our_constraints:
            assert c.has_annotation_type(_ExprTaint), f"top-level constraint dropped Expression annotation: {c!r}"
            inner_exprs = [a for a in c.args if hasattr(a, "op") and a.op == "__add__"]
            assert inner_exprs, f"expected inner __add__ Expression in {c!r}"
            for inner in inner_exprs:
                assert inner.has_annotation_type(_ExprTaint), (
                    f"inner Expression dropped annotation on roundtrip: {inner!r}, annotations={inner.annotations}"
                )


class TestEdgeCases:
    """End-to-end edge-case tests for RustExplorationManager (angr-32ky).

    These exercise scenarios that have caused bugs historically: non-branching
    binaries, wide symbolic bitvectors, exhausted exploration without find,
    avoid-only configurations, LAZY_SOLVES, re-entrant exploration, and
    store/load roundtrip on the same symbolic address.
    """

    @pytest.fixture(autouse=True)
    def _isolate_class_caches(self):
        """RustExplorationManager._init_cache is class-level and caches the
        post-init state, including user stores added before manager construction.
        Without this clear, user stores from one test bleed into the next via
        the cached state, breaking memory.load assertions. _apply_state_metadata
        copies constraints/options but not memory pages, so the bleed is silent.
        """
        from angr.exploration.rust_manager import RustExplorationManager

        RustExplorationManager._init_cache.clear()
        yield
        RustExplorationManager._init_cache.clear()

    def test_no_branches_single_basic_block_expression_store(self, fauxware_project):
        """Storing an Expression (not a leaf BVS or BVV) into memory and
        loading it back must roundtrip correctly without forking, even when
        the executing block has no conditional branches."""
        import claripy

        state = fauxware_project.factory.entry_state()
        x = claripy.BVS("nobranch_expr_x", 32)
        # Inner Expression: not a leaf — uses __add__ + Concat-equivalent op.
        expr = (x + 0x100) ^ 0xDEADBEEF
        addr = 0x500200
        state.memory.store(addr, expr, endness=state.arch.memory_endness)
        # Single-step the manager: 1 step keeps us inside the entry basic
        # block (fauxware's _start has no branches in the first block).
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.run(max_steps=1)

        # No forks: still a single state in active (or moved to deadended).
        all_states = list(mgr.active) + list(mgr.deadended)
        assert len(all_states) >= 1, "expected at least one state after 1 step"
        s = all_states[0]
        loaded = s.memory.load(addr, 4, endness=s.arch.memory_endness)
        # Solve under x == 0: expected value = (0 + 0x100) ^ 0xDEADBEEF.
        s.solver.add(x == 0)
        assert s.solver.eval(loaded) == ((0x100) ^ 0xDEADBEEF), (
            f"Expression roundtrip via store/load failed; got {s.solver.eval(loaded):#x}"
        )

    def test_wide_symbolic_value_in_memory_256bit(self, fauxware_project):
        """A 256-bit symbolic value stored in memory survives the
        manager-init export/import roundtrip."""
        import claripy

        state = fauxware_project.factory.entry_state()
        wide = claripy.BVS("wide_sym_256", 256)
        addr = 0x500300
        state.memory.store(addr, wide, endness=state.arch.memory_endness)
        # Constrain low byte so we can verify after roundtrip.
        state.solver.add(claripy.Extract(7, 0, wide) == 0x42)

        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.run(max_steps=1)
        all_states = list(mgr.active) + list(mgr.deadended)
        assert all_states, "expected at least one state after 1 step"
        s = all_states[0]
        loaded = s.memory.load(addr, 32, endness=s.arch.memory_endness)
        assert loaded.length == 256
        # Low byte must satisfy the constraint.
        low_byte = s.solver.eval(claripy.Extract(7, 0, loaded))
        assert low_byte == 0x42, f"low byte of 256-bit BV roundtrip failed: {low_byte:#x}"

    def test_wide_symbolic_value_in_memory_512bit(self, fauxware_project):
        """A 512-bit symbolic value stored in memory survives the
        manager-init export/import roundtrip."""
        import claripy

        state = fauxware_project.factory.entry_state()
        wide = claripy.BVS("wide_sym_512", 512)
        addr = 0x500400
        state.memory.store(addr, wide, endness=state.arch.memory_endness)
        state.solver.add(claripy.Extract(7, 0, wide) == 0xAB)
        state.solver.add(claripy.Extract(511, 504, wide) == 0xCD)

        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.run(max_steps=1)
        all_states = list(mgr.active) + list(mgr.deadended)
        assert all_states, "expected at least one state after 1 step"
        s = all_states[0]
        loaded = s.memory.load(addr, 64, endness=s.arch.memory_endness)
        assert loaded.length == 512
        assert s.solver.eval(claripy.Extract(7, 0, loaded)) == 0xAB
        assert s.solver.eval(claripy.Extract(511, 504, loaded)) == 0xCD

    def test_explore_with_zero_find_addresses(self, fauxware_project):
        """explore() with find=None must drain all active states without
        crashing; nothing ends up in found."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        # No find set anywhere — exploration runs until active is empty (or
        # max_steps, whichever first). Use a generous step budget so fauxware
        # naturally terminates in deadended.
        mgr.explore(max_steps=50000)

        assert len(mgr.found) == 0, f"found should be empty with no find, got {len(mgr.found)}"
        # active drains to other stashes — total state count is preserved.
        total = len(mgr.active) + len(mgr.deadended) + len(mgr.avoid) + len(mgr.errored)
        assert total >= 1, f"all states vanished: counts={mgr.stash_counts()}"

    def test_only_avoid_addresses_no_find(self, fauxware_project):
        """An exploration configured with avoid (no find) routes states that
        hit avoid into the avoid stash; remaining states deadend."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        # 0x4006fd is the rejection branch in fauxware. Without find, the
        # explore loop should still dispatch avoid via the address set.
        mgr.explore(avoid=0x4006FD, max_steps=50000)

        assert len(mgr.found) == 0, "no find configured — found must be empty"
        # At least one path through fauxware reaches the rejection branch when
        # auth values are unconstrained.
        assert len(mgr.avoid) >= 1, f"expected at least one avoided state; counts={mgr.stash_counts()}"

    def test_lazy_solves_option_explore(self, fauxware_project):
        """Exploration with LAZY_SOLVES enabled completes and finds the
        target state. LAZY_SOLVES defers the per-branch satisfiability check;
        the Rust engine must still produce a satisfiable found state."""

        state = fauxware_project.factory.entry_state(
            add_options={angr.sim_options.LAZY_SOLVES},
        )
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ED, avoid=0x4006FD, max_steps=50000)

        assert len(mgr.found) >= 1, f"LAZY_SOLVES exploration found nothing; counts={mgr.stash_counts()}"
        # The found state's constraints must still be satisfiable.
        assert mgr.found[0].solver.satisfiable(), "found state under LAZY_SOLVES is not satisfiable"

    def test_multiple_explores_on_same_manager(self, fauxware_project):
        """Re-entrant exploration: calling explore() twice on the same
        manager continues from where the previous call left off without
        resetting found/active stashes."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        # First explore: small step budget — likely doesn't reach target.
        mgr.explore(find=0x4006ED, max_steps=50)
        first_found = len(mgr.found)
        first_total_steps = mgr.stats.get("total_steps", 0)

        # Second explore: continue with a larger budget. Must not crash and
        # must accumulate steps on top of the first call.
        mgr.explore(find=0x4006ED, max_steps=50000)
        second_total_steps = mgr.stats.get("total_steps", 0)

        assert second_total_steps >= first_total_steps, (
            f"step counter regressed across explores: {first_total_steps} -> {second_total_steps}"
        )
        # Re-entrant explore must eventually find the target.
        assert len(mgr.found) >= max(1, first_found), (
            f"found stash regressed across explores; first={first_found}, second={len(mgr.found)}"
        )

    def test_symbolic_store_then_load_same_address(self, fauxware_project):
        """A symbolic value stored at an address and immediately loaded from
        the same address must roundtrip — solver eval under a unique
        assignment yields the assigned value."""
        import claripy

        state = fauxware_project.factory.entry_state()
        addr = 0x500500
        sym = claripy.BVS("rw_same_addr_sym", 64)
        state.memory.store(addr, sym, endness=state.arch.memory_endness)

        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.run(max_steps=1)
        all_states = list(mgr.active) + list(mgr.deadended)
        assert all_states, "expected at least one state after 1 step"
        s = all_states[0]
        # Read back from the same address; constrain symbol; verify load
        # follows the constraint.
        loaded = s.memory.load(addr, 8, endness=s.arch.memory_endness)
        s.solver.add(sym == 0x1122334455667788)
        assert s.solver.eval(loaded) == 0x1122334455667788, (
            f"store/load roundtrip on same symbolic address failed: got {s.solver.eval(loaded):#x}"
        )

    def test_rejected_options_emit_warning(self, fauxware_project):
        """Setting an option tagged ``(b) explicitly reject`` in
        docs/advanced-topics/rust_engine.rst must emit a UserWarning at
        state-add time. Without this signal, users silently get divergent
        behavior from the Python engine.
        """

        # Pick two options that stay in _REJECTED_OPTION_NAMES (warn-only).
        # CALLLESS and DO_RET_EMULATION were promoted to raise in angr-cf9h,
        # so this test now exercises the SimMemory error-handling pair.
        state = fauxware_project.factory.entry_state(
            add_options={
                angr.sim_options.UNINITIALIZED_ACCESS_AWARENESS,
                angr.sim_options.BEST_EFFORT_MEMORY_STORING,
            },
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            RustExplorationManager(fauxware_project, [state])

        messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any("UNINITIALIZED_ACCESS_AWARENESS" in m for m in messages), (
            f"expected UNINITIALIZED_ACCESS_AWARENESS warning; got {messages!r}"
        )
        assert any("BEST_EFFORT_MEMORY_STORING" in m for m in messages), (
            f"expected BEST_EFFORT_MEMORY_STORING warning; got {messages!r}"
        )

    @pytest.mark.parametrize(
        "option_name",
        [
            "TRACK_MEMORY_ACTIONS",
            "TRACK_REGISTER_ACTIONS",
            "TRACK_TMP_ACTIONS",
            "TRACK_JMP_ACTIONS",
            "TRACK_OP_ACTIONS",
        ],
    )
    def test_action_tracking_options_raise_at_construction(
        self,
        fauxware_project,
        option_name,
    ):
        """The TRACK_*_ACTIONS family must raise NotImplementedError at
        RustExplorationManager construction. Rust does not emit SimAction
        records, so silently honoring these would hand the user an empty
        ``state.history.actions`` stream — a hard-to-diagnose divergence.
        Acceptance for angr-xghv. TRACK_ACTION_HISTORY was demoted
        from this list in angr-fkvt (2026-06-06) — it is a metadata
        flag consulted only by preconstrainer.py and does not by itself
        gate action recording; see test_track_action_history_does_not_raise.
        """

        state = fauxware_project.factory.entry_state(
            add_options={getattr(angr.sim_options, option_name)},
        )
        with pytest.raises(NotImplementedError) as exc:
            RustExplorationManager(fauxware_project, [state])
        msg = str(exc.value)
        assert option_name in msg, f"error must name the option: {msg!r}"
        assert "Python engine" in msg, f"error must point users to the Python engine: {msg!r}"

    def test_track_action_history_does_not_raise(self, fauxware_project):
        """TRACK_ACTION_HISTORY alone must not raise — it is a metadata
        flag consulted by preconstrainer.py (and historically by
        successors.py, now commented out) but does not gate action
        recording itself. The TRACK_*_ACTIONS family is the actual
        recording switch and remains raise-listed.

        Acceptance for angr-fkvt: unblocks insomnihack_aeg (angr-86c4)
        whose solve.py sets ``{REVERSE_MEMORY_NAME_MAP,
        TRACK_ACTION_HISTORY}``. Both engines produce an empty
        ``state.history.actions`` when only this option is set, so
        downstream code that consults the flag (preconstrainer's
        clear/restore pattern) behaves identically.
        """

        state = fauxware_project.factory.entry_state(
            add_options={angr.sim_options.TRACK_ACTION_HISTORY},
        )
        # Construction (and the entry-state copy that happens in
        # _add_rust_state) must not raise.
        mgr = RustExplorationManager(fauxware_project, [state])

        # The option must reach the proxy so preconstrainer's standard
        # `o.X in self.state.options` check sees it. Use the initial
        # active state — fauxware deadends quickly so we check before
        # stepping rather than after.
        proxy = mgr.active[0]
        assert angr.sim_options.TRACK_ACTION_HISTORY in proxy.options, (
            "option must be preserved on the proxy so preconstrainer can "
            "see it via the standard `o.X in self.state.options` check"
        )

        # And a step doesn't raise either (the option has no Rust-side
        # action-recording wiring; stepping is a no-op w.r.t. the option).
        mgr.step()

    def test_action_tracking_options_raise_lists_all(self, fauxware_project):
        """When multiple TRACK_*_ACTIONS options are set, the error lists
        all of them so the user can disable them in one pass.
        """

        state = fauxware_project.factory.entry_state(
            add_options={
                angr.sim_options.TRACK_MEMORY_ACTIONS,
                angr.sim_options.TRACK_REGISTER_ACTIONS,
            },
        )
        with pytest.raises(NotImplementedError) as exc:
            RustExplorationManager(fauxware_project, [state])
        msg = str(exc.value)
        assert "TRACK_MEMORY_ACTIONS" in msg
        assert "TRACK_REGISTER_ACTIONS" in msg

    def test_concretize_option_raises_at_construction(self, fauxware_project):
        """CONCRETIZE must raise NotImplementedError at manager construction.
        Rust does not honor SimSolver's BatchedConcretizationBacker, so
        silently accepting it would mean symbolic-driven analyses run
        symbolically instead of eagerly concretizing — a hard-to-diagnose
        semantic divergence. Acceptance for angr-gmrc.
        """

        state = fauxware_project.factory.entry_state(
            add_options={angr.sim_options.CONCRETIZE},
        )
        with pytest.raises(NotImplementedError) as exc:
            RustExplorationManager(fauxware_project, [state])
        msg = str(exc.value)
        assert "CONCRETIZE" in msg, f"error must name the option: {msg!r}"
        assert "Python engine" in msg, f"error must point users to the Python engine: {msg!r}"

    def test_conservative_write_strategy_raises_at_construction(
        self,
        fauxware_project,
    ):
        """CONSERVATIVE_WRITE_STRATEGY must raise NotImplementedError at
        manager construction. Rust's SymbolicMemory always concretizes
        within strategy limits and does not honor the option's
        "refuse to concretize on range-check failure" semantics, so
        silently accepting it would mask the user's intent to keep the
        analysis conservative. Acceptance for angr-csmm.
        """

        state = fauxware_project.factory.entry_state(
            add_options={angr.sim_options.CONSERVATIVE_WRITE_STRATEGY},
        )
        with pytest.raises(NotImplementedError) as exc:
            RustExplorationManager(fauxware_project, [state])
        msg = str(exc.value)
        assert "CONSERVATIVE_WRITE_STRATEGY" in msg, f"error must name the option: {msg!r}"
        assert "Python engine" in msg, f"error must point users to the Python engine: {msg!r}"

    def test_do_ret_emulation_option_raises_at_construction(
        self,
        fauxware_project,
    ):
        """DO_RET_EMULATION must raise NotImplementedError at manager
        construction. The Python engine emits an emulated ret successor at
        every ret site; Rust does not emulate rets at all, so the successor
        set silently differs. Callable workflows are the typical caller and
        would lose the emulated successor. Acceptance for angr-cf9h.
        """

        state = fauxware_project.factory.entry_state(
            add_options={angr.sim_options.DO_RET_EMULATION},
        )
        with pytest.raises(NotImplementedError) as exc:
            RustExplorationManager(fauxware_project, [state])
        msg = str(exc.value)
        assert "DO_RET_EMULATION" in msg, f"error must name the option: {msg!r}"
        assert "Python engine" in msg, f"error must point users to the Python engine: {msg!r}"

    def test_callless_option_raises_at_construction(self, fauxware_project):
        """CALLLESS must raise NotImplementedError at manager construction.
        The Python engine replaces each call with an unconstraining of the
        return register so Callable can short-circuit function bodies; Rust
        has no equivalent path and would step into the callee, structurally
        diverging from the Callable contract. Acceptance for angr-cf9h.
        """

        state = fauxware_project.factory.entry_state(
            add_options={angr.sim_options.CALLLESS},
        )
        with pytest.raises(NotImplementedError) as exc:
            RustExplorationManager(fauxware_project, [state])
        msg = str(exc.value)
        assert "CALLLESS" in msg, f"error must name the option: {msg!r}"
        assert "Python engine" in msg, f"error must point users to the Python engine: {msg!r}"

    def test_efficient_state_merging_option_does_not_raise(
        self,
        fauxware_project,
    ):
        """EFFICIENT_STATE_MERGING must NOT raise (demoted angr-op0dn.11.6).

        It was raise-listed (angr-n129) while merge went through the Python
        export path, which needs the SimStateHistory common-ancestor walk the
        option feeds. Merge is now native — RustExplorationManager.merge()
        runs the M3-4 fast path (merge_states, no export) and the M3-5 native
        ManualMergepoint technique forks-and-merges in Rust — so the option's
        Python rationale is moot and it is honored by not raising. Veritesting
        auto-adds it; honoring it lets such a state explore under the native
        merge machinery instead of hard-failing at the manager boundary.
        """

        state = fauxware_project.factory.entry_state(
            add_options={angr.sim_options.EFFICIENT_STATE_MERGING},
        )
        # Must construct without raising.
        mgr = RustExplorationManager(fauxware_project, [state])
        assert mgr is not None

    def test_symbol_fill_unconstrained_registers_option_raises_at_construction(
        self,
        fauxware_project,
    ):
        """SYMBOL_FILL_UNCONSTRAINED_REGISTERS must raise NotImplementedError
        at manager construction. The Python filler creates a fresh symbolic
        BVS on every read of an uninitialized register; the Rust RegisterFile
        always returns concrete zero from its vec![0; size] storage with no
        "uninitialized" marker. A user who opted into symbolic-fill would
        silently get concrete-zero registers and paths driven by
        unconstrained initial register values would simply not be explored.
        The MEMORY variant SYMBOL_FILL_UNCONSTRAINED_MEMORY is NOT promoted
        because Rust's load_concrete_lazy already defaults to symbolic-fill
        when zero_fill_unconstrained is unset. Acceptance for angr-apre.
        """

        state = fauxware_project.factory.entry_state(
            add_options={angr.sim_options.SYMBOL_FILL_UNCONSTRAINED_REGISTERS},
        )
        with pytest.raises(NotImplementedError) as exc:
            RustExplorationManager(fauxware_project, [state])
        msg = str(exc.value)
        assert "SYMBOL_FILL_UNCONSTRAINED_REGISTERS" in msg, f"error must name the option: {msg!r}"
        assert "Python engine" in msg, f"error must point users to the Python engine: {msg!r}"

    def test_symbol_fill_unconstrained_memory_option_does_not_raise(
        self,
        fauxware_project,
    ):
        """SYMBOL_FILL_UNCONSTRAINED_MEMORY must NOT raise — Rust's
        load_concrete_lazy in native/angr/src/memory/load.rs falls back to a
        fresh symbolic BVS when zero_fill_unconstrained is unset, so the
        memory variant matches Python's symbolic-fill behavior. Only the
        REGISTERS variant is a silent divergence. Acceptance for angr-apre.
        """

        state = fauxware_project.factory.entry_state(
            add_options={angr.sim_options.SYMBOL_FILL_UNCONSTRAINED_MEMORY},
        )
        # Must not raise; manager construction succeeds.
        mgr = RustExplorationManager(fauxware_project, [state])
        assert mgr is not None

    @pytest.mark.parametrize(
        "option_name",
        [
            "BYPASS_ERRORED_IROP",
            "BYPASS_ERRORED_IRCCALL",
            "BYPASS_ERRORED_IRSTMT",
        ],
    )
    def test_bypass_errored_options_raise_at_construction(
        self,
        fauxware_project,
        option_name,
    ):
        """The BYPASS_ERRORED_* family must raise NotImplementedError at
        manager construction. Python's HeavyResilienceMixin catches
        SimError raised during op/ccall/stmt evaluation and substitutes a
        default value; Rust's interpreter maps Op / TypeMismatch /
        InvalidIR errors to FallbackStrategy::Panic and moves the state
        to the errored stash without ever falling back to Python — the
        bypass never fires. Silent divergence from the Python engine.
        Acceptance for angr-6rz8.
        """

        state = fauxware_project.factory.entry_state(
            add_options={getattr(angr.sim_options, option_name)},
        )
        with pytest.raises(NotImplementedError) as exc:
            RustExplorationManager(fauxware_project, [state])
        msg = str(exc.value)
        assert option_name in msg, f"error must name the option: {msg!r}"
        assert "Python engine" in msg, f"error must point users to the Python engine: {msg!r}"

    @pytest.mark.parametrize(
        "option_name",
        [
            "BYPASS_UNSUPPORTED_IROP",
            "BYPASS_UNSUPPORTED_IREXPR",
            "BYPASS_UNSUPPORTED_IRSTMT",
            "BYPASS_UNSUPPORTED_IRDIRTY",
            "BYPASS_UNSUPPORTED_IRCCALL",
            "BYPASS_UNSUPPORTED_SYSCALL",
            "UNSUPPORTED_BYPASS_ZERO_DEFAULT",
            "UNSUPPORTED_FORCE_CONCRETIZE",
        ],
    )
    def test_bypass_unsupported_options_honored_silently(
        self,
        fauxware_project,
        option_name,
    ):
        """The BYPASS_UNSUPPORTED_* family (and the two modifier options
        UNSUPPORTED_BYPASS_ZERO_DEFAULT / UNSUPPORTED_FORCE_CONCRETIZE)
        must NOT raise and NOT warn at manager construction. Rust routes
        unsupported VEX features through FallbackStrategy::PythonCallback
        (interpreter/mod.rs:268-278), which re-runs the failing block
        through Python's HeavyResilienceMixin — so these options are
        honored transparently and need neither raise nor warn entries.
        BYPASS_UNSUPPORTED_IREXPR and BYPASS_UNSUPPORTED_IRSTMT are
        vestigial (defined but not consulted anywhere); they pass
        through vacuously. Acceptance for angr-6rz8.
        """

        state = fauxware_project.factory.entry_state(
            add_options={getattr(angr.sim_options, option_name)},
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mgr = RustExplorationManager(fauxware_project, [state])
        assert mgr is not None
        target = [w for w in caught if issubclass(w.category, UserWarning) and option_name in str(w.message)]
        assert not target, (
            f"{option_name} must not emit a UserWarning at construction; got {[str(w.message) for w in target]!r}"
        )

    def test_bypass_veritesting_exceptions_warns(self, fauxware_project):
        """BYPASS_VERITESTING_EXCEPTIONS is consulted only by
        analyses/veritesting.py; the Rust manager does not yet drive the
        Veritesting technique (angr-op0dn.11.7 is still open). Outside
        Veritesting the option is a no-op. It travels with the
        ``angr.options.resilience`` bundle, so reject-with-warn (not
        raise) keeps resilience-bundle users alive while still signaling
        the divergence. Acceptance for angr-6rz8.
        """

        state = fauxware_project.factory.entry_state(
            add_options={angr.sim_options.BYPASS_VERITESTING_EXCEPTIONS},
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            RustExplorationManager(fauxware_project, [state])
        messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any("BYPASS_VERITESTING_EXCEPTIONS" in m for m in messages), (
            f"expected BYPASS_VERITESTING_EXCEPTIONS warning; got {messages!r}"
        )

    def test_rejected_options_warn_once_per_manager(self, fauxware_project):
        """The warning fires once per option per manager, not per state added."""

        # CALLLESS was promoted to raise in angr-cf9h; use a still-warn-only
        # option so the test exercises the warn-once latch.
        s1 = fauxware_project.factory.entry_state(
            add_options={angr.sim_options.UNINITIALIZED_ACCESS_AWARENESS},
        )
        s2 = s1.copy()
        s3 = s1.copy()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            RustExplorationManager(fauxware_project, [s1, s2, s3])

        target_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning) and "UNINITIALIZED_ACCESS_AWARENESS" in str(w.message)
        ]
        assert len(target_warnings) == 1, (
            f"expected exactly one UNINITIALIZED_ACCESS_AWARENESS warning across 3 states, got {len(target_warnings)}"
        )

    def test_default_state_options_do_not_warn_for_non_rejected(self, fauxware_project):
        """A plain entry_state() must NOT raise a UserWarning for any option
        outside ``_REJECTED_OPTION_NAMES``. Two options ship in the default
        ``symbolic`` mode bundle (TRACK_CONSTRAINT_ACTIONS, TRACK_MEMORY_MAPPING)
        and ARE intentionally in the rejected set — those will warn — but no
        unrelated option should trigger.
        """
        from angr.exploration.rust_manager import _REJECTED_OPTION_NAMES

        state = fauxware_project.factory.entry_state()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            RustExplorationManager(fauxware_project, [state])

        for w in caught:
            if not issubclass(w.category, UserWarning):
                continue
            msg = str(w.message)
            # A warning is only acceptable if it names an option in the
            # rejected set; spurious warnings for other options would mean
            # the matrix has drifted.
            assert any(name in msg for name in _REJECTED_OPTION_NAMES), (
                f"unexpected UserWarning from default entry_state: {msg!r}"
            )

    def test_history_actions_read_warns_under_rust(self, fauxware_project):
        """angr-383x: TRACK_CONSTRAINT_ACTIONS / TRACK_MEMORY_MAPPING ship in
        the default `symbolic` bundle, so they can't be rejected on add()
        without spamming every entry_state(). Instead, reading
        ``state.history.actions`` (or ``.events``) on a Rust-owned materialized
        state must emit a UserWarning so users see a signal the empty stream
        is a Rust-engine limitation, not an actually-empty history.
        """
        from angr.exploration.rust_state_export import _RustOwnedSimStateHistory

        # Reset the process-wide warn-once latch so the test is order-independent
        _RustOwnedSimStateHistory._WARNED = False

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.run(max_steps=1)
        materialized = list(mgr.active) + list(mgr.deadended)
        assert materialized, "expected at least one materialized state"
        s = materialized[0]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = s.history.actions

        history_warnings = [
            w for w in caught if issubclass(w.category, UserWarning) and "state.history.actions" in str(w.message)
        ]
        assert len(history_warnings) == 1, (
            f"expected exactly one history.actions warning, got {[str(w.message) for w in caught]!r}"
        )
        assert "Rust engine" in str(history_warnings[0].message)

    def test_history_events_read_warns_under_rust(self, fauxware_project):
        """Sibling of the actions test for state.history.events."""
        from angr.exploration.rust_state_export import _RustOwnedSimStateHistory

        _RustOwnedSimStateHistory._WARNED = False

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.run(max_steps=1)
        materialized = list(mgr.active) + list(mgr.deadended)
        assert materialized, "expected at least one materialized state"
        s = materialized[0]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = s.history.events

        history_warnings = [
            w for w in caught if issubclass(w.category, UserWarning) and "state.history.events" in str(w.message)
        ]
        assert len(history_warnings) == 1, (
            f"expected exactly one history.events warning, got {[str(w.message) for w in caught]!r}"
        )

    def test_history_actions_warn_once_process_wide(self, fauxware_project):
        """The warn-once latch is process-wide: once fired, no subsequent
        ``state.history.actions`` read (on any state, in any manager) re-warns.
        Default-bundle users who never read .actions get zero warnings; those
        who do, get exactly one."""
        from angr.exploration.rust_state_export import _RustOwnedSimStateHistory

        _RustOwnedSimStateHistory._WARNED = False

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.run(max_steps=1)
        states = list(mgr.active) + list(mgr.deadended)
        assert len(states) >= 1
        s = states[0]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = s.history.actions
            _ = s.history.actions  # second read on same state
            _ = s.history.events  # different attribute, same latch

        history_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and ("state.history.actions" in str(w.message) or "state.history.events" in str(w.message))
        ]
        assert len(history_warnings) == 1, f"expected one process-wide warning, got {len(history_warnings)}"

    def test_history_warning_does_not_fire_when_actions_not_read(self, fauxware_project):
        """Users who run RustExplorationManager but never touch
        state.history.actions/.events must see zero history-related warnings,
        even though TRACK_CONSTRAINT_ACTIONS is in the default `symbolic`
        bundle (the whole reason this strategy exists)."""
        from angr.exploration.rust_state_export import _RustOwnedSimStateHistory

        _RustOwnedSimStateHistory._WARNED = False

        state = fauxware_project.factory.entry_state()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mgr = RustExplorationManager(fauxware_project, [state])
            mgr.run(max_steps=1)
            # Touch unrelated history attributes — bbl_addrs should be fine.
            for s in list(mgr.active) + list(mgr.deadended):
                _ = s.history.recent_bbl_addrs

        history_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and ("state.history.actions" in str(w.message) or "state.history.events" in str(w.message))
        ]
        assert history_warnings == [], (
            f"expected zero history warnings when .actions/.events not read; "
            f"got {[str(w.message) for w in history_warnings]!r}"
        )

    def test_recent_bbl_addrs_uses_tail_ffi_not_export_state(self, fauxware_project):
        """angr-kwpi.1: RustHistoryProxy.recent_bbl_addrs must use the
        lightweight get_state_bbl_history_tail FFI accessor, not the heavy
        export_state path. Verifies (1) the FFI accessor exists and returns
        the tail of history, (2) the proxy delegates to the tail accessor
        and NOT export_state.
        """
        from angr.exploration.rust_state_proxy import RustHistoryProxy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.run(max_steps=2)

        sids = mgr._rust_mgr.get_state_ids("active") or mgr._rust_mgr.get_state_ids("deadended")
        assert sids, "expected at least one state to inspect"
        sid = sids[0]

        # Direct FFI exercise: the accessor exists and returns a list.
        tail = mgr._rust_mgr.get_state_bbl_history_tail(sid, 256)
        assert isinstance(tail, list)
        assert len(tail) <= RustHistoryProxy._RECENT_TAIL_DEFAULT

        # Behavioral check via a stand-in manager that records which
        # accessors the proxy calls. PyO3 classes do not allow attribute
        # patching, so wrap with a recorder rather than monkeypatching.
        class _RecordingMgr:
            def __init__(self, inner):
                self._inner = inner
                self.tail_calls = []
                self.export_calls = 0

            def get_state_bbl_history_tail(self, state_id, n):
                self.tail_calls.append((state_id, n))
                return self._inner.get_state_bbl_history_tail(state_id, n)

            def export_state(self, state_id):
                self.export_calls += 1
                return self._inner.export_state(state_id)

        rec = _RecordingMgr(mgr._rust_mgr)
        proxy = RustHistoryProxy(rec, sid)
        addrs = proxy.recent_bbl_addrs
        _ = proxy.bbl_addrs  # delegate
        _ = proxy.block_count  # delegate

        assert rec.export_calls == 0, "recent_bbl_addrs must not export_state"
        assert rec.tail_calls == [(sid, RustHistoryProxy._RECENT_TAIL_DEFAULT)], (
            f"expected one tail call with default N, got {rec.tail_calls}"
        )
        assert isinstance(addrs, list)
        assert addrs == tail

    def test_stash_iteration_does_not_materialize_simstate(self, fauxware_project):
        """angr-kwpi.2: iterating ``mgr.active`` / ``mgr.found`` / etc. must
        return ``_LazySimStateRef`` wrappers that defer SimState materialization
        until first attribute access.

        Asserts:
          1. Stash properties return ``list`` instances (preserves the
             ``isinstance(active, list)`` contract).
          2. Iterating without touching attrs does NOT call the manager's
             ``_materialize_single_state`` (no heavy plugin restore / sync).
          3. Touching any non-private attribute (``state.solver``,
             ``state.regs``, etc.) triggers materialization exactly once
             per state id; subsequent accesses are cached.
          4. ``mgr.active[0] is mgr.active[0]`` (wrapper identity preserved
             across repeated stash reads).
        """
        from angr.exploration.rust_state_export import _LazySimStateRef

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.step(n=1)

        # (1) Stash returns a list.
        active = mgr.active
        assert isinstance(active, list), "mgr.active must return a list"
        assert active, "expected at least one active state after one step"
        assert all(isinstance(s, _LazySimStateRef) for s in active), "stash entries must be _LazySimStateRef wrappers"

        # (2) Iteration without attribute access does not materialize.
        materialize_calls = []
        orig_materialize = mgr._materialize_single_state

        def counting_materialize(state_id):
            materialize_calls.append(state_id)
            return orig_materialize(state_id)

        mgr._materialize_single_state = counting_materialize
        try:
            for s in mgr.active:
                # touching __slots__ attrs does not trigger __getattr__
                _ = s._lazy_state_id
            _ = len(mgr.active)
            assert mgr.active[0] is mgr.active[0], "wrapper identity must be preserved across repeated stash reads"
            assert materialize_calls == [], (
                f"iteration / len / index should not materialize SimStates; got {materialize_calls!r}"
            )

            # (3) Accessing a real attribute triggers materialization once.
            first = mgr.active[0]
            sid = first._lazy_state_id
            _ = first.solver  # this triggers materialization
            assert materialize_calls == [sid], (
                f"first .solver access should materialize once; got {materialize_calls!r}"
            )
            # Note: each attribute access re-enters _materialize() which is
            # cheap once rust_fully_synced is True (no plugin/sync work), but
            # the call count still increments. Verify cached path is taken
            # (no re-sync) by checking rust_fully_synced stays set.
            cached = mgr._state_cache[sid]
            assert getattr(cached.scratch, "rust_fully_synced", False), (
                "rust_fully_synced should remain set on a hot lazy ref"
            )
        finally:
            mgr._materialize_single_state = orig_materialize

    def test_lazy_ref_private_name_delegation_contract(self, fauxware_project):
        """angr-0xyq2 Phase 4: ``_LazySimStateRef`` private-name rules.

        ``SimProcedure.execute(found, ...)`` calls ``state._inspect`` (the
        asisctffinals2015_license inline-strlen pattern), so the wrapper must
        forward angr-core private entry points — while still refusing to
        materialize on arbitrary private probes (IPython ``_repr_html_``-style)
        and always refusing dunders.
        """
        from angr.exploration.rust_state_export import _LazySimStateRef

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        # Seed states are cached at _add_rust_state time, so step until a
        # FORKED successor (fresh Rust-side id, no cached SimState) shows up.
        fresh = None
        for _ in range(40):
            mgr.step(n=1)
            fresh = next(
                (s for s in mgr.active if s._lazy_state_id not in mgr._state_cache),
                None,
            )
            if fresh is not None:
                break
        assert fresh is not None, "expected an uncached forked successor"
        assert isinstance(fresh, _LazySimStateRef)
        sid = fresh._lazy_state_id

        # (1) Arbitrary private probe on an UNmaterialized ref: refuse
        # without materializing.
        with pytest.raises(AttributeError):
            _ = fresh._repr_html_
        assert sid not in mgr._state_cache, "private probe must not materialize"
        # Dunders likewise (functools/decorator protocol probes).
        with pytest.raises(AttributeError):
            _ = fresh.__wrapped__
        assert sid not in mgr._state_cache

        # (2) The angr-core allowlist (_inspect et al.) materializes on
        # demand, like a public attribute.
        inspect_method = fresh._inspect
        assert callable(inspect_method)
        assert sid in mgr._state_cache, "_inspect must materialize the state"

        # (3) Once materialized, other private names delegate to the real
        # SimState (delegation is free at that point)...
        assert fresh._ip is mgr._state_cache[sid]._ip or fresh._ip.op == mgr._state_cache[sid]._ip.op
        with pytest.raises(AttributeError):
            _ = fresh._no_such_private_attr  # ...but misses still raise.
        # Dunders keep raising even when materialized.
        with pytest.raises(AttributeError):
            _ = fresh.__wrapped__
