"""Tests for RustExplorationManager.

This module tests the Rust-native exploration manager for symbolic execution.
"""

from __future__ import annotations

import os

import pytest

import angr
from angr.errors import AngrCallableMultistateError

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


class TestZ3TacticEnvVar:
    """The ANGR_Z3_TACTIC env var selects the Z3 solver construction strategy.

    The spec is read once via OnceLock on the first call to build_solver, so
    these tests must run in subprocesses to vary the env var. Each subprocess
    asserts the solver still produces correct answers under the requested
    spec — a regression here means we shipped a tactic that breaks
    correctness on real claripy ASTs (sat-preprocess:qfbv was rejected for
    exactly this — see angr-ya00).
    """

    @staticmethod
    def _run_in_subprocess(tactic_env_value):
        """Spawn a fresh interpreter with `ANGR_Z3_TACTIC=<value>` set and
        run a known constraint problem (x*7 + 3 == 24, expected x==3) via
        RustSolverContext. Returns (eval_result, satisfiable_result).
        """
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            """
            import sys
            import claripy
            from angr.exploration.rust_manager import _setup_shared_z3_context
            _setup_shared_z3_context()
            from angr.rustylib.vex_engine import RustSolverContext

            ctx = RustSolverContext()
            x = claripy.BVS("x", 32)
            ctx.add_constraint_ast(x * 7 + 3 == 24)
            sat = ctx.satisfiable()
            val = ctx.eval(x) if sat else None
            sys.stdout.write(f"{int(sat)}|{val}")
            """
        )
        env = dict(os.environ)
        if tactic_env_value is None:
            env.pop("ANGR_Z3_TACTIC", None)
        else:
            env["ANGR_Z3_TACTIC"] = tactic_env_value
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        assert proc.returncode == 0, (
            f"subprocess failed (env={tactic_env_value!r}):\nstdout: {proc.stdout!r}\nstderr: {proc.stderr[-2000:]!r}"
        )
        sat_str, val_str = proc.stdout.strip().split("|")
        return int(sat_str), int(val_str) if val_str != "None" else None

    def test_default_tactic_solves_simple_bv(self):
        """No env var → default Solver::new()."""
        sat, val = self._run_in_subprocess(None)
        assert sat == 1
        assert val == 3

    def test_qfbv_tactic_solves_simple_bv(self):
        """ANGR_Z3_TACTIC=qfbv → Z3 qfbv preset. Bimodal-bench winner per
        angr-ya00 (fairlight 6.2x, sokohashv2 2.6x) but regresses
        csgames2018 and flareon2015_2 — opt-in only."""
        sat, val = self._run_in_subprocess("qfbv")
        assert sat == 1
        assert val == 3

    def test_pipeline_tactic_solves_simple_bv(self):
        """Colon-separated pipeline parses and composes via and_then."""
        sat, val = self._run_in_subprocess("simplify:qfbv")
        assert sat == 1
        assert val == 3

    def test_qfbv_smart_tactic_solves_simple_bv(self):
        """qfbv_smart builds cond(num-consts > N, qfbv, smt). Whichever
        branch fires must still return the correct answer."""
        sat, val = self._run_in_subprocess("qfbv_smart")
        assert sat == 1
        assert val == 3

    def test_unknown_pipeline_name_fails_fast(self):
        """An invalid tactic name must panic loudly rather than silently
        falling back — caught by `Z3_mk_tactic` returning None."""
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            """
            from angr.exploration.rust_manager import _setup_shared_z3_context
            _setup_shared_z3_context()
            from angr.rustylib.vex_engine import RustSolverContext
            try:
                RustSolverContext()
            except BaseException as e:
                print(type(e).__name__)
            """
        )
        env = dict(os.environ)
        env["ANGR_Z3_TACTIC"] = "no-such-tactic-name-12345"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        # Either a non-zero exit (panic via build_solver) or an exception name
        # in stdout is acceptable; the contract is "loud failure, not silent".
        assert proc.returncode != 0 or "Error" in proc.stdout or "Exception" in proc.stdout, (
            f"expected loud failure with invalid tactic; got:\n"
            f"  rc={proc.returncode}\n  stdout={proc.stdout!r}\n  stderr={proc.stderr[-500:]!r}"
        )


class TestSolverOutputCorrectness:
    """Tests verifying solver eval() returns correct values for known constraint systems.

    These complement existing tests (which check satisfiable/count) by checking
    that the actual solved values are correct.
    """

    @classmethod
    def setup_class(cls):
        """Ensure the shared Z3 context is initialized for solver tests."""
        from angr.exploration.rust_manager import _setup_shared_z3_context

        _setup_shared_z3_context()

    def _make_ctx(self):
        from angr.rustylib.vex_engine import RustSolverContext

        return RustSolverContext()

    def test_single_equality(self):
        """x == 42 should eval to exactly 42."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 42)
        assert ctx.eval(x) == 42

    def test_arithmetic_chain(self):
        """x + 10 == 50 should give x == 40."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x + 10 == 50)
        assert ctx.eval(x) == 40

    def test_bitwise_and_mask(self):
        """x & 0xFF == 0x41 constrains the low byte to 'A'."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x & 0xFF == 0x41)
        val = ctx.eval(x)
        assert (val & 0xFF) == 0x41

    def test_xor_constraint(self):
        """x ^ key == target should give x == key ^ target."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        key = 0xDEADBEEF
        target = 0x12345678
        ctx.add_constraint_ast(x ^ key == target)
        assert ctx.eval(x) == (key ^ target)

    def test_shift_left(self):
        """(x << 4) == 0x120 should give x == 0x12."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x << 4 == 0x120)
        # x could be anything where (x << 4) == 0x120, i.e. low 4 bits of x are lost
        # but (x << 4) forces low 4 bits to 0, so x must be 0x12
        val = ctx.eval(x)
        assert (val << 4) & 0xFFFFFFFF == 0x120

    def test_extract_byte(self):
        """Extract byte 1 (bits 15:8) == 0xBE constrains that byte."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(claripy.Extract(15, 8, x) == 0xBE)
        val = ctx.eval(x)
        assert ((val >> 8) & 0xFF) == 0xBE

    def test_concat_constraint(self):
        """Concat(a, b) == 0xAABB constrains both a and b."""
        import claripy

        ctx = self._make_ctx()
        a = claripy.BVS("a", 8)
        b = claripy.BVS("b", 8)
        ctx.add_constraint_ast(claripy.Concat(a, b) == 0xAABB)
        assert ctx.eval(a) == 0xAA
        assert ctx.eval(b) == 0xBB

    def test_signed_comparison(self):
        """Signed comparison: x >s -5 and x <s 5 should give value in (-5, 5)."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(claripy.SGT(x, claripy.BVV(-5, 32)))
        ctx.add_constraint_ast(claripy.SLT(x, claripy.BVV(5, 32)))
        val = ctx.eval(x)
        # Interpret as signed 32-bit
        signed_val = val - 0x100000000 if val >= 0x80000000 else val
        assert -5 < signed_val < 5

    def test_multi_variable_system(self):
        """System of equations: x + y == 100, x - y == 20 => x=60, y=40."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        y = claripy.BVS("y", 32)
        ctx.add_constraint_ast(x + y == 100)
        ctx.add_constraint_ast(x - y == 20)
        assert ctx.eval(x) == 60
        assert ctx.eval(y) == 40

    def test_eval_upto_exact_range(self):
        """eval_upto on tightly constrained variable returns all valid values."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(claripy.UGE(x, 0x41))  # >= 'A'
        ctx.add_constraint_ast(claripy.ULE(x, 0x45))  # <= 'E'
        results = ctx.eval_upto(x, 10)
        assert set(results) == {0x41, 0x42, 0x43, 0x44, 0x45}

    def test_min_max_with_complex_constraints(self):
        """min/max with multiple overlapping constraints."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x >= 100)
        ctx.add_constraint_ast(x <= 200)
        ctx.add_constraint_ast(x >= 150)  # tightens lower bound
        assert ctx.min(x, signed=False) == 150
        assert ctx.max(x, signed=False) == 200

    def test_fork_preserves_values(self):
        """Forked solver preserves parent constraints and returns correct values."""
        import claripy

        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        y = claripy.BVS("y", 32)
        ctx.add_constraint_ast(x == 10)
        ctx.add_constraint_ast(y >= 20)
        ctx.add_constraint_ast(y <= 30)

        forked = ctx.fork()
        forked.add_constraint_ast(y == 25)

        # Forked context: both constraints hold
        assert forked.eval(x) == 10
        assert forked.eval(y) == 25

        # Parent: x still 10, y still in range
        assert ctx.eval(x) == 10
        vy = ctx.eval(y)
        assert 20 <= vy <= 30


class TestCallableStepFunc:
    """Integration tests for Callable/step_func flow with RustExplorationManager.

    This flow (used by flareon2015_10 and similar) calls run(step_func=...) where
    step_func invokes prune() after each step to remove unsatisfiable states.
    It has historically had multiple bugs (see flareon10-three-bugs memory).
    """

    def test_step_func_called_per_step(self, fauxware_project):
        """step_func is called after each execution step."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        call_count = 0

        def count_steps(sm):
            nonlocal call_count
            call_count += 1

        mgr.run(step_func=count_steps, n=5)
        assert call_count == 5, f"step_func should be called 5 times, got {call_count}"

    def test_step_func_stops_when_no_active(self, fauxware_project):
        """run(step_func=...) stops when active stash is empty."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        call_count = 0

        def counting_step(sm):
            nonlocal call_count
            call_count += 1

        # Run with high n limit -- should stop when active is empty
        mgr.run(step_func=counting_step, n=100000)
        assert call_count > 0, "step_func should have been called at least once"
        assert call_count < 100000, "Should have stopped before n limit"

    def test_prune_removes_unsat_states(self, fauxware_project):
        """prune() removes unsatisfiable states from active stash."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        # Step a few times to get some states
        mgr.step()

        # The default prune (no filter_func) should keep satisfiable states
        active_before = len(mgr.active)
        mgr.prune()
        active_after = len(mgr.active)
        # All states from normal execution should be satisfiable
        assert active_after == active_before, "Normal states should all be satisfiable"

    def test_step_func_with_prune(self, fauxware_project):
        """step_func that prunes works correctly (Callable pattern)."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        prune_count = 0

        def prune_step(sm):
            nonlocal prune_count
            sm.prune()
            prune_count += 1

        # Run with step_func that prunes after each step.
        # May stop before n=20 if active stash empties (all states deadend).
        mgr.run(step_func=prune_step, n=20)
        assert prune_count > 0, "step_func should have been called at least once"
        assert prune_count <= 20, f"step_func should not exceed n, got {prune_count}"

    def test_unstash_from_deadended(self, fauxware_project):
        """unstash(from_stash='deadended') moves states to active (Callable pattern)."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        # Run with step_func and prune -- states will eventually deadend
        mgr.run(step_func=lambda sm: sm.prune(), n=100000)

        # Check deadended stash
        deadended_count = len(mgr.deadended)
        assert deadended_count > 0, "Some states should have deadended"

        # unstash should move deadended to active
        mgr.unstash(from_stash="deadended")
        assert len(mgr.deadended) == 0, "Deadended should be empty after unstash"
        assert len(mgr.active) >= deadended_count

    def test_prune_with_filter_func(self, fauxware_project):
        """prune(filter_func=...) keeps only matching states."""

        ACCEPTED = 0x4006ED

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        # Explore to find accepted states
        mgr.explore(find=ACCEPTED, max_steps=50000)
        found_count = len(mgr.found)
        assert found_count > 0, "Should find at least one state"

        # Move found + deadended to active for testing prune with filter
        mgr.unstash(from_stash="found", to_stash="active")
        mgr.unstash(from_stash="deadended", to_stash="active")
        total_active = len(mgr.active)
        assert total_active > 0

        # Prune with filter: keep only states at ACCEPTED
        mgr.prune(filter_func=lambda s: s.addr == ACCEPTED)
        # Should have kept the found states
        assert len(mgr.active) == found_count

    def test_move_filter_func_sees_rust_state_proxy(self, fauxware_project):
        """move(filter_func=...) passes a RustStateProxy when the predicate stays read-only.

        Verifies the proxy fast-path added for angr-9jly: filter_func receives a
        lightweight RustStateProxy rather than a fully reconstructed SimState
        when the predicate only touches proxy-supported attributes (e.g., addr).
        """
        from angr.exploration.rust_state_proxy import RustStateProxy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.step()  # populate active with forks so move has something to filter

        assert len(mgr.active) > 0
        active_before = len(mgr.active)
        proxy_seen = []

        def keep_all(s):
            proxy_seen.append(isinstance(s, RustStateProxy))
            return True

        mgr.move(from_stash="active", to_stash="found", filter_func=keep_all)

        assert len(proxy_seen) == active_before, "filter should see every active state"
        assert all(proxy_seen), "every filter call should receive a RustStateProxy"
        assert len(mgr.found) == active_before
        assert len(mgr.active) == 0

    def test_drop_filter_func_sees_rust_state_proxy(self, fauxware_project):
        """drop(filter_func=...) passes a RustStateProxy when the predicate stays read-only."""
        from angr.exploration.rust_state_proxy import RustStateProxy

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.step()

        assert len(mgr.active) > 0
        active_before = len(mgr.active)
        proxy_seen = []

        def drop_none(s):
            proxy_seen.append(isinstance(s, RustStateProxy))
            return False  # never drop

        mgr.drop(stash="active", filter_func=drop_none)

        assert len(proxy_seen) == active_before, "filter should see every active state"
        assert all(proxy_seen), "every filter call should receive a RustStateProxy"
        assert len(mgr.active) == active_before, "no state should have been dropped"

    def test_callable_with_rust_engine(self, fauxware_project):
        """Full Callable flow works with use_rust_engine=True.

        This tests the complete Callable pipeline:
        1. factory.callable() creates the callable
        2. Callable.perform_call() creates a simulation_manager
        3. run(step_func=_step_func) iterates with pruning
        4. unstash + prune(filter_func=...) post-processes results
        """
        # Monkey-patch simulation_manager to use Rust engine for this test
        original_sm = fauxware_project.factory.simulation_manager

        def rust_sm(thing=None, **kwargs):
            kwargs.pop("use_rust_engine", None)
            kwargs.pop("techniques", None)  # RustExplorationManager doesn't take techniques
            if thing is None:
                thing = [fauxware_project.factory.entry_state()]
            elif isinstance(thing, angr.SimState):
                thing = [thing]
            return RustExplorationManager(fauxware_project, active_states=thing)

        try:
            fauxware_project.factory.simulation_manager = rust_sm

            # authenticate() opens the username as a file and compares the
            # password against its 8 bytes. Those bytes are symbolic (no such
            # file exists), so the strcmp branch splits and a concrete_only
            # Callable raises — which is exactly what the VANILLA Python engine
            # does on this same call. Rust used to return a concrete 0 here
            # only because the read of the natively-opened fd bounced to Python,
            # found no such fd in state.posix, and silently wrote nothing,
            # leaving stored_pw concrete. The native read now mints the symbolic
            # bytes (angr-gorvf.15), restoring Python parity. What this test
            # pins is the Callable -> Rust pipeline reaching the split at all
            # (factory.callable -> perform_call -> run(step_func=...) -> the
            # step_func's multistate check).
            # authenticate is at 0x400664
            authenticate = fauxware_project.factory.callable(
                0x400664,
                prototype="int authenticate(char *username, char *password)",
                concrete_only=True,
            )
            with pytest.raises(AngrCallableMultistateError):
                authenticate(b"SOSNEAKY\x00", b"anything\x00")
        finally:
            fauxware_project.factory.simulation_manager = original_sm

    def test_run_until_predicate(self, fauxware_project):
        """run(step_func=..., until=...) stops when until returns True."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        call_count = 0

        def count_step(sm):
            nonlocal call_count
            call_count += 1

        # Stop after 3 step_func calls via until
        mgr.run(step_func=count_step, until=lambda sm: call_count >= 3, n=100)
        assert call_count == 3, f"Should stop after 3 calls, got {call_count}"

    def test_drop_terminal_states_false_during_step_func(self, fauxware_project):
        """run(step_func=...) sets drop_terminal_states=False so deadended states survive."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        found_deadended = False

        def check_deadended(sm):
            nonlocal found_deadended
            if len(sm.deadended) > 0:
                found_deadended = True

        mgr.run(step_func=check_deadended, n=100000)
        # With drop_terminal_states=False, deadended states should be visible
        # during step_func calls (or at end)
        assert found_deadended or len(mgr.deadended) > 0, (
            "Deadended states should be preserved during step_func execution"
        )


class TestCallStackTracking:
    """Tests for call stack tracking in the Rust engine."""

    def test_call_stack_on_state_snapshot(self, fauxware_project):
        """Test that call stack is available on exported state snapshots."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ED)

        # Without a find target the found stash is empty and the loop below would
        # never run, making the snapshot/call-stack assertions vacuous.
        assert len(mgr.found) > 0, "expected at least one found state to snapshot"
        # Found states should have call stack accessible via snapshot. Found
        # proxies don't carry _rust_state_id, so resolve ids via the Rust mgr
        # (mirrors test_call_stack_on_found_states).
        found_ids = mgr._rust_mgr.get_state_ids("found")
        assert len(found_ids) == len(mgr.found)
        for state_id in found_ids:
            snapshot = mgr._rust_mgr.export_state(state_id)
            call_stack = snapshot.get_call_stack()
            # Call stack is a list of (call_site, callee, ret_addr, sp) tuples
            assert isinstance(call_stack, list)
            depth = snapshot.get_call_stack_depth()
            assert depth == len(call_stack)

    def test_call_stack_api_on_low_level_manager(self):
        """Test get_state_call_stack on the low-level Rust exploration manager."""
        mgr = _RustExplorationManager("amd64")
        state_id = mgr.create_state("active")
        # New state should have empty call stack
        call_stack = mgr.get_state_call_stack(state_id)
        depth = mgr.get_state_call_stack_depth(state_id)
        assert isinstance(call_stack, list)
        assert call_stack == []
        assert depth == 0

    def test_call_stack_on_found_states(self, fauxware_project):
        """Test that found state snapshots contain call stack data."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ED)

        assert len(mgr.found) > 0
        # Check that the found state IDs are accessible via the Rust manager
        found_ids = mgr._rust_mgr.get_state_ids("found")
        for state_id in found_ids:
            snapshot = mgr._rust_mgr.export_state(state_id)
            call_stack = snapshot.get_call_stack()
            depth = snapshot.get_call_stack_depth()
            assert isinstance(call_stack, list)
            assert depth == len(call_stack)
            # Each entry is a tuple of 4 integers
            for entry in call_stack:
                assert len(entry) == 4
                assert all(isinstance(v, int) for v in entry)

    def test_call_stack_on_unit_state(self):
        """Test call stack on a standalone RustSimState."""
        state = RustSimState("amd64", True)
        # New state should have empty call stack
        snapshot = state.export_full()
        assert snapshot.get_call_stack() == []
        assert snapshot.get_call_stack_depth() == 0


class TestDetailedHistory:
    """Tests for detailed execution history tracking."""

    def test_detailed_history_on_unit_state(self):
        """Test that new state has empty detailed history."""
        state = RustSimState("amd64", True)
        snapshot = state.export_full()
        assert snapshot.get_detailed_history() == []

    def test_detailed_history_on_found_states(self, fauxware_project):
        """Test that found states have non-empty detailed history."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ED)

        assert len(mgr.found) > 0
        found_ids = mgr._rust_mgr.get_state_ids("found")
        for state_id in found_ids:
            snapshot = mgr._rust_mgr.export_state(state_id)
            history = snapshot.get_detailed_history()
            assert isinstance(history, list)
            assert len(history) > 0, "Found state should have non-empty history"
            # Each entry is (addr, jumpkind, jump_target)
            for entry in history:
                assert len(entry) == 3
                addr, jumpkind, target = entry
                assert isinstance(addr, int) and addr > 0
                assert isinstance(jumpkind, int) and 0 <= jumpkind <= 4
                assert isinstance(target, int) and target > 0

    def test_detailed_history_str(self, fauxware_project):
        """Test get_detailed_history_str returns human-readable jumpkinds."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ED)

        assert len(mgr.found) > 0
        found_ids = mgr._rust_mgr.get_state_ids("found")
        snapshot = mgr._rust_mgr.export_state(found_ids[0])
        history_str = snapshot.get_detailed_history_str()
        assert len(history_str) > 0
        valid_jumpkinds = {"Ijk_Boring", "Ijk_Call", "Ijk_Ret", "Ijk_Sys_syscall", "Ijk_Other"}
        for _addr, jk_str, _target in history_str:
            assert jk_str in valid_jumpkinds, f"Unknown jumpkind: {jk_str}"

    def test_detailed_history_has_calls(self, fauxware_project):
        """Test that fauxware's history contains Ijk_Call entries."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ED)

        assert len(mgr.found) > 0
        found_ids = mgr._rust_mgr.get_state_ids("found")
        history = mgr._rust_mgr.get_state_detailed_history(found_ids[0])
        # fauxware calls authenticate() and other functions — should have Call entries
        jumpkinds = [jk for _, jk, _ in history]
        # 1 = Ijk_Call
        assert 1 in jumpkinds, "History should contain at least one Ijk_Call"

    def test_detailed_history_manager_api(self):
        """Test get_state_detailed_history on the low-level manager."""
        mgr = _RustExplorationManager("amd64")
        state_id = mgr.create_state("active")
        history = mgr.get_state_detailed_history(state_id)
        assert isinstance(history, list)
        assert history == []

    def test_max_history_get_set_default(self):
        """Manager exposes a configurable per-state history cap (default 1000)."""
        mgr = _RustExplorationManager("amd64")
        assert mgr.get_max_history() == 1000

        mgr.set_max_history(50)
        assert mgr.get_max_history() == 50

        mgr.set_max_history(0)
        assert mgr.get_max_history() == 0

    def test_max_history_caps_recorded_history(self, fauxware_project):
        """A tight max_history cap bounds detailed_history during exploration."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state], max_history=5)
        mgr.explore(find=0x4006ED)

        assert len(mgr.found) > 0
        found_ids = mgr._rust_mgr.get_state_ids("found")
        for state_id in found_ids:
            snap = mgr._rust_mgr.export_state(state_id)
            history = snap.get_detailed_history()
            # Cap is 5 — buffer must not exceed it. Fauxware exploration
            # records >>5 entries before reaching the find target, so the
            # ring buffer must have evicted older entries down to the cap.
            assert len(history) <= 5
            assert len(history) > 0

    def test_max_history_default_bounds_long_run(self, fauxware_project):
        """Default cap (1000) keeps detailed_history bounded on a real run."""

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ED)

        assert len(mgr.found) > 0
        found_ids = mgr._rust_mgr.get_state_ids("found")
        for state_id in found_ids:
            snap = mgr._rust_mgr.export_state(state_id)
            history = snap.get_detailed_history()
            # Default cap is 1000 — must hold for any long-running exploration.
            assert len(history) <= 1000

    def test_max_history_retroactive_trim(self):
        """Lowering max_history below the current buffer length trims FIFO.

        Without the retroactive trim, `add_to_history` only removes one
        entry per push when over cap, so a state with 100 entries would
        never converge to a smaller cap; the buffer would stay bloated
        until enough new pushes happened to drain it. Verifies both
        the basic `history` (Vec<u64>) and `detailed_history` paths.
        """
        state = RustSimState("amd64")
        # Disable cap and load up the buffers.
        state.set_max_history(0)
        for i in range(20):
            state.add_history(0x1000 + i)
            state.add_detailed_history(0x1000 + i, 0, 0)
        assert len(state.history()) == 20
        assert len(state.detailed_history()) == 20

        # Retroactively shrink the cap; both buffers must be FIFO-trimmed
        # to the most-recent 4 entries immediately.
        state.set_max_history(4)
        assert state.get_max_history() == 4
        kept_history = state.history()
        assert kept_history == [0x1010, 0x1011, 0x1012, 0x1013]
        kept_detailed = state.detailed_history()
        assert len(kept_detailed) == 4
        assert kept_detailed[0][0] == 0x1010  # oldest kept entry
        assert kept_detailed[3][0] == 0x1013  # newest entry

    def test_max_history_fifo_eviction_via_add(self):
        """add_detailed_history evicts the oldest entry first when over cap.

        Pushing past the cap one entry at a time must drop the head of
        the buffer, not the tail or a random index. Without FIFO order
        the kept window would not be the most-recent N entries.
        """
        state = RustSimState("amd64")
        state.set_max_history(3)
        for i in range(8):
            state.add_detailed_history(0x2000 + i, i & 0xFF, 0)
        kept = state.detailed_history()
        assert len(kept) == 3
        # The most-recent 3 entries (0x2005, 0x2006, 0x2007) survive.
        assert [entry[0] for entry in kept] == [0x2005, 0x2006, 0x2007]
        # Jumpkind/jump_target travel with the address through eviction.
        assert kept[0][1] == 5
        assert kept[2][1] == 7

    def test_max_history_inherited_through_fork(self):
        """Forking a state copies max_history (and the existing buffer).

        Lowering the cap on the parent post-fork must not touch the
        child's cap — they're independent fields after the clone.
        Conversely, the child must not silently revert to the default
        (1000) on fork.
        """
        parent = RustSimState("amd64")
        parent.set_max_history(7)
        for i in range(3):
            parent.add_detailed_history(0x3000 + i, 0, 0)

        child = parent.fork()
        # Cap propagates and the existing buffer is copied.
        assert child.get_max_history() == 7
        assert len(child.detailed_history()) == 3

        # Caps are independent post-fork: changing one must not affect
        # the other (separate fields, not shared state).
        parent.set_max_history(2)
        assert child.get_max_history() == 7
        # Parent buffer trimmed to 2 by the retroactive cap.
        assert len(parent.detailed_history()) == 2

        # Push past the inherited cap on the child only; FIFO eviction
        # kicks in at 7 entries.
        for i in range(10):
            child.add_detailed_history(0x4000 + i, 0, 0)
        kept = child.detailed_history()
        assert len(kept) == 7
        # Oldest 3 (0x3000-0x3002) and the first 3 of 0x4000-0x4009
        # are evicted; the tail 0x4003-0x4009 survives.
        assert [entry[0] for entry in kept] == [
            0x4003,
            0x4004,
            0x4005,
            0x4006,
            0x4007,
            0x4008,
            0x4009,
        ]


class TestNativeTechniques:
    """Tests for native exploration technique hooks in Rust."""

    def test_register_length_limiter(self):
        """Test that LengthLimiter can be registered natively."""
        mgr = _RustExplorationManager("amd64")
        assert mgr.native_technique_count() == 0
        mgr.register_length_limiter(100, False)
        assert mgr.native_technique_count() == 1
        mgr.register_length_limiter(200, True)
        assert mgr.native_technique_count() == 2
        mgr.clear_native_techniques()
        assert mgr.native_technique_count() == 0

    def test_register_timeout(self):
        """Test that Timeout can be registered natively."""
        mgr = _RustExplorationManager("amd64")
        mgr.register_timeout(10.0)
        assert mgr.native_technique_count() == 1

    def test_register_loop_bound(self):
        """Test that LoopBound can be registered natively."""
        mgr = _RustExplorationManager("amd64")
        mgr.register_loop_bound(5, "spinning")
        assert mgr.native_technique_count() == 1

    def test_length_limiter_via_use_technique(self, fauxware_project):
        """Test LengthLimiter registered through use_technique()."""
        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=True)

        from angr.exploration_techniques import LengthLimiter

        tech = LengthLimiter(max_length=50)
        mgr.use_technique(tech)

        # Verify it was registered natively
        assert getattr(tech, "_native_length_limiter", False), "LengthLimiter should be marked as native"
        assert mgr._rust_mgr.native_technique_count() == 1

    def test_length_limiter_cuts_long_paths(self, fauxware_project):
        """Test that LengthLimiter actually moves states to 'cut' stash."""
        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=True)

        from angr.exploration_techniques import LengthLimiter

        mgr.use_technique(LengthLimiter(max_length=5))

        # Explore — paths beyond 5 blocks should be cut
        mgr.run(max_steps=200)

        # Should have states in 'cut' stash
        cut_ids = list(mgr._rust_mgr.get_state_ids("cut"))
        assert len(cut_ids) > 0, "LengthLimiter should have moved some states to 'cut'"

    def test_timeout_via_use_technique(self, fauxware_project):
        """Test Timeout registered through use_technique()."""
        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=True)

        from angr.exploration_techniques import Timeout

        tech = Timeout(timeout=30)
        mgr.use_technique(tech)

        assert getattr(tech, "_native_timeout", False), "Timeout should be marked as native"
        assert mgr._rust_mgr.native_technique_count() == 1

    def test_multiple_native_techniques(self):
        """Test that multiple native techniques can coexist."""
        mgr = _RustExplorationManager("amd64")
        mgr.register_length_limiter(100, False)
        mgr.register_timeout(30.0)
        mgr.register_loop_bound(10, "spinning")
        assert mgr.native_technique_count() == 3

    def test_loop_seer_via_use_technique(self, fauxware_project):
        """LoopSeer(bound=N) routes to the native register_loop_bound."""
        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=True)

        from angr.exploration_techniques import LoopSeer

        mgr.use_technique(LoopSeer(bound=8))
        assert mgr._rust_mgr.native_technique_count() == 1

    def test_local_loop_seer_via_use_technique(self, fauxware_project):
        """LocalLoopSeer(bound=N) routes to the native register_loop_bound."""
        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=True)

        from angr.exploration_techniques import LocalLoopSeer

        mgr.use_technique(LocalLoopSeer(bound=8))
        assert mgr._rust_mgr.native_technique_count() == 1

    def test_loop_seer_no_bound_no_native(self, fauxware_project):
        """A bound-less LoopSeer has nothing to enforce, so no native technique."""
        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=True)

        from angr.exploration_techniques import LoopSeer

        mgr.use_technique(LoopSeer())  # bound defaults to None
        assert mgr._rust_mgr.native_technique_count() == 0

    def test_loop_seer_bound_reached_callback_skips_native(self, fauxware_project):
        """A bound_reached callback can't run in Rust, so we skip the native bound."""
        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=True)

        from angr.exploration_techniques import LoopSeer

        mgr.use_technique(LoopSeer(bound=8, bound_reached=lambda seer, succ: None))
        assert mgr._rust_mgr.native_technique_count() == 0

    def test_local_loop_seer_routes_to_spinning(self, fauxware_project):
        """LocalLoopSeer(bound=N) routes over-bound states to 'spinning'.

        fauxware's authentication path revisits the same blocks (read/strcmp
        loop) more than twice, so a tight bound=2 trips the native back-edge
        heuristic and moves the offending state to the default 'spinning'
        discard stash, matching the Python engine's LocalLoopSeer behavior.
        A synthetic ``jmp $`` blob can't be used here: the uniqueness filter
        collapses the identical repeated state before the bound is reached.
        """
        from angr.exploration_techniques import LocalLoopSeer

        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=True)

        mgr.use_technique(LocalLoopSeer(bound=2))
        assert mgr._rust_mgr.native_technique_count() == 1

        mgr.run(max_steps=120)

        # The tight bound must have routed at least one state to 'spinning',
        # and exploration must have drained the active stash.
        assert len(list(mgr._rust_mgr.get_state_ids("active"))) == 0
        assert len(list(mgr._rust_mgr.get_state_ids("spinning"))) > 0


class TestTechniqueFilterReevaluation:
    """Re-filter semantics for ExplorationTechnique.filter() (angr-j1ue).

    Rust state ids persist across steps for non-forking states, so a filter()
    that depends on a state's evolving address/stdout must be re-evaluated on
    each signature change — not skipped after one pass.
    """

    def test_persistent_state_refiltered_on_signature_change(self, fauxware_project):
        """A filter is re-invoked for a persistent id once its signature changes.

        Regression for angr-j1ue: under the old once-per-id contract each Rust
        state id was filtered exactly once and never re-checked, so a filter
        whose verdict depends on evolving state (here: how many times the same
        id has been presented) could never fire after the first sweep. The
        change-detection cache re-runs the filter when (addr, stdout_len) moves,
        so the second invocation lands and the state is routed.
        """
        from angr.exploration_techniques import ExplorationTechnique

        class _SecondSightTagger(ExplorationTechnique):
            """Move a state to 'tagged' the SECOND time its id is filtered.

            Reaching a second filter call for one id is only possible if the id
            is re-evaluated after its first sweep — exactly the behavior the fix
            restores.
            """

            def __init__(self):
                super().__init__()
                self.calls = {}

            def filter(self, simgr, state, **kwargs):
                sid = state._state_id
                self.calls[sid] = self.calls.get(sid, 0) + 1
                if self.calls[sid] >= 2:
                    return "tagged"
                return None

        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=True)
        tech = _SecondSightTagger()
        mgr.use_technique(tech)
        # Step one block at a time so a filter sweep runs after each step (run()
        # batches steps and would sweep too coarsely to observe a persistent id
        # twice). The entry path stays linear for several blocks, keeping id 1.
        for _ in range(8):
            if not mgr._rust_mgr.has_active_states():
                break
            mgr.step()

        # The regression would cap every id at exactly one call; the fix lets a
        # persistent id be re-filtered after its signature changes, which both
        # bumps a call count past 1 and routes the state to 'tagged'.
        assert max(tech.calls.values(), default=0) >= 2, (
            "a persistent id must be re-filtered after its signature changes"
        )
        assert list(mgr._rust_mgr.get_state_ids("tagged")), (
            "the second-sight filter must have routed a state to 'tagged'"
        )

    def test_named_check_uniqueness_takes_native_path(self, fauxware_project):
        """A technique named CheckUniqueness registers natively, bypassing the guard.

        use_technique() keys native uniqueness registration on the class NAME
        "CheckUniqueness" (the grub-style custom technique). On x86_64 it always
        registers natively, so the Python filter() — and the
        _MONOTONIC_FILTER_TECHNIQUES guard that protects it — never runs. This
        is why the re-filter change cannot alter CheckUniqueness behavior.
        """
        from angr.exploration.rust_techniques import _MONOTONIC_FILTER_TECHNIQUES
        from angr.exploration_techniques import ExplorationTechnique

        assert "CheckUniqueness" in _MONOTONIC_FILTER_TECHNIQUES

        class CheckUniqueness(ExplorationTechnique):
            def filter(self, simgr, state, **kwargs):
                return None

        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=True)
        tech = CheckUniqueness()
        mgr.use_technique(tech)

        # Native registration means the Python filter (and its monotonic guard)
        # is bypassed entirely, so the re-filter change is a no-op for it.
        assert getattr(tech, "_native_uniqueness", False), "CheckUniqueness must register natively on x86_64"
        mgr.run(max_steps=30)


class TestExplorationTechniqueStepHookDispatch:
    """ExplorationTechnique.step() hook dispatch against RustStateProxy.

    angr-rqvq: prior to this change the Rust manager only invoked filter()
    and complete() hooks, so techniques whose effect lived in step() (e.g.
    MemoryWatcher, Spiller, StochasticSearch, Director, DrillerCore) were
    silently no-op'd. These tests pin down the new dispatch contract.
    """

    def _step_counting_tech(self):
        """Build a tech that increments a counter inside step() and delegates."""
        from angr.exploration_techniques import ExplorationTechnique

        class _StepCounter(ExplorationTechnique):
            def __init__(self):
                super().__init__()
                self.step_calls = 0
                self.last_simgr = None

            def step(self, simgr, stash="active", **kwargs):
                self.step_calls += 1
                self.last_simgr = simgr
                simgr.step(stash=stash, **kwargs)

        return _StepCounter()

    def test_step_hook_dispatched_during_run(self, fauxware_project):
        """step() runs at least once per Rust batch under run()."""
        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=True)
        tech = self._step_counting_tech()
        mgr.use_technique(tech)

        mgr.run(max_steps=10)

        assert tech.step_calls >= 1, "step() hook should have been invoked at least once"

    def test_step_hook_simgr_is_proxy(self, fauxware_project):
        """The simgr arg to step() is a RustSimulationManagerProxy."""
        from angr.exploration.rust_state_proxy import RustSimulationManagerProxy

        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=True)
        tech = self._step_counting_tech()
        mgr.use_technique(tech)

        mgr.run(max_steps=5)

        assert isinstance(tech.last_simgr, RustSimulationManagerProxy)

    def test_step_hook_dispatched_during_step(self, fauxware_project):
        """step() dispatches through the explicit mgr.step(n) path too."""
        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=True)
        tech = self._step_counting_tech()
        mgr.use_technique(tech)

        mgr.step(n=3)

        assert tech.step_calls >= 1

    def test_step_hooks_compose_lifo(self, fauxware_project):
        """Multiple step() hooks compose in LIFO order (matches SimulationManager)."""
        from angr.exploration_techniques import ExplorationTechnique

        order = []

        class _Recorder(ExplorationTechnique):
            def __init__(self, name):
                super().__init__()
                self.name = name

            def step(self, simgr, stash="active", **kwargs):
                order.append(("enter", self.name))
                simgr.step(stash=stash, **kwargs)
                order.append(("exit", self.name))

        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=True)
        mgr.use_technique(_Recorder("first"))
        mgr.use_technique(_Recorder("second"))

        mgr.run(max_steps=2)

        # The last-registered tech wraps the earlier one: outer = second,
        # inner = first.
        assert ("enter", "second") in order
        assert ("enter", "first") in order
        # second wraps first — second's enter precedes first's enter
        enter_second = order.index(("enter", "second"))
        enter_first = order.index(("enter", "first"))
        assert enter_second < enter_first, f"Expected LIFO compose, got {order}"

    def test_step_hook_exception_falls_back(self, fauxware_project):
        """If a step() hook raises, run loop still advances via fallback."""
        from angr.exploration_techniques import ExplorationTechnique

        class _Raiser(ExplorationTechnique):
            def step(self, simgr, stash="active", **kwargs):
                raise RuntimeError("technique step intentionally raised")

        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=True)
        mgr.use_technique(_Raiser())

        # No exception escapes; exploration completes.
        mgr.run(max_steps=5)

        # The fallback must keep the run loop advancing despite the raising hook.
        # stash_counts() being non-None is true for any manager and proves
        # nothing; on linear fauxware the single active state steps forward
        # without forking/draining, so assert the canonical progress counter
        # (mgr.stats["steps"]) actually advanced.
        assert mgr.stats["steps"] > 0, f"run made no progress despite fallback: stats={mgr.stats}"

    def test_no_step_hook_means_no_dispatch_overhead(self, fauxware_project):
        """Techniques without step() override should NOT trigger dispatch."""
        from angr.exploration_techniques import ExplorationTechnique

        class _FilterOnly(ExplorationTechnique):
            def filter(self, simgr, state, **kwargs):
                return None

        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=True)
        mgr.use_technique(_FilterOnly())

        assert not mgr._has_technique_step_hooks()

    def test_native_dfs_does_not_dispatch_step(self, fauxware_project):
        """Native DFS handles stepping internally; its step() must NOT dispatch."""
        from angr.exploration_techniques import DFS

        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=True)
        mgr.use_technique(DFS())

        # DFS overrides step() in Python, but we shadow it natively in the
        # Rust manager — _has_technique_step_hooks() must skip it to avoid
        # double-dispatch (and double-stepping).
        assert not mgr._has_technique_step_hooks(), "Native DFS step() must NOT be re-dispatched"

    def test_proxy_step_without_callback_raises(self):
        """Direct RustSimulationManagerProxy.step() with no callback raises."""
        from angr.exploration.rust_state_proxy import RustSimulationManagerProxy

        # No rust_mgr needed — we expect to fail before touching it.
        proxy = RustSimulationManagerProxy(rust_mgr=None)
        with pytest.raises(NotImplementedError):
            proxy.step()

    def test_proxy_successors_raises_not_implemented(self):
        """Successors() must raise — silent no-op masks tech bugs."""
        from angr.exploration.rust_state_proxy import RustSimulationManagerProxy

        proxy = RustSimulationManagerProxy(rust_mgr=None)
        with pytest.raises(NotImplementedError):
            proxy.successors(state=None)

    def test_proxy_step_state_raises_not_implemented(self):
        """step_state() must raise — silent no-op masks tech bugs."""
        from angr.exploration.rust_state_proxy import RustSimulationManagerProxy

        proxy = RustSimulationManagerProxy(rust_mgr=None)
        with pytest.raises(NotImplementedError):
            proxy.step_state(state=None)


class TestVexOptLevel:
    """Tests for VEX optimization level control."""

    def test_set_get_vex_opt_level(self):
        """Test setting and getting VEX optimization level."""
        mgr = _RustExplorationManager("amd64")
        assert mgr.get_vex_opt_level() is None
        mgr.set_vex_opt_level(0)
        assert mgr.get_vex_opt_level() == 0
        mgr.set_vex_opt_level(2)
        assert mgr.get_vex_opt_level() == 2
        mgr.set_vex_opt_level(None)
        assert mgr.get_vex_opt_level() is None

    def test_per_address_override(self):
        """Test per-address VEX optimization level overrides."""
        mgr = _RustExplorationManager("amd64")
        mgr.set_vex_opt_level(1)  # global level

        # Per-address override
        mgr.set_vex_opt_level_override(0x401000, 0)
        assert mgr.resolve_vex_opt_level(0x401000) == 0  # override
        assert mgr.resolve_vex_opt_level(0x402000) == 1  # global fallback

        # Remove override
        mgr.remove_vex_opt_level_override(0x401000)
        assert mgr.resolve_vex_opt_level(0x401000) == 1  # falls back to global

    def test_clear_overrides(self):
        """Test clearing all per-address overrides."""
        mgr = _RustExplorationManager("amd64")
        mgr.set_vex_opt_level_override(0x401000, 0)
        mgr.set_vex_opt_level_override(0x402000, 2)
        mgr.clear_vex_opt_level_overrides()
        assert mgr.resolve_vex_opt_level(0x401000) is None
        assert mgr.resolve_vex_opt_level(0x402000) is None

    def test_opt_level_with_exploration(self, fauxware_project):
        """Test that opt_level doesn't break exploration."""
        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=True)

        # Set opt_level 0 (no optimization) and verify exploration still works
        mgr._rust_mgr.set_vex_opt_level(0)
        mgr.explore(find=0x4006ED)
        assert len(mgr.found) > 0, "Should find target with opt_level=0"


class TestStateMerging:
    """Tests for state merging with symbolic merge conditions."""

    def test_merge_states_basic(self):
        """Test merging two states produces a valid merged state."""
        mgr = _RustExplorationManager("amd64")
        sid1 = mgr.create_state("active")
        sid2 = mgr.create_state("active")

        merged_id = mgr.merge_states([sid1, sid2], "merged")
        merged_ids = mgr.get_state_ids("merged")
        assert merged_id in merged_ids

    def test_merge_states_requires_two(self):
        """Test that merge_states requires at least 2 states."""
        mgr = _RustExplorationManager("amd64")
        sid1 = mgr.create_state("active")

        with pytest.raises(ValueError, match="at least 2"):
            mgr.merge_states([sid1], "merged")

    def test_merge_states_invalid_id(self):
        """Test that merge_states raises on invalid state IDs."""
        mgr = _RustExplorationManager("amd64")
        sid1 = mgr.create_state("active")

        with pytest.raises(ValueError, match=r"state .* not found"):
            mgr.merge_states([sid1, 999999], "merged")

    def test_merge_three_states(self):
        """Test merging three states."""
        mgr = _RustExplorationManager("amd64")
        sid1 = mgr.create_state("active")
        sid2 = mgr.create_state("active")
        sid3 = mgr.create_state("active")

        merged_id = mgr.merge_states([sid1, sid2, sid3], "merged")
        merged_ids = mgr.get_state_ids("merged")
        assert merged_id in merged_ids

    def test_merge_states_default_stash(self):
        """Test merge_states with default dest_stash."""
        mgr = _RustExplorationManager("amd64")
        sid1 = mgr.create_state("stash1")
        sid2 = mgr.create_state("stash2")

        merged_id = mgr.merge_states([sid1, sid2])
        # Default dest_stash is "active"
        active_ids = mgr.get_state_ids("active")
        assert merged_id in active_ids


class TestNativeMergeFastPath:
    """M3-4 (angr-op0dn.11.4): RustExplorationManager.merge() native fast path.

    When ``merge_func`` and ``merge_key`` are both None, same-pc native states
    are merged in-Rust via ``merge_states`` with NO export -> Python
    ``state.merge()`` -> re-import round trip. ``stats()['states_merged_native']``
    counts the states consumed by that path; a custom ``merge_func`` still
    exports and leaves the counter at 0.
    """

    def test_native_merge_same_pc_no_export(self, fauxware_project):
        """Two same-pc states merge to 1 with states_merged_native == 2 and
        zero exports (get_state_by_id never called on the native path)."""
        s1 = fauxware_project.factory.entry_state()
        s2 = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [s1, s2])
        assert len(mgr._rust_mgr.get_state_ids("active")) == 2

        exports = []
        orig_get = mgr.get_state_by_id
        mgr.get_state_by_id = lambda sid: (exports.append(sid), orig_get(sid))[1]

        mgr.merge()

        assert exports == [], "native path must not export states"
        assert len(mgr._rust_mgr.get_state_ids("active")) == 1
        assert mgr.stats["states_merged_native"] == 2

    def test_native_merge_singletons_untouched(self, fauxware_project):
        """A single active state is a no-op (nothing to merge) and does not
        increment the native counter."""
        s1 = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [s1])
        mgr.merge()
        assert len(mgr._rust_mgr.get_state_ids("active")) == 1
        assert mgr.stats["states_merged_native"] == 0

    def test_custom_merge_func_takes_python_path(self, fauxware_project):
        """A custom merge_func exports states (Python path) and leaves
        states_merged_native at 0."""
        s1 = fauxware_project.factory.entry_state()
        s2 = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [s1, s2])

        exports = []
        orig_get = mgr.get_state_by_id
        mgr.get_state_by_id = lambda sid: (exports.append(sid), orig_get(sid))[1]

        # Return one arm directly; the point is only that the exporting Python
        # path is taken (built-in merge of freshly-exported entry states hits an
        # unrelated history-plugin-class mismatch), not the merge result itself.
        def _merge_func(base, *others):
            return base

        mgr.merge(merge_func=_merge_func)

        assert exports, "custom merge_func must take the exporting Python path"
        assert mgr.stats["states_merged_native"] == 0
        assert len(mgr._rust_mgr.get_state_ids("active")) == 1


class TestManualMergepointNative:
    """M3-5 (angr-op0dn.11.5): ManualMergepoint routes to the native
    ``register_merge_point`` technique, and its Python ``step()`` hook is
    suppressed so the native and Python implementations do not both run.
    """

    def test_manual_mergepoint_registers_native(self, fauxware_project):
        """ManualMergepoint(addr) registers exactly one native technique and
        flags itself as natively handled."""
        state = fauxware_project.factory.entry_state()
        mgr = fauxware_project.factory.simulation_manager(state, use_rust_engine=True)

        from angr.exploration_techniques import ManualMergepoint

        tech = ManualMergepoint(0x400000, wait_counter=5)
        mgr.use_technique(tech)

        assert mgr._rust_mgr.native_technique_count() == 1
        assert getattr(tech, "_native_merge_point", False) is True

    def test_manual_mergepoint_step_hook_suppressed(self):
        """The native MergePoint replaces the Python step() hook, so
        _has_dispatched_step_hook must return False for ManualMergepoint."""
        from angr.exploration.rust_techniques import (
            _NATIVE_STEP_TECH_NAMES,
            _has_dispatched_step_hook,
        )
        from angr.exploration_techniques import ManualMergepoint

        assert "ManualMergepoint" in _NATIVE_STEP_TECH_NAMES
        assert _has_dispatched_step_hook(ManualMergepoint(0x400000)) is False
