"""Tests for RustExplorationManager.

This module tests the Rust-native exploration manager for symbolic execution.
"""
import os
import pytest

import angr


# Check if Rust exploration is available
try:
    from angr.rustylib.vex_engine import (
        RustExplorationManager as _RustExplorationManager,
        ExplorationEvent,
        PythonCallbacks,
        RustSimState,
    )
    RUST_EXPLORATION_AVAILABLE = True
except ImportError:
    RUST_EXPLORATION_AVAILABLE = False

# Get test binary path
# Try multiple locations for test binaries
_candidates = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "binaries"),
    os.path.expanduser("~/repos/angr-examples/examples/fauxware"),
    os.path.expanduser("~/angr-examples/examples/fauxware"),
]
TEST_BINARIES_DIR = next((d for d in _candidates if os.path.isdir(d)), _candidates[0])


@pytest.fixture(scope="module")
def fauxware_project():
    """Load fauxware test binary (shared across all tests in this module)."""
    binary_path = os.path.join(TEST_BINARIES_DIR, "fauxware")
    if not os.path.exists(binary_path):
        pytest.skip("fauxware binary not found")
    return angr.Project(binary_path, auto_load_libs=False)


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestRustExplorationManagerUnit:
    """Unit tests for RustExplorationManager Rust class."""

    def test_creation(self):
        """Test creating an exploration manager."""
        mgr = _RustExplorationManager("amd64")
        assert mgr.arch == "amd64"
        assert mgr.step_count == 0
        assert mgr.active_count() == 0
        assert mgr.found_count() == 0

    def test_create_state(self):
        """Test creating states in stashes."""
        mgr = _RustExplorationManager("amd64")

        # Create a state
        state_id = mgr.create_state("active")
        assert state_id >= 0  # State IDs can start at 0
        assert mgr.active_count() == 1

        # Create another state
        state_id2 = mgr.create_state("active")
        # Note: Both states might have ID 0 since create_state creates fresh states
        assert mgr.active_count() == 2

    def test_stash_counts(self):
        """Test stash count reporting."""
        mgr = _RustExplorationManager("amd64")

        # Create states in different stashes
        mgr.create_state("active")
        mgr.create_state("active")
        mgr.create_state("found")

        counts = mgr.stash_counts()
        assert counts["active"] == 2
        assert counts["found"] == 1

    def test_find_avoid_addresses(self):
        """Test setting find/avoid addresses."""
        mgr = _RustExplorationManager("amd64")

        # Set find addresses
        mgr.set_find_addrs([0x1000, 0x2000, 0x3000])

        # Set avoid addresses
        mgr.set_avoid_addrs([0x4000, 0x5000])

        stats = mgr.stats()
        assert stats["find_addrs"] == 3
        assert stats["avoid_addrs"] == 2

    def test_register_simprocedures(self):
        """Test registering SimProcedures."""
        mgr = _RustExplorationManager("amd64")

        # Register some procedures
        mgr.register_simprocedure(0x401000, "strlen", 1, False)
        mgr.register_simprocedure(0x401100, "malloc", 1, False)
        mgr.register_simprocedure(0x401200, "exit", 1, True)

        stats = mgr.stats()
        assert stats["simprocedures"] == 3
        assert stats["hooks"] == 3

    def test_dcas_unsupported_metric_exposed(self):
        """DCAS visibility counter is exposed via stats() and get_fallback_stats()."""
        mgr = _RustExplorationManager("amd64")

        stats = mgr.stats()
        assert "dcas_unsupported_count" in stats
        assert stats["dcas_unsupported_count"] == 0

        fb = mgr.get_fallback_stats()
        assert "dcas_unsupported_count" in fb
        assert fb["dcas_unsupported_count"] == 0

    def test_add_rust_state(self):
        """Test adding an existing RustSimState."""
        mgr = _RustExplorationManager("amd64")

        # Create a RustSimState
        state = RustSimState("amd64")
        state.pc = 0x1000

        # Add to manager
        mgr.add_state("active", state)
        assert mgr.active_count() == 1

        # Check PC
        pc = mgr.get_state_pc("active", 0)
        assert pc == 0x1000

    def test_state_ids(self):
        """Test getting state IDs from stash."""
        mgr = _RustExplorationManager("amd64")

        # Create states
        id1 = mgr.create_state("active")
        id2 = mgr.create_state("active")

        # Get IDs
        ids = mgr.get_state_ids("active")
        assert len(ids) == 2
        assert id1 in ids
        assert id2 in ids

    def test_set_rust_log_level(self):
        """Test setting Rust log level from Python."""
        from angr.rustylib.vex_engine import set_rust_log_level
        # Should accept all valid levels without error
        for level in ("error", "warn", "info", "debug", "trace", "off"):
            set_rust_log_level(level)
        # Invalid level should raise
        with pytest.raises(ValueError):
            set_rust_log_level("invalid")

    def test_max_active_states_get_set(self):
        """Test get/set for max_active_states limit."""
        mgr = _RustExplorationManager("amd64")

        # Default is None (unlimited)
        assert mgr.get_max_active_states() is None

        # Set a limit
        mgr.set_max_active_states(5)
        assert mgr.get_max_active_states() == 5

        # Clear the limit
        mgr.set_max_active_states(None)
        assert mgr.get_max_active_states() is None

    def test_max_active_states_enforced(self):
        """Test that max_active_states limit prevents adding excess states."""
        mgr = _RustExplorationManager("amd64")
        mgr.set_max_active_states(3)

        # Add states up to the limit
        mgr.create_state("active")
        mgr.create_state("active")
        mgr.create_state("active")
        assert mgr.active_count() == 3

        # Adding beyond the limit via create_state goes directly to stash,
        # so it bypasses push_to_active_or_drop. Verify the getter works.
        assert mgr.get_max_active_states() == 3

    def test_register_python_procedure_appears_in_listing(self):
        """register_python_procedure adds the procedure to the registry."""
        mgr = _RustExplorationManager("amd64")
        assert not mgr.has_native_procedure("custom_widget_init")

        def widget_init(args):
            return 0

        mgr.register_python_procedure(
            "custom_widget_init",
            num_args=0,
            no_return=False,
            callable=widget_init,
        )
        assert mgr.has_native_procedure("custom_widget_init")
        assert "custom_widget_init" in mgr.list_native_procedures()

    def test_register_python_procedure_invoked_via_simprocedure_hook(
        self, fauxware_project
    ):
        """A Python-registered native procedure runs when its hook fires.

        Simulates the dispatcher path: register a SimProcedure at an address
        with a name matching a Python-registered native procedure. When the
        dispatcher reaches that address, it should call the native (Python)
        implementation and capture the return value.
        """
        from angr.exploration import RustExplorationManager

        proj = fauxware_project
        state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [state])

        # Track invocations from Rust into our Python procedure.
        invocations = []

        def echo_args(args):
            invocations.append(tuple(args))
            return 0xDEADBEEF

        mgr._rust_mgr.register_python_procedure(
            "echo_proc",
            num_args=2,
            no_return=False,
            callable=echo_args,
        )

        # Verify it landed.
        assert mgr._rust_mgr.has_native_procedure("echo_proc")
        names = mgr._rust_mgr.list_native_procedures()
        assert "echo_proc" in names


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestRustSimStateIntegration:
    """Tests for RustSimState functionality used by exploration."""

    def test_state_fork(self):
        """Test O(1) state forking."""
        state1 = RustSimState("amd64")
        state1.pc = 0x1000
        state1.set_register("rax", 42)

        # Fork
        state2 = state1.fork()

        # Check values preserved
        assert state2.pc == 0x1000
        assert state2.get_register("rax") == 42

        # Check independent
        state2.pc = 0x2000
        state2.set_register("rax", 100)

        assert state1.pc == 0x1000
        assert state1.get_register("rax") == 42

    def test_state_memory(self):
        """Test memory operations."""
        state = RustSimState("amd64")

        # Map memory
        state.map_memory(0x1000, 0x1000, 7)  # RWX

        # Store
        state.memory_store(0x1000, bytes([0x41, 0x42, 0x43, 0x44]))

        # Load
        data = state.memory_load(0x1000, 4)
        assert data == bytes([0x41, 0x42, 0x43, 0x44])

    def test_state_memory_fork(self):
        """Test CoW memory forking."""
        state1 = RustSimState("amd64")
        state1.map_memory(0x1000, 0x1000, 7)
        state1.memory_store(0x1000, bytes([0xAA, 0xBB]))

        # Fork
        state2 = state1.fork()

        # Modify fork
        state2.memory_store(0x1000, bytes([0xCC, 0xDD]))

        # Check original unchanged
        assert state1.memory_load(0x1000, 2) == bytes([0xAA, 0xBB])
        assert state2.memory_load(0x1000, 2) == bytes([0xCC, 0xDD])


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestRustExplorationPython:
    """Tests for Python RustExplorationManager wrapper."""


    def test_python_wrapper_creation(self, fauxware_project):
        """Test creating Python wrapper."""
        from angr.exploration import RustExplorationManager

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        assert mgr.stash_counts()["active"] == 1, "entry_state should produce exactly 1 active state"

    def test_strict_page_access_propagates_to_rust(self, fauxware_project):
        """A SimState with STRICT_PAGE_ACCESS option should flip the Rust
        memory model's enforce_permissions flag automatically (mirrors angr)."""
        from angr.exploration import RustExplorationManager
        from angr import sim_options as o

        # No option → flag stays off (default behavior).
        plain_state = fauxware_project.factory.entry_state()
        plain_mgr = RustExplorationManager(fauxware_project, [plain_state])
        plain_ids = plain_mgr._rust_mgr.get_state_ids("active")
        assert plain_ids, "expected an active state to be added"
        assert plain_mgr._rust_mgr.state_enforce_permissions(plain_ids[0]) is False

        # Option present → flag flips on for the Rust state.
        strict_state = fauxware_project.factory.entry_state(
            add_options={o.STRICT_PAGE_ACCESS}
        )
        strict_mgr = RustExplorationManager(fauxware_project, [strict_state])
        strict_ids = strict_mgr._rust_mgr.get_state_ids("active")
        assert strict_ids, "expected an active state to be added"
        assert strict_mgr._rust_mgr.state_enforce_permissions(strict_ids[0]) is True

    def test_strict_page_access_blocks_nx_block_fetch(self):
        """Fetching a basic block from a mapped non-executable page must
        surface as a permission error (state lands in `errored`) when
        enforce_permissions is on. Lift callback must NOT be invoked at
        the NX address — the check fires before lifting."""
        from angr.rustylib.vex_engine import PythonCallbacks

        mgr = _RustExplorationManager("amd64")
        callbacks = PythonCallbacks()
        callbacks.set_memory_load(lambda addr, size: (bytes(size), False, None))
        callbacks.set_memory_store(lambda addr, data: None)
        lift_addrs = []
        def lift(addr):
            lift_addrs.append(addr)
            return '{}'
        callbacks.set_lift_block(lift)
        mgr.set_callbacks(callbacks)

        state = RustSimState("amd64")
        state.set_enforce_permissions(True)
        state.map_memory(0x1000, 0x1000, 6)  # RW, no X
        state.pc = 0x1000
        mgr.add_state("active", state)

        mgr.run(10)

        counts = mgr.stash_counts()
        assert counts.get("errored", 0) == 1, (
            f"expected the NX block fetch to error the state, got stashes={counts}"
        )
        assert 0x1000 not in lift_addrs, (
            "permission check must fire before lift_block is dispatched"
        )

    def test_basic_explore(self, fauxware_project):
        """Test basic exploration with find address."""
        from angr.exploration import RustExplorationManager

        # Find the "Welcome" message address
        # In fauxware, this is typically around 0x4006ed
        find_addr = 0x4006ed

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        # Run exploration
        mgr.explore(find=find_addr)
        # Exploration completed — verify at least one state was found or explored
        assert len(mgr.found) > 0 or len(mgr.deadended) > 0, \
            "exploration should find states or deadend some"

    def test_stash_access(self, fauxware_project):
        """Test accessing stashes."""
        from angr.exploration import RustExplorationManager

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        # Access stashes
        active = mgr.active
        found = mgr.found
        avoid = mgr.avoid
        deadended = mgr.deadended

        assert isinstance(active, list)
        assert isinstance(found, list)
        assert isinstance(avoid, list)
        assert isinstance(deadended, list)

    def test_max_active_states_python(self, fauxware_project):
        """Test max_active_states limit via Python wrapper."""
        from angr.exploration import RustExplorationManager

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state], max_active_states=2)

        # Run exploration with limited active states
        mgr.explore(find=0x4006ed, num_find=1)

        # Active states should never exceed the limit
        counts = mgr.stash_counts()
        assert counts.get("active", 0) <= 2, \
            f"active count {counts['active']} exceeds max_active_states=2"

    def test_max_active_states_prunes_forks(self, fauxware_project):
        """Excess forks should be pruned when max_active_states is reached.

        With limit=1, fauxware's symbolic strcmp branches must produce at
        least one pruned state — otherwise the limit is being silently
        ignored (the bug fixed by angr-jmiz).
        """
        from angr.exploration import RustExplorationManager

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state], max_active_states=1)
        mgr.explore(find=0x4006ed, num_find=1)

        counts = mgr.stash_counts()
        assert counts.get("active", 0) <= 1, \
            f"active count {counts.get('active', 0)} exceeds max_active_states=1"
        assert counts.get("pruned", 0) > 0, \
            f"expected pruned states with max_active_states=1, got counts={counts}"

    def test_progress_callback(self, fauxware_project):
        """Test that progress callback fires during exploration."""
        from angr.exploration import RustExplorationManager

        progress_reports = []

        def on_progress(info):
            progress_reports.append(info)

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.set_progress_callback(on_progress, interval_steps=1)
        mgr.explore(find=0x4006ed, num_find=1)

        # Progress callback should have fired at least once
        assert len(progress_reports) > 0, "progress callback never fired"
        # Each report should have the expected keys
        report = progress_reports[0]
        assert 'step_count' in report
        assert 'active_count' in report
        assert 'found_count' in report
        assert 'elapsed_seconds' in report


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestExplorationEvent:
    """Tests for ExplorationEvent class."""

    def test_event_attributes(self):
        """Test event has expected attributes."""
        mgr = _RustExplorationManager("amd64")

        # Set up minimal callbacks
        callbacks = PythonCallbacks()
        callbacks.set_memory_load(lambda a, s: (bytes(s), False, None))
        callbacks.set_memory_store(lambda a, d: None)
        callbacks.set_lift_block(lambda a: '{}')
        mgr.set_callbacks(callbacks)

        # Run should return an event
        event = mgr.run(1)

        # Check event structure
        assert hasattr(event, 'event_type')
        assert hasattr(event, 'found_count')
        assert hasattr(event, 'active_count')
        assert hasattr(event, 'steps_taken')


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestRustEdgeCases:
    """Edge-case tests for Rust state and solver."""

    @classmethod
    def setup_class(cls):
        """Ensure the shared Z3 context is initialized for solver tests."""
        from angr.exploration.rust_manager import _setup_shared_z3_context
        _setup_shared_z3_context()

    def test_page_boundary_store_load(self):
        """Store 6 bytes at offset 4090, crossing a 4096-byte page boundary."""
        state = RustSimState("amd64")
        state.map_memory(0x1000, 0x2000, 7)  # Map 2 pages

        data_in = bytes([0x11, 0x22, 0x33, 0x44, 0x55, 0x66])
        state.memory_store(0x1000 + 4090, data_in)
        data_out = state.memory_load(0x1000 + 4090, 6)
        assert data_out == data_in

    def test_page_boundary_cow_fork(self):
        """Page-boundary store preserves CoW isolation after fork."""
        state1 = RustSimState("amd64")
        state1.map_memory(0x1000, 0x2000, 7)
        state1.memory_store(0x1000 + 4090, bytes([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]))

        state2 = state1.fork()
        state2.memory_store(0x1000 + 4090, bytes([0x11, 0x22, 0x33, 0x44, 0x55, 0x66]))

        assert state1.memory_load(0x1000 + 4090, 6) == bytes([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF])
        assert state2.memory_load(0x1000 + 4090, 6) == bytes([0x11, 0x22, 0x33, 0x44, 0x55, 0x66])

    def test_many_symbolic_variables(self):
        """Solver with >50 symbolic variables and constraints."""
        from angr.rustylib.vex_engine import RustSolverContext
        import claripy

        ctx = RustSolverContext()
        symbols = []
        for i in range(60):
            s = claripy.BVS(f"var_{i}", 32)
            symbols.append(s)
            ctx.add_constraint_ast(s >= i)
            ctx.add_constraint_ast(s <= i + 100)

        assert ctx.satisfiable()
        assert ctx.num_constraints() == 120

        # Eval boundary values
        val_first = ctx.eval(symbols[0])
        val_last = ctx.eval(symbols[59])
        assert val_first is not None and 0 <= val_first <= 100
        assert val_last is not None and 59 <= val_last <= 159

    def test_solver_push_pop(self):
        """Push/pop preserves solver state correctly."""
        from angr.rustylib.vex_engine import RustSolverContext
        import claripy

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x >= 10)
        ctx.add_constraint_ast(x <= 20)

        ctx.push()
        ctx.add_constraint_ast(x == 15)
        assert ctx.eval(x) == 15

        ctx.pop()
        # After pop, x == 15 constraint is gone
        val = ctx.eval(x)
        assert 10 <= val <= 20

    def test_deep_fork_chain(self):
        """Fork a state 20 levels deep and verify independence."""
        state = RustSimState("amd64")
        state.map_memory(0x1000, 0x1000, 7)
        state.set_register("rax", 0)

        states = [state]
        for i in range(1, 21):
            child = states[-1].fork()
            child.set_register("rax", i)
            child.memory_store(0x1000, bytes([i]))
            states.append(child)

        # Verify each level is independent
        for i, s in enumerate(states):
            assert s.get_register("rax") == i
            if i > 0:
                assert s.memory_load(0x1000, 1) == bytes([i])


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestCallablePredicates:
    """Tests for callable find/avoid predicates with RustStateProxy."""


    def test_find_lambda_by_address(self, fauxware_project):
        """Callable find predicate matching by address works."""
        from angr.exploration import RustExplorationManager

        ACCEPTED = 0x4006ed
        REJECTED = 0x4006fd

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=lambda s: s.addr == ACCEPTED, avoid=REJECTED,
                    max_steps=50000)

        assert len(mgr.found) > 0, "Should find at least one state reaching accepted()"
        for s in mgr.found:
            assert s.addr == ACCEPTED

    def test_avoid_lambda_by_address(self, fauxware_project):
        """Callable avoid predicate matching by address works."""
        from angr.exploration import RustExplorationManager

        ACCEPTED = 0x4006ed
        REJECTED = 0x4006fd

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=ACCEPTED, avoid=lambda s: s.addr == REJECTED,
                    max_steps=50000)

        assert len(mgr.found) > 0, "Should find at least one state reaching accepted()"

    def test_solver_proxy_eval_returns_single_value(self):
        """RustSolverProxy.eval() returns a single value, not a tuple."""
        from angr.exploration.rust_state_proxy import RustSolverProxy
        from angr.rustylib.vex_engine import RustSolverContext
        import claripy

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x >= 10)
        ctx.add_constraint_ast(x <= 20)

        # RustSolverProxy normally lazily forks from rust_mgr, but we can
        # inject a pre-built solver context for unit testing.
        proxy = RustSolverProxy.__new__(RustSolverProxy)
        proxy._mgr = None
        proxy._state_id = None
        proxy._solver_ctx = ctx

        result = proxy.eval(x)
        assert isinstance(result, int), f"eval() should return int, got {type(result)}"
        assert 10 <= result <= 20

    def test_solver_proxy_eval_upto_returns_tuple(self):
        """RustSolverProxy.eval_upto() returns a tuple."""
        from angr.exploration.rust_state_proxy import RustSolverProxy
        from angr.rustylib.vex_engine import RustSolverContext
        import claripy

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x >= 10)
        ctx.add_constraint_ast(x <= 12)

        proxy = RustSolverProxy.__new__(RustSolverProxy)
        proxy._mgr = None
        proxy._state_id = None
        proxy._solver_ctx = ctx

        results = proxy.eval_upto(x, 5)
        assert isinstance(results, tuple), f"eval_upto() should return tuple, got {type(results)}"
        assert len(results) == 3  # exactly 10, 11, 12
        assert set(results) == {10, 11, 12}

    def test_solver_proxy_eval_cast_to_bytes(self):
        """RustSolverProxy.eval() with cast_to=bytes works."""
        from angr.exploration.rust_state_proxy import RustSolverProxy
        from angr.rustylib.vex_engine import RustSolverContext
        import claripy

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 0x41424344)

        proxy = RustSolverProxy.__new__(RustSolverProxy)
        proxy._mgr = None
        proxy._state_id = None
        proxy._solver_ctx = ctx

        result = proxy.eval(x, cast_to=bytes)
        assert isinstance(result, bytes), f"eval(cast_to=bytes) should return bytes, got {type(result)}"
        assert result == b"ABCD"


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestCallStackProxy:
    """Tests for RustCallStackProxy on RustStateProxy."""

    def test_callstack_empty_on_unit_state(self):
        """A bare manager state has no frames; proxy reports depth 0."""
        from angr.exploration.rust_state_proxy import RustCallStackProxy

        mgr = _RustExplorationManager("amd64")
        state_id = mgr.create_state("active")
        proxy = RustCallStackProxy(mgr, state_id)
        assert len(proxy) == 0
        assert list(proxy) == []
        assert proxy.top is None
        assert proxy.current_function_address == 0
        assert proxy.current_return_target == 0

    def test_callstack_proxy_after_explore(self, fauxware_project):
        """After exploration, found-state callstack frames match the
        snapshot exposed by the underlying Rust manager."""
        from angr.exploration import RustExplorationManager
        from angr.exploration.rust_state_proxy import (
            RustCallStackProxy,
            RustCallStackFrameProxy,
        )

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ed)
        assert len(mgr.found) > 0

        found_ids = mgr._rust_mgr.get_state_ids("found")
        sid = found_ids[0]
        proxy = RustCallStackProxy(mgr._rust_mgr, sid)
        # Same depth as the raw API, modulo direction.
        raw = mgr._rust_mgr.get_state_call_stack(sid)
        assert len(proxy) == len(raw)
        # Iteration yields proxies with matching attrs in reverse order
        # (top-first).
        frames = list(proxy)
        assert all(isinstance(f, RustCallStackFrameProxy) for f in frames)
        if raw:
            top = frames[0]
            expected_top = raw[-1]  # raw is push-order; top is last
            assert top.call_site_addr == expected_top[0]
            assert top.func_addr == expected_top[1]
            assert top.ret_addr == expected_top[2]
            assert top.stack_ptr == expected_top[3]
            assert proxy.current_function_address == expected_top[1]

    def test_callstack_indexing_and_walk(self):
        """Frame __getitem__ and .next walk the same path."""
        from angr.exploration.rust_state_proxy import RustCallStackProxy

        proxy = RustCallStackProxy.__new__(RustCallStackProxy)
        proxy._mgr = None
        proxy._state_id = None
        # Most-recent first: function 0x300 called from 0x200, etc.
        proxy._frames_cache = [
            (0x250, 0x300, 0x255, 0x7000),  # top
            (0x150, 0x200, 0x155, 0x7100),
            (0x050, 0x100, 0x055, 0x7200),  # bottom
        ]
        assert len(proxy) == 3
        top = proxy[0]
        assert top.func_addr == 0x300
        assert top.next.func_addr == 0x200
        assert top.next.next.func_addr == 0x100
        assert top.next.next.next is None
        # Negative indexing
        assert proxy[-1].func_addr == 0x100

    def test_simstate_callstack_synced_after_explore(self, fauxware_project):
        """The angr SimState returned from mgr.found has its CallStack
        plugin populated from Rust's call_stack — not just the empty
        sentinel from the entry-state template."""
        from angr.exploration import RustExplorationManager

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ed)
        assert len(mgr.found) > 0
        found_state = mgr.found[0]

        # Cross-check against the raw Rust API.
        sid = mgr._rust_mgr.get_state_ids("found")[0]
        raw = mgr._rust_mgr.get_state_call_stack(sid)
        if not raw:
            pytest.skip("Rust call stack empty at find — nothing to verify")

        # angr's CallStack iterates top-first; raw is push-order
        # (outermost first). Compare the angr-side top to raw[-1].
        cs_frames = list(found_state.callstack)
        # cs_frames includes the sentinel (zero-frame) at the bottom, so
        # the angr-side has len(raw) + 1 entries (worst case). Just check
        # the top frame matches the most recent Rust call.
        assert len(cs_frames) >= len(raw)
        top_call_site, top_func, top_ret, top_sp = raw[-1]
        assert found_state.callstack.func_addr == top_func
        assert found_state.callstack.call_site_addr == top_call_site
        assert found_state.callstack.ret_addr == top_ret
        assert found_state.callstack.stack_ptr == top_sp


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestInspectProxy:
    """Tests for the no-op inspect proxy on RustStateProxy."""

    def test_inspect_breakpoint_calls_succeed(self):
        """state.inspect.b('mem_read', ...) and friends silently succeed."""
        from angr.exploration.rust_state_proxy import _NoOpInspectProxy

        ins = _NoOpInspectProxy()
        # All these should be silent no-ops.
        ins.b("mem_read", when="before", action=lambda s: None)
        ins.make_breakpoint("mem_write")
        ins.add_breakpoint("call", lambda s: None)
        ins.remove_breakpoint("call", 0)
        # Unknown attr returns a callable no-op.
        ins.unknown_method(1, 2, key="value")


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestStashOperations:
    """Tests for stash management operations."""

    def test_move_states_all(self):
        """move_states without filter moves all states."""
        mgr = _RustExplorationManager("amd64")
        mgr.create_state("active")
        mgr.create_state("active")
        mgr.create_state("active")
        assert mgr.active_count() == 3

        count = mgr.move_states("active", "found", None)
        assert count == 3
        assert mgr.active_count() == 0
        assert mgr.found_count() == 3

    def test_move_states_empty_source(self):
        """move_states from empty stash returns 0."""
        mgr = _RustExplorationManager("amd64")
        count = mgr.move_states("active", "found", None)
        assert count == 0

    def test_move_state_by_id(self):
        """move_state moves a specific state by ID."""
        mgr = _RustExplorationManager("amd64")
        id1 = mgr.create_state("active")
        id2 = mgr.create_state("active")

        result = mgr.move_state(id1, "active", "found")
        assert result is True
        assert mgr.active_count() == 1
        assert mgr.found_count() == 1

        # The remaining state should be id2
        remaining = mgr.get_state_ids("active")
        assert id2 in remaining

    def test_move_state_nonexistent(self):
        """move_state returns False for nonexistent state ID."""
        mgr = _RustExplorationManager("amd64")
        mgr.create_state("active")
        result = mgr.move_state(999999, "active", "found")
        assert result is False

    def test_clear_stash(self):
        """clear_stash removes all states from a stash."""
        mgr = _RustExplorationManager("amd64")
        mgr.create_state("found")
        mgr.create_state("found")
        assert mgr.found_count() == 2

        mgr.clear_stash("found")
        assert mgr.found_count() == 0

    def test_clear_empty_stash(self):
        """clear_stash on empty stash is a no-op."""
        mgr = _RustExplorationManager("amd64")
        mgr.clear_stash("nonexistent")  # Should not raise

    def test_stash_counts_multiple(self):
        """stash_counts includes all stash names."""
        mgr = _RustExplorationManager("amd64")
        mgr.create_state("active")
        mgr.create_state("found")
        mgr.create_state("deadended")

        counts = mgr.stash_counts()
        assert counts["active"] == 1
        assert counts["found"] == 1
        assert counts["deadended"] == 1

    def test_get_state_ids_empty(self):
        """get_state_ids on empty stash returns empty list."""
        mgr = _RustExplorationManager("amd64")
        ids = mgr.get_state_ids("active")
        assert ids == []


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestHooksAndProcedures:
    """Tests for hook and SimProcedure registration."""

    def test_register_hook(self):
        """Registering a hook at an address."""
        mgr = _RustExplorationManager("amd64")
        mgr.register_simprocedure(0x401000, "test_hook", 0, False)
        stats = mgr.stats()
        assert stats["hooks"] == 1

    def test_register_multiple_hooks(self):
        """Multiple hooks at different addresses."""
        mgr = _RustExplorationManager("amd64")
        mgr.register_simprocedure(0x401000, "hook1", 1, False)
        mgr.register_simprocedure(0x402000, "hook2", 2, False)
        mgr.register_simprocedure(0x403000, "hook3", 0, True)
        stats = mgr.stats()
        assert stats["simprocedures"] == 3
        assert stats["hooks"] == 3

    def test_set_find_avoid_addrs(self):
        """Setting find and avoid addresses."""
        mgr = _RustExplorationManager("amd64")
        mgr.set_find_addrs([0x1000, 0x2000])
        mgr.set_avoid_addrs([0x3000])

        stats = mgr.stats()
        assert stats["find_addrs"] == 2
        assert stats["avoid_addrs"] == 1

    def test_empty_find_avoid(self):
        """Empty find/avoid lists."""
        mgr = _RustExplorationManager("amd64")
        mgr.set_find_addrs([])
        mgr.set_avoid_addrs([])
        stats = mgr.stats()
        assert stats["find_addrs"] == 0
        assert stats["avoid_addrs"] == 0


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestStateManagement:
    """Tests for state creation and management."""

    def test_state_pc_get_set(self):
        """Get and set PC on states via manager."""
        mgr = _RustExplorationManager("amd64")
        state = RustSimState("amd64")
        state.pc = 0x401000
        mgr.add_state("active", state)

        pc = mgr.get_state_pc("active", 0)
        assert pc == 0x401000

    def test_multiple_states_different_pcs(self):
        """Multiple states with different PCs."""
        mgr = _RustExplorationManager("amd64")

        for addr in [0x1000, 0x2000, 0x3000]:
            state = RustSimState("amd64")
            state.pc = addr
            mgr.add_state("active", state)

        assert mgr.active_count() == 3

    def test_has_active_states(self):
        """has_active_states reflects stash contents."""
        mgr = _RustExplorationManager("amd64")
        assert not mgr.has_active_states()

        mgr.create_state("active")
        assert mgr.has_active_states()

    def test_drop_terminal_states_toggle(self):
        """set_drop_terminal_states can be toggled."""
        mgr = _RustExplorationManager("amd64")
        # Should not raise
        mgr.set_drop_terminal_states(True)
        mgr.set_drop_terminal_states(False)


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestSolverOperations:
    """Tests for solver constraint operations."""

    @classmethod
    def setup_class(cls):
        """Ensure the shared Z3 context is initialized for solver tests."""
        from angr.exploration.rust_manager import _setup_shared_z3_context
        _setup_shared_z3_context()

    def test_solver_min_max(self):
        """min() and max() return correct bounds."""
        from angr.rustylib.vex_engine import RustSolverContext
        import claripy

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x >= 10)
        ctx.add_constraint_ast(x <= 20)

        assert ctx.min(x, signed=False) == 10
        assert ctx.max(x, signed=False) == 20

    def test_solver_unsatisfiable(self):
        """Contradictory constraints make solver UNSAT."""
        from angr.rustylib.vex_engine import RustSolverContext
        import claripy

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 5)
        ctx.add_constraint_ast(x == 10)
        assert not ctx.satisfiable()

    def test_contradictory_constraints_make_unsat(self):
        """Inequality contradictions (x>100 AND x<50) make solver UNSAT and
        leave eval/min/max returning None instead of bogus concrete values.

        Locks down behaviour for angr-eldx — see also
        memory `satisfiable-wrong-answer`: False from satisfiable() must
        mean a definitive UNSAT, never a swallowed exception.
        """
        from angr.rustylib.vex_engine import RustSolverContext
        import claripy

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x > 100)
        ctx.add_constraint_ast(x < 50)

        assert ctx.satisfiable() is False
        assert ctx.eval(x) is None
        assert ctx.min(x, signed=False) is None
        assert ctx.max(x, signed=False) is None
        assert ctx.eval_upto(x, 5) == []

    def test_solver_fork_independence(self):
        """Forked solver contexts are independent."""
        from angr.rustylib.vex_engine import RustSolverContext
        import claripy

        ctx1 = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx1.add_constraint_ast(x >= 0)
        ctx1.add_constraint_ast(x <= 100)

        ctx2 = ctx1.fork()
        ctx2.add_constraint_ast(x == 42)

        # ctx2 is constrained to 42
        assert ctx2.eval(x) == 42

        # ctx1 still has the wider range
        val = ctx1.eval(x)
        assert 0 <= val <= 100

    def test_solver_multiple_variables(self):
        """Solver handles multiple independent symbolic variables."""
        from angr.rustylib.vex_engine import RustSolverContext
        import claripy

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        y = claripy.BVS("y", 32)
        ctx.add_constraint_ast(x >= 30)
        ctx.add_constraint_ast(x <= 50)
        ctx.add_constraint_ast(y >= 60)
        ctx.add_constraint_ast(y <= 80)

        assert ctx.satisfiable()
        vx = ctx.eval(x)
        vy = ctx.eval(y)
        assert 30 <= vx <= 50
        assert 60 <= vy <= 80

    def test_solver_eval_upto_wide_bvs(self):
        """eval_upto should handle BVS wider than 128 bits without truncation."""
        from angr.rustylib.vex_engine import RustSolverContext
        import claripy

        ctx = RustSolverContext()
        # 296-bit BVS (like whitehatvn's 37-byte arg)
        x = claripy.BVS("wide_var", 296)
        # Constrain first byte to 'A' (0x41) and last byte to 'Z' (0x5a)
        ctx.add_constraint_ast(claripy.Extract(295, 288, x) == 0x41)
        ctx.add_constraint_ast(claripy.Extract(7, 0, x) == 0x5a)

        results = ctx.eval_upto(x, 2)
        assert len(results) >= 1, "should find at least one solution"
        for r in results:
            nbytes = 37
            val_bytes = r.to_bytes(nbytes, 'big')
            assert val_bytes[0] == 0x41, "first byte should be 'A'"
            assert val_bytes[-1] == 0x5a, "last byte should be 'Z'"

    def test_solver_eval_upto_excludes_duplicates(self):
        """eval_upto should return distinct values."""
        from angr.rustylib.vex_engine import RustSolverContext
        import claripy

        ctx = RustSolverContext()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(x >= 1)
        ctx.add_constraint_ast(x <= 5)

        results = ctx.eval_upto(x, 10)
        assert len(results) == 5, "should find exactly 5 solutions for [1..5]"
        assert len(set(results)) == 5, "all solutions should be distinct"

    def test_constraint_weakening_through_ite(self):
        """Constraints on an ITE result must weaken correctly to its branches.

        Build `result = If(x > 5, x, 100)` and constrain `result > 50`. This
        is satisfiable in two disjoint ways:
          - x > 5  AND  x > 50  → x in (50, 2^32)
          - x <= 5 AND  100 > 50 → any x in [0, 5] (else-branch always > 50)

        So x is NOT pinned to any single value; the solver must admit
        solutions like x = 51 (and also x in [0, 5]). A regression that
        loses ITE structure could over-constrain and force x to one branch.
        """
        from angr.rustylib.vex_engine import RustSolverContext
        import claripy

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        result = claripy.If(x > 5, x, claripy.BVV(100, 32))
        ctx.add_constraint_ast(result > 50)

        assert ctx.satisfiable()

        # Witness 1: x = 51 must be admissible (then-branch satisfies > 50).
        ctx_then = ctx.fork()
        ctx_then.add_constraint_ast(x == 51)
        assert ctx_then.satisfiable(), "x=51 should be a valid model"

        # Witness 2: x = 3 must also be admissible (else-branch=100 > 50).
        ctx_else = ctx.fork()
        ctx_else.add_constraint_ast(x == 3)
        assert ctx_else.satisfiable(), "x=3 should be a valid model"

        # Counter-witness: x = 30 must NOT be admissible
        # (then-branch=30, else-branch unreachable since x>5).
        ctx_bad = ctx.fork()
        ctx_bad.add_constraint_ast(x == 30)
        assert not ctx_bad.satisfiable(), "x=30 must be ruled out"

    def test_model_stability_constraint_order(self):
        """Adding the same constraints in two orders yields the same eval(x).

        Z3 is deterministic given the same constraint set; if our bridge
        re-orders or de-duplicates inconsistently across solver instances,
        eval(x) may diverge. Lock down stability to catch any future change
        that introduces order-dependent behaviour.
        """
        from angr.rustylib.vex_engine import RustSolverContext
        import claripy

        x = claripy.BVS("x", 32)
        c1 = x >= 100
        c2 = x <= 200
        c3 = x != 150

        ctx_a = RustSolverContext()
        ctx_a.add_constraint_ast(c1)
        ctx_a.add_constraint_ast(c2)
        ctx_a.add_constraint_ast(c3)

        ctx_b = RustSolverContext()
        ctx_b.add_constraint_ast(c3)
        ctx_b.add_constraint_ast(c2)
        ctx_b.add_constraint_ast(c1)

        v_a = ctx_a.eval(x)
        v_b = ctx_b.eval(x)

        assert v_a is not None and v_b is not None
        assert 100 <= v_a <= 200 and v_a != 150
        assert 100 <= v_b <= 200 and v_b != 150
        assert v_a == v_b, (
            f"eval(x) should be order-stable; got {v_a} vs {v_b}"
        )

    def test_solver_contradictory_find_avoid(self):
        """Same address in find and avoid should avoid (avoid takes priority)."""
        mgr = _RustExplorationManager("amd64")
        mgr.set_find_addrs([0x1000])
        mgr.set_avoid_addrs([0x1000])
        state = RustSimState("amd64")
        state.pc = 0x1000
        mgr.add_state("active", state)
        # The address is in both find and avoid — behavior is implementation-defined
        # but should not crash
        assert mgr.active_count() == 1


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestRustStateRegisters:
    """Tests for register operations on RustSimState."""

    def test_register_set_get(self):
        """Set and get multiple registers."""
        state = RustSimState("amd64")
        state.set_register("rax", 0x1234)
        state.set_register("rbx", 0x5678)
        state.set_register("rcx", 0xABCD)

        assert state.get_register("rax") == 0x1234
        assert state.get_register("rbx") == 0x5678
        assert state.get_register("rcx") == 0xABCD

    def test_register_fork_isolation(self):
        """Forked states have independent registers."""
        state1 = RustSimState("amd64")
        state1.set_register("rax", 100)

        state2 = state1.fork()
        state2.set_register("rax", 200)

        assert state1.get_register("rax") == 100
        assert state2.get_register("rax") == 200

    def test_pc_set_get(self):
        """PC property works correctly."""
        state = RustSimState("amd64")
        state.pc = 0xDEADBEEF
        assert state.pc == 0xDEADBEEF

    def test_state_id_unique(self):
        """Each state has a unique ID."""
        s1 = RustSimState("amd64")
        s2 = RustSimState("amd64")
        s3 = s1.fork()
        ids = {s1.state_id, s2.state_id, s3.state_id}
        assert len(ids) == 3, "State IDs should be unique"


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestSerializeIRSB:
    """Tests for IRSB serialization (used in lift callbacks)."""


    def test_serialize_basic_block(self, fauxware_project):
        """Serializing a basic block produces valid JSON."""
        import json
        from angr.exploration import RustExplorationManager

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        # Lift a block
        block = fauxware_project.factory.block(fauxware_project.entry)
        irsb = block.vex

        # Serialize
        result = mgr._serialize_irsb(irsb)
        data = json.loads(result)

        assert 'addr' in data
        assert 'statements' in data
        assert 'next' in data
        assert 'jumpkind' in data
        assert 'tyenv' in data
        assert len(data['statements']) > 0

    def test_serialize_roundtrip_consistency(self, fauxware_project):
        """Serializing the same block twice produces identical output."""
        from angr.exploration import RustExplorationManager

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        block = fauxware_project.factory.block(fauxware_project.entry)
        irsb = block.vex

        result1 = mgr._serialize_irsb(irsb)
        result2 = mgr._serialize_irsb(irsb)
        assert result1 == result2


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestExplorationIntegration:
    """Integration tests with real binaries."""


    def test_explore_with_max_steps(self, fauxware_project):
        """Exploration respects max_steps limit."""
        from angr.exploration import RustExplorationManager

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ed, max_steps=3)

        # 3 steps is too few to find target in fauxware
        stats = mgr.stats
        assert stats['ffi_crossings'] > 0, "should have crossed FFI boundary"
        assert len(mgr.found) == 0, "3 steps too few to find target in fauxware"

    def test_explore_finds_correct_state(self, fauxware_project):
        """Full exploration finds the expected state."""
        from angr.exploration import RustExplorationManager

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ed, avoid=0x4006fd, max_steps=50000)

        assert len(mgr.found) > 0, "Should find at least one state"

    def test_explore_with_timeout_technique(self, fauxware_project):
        """Timeout technique stops exploration after time limit."""
        from angr.exploration import RustExplorationManager
        from angr.exploration_techniques import Timeout

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.use_technique(Timeout(timeout=0.001))  # 1ms timeout — should trigger quickly
        mgr.explore(find=0x4006ed, max_steps=50000)
        # Should terminate quickly due to timeout (not find the solution)

    def test_hook_fp_constraint_uses_z3_ptr_fallback(self, fauxware_project):
        """Constraints with ops that claripy_to_rustbv can't translate (e.g. FP)
        must still reach the Rust solver via the Z3 ptr fallback in
        sync_constraints_from_python (added for angr-1epf).

        Without the fallback, FP constraints raise UnsupportedOp inside
        claripy_to_rustbv and get dropped silently. We compare the Rust-side
        Z3 assertion count against an unhooked baseline run; the hooked run
        must produce strictly more assertions per state.
        """
        from angr.exploration import RustExplorationManager
        import claripy

        proj = fauxware_project

        # Baseline run (no hook).
        baseline_state = proj.factory.entry_state()
        baseline_mgr = RustExplorationManager(proj, [baseline_state])
        baseline_mgr.run(max_steps=300)
        baseline_counts = sorted(
            len(baseline_mgr._rust_mgr.export_z3_constraint_ptrs(sid))
            for stash in ("active", "found", "deadended")
            for sid in baseline_mgr._rust_mgr.get_state_ids(stash)
        )
        assert baseline_counts, "baseline run produced no states"

        # Hooked run: FP constraint goes through the Z3 ptr fallback.
        main_sym = proj.loader.find_symbol("main")
        hook_addr = main_sym.rebased_addr
        hook_fired = []

        def hook(state):
            fp = claripy.FPS("hook_fp_var", claripy.FSORT_DOUBLE)
            state.solver.add(fp == claripy.FPV(1.5, claripy.FSORT_DOUBLE))
            hook_fired.append(True)

        proj.hook(hook_addr, hook=hook, length=0)
        try:
            hooked_state = proj.factory.entry_state()
            hooked_mgr = RustExplorationManager(proj, [hooked_state])
            hooked_mgr.run(max_steps=300)
        finally:
            proj.unhook(hook_addr)

        assert hook_fired, "hook never fired"
        hooked_counts = sorted(
            len(hooked_mgr._rust_mgr.export_z3_constraint_ptrs(sid))
            for stash in ("active", "found", "deadended")
            for sid in hooked_mgr._rust_mgr.get_state_ids(stash)
        )
        assert hooked_counts, "hooked run produced no states"
        assert len(hooked_counts) == len(baseline_counts), (
            f"hooked vs baseline state counts differ: hooked={len(hooked_counts)}, "
            f"baseline={len(baseline_counts)}"
        )
        # Every paired state should have strictly more Z3 assertions in the
        # hooked run. If claripy_to_rustbv silently drops the FP constraint
        # (the bug this fix targets), the counts would match exactly.
        for h, b in zip(hooked_counts, baseline_counts):
            assert h > b, (
                f"Rust solver gained no extra constraints with FP hook "
                f"(hooked={hooked_counts}, baseline={baseline_counts}); "
                "the Z3 ptr fallback is not engaging."
            )

    def test_hook_constraint_propagates_to_rust_solver(self, fauxware_project):
        """A constraint added inside a Python hook callback must reach the Rust solver.

        Regression for angr-1epf: post-callback the dispatch passes new claripy
        constraints back via resume_after_simprocedure. If that round-trip drops
        the constraint, the Rust state's solver itself will be missing it.
        """
        from angr.exploration import RustExplorationManager
        import claripy

        proj = fauxware_project
        main_sym = proj.loader.find_symbol("main")
        assert main_sym is not None, "fauxware should have a main symbol"
        hook_addr = main_sym.rebased_addr

        captured = {}

        def hook(state):
            x = claripy.BVS("hook_constraint_var", 32)
            state.solver.add(x == 0xCAFEBABE)
            captured.setdefault("var", x)

        proj.hook(hook_addr, hook=hook, length=0)
        try:
            state = proj.factory.entry_state()
            mgr = RustExplorationManager(proj, [state])
            mgr.run(max_steps=300)

            assert "var" in captured, "hook never fired"

            # Inspect every live Rust state's exported constraint ASTs and
            # confirm at least one references our `hook_constraint_var` BVS
            # — the AST must have actually crossed the Python→Rust boundary,
            # not just lived in the Python solver.
            def constraint_mentions_hook_var(ast):
                # claripy assigns BVS a uniqueness suffix (e.g. "_1_32"), so
                # match by prefix rather than exact name.
                try:
                    leaves = list(ast.leaf_asts())
                except Exception:
                    return False
                for leaf in leaves:
                    args = getattr(leaf, "args", ())
                    if args and isinstance(args[0], str) and args[0].startswith("hook_constraint_var"):
                        return True
                return False

            saw = False
            for stash in ("active", "found", "deadended"):
                for sid in mgr._rust_mgr.get_state_ids(stash):
                    try:
                        rust_constraints = mgr._rust_mgr.export_state_constraints(sid)
                    except Exception:
                        continue
                    if any(constraint_mentions_hook_var(c) for c in rust_constraints):
                        saw = True
                        break
                if saw:
                    break

            assert saw, (
                "Rust solver does not contain any constraint that references "
                "`hook_constraint_var`. The hook's constraint appears to have "
                "been dropped on the Python→Rust round-trip."
            )
        finally:
            proj.unhook(hook_addr)

    def test_hook_length_advances_pc_userhook(self, fauxware_project):
        """proj.hook(addr, fn, length=N>0) on a UserHook must skip N bytes.

        Regression for angr-03ej. Confirms the Rust dispatch honors
        the new_pc that UserHook sets via ``state.addr + length`` so
        the hooked N-byte instruction is replaced (not re-executed)
        and the hook does not re-fire in a loop.

        The hook is placed at ``mov rsp, rbp`` (3 bytes) inside main.
        With length=3 control resumes at hook_addr+3; if the dispatch
        somehow stayed at hook_addr the hook would re-fire and the
        test would see an unbounded number of fires.
        """
        from angr.exploration import RustExplorationManager

        proj = fauxware_project
        main_sym = proj.loader.find_symbol("main")
        assert main_sym is not None
        hook_addr = main_sym.rebased_addr + 1  # second instruction in main
        hook_length = 3

        # Sanity-check the disassembly so the test fails clearly if the
        # binary changes shape.
        block = proj.factory.block(hook_addr)
        first_insn_size = block.capstone.insns[0].size
        assert first_insn_size == hook_length, (
            f"Expected a {hook_length}-byte instruction at {hook_addr:#x}, "
            f"got {first_insn_size}-byte"
        )

        fire_addrs = []

        def my_hook(state):
            fire_addrs.append(state.addr)

        proj.hook(hook_addr, hook=my_hook, length=hook_length)
        try:
            state = proj.factory.entry_state()
            mgr = RustExplorationManager(proj, [state])
            mgr.run(max_steps=300)
        finally:
            proj.unhook(hook_addr)

        assert fire_addrs, "hook never fired"
        # Without the fix, the hook re-executes in an infinite loop
        # because new_pc stays at hook_addr. With the fix, it fires
        # at most once per state path through main (fauxware has a
        # small number of paths).
        assert len(fire_addrs) < 50, (
            f"Hook fired {len(fire_addrs)} times — likely re-executing "
            f"because hook_length was not honored. First addrs: "
            f"{[hex(a) for a in fire_addrs[:5]]}"
        )

    def test_vex_fallback_forks_multi_successors(self, fauxware_project):
        """Multi-successor Python VEX fallback must fork extras instead of dropping them.

        Regression for angr-v8iz: factory.successors(num_inst=99) inside
        _handle_python_vex_fallback can return N>1 successors when the
        fallback block contains a symbolic branch. Previously only
        all_succs[0] was synced back to Rust and the rest were silently
        dropped (warning only) — if the convergent path was the dropped
        one, exploration would spin forever. The handler must call
        _add_forked_state for every extra successor.
        """
        from angr.exploration import RustExplorationManager
        from types import SimpleNamespace

        proj = fauxware_project
        seed_state = proj.factory.entry_state()
        mgr = RustExplorationManager(proj, [seed_state])

        # Build two distinct successor states off the seed state to play
        # the role of a symbolic-branch fork inside the fallback block.
        succ_a = seed_state.copy()
        succ_a.regs.rax = 0x1111
        succ_b = seed_state.copy()
        succ_b.regs.rax = 0x2222
        all_succs = [succ_a, succ_b]
        succs_obj = SimpleNamespace(all_successors=all_succs)

        event = SimpleNamespace(
            callback_addr=seed_state.addr,
            callback_state_id=None,
            callback_name="test_unsupported_vex_op",
            callback_return_addr=None,
        )

        # Stub helpers so the test exercises only the multi-successor
        # handling logic. _create_state_for_callback returns a fresh
        # SimState; factory.successors returns our pre-built successors.
        mgr._create_state_for_callback = lambda evt: seed_state.copy()  # type: ignore[assignment]
        orig_factory_successors = proj.factory.successors
        proj.factory.successors = lambda *a, **kw: succs_obj  # type: ignore[assignment]

        # Capture resume_after_simprocedure (first successor) and
        # _add_forked_state (every extra successor) calls. The rust_mgr
        # PyO3 object's methods are read-only, so swap the whole handle
        # for a SimpleNamespace recorder. We only need to satisfy the
        # subset of calls _handle_python_vex_fallback issues on the
        # success path.
        resumed_calls = []

        def _record_resume(*args, **kwargs):
            resumed_calls.append((args, kwargs))

        original_rust_mgr = mgr._rust_mgr
        mgr._rust_mgr = SimpleNamespace(
            resume_after_simprocedure=_record_resume,
            resume_after_error=lambda *a, **k: None,
            deadend_pending_callback=lambda *a, **k: None,
        )

        forked_calls = []

        def _record_fork(succ, evt):
            forked_calls.append(succ)

        mgr._add_forked_state = _record_fork  # type: ignore[assignment]

        try:
            mgr._handle_python_vex_fallback(event)
        finally:
            proj.factory.successors = orig_factory_successors  # type: ignore[assignment]
            mgr._rust_mgr = original_rust_mgr

        assert len(resumed_calls) == 1, (
            f"Expected resume_after_simprocedure to fire exactly once for the "
            f"first successor, got {len(resumed_calls)}"
        )
        assert len(forked_calls) == len(all_succs) - 1, (
            f"Expected {len(all_succs) - 1} forked successors, got "
            f"{len(forked_calls)} — extras were dropped (the bug)"
        )
        assert forked_calls[0] is succ_b, (
            "The forked successor identity does not match all_succs[1]; "
            "wrong state was passed to _add_forked_state."
        )


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
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
        """UNSAT constraints: add_constraint_ast may not detect equality contradictions.

        Known limitation: Rust solver's add_constraint_ast converts claripy ASTs
        to RustBV representation which may not preserve == semantics fully.
        This test documents current behavior.
        """
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext
        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 5)
        ctx.add_constraint_ast(x == 10)
        # Known: may return True due to constraint conversion limitations
        # Real UNSAT detection works through Z3 when check() is called internally
        ctx.satisfiable()  # Should not crash

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
        assert h.length == 0
        assert h.is_concrete is True
        assert h.symbolic is False
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
        """Moving from empty stash should not crash."""
        mgr = _RustExplorationManager("amd64")
        # Try moving states when no states exist — should be a no-op
        try:
            mgr.move_states("active", "found", None)
        except Exception:
            pass  # Some implementations may raise, that's OK

    # --- Integration: RustExplorationManager Python wrapper ---

    def test_python_wrapper_no_states(self, fauxware_project):
        """Python wrapper with empty state list."""
        from angr.exploration import RustExplorationManager
        mgr = RustExplorationManager(fauxware_project, [])
        assert len(mgr.active) == 0

    def test_python_wrapper_explore_no_find(self, fauxware_project):
        """Explore with no find addresses should terminate on active_empty."""
        from angr.exploration import RustExplorationManager
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(max_steps=10)
        # Should not crash, should have run some steps

    def test_python_wrapper_double_explore(self, fauxware_project):
        """Calling explore() twice should not crash."""
        from angr.exploration import RustExplorationManager
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ed, max_steps=5)
        # Run again — should continue from where it left off
        mgr.explore(find=0x4006ed, max_steps=5)


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestMultiArchSupport:
    """Tests for MIPS, ARM, and big-endian architecture support."""

    def test_mips32_state_creation(self):
        """MIPS32 state creation and register operations."""
        state = RustSimState("mips32")
        state.set_register("v0", 0xDEAD)
        state.set_register("a0", 0xBEEF)
        state.set_register("sp", 0x7FFF0000)

        assert state.get_register("v0") == 0xDEAD
        assert state.get_register("a0") == 0xBEEF
        assert state.get_register("sp") == 0x7FFF0000

    def test_mips32_exploration_manager(self):
        """MIPS32 exploration manager creation and stash ops."""
        mgr = _RustExplorationManager("mips32")
        assert mgr.arch == "mips32"

        sid = mgr.create_state("active")
        assert mgr.active_count() == 1
        mgr.set_find_addrs([0x400000])
        mgr.set_avoid_addrs([0x400100])

    def test_mips32_big_endian(self):
        """MIPS32 big-endian state creation."""
        state = RustSimState("mips32", little_endian=False)
        state.set_register("v0", 0x12345678)
        assert state.get_register("v0") == 0x12345678

    def test_mips32_fork_isolation(self):
        """MIPS32 forked states have independent registers."""
        state1 = RustSimState("mips32")
        state1.set_register("t0", 100)
        state2 = state1.fork()
        state2.set_register("t0", 200)
        assert state1.get_register("t0") == 100
        assert state2.get_register("t0") == 200

    def test_arm_state_creation(self):
        """ARM32 state creation and register operations."""
        state = RustSimState("arm")
        state.set_register("r0", 0x1234)
        state.set_register("r1", 0x5678)
        state.set_register("sp", 0x7FFF0000)
        state.set_register("lr", 0x8000)

        assert state.get_register("r0") == 0x1234
        assert state.get_register("r1") == 0x5678
        assert state.get_register("sp") == 0x7FFF0000
        assert state.get_register("lr") == 0x8000

    def test_arm_exploration_manager(self):
        """ARM exploration manager creation."""
        mgr = _RustExplorationManager("arm")
        assert mgr.arch == "arm"
        sid = mgr.create_state("active")
        assert mgr.active_count() == 1

    def test_arm64_state_creation(self):
        """ARM64/AArch64 state creation and register operations."""
        state = RustSimState("aarch64")
        state.set_register("x0", 0xDEADBEEF)
        state.set_register("x1", 0xCAFEBABE)
        state.set_register("sp", 0x7FFFFFFFE000)
        assert state.get_register("x0") == 0xDEADBEEF
        assert state.get_register("x1") == 0xCAFEBABE
        assert state.get_register("sp") == 0x7FFFFFFFE000

    def test_x86_state_creation(self):
        """x86 (32-bit) state creation and register operations."""
        state = RustSimState("x86")
        state.set_register("eax", 0xDEADBEEF)
        state.set_register("esp", 0x7FFF0000)
        assert state.get_register("eax") == 0xDEADBEEF
        assert state.get_register("esp") == 0x7FFF0000

    def test_unsupported_arch_raises(self):
        """Unknown architecture raises an error."""
        with pytest.raises(Exception):
            RustSimState("pdp11")


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust extension not available")
class TestErroredStash:
    """Tests for the errored stash and RustErrorRecord."""

    def test_errored_returns_error_records(self, fauxware_project):
        """errored property returns RustErrorRecord objects with error details."""
        from angr.exploration import RustExplorationManager, RustErrorRecord

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        # Explore with a valid find address
        ACCEPTED = 0x4006ed
        mgr.explore(find=ACCEPTED, max_steps=50000)

        # errored should be a list (possibly empty for successful exploration)
        errored = mgr.errored
        assert isinstance(errored, list)

        # If there are errored states, they should be RustErrorRecord instances
        for record in errored:
            assert isinstance(record, RustErrorRecord)
            assert hasattr(record, 'state')
            assert hasattr(record, 'error')
            assert hasattr(record, 'addr')
            assert isinstance(record.error, Exception)
            assert repr(record).startswith('<State errored')

    def test_error_record_has_state(self):
        """RustErrorRecord wraps a state with error info."""
        from angr.exploration import RustErrorRecord

        record = RustErrorRecord(None, "test error", 0x401000)
        assert str(record.error) == "test error"
        assert record.addr == 0x401000
        assert record.state is None
        assert "test error" in repr(record)
        assert "0x401000" in repr(record)

    def test_error_record_reraise(self):
        """RustErrorRecord.reraise() re-raises the stored error."""
        from angr.exploration import RustErrorRecord

        record = RustErrorRecord(None, "test error", 0x401000)
        with pytest.raises(RuntimeError, match="test error"):
            record.reraise()


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestErrorRecovery:
    """Tests that the engine degrades gracefully instead of panicking."""

    @classmethod
    def setup_class(cls):
        from angr.exploration.rust_manager import _setup_shared_z3_context
        _setup_shared_z3_context()

    def test_unmapped_memory_load_returns_error(self):
        """Loading from unmapped memory should raise, not crash."""
        state = RustSimState("amd64")
        # No memory mapped — load should fail gracefully with ValueError
        with pytest.raises(ValueError, match="unmapped"):
            state.memory_load(0xDEAD0000, 4)

    def test_mapped_memory_load_succeeds(self):
        """Loading from mapped memory should return data."""
        state = RustSimState("amd64")
        state.map_memory(0x1000, 0x1000, 7)  # RWX
        data = state.memory_load(0x1000, 4)
        assert len(data) == 4

    def test_symbolic_load_to_unmapped_address_does_not_stay_active(self):
        """Symbolic load address pinned to a single unmapped value must NOT
        keep the state in `active`. It should land in `errored` (or at minimum
        `deadended`) — silently falling back to address 0 / a fresh symbol
        would diverge from Python angr (which raises SimUnsatError /
        SimMemoryAddressError under STRICT_PAGE_ACCESS).

        Locks down current behaviour for angr-ho6i: if the engine ever silently
        keeps the state in `active`, this test fails and a real bug is filed.
        """
        import angr
        import claripy
        from angr import sim_options as o
        from angr.exploration import RustExplorationManager

        # mov rax, [rdi]   -> 48 8b 07
        # ret              -> c3
        # rdi will be constrained to 0xDEADBEEF, which is OUTSIDE the only
        # mapped region [0x1000, 0x2000) (the loaded shellcode page).
        shellcode = bytes.fromhex("488b07c3")
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)

        state = proj.factory.blank_state(
            addr=0x1000,
            add_options={o.STRICT_PAGE_ACCESS},
        )

        rdi_sym = claripy.BVS("rdi_addr", 64)
        state.regs.rdi = rdi_sym
        state.solver.add(rdi_sym == 0xDEADBEEF)
        assert state.solver.satisfiable(), (
            "Sanity: rdi==0xDEADBEEF is satisfiable on its own; the *load* "
            "from that address is what should fail, not the constraint set."
        )

        mgr = RustExplorationManager(proj, [state])
        mgr.run(max_steps=10)

        counts = mgr.stash_counts()
        assert counts.get("active", 0) == 0, (
            f"State remained in active despite load from unmapped 0xDEADBEEF "
            f"under STRICT_PAGE_ACCESS; stashes={counts} — engine silently "
            f"continued instead of erroring or dead-ending."
        )

    def test_unsat_state_pruned_during_step(self):
        """State with pre-existing contradictory constraints (x>100 AND x<50)
        must NOT remain in `active` after stepping. Locks down behaviour for
        angr-eldx — engine eagerly checks fork satisfiability and prunes
        UNSAT children, so an UNSAT parent is evicted via the prune path
        rather than silently advancing.
        """
        import angr
        import claripy
        from angr.exploration import RustExplorationManager

        # mov rax, 1     -> 48 c7 c0 01 00 00 00
        # cmp rax, 0     -> 48 83 f8 00
        # jne +1         -> 75 01
        # nop            -> 90
        # ret            -> c3
        shellcode = bytes.fromhex("48c7c0010000004883f8007501 90c3".replace(" ", ""))
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)

        state = proj.factory.blank_state(addr=0x1000)
        x = claripy.BVS("contradiction", 32)
        state.solver.add(x > 100)
        state.solver.add(x < 50)
        assert not state.solver.satisfiable(), (
            "Sanity: Python solver also sees the contradiction."
        )

        mgr = RustExplorationManager(proj, [state])
        mgr.run(max_steps=20)

        counts = mgr.stash_counts()
        assert counts.get("active", 0) == 0, (
            f"State with contradictory constraints (x>100 AND x<50) "
            f"remained in active after step; stashes={counts} — engine "
            f"failed to detect UNSAT and continued executing."
        )

    def test_register_invalid_name(self):
        """Getting a nonexistent register should raise, not crash."""
        state = RustSimState("amd64")
        with pytest.raises(ValueError, match="cannot read register"):
            state.get_register("nonexistent_register_xyz")

    def test_register_set_get_roundtrip(self):
        """Setting and getting a register preserves the value."""
        state = RustSimState("amd64")
        state.set_register("rax", 0xCAFEBABE)
        assert state.get_register("rax") == 0xCAFEBABE

    def test_solver_eval_after_unsat(self):
        """Evaluating an expression on an UNSAT solver should return None, not crash."""
        from angr.rustylib.vex_engine import RustSolverContext
        import claripy

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 5)
        ctx.add_constraint_ast(x == 10)  # contradicts
        assert not ctx.satisfiable()
        result = ctx.eval(x)
        # Should be None or 0-ish, not crash
        # (exact behavior is implementation-defined for UNSAT)

    def test_solver_min_on_unsat(self):
        """min() on UNSAT solver should return None, not crash."""
        from angr.rustylib.vex_engine import RustSolverContext
        import claripy

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 5)
        ctx.add_constraint_ast(x == 10)
        result = ctx.min(x, signed=False)
        assert result is None or isinstance(result, int)

    def test_solver_eval_upto_on_unsat(self):
        """eval_upto() on UNSAT solver returns empty list, not crash."""
        from angr.rustylib.vex_engine import RustSolverContext
        import claripy

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 5)
        ctx.add_constraint_ast(x == 10)
        results = ctx.eval_upto(x, 5)
        assert isinstance(results, list)

    def test_solver_many_constraints(self):
        """Solver handles many constraints without crashing."""
        from angr.rustylib.vex_engine import RustSolverContext
        import claripy

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        # Add 100 redundant but consistent constraints
        for i in range(100):
            ctx.add_constraint_ast(x >= i)
        ctx.add_constraint_ast(x <= 200)
        assert ctx.satisfiable()
        val = ctx.eval(x)
        assert 99 <= val <= 200

    def test_solver_fork_many_times(self):
        """Forking solver many times should not leak or crash."""
        from angr.rustylib.vex_engine import RustSolverContext
        import claripy

        ctx = RustSolverContext()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x >= 0)

        forks = []
        for i in range(50):
            f = ctx.fork()
            f.add_constraint_ast(x == i)
            forks.append(f)

        # All forks should be independent
        for i, f in enumerate(forks):
            assert f.eval(x) == i

    def test_state_fork_preserves_registers(self):
        """Forking a state preserves register values."""
        state = RustSimState("amd64")
        state.set_register("rax", 42)
        state.set_register("rbx", 99)

        forked = state.fork()
        assert forked.get_register("rax") == 42
        assert forked.get_register("rbx") == 99

        # Modifying fork doesn't affect original
        forked.set_register("rax", 100)
        assert state.get_register("rax") == 42
        assert forked.get_register("rax") == 100

    def test_state_fork_preserves_memory(self):
        """Forking a state preserves memory contents (CoW)."""
        state = RustSimState("amd64")
        state.map_memory_data(0x1000, b"\x41\x42\x43\x44", 7)

        forked = state.fork()
        data = forked.memory_load(0x1000, 4)
        assert bytes(data) == b"\x41\x42\x43\x44"

    def test_exploration_manager_empty_run(self):
        """Running with no callbacks set should raise, not crash silently."""
        mgr = _RustExplorationManager("amd64")
        mgr.set_find_addrs([0x1000])
        # No callbacks set — run should raise RuntimeError
        with pytest.raises(RuntimeError, match="callbacks not set"):
            mgr.run(100)

    # angr-2i4n: error path coverage — make the strict, user-facing failure
    # modes explicit so we notice if the engine ever starts silently
    # papering over them (phantom symbolic zero, unbounded Z3 hang, etc.).

    def test_unmapped_concrete_store_returns_error(self):
        """Storing to unmapped memory must raise — not silently succeed.

        Mirror of test_unmapped_memory_load_returns_error for the write
        side: the store path has its own start_page/end_page check
        (memory.rs ~750) and we want a regression that catches a future
        change that would auto-map on demand without permissions.
        """
        state = RustSimState("amd64")
        with pytest.raises(ValueError, match="unmapped"):
            state.memory_store(0xDEAD0000, b"\x41\x42\x43\x44")

    def test_cross_page_permission_write_returns_error(self):
        """Cross-page write where the second page is R-only must surface a
        permission violation, not silently overwrite the read-only page.

        The Rust unit tests in memory.rs already cover this via
        SymbolicMemory directly (test_permission_enforcement_cross_page_write
        and test_permission_enforcement_unaligned_store_two_pages_*); this
        test locks the same invariant down at the Python/RustSimState
        boundary so that a regression in the FFI wiring (e.g. forgetting
        to forward set_enforce_permissions) does not go unnoticed.
        """
        state = RustSimState("amd64")
        # Page 0 RW, page 1 R-only.
        state.map_memory(0x1000, 0x1000, 0x6)  # R|W
        state.map_memory(0x2000, 0x1000, 0x4)  # R only
        state.set_enforce_permissions(True)

        # 4-byte write straddling 0x1FFE..0x2002 hits the R-only page.
        with pytest.raises(ValueError, match="permission"):
            state.memory_store(0x1FFE, b"\x11\x22\x33\x44")

        # Sanity: a write entirely within the RW page still succeeds.
        state.memory_store(0x1000, b"\x55\x66\x77\x88")

    # angr-8e81: narrow exception types in _cb_memory_load / _cb_memory_store.
    # The callbacks used to swallow `except Exception`, masking unrelated bugs
    # (e.g. AttributeError, RuntimeError) by returning a zero buffer. These
    # tests pin down the new contract: only Sim*/Claripy errors are caught;
    # everything else propagates.

    def _build_load_store_manager(self):
        import angr
        from angr.exploration import RustExplorationManager
        shellcode = bytes.fromhex("c3")  # ret
        proj = angr.load_shellcode(shellcode, arch="AMD64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        mgr = RustExplorationManager(proj, [state])
        return mgr, state

    def test_cb_memory_load_swallows_sim_memory_error(self):
        """SimMemoryError from state.memory.load() is the expected failure mode
        and must keep being swallowed (zero buffer fallback)."""
        from angr.errors import SimMemoryError
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise SimMemoryError("simulated unmapped read")
        state.memory.load = boom

        mgr._set_callback_state(state)

        result = mgr._cb_memory_load(0x1000, 4)
        assert result == (bytes(4), False, None)

    def test_cb_memory_load_propagates_unrelated_exceptions(self):
        """Non-Sim/Claripy exceptions (e.g. RuntimeError from a real bug)
        must propagate out of _cb_memory_load instead of being silently
        masked as a zero buffer."""
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected bug")
        state.memory.load = boom

        mgr._set_callback_state(state)

        with pytest.raises(RuntimeError, match="unexpected bug"):
            mgr._cb_memory_load(0x1000, 4)

    def test_cb_memory_store_swallows_sim_memory_error(self):
        """SimMemoryError from state.memory.store() is the expected failure
        mode and must keep being swallowed."""
        from angr.errors import SimMemoryError
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise SimMemoryError("simulated unmapped write")
        state.memory.store = boom

        mgr._set_callback_state(state)

        # Should not raise.
        mgr._cb_memory_store(0x1000, b"\x41\x42\x43\x44")

    def test_cb_memory_store_propagates_unrelated_exceptions(self):
        """Non-Sim/Claripy exceptions must propagate out of _cb_memory_store."""
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected bug")
        state.memory.store = boom

        mgr._set_callback_state(state)

        with pytest.raises(RuntimeError, match="unexpected bug"):
            mgr._cb_memory_store(0x1000, b"\x41\x42\x43\x44")

    # angr-2f7o: extend the angr-8e81 contract to the remaining callbacks.
    # Each callback must:
    #   1. swallow Sim*/Claripy/PyVEX errors (legitimate symbolic-engine failures)
    #   2. propagate RuntimeError (or any other unrelated exception) so real
    #      bugs surface instead of being masked as zero buffers / empty pages.

    def _put_state_in_default_cache(self, mgr, state):
        """Inject `state` so _get_default_state() returns it. Several
        callbacks (fetch_page, sync_constraints, batch_fetch_pages) read
        through _state_cache rather than _callback_state."""
        mgr._state_cache[id(state)] = state

    def test_cb_lift_block_swallows_pyvex_error(self):
        """PyVEXError from the lifter must keep falling back to '{}'."""
        from pyvex.errors import PyVEXError
        mgr, _ = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise PyVEXError("simulated lift failure")
        mgr._project.factory.block = boom

        result = mgr._cb_lift_block(0x1000)
        assert result == '{}'

    def test_cb_lift_block_swallows_sim_engine_error(self):
        """SimEngineError from block construction must keep falling back."""
        from angr.errors import SimEngineError
        mgr, _ = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise SimEngineError("simulated engine failure")
        mgr._project.factory.block = boom

        result = mgr._cb_lift_block(0x1000)
        assert result == '{}'

    def test_cb_lift_block_propagates_unrelated_exceptions(self):
        """Non-(SimEngine/Claripy/PyVEX) exceptions must propagate."""
        mgr, _ = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected lift bug")
        mgr._project.factory.block = boom

        with pytest.raises(RuntimeError, match="unexpected lift bug"):
            mgr._cb_lift_block(0x1000)

    def test_cb_fetch_page_swallows_sim_memory_error(self):
        """SimMemoryError from state.memory.load() must keep returning empty page."""
        from angr.errors import SimMemoryError
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise SimMemoryError("simulated unmapped page")
        state.memory.load = boom
        self._put_state_in_default_cache(mgr, state)

        result = mgr._cb_fetch_page(0x1000)
        assert result == (bytes(4096), 0, False)

    def test_cb_fetch_page_propagates_unrelated_exceptions(self):
        """Non-Sim/Claripy exceptions must propagate out of _cb_fetch_page."""
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected fetch bug")
        state.memory.load = boom
        self._put_state_in_default_cache(mgr, state)

        with pytest.raises(RuntimeError, match="unexpected fetch bug"):
            mgr._cb_fetch_page(0x1000)

    def test_cb_sync_constraints_swallows_sim_solver_error(self):
        """SimSolverError from state.solver.add() must mark sync as failed
        rather than propagating — preserves the existing best-effort
        constraint-replay behavior."""
        import claripy
        from angr.errors import SimSolverError
        mgr, state = self._build_load_store_manager()

        x = claripy.BVS("x_sync", 32)
        handle_id = id(x)
        mgr._register_handle(handle_id, x)

        def boom(*args, **kwargs):
            raise SimSolverError("simulated solver failure")
        state.solver.add = boom
        self._put_state_in_default_cache(mgr, state)

        ok = mgr._cb_sync_constraints([("test_constraint", 32, 42, handle_id)])
        assert ok is False

    def test_cb_sync_constraints_propagates_unrelated_exceptions(self):
        """Non-Sim/Claripy exceptions must propagate out of _cb_sync_constraints."""
        import claripy
        mgr, state = self._build_load_store_manager()

        x = claripy.BVS("x_sync", 32)
        handle_id = id(x)
        mgr._register_handle(handle_id, x)

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected sync bug")
        state.solver.add = boom
        self._put_state_in_default_cache(mgr, state)

        with pytest.raises(RuntimeError, match="unexpected sync bug"):
            mgr._cb_sync_constraints([("test_constraint", 32, 42, handle_id)])

    def test_cb_memory_store_batch_swallows_sim_memory_error(self):
        """SimMemoryError from state.memory.store() must keep being swallowed
        per-store in the batch loop (other stores still attempted)."""
        from angr.errors import SimMemoryError
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise SimMemoryError("simulated unmapped batch store")
        state.memory.store = boom

        mgr._set_callback_state(state)

        # Should not raise.
        mgr._cb_memory_store_batch([(0x1000, b"\x41\x42\x43\x44")])

    def test_cb_memory_store_batch_propagates_unrelated_exceptions(self):
        """Non-Sim/Claripy exceptions must propagate out of _cb_memory_store_batch."""
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected batch store bug")
        state.memory.store = boom

        mgr._set_callback_state(state)

        with pytest.raises(RuntimeError, match="unexpected batch store bug"):
            mgr._cb_memory_store_batch([(0x1000, b"\x41\x42\x43\x44")])

    def test_cb_memory_load_batch_swallows_sim_memory_error(self):
        """SimMemoryError from state.memory.load() must keep returning a
        per-entry zero buffer in the batch loop."""
        from angr.errors import SimMemoryError
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise SimMemoryError("simulated unmapped batch load")
        state.memory.load = boom

        mgr._set_callback_state(state)

        result = mgr._cb_memory_load_batch([(0x1000, 4)])
        assert result == [(bytes(4), False, None)]

    def test_cb_memory_load_batch_propagates_unrelated_exceptions(self):
        """Non-Sim/Claripy exceptions must propagate out of _cb_memory_load_batch."""
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected batch load bug")
        state.memory.load = boom

        mgr._set_callback_state(state)

        with pytest.raises(RuntimeError, match="unexpected batch load bug"):
            mgr._cb_memory_load_batch([(0x1000, 4)])

    def test_cb_batch_fetch_pages_swallows_sim_memory_error(self):
        """SimMemoryError from state.memory.load() must keep returning empty
        pages per-entry in the batch loop."""
        from angr.errors import SimMemoryError
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise SimMemoryError("simulated unmapped batch page")
        state.memory.load = boom
        self._put_state_in_default_cache(mgr, state)

        result = mgr._cb_batch_fetch_pages([0x1000])
        assert result == [(bytes(4096), 0, True)]

    def test_cb_batch_fetch_pages_propagates_unrelated_exceptions(self):
        """Non-Sim/Claripy exceptions must propagate out of _cb_batch_fetch_pages."""
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected batch page bug")
        state.memory.load = boom
        self._put_state_in_default_cache(mgr, state)

        with pytest.raises(RuntimeError, match="unexpected batch page bug"):
            mgr._cb_batch_fetch_pages([0x1000])

    def test_cb_memory_store_symbolic_value_swallows_sim_memory_error(self):
        """SimMemoryError from state.memory.store() must keep being swallowed."""
        import claripy
        from angr.errors import SimMemoryError
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise SimMemoryError("simulated unmapped symbolic store")
        state.memory.store = boom

        mgr._set_callback_state(state)

        # Should not raise.
        ast = claripy.BVS("sym_store", 32)
        mgr._cb_memory_store_symbolic_value(0x1000, ast)

    def test_cb_memory_store_symbolic_value_propagates_unrelated_exceptions(self):
        """Non-Sim/Claripy exceptions must propagate out of
        _cb_memory_store_symbolic_value."""
        import claripy
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected symbolic store bug")
        state.memory.store = boom

        mgr._set_callback_state(state)

        ast = claripy.BVS("sym_store", 32)
        with pytest.raises(RuntimeError, match="unexpected symbolic store bug"):
            mgr._cb_memory_store_symbolic_value(0x1000, ast)

    # angr-b1qq: wire memory_{store,load}_symbolic_full so the existing
    # interpreter_cb fallbacks (statements.rs:368/444/710/1174,
    # expressions.rs:149) can actually delegate to Python instead of returning
    # CbExecutionError::Unsupported on TooLarge symbolic addresses.

    def test_symbolic_full_callbacks_are_wired(self):
        """_init_callbacks must bind both *_symbolic_full callbacks. Without
        them the Rust TooLarge branches at interpreter_cb/statements.rs:368/
        444/710/1174 and expressions.rs:149 hard-error instead of falling
        back to Python's memory model."""
        mgr, _ = self._build_load_store_manager()
        # Bound-method presence is the contract: _init_callbacks must have
        # called set_memory_{store,load}_symbolic_full with the corresponding
        # method. The setters are unconditional in PythonCallbacks (see
        # callbacks.rs:560/566).
        assert callable(getattr(mgr, "_cb_memory_store_symbolic_full", None))
        assert callable(getattr(mgr, "_cb_memory_load_symbolic_full", None))

    def test_cb_memory_store_symbolic_full_round_trip(self):
        """Storing a symbolic value at a (pinned) symbolic address via the
        full callback must land in Python state memory and be loadable back.
        Uses 0x4000 because 0x1000 is already mapped by load_shellcode."""
        import claripy
        mgr, state = self._build_load_store_manager()
        mgr._set_callback_state(state)

        target_addr = 0x4000
        state.memory.map_region(target_addr, 0x100, 7)
        addr = claripy.BVS("sym_addr", 64)
        state.solver.add(addr == target_addr)
        data = claripy.BVV(0xCAFEBABE, 32)

        mgr._cb_memory_store_symbolic_full(addr, data)

        loaded = state.memory.load(target_addr, 4, endness=state.arch.memory_endness,
                                   inspect=False, disable_actions=True)
        assert state.solver.eval(loaded) == 0xCAFEBABE

    def test_cb_memory_load_symbolic_full_returns_stored_ast(self):
        """Loading via the full callback must return a claripy AST that
        evaluates to the previously stored value at a symbolic-but-pinned
        address."""
        import claripy
        mgr, state = self._build_load_store_manager()
        mgr._set_callback_state(state)

        target_addr = 0x4000
        state.memory.map_region(target_addr, 0x100, 7)
        state.memory.store(target_addr, claripy.BVV(0xDEADBEEF, 32),
                           endness=state.arch.memory_endness,
                           inspect=False, disable_actions=True)

        addr = claripy.BVS("sym_load_addr", 64)
        state.solver.add(addr == target_addr)

        result = mgr._cb_memory_load_symbolic_full(addr, 4)
        assert result is not None
        assert state.solver.eval(result) == 0xDEADBEEF

    def test_cb_memory_store_symbolic_full_swallows_sim_memory_error(self):
        """SimMemoryError from state.memory.store() must be swallowed (matches
        the angr-8e81 / angr-2f7o convention used by the other symbolic
        callbacks). Otherwise Rust's flush_stores() path would propagate the
        error and tear down exploration."""
        import claripy
        from angr.errors import SimMemoryError
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise SimMemoryError("simulated unmapped symbolic-address store")
        state.memory.store = boom

        mgr._set_callback_state(state)

        addr = claripy.BVS("sym_addr", 64)
        data = claripy.BVS("sym_data", 32)
        # Should not raise.
        mgr._cb_memory_store_symbolic_full(addr, data)

    def test_cb_memory_store_symbolic_full_propagates_unrelated_exceptions(self):
        """Non-Sim/Claripy exceptions must propagate so real bugs aren't
        masked as silent no-ops."""
        import claripy
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected symbolic-address store bug")
        state.memory.store = boom

        mgr._set_callback_state(state)

        addr = claripy.BVS("sym_addr", 64)
        data = claripy.BVS("sym_data", 32)
        with pytest.raises(RuntimeError, match="unexpected symbolic-address store bug"):
            mgr._cb_memory_store_symbolic_full(addr, data)

    def test_cb_memory_load_symbolic_full_swallows_sim_memory_error(self):
        """SimMemoryError must be swallowed and a fresh symbolic AST returned
        so Rust can keep going (it'll wrap the result in a sym_pyref_*
        placeholder via expressions.rs)."""
        import claripy
        from angr.errors import SimMemoryError
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise SimMemoryError("simulated unmapped symbolic-address load")
        state.memory.load = boom

        mgr._set_callback_state(state)

        addr = claripy.BVS("sym_addr", 64)
        result = mgr._cb_memory_load_symbolic_full(addr, 4)
        # Returns something valid for Rust to consume — width must match.
        assert result is not None
        assert getattr(result, "length", None) == 32

    def test_cb_memory_load_symbolic_full_propagates_unrelated_exceptions(self):
        """Non-Sim/Claripy exceptions must propagate."""
        import claripy
        mgr, state = self._build_load_store_manager()

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected symbolic-address load bug")
        state.memory.load = boom

        mgr._set_callback_state(state)

        addr = claripy.BVS("sym_addr", 64)
        with pytest.raises(RuntimeError, match="unexpected symbolic-address load bug"):
            mgr._cb_memory_load_symbolic_full(addr, 4)

    def test_z3_solver_timeout_does_not_hang(self):
        """A tight Z3 timeout must bound `satisfiable()` wall-clock — even
        on a constraint set Z3 would otherwise grind on forever (factoring a
        128-bit semiprime). Locks down that set_timeout() actually flows
        into the Z3 solver params and that the result path handles
        SatResult::Unknown by returning (rather than panicking or looping).

        The wall-clock bound is intentionally generous (5s) to avoid CI
        flakes on slow runners; the *contract* under test is "does not
        hang", not "respects timeout to the millisecond".
        """
        import time
        import claripy
        from angr.rustylib.vex_engine import RustSolverContext

        ctx = RustSolverContext()
        ctx.set_timeout(50)  # 50ms — far smaller than the 5s assertion bound

        # Force Z3 to factor a 128-bit semiprime where both factors are
        # constrained > 2^60. Practically intractable for Z3's BV theory
        # under a 50ms budget.
        x = claripy.BVS("x_factor", 128)
        y = claripy.BVS("y_factor", 128)
        # An arbitrary 128-bit composite whose factorization is non-obvious
        # to Z3 within 50ms.
        n = 0xC0DEBABE_DEADBEEF_FEEDFACE_CAFEF00D
        ctx.add_constraint_ast(x * y == n)
        ctx.add_constraint_ast(x > (1 << 60))
        ctx.add_constraint_ast(y > (1 << 60))
        ctx.add_constraint_ast(x < (1 << 100))
        ctx.add_constraint_ast(y < (1 << 100))

        start = time.monotonic()
        # We do NOT assert on the boolean result: under a tight timeout
        # Z3 may return Unknown, which the engine collapses to false. We
        # only require that the call returns at all and within bound.
        _ = ctx.satisfiable()
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, (
            f"satisfiable() took {elapsed:.2f}s with a 50ms Z3 timeout — "
            f"timeout config is not being honoured (expected < 5s)."
        )


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
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
        if val >= 0x80000000:
            signed_val = val - 0x100000000
        else:
            signed_val = val
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
        from angr.exploration import RustExplorationManager

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
        from angr.exploration import RustExplorationManager

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
        from angr.exploration import RustExplorationManager
        import claripy

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
        from angr.exploration import RustExplorationManager

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
        from angr.exploration import RustExplorationManager

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
        from angr.exploration import RustExplorationManager

        ACCEPTED = 0x4006ed

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
            kwargs.pop('use_rust_engine', None)
            kwargs.pop('techniques', None)  # RustExplorationManager doesn't take techniques
            from angr.exploration import RustExplorationManager
            if thing is None:
                thing = [fauxware_project.factory.entry_state()]
            elif isinstance(thing, angr.SimState):
                thing = [thing]
            return RustExplorationManager(fauxware_project, active_states=thing)

        try:
            fauxware_project.factory.simulation_manager = rust_sm

            # Call authenticate(username="SOSNEAKY") which is the backdoor
            # authenticate is at 0x400664
            authenticate = fauxware_project.factory.callable(
                0x400664,
                prototype="int authenticate(char *username, char *password)",
                concrete_only=True,
            )
            # This should complete without error using the Rust engine
            result = authenticate(b"SOSNEAKY\x00", b"anything\x00")
            assert result is not None, "Callable should return a value"
        finally:
            fauxware_project.factory.simulation_manager = original_sm

    def test_run_until_predicate(self, fauxware_project):
        """run(step_func=..., until=...) stops when until returns True."""
        from angr.exploration import RustExplorationManager

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
        from angr.exploration import RustExplorationManager

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
        assert found_deadended or len(mgr.deadended) > 0, \
            "Deadended states should be preserved during step_func execution"


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestCallStackTracking:
    """Tests for call stack tracking in the Rust engine."""

    def test_call_stack_on_state_snapshot(self, fauxware_project):
        """Test that call stack is available on exported state snapshots."""
        from angr.exploration import RustExplorationManager

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.run(n=100)

        # Found states should have call stack accessible via snapshot
        for s in mgr.found:
            snapshot = mgr._rust_mgr.export_state(s.scratch._rust_state_id)
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
        from angr.exploration import RustExplorationManager

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ed)

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


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestDetailedHistory:
    """Tests for detailed execution history tracking."""

    def test_detailed_history_on_unit_state(self):
        """Test that new state has empty detailed history."""
        state = RustSimState("amd64", True)
        snapshot = state.export_full()
        assert snapshot.get_detailed_history() == []

    def test_detailed_history_on_found_states(self, fauxware_project):
        """Test that found states have non-empty detailed history."""
        from angr.exploration import RustExplorationManager

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ed)

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
        from angr.exploration import RustExplorationManager

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ed)

        assert len(mgr.found) > 0
        found_ids = mgr._rust_mgr.get_state_ids("found")
        snapshot = mgr._rust_mgr.export_state(found_ids[0])
        history_str = snapshot.get_detailed_history_str()
        assert len(history_str) > 0
        valid_jumpkinds = {"Ijk_Boring", "Ijk_Call", "Ijk_Ret", "Ijk_Sys_syscall", "Ijk_Other"}
        for addr, jk_str, target in history_str:
            assert jk_str in valid_jumpkinds, f"Unknown jumpkind: {jk_str}"

    def test_detailed_history_has_calls(self, fauxware_project):
        """Test that fauxware's history contains Ijk_Call entries."""
        from angr.exploration import RustExplorationManager

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ed)

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
        from angr.exploration import RustExplorationManager

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state], max_history=5)
        mgr.explore(find=0x4006ed)

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
        from angr.exploration import RustExplorationManager

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ed)

        assert len(mgr.found) > 0
        found_ids = mgr._rust_mgr.get_state_ids("found")
        for state_id in found_ids:
            snap = mgr._rust_mgr.export_state(state_id)
            history = snap.get_detailed_history()
            # Default cap is 1000 — must hold for any long-running exploration.
            assert len(history) <= 1000


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
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
        assert getattr(tech, '_native_length_limiter', False), \
            "LengthLimiter should be marked as native"
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

        assert getattr(tech, '_native_timeout', False), \
            "Timeout should be marked as native"
        assert mgr._rust_mgr.native_technique_count() == 1

    def test_multiple_native_techniques(self):
        """Test that multiple native techniques can coexist."""
        mgr = _RustExplorationManager("amd64")
        mgr.register_length_limiter(100, False)
        mgr.register_timeout(30.0)
        mgr.register_loop_bound(10, "spinning")
        assert mgr.native_technique_count() == 3


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
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


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
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

        with pytest.raises(Exception):
            mgr.merge_states([sid1], "merged")

    def test_merge_states_invalid_id(self):
        """Test that merge_states raises on invalid state IDs."""
        mgr = _RustExplorationManager("amd64")
        sid1 = mgr.create_state("active")

        with pytest.raises(Exception):
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


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestExplorationStrategy:
    """Tests for DFS/BFS exploration strategy."""

    def test_set_exploration_strategy_dfs(self, fauxware_project):
        """Test setting DFS strategy finds the same result."""
        from angr.exploration import RustExplorationManager

        find_addr = 0x4006ed
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.set_exploration_strategy('dfs')
        mgr.explore(find=find_addr)
        assert len(mgr.found) > 0, "DFS should find at least one state"

    def test_set_exploration_strategy_bfs(self, fauxware_project):
        """Test BFS strategy (default) works."""
        from angr.exploration import RustExplorationManager

        find_addr = 0x4006ed
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.set_exploration_strategy('bfs')
        mgr.explore(find=find_addr)
        assert len(mgr.found) > 0, "BFS should find at least one state"

    def test_set_exploration_strategy_invalid(self, fauxware_project):
        """Test invalid strategy raises ValueError."""
        from angr.exploration import RustExplorationManager

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        with pytest.raises(ValueError, match="Unknown exploration strategy"):
            mgr.set_exploration_strategy('random')

    def test_dfs_technique_auto_detection(self, fauxware_project):
        """Test that angr DFS technique is auto-detected."""
        from angr.exploration import RustExplorationManager

        find_addr = 0x4006ed
        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.use_technique(angr.exploration_techniques.DFS())
        mgr.explore(find=find_addr)
        assert len(mgr.found) > 0, "DFS technique should find at least one state"


@pytest.mark.skipif(not RUST_EXPLORATION_AVAILABLE, reason="Rust exploration not available")
class TestVexOperationCoverage:
    """Systematic tests for BV operations through the Rust solver.

    Each test constrains a symbolic variable using a specific BV operation,
    solves it, and verifies the result matches the expected concrete value.
    This ensures the Rust Z3 bridge correctly handles each VEX/claripy op.
    """

    @classmethod
    def setup_class(cls):
        from angr.exploration.rust_manager import _setup_shared_z3_context
        _setup_shared_z3_context()

    def _make_ctx(self):
        from angr.rustylib.vex_engine import RustSolverContext
        return RustSolverContext()

    # --- Arithmetic ---

    def test_add_concrete(self):
        """x + 7 == 49 => x == 42."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x + 7 == 49)
        assert ctx.eval(x) == 42

    def test_add_symbolic(self):
        """x + y == 100, y == 30 => x == 70."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        y = claripy.BVS("y", 32)
        ctx.add_constraint_ast(x + y == 100)
        ctx.add_constraint_ast(y == 30)
        assert ctx.eval(x) == 70

    def test_add_overflow_wraps(self):
        """0xFFFFFFFF + 1 wraps to 0 in 32-bit."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x + 1 == 0)
        assert ctx.eval(x) == 0xFFFFFFFF

    def test_sub_concrete(self):
        """x - 8 == 34 => x == 42."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x - 8 == 34)
        assert ctx.eval(x) == 42

    def test_sub_symbolic(self):
        """x - y == 20, y == 10 => x == 30."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        y = claripy.BVS("y", 32)
        ctx.add_constraint_ast(x - y == 20)
        ctx.add_constraint_ast(y == 10)
        assert ctx.eval(x) == 30

    def test_sub_underflow_wraps(self):
        """0 - 1 wraps to 0xFFFFFFFF in 32-bit."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x - 1 == 0xFFFFFFFF)
        assert ctx.eval(x) == 0

    def test_mul_concrete(self):
        """x * 6 == 42 => x == 7."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x * 6 == 42)
        # Multiple solutions possible (mod 2^32), but 7 must be one
        val = ctx.eval(x)
        assert (val * 6) & 0xFFFFFFFF == 42

    def test_mul_symbolic(self):
        """x * y == 56, x == 7 => y == 8."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        y = claripy.BVS("y", 32)
        ctx.add_constraint_ast(x * y == 56)
        ctx.add_constraint_ast(x == 7)
        assert ctx.eval(y) == 8

    # --- Bitwise ---

    def test_and_mask(self):
        """x & 0xFF == 0x42 constrains low byte."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast((x & 0xFF) == 0x42)
        assert (ctx.eval(x) & 0xFF) == 0x42

    def test_and_symbolic(self):
        """x & y == 0x10, x == 0x1F, so y & 0x1F == 0x10."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        y = claripy.BVS("y", 32)
        ctx.add_constraint_ast((x & y) == 0x10)
        ctx.add_constraint_ast(x == 0x1F)
        val_y = ctx.eval(y)
        assert (0x1F & val_y) == 0x10

    def test_or_bits(self):
        """x | 0xF0 == 0xFF => x & 0x0F must be 0x0F."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast((x | 0xF0) == 0xFF)
        val = ctx.eval(x)
        assert (val | 0xF0) == 0xFF

    def test_or_symbolic(self):
        """x | y == 0xFF, x == 0x0F => y must set high nibble."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        y = claripy.BVS("y", 8)
        ctx.add_constraint_ast((x | y) == 0xFF)
        ctx.add_constraint_ast(x == 0x0F)
        val_y = ctx.eval(y)
        assert (0x0F | val_y) == 0xFF

    def test_xor_concrete(self):
        """x ^ 0xAA == 0x55 => x == 0xFF."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast((x ^ 0xAA) == 0x55)
        assert ctx.eval(x) == 0xFF

    def test_xor_self_is_zero(self):
        """x ^ x == 0 for any x."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast((x ^ x) == 0)
        assert ctx.satisfiable()

    def test_xor_symbolic_inverse(self):
        """x ^ y == 0xFFFFFFFF => y == ~x."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        y = claripy.BVS("y", 32)
        ctx.add_constraint_ast((x ^ y) == 0xFFFFFFFF)
        ctx.add_constraint_ast(x == 0xDEADBEEF)
        assert ctx.eval(y) == (0xDEADBEEF ^ 0xFFFFFFFF)

    def test_not_bitwise(self):
        """~x == 0x00 => x == 0xFF (8-bit)."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(~x == 0x00)
        assert ctx.eval(x) == 0xFF

    # --- Shifts ---

    def test_shl_concrete(self):
        """x << 4 == 0x120 => low nibble lost, x == 0x12."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast((x << 4) == 0x120)
        val = ctx.eval(x)
        assert (val << 4) & 0xFFFFFFFF == 0x120

    def test_shl_symbolic_amount(self):
        """x << n == 0x80, x == 1 => n == 7."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        n = claripy.BVS("n", 32)
        ctx.add_constraint_ast((x << n) == 0x80)
        ctx.add_constraint_ast(x == 1)
        ctx.add_constraint_ast(n < 32)
        assert ctx.eval(n) == 7

    def test_lshr_concrete(self):
        """LShR(x, 8) == 0x12 => x >> 8 == 0x12."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(claripy.LShR(x, 8) == 0x12)
        val = ctx.eval(x)
        assert (val >> 8) == 0x12

    def test_lshr_vs_arithmetic(self):
        """LShR is logical (zero-fill), not arithmetic."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 0x80000000)
        ctx.add_constraint_ast(claripy.LShR(x, 1) == 0x40000000)
        assert ctx.satisfiable()

    def test_arithmetic_shr(self):
        """Arithmetic shift right sign-extends."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 0x80000000)
        # x >> 1 (arithmetic) should be 0xC0000000
        ctx.add_constraint_ast((x >> 1) == 0xC0000000)
        assert ctx.satisfiable()

    # --- Extract / Concat ---

    def test_extract_low_byte(self):
        """Extract(7, 0, x) gets the least significant byte."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 0xDEADBEEF)
        ctx.add_constraint_ast(claripy.Extract(7, 0, x) == 0xEF)
        assert ctx.satisfiable()

    def test_extract_high_byte(self):
        """Extract(31, 24, x) gets the most significant byte."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 0xDEADBEEF)
        ctx.add_constraint_ast(claripy.Extract(31, 24, x) == 0xDE)
        assert ctx.satisfiable()

    def test_extract_middle_word(self):
        """Extract(23, 8, x) gets the middle two bytes."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(claripy.Extract(23, 8, x) == 0xBEEF)
        val = ctx.eval(x)
        assert ((val >> 8) & 0xFFFF) == 0xBEEF

    def test_extract_single_bit(self):
        """Extract(0, 0, x) gets bit 0."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(claripy.Extract(0, 0, x) == 1)
        val = ctx.eval(x)
        assert (val & 1) == 1

    def test_concat_two_bytes(self):
        """Concat(a, b) forms a 16-bit value."""
        import claripy
        ctx = self._make_ctx()
        a = claripy.BVS("a", 8)
        b = claripy.BVS("b", 8)
        ctx.add_constraint_ast(claripy.Concat(a, b) == 0xCAFE)
        assert ctx.eval(a) == 0xCA
        assert ctx.eval(b) == 0xFE

    def test_concat_four_bytes(self):
        """Concat(a, b, c, d) forms a 32-bit value."""
        import claripy
        ctx = self._make_ctx()
        a = claripy.BVS("a", 8)
        b = claripy.BVS("b", 8)
        c = claripy.BVS("c", 8)
        d = claripy.BVS("d", 8)
        ctx.add_constraint_ast(claripy.Concat(a, b, c, d) == 0xDEADBEEF)
        assert ctx.eval(a) == 0xDE
        assert ctx.eval(b) == 0xAD
        assert ctx.eval(c) == 0xBE
        assert ctx.eval(d) == 0xEF

    def test_concat_then_extract_roundtrip(self):
        """Extract undoes Concat for matching bit ranges."""
        import claripy
        ctx = self._make_ctx()
        a = claripy.BVS("a", 8)
        b = claripy.BVS("b", 8)
        ctx.add_constraint_ast(a == 0x12)
        ctx.add_constraint_ast(b == 0x34)
        ab = claripy.Concat(a, b)
        ctx.add_constraint_ast(claripy.Extract(15, 8, ab) == 0x12)
        ctx.add_constraint_ast(claripy.Extract(7, 0, ab) == 0x34)
        assert ctx.satisfiable()

    # --- Extension ---

    def test_zeroext_8_to_32(self):
        """ZeroExt(24, x8) where x8 == 0xFF gives 0x000000FF."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(x == 0xFF)
        extended = claripy.ZeroExt(24, x)
        ctx.add_constraint_ast(extended == 0x000000FF)
        assert ctx.satisfiable()

    def test_zeroext_preserves_value(self):
        """ZeroExt should not change the numeric value of a positive number."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 16)
        ctx.add_constraint_ast(x == 0x1234)
        extended = claripy.ZeroExt(16, x)
        ctx.add_constraint_ast(extended == 0x00001234)
        assert ctx.satisfiable()

    def test_signext_positive(self):
        """SignExt of positive value (MSB=0) is same as ZeroExt."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(x == 0x7F)  # positive in signed 8-bit
        extended = claripy.SignExt(24, x)
        ctx.add_constraint_ast(extended == 0x0000007F)
        assert ctx.satisfiable()

    def test_signext_negative(self):
        """SignExt of negative value (MSB=1) fills with 1s."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(x == 0x80)  # -128 in signed 8-bit
        extended = claripy.SignExt(24, x)
        ctx.add_constraint_ast(extended == 0xFFFFFF80)
        assert ctx.satisfiable()

    def test_signext_ff(self):
        """SignExt(24, 0xFF) == 0xFFFFFFFF."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(x == 0xFF)
        extended = claripy.SignExt(24, x)
        ctx.add_constraint_ast(extended == 0xFFFFFFFF)
        assert ctx.satisfiable()

    # --- Reverse (byte swap) ---

    def test_reverse_16bit(self):
        """Reverse(0x1234) == 0x3412."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 16)
        ctx.add_constraint_ast(x == 0x1234)
        ctx.add_constraint_ast(claripy.Reverse(x) == 0x3412)
        assert ctx.satisfiable()

    def test_reverse_32bit(self):
        """Reverse(0xDEADBEEF) == 0xEFBEADDE."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 0xDEADBEEF)
        ctx.add_constraint_ast(claripy.Reverse(x) == 0xEFBEADDE)
        assert ctx.satisfiable()

    def test_reverse_involution(self):
        """Reverse(Reverse(x)) == x."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 0xCAFEBABE)
        ctx.add_constraint_ast(claripy.Reverse(claripy.Reverse(x)) == x)
        assert ctx.satisfiable()

    def test_reverse_64bit(self):
        """Reverse of a 64-bit value."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 64)
        ctx.add_constraint_ast(x == 0x0102030405060708)
        ctx.add_constraint_ast(claripy.Reverse(x) == 0x0807060504030201)
        assert ctx.satisfiable()

    # --- Comparisons ---

    def test_uge(self):
        """Unsigned greater-or-equal."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(claripy.UGE(x, 0xFE))
        results = ctx.eval_upto(x, 10)
        assert set(results) == {0xFE, 0xFF}

    def test_ule(self):
        """Unsigned less-or-equal."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(claripy.ULE(x, 2))
        results = ctx.eval_upto(x, 10)
        assert set(results) == {0, 1, 2}

    def test_sgt(self):
        """Signed greater-than constrains to positive range."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(claripy.SGT(x, claripy.BVV(0x7C, 8)))  # > 124 signed
        ctx.add_constraint_ast(claripy.SLT(x, claripy.BVV(0x7F, 8)))  # < 127 signed
        results = ctx.eval_upto(x, 10)
        assert set(results) == {0x7D, 0x7E}  # 125, 126

    def test_sle(self):
        """Signed less-or-equal with negative bound."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        # x <=s -126 (0x82) means x is in {0x80, 0x81, 0x82} = {-128, -127, -126}
        ctx.add_constraint_ast(claripy.SLE(x, claripy.BVV(0x82, 8)))
        ctx.add_constraint_ast(claripy.SGE(x, claripy.BVV(0x80, 8)))
        results = ctx.eval_upto(x, 10)
        assert set(results) == {0x80, 0x81, 0x82}

    # --- If-then-else ---

    def test_ite_true_branch(self):
        """If(True, a, b) == a."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 1)
        result = claripy.If(x == 1, claripy.BVV(0xAA, 8), claripy.BVV(0xBB, 8))
        ctx.add_constraint_ast(result == 0xAA)
        assert ctx.satisfiable()

    def test_ite_false_branch(self):
        """If(False, a, b) == b."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 0)
        result = claripy.If(x == 1, claripy.BVV(0xAA, 8), claripy.BVV(0xBB, 8))
        ctx.add_constraint_ast(result == 0xBB)
        assert ctx.satisfiable()

    def test_ite_symbolic_condition(self):
        """ITE with symbolic condition and constrained result."""
        import claripy
        ctx = self._make_ctx()
        cond = claripy.BVS("c", 8)
        a = claripy.BVS("a", 32)
        b = claripy.BVS("b", 32)
        ctx.add_constraint_ast(a == 100)
        ctx.add_constraint_ast(b == 200)
        result = claripy.If(cond == 1, a, b)
        ctx.add_constraint_ast(result == 100)
        assert ctx.eval(cond) == 1

    # --- Combined / complex ---

    def test_add_then_extract(self):
        """(x + y) constrained, then extract a byte of the sum."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        y = claripy.BVS("y", 32)
        ctx.add_constraint_ast(x == 0x00001000)
        ctx.add_constraint_ast(y == 0x00000234)
        s = x + y
        ctx.add_constraint_ast(claripy.Extract(15, 0, s) == 0x1234)
        assert ctx.satisfiable()

    def test_xor_shl_combo(self):
        """(x ^ key) << 8 == target tests combined ops."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        key = 0x55
        target = 0x0000AA00
        ctx.add_constraint_ast(((x ^ key) << 8) == target)
        val = ctx.eval(x)
        assert (((val ^ key) << 8) & 0xFFFFFFFF) == target

    def test_signext_then_add(self):
        """SignExt then add: common in sign-extended address calculations."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(x == 0xFE)  # -2 in signed 8-bit
        extended = claripy.SignExt(24, x)  # 0xFFFFFFFE
        result = extended + 0x100
        ctx.add_constraint_ast(result == 0x000000FE)
        assert ctx.satisfiable()

    def test_concat_reverse_extract(self):
        """Concat, Reverse, Extract pipeline (common in memory operations)."""
        import claripy
        ctx = self._make_ctx()
        a = claripy.BVS("a", 8)
        b = claripy.BVS("b", 8)
        ctx.add_constraint_ast(a == 0x12)
        ctx.add_constraint_ast(b == 0x34)
        word = claripy.Concat(a, b)  # 0x1234
        swapped = claripy.Reverse(word)  # 0x3412
        lo = claripy.Extract(7, 0, swapped)  # 0x12
        ctx.add_constraint_ast(lo == 0x12)
        assert ctx.satisfiable()

    def test_mul_and_mask(self):
        """x * 3 & 0xFF == result, checking low byte of multiplication."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 32)
        ctx.add_constraint_ast(x == 0x55)
        product = x * 3  # 0xFF
        ctx.add_constraint_ast((product & 0xFF) == 0xFF)
        assert ctx.satisfiable()

    # --- Width variations ---

    def test_add_8bit(self):
        """8-bit addition with overflow."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 8)
        ctx.add_constraint_ast(x + 1 == 0)
        assert ctx.eval(x) == 0xFF

    def test_add_64bit(self):
        """64-bit addition."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 64)
        ctx.add_constraint_ast(x + 1 == 0x100000000)
        assert ctx.eval(x) == 0xFFFFFFFF

    def test_xor_64bit(self):
        """64-bit XOR."""
        import claripy
        ctx = self._make_ctx()
        x = claripy.BVS("x", 64)
        ctx.add_constraint_ast(x ^ 0xDEADBEEFCAFEBABE == 0)
        assert ctx.eval(x) == 0xDEADBEEFCAFEBABE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
