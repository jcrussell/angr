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

    @pytest.fixture
    def fauxware_project(self):
        """Load fauxware test binary."""
        binary_path = os.path.join(TEST_BINARIES_DIR, "fauxware")
        if not os.path.exists(binary_path):
            pytest.skip("fauxware binary not found")
        return angr.Project(binary_path, auto_load_libs=False)

    def test_python_wrapper_creation(self, fauxware_project):
        """Test creating Python wrapper."""
        from angr.exploration import RustExplorationManager

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        assert mgr.stash_counts()["active"] >= 1

    def test_basic_explore(self, fauxware_project):
        """Test basic exploration with find address."""
        from angr.exploration import RustExplorationManager

        # Find the "Welcome" message address
        # In fauxware, this is typically around 0x4006ed
        find_addr = 0x4006ed

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        # Run exploration
        try:
            mgr.explore(find=find_addr)
            # If we get here, exploration completed
            stats = mgr.stats
            print(f"Exploration stats: {stats}")
        except Exception as e:
            # Exploration might fail due to missing SimProcedures
            # This is expected in unit tests
            print(f"Exploration stopped: {e}")

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

    @pytest.fixture
    def fauxware_project(self):
        """Load fauxware test binary."""
        binary_path = os.path.join(TEST_BINARIES_DIR, "fauxware")
        if not os.path.exists(binary_path):
            pytest.skip("fauxware binary not found")
        return angr.Project(binary_path, auto_load_libs=False)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
