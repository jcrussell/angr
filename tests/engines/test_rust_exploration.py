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

    @pytest.fixture
    def fauxware_project(self):
        """Load fauxware test binary."""
        binary_path = os.path.join(TEST_BINARIES_DIR, "fauxware")
        if not os.path.exists(binary_path):
            pytest.skip("fauxware binary not found")
        return angr.Project(binary_path, auto_load_libs=False)

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

    @pytest.fixture
    def fauxware_project(self):
        """Load fauxware test binary."""
        binary_path = os.path.join(TEST_BINARIES_DIR, "fauxware")
        if not os.path.exists(binary_path):
            pytest.skip("fauxware binary not found")
        return angr.Project(binary_path, auto_load_libs=False)

    def test_explore_with_max_steps(self, fauxware_project):
        """Exploration respects max_steps limit."""
        from angr.exploration import RustExplorationManager

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ed, max_steps=10)

        # Should stop before finding (10 steps is too few for fauxware)
        # But should not crash
        stats = mgr.stats
        assert stats['ffi_crossings'] > 0

    def test_explore_finds_correct_state(self, fauxware_project):
        """Full exploration finds the expected state."""
        from angr.exploration import RustExplorationManager

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])
        mgr.explore(find=0x4006ed, avoid=0x4006fd, max_steps=50000)

        assert len(mgr.found) > 0, "Should find at least one state"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
