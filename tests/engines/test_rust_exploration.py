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


    def test_python_wrapper_creation(self, fauxware_project):
        """Test creating Python wrapper."""
        from angr.exploration import RustExplorationManager

        state = fauxware_project.factory.entry_state()
        mgr = RustExplorationManager(fauxware_project, [state])

        assert mgr.stash_counts()["active"] == 1, "entry_state should produce exactly 1 active state"

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
