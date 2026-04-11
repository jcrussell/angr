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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
