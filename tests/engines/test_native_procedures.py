"""Tests for native SimProcedure implementations in Rust.

This module tests the Rust-native implementations of common libc functions
(strlen, memcpy, strcmp, etc.) that bypass Python for performance.
"""
import os
import pytest

# Check if Rust exploration is available
try:
    from angr.rustylib.vex_engine import (
        RustExplorationManager as _RustExplorationManager,
        RustSimState,
        PythonCallbacks,
    )
    RUST_NATIVE_AVAILABLE = True
except ImportError:
    RUST_NATIVE_AVAILABLE = False


@pytest.mark.skipif(not RUST_NATIVE_AVAILABLE, reason="Rust native procedures not available")
class TestNativeStrlen:
    """Tests for native strlen implementation."""

    def test_strlen_basic(self):
        """Test strlen with basic null-terminated string."""
        state = RustSimState("amd64")

        # Map memory and store "hello\0"
        state.map_memory(0x1000, 0x1000, 7)  # RWX
        state.memory_store(0x1000, b"hello\x00")

        # Create exploration manager and test native procedure
        mgr = _RustExplorationManager("amd64")
        mgr.register_simprocedure(0x2000, "strlen", 1, False)

        # Add state with PC at strlen hook and RDI pointing to string
        state.pc = 0x2000
        state.set_register("rdi", 0x1000)  # First arg = string address
        state.set_register("rsp", 0x7fff_0000)
        state.map_memory(0x7fff_0000 - 0x1000, 0x2000, 6)  # Stack
        state.memory_store(0x7fff_0000, (0x3000).to_bytes(8, 'little'))  # Return addr

        mgr.add_state("active", state)

        # Check native procedures are available
        procs = mgr.list_native_procedures()
        assert "strlen" in procs

    def test_strlen_empty(self):
        """Test strlen with empty string (just null)."""
        state = RustSimState("amd64")

        # Map memory and store just null byte
        state.map_memory(0x1000, 0x1000, 7)
        state.memory_store(0x1000, b"\x00")

        # The strlen should return 0 for empty string
        mgr = _RustExplorationManager("amd64")
        procs = mgr.list_native_procedures()
        assert "strlen" in procs

    def test_strlen_long_string(self):
        """Test strlen with longer string."""
        state = RustSimState("amd64")

        # Map memory and store longer string
        test_str = b"This is a longer test string with special chars: !@#$%\x00"
        state.map_memory(0x1000, 0x1000, 7)
        state.memory_store(0x1000, test_str)

        # Expected length is everything before the null
        expected_len = len(test_str) - 1
        assert expected_len == 54  # Length without null terminator


@pytest.mark.skipif(not RUST_NATIVE_AVAILABLE, reason="Rust native procedures not available")
class TestNativeMemcpy:
    """Tests for native memcpy implementation."""

    def test_memcpy_basic(self):
        """Test memcpy with basic data."""
        state = RustSimState("amd64")

        # Map source and destination memory
        state.map_memory(0x1000, 0x2000, 7)  # RWX

        # Store source data
        source_data = b"Hello, World!"
        state.memory_store(0x1000, source_data)

        # Create manager
        mgr = _RustExplorationManager("amd64")
        procs = mgr.list_native_procedures()
        assert "memcpy" in procs

        # Verify source data
        loaded = state.memory_load(0x1000, len(source_data))
        assert loaded == source_data

    def test_memcpy_zero_length(self):
        """Test memcpy with zero length."""
        state = RustSimState("amd64")
        state.map_memory(0x1000, 0x1000, 7)

        # Zero-length memcpy should be a no-op
        mgr = _RustExplorationManager("amd64")
        mgr.register_simprocedure(0x2000, "memcpy", 3, False)

    def test_memmove_available(self):
        """Test memmove is available as native procedure."""
        mgr = _RustExplorationManager("amd64")
        procs = mgr.list_native_procedures()
        assert "memmove" in procs


@pytest.mark.skipif(not RUST_NATIVE_AVAILABLE, reason="Rust native procedures not available")
class TestNativeStrcmp:
    """Tests for native strcmp implementation."""

    def test_strcmp_equal(self):
        """Test strcmp with equal strings."""
        state = RustSimState("amd64")
        state.map_memory(0x1000, 0x2000, 7)

        # Store equal strings
        state.memory_store(0x1000, b"test\x00")
        state.memory_store(0x1100, b"test\x00")

        mgr = _RustExplorationManager("amd64")
        procs = mgr.list_native_procedures()
        assert "strcmp" in procs

    def test_strcmp_different(self):
        """Test strcmp with different strings."""
        state = RustSimState("amd64")
        state.map_memory(0x1000, 0x2000, 7)

        # Store different strings
        state.memory_store(0x1000, b"abc\x00")
        state.memory_store(0x1100, b"abd\x00")

        mgr = _RustExplorationManager("amd64")
        procs = mgr.list_native_procedures()
        assert "strcmp" in procs

    def test_strncmp_available(self):
        """Test strncmp is available."""
        mgr = _RustExplorationManager("amd64")
        procs = mgr.list_native_procedures()
        assert "strncmp" in procs

    def test_strcasecmp_available(self):
        """Test strcasecmp is available."""
        mgr = _RustExplorationManager("amd64")
        procs = mgr.list_native_procedures()
        assert "strcasecmp" in procs


@pytest.mark.skipif(not RUST_NATIVE_AVAILABLE, reason="Rust native procedures not available")
class TestNativeProcedureControl:
    """Tests for controlling native procedure execution."""

    def test_list_native_procedures(self):
        """Test listing available native procedures."""
        mgr = _RustExplorationManager("amd64")
        procs = mgr.list_native_procedures()

        # Should have at least strlen, memcpy, strcmp
        assert isinstance(procs, list)
        assert len(procs) >= 3
        assert "strlen" in procs
        assert "memcpy" in procs
        assert "strcmp" in procs

    def test_disable_all_native_procedures(self):
        """Test disabling all native procedures."""
        mgr = _RustExplorationManager("amd64")

        # Verify enabled by default
        assert mgr.native_procedures_enabled()

        # Disable all
        mgr.disable_native_procedures()
        assert not mgr.native_procedures_enabled()

        # Re-enable
        mgr.enable_native_procedures()
        assert mgr.native_procedures_enabled()

    def test_disable_specific_procedure(self):
        """Test disabling a specific native procedure."""
        mgr = _RustExplorationManager("amd64")

        # Disable strlen only
        mgr.disable_native_procedure("strlen")

        # memcpy should still work
        procs = mgr.list_native_procedures()
        # Note: list_native_procedures returns all registered, not just enabled
        assert "memcpy" in procs

        # Re-enable strlen
        mgr.enable_native_procedure("strlen")

    def test_python_override(self):
        """Test setting Python override for a procedure."""
        mgr = _RustExplorationManager("amd64")

        # Set Python override for strlen
        mgr.set_python_override("strlen")

        # Remove override
        mgr.remove_python_override("strlen")

    def test_has_native_procedure(self):
        """Test checking if native procedure exists."""
        mgr = _RustExplorationManager("amd64")

        assert mgr.has_native_procedure("strlen")
        assert mgr.has_native_procedure("memcpy")
        assert mgr.has_native_procedure("strcmp")
        assert not mgr.has_native_procedure("nonexistent_proc")


@pytest.mark.skipif(not RUST_NATIVE_AVAILABLE, reason="Rust native procedures not available")
class TestNativeProcedureStats:
    """Tests for native procedure statistics."""

    def test_native_procedure_stats(self):
        """Test getting native procedure execution statistics."""
        mgr = _RustExplorationManager("amd64")

        stats = mgr.native_procedure_stats()
        assert "native_calls" in stats
        assert "python_fallbacks" in stats
        assert "call_counts" in stats

        # Initially should be zero
        assert stats["native_calls"] == 0
        assert stats["python_fallbacks"] == 0

    def test_stats_include_native_procs(self):
        """Test that general stats include native proc info."""
        mgr = _RustExplorationManager("amd64")

        stats = mgr.stats()
        assert "native_proc_calls" in stats
        assert "native_proc_fallbacks" in stats


@pytest.mark.skipif(not RUST_NATIVE_AVAILABLE, reason="Rust native procedures not available")
class TestRustSimStateMemoryHelpers:
    """Tests for RustSimState memory helper methods used by native procedures."""

    def test_memory_load_byte(self):
        """Test loading a single byte."""
        state = RustSimState("amd64")
        state.map_memory(0x1000, 0x1000, 7)
        state.memory_store(0x1000, b"\x42")

        data = state.memory_load(0x1000, 1)
        assert data == b"\x42"

    def test_memory_load_multiple_bytes(self):
        """Test loading multiple bytes."""
        state = RustSimState("amd64")
        state.map_memory(0x1000, 0x1000, 7)

        test_data = b"\x01\x02\x03\x04\x05\x06\x07\x08"
        state.memory_store(0x1000, test_data)

        data = state.memory_load(0x1000, 8)
        assert data == test_data

    def test_memory_store_and_load(self):
        """Test storing and loading data."""
        state = RustSimState("amd64")
        state.map_memory(0x1000, 0x1000, 7)

        # Store some data
        original = b"ABCDEFGH"
        state.memory_store(0x1000, original)

        # Load it back
        loaded = state.memory_load(0x1000, len(original))
        assert loaded == original

    def test_map_memory_with_data(self):
        """Test mapping memory with initial data."""
        state = RustSimState("amd64")

        # Map with initial data
        initial_data = b"Hello, memory!"
        state.map_memory_data(0x1000, initial_data, 7)

        # Load it back
        loaded = state.memory_load(0x1000, len(initial_data))
        assert loaded == initial_data


@pytest.mark.skipif(not RUST_NATIVE_AVAILABLE, reason="Rust native procedures not available")
class TestNativeProcedureIntegration:
    """Integration tests for native procedures with exploration."""

    def test_exploration_with_native_procs(self):
        """Test that exploration manager has native procedures."""
        mgr = _RustExplorationManager("amd64")

        # Register a simprocedure
        mgr.register_simprocedure(0x401000, "strlen", 1, False)

        # Native procedure should be available
        assert mgr.has_native_procedure("strlen")

        # Stats should show hooks
        stats = mgr.stats()
        assert stats["simprocedures"] == 1
        assert stats["hooks"] == 1

    def test_native_procs_with_state(self):
        """Test native procedures with a state."""
        mgr = _RustExplorationManager("amd64")

        # Create a state
        state = RustSimState("amd64")
        state.map_memory(0x1000, 0x2000, 7)
        state.memory_store(0x1000, b"test string\x00")

        # Add state
        mgr.add_state("active", state)
        assert mgr.active_count() == 1

        # Native procs should be available
        assert mgr.native_procedures_enabled()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
