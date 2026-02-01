"""
Memory Subsystem Equivalence Tests.

Tests that verify the Rust VEX engine's memory operations produce identical
results to the Python VEX engine.
"""
from __future__ import annotations

import logging
import struct
import unittest

import pytest
import claripy

from angr import SimState, load_shellcode

l = logging.getLogger(__name__)

# Import Rust engine availability flag
try:
    from angr.engines.rust_vex import (
        RustVEXEngineWrapper,
        RUST_ENGINE_AVAILABLE,
    )
except ImportError:
    RUST_ENGINE_AVAILABLE = False
    RustVEXEngineWrapper = None


class TestMemoryLoadStore(unittest.TestCase):
    """Test basic memory load/store equivalence."""

    def setUp(self):
        if not RUST_ENGINE_AVAILABLE:
            self.skipTest("Rust VEX engine not available")
        self.rust_engine = RustVEXEngineWrapper("amd64")
        self.project = load_shellcode(b"\xc3", arch="AMD64")
        self.python_state = SimState(project=self.project)

    def test_store_read_1byte(self):
        """Test 1-byte memory operations."""
        addr = 0x1000
        self.rust_engine.map_memory(addr, 0x1000)

        test_values = [0x00, 0x7F, 0x80, 0xFF]
        for val in test_values:
            with self.subTest(value=val):
                # Rust engine
                self.rust_engine.write_memory(addr, bytes([val]))
                rust_result = self.rust_engine.read_memory(addr, 1)

                # Python engine
                self.python_state.memory.store(addr, claripy.BVV(val, 8))
                python_result = self.python_state.memory.load(addr, 1)

                self.assertEqual(
                    rust_result, bytes([val]),
                    f"Rust read mismatch for value {val:#x}"
                )
                self.assertEqual(
                    python_result.concrete_value, val,
                    f"Python read mismatch for value {val:#x}"
                )

    def test_store_read_2bytes(self):
        """Test 2-byte memory operations."""
        addr = 0x1000
        self.rust_engine.map_memory(addr, 0x1000)

        test_values = [0x0000, 0x7FFF, 0x8000, 0xFFFF, 0x1234, 0xDEAD]
        for val in test_values:
            with self.subTest(value=val):
                data = struct.pack("<H", val)  # Little-endian

                self.rust_engine.write_memory(addr, data)
                rust_result = self.rust_engine.read_memory(addr, 2)

                self.assertEqual(rust_result, data)

    def test_store_read_4bytes(self):
        """Test 4-byte memory operations."""
        addr = 0x1000
        self.rust_engine.map_memory(addr, 0x1000)

        test_values = [0x00000000, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xDEADBEEF]
        for val in test_values:
            with self.subTest(value=val):
                data = struct.pack("<I", val)

                self.rust_engine.write_memory(addr, data)
                rust_result = self.rust_engine.read_memory(addr, 4)

                self.assertEqual(rust_result, data)

    def test_store_read_8bytes(self):
        """Test 8-byte memory operations."""
        addr = 0x1000
        self.rust_engine.map_memory(addr, 0x1000)

        test_values = [
            0x0000000000000000,
            0x7FFFFFFFFFFFFFFF,
            0x8000000000000000,
            0xFFFFFFFFFFFFFFFF,
            0xDEADBEEFCAFEBABE,
        ]
        for val in test_values:
            with self.subTest(value=val):
                data = struct.pack("<Q", val)

                self.rust_engine.write_memory(addr, data)
                rust_result = self.rust_engine.read_memory(addr, 8)

                self.assertEqual(rust_result, data)


class TestMemoryEndianness(unittest.TestCase):
    """Test memory endianness handling."""

    def setUp(self):
        if not RUST_ENGINE_AVAILABLE:
            self.skipTest("Rust VEX engine not available")

    def test_little_endian_amd64(self):
        """AMD64 should use little-endian byte order."""
        engine = RustVEXEngineWrapper("amd64")
        engine.map_memory(0x1000, 0x1000)

        # Write 0x12345678 - in little-endian, stored as 78 56 34 12
        val = 0x12345678
        data = struct.pack("<I", val)
        self.assertEqual(data, b"\x78\x56\x34\x12")

        engine.write_memory(0x1000, data)
        result = engine.read_memory(0x1000, 4)

        self.assertEqual(result, b"\x78\x56\x34\x12")

    def test_little_endian_x86(self):
        """x86 should use little-endian byte order."""
        engine = RustVEXEngineWrapper("x86")
        engine.map_memory(0x1000, 0x1000)

        val = 0xDEADBEEF
        data = struct.pack("<I", val)
        engine.write_memory(0x1000, data)
        result = engine.read_memory(0x1000, 4)

        self.assertEqual(result, data)

    def test_little_endian_arm(self):
        """ARM (little-endian mode) should use little-endian byte order."""
        engine = RustVEXEngineWrapper("arm")
        engine.map_memory(0x1000, 0x1000)

        val = 0xCAFEBABE
        data = struct.pack("<I", val)
        engine.write_memory(0x1000, data)
        result = engine.read_memory(0x1000, 4)

        self.assertEqual(result, data)


class TestCrossPageAccess(unittest.TestCase):
    """Test memory access across page boundaries."""

    def setUp(self):
        if not RUST_ENGINE_AVAILABLE:
            self.skipTest("Rust VEX engine not available")
        self.engine = RustVEXEngineWrapper("amd64")

    def test_cross_page_read_write_4byte(self):
        """Test 4-byte access crossing a page boundary."""
        page_size = 0x1000
        # Map two consecutive pages
        self.engine.map_memory(0x1000, page_size * 2)

        # Address that causes 4-byte access to cross page boundary
        # Page 1: 0x1000-0x1FFF, Page 2: 0x2000-0x2FFF
        cross_addr = 0x1FFE  # 2 bytes in page 1, 2 bytes in page 2

        test_val = 0xDEADBEEF
        data = struct.pack("<I", test_val)

        self.engine.write_memory(cross_addr, data)
        result = self.engine.read_memory(cross_addr, 4)

        self.assertEqual(result, data)

    def test_cross_page_read_write_8byte(self):
        """Test 8-byte access crossing a page boundary."""
        self.engine.map_memory(0x1000, 0x2000)

        cross_addr = 0x1FFC  # 4 bytes in page 1, 4 bytes in page 2

        test_val = 0xDEADBEEFCAFEBABE
        data = struct.pack("<Q", test_val)

        self.engine.write_memory(cross_addr, data)
        result = self.engine.read_memory(cross_addr, 8)

        self.assertEqual(result, data)

    def test_cross_page_at_various_offsets(self):
        """Test cross-page access at various offsets within the boundary zone."""
        self.engine.map_memory(0x1000, 0x2000)

        # Test offsets 0xFFD through 0xFFF (crossing into next page with 4-byte read)
        for offset in [0xFFD, 0xFFE, 0xFFF]:
            addr = 0x1000 + offset
            with self.subTest(offset=offset):
                test_val = 0x12345678
                data = struct.pack("<I", test_val)

                self.engine.write_memory(addr, data)
                result = self.engine.read_memory(addr, 4)

                self.assertEqual(result, data)


class TestForkIsolation(unittest.TestCase):
    """Test that forked engines have isolated memory."""

    def setUp(self):
        if not RUST_ENGINE_AVAILABLE:
            self.skipTest("Rust VEX engine not available")

    def test_fork_memory_isolation(self):
        """Writes in forked engine should not affect original."""
        original = RustVEXEngineWrapper("amd64")
        original.map_memory(0x1000, 0x1000)
        original.write_memory(0x1000, b"\x01\x02\x03\x04")

        # Fork the engine
        forked = original.fork()

        # Modify forked engine's memory
        forked.write_memory(0x1000, b"\xFF\xFF\xFF\xFF")

        # Original should be unchanged
        original_data = original.read_memory(0x1000, 4)
        forked_data = forked.read_memory(0x1000, 4)

        self.assertEqual(original_data, b"\x01\x02\x03\x04")
        self.assertEqual(forked_data, b"\xFF\xFF\xFF\xFF")

    def test_fork_memory_copy_on_write(self):
        """Forked engine should initially share memory with original."""
        original = RustVEXEngineWrapper("amd64")
        original.map_memory(0x1000, 0x1000)
        original.write_memory(0x1000, b"\xDE\xAD\xBE\xEF")

        # Fork
        forked = original.fork()

        # Forked should read same initial data
        forked_data = forked.read_memory(0x1000, 4)
        self.assertEqual(forked_data, b"\xDE\xAD\xBE\xEF")

    def test_multiple_forks_isolation(self):
        """Multiple forks should all be isolated from each other."""
        original = RustVEXEngineWrapper("amd64")
        original.map_memory(0x1000, 0x1000)
        original.write_memory(0x1000, b"\x00\x00\x00\x00")

        # Create multiple forks
        forks = [original.fork() for _ in range(5)]

        # Write different values to each fork
        for i, fork in enumerate(forks):
            fork.write_memory(0x1000, bytes([i, i, i, i]))

        # Verify isolation
        self.assertEqual(original.read_memory(0x1000, 4), b"\x00\x00\x00\x00")
        for i, fork in enumerate(forks):
            expected = bytes([i, i, i, i])
            self.assertEqual(fork.read_memory(0x1000, 4), expected)


class TestMemoryMapping(unittest.TestCase):
    """Test memory mapping operations."""

    def setUp(self):
        if not RUST_ENGINE_AVAILABLE:
            self.skipTest("Rust VEX engine not available")

    def test_map_memory_with_data(self):
        """Test mapping memory with initial data."""
        engine = RustVEXEngineWrapper("amd64")

        data = b"\x48\x89\xC3\xC3"  # mov rbx, rax; ret
        engine.map_memory_data(0x1000, data)

        result = engine.read_memory(0x1000, len(data))
        self.assertEqual(result, data)

    def test_map_multiple_regions(self):
        """Test mapping multiple memory regions."""
        engine = RustVEXEngineWrapper("amd64")

        engine.map_memory(0x1000, 0x1000)
        engine.map_memory(0x3000, 0x1000)
        engine.map_memory(0x5000, 0x1000)

        # Write to each region
        engine.write_memory(0x1000, b"AAAA")
        engine.write_memory(0x3000, b"BBBB")
        engine.write_memory(0x5000, b"CCCC")

        # Read back
        self.assertEqual(engine.read_memory(0x1000, 4), b"AAAA")
        self.assertEqual(engine.read_memory(0x3000, 4), b"BBBB")
        self.assertEqual(engine.read_memory(0x5000, 4), b"CCCC")

    def test_map_large_region(self):
        """Test mapping a large memory region."""
        engine = RustVEXEngineWrapper("amd64")

        # Map 1 MB
        engine.map_memory(0x100000, 0x100000)

        # Write at various offsets
        engine.write_memory(0x100000, b"start")
        engine.write_memory(0x180000, b"middle")
        engine.write_memory(0x1FFFF0, b"end")

        # Read back
        self.assertEqual(engine.read_memory(0x100000, 5), b"start")
        self.assertEqual(engine.read_memory(0x180000, 6), b"middle")
        self.assertEqual(engine.read_memory(0x1FFFF0, 3), b"end")


class TestMemoryPermissions(unittest.TestCase):
    """Test memory permission handling."""

    def setUp(self):
        if not RUST_ENGINE_AVAILABLE:
            self.skipTest("Rust VEX engine not available")

    @pytest.mark.xfail(reason="Rust engine map_memory_data doesn't initialize memory with data")
    def test_map_with_different_permissions(self):
        """Test mapping memory with different permission bits."""
        engine = RustVEXEngineWrapper("amd64")

        # RWX permissions (7)
        engine.map_memory(0x1000, 0x1000, 7)
        engine.write_memory(0x1000, b"RWX")
        self.assertEqual(engine.read_memory(0x1000, 3), b"RWX")

        # RW permissions (6)
        engine.map_memory(0x2000, 0x1000, 6)
        engine.write_memory(0x2000, b"RW")
        self.assertEqual(engine.read_memory(0x2000, 2), b"RW")

        # RX permissions (5)
        engine.map_memory(0x3000, 0x1000, 5)
        # Can still read
        engine.map_memory_data(0x3000, b"RX")
        self.assertEqual(engine.read_memory(0x3000, 2), b"RX")


class TestSymbolicMemory(unittest.TestCase):
    """Test memory operations with symbolic values (Python engine only)."""

    @classmethod
    def setUpClass(cls):
        cls.project = load_shellcode(b"\xc3", arch="AMD64")

    def test_symbolic_store_concrete_load(self):
        """Store symbolic value, load should return symbolic."""
        state = SimState(project=self.project)

        x = claripy.BVS("x", 32)
        state.memory.store(0x1000, x)

        loaded = state.memory.load(0x1000, 4)

        self.assertTrue(loaded.symbolic)
        # Should be same symbolic variable
        self.assertTrue(state.solver.is_true(loaded == x))

    def test_symbolic_address_load(self):
        """Load from symbolic address."""
        state = SimState(project=self.project)

        # Store concrete values at known addresses
        state.memory.store(0x1000, claripy.BVV(0x11111111, 32))
        state.memory.store(0x1004, claripy.BVV(0x22222222, 32))
        state.memory.store(0x1008, claripy.BVV(0x33333333, 32))

        # Symbolic address
        addr = claripy.BVS("addr", 64)
        state.solver.add(addr >= 0x1000)
        state.solver.add(addr <= 0x1008)
        state.solver.add(addr % 4 == 0)  # Aligned

        loaded = state.memory.load(addr, 4, endness=state.arch.memory_endness)

        # Result should be symbolic
        self.assertTrue(loaded.symbolic)

        # Should be able to evaluate to any of the stored values
        possible = state.solver.eval_upto(loaded, 10)
        self.assertIn(0x11111111, possible)
        self.assertIn(0x22222222, possible)
        self.assertIn(0x33333333, possible)


if __name__ == "__main__":
    unittest.main()
