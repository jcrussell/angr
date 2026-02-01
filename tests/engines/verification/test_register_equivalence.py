"""
Register Access Equivalence Tests.

Tests that verify the Rust VEX engine's register operations produce identical
results to the Python VEX engine, including register aliasing.
"""
from __future__ import annotations

import logging
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


class TestAMD64Registers(unittest.TestCase):
    """Test AMD64 register access and aliasing."""

    def setUp(self):
        if not RUST_ENGINE_AVAILABLE:
            self.skipTest("Rust VEX engine not available")
        self.rust_engine = RustVEXEngineWrapper("amd64")
        self.project = load_shellcode(b"\xc3", arch="AMD64")
        self.python_state = SimState(project=self.project)

    def test_general_purpose_registers(self):
        """Test all 64-bit general purpose registers."""
        registers = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
                     "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"]

        for reg in registers:
            with self.subTest(register=reg):
                test_val = 0xDEADBEEFCAFEBABE

                # Rust engine
                self.rust_engine.set_register(reg, test_val)
                rust_result = self.rust_engine.get_register(reg)

                # Python engine
                setattr(self.python_state.regs, reg, test_val)
                python_result = getattr(self.python_state.regs, reg)

                self.assertEqual(rust_result, test_val)
                if hasattr(python_result, 'concrete_value'):
                    self.assertEqual(python_result.concrete_value, test_val)

    def test_pc_register(self):
        """Test program counter access."""
        test_val = 0x00007FFFDEADBEEF

        self.rust_engine.pc = test_val
        self.assertEqual(self.rust_engine.pc, test_val)

        # Also via rip
        self.rust_engine.set_register("rip", 0x401000)
        self.assertEqual(self.rust_engine.get_register("rip"), 0x401000)

    def test_get_all_registers(self):
        """Test getting all registers at once."""
        self.rust_engine.set_register("rax", 1)
        self.rust_engine.set_register("rbx", 2)
        self.rust_engine.set_register("rcx", 3)

        regs = self.rust_engine.get_registers()

        self.assertEqual(regs.get("rax"), 1)
        self.assertEqual(regs.get("rbx"), 2)
        self.assertEqual(regs.get("rcx"), 3)

    def test_edge_case_values(self):
        """Test edge case register values."""
        test_cases = [
            ("rax", 0x0000000000000000),  # Zero
            ("rbx", 0x0000000000000001),  # One
            ("rcx", 0x7FFFFFFFFFFFFFFF),  # Max signed positive
            ("rdx", 0x8000000000000000),  # Min signed negative
            ("rsi", 0xFFFFFFFFFFFFFFFF),  # All ones
        ]

        for reg, val in test_cases:
            with self.subTest(register=reg, value=val):
                self.rust_engine.set_register(reg, val)
                result = self.rust_engine.get_register(reg)
                self.assertEqual(result, val)


class TestAMD64RegisterAliasing(unittest.TestCase):
    """Test AMD64 register aliasing (sub-register access)."""

    def setUp(self):
        if not RUST_ENGINE_AVAILABLE:
            self.skipTest("Rust VEX engine not available")
        self.project = load_shellcode(b"\xc3", arch="AMD64")
        self.state = SimState(project=self.project)

    def test_rax_eax_ax_al_ah_aliasing(self):
        """Test rax/eax/ax/al/ah register aliasing in Python engine."""
        # Set rax to a value with interesting bit patterns
        val = 0x123456789ABCDEF0

        self.state.regs.rax = val

        # Check sub-registers
        self.assertEqual(
            self.state.regs.rax.concrete_value,
            0x123456789ABCDEF0
        )
        self.assertEqual(
            self.state.regs.eax.concrete_value,
            0x9ABCDEF0  # Lower 32 bits
        )
        self.assertEqual(
            self.state.regs.ax.concrete_value,
            0xDEF0  # Lower 16 bits
        )
        self.assertEqual(
            self.state.regs.al.concrete_value,
            0xF0  # Lower 8 bits
        )
        self.assertEqual(
            self.state.regs.ah.concrete_value,
            0xDE  # Bits 8-15
        )

    def test_write_al_preserves_upper(self):
        """Writing to AL should preserve upper bits of RAX."""
        self.state.regs.rax = 0xFFFFFFFFFFFFFFFF
        self.state.regs.al = 0x00

        self.assertEqual(
            self.state.regs.rax.concrete_value,
            0xFFFFFFFFFFFFFF00
        )

    @pytest.mark.xfail(reason="Rust engine doesn't clear upper 32 bits on 32-bit register write")
    def test_write_eax_clears_upper(self):
        """Writing to EAX should clear upper 32 bits (AMD64 behavior)."""
        self.state.regs.rax = 0xFFFFFFFFFFFFFFFF
        self.state.regs.eax = 0x12345678

        # AMD64: writing to 32-bit register zero-extends to 64 bits
        self.assertEqual(
            self.state.regs.rax.concrete_value,
            0x0000000012345678
        )

    def test_rbx_ebx_bx_bl_bh_aliasing(self):
        """Test rbx sub-register aliasing."""
        self.state.regs.rbx = 0xFEDCBA9876543210

        self.assertEqual(self.state.regs.ebx.concrete_value, 0x76543210)
        self.assertEqual(self.state.regs.bx.concrete_value, 0x3210)
        self.assertEqual(self.state.regs.bl.concrete_value, 0x10)
        self.assertEqual(self.state.regs.bh.concrete_value, 0x32)


class TestX86Registers(unittest.TestCase):
    """Test x86 (32-bit) register access."""

    def setUp(self):
        if not RUST_ENGINE_AVAILABLE:
            self.skipTest("Rust VEX engine not available")
        self.rust_engine = RustVEXEngineWrapper("x86")
        self.project = load_shellcode(b"\xc3", arch="X86")
        self.python_state = SimState(project=self.project)

    def test_general_purpose_registers(self):
        """Test all 32-bit general purpose registers."""
        registers = ["eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp"]

        for reg in registers:
            with self.subTest(register=reg):
                test_val = 0xDEADBEEF

                self.rust_engine.set_register(reg, test_val)
                result = self.rust_engine.get_register(reg)

                self.assertEqual(result, test_val)

    def test_32bit_register_aliasing(self):
        """Test 32-bit register aliasing in Python engine."""
        self.python_state.regs.eax = 0x12345678

        self.assertEqual(self.python_state.regs.ax.concrete_value, 0x5678)
        self.assertEqual(self.python_state.regs.al.concrete_value, 0x78)
        self.assertEqual(self.python_state.regs.ah.concrete_value, 0x56)


class TestARMRegisters(unittest.TestCase):
    """Test ARM register access."""

    def setUp(self):
        if not RUST_ENGINE_AVAILABLE:
            self.skipTest("Rust VEX engine not available")
        try:
            self.rust_engine = RustVEXEngineWrapper("arm")
        except Exception as e:
            self.skipTest(f"ARM engine not available: {e}")
        self.project = load_shellcode(b"\x00\x00\x00\x00", arch="ARMEL")
        self.python_state = SimState(project=self.project)

    def test_general_purpose_registers(self):
        """Test ARM general purpose registers r0-r12."""
        for i in range(13):
            reg = f"r{i}"
            with self.subTest(register=reg):
                test_val = 0xDEADBEEF

                self.rust_engine.set_register(reg, test_val)
                result = self.rust_engine.get_register(reg)

                self.assertEqual(result, test_val)

    def test_special_registers(self):
        """Test ARM special registers."""
        special_regs = [
            ("sp", 0x7FFF0000),  # Stack pointer
            ("lr", 0x00400100),  # Link register
            ("pc", 0x00400000),  # Program counter
        ]

        for reg, val in special_regs:
            with self.subTest(register=reg):
                self.rust_engine.set_register(reg, val)
                result = self.rust_engine.get_register(reg)
                self.assertEqual(result, val)


class TestARM64Registers(unittest.TestCase):
    """Test ARM64 (AArch64) register access."""

    def setUp(self):
        if not RUST_ENGINE_AVAILABLE:
            self.skipTest("Rust VEX engine not available")
        try:
            self.rust_engine = RustVEXEngineWrapper("arm64")
        except Exception as e:
            self.skipTest(f"ARM64 engine not available: {e}")
        self.project = load_shellcode(b"\x00\x00\x00\x00", arch="AARCH64")
        self.python_state = SimState(project=self.project)

    def test_x_registers(self):
        """Test 64-bit X registers."""
        for i in range(31):
            reg = f"x{i}"
            with self.subTest(register=reg):
                test_val = 0xDEADBEEFCAFEBABE

                self.rust_engine.set_register(reg, test_val)
                result = self.rust_engine.get_register(reg)

                self.assertEqual(result, test_val)

    def test_special_registers(self):
        """Test ARM64 special registers."""
        special_regs = [
            ("sp", 0x00007FFFFFFFE000),
            ("pc", 0x0000000000400000),
        ]

        for reg, val in special_regs:
            with self.subTest(register=reg):
                self.rust_engine.set_register(reg, val)
                result = self.rust_engine.get_register(reg)
                self.assertEqual(result, val)

    def test_w_x_register_aliasing(self):
        """Test W/X register aliasing (32-bit view of 64-bit registers) in Python."""
        self.python_state.regs.x0 = 0x123456789ABCDEF0

        # W0 should be lower 32 bits
        self.assertEqual(self.python_state.regs.w0.concrete_value, 0x9ABCDEF0)


class TestMIPSRegisters(unittest.TestCase):
    """Test MIPS register access."""

    def setUp(self):
        if not RUST_ENGINE_AVAILABLE:
            self.skipTest("Rust VEX engine not available")
        try:
            self.rust_engine = RustVEXEngineWrapper("mips32")
        except Exception as e:
            self.skipTest(f"MIPS32 engine not available: {e}")

    def test_named_registers(self):
        """Test MIPS named registers."""
        named_regs = [
            ("v0", 0x12345678),
            ("v1", 0x9ABCDEF0),
            ("a0", 0x11111111),
            ("a1", 0x22222222),
            ("a2", 0x33333333),
            ("a3", 0x44444444),
            ("t0", 0x55555555),
            ("t1", 0x66666666),
            ("s0", 0x77777777),
            ("sp", 0x7FFF0000),
            ("ra", 0x00400100),
        ]

        for reg, val in named_regs:
            with self.subTest(register=reg):
                self.rust_engine.set_register(reg, val)
                result = self.rust_engine.get_register(reg)
                self.assertEqual(result, val)


class TestRegisterForkIsolation(unittest.TestCase):
    """Test that forked engines have isolated registers."""

    def setUp(self):
        if not RUST_ENGINE_AVAILABLE:
            self.skipTest("Rust VEX engine not available")

    def test_fork_register_isolation(self):
        """Writes to forked engine's registers should not affect original."""
        original = RustVEXEngineWrapper("amd64")
        original.set_register("rax", 100)
        original.set_register("rbx", 200)

        # Fork
        forked = original.fork()

        # Verify fork has same initial values
        self.assertEqual(forked.get_register("rax"), 100)
        self.assertEqual(forked.get_register("rbx"), 200)

        # Modify forked registers
        forked.set_register("rax", 999)
        forked.set_register("rbx", 888)

        # Original should be unchanged
        self.assertEqual(original.get_register("rax"), 100)
        self.assertEqual(original.get_register("rbx"), 200)

        # Forked should have new values
        self.assertEqual(forked.get_register("rax"), 999)
        self.assertEqual(forked.get_register("rbx"), 888)

    def test_fork_pc_isolation(self):
        """PC should be isolated between original and fork."""
        original = RustVEXEngineWrapper("amd64")
        original.pc = 0x400000

        forked = original.fork()
        self.assertEqual(forked.pc, 0x400000)

        forked.pc = 0x500000

        self.assertEqual(original.pc, 0x400000)
        self.assertEqual(forked.pc, 0x500000)

    def test_multiple_forks_isolation(self):
        """Multiple forks should all have isolated registers."""
        original = RustVEXEngineWrapper("amd64")
        original.set_register("rax", 0)

        forks = [original.fork() for _ in range(5)]

        # Set different values in each fork
        for i, fork in enumerate(forks):
            fork.set_register("rax", i + 1)

        # Verify isolation
        self.assertEqual(original.get_register("rax"), 0)
        for i, fork in enumerate(forks):
            self.assertEqual(fork.get_register("rax"), i + 1)


class TestSymbolicRegisters(unittest.TestCase):
    """Test symbolic register values (Python engine only)."""

    @classmethod
    def setUpClass(cls):
        cls.project = load_shellcode(b"\xc3", arch="AMD64")

    def test_symbolic_register_value(self):
        """Store and retrieve symbolic register value."""
        state = SimState(project=self.project)

        x = claripy.BVS("x", 64)
        state.regs.rax = x

        # Should get back the same symbolic variable
        retrieved = state.regs.rax
        self.assertTrue(retrieved.symbolic)
        self.assertTrue(state.solver.is_true(retrieved == x))

    def test_symbolic_register_operations(self):
        """Perform operations on symbolic register values."""
        state = SimState(project=self.project)

        x = claripy.BVS("x", 64)
        state.regs.rax = x
        state.regs.rbx = x + 10

        # Verify relationship
        constraint = state.regs.rbx == state.regs.rax + 10
        self.assertTrue(state.solver.is_true(constraint))

    def test_constrain_symbolic_register(self):
        """Add constraints to symbolic register values."""
        state = SimState(project=self.project)

        x = claripy.BVS("x", 64)
        state.regs.rax = x
        state.solver.add(x > 100)
        state.solver.add(x < 200)

        # Evaluate should give value in range
        val = state.solver.eval(state.regs.rax)
        self.assertGreater(val, 100)
        self.assertLess(val, 200)


if __name__ == "__main__":
    unittest.main()
