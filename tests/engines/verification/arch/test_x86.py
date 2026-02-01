"""
x86 (32-bit) Architecture-Specific Tests.

Tests for x86-specific VEX operations and instructions.
"""
from __future__ import annotations

import logging
import unittest

import claripy

import angr
from angr import SimState, load_shellcode
from angr.engines import HeavyVEXMixin

l = logging.getLogger(__name__)

try:
    from angr.engines.rust_vex import RUST_ENGINE_AVAILABLE, RustVEXEngineWrapper
except ImportError:
    RUST_ENGINE_AVAILABLE = False
    RustVEXEngineWrapper = None


class TestX86Operations(unittest.TestCase):
    """Test x86 32-bit operations."""

    @classmethod
    def setUpClass(cls):
        cls.project = load_shellcode(b"\xc3", arch="X86")
        cls.engine = HeavyVEXMixin(cls.project)
        cls.state = SimState(project=cls.project)

    def _run_op(self, op_name: str, *args):
        self.engine.state = self.state
        return self.engine._perform_vex_expr_Op(op_name, list(args))

    def test_32bit_add(self):
        """Test 32-bit addition."""
        a = claripy.BVV(0x7FFFFFFF, 32)
        b = claripy.BVV(1, 32)

        result = self._run_op("Iop_Add32", a, b)
        self.assertEqual(result.concrete_value, 0x80000000)

    def test_32bit_mul_overflow(self):
        """Test 32-bit multiply with overflow."""
        a = claripy.BVV(0x10000, 32)
        b = claripy.BVV(0x10000, 32)

        result = self._run_op("Iop_Mul32", a, b)
        # 0x10000 * 0x10000 = 0x100000000, truncated to 0
        self.assertEqual(result.concrete_value, 0)


class TestX86SegmentRegisters(unittest.TestCase):
    """Test x86 segment register handling."""

    def test_segment_register_access(self):
        """Test segment register access in Python engine."""
        p = load_shellcode(b"\xc3", arch="X86")
        state = SimState(project=p)

        # Set and get FS segment
        state.regs.fs = 0x0023
        result = state.regs.fs.concrete_value
        self.assertEqual(result, 0x0023)


class TestX86StackOperations(unittest.TestCase):
    """Test x86 stack operations."""

    def test_push_decrements_esp(self):
        """PUSH should decrement ESP by 4."""
        # push eax
        code = b"\x50\xCC"

        p = load_shellcode(code, arch="X86")
        state = p.factory.blank_state(
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        state.regs.esp = 0x7FFFF000
        state.memory.store(0x7FFFE000, b"\x00" * 0x1000)

        original_esp = state.regs.esp.concrete_value
        successor = state.step(num_inst=1)

        if successor.successors:
            final = successor.successors[0]
            new_esp = final.regs.esp.concrete_value
            self.assertEqual(new_esp, original_esp - 4)

    def test_pop_increments_esp(self):
        """POP should increment ESP by 4."""
        # pop eax
        code = b"\x58\xCC"

        p = load_shellcode(code, arch="X86")
        state = p.factory.blank_state(
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        state.regs.esp = 0x7FFFF000
        state.memory.store(0x7FFFE000, b"\x00" * 0x1000)

        original_esp = state.regs.esp.concrete_value
        successor = state.step(num_inst=1)

        if successor.successors:
            final = successor.successors[0]
            new_esp = final.regs.esp.concrete_value
            self.assertEqual(new_esp, original_esp + 4)


class TestX86RegisterAliasing(unittest.TestCase):
    """Test x86 register aliasing."""

    def test_eax_ax_al_ah_aliasing(self):
        """Test EAX sub-register aliasing."""
        p = load_shellcode(b"\xc3", arch="X86")
        state = SimState(project=p)

        state.regs.eax = 0x12345678

        self.assertEqual(state.regs.eax.concrete_value, 0x12345678)
        self.assertEqual(state.regs.ax.concrete_value, 0x5678)
        self.assertEqual(state.regs.al.concrete_value, 0x78)
        self.assertEqual(state.regs.ah.concrete_value, 0x56)

    def test_write_al_preserves_upper(self):
        """Writing AL should preserve upper bits."""
        p = load_shellcode(b"\xc3", arch="X86")
        state = SimState(project=p)

        state.regs.eax = 0xFFFFFFFF
        state.regs.al = 0x00

        self.assertEqual(state.regs.eax.concrete_value, 0xFFFFFF00)

    def test_write_ah_preserves_other(self):
        """Writing AH should preserve AL and upper bits."""
        p = load_shellcode(b"\xc3", arch="X86")
        state = SimState(project=p)

        state.regs.eax = 0x12345678
        state.regs.ah = 0xFF

        self.assertEqual(state.regs.eax.concrete_value, 0x1234FF78)


class TestX86RustEngineIntegration(unittest.TestCase):
    """Test x86-specific Rust engine functionality."""

    def setUp(self):
        if not RUST_ENGINE_AVAILABLE:
            self.skipTest("Rust VEX engine not available")
        self.engine = RustVEXEngineWrapper("x86")

    def test_32bit_register_values(self):
        """Test that 32-bit values are handled correctly."""
        test_vals = [
            0x00000000,
            0x7FFFFFFF,
            0x80000000,
            0xFFFFFFFF,
            0xDEADBEEF,
        ]

        for val in test_vals:
            with self.subTest(value=val):
                self.engine.set_register("eax", val)
                result = self.engine.get_register("eax")
                self.assertEqual(result, val)

    def test_all_gp_registers(self):
        """Test all 8 general purpose registers."""
        registers = ["eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp"]

        for i, reg in enumerate(registers):
            with self.subTest(register=reg):
                val = 0x10000000 + i
                self.engine.set_register(reg, val)
                result = self.engine.get_register(reg)
                self.assertEqual(result, val)


if __name__ == "__main__":
    unittest.main()
