"""
AMD64 Architecture-Specific Tests.

Tests for AMD64-specific VEX operations and instructions.
"""
from __future__ import annotations

import logging
import unittest

import claripy

import angr
from angr import SimState, load_shellcode
from angr.engines import HeavyVEXMixin
import angr.engines.vex.claripy.ccall as s_ccall

l = logging.getLogger(__name__)

try:
    from angr.engines.rust_vex import RUST_ENGINE_AVAILABLE, RustVEXEngineWrapper
except ImportError:
    RUST_ENGINE_AVAILABLE = False
    RustVEXEngineWrapper = None


class TestAMD64FlagCalculations(unittest.TestCase):
    """Test AMD64 flag calculations (CF, PF, AF, ZF, SF, OF)."""

    @classmethod
    def setUpClass(cls):
        cls.project = load_shellcode(b"\xc3", arch="AMD64")
        cls.state = SimState(project=cls.project)

    def test_add_flags_no_overflow(self):
        """ADD without overflow should set flags correctly."""
        arg_l = claripy.BVV(1, 8)
        arg_r = claripy.BVV(1, 8)

        cf, pf, af, zf, sf, of = s_ccall.pc_actions_ADD(
            self.state, 8, arg_l, arg_r, 0, platform="AMD64"
        )

        self.assertTrue(self.state.solver.is_true(cf == 0))
        self.assertTrue(self.state.solver.is_true(zf == 0))
        self.assertTrue(self.state.solver.is_true(sf == 0))
        self.assertTrue(self.state.solver.is_true(of == 0))

    def test_add_flags_with_overflow(self):
        """ADD with overflow should set CF correctly."""
        arg_l = claripy.BVV(0xFFFFFFFF, 32)
        arg_r = claripy.BVV(1, 32)

        cf, pf, af, zf, sf, of = s_ccall.pc_actions_ADD(
            self.state, 32, arg_l, arg_r, 0, platform="AMD64"
        )

        self.assertTrue(self.state.solver.is_true(cf == 1))  # Carry occurred
        self.assertTrue(self.state.solver.is_true(zf == 1))  # Result is 0

    def test_add_flags_signed_overflow(self):
        """ADD with signed overflow should set OF correctly."""
        # 0x7FFFFFFF + 1 = 0x80000000 (positive + positive = negative)
        arg_l = claripy.BVV(0x7FFFFFFF, 32)
        arg_r = claripy.BVV(1, 32)

        cf, pf, af, zf, sf, of = s_ccall.pc_actions_ADD(
            self.state, 32, arg_l, arg_r, 0, platform="AMD64"
        )

        self.assertTrue(self.state.solver.is_true(of == 1))  # Signed overflow
        self.assertTrue(self.state.solver.is_true(sf == 1))  # Result is negative

    def test_sub_flags_no_borrow(self):
        """SUB without borrow should set flags correctly."""
        arg_l = claripy.BVV(5, 8)
        arg_r = claripy.BVV(3, 8)

        cf, pf, af, zf, sf, of = s_ccall.pc_actions_SUB(
            self.state, 8, arg_l, arg_r, 0, platform="AMD64"
        )

        self.assertTrue(self.state.solver.is_true(cf == 0))  # No borrow
        self.assertTrue(self.state.solver.is_true(zf == 0))  # Result != 0

    def test_sub_flags_equal(self):
        """SUB of equal values should set ZF."""
        arg_l = claripy.BVV(5, 8)
        arg_r = claripy.BVV(5, 8)

        cf, pf, af, zf, sf, of = s_ccall.pc_actions_SUB(
            self.state, 8, arg_l, arg_r, 0, platform="AMD64"
        )

        self.assertTrue(self.state.solver.is_true(zf == 1))  # Result is 0

    def test_sub_flags_with_borrow(self):
        """SUB with borrow should set CF."""
        arg_l = claripy.BVV(3, 8)
        arg_r = claripy.BVV(5, 8)

        cf, pf, af, zf, sf, of = s_ccall.pc_actions_SUB(
            self.state, 8, arg_l, arg_r, 0, platform="AMD64"
        )

        self.assertTrue(self.state.solver.is_true(cf == 1))  # Borrow occurred
        self.assertTrue(self.state.solver.is_true(sf == 1))  # Result is negative


class TestAMD6464BitOperations(unittest.TestCase):
    """Test 64-bit specific operations."""

    @classmethod
    def setUpClass(cls):
        cls.project = load_shellcode(b"\xc3", arch="AMD64")
        cls.engine = HeavyVEXMixin(cls.project)
        cls.state = SimState(project=cls.project)

    def _run_op(self, op_name: str, *args):
        self.engine.state = self.state
        return self.engine._perform_vex_expr_Op(op_name, list(args))

    def test_64bit_add_no_truncation(self):
        """64-bit add should not truncate."""
        a = claripy.BVV(0x7FFFFFFFFFFFFFFF, 64)
        b = claripy.BVV(1, 64)

        result = self._run_op("Iop_Add64", a, b)

        self.assertEqual(result.concrete_value, 0x8000000000000000)

    def test_64bit_mul_no_truncation(self):
        """64-bit multiply should truncate to 64 bits."""
        a = claripy.BVV(0x100000000, 64)  # 2^32
        b = claripy.BVV(0x100000000, 64)  # 2^32

        result = self._run_op("Iop_Mul64", a, b)

        # 2^32 * 2^32 = 2^64, which truncates to 0 in 64 bits
        self.assertEqual(result.concrete_value, 0)

    def test_64bit_shift(self):
        """64-bit shift should work across full width."""
        val = claripy.BVV(1, 64)
        shift = claripy.BVV(63, 8)

        result = self._run_op("Iop_Shl64", val, shift)

        self.assertEqual(result.concrete_value, 0x8000000000000000)


class TestAMD64REXPrefixBehavior(unittest.TestCase):
    """Test REX prefix handling for 64-bit registers."""

    def test_rex_register_extension(self):
        """Test that R8-R15 work correctly."""
        # mov r8, 0xDEADBEEFCAFEBABE
        code = b"\x49\xB8" + b"\xBE\xBA\xFE\xCA\xEF\xBE\xAD\xDE" + b"\xCC"

        p = load_shellcode(code, arch="AMD64")
        state = p.factory.blank_state(
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        successor = state.step(num_inst=1)

        if successor.successors:
            final = successor.successors[0]
            self.assertEqual(final.regs.r8.concrete_value, 0xDEADBEEFCAFEBABE)

    def test_64bit_operand_size(self):
        """Test that 64-bit operand size works."""
        # mov rax, rbx
        code = b"\x48\x89\xD8" + b"\xCC"

        p = load_shellcode(code, arch="AMD64")
        state = p.factory.blank_state(
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        state.regs.rbx = 0xFFFFFFFFFFFFFFFF
        successor = state.step(num_inst=1)

        if successor.successors:
            final = successor.successors[0]
            self.assertEqual(final.regs.rax.concrete_value, 0xFFFFFFFFFFFFFFFF)


class TestAMD64SSERegisters(unittest.TestCase):
    """Test SSE/XMM register operations."""

    def test_xmm_register_access(self):
        """Test XMM register read/write."""
        p = load_shellcode(b"\xc3", arch="AMD64")
        state = SimState(project=p)

        # Set XMM0 to a 128-bit value
        val = 0x123456789ABCDEF0FEDCBA9876543210
        state.regs.xmm0 = val

        result = state.regs.xmm0.concrete_value
        self.assertEqual(result, val)

    def test_xmm_low_quadword(self):
        """Test XMM low 64-bit access."""
        p = load_shellcode(b"\xc3", arch="AMD64")
        state = SimState(project=p)

        state.regs.xmm0 = 0x123456789ABCDEF0FEDCBA9876543210

        # xmm0lq is the low 64 bits
        low_qword = state.regs.xmm0lq.concrete_value
        self.assertEqual(low_qword, 0xFEDCBA9876543210)


class TestAMD64RustEngineIntegration(unittest.TestCase):
    """Test AMD64-specific Rust engine functionality."""

    def setUp(self):
        if not RUST_ENGINE_AVAILABLE:
            self.skipTest("Rust VEX engine not available")
        self.engine = RustVEXEngineWrapper("amd64")

    def test_64bit_register_values(self):
        """Test that 64-bit values are preserved correctly."""
        test_vals = [
            0x0000000000000000,
            0x7FFFFFFFFFFFFFFF,
            0x8000000000000000,
            0xFFFFFFFFFFFFFFFF,
            0xDEADBEEFCAFEBABE,
        ]

        for val in test_vals:
            with self.subTest(value=val):
                self.engine.set_register("rax", val)
                result = self.engine.get_register("rax")
                self.assertEqual(result, val)

    def test_all_gp_registers(self):
        """Test all 16 general purpose registers."""
        registers = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
                     "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"]

        for i, reg in enumerate(registers):
            with self.subTest(register=reg):
                val = 0x1000000000000000 + i
                self.engine.set_register(reg, val)
                result = self.engine.get_register(reg)
                self.assertEqual(result, val)

    def test_64bit_memory_access(self):
        """Test 64-bit memory read/write."""
        self.engine.map_memory(0x1000, 0x1000)

        val = 0xDEADBEEFCAFEBABE
        data = val.to_bytes(8, "little")

        self.engine.write_memory(0x1000, data)
        result = self.engine.read_memory(0x1000, 8)

        self.assertEqual(result, data)


if __name__ == "__main__":
    unittest.main()
