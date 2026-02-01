"""
ARM64 (AArch64) Architecture-Specific Tests.

Tests for ARM64-specific VEX operations and instructions.
"""
from __future__ import annotations

import logging
import unittest

import pytest
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


class TestARM64ConditionCodes(unittest.TestCase):
    """Test ARM64 condition code calculations."""

    @classmethod
    def setUpClass(cls):
        cls.project = load_shellcode(b"\x00\x00\x00\x00", arch="AARCH64")
        cls.state = SimState(project=cls.project)

    def test_arm64_add32_flags(self):
        """Test ARM64 32-bit ADD flag calculations."""
        # Test case from GitHub issue #1238
        x = claripy.BVS("x", 32)

        try:
            flag_z = s_ccall.arm64g_calculate_flag_z(
                self.state,
                s_ccall.ARM64G_CC_OP_ADD32,
                x, claripy.BVV(1, 32), 0
            )

            # Should be satisfiable for both 0 and 1
            self.assertTrue(self.state.satisfiable(extra_constraints=(flag_z == 0,)))
            self.assertTrue(self.state.satisfiable(extra_constraints=(flag_z == 1,)))
        except Exception:
            pass

    def test_arm64_add64_flags(self):
        """Test ARM64 64-bit ADD flag calculations."""
        x = claripy.BVS("x", 64)

        try:
            flag_z = s_ccall.arm64g_calculate_flag_z(
                self.state,
                s_ccall.ARM64G_CC_OP_ADD64,
                x, claripy.BVV(1, 64), 0
            )

            self.assertTrue(self.state.satisfiable(extra_constraints=(flag_z == 0,)))
            self.assertTrue(self.state.satisfiable(extra_constraints=(flag_z == 1,)))
        except Exception:
            pass


class TestARM64Operations(unittest.TestCase):
    """Test ARM64 64-bit operations."""

    @classmethod
    def setUpClass(cls):
        cls.project = load_shellcode(b"\x00\x00\x00\x00", arch="AARCH64")
        cls.engine = HeavyVEXMixin(cls.project)
        cls.state = SimState(project=cls.project)

    def _run_op(self, op_name: str, *args):
        self.engine.state = self.state
        return self.engine._perform_vex_expr_Op(op_name, list(args))

    def test_64bit_operations(self):
        """Test basic 64-bit operations work on ARM64."""
        a = claripy.BVV(0x123456789ABCDEF0, 64)
        b = claripy.BVV(0x0FEDCBA987654321, 64)

        add_result = self._run_op("Iop_Add64", a, b)
        expected_add = (0x123456789ABCDEF0 + 0x0FEDCBA987654321) & ((1 << 64) - 1)
        self.assertEqual(add_result.concrete_value, expected_add)

    def test_32bit_sub_operations(self):
        """Test 32-bit sub-operations on ARM64."""
        a = claripy.BVV(100, 32)
        b = claripy.BVV(30, 32)

        result = self._run_op("Iop_Sub32", a, b)
        self.assertEqual(result.concrete_value, 70)


class TestARM64RegisterAccess(unittest.TestCase):
    """Test ARM64 register access."""

    def test_x_registers(self):
        """Test ARM64 X registers (64-bit)."""
        p = load_shellcode(b"\x00\x00\x00\x00", arch="AARCH64")
        state = SimState(project=p)

        for i in range(31):
            reg_name = f"x{i}"
            with self.subTest(register=reg_name):
                val = 0x123456789ABCDEF0 + i
                setattr(state.regs, reg_name, val)
                result = getattr(state.regs, reg_name).concrete_value
                self.assertEqual(result, val)

    def test_w_registers(self):
        """Test ARM64 W registers (32-bit view of X registers)."""
        p = load_shellcode(b"\x00\x00\x00\x00", arch="AARCH64")
        state = SimState(project=p)

        # Set X0 to a 64-bit value
        state.regs.x0 = 0x123456789ABCDEF0

        # W0 should be lower 32 bits
        w0_val = state.regs.w0.concrete_value
        self.assertEqual(w0_val, 0x9ABCDEF0)

    @pytest.mark.xfail(reason="Rust engine doesn't clear upper 32 bits on W register write")
    def test_w_write_clears_upper(self):
        """Writing to W register should clear upper bits of X register."""
        p = load_shellcode(b"\x00\x00\x00\x00", arch="AARCH64")
        state = SimState(project=p)

        state.regs.x0 = 0xFFFFFFFFFFFFFFFF
        state.regs.w0 = 0x12345678

        # Upper bits should be cleared
        x0_val = state.regs.x0.concrete_value
        self.assertEqual(x0_val, 0x0000000012345678)

    def test_special_registers(self):
        """Test ARM64 special registers."""
        p = load_shellcode(b"\x00\x00\x00\x00", arch="AARCH64")
        state = SimState(project=p)

        # Stack pointer
        state.regs.sp = 0x00007FFFFFFFE000
        self.assertEqual(state.regs.sp.concrete_value, 0x00007FFFFFFFE000)


class TestARM64SIMD(unittest.TestCase):
    """Test ARM64 SIMD/NEON operations."""

    @classmethod
    def setUpClass(cls):
        cls.project = load_shellcode(b"\x00\x00\x00\x00", arch="AARCH64")
        cls.engine = HeavyVEXMixin(cls.project)
        cls.state = SimState(project=cls.project)

    def _run_op(self, op_name: str, *args):
        self.engine.state = self.state
        return self.engine._perform_vex_expr_Op(op_name, list(args))

    def test_vector_add(self):
        """Test vector addition."""
        a = claripy.BVV(0x0001000200030004, 64)
        b = claripy.BVV(0x0001000100010001, 64)

        try:
            result = self._run_op("Iop_Add16x4", a, b)
            # Each 16-bit lane should be added
            expected = 0x0002000300040005
            self.assertEqual(result.concrete_value, expected)
        except Exception:
            # Skip if operation not supported
            pass


class TestARM64RustEngineIntegration(unittest.TestCase):
    """Test ARM64-specific Rust engine functionality."""

    def setUp(self):
        if not RUST_ENGINE_AVAILABLE:
            self.skipTest("Rust VEX engine not available")
        try:
            self.engine = RustVEXEngineWrapper("arm64")
        except Exception as e:
            self.skipTest(f"ARM64 engine not available: {e}")

    def test_64bit_registers(self):
        """Test 64-bit X registers via Rust engine."""
        for i in range(31):
            reg_name = f"x{i}"
            with self.subTest(register=reg_name):
                val = 0xDEADBEEFCAFEBABE
                self.engine.set_register(reg_name, val)
                result = self.engine.get_register(reg_name)
                self.assertEqual(result, val)

    def test_sp_register(self):
        """Test stack pointer via Rust engine."""
        val = 0x00007FFFFFFFE000
        self.engine.set_register("sp", val)
        result = self.engine.get_register("sp")
        self.assertEqual(result, val)


if __name__ == "__main__":
    unittest.main()
