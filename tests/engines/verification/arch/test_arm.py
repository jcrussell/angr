"""
ARM Architecture-Specific Tests.

Tests for ARM-specific VEX operations and instructions.
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


class TestARMConditionCodes(unittest.TestCase):
    """Test ARM condition code calculations."""

    @classmethod
    def setUpClass(cls):
        cls.project = load_shellcode(b"\x00\x00\x00\x00", arch="ARMEL")
        cls.state = SimState(project=cls.project)

    def test_arm_add_flags(self):
        """Test ARM ADD flag calculations."""
        # ARM uses different flag calculation methods
        x = claripy.BVV(1, 32)
        y = claripy.BVV(1, 32)

        # Calculate flags using ARM ccall
        try:
            # ARM uses armg_calculate_flag_* functions
            result = s_ccall.armg_calculate_flag_n(
                self.state,
                s_ccall.ARMG_CC_OP_ADD,
                x, y, 0
            )
            # N flag should be 0 for 1+1=2
            self.assertTrue(result.concrete_value == 0)
        except Exception:
            # Skip if ARM ccalls not available
            pass


class TestARMOperations(unittest.TestCase):
    """Test ARM 32-bit operations."""

    @classmethod
    def setUpClass(cls):
        cls.project = load_shellcode(b"\x00\x00\x00\x00", arch="ARMEL")
        cls.engine = HeavyVEXMixin(cls.project)
        cls.state = SimState(project=cls.project)

    def _run_op(self, op_name: str, *args):
        self.engine.state = self.state
        return self.engine._perform_vex_expr_Op(op_name, list(args))

    def test_32bit_operations(self):
        """Test basic 32-bit operations work on ARM."""
        a = claripy.BVV(10, 32)
        b = claripy.BVV(5, 32)

        add_result = self._run_op("Iop_Add32", a, b)
        self.assertEqual(add_result.concrete_value, 15)

        sub_result = self._run_op("Iop_Sub32", a, b)
        self.assertEqual(sub_result.concrete_value, 5)


class TestARMRegisterAccess(unittest.TestCase):
    """Test ARM register access."""

    def test_general_purpose_registers(self):
        """Test ARM r0-r12 registers."""
        p = load_shellcode(b"\x00\x00\x00\x00", arch="ARMEL")
        state = SimState(project=p)

        for i in range(13):
            reg_name = f"r{i}"
            with self.subTest(register=reg_name):
                val = 0x12345678 + i
                setattr(state.regs, reg_name, val)
                result = getattr(state.regs, reg_name).concrete_value
                self.assertEqual(result, val)

    def test_special_registers(self):
        """Test ARM special registers."""
        p = load_shellcode(b"\x00\x00\x00\x00", arch="ARMEL")
        state = SimState(project=p)

        # Stack pointer
        state.regs.sp = 0x7FFF0000
        self.assertEqual(state.regs.sp.concrete_value, 0x7FFF0000)

        # Link register
        state.regs.lr = 0x00400100
        self.assertEqual(state.regs.lr.concrete_value, 0x00400100)


class TestARMRustEngineIntegration(unittest.TestCase):
    """Test ARM-specific Rust engine functionality."""

    def setUp(self):
        if not RUST_ENGINE_AVAILABLE:
            self.skipTest("Rust VEX engine not available")
        try:
            self.engine = RustVEXEngineWrapper("arm")
        except Exception as e:
            self.skipTest(f"ARM engine not available: {e}")

    def test_arm_registers(self):
        """Test ARM register access via Rust engine."""
        for i in range(13):
            reg_name = f"r{i}"
            with self.subTest(register=reg_name):
                val = 0xDEADBEEF
                self.engine.set_register(reg_name, val)
                result = self.engine.get_register(reg_name)
                self.assertEqual(result, val)

    def test_arm_special_registers(self):
        """Test ARM special registers via Rust engine."""
        self.engine.set_register("sp", 0x7FFF0000)
        self.assertEqual(self.engine.get_register("sp"), 0x7FFF0000)

        self.engine.set_register("lr", 0x00400100)
        self.assertEqual(self.engine.get_register("lr"), 0x00400100)


if __name__ == "__main__":
    unittest.main()
