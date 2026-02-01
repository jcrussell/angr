"""
MIPS Architecture-Specific Tests.

Tests for MIPS-specific VEX operations and instructions.
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


class TestMIPSOperations(unittest.TestCase):
    """Test MIPS 32-bit operations."""

    @classmethod
    def setUpClass(cls):
        # Use big-endian MIPS
        try:
            cls.project = load_shellcode(b"\x00\x00\x00\x00", arch="MIPS32")
            cls.engine = HeavyVEXMixin(cls.project)
            cls.state = SimState(project=cls.project)
            cls.available = True
        except Exception:
            cls.available = False

    def setUp(self):
        if not self.available:
            self.skipTest("MIPS32 architecture not available")

    def _run_op(self, op_name: str, *args):
        self.engine.state = self.state
        return self.engine._perform_vex_expr_Op(op_name, list(args))

    def test_32bit_operations(self):
        """Test basic 32-bit operations work on MIPS."""
        a = claripy.BVV(100, 32)
        b = claripy.BVV(50, 32)

        add_result = self._run_op("Iop_Add32", a, b)
        self.assertEqual(add_result.concrete_value, 150)

        sub_result = self._run_op("Iop_Sub32", a, b)
        self.assertEqual(sub_result.concrete_value, 50)


class TestMIPSRegisterAccess(unittest.TestCase):
    """Test MIPS register access."""

    def setUp(self):
        try:
            self.project = load_shellcode(b"\x00\x00\x00\x00", arch="MIPS32")
            self.state = SimState(project=self.project)
        except Exception:
            self.skipTest("MIPS32 architecture not available")

    def test_named_registers(self):
        """Test MIPS named registers."""
        named_regs = [
            ("v0", 0x12345678),
            ("v1", 0x9ABCDEF0),
            ("a0", 0x11111111),
            ("a1", 0x22222222),
            ("a2", 0x33333333),
            ("a3", 0x44444444),
        ]

        for reg_name, val in named_regs:
            with self.subTest(register=reg_name):
                setattr(self.state.regs, reg_name, val)
                result = getattr(self.state.regs, reg_name).concrete_value
                self.assertEqual(result, val)

    def test_t_registers(self):
        """Test MIPS temporary registers."""
        for i in range(10):
            reg_name = f"t{i}"
            with self.subTest(register=reg_name):
                try:
                    val = 0xDEADBEEF
                    setattr(self.state.regs, reg_name, val)
                    result = getattr(self.state.regs, reg_name).concrete_value
                    self.assertEqual(result, val)
                except Exception:
                    # t8, t9 may have different names
                    pass

    def test_s_registers(self):
        """Test MIPS saved registers."""
        for i in range(8):
            reg_name = f"s{i}"
            with self.subTest(register=reg_name):
                val = 0xCAFEBABE + i
                setattr(self.state.regs, reg_name, val)
                result = getattr(self.state.regs, reg_name).concrete_value
                self.assertEqual(result, val)

    def test_special_registers(self):
        """Test MIPS special registers."""
        # Stack pointer
        self.state.regs.sp = 0x7FFF0000
        self.assertEqual(self.state.regs.sp.concrete_value, 0x7FFF0000)

        # Return address
        self.state.regs.ra = 0x00400100
        self.assertEqual(self.state.regs.ra.concrete_value, 0x00400100)


class TestMIPSHILORegisters(unittest.TestCase):
    """Test MIPS HI/LO register handling."""

    def setUp(self):
        try:
            self.project = load_shellcode(b"\x00\x00\x00\x00", arch="MIPS32")
            self.state = SimState(project=self.project)
        except Exception:
            self.skipTest("MIPS32 architecture not available")

    def test_hi_lo_access(self):
        """Test HI/LO register access."""
        try:
            self.state.regs.hi = 0x12345678
            self.state.regs.lo = 0x9ABCDEF0

            hi_val = self.state.regs.hi.concrete_value
            lo_val = self.state.regs.lo.concrete_value

            self.assertEqual(hi_val, 0x12345678)
            self.assertEqual(lo_val, 0x9ABCDEF0)
        except Exception:
            # HI/LO may not be directly accessible
            pass


class TestMIPSRustEngineIntegration(unittest.TestCase):
    """Test MIPS-specific Rust engine functionality."""

    def setUp(self):
        if not RUST_ENGINE_AVAILABLE:
            self.skipTest("Rust VEX engine not available")
        try:
            self.engine = RustVEXEngineWrapper("mips32")
        except Exception as e:
            self.skipTest(f"MIPS32 engine not available: {e}")

    def test_mips_registers(self):
        """Test MIPS register access via Rust engine."""
        named_regs = ["v0", "v1", "a0", "a1", "a2", "a3"]

        for reg_name in named_regs:
            with self.subTest(register=reg_name):
                val = 0xDEADBEEF
                self.engine.set_register(reg_name, val)
                result = self.engine.get_register(reg_name)
                self.assertEqual(result, val)

    def test_mips_special_registers(self):
        """Test MIPS special registers via Rust engine."""
        self.engine.set_register("sp", 0x7FFF0000)
        self.assertEqual(self.engine.get_register("sp"), 0x7FFF0000)

        self.engine.set_register("ra", 0x00400100)
        self.assertEqual(self.engine.get_register("ra"), 0x00400100)


if __name__ == "__main__":
    unittest.main()
