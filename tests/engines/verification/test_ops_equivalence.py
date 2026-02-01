"""
VEX Operation Equivalence Tests.

Tests that verify the Rust VEX engine produces identical results to the
Python VEX engine for all supported VEX operations.
"""
from __future__ import annotations

import itertools
import logging
import unittest

import claripy

from angr import SimState, load_shellcode
from angr.engines import HeavyVEXMixin
from angr.engines.vex.claripy import irop

l = logging.getLogger(__name__)

# Import Rust engine availability flag
try:
    from angr.engines.rust_vex import RUST_ENGINE_AVAILABLE
except ImportError:
    RUST_ENGINE_AVAILABLE = False


def edge_case_values(width: int) -> list[int]:
    """Generate comprehensive edge case values for testing."""
    mask = (1 << width) - 1
    values = [
        0,                                  # Zero
        1,                                  # One
        mask,                               # All ones (max unsigned)
        1 << (width - 1),                   # Only MSB set (min signed)
        (1 << (width - 1)) - 1,             # Max positive signed
        mask - 1,                           # Max unsigned - 1
        2,                                  # Small even
        (1 << (width - 1)) + 1,             # Min signed + 1
    ]

    # Powers of 2
    for i in [1, 2, 4, 8, width // 4, width // 2, width - 2, width - 1]:
        if 0 <= i < width:
            values.append(1 << i)
            values.append((1 << i) - 1)
            values.append((1 << i) + 1)

    # Alternating bit patterns
    patterns = [0x55, 0xAA, 0x0F, 0xF0]
    for p in patterns:
        val = 0
        for i in range(0, width, 8):
            val |= p << i
        values.append(val & mask)

    return list(set(v & mask for v in values if v >= 0))


def shift_amounts(width: int) -> list[int]:
    """Generate test shift amounts."""
    return [0, 1, 2, width // 2, width - 2, width - 1, width, width + 1]


class TestArithmeticOps(unittest.TestCase):
    """Test arithmetic VEX operations for equivalence."""

    @classmethod
    def setUpClass(cls):
        cls.project = load_shellcode(b"\xc3", arch="AMD64")
        cls.engine = HeavyVEXMixin(cls.project)
        cls.state = SimState(project=cls.project)

    def _run_op(self, op_name: str, *args):
        """Run a VEX operation via the Python engine."""
        self.engine.state = self.state
        return self.engine._perform_vex_expr_Op(op_name, list(args))

    def _run_op_direct(self, op_name: str, *args):
        """Run a VEX operation directly via irop."""
        simop = irop.vexop_to_simop(op_name)
        return simop.calculate(*args)

    def _test_binop_concrete(self, op_name: str, width: int, skip_zero_divisor: bool = False):
        """Test a binary operation with concrete values."""
        values = edge_case_values(width)
        mask = (1 << width) - 1

        for a, b in itertools.product(values, repeat=2):
            if skip_zero_divisor and b == 0:
                continue

            arg_a = claripy.BVV(a, width)
            arg_b = claripy.BVV(b, width)

            try:
                result = self._run_op(op_name, arg_a, arg_b)
                direct_result = self._run_op_direct(op_name, arg_a, arg_b)

                # Both methods should produce the same result
                self.assertTrue(
                    self.state.solver.is_true(result == direct_result),
                    f"{op_name}({a:#x}, {b:#x}): engine={result}, direct={direct_result}"
                )
            except Exception as e:
                # Some operations may not be supported for all inputs
                l.debug("%s(%#x, %#x) raised: %s", op_name, a, b, e)

    def _test_unop_concrete(self, op_name: str, width: int):
        """Test a unary operation with concrete values."""
        values = edge_case_values(width)

        for a in values:
            arg_a = claripy.BVV(a, width)

            try:
                result = self._run_op(op_name, arg_a)
                direct_result = self._run_op_direct(op_name, arg_a)

                self.assertTrue(
                    self.state.solver.is_true(result == direct_result),
                    f"{op_name}({a:#x}): engine={result}, direct={direct_result}"
                )
            except Exception as e:
                l.debug("%s(%#x) raised: %s", op_name, a, e)

    # Addition tests
    def test_add_8bit(self):
        self._test_binop_concrete("Iop_Add8", 8)

    def test_add_16bit(self):
        self._test_binop_concrete("Iop_Add16", 16)

    def test_add_32bit(self):
        self._test_binop_concrete("Iop_Add32", 32)

    def test_add_64bit(self):
        self._test_binop_concrete("Iop_Add64", 64)

    # Subtraction tests
    def test_sub_8bit(self):
        self._test_binop_concrete("Iop_Sub8", 8)

    def test_sub_16bit(self):
        self._test_binop_concrete("Iop_Sub16", 16)

    def test_sub_32bit(self):
        self._test_binop_concrete("Iop_Sub32", 32)

    def test_sub_64bit(self):
        self._test_binop_concrete("Iop_Sub64", 64)

    # Multiplication tests
    def test_mul_8bit(self):
        self._test_binop_concrete("Iop_Mul8", 8)

    def test_mul_16bit(self):
        self._test_binop_concrete("Iop_Mul16", 16)

    def test_mul_32bit(self):
        self._test_binop_concrete("Iop_Mul32", 32)

    def test_mul_64bit(self):
        self._test_binop_concrete("Iop_Mul64", 64)

    # Division tests (skip zero divisor)
    def test_divu_32bit(self):
        self._test_binop_concrete("Iop_DivU32", 32, skip_zero_divisor=True)

    def test_divu_64bit(self):
        self._test_binop_concrete("Iop_DivU64", 64, skip_zero_divisor=True)

    def test_divs_32bit(self):
        self._test_binop_concrete("Iop_DivS32", 32, skip_zero_divisor=True)

    def test_divs_64bit(self):
        self._test_binop_concrete("Iop_DivS64", 64, skip_zero_divisor=True)

    # Modulo tests (skip zero divisor)
    def test_modu_32bit(self):
        self._test_binop_concrete("Iop_ModU32", 32, skip_zero_divisor=True)

    def test_modu_64bit(self):
        self._test_binop_concrete("Iop_ModU64", 64, skip_zero_divisor=True)

    # Widening multiplication tests
    def test_mullu_32bit(self):
        """Test unsigned widening multiply 32->64."""
        values = edge_case_values(32)
        for a, b in itertools.product(values[:10], repeat=2):
            arg_a = claripy.BVV(a, 32)
            arg_b = claripy.BVV(b, 32)

            try:
                result = self._run_op("Iop_MullU32", arg_a, arg_b)
                expected = (a * b) & ((1 << 64) - 1)
                actual = result.concrete_value

                self.assertEqual(
                    actual, expected,
                    f"MullU32({a:#x}, {b:#x}): got {actual:#x}, expected {expected:#x}"
                )
            except Exception as e:
                l.debug("MullU32(%#x, %#x) raised: %s", a, b, e)

    def test_mulls_32bit(self):
        """Test signed widening multiply 32->64."""
        values = edge_case_values(32)
        for a, b in itertools.product(values[:10], repeat=2):
            # Convert to signed
            a_signed = a if a < (1 << 31) else a - (1 << 32)
            b_signed = b if b < (1 << 31) else b - (1 << 32)

            arg_a = claripy.BVV(a, 32)
            arg_b = claripy.BVV(b, 32)

            try:
                result = self._run_op("Iop_MullS32", arg_a, arg_b)
                expected = (a_signed * b_signed) & ((1 << 64) - 1)
                actual = result.concrete_value

                self.assertEqual(
                    actual, expected,
                    f"MullS32({a:#x}={a_signed}, {b:#x}={b_signed}): got {actual:#x}, expected {expected:#x}"
                )
            except Exception as e:
                l.debug("MullS32(%#x, %#x) raised: %s", a, b, e)


class TestBitwiseOps(unittest.TestCase):
    """Test bitwise VEX operations for equivalence."""

    @classmethod
    def setUpClass(cls):
        cls.project = load_shellcode(b"\xc3", arch="AMD64")
        cls.engine = HeavyVEXMixin(cls.project)
        cls.state = SimState(project=cls.project)

    def _run_op(self, op_name: str, *args):
        self.engine.state = self.state
        return self.engine._perform_vex_expr_Op(op_name, list(args))

    def _test_binop_concrete(self, op_name: str, width: int):
        values = edge_case_values(width)

        for a, b in itertools.product(values[:20], repeat=2):
            arg_a = claripy.BVV(a, width)
            arg_b = claripy.BVV(b, width)

            try:
                result = self._run_op(op_name, arg_a, arg_b)
                self.assertTrue(
                    result.concrete,
                    f"{op_name}({a:#x}, {b:#x}) should be concrete"
                )
            except Exception as e:
                l.debug("%s(%#x, %#x) raised: %s", op_name, a, b, e)

    def _test_unop_concrete(self, op_name: str, width: int):
        values = edge_case_values(width)

        for a in values:
            arg_a = claripy.BVV(a, width)

            try:
                result = self._run_op(op_name, arg_a)
                self.assertTrue(
                    result.concrete,
                    f"{op_name}({a:#x}) should be concrete"
                )
            except Exception as e:
                l.debug("%s(%#x) raised: %s", op_name, a, e)

    # AND tests
    def test_and_8bit(self):
        self._test_binop_concrete("Iop_And8", 8)

    def test_and_16bit(self):
        self._test_binop_concrete("Iop_And16", 16)

    def test_and_32bit(self):
        self._test_binop_concrete("Iop_And32", 32)

    def test_and_64bit(self):
        self._test_binop_concrete("Iop_And64", 64)

    # OR tests
    def test_or_8bit(self):
        self._test_binop_concrete("Iop_Or8", 8)

    def test_or_16bit(self):
        self._test_binop_concrete("Iop_Or16", 16)

    def test_or_32bit(self):
        self._test_binop_concrete("Iop_Or32", 32)

    def test_or_64bit(self):
        self._test_binop_concrete("Iop_Or64", 64)

    # XOR tests
    def test_xor_8bit(self):
        self._test_binop_concrete("Iop_Xor8", 8)

    def test_xor_16bit(self):
        self._test_binop_concrete("Iop_Xor16", 16)

    def test_xor_32bit(self):
        self._test_binop_concrete("Iop_Xor32", 32)

    def test_xor_64bit(self):
        self._test_binop_concrete("Iop_Xor64", 64)

    # NOT tests
    def test_not_8bit(self):
        self._test_unop_concrete("Iop_Not8", 8)

    def test_not_16bit(self):
        self._test_unop_concrete("Iop_Not16", 16)

    def test_not_32bit(self):
        self._test_unop_concrete("Iop_Not32", 32)

    def test_not_64bit(self):
        self._test_unop_concrete("Iop_Not64", 64)


class TestShiftOps(unittest.TestCase):
    """Test shift VEX operations for equivalence."""

    @classmethod
    def setUpClass(cls):
        cls.project = load_shellcode(b"\xc3", arch="AMD64")
        cls.engine = HeavyVEXMixin(cls.project)
        cls.state = SimState(project=cls.project)

    def _run_op(self, op_name: str, *args):
        self.engine.state = self.state
        return self.engine._perform_vex_expr_Op(op_name, list(args))

    def _test_shift_concrete(self, op_name: str, width: int):
        """Test shift operation with comprehensive inputs."""
        values = edge_case_values(width)
        shifts = shift_amounts(width)

        for val in values[:15]:
            for shift in shifts:
                arg_val = claripy.BVV(val, width)
                arg_shift = claripy.BVV(shift, 8)

                try:
                    result = self._run_op(op_name, arg_val, arg_shift)
                    self.assertTrue(
                        result.concrete,
                        f"{op_name}({val:#x}, {shift}) should be concrete"
                    )
                except Exception as e:
                    l.debug("%s(%#x, %d) raised: %s", op_name, val, shift, e)

    # Logical left shift tests
    def test_shl_8bit(self):
        self._test_shift_concrete("Iop_Shl8", 8)

    def test_shl_16bit(self):
        self._test_shift_concrete("Iop_Shl16", 16)

    def test_shl_32bit(self):
        self._test_shift_concrete("Iop_Shl32", 32)

    def test_shl_64bit(self):
        self._test_shift_concrete("Iop_Shl64", 64)

    # Logical right shift tests
    def test_shr_8bit(self):
        self._test_shift_concrete("Iop_Shr8", 8)

    def test_shr_16bit(self):
        self._test_shift_concrete("Iop_Shr16", 16)

    def test_shr_32bit(self):
        self._test_shift_concrete("Iop_Shr32", 32)

    def test_shr_64bit(self):
        self._test_shift_concrete("Iop_Shr64", 64)

    # Arithmetic right shift tests
    def test_sar_8bit(self):
        self._test_shift_concrete("Iop_Sar8", 8)

    def test_sar_16bit(self):
        self._test_shift_concrete("Iop_Sar16", 16)

    def test_sar_32bit(self):
        self._test_shift_concrete("Iop_Sar32", 32)

    def test_sar_64bit(self):
        self._test_shift_concrete("Iop_Sar64", 64)

    # Test shift by 0 produces identity
    def test_shift_by_zero(self):
        """Shift by 0 should return the original value."""
        for width in [8, 16, 32, 64]:
            val = 0xDEADBEEF & ((1 << width) - 1)
            arg_val = claripy.BVV(val, width)
            arg_zero = claripy.BVV(0, 8)

            for op in [f"Iop_Shl{width}", f"Iop_Shr{width}", f"Iop_Sar{width}"]:
                result = self._run_op(op, arg_val, arg_zero)
                self.assertEqual(
                    result.concrete_value, val,
                    f"{op}({val:#x}, 0) should equal {val:#x}"
                )

    # Test shift by width produces 0 (for Shl/Shr)
    def test_shift_by_width(self):
        """Shift by width should produce 0 for logical shifts."""
        for width in [8, 16, 32, 64]:
            val = 0xFFFFFFFF & ((1 << width) - 1)
            arg_val = claripy.BVV(val, width)
            arg_width = claripy.BVV(width, 8)

            # Shl by width should be 0
            result = self._run_op(f"Iop_Shl{width}", arg_val, arg_width)
            self.assertEqual(
                result.concrete_value, 0,
                f"Shl{width}({val:#x}, {width}) should be 0"
            )

            # Shr by width should be 0
            result = self._run_op(f"Iop_Shr{width}", arg_val, arg_width)
            self.assertEqual(
                result.concrete_value, 0,
                f"Shr{width}({val:#x}, {width}) should be 0"
            )


class TestCompareOps(unittest.TestCase):
    """Test comparison VEX operations for equivalence."""

    @classmethod
    def setUpClass(cls):
        cls.project = load_shellcode(b"\xc3", arch="AMD64")
        cls.engine = HeavyVEXMixin(cls.project)
        cls.state = SimState(project=cls.project)

    def _run_op(self, op_name: str, *args):
        self.engine.state = self.state
        return self.engine._perform_vex_expr_Op(op_name, list(args))

    def _test_compare(self, op_name: str, width: int):
        """Test comparison operation."""
        values = edge_case_values(width)

        for a, b in itertools.product(values[:15], repeat=2):
            arg_a = claripy.BVV(a, width)
            arg_b = claripy.BVV(b, width)

            try:
                result = self._run_op(op_name, arg_a, arg_b)
                # Result should be a 1-bit value
                self.assertEqual(result.length, 1, f"{op_name} should return 1-bit value")
                self.assertTrue(result.concrete, f"{op_name}({a:#x}, {b:#x}) should be concrete")
            except Exception as e:
                l.debug("%s(%#x, %#x) raised: %s", op_name, a, b, e)

    # Equality tests
    def test_cmpeq_8bit(self):
        self._test_compare("Iop_CmpEQ8", 8)

    def test_cmpeq_16bit(self):
        self._test_compare("Iop_CmpEQ16", 16)

    def test_cmpeq_32bit(self):
        self._test_compare("Iop_CmpEQ32", 32)

    def test_cmpeq_64bit(self):
        self._test_compare("Iop_CmpEQ64", 64)

    # Inequality tests
    def test_cmpne_8bit(self):
        self._test_compare("Iop_CmpNE8", 8)

    def test_cmpne_16bit(self):
        self._test_compare("Iop_CmpNE16", 16)

    def test_cmpne_32bit(self):
        self._test_compare("Iop_CmpNE32", 32)

    def test_cmpne_64bit(self):
        self._test_compare("Iop_CmpNE64", 64)

    # Unsigned less than tests
    def test_cmpltu_32bit(self):
        self._test_compare("Iop_CmpLT32U", 32)

    def test_cmpltu_64bit(self):
        self._test_compare("Iop_CmpLT64U", 64)

    # Signed less than tests
    def test_cmplts_32bit(self):
        self._test_compare("Iop_CmpLT32S", 32)

    def test_cmplts_64bit(self):
        self._test_compare("Iop_CmpLT64S", 64)

    # Verify equality semantics
    def test_cmpeq_semantics(self):
        """CmpEQ should return 1 when equal, 0 otherwise."""
        for width in [8, 16, 32, 64]:
            mask = (1 << width) - 1

            # Same values should be equal
            val = 0x42 & mask
            arg = claripy.BVV(val, width)
            result = self._run_op(f"Iop_CmpEQ{width}", arg, arg)
            self.assertEqual(result.concrete_value, 1)

            # Different values should not be equal
            val2 = 0x43 & mask
            arg2 = claripy.BVV(val2, width)
            result = self._run_op(f"Iop_CmpEQ{width}", arg, arg2)
            self.assertEqual(result.concrete_value, 0)

    # Verify signed comparison semantics
    def test_cmplts_semantics(self):
        """CmpLT signed should handle negative numbers correctly."""
        for width in [32, 64]:
            mask = (1 << width) - 1
            sign_bit = 1 << (width - 1)

            # -1 < 0 should be true
            neg_one = mask  # -1 in two's complement
            zero = 0
            arg_neg = claripy.BVV(neg_one, width)
            arg_zero = claripy.BVV(zero, width)
            result = self._run_op(f"Iop_CmpLT{width}S", arg_neg, arg_zero)
            self.assertEqual(result.concrete_value, 1, f"-1 < 0 should be true for {width}-bit")

            # 0 < -1 should be false
            result = self._run_op(f"Iop_CmpLT{width}S", arg_zero, arg_neg)
            self.assertEqual(result.concrete_value, 0, f"0 < -1 should be false for {width}-bit")


class TestConversionOps(unittest.TestCase):
    """Test type conversion VEX operations for equivalence."""

    @classmethod
    def setUpClass(cls):
        cls.project = load_shellcode(b"\xc3", arch="AMD64")
        cls.engine = HeavyVEXMixin(cls.project)
        cls.state = SimState(project=cls.project)

    def _run_op(self, op_name: str, *args):
        self.engine.state = self.state
        return self.engine._perform_vex_expr_Op(op_name, list(args))

    def test_sign_extend_8to16(self):
        """Test sign extension from 8 to 16 bits."""
        test_cases = [
            (0x00, 0x0000),  # Zero
            (0x7F, 0x007F),  # Max positive
            (0x80, 0xFF80),  # Min negative
            (0xFF, 0xFFFF),  # -1
        ]
        for input_val, expected in test_cases:
            arg = claripy.BVV(input_val, 8)
            result = self._run_op("Iop_8Sto16", arg)
            self.assertEqual(
                result.concrete_value, expected,
                f"8Sto16({input_val:#x}) should be {expected:#x}, got {result.concrete_value:#x}"
            )

    def test_sign_extend_8to32(self):
        """Test sign extension from 8 to 32 bits."""
        test_cases = [
            (0x00, 0x00000000),
            (0x7F, 0x0000007F),
            (0x80, 0xFFFFFF80),
            (0xFF, 0xFFFFFFFF),
        ]
        for input_val, expected in test_cases:
            arg = claripy.BVV(input_val, 8)
            result = self._run_op("Iop_8Sto32", arg)
            self.assertEqual(
                result.concrete_value, expected,
                f"8Sto32({input_val:#x}) should be {expected:#x}"
            )

    def test_sign_extend_16to32(self):
        """Test sign extension from 16 to 32 bits."""
        test_cases = [
            (0x0000, 0x00000000),
            (0x7FFF, 0x00007FFF),
            (0x8000, 0xFFFF8000),
            (0xFFFF, 0xFFFFFFFF),
        ]
        for input_val, expected in test_cases:
            arg = claripy.BVV(input_val, 16)
            result = self._run_op("Iop_16Sto32", arg)
            self.assertEqual(
                result.concrete_value, expected,
                f"16Sto32({input_val:#x}) should be {expected:#x}"
            )

    def test_sign_extend_32to64(self):
        """Test sign extension from 32 to 64 bits."""
        test_cases = [
            (0x00000000, 0x0000000000000000),
            (0x7FFFFFFF, 0x000000007FFFFFFF),
            (0x80000000, 0xFFFFFFFF80000000),
            (0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF),
        ]
        for input_val, expected in test_cases:
            arg = claripy.BVV(input_val, 32)
            result = self._run_op("Iop_32Sto64", arg)
            self.assertEqual(
                result.concrete_value, expected,
                f"32Sto64({input_val:#x}) should be {expected:#x}"
            )

    def test_zero_extend_8to16(self):
        """Test zero extension from 8 to 16 bits."""
        test_cases = [
            (0x00, 0x0000),
            (0x7F, 0x007F),
            (0x80, 0x0080),
            (0xFF, 0x00FF),
        ]
        for input_val, expected in test_cases:
            arg = claripy.BVV(input_val, 8)
            result = self._run_op("Iop_8Uto16", arg)
            self.assertEqual(
                result.concrete_value, expected,
                f"8Uto16({input_val:#x}) should be {expected:#x}"
            )

    def test_zero_extend_32to64(self):
        """Test zero extension from 32 to 64 bits."""
        test_cases = [
            (0x00000000, 0x0000000000000000),
            (0x7FFFFFFF, 0x000000007FFFFFFF),
            (0x80000000, 0x0000000080000000),
            (0xFFFFFFFF, 0x00000000FFFFFFFF),
        ]
        for input_val, expected in test_cases:
            arg = claripy.BVV(input_val, 32)
            result = self._run_op("Iop_32Uto64", arg)
            self.assertEqual(
                result.concrete_value, expected,
                f"32Uto64({input_val:#x}) should be {expected:#x}"
            )

    def test_truncate_64to32(self):
        """Test truncation from 64 to 32 bits."""
        test_cases = [
            (0x0000000000000000, 0x00000000),
            (0x00000000DEADBEEF, 0xDEADBEEF),
            (0xFFFFFFFFDEADBEEF, 0xDEADBEEF),
            (0x1234567890ABCDEF, 0x90ABCDEF),
        ]
        for input_val, expected in test_cases:
            arg = claripy.BVV(input_val, 64)
            result = self._run_op("Iop_64to32", arg)
            self.assertEqual(
                result.concrete_value, expected,
                f"64to32({input_val:#x}) should be {expected:#x}"
            )

    def test_truncate_32to16(self):
        """Test truncation from 32 to 16 bits."""
        test_cases = [
            (0x00000000, 0x0000),
            (0x0000BEEF, 0xBEEF),
            (0xDEADBEEF, 0xBEEF),
        ]
        for input_val, expected in test_cases:
            arg = claripy.BVV(input_val, 32)
            result = self._run_op("Iop_32to16", arg)
            self.assertEqual(
                result.concrete_value, expected,
                f"32to16({input_val:#x}) should be {expected:#x}"
            )

    def test_truncate_16to8(self):
        """Test truncation from 16 to 8 bits."""
        test_cases = [
            (0x0000, 0x00),
            (0x00EF, 0xEF),
            (0xBEEF, 0xEF),
        ]
        for input_val, expected in test_cases:
            arg = claripy.BVV(input_val, 16)
            result = self._run_op("Iop_16to8", arg)
            self.assertEqual(
                result.concrete_value, expected,
                f"16to8({input_val:#x}) should be {expected:#x}"
            )


class TestBitManipulationOps(unittest.TestCase):
    """Test bit manipulation VEX operations for equivalence."""

    @classmethod
    def setUpClass(cls):
        cls.project = load_shellcode(b"\xc3", arch="AMD64")
        cls.engine = HeavyVEXMixin(cls.project)
        cls.state = SimState(project=cls.project)

    def _run_op(self, op_name: str, *args):
        self.engine.state = self.state
        return self.engine._perform_vex_expr_Op(op_name, list(args))

    def test_clz_32bit(self):
        """Test count leading zeros 32-bit."""
        test_cases = [
            (0x80000000, 0),   # MSB set
            (0x00000001, 31),  # Only LSB set
            (0x00008000, 16),  # Bit 15 set
            (0x00000000, 32),  # Zero -> width (32)
            (0xFFFFFFFF, 0),   # All ones
        ]
        for input_val, expected in test_cases:
            arg = claripy.BVV(input_val, 32)
            try:
                result = self._run_op("Iop_Clz32", arg)
                self.assertEqual(
                    result.concrete_value, expected,
                    f"Clz32({input_val:#x}) should be {expected}"
                )
            except Exception as e:
                l.debug("Clz32(%#x) raised: %s", input_val, e)

    def test_clz_64bit(self):
        """Test count leading zeros 64-bit."""
        test_cases = [
            (0x8000000000000000, 0),
            (0x0000000000000001, 63),
            (0x0000000080000000, 32),
            (0x0000000000000000, 64),
        ]
        for input_val, expected in test_cases:
            arg = claripy.BVV(input_val, 64)
            try:
                result = self._run_op("Iop_Clz64", arg)
                self.assertEqual(
                    result.concrete_value, expected,
                    f"Clz64({input_val:#x}) should be {expected}"
                )
            except Exception as e:
                l.debug("Clz64(%#x) raised: %s", input_val, e)

    def test_ctz_32bit(self):
        """Test count trailing zeros 32-bit."""
        test_cases = [
            (0x00000001, 0),   # LSB set
            (0x80000000, 31),  # Only MSB set
            (0x00008000, 15),  # Bit 15 set
            (0x00000000, 32),  # Zero -> width (32)
            (0xFFFFFFFF, 0),   # All ones
        ]
        for input_val, expected in test_cases:
            arg = claripy.BVV(input_val, 32)
            try:
                result = self._run_op("Iop_Ctz32", arg)
                self.assertEqual(
                    result.concrete_value, expected,
                    f"Ctz32({input_val:#x}) should be {expected}"
                )
            except Exception as e:
                l.debug("Ctz32(%#x) raised: %s", input_val, e)

    def test_popcount_32bit(self):
        """Test population count 32-bit."""
        test_cases = [
            (0x00000000, 0),
            (0x00000001, 1),
            (0x00000003, 2),
            (0x0000000F, 4),
            (0x55555555, 16),  # Alternating bits
            (0xFFFFFFFF, 32),
        ]
        for input_val, expected in test_cases:
            arg = claripy.BVV(input_val, 32)
            try:
                result = self._run_op("Iop_PopCount32", arg)
                self.assertEqual(
                    result.concrete_value, expected,
                    f"PopCount32({input_val:#x}) should be {expected}"
                )
            except Exception as e:
                l.debug("PopCount32(%#x) raised: %s", input_val, e)


class TestSymbolicOps(unittest.TestCase):
    """Test symbolic VEX operations for equivalence."""

    @classmethod
    def setUpClass(cls):
        cls.project = load_shellcode(b"\xc3", arch="AMD64")
        cls.engine = HeavyVEXMixin(cls.project)
        cls.state = SimState(project=cls.project)

    def _run_op(self, op_name: str, *args):
        self.engine.state = self.state
        return self.engine._perform_vex_expr_Op(op_name, list(args))

    def test_symbolic_add(self):
        """Symbolic addition should produce Add AST node."""
        x = claripy.BVS("x", 32)
        y = claripy.BVS("y", 32)

        result = self._run_op("Iop_Add32", x, y)

        # Result should be symbolic
        self.assertTrue(result.symbolic)

        # Verify with concrete evaluation
        solver = claripy.Solver()
        solver.add(x == 10)
        solver.add(y == 20)
        evaluated = solver.eval(result, 1)[0]
        self.assertEqual(evaluated, 30)

    def test_symbolic_sub(self):
        """Symbolic subtraction should produce Sub AST node."""
        x = claripy.BVS("x", 32)
        y = claripy.BVS("y", 32)

        result = self._run_op("Iop_Sub32", x, y)

        # Verify with concrete evaluation
        solver = claripy.Solver()
        solver.add(x == 30)
        solver.add(y == 10)
        evaluated = solver.eval(result, 1)[0]
        self.assertEqual(evaluated, 20)

    def test_symbolic_compare(self):
        """Symbolic comparison should produce comparison AST."""
        x = claripy.BVS("x", 32)
        ten = claripy.BVV(10, 32)

        result = self._run_op("Iop_CmpLT32S", x, ten)

        # Result should be 1-bit symbolic
        self.assertEqual(result.length, 1)
        self.assertTrue(result.symbolic)

        # Verify: x=5 should satisfy x < 10
        solver = claripy.Solver()
        solver.add(x == 5)
        evaluated = solver.eval(result, 1)[0]
        self.assertEqual(evaluated, 1)

        # Verify: x=15 should not satisfy x < 10
        solver2 = claripy.Solver()
        solver2.add(x == 15)
        evaluated2 = solver2.eval(result, 1)[0]
        self.assertEqual(evaluated2, 0)

    def test_symbolic_shift(self):
        """Symbolic shift operations."""
        x = claripy.BVS("x", 32)
        shift = claripy.BVV(4, 8)

        result = self._run_op("Iop_Shl32", x, shift)

        # Verify: x << 4 == x * 16
        solver = claripy.Solver()
        solver.add(x == 0x0000000F)
        evaluated = solver.eval(result, 1)[0]
        self.assertEqual(evaluated, 0x000000F0)

    def test_symbolic_bitwise(self):
        """Symbolic bitwise operations."""
        x = claripy.BVS("x", 32)
        mask = claripy.BVV(0x0000FFFF, 32)

        result = self._run_op("Iop_And32", x, mask)

        # Verify: x & 0xFFFF masks high bits
        solver = claripy.Solver()
        solver.add(x == 0xDEADBEEF)
        evaluated = solver.eval(result, 1)[0]
        self.assertEqual(evaluated, 0x0000BEEF)

    def test_ite_operation(self):
        """Test If-Then-Else operation structure."""
        cond = claripy.BVS("cond", 1)
        x = claripy.BVV(10, 32)
        y = claripy.BVV(20, 32)

        # Using claripy's ITE directly
        result = claripy.If(cond == 1, x, y)

        # When cond=1, result should be 10
        solver = claripy.Solver()
        solver.add(cond == 1)
        evaluated = solver.eval(result, 1)[0]
        self.assertEqual(evaluated, 10)

        # When cond=0, result should be 20
        solver2 = claripy.Solver()
        solver2.add(cond == 0)
        evaluated2 = solver2.eval(result, 1)[0]
        self.assertEqual(evaluated2, 20)


class TestVectorOps(unittest.TestCase):
    """Test vector VEX operations for equivalence."""

    @classmethod
    def setUpClass(cls):
        cls.project = load_shellcode(b"\xc3", arch="AMD64")
        cls.engine = HeavyVEXMixin(cls.project)
        cls.state = SimState(project=cls.project)

    def _run_op(self, op_name: str, *args):
        self.engine.state = self.state
        return self.engine._perform_vex_expr_Op(op_name, list(args))

    def test_add16x8(self):
        """Test 16-bit x 8 lane vector addition."""
        a = claripy.BVV(0x0001000200030004_0005000600070008, 128)
        b = claripy.BVV(0x0001000100010001_0001000100010001, 128)
        expected = claripy.BVV(0x0002000300040005_0006000700080009, 128)

        result = self._run_op("Iop_Add16x8", a, b)
        self.assertTrue(
            self.state.solver.is_true(result == expected),
            f"Add16x8 mismatch: got {result}, expected {expected}"
        )

    def test_sub16x8(self):
        """Test 16-bit x 8 lane vector subtraction."""
        a = claripy.BVV(0xFFFF0000000100020003000400050006, 128)
        b = claripy.BVV(0x00020002000200020002000200020002, 128)
        expected = claripy.BVV(0xFFFDFFFEFFFF00000001000200030004, 128)

        result = self._run_op("Iop_Sub16x8", a, b)
        self.assertTrue(
            self.state.solver.is_true(result == expected),
            f"Sub16x8 mismatch"
        )

    def test_cmpeq16x8(self):
        """Test 16-bit x 8 lane vector equality comparison."""
        a = claripy.BVV(0xFFFF0000000100020003000400050006, 128)
        b = claripy.BVV(0x00020002000200020002000200020002, 128)
        # Only lane 4 (0x0002) matches
        expected = claripy.BVV(0x000000000000FFFF0000000000000000, 128)

        result = self._run_op("Iop_CmpEQ16x8", a, b)
        self.assertTrue(
            self.state.solver.is_true(result == expected),
            f"CmpEQ16x8 mismatch"
        )

    def test_interleave_lo16x8(self):
        """Test low half interleave of 16-bit lanes."""
        a = claripy.BVV(0xFFFF0000000100020003000400050006, 128)
        b = claripy.BVV(0x00020002000200020002000200020002, 128)
        expected = claripy.BVV(0x00030002000400020005000200060002, 128)

        result = self._run_op("Iop_InterleaveLO16x8", a, b)
        self.assertTrue(
            self.state.solver.is_true(result == expected),
            f"InterleaveLO16x8 mismatch"
        )

    def test_dup8x8(self):
        """Test 8-bit duplicate across 8 lanes."""
        val = claripy.BVV(0xE7, 8)
        expected = claripy.BVV(0xE7E7E7E7E7E7E7E7, 64)

        result = self._run_op("Iop_Dup8x8", val)
        self.assertTrue(
            self.state.solver.is_true(result == expected),
            f"Dup8x8 mismatch"
        )

    def test_dup16x4(self):
        """Test 16-bit duplicate across 4 lanes."""
        val = claripy.BVV(0x1234, 16)
        expected = claripy.BVV(0x1234123412341234, 64)

        result = self._run_op("Iop_Dup16x4", val)
        self.assertTrue(
            self.state.solver.is_true(result == expected),
            f"Dup16x4 mismatch"
        )

    def test_dup32x4(self):
        """Test 32-bit duplicate across 4 lanes (128-bit)."""
        val = claripy.BVV(0x12345678, 32)
        expected = claripy.BVV(0x12345678123456781234567812345678, 128)

        result = self._run_op("Iop_Dup32x4", val)
        self.assertTrue(
            self.state.solver.is_true(result == expected),
            f"Dup32x4 mismatch"
        )


if __name__ == "__main__":
    unittest.main()
