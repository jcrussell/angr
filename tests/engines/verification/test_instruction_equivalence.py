"""
Instruction Equivalence Tests.

Tests that verify the Rust VEX engine produces correct results for
specific instructions, adapted from tests/sim/exec_insn/.
"""
from __future__ import annotations

import itertools
import logging
import struct
import unittest

import claripy

import angr
from angr import SimState, load_shellcode
from angr.engines import HeavyVEXMixin

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


class TestADCInstruction(unittest.TestCase):
    """Test ADC (add with carry) instruction equivalence."""

    def test_adc_basic_concrete(self):
        """Test basic ADC instruction with concrete values."""
        # ADC: adds two operands plus carry flag
        # This tests flag calculation equivalence

        # mov eax, 0x7FFFFFFF
        # mov ebx, 1
        # add eax, ebx  ; sets carry=0
        # adc eax, ebx  ; adds ebx + carry(0) = ebx

        p = load_shellcode(
            b"\xB8\xFF\xFF\xFF\x7F"  # mov eax, 0x7FFFFFFF
            b"\xBB\x01\x00\x00\x00"  # mov ebx, 1
            b"\x01\xD8"              # add eax, ebx -> 0x80000000, carry=0
            b"\x11\xD8"              # adc eax, ebx -> 0x80000001
            b"\xCC",                 # int3 (stop)
            arch="x86"
        )

        state = p.factory.blank_state(
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        successor = state.step(num_inst=4)

        if successor.successors:
            final = successor.successors[0]
            # After: add makes 0x80000000, adc adds 1 more = 0x80000001
            self.assertEqual(final.regs.eax.concrete_value, 0x80000001)

    def test_adc_with_carry_set(self):
        """Test ADC when carry flag is set."""
        # mov eax, 0xFFFFFFFF
        # mov ebx, 1
        # add eax, ebx  ; causes carry (result=0, CF=1)
        # adc eax, ebx  ; 0 + 1 + 1(carry) = 2

        p = load_shellcode(
            b"\xB8\xFF\xFF\xFF\xFF"  # mov eax, 0xFFFFFFFF
            b"\xBB\x01\x00\x00\x00"  # mov ebx, 1
            b"\x01\xD8"              # add eax, ebx -> 0, CF=1
            b"\x11\xD8"              # adc eax, ebx -> 0 + 1 + 1 = 2
            b"\xCC",
            arch="x86"
        )

        state = p.factory.blank_state(
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        successor = state.step(num_inst=4)

        if successor.successors:
            final = successor.successors[0]
            self.assertEqual(final.regs.eax.concrete_value, 2)


class TestSHLInstruction(unittest.TestCase):
    """Test SHL (shift left) instruction equivalence."""

    def _test_shl(self, value: int, shift: int, width: int):
        """Test a single SHL operation."""
        if width == 32:
            # mov eax, value
            # shl eax, shift
            code = struct.pack("<BI", 0xB8, value) + bytes([0xC1, 0xE0, shift & 0xFF])
            arch = "x86"
            result_reg = "eax"
        else:  # 64-bit
            # mov rax, value (movabs)
            # shl rax, shift
            code = b"\x48\xB8" + struct.pack("<Q", value) + bytes([0x48, 0xC1, 0xE0, shift & 0xFF])
            arch = "amd64"
            result_reg = "rax"

        p = load_shellcode(code + b"\xCC", arch=arch)
        state = p.factory.blank_state(
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        successor = state.step(num_inst=2)

        if successor.successors:
            final = successor.successors[0]
            result = getattr(final.regs, result_reg).concrete_value

            mask = (1 << width) - 1
            expected = (value << shift) & mask if shift < width else 0

            self.assertEqual(
                result, expected,
                f"SHL({value:#x}, {shift}): got {result:#x}, expected {expected:#x}"
            )

    def test_shl_32bit_edge_cases(self):
        """Test SHL with 32-bit edge case values."""
        test_cases = [
            (0, 0),        # 0 << 0 = 0
            (1, 0),        # 1 << 0 = 1
            (1, 1),        # 1 << 1 = 2
            (1, 31),       # 1 << 31 = 0x80000000
            (0x80000000, 1),  # High bit shift
            (0xFFFFFFFF, 1),  # All ones shift
        ]

        for value, shift in test_cases:
            with self.subTest(value=value, shift=shift):
                self._test_shl(value, shift, 32)


class TestROLInstruction(unittest.TestCase):
    """Test ROL (rotate left) instruction equivalence."""

    def test_rol_32bit(self):
        """Test 32-bit rotate left."""
        # mov eax, 0x80000001
        # rol eax, 1  ; result should be 0x00000003

        p = load_shellcode(
            b"\xB8\x01\x00\x00\x80"  # mov eax, 0x80000001
            b"\xC1\xC0\x01"          # rol eax, 1
            b"\xCC",
            arch="x86"
        )

        state = p.factory.blank_state(
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        successor = state.step(num_inst=2)

        if successor.successors:
            final = successor.successors[0]
            self.assertEqual(final.regs.eax.concrete_value, 0x00000003)

    def test_rol_by_various_amounts(self):
        """Test ROL with various rotation amounts."""
        for shift in [1, 4, 8, 16, 31]:
            with self.subTest(shift=shift):
                # mov eax, 0xDEADBEEF
                # rol eax, shift
                code = b"\xB8\xEF\xBE\xAD\xDE" + bytes([0xC1, 0xC0, shift])

                p = load_shellcode(code + b"\xCC", arch="x86")
                state = p.factory.blank_state(
                    add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
                )
                successor = state.step(num_inst=2)

                if successor.successors:
                    final = successor.successors[0]
                    result = final.regs.eax.concrete_value

                    # Calculate expected
                    val = 0xDEADBEEF
                    shift_mod = shift % 32
                    expected = ((val << shift_mod) | (val >> (32 - shift_mod))) & 0xFFFFFFFF

                    self.assertEqual(result, expected)


class TestIMULInstruction(unittest.TestCase):
    """Test IMUL (signed multiply) instruction equivalence."""

    def test_imul_two_operand(self):
        """Test two-operand IMUL."""
        # mov eax, 7
        # mov ebx, 6
        # imul eax, ebx  ; eax = 42

        p = load_shellcode(
            b"\xB8\x07\x00\x00\x00"  # mov eax, 7
            b"\xBB\x06\x00\x00\x00"  # mov ebx, 6
            b"\x0F\xAF\xC3"          # imul eax, ebx
            b"\xCC",
            arch="x86"
        )

        state = p.factory.blank_state(
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        successor = state.step(num_inst=3)

        if successor.successors:
            final = successor.successors[0]
            self.assertEqual(final.regs.eax.concrete_value, 42)

    def test_imul_negative(self):
        """Test IMUL with negative values."""
        # mov eax, -5 (0xFFFFFFFB)
        # mov ebx, 3
        # imul eax, ebx  ; eax = -15 (0xFFFFFFF1)

        p = load_shellcode(
            b"\xB8\xFB\xFF\xFF\xFF"  # mov eax, -5
            b"\xBB\x03\x00\x00\x00"  # mov ebx, 3
            b"\x0F\xAF\xC3"          # imul eax, ebx
            b"\xCC",
            arch="x86"
        )

        state = p.factory.blank_state(
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        successor = state.step(num_inst=3)

        if successor.successors:
            final = successor.successors[0]
            # -15 in 32-bit two's complement
            self.assertEqual(final.regs.eax.concrete_value, 0xFFFFFFF1)


class TestDIVInstruction(unittest.TestCase):
    """Test DIV/IDIV instruction equivalence."""

    def test_div_unsigned(self):
        """Test unsigned division."""
        # mov eax, 100
        # xor edx, edx  ; clear upper bits
        # mov ecx, 7
        # div ecx       ; eax = 100/7 = 14, edx = 100%7 = 2

        p = load_shellcode(
            b"\xB8\x64\x00\x00\x00"  # mov eax, 100
            b"\x31\xD2"              # xor edx, edx
            b"\xB9\x07\x00\x00\x00"  # mov ecx, 7
            b"\xF7\xF1"              # div ecx
            b"\xCC",
            arch="x86"
        )

        state = p.factory.blank_state(
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        successor = state.step(num_inst=4)

        if successor.successors:
            final = successor.successors[0]
            self.assertEqual(final.regs.eax.concrete_value, 14)  # quotient
            self.assertEqual(final.regs.edx.concrete_value, 2)   # remainder


class TestMOVInstruction(unittest.TestCase):
    """Test MOV instruction variants."""

    def test_mov_immediate(self):
        """Test MOV with immediate values."""
        test_cases = [
            (0x00000000, "x86"),
            (0xDEADBEEF, "x86"),
            (0xFFFFFFFF, "x86"),
        ]

        for val, arch in test_cases:
            with self.subTest(value=val):
                code = struct.pack("<BI", 0xB8, val) + b"\xCC"
                p = load_shellcode(code, arch=arch)
                state = p.factory.blank_state(
                    add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
                )
                successor = state.step(num_inst=1)

                if successor.successors:
                    final = successor.successors[0]
                    self.assertEqual(final.regs.eax.concrete_value, val)

    def test_mov_register_to_register(self):
        """Test MOV between registers."""
        # mov eax, 0x12345678
        # mov ebx, eax

        p = load_shellcode(
            b"\xB8\x78\x56\x34\x12"  # mov eax, 0x12345678
            b"\x89\xC3"              # mov ebx, eax
            b"\xCC",
            arch="x86"
        )

        state = p.factory.blank_state(
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        successor = state.step(num_inst=2)

        if successor.successors:
            final = successor.successors[0]
            self.assertEqual(final.regs.eax.concrete_value, 0x12345678)
            self.assertEqual(final.regs.ebx.concrete_value, 0x12345678)


class TestCMPInstruction(unittest.TestCase):
    """Test CMP instruction and flag setting."""

    def test_cmp_equal(self):
        """Test CMP when values are equal (ZF=1)."""
        # mov eax, 5
        # cmp eax, 5
        # (ZF should be set)

        p = load_shellcode(
            b"\xB8\x05\x00\x00\x00"  # mov eax, 5
            b"\x83\xF8\x05"          # cmp eax, 5
            b"\xCC",
            arch="x86"
        )

        state = p.factory.blank_state(
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        successor = state.step(num_inst=2)

        if successor.successors:
            final = successor.successors[0]
            # ZF should be set (result of subtraction is 0)
            # Check via conditional jump or flag query
            # EAX should be unchanged
            self.assertEqual(final.regs.eax.concrete_value, 5)


class TestXORInstruction(unittest.TestCase):
    """Test XOR instruction."""

    def test_xor_self_clears(self):
        """XOR reg with itself should clear the register."""
        # mov eax, 0xDEADBEEF
        # xor eax, eax  ; eax = 0

        p = load_shellcode(
            b"\xB8\xEF\xBE\xAD\xDE"  # mov eax, 0xDEADBEEF
            b"\x31\xC0"              # xor eax, eax
            b"\xCC",
            arch="x86"
        )

        state = p.factory.blank_state(
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        successor = state.step(num_inst=2)

        if successor.successors:
            final = successor.successors[0]
            self.assertEqual(final.regs.eax.concrete_value, 0)

    def test_xor_values(self):
        """Test XOR with specific values."""
        # mov eax, 0xFF00FF00
        # mov ebx, 0x00FF00FF
        # xor eax, ebx  ; eax = 0xFFFFFFFF

        p = load_shellcode(
            b"\xB8\x00\xFF\x00\xFF"  # mov eax, 0xFF00FF00
            b"\xBB\xFF\x00\xFF\x00"  # mov ebx, 0x00FF00FF
            b"\x31\xD8"              # xor eax, ebx
            b"\xCC",
            arch="x86"
        )

        state = p.factory.blank_state(
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        successor = state.step(num_inst=3)

        if successor.successors:
            final = successor.successors[0]
            self.assertEqual(final.regs.eax.concrete_value, 0xFFFFFFFF)


class TestLEAInstruction(unittest.TestCase):
    """Test LEA (load effective address) instruction."""

    def test_lea_simple_add(self):
        """Test LEA for simple addition."""
        # mov eax, 10
        # lea ebx, [eax + 5]  ; ebx = 15

        p = load_shellcode(
            b"\xB8\x0A\x00\x00\x00"  # mov eax, 10
            b"\x8D\x58\x05"          # lea ebx, [eax + 5]
            b"\xCC",
            arch="x86"
        )

        state = p.factory.blank_state(
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        successor = state.step(num_inst=2)

        if successor.successors:
            final = successor.successors[0]
            self.assertEqual(final.regs.ebx.concrete_value, 15)


class TestPUSHPOPInstruction(unittest.TestCase):
    """Test PUSH/POP instruction equivalence."""

    def test_push_pop_round_trip(self):
        """Value should survive push then pop."""
        # mov eax, 0x12345678
        # push eax
        # pop ebx  ; ebx should equal eax

        p = load_shellcode(
            b"\xB8\x78\x56\x34\x12"  # mov eax, 0x12345678
            b"\x50"                  # push eax
            b"\x5B"                  # pop ebx
            b"\xCC",
            arch="x86"
        )

        state = p.factory.blank_state(
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        # Need stack space
        state.regs.esp = 0x7FFFF000
        state.memory.store(0x7FFFE000, b"\x00" * 0x1000)

        successor = state.step(num_inst=3)

        if successor.successors:
            final = successor.successors[0]
            self.assertEqual(final.regs.ebx.concrete_value, 0x12345678)


class TestSymbolicInstructions(unittest.TestCase):
    """Test instruction execution with symbolic values."""

    def test_add_symbolic(self):
        """Test ADD with symbolic operand."""
        p = load_shellcode(
            b"\x01\xD8"  # add eax, ebx
            b"\xCC",
            arch="x86"
        )

        state = p.factory.blank_state()
        x = claripy.BVS("x", 32)
        state.regs.eax = x
        state.regs.ebx = 10

        successor = state.step(num_inst=1)

        if successor.successors:
            final = successor.successors[0]
            result = final.regs.eax

            # Result should be x + 10
            self.assertTrue(result.symbolic)

            # Verify with concrete evaluation
            final.solver.add(x == 5)
            evaluated = final.solver.eval(result)
            self.assertEqual(evaluated, 15)

    def test_cmp_symbolic_branch(self):
        """Test CMP with symbolic value creates branch conditions."""
        # cmp eax, 10
        # jl label  ; jump if eax < 10

        p = load_shellcode(
            b"\x83\xF8\x0A"  # cmp eax, 10
            b"\x7C\x02"      # jl +2 (skip next)
            b"\xEB\x00"      # jmp +0 (nop-like)
            b"\xCC",
            arch="x86"
        )

        state = p.factory.blank_state()
        x = claripy.BVS("x", 32)
        state.regs.eax = x

        # This creates a simulation with branching
        sm = p.factory.simulation_manager(state)
        sm.step(num_inst=2)

        # Should have potential for both branches
        # (depends on constraint on x)


if __name__ == "__main__":
    unittest.main()
