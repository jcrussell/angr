"""
Minimal differential tests for Rust vs Python VEX engines.

These are the simplest possible tests to verify the differential testing
framework works correctly. Start here before running larger test suites.

IMPORTANT: The Rust VEX engine currently only supports block structure (IMark,
exit) but not actual VEX operations. The `step()` method executes the block
structure but doesn't execute the actual instructions. This means differential
testing at the instruction level is not yet possible.

Current Status:
- Block navigation works (PC advances correctly)
- Register read/write works
- Memory read/write works
- Actual instruction execution: NOT YET IMPLEMENTED

Tests in this file are marked as xfail until the Rust engine supports
actual VEX operation execution.
"""
from __future__ import annotations

import pytest

# Import Rust engine availability
try:
    from angr.engines.rust_vex import RUST_ENGINE_AVAILABLE, RustVEXEngineWrapper
except ImportError:
    RUST_ENGINE_AVAILABLE = False
    RustVEXEngineWrapper = None

import angr

pytestmark = pytest.mark.rust_engine

# Mark for tests that require full VEX execution (not yet implemented)
rust_vex_execution = pytest.mark.xfail(
    reason="Rust VEX engine does not yet execute actual VEX operations",
    strict=False  # Don't fail if it unexpectedly passes (future fix)
)


@pytest.fixture
def skip_if_no_rust():
    """Skip test if Rust engine is not available."""
    if not RUST_ENGINE_AVAILABLE:
        pytest.skip("Rust VEX engine not available")


@rust_vex_execution
class TestMinimalDifferential:
    """Minimal differential tests - one for each basic operation type."""

    def test_add_differential(self, skip_if_no_rust):
        """Simplest differential test: ADD instruction."""
        # x86: add eax, ebx (0x01 0xD8)
        code = bytes([0x01, 0xD8])
        code_base = 0x1000

        # Rust execution
        rust = RustVEXEngineWrapper("x86")
        rust.map_memory_data(code_base, code)
        rust.set_register("eax", 5)
        rust.set_register("ebx", 3)
        rust.pc = code_base
        rust.step()
        rust_eax = rust.get_register("eax")

        # Python execution
        proj = angr.load_shellcode(code, arch="X86", load_address=code_base)
        state = proj.factory.blank_state(
            addr=code_base,
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        state.regs.eax = 5
        state.regs.ebx = 3
        succ = state.step(num_inst=1)
        python_state = succ.successors[0]
        python_eax = python_state.solver.eval(python_state.regs.eax)

        assert rust_eax == python_eax == 8, f"ADD: Rust={rust_eax}, Python={python_eax}, expected=8"

    def test_sub_differential(self, skip_if_no_rust):
        """Differential test: SUB instruction."""
        # x86: sub eax, ebx (0x29 0xD8)
        code = bytes([0x29, 0xD8])
        code_base = 0x1000

        # Rust execution
        rust = RustVEXEngineWrapper("x86")
        rust.map_memory_data(code_base, code)
        rust.set_register("eax", 10)
        rust.set_register("ebx", 3)
        rust.pc = code_base
        rust.step()
        rust_eax = rust.get_register("eax")

        # Python execution
        proj = angr.load_shellcode(code, arch="X86", load_address=code_base)
        state = proj.factory.blank_state(
            addr=code_base,
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        state.regs.eax = 10
        state.regs.ebx = 3
        succ = state.step(num_inst=1)
        python_state = succ.successors[0]
        python_eax = python_state.solver.eval(python_state.regs.eax)

        assert rust_eax == python_eax == 7, f"SUB: Rust={rust_eax}, Python={python_eax}, expected=7"

    def test_xor_differential(self, skip_if_no_rust):
        """Differential test: XOR instruction."""
        # x86: xor eax, ebx (0x31 0xD8)
        code = bytes([0x31, 0xD8])
        code_base = 0x1000

        # Rust execution
        rust = RustVEXEngineWrapper("x86")
        rust.map_memory_data(code_base, code)
        rust.set_register("eax", 0xFF00FF00)
        rust.set_register("ebx", 0x0F0F0F0F)
        rust.pc = code_base
        rust.step()
        rust_eax = rust.get_register("eax")

        # Python execution
        proj = angr.load_shellcode(code, arch="X86", load_address=code_base)
        state = proj.factory.blank_state(
            addr=code_base,
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        state.regs.eax = 0xFF00FF00
        state.regs.ebx = 0x0F0F0F0F
        succ = state.step(num_inst=1)
        python_state = succ.successors[0]
        python_eax = python_state.solver.eval(python_state.regs.eax)

        expected = 0xFF00FF00 ^ 0x0F0F0F0F
        assert rust_eax == python_eax == expected, f"XOR: Rust={rust_eax:#x}, Python={python_eax:#x}, expected={expected:#x}"

    def test_mul_differential(self, skip_if_no_rust):
        """Differential test: MUL instruction (unsigned multiply)."""
        # x86: mul ebx (0xF7 0xE3)
        code = bytes([0xF7, 0xE3])
        code_base = 0x1000

        # Rust execution
        rust = RustVEXEngineWrapper("x86")
        rust.map_memory_data(code_base, code)
        rust.set_register("eax", 100)
        rust.set_register("ebx", 200)
        rust.set_register("edx", 0)
        rust.pc = code_base
        rust.step()
        rust_eax = rust.get_register("eax")
        rust_edx = rust.get_register("edx")

        # Python execution
        proj = angr.load_shellcode(code, arch="X86", load_address=code_base)
        state = proj.factory.blank_state(
            addr=code_base,
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        state.regs.eax = 100
        state.regs.ebx = 200
        state.regs.edx = 0
        succ = state.step(num_inst=1)
        python_state = succ.successors[0]
        python_eax = python_state.solver.eval(python_state.regs.eax)
        python_edx = python_state.solver.eval(python_state.regs.edx)

        expected = 100 * 200  # 20000, fits in eax
        assert rust_eax == python_eax == expected, f"MUL eax: Rust={rust_eax}, Python={python_eax}, expected={expected}"
        assert rust_edx == python_edx == 0, f"MUL edx: Rust={rust_edx}, Python={python_edx}, expected=0"

    def test_mov_differential(self, skip_if_no_rust):
        """Differential test: MOV register-to-register."""
        # x86: mov eax, ebx (0x89 0xD8)
        code = bytes([0x89, 0xD8])
        code_base = 0x1000

        # Rust execution
        rust = RustVEXEngineWrapper("x86")
        rust.map_memory_data(code_base, code)
        rust.set_register("eax", 0)
        rust.set_register("ebx", 0xDEADBEEF)
        rust.pc = code_base
        rust.step()
        rust_eax = rust.get_register("eax")

        # Python execution
        proj = angr.load_shellcode(code, arch="X86", load_address=code_base)
        state = proj.factory.blank_state(
            addr=code_base,
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        state.regs.eax = 0
        state.regs.ebx = 0xDEADBEEF
        succ = state.step(num_inst=1)
        python_state = succ.successors[0]
        python_eax = python_state.solver.eval(python_state.regs.eax)

        assert rust_eax == python_eax == 0xDEADBEEF, f"MOV: Rust={rust_eax:#x}, Python={python_eax:#x}"

    def test_shl_differential(self, skip_if_no_rust):
        """Differential test: SHL (shift left) instruction."""
        # x86: shl eax, cl (0xD3 0xE0)
        code = bytes([0xD3, 0xE0])
        code_base = 0x1000

        # Rust execution
        rust = RustVEXEngineWrapper("x86")
        rust.map_memory_data(code_base, code)
        rust.set_register("eax", 1)
        rust.set_register("ecx", 4)
        rust.pc = code_base
        rust.step()
        rust_eax = rust.get_register("eax")

        # Python execution
        proj = angr.load_shellcode(code, arch="X86", load_address=code_base)
        state = proj.factory.blank_state(
            addr=code_base,
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        state.regs.eax = 1
        state.regs.ecx = 4
        succ = state.step(num_inst=1)
        python_state = succ.successors[0]
        python_eax = python_state.solver.eval(python_state.regs.eax)

        expected = 1 << 4  # 16
        assert rust_eax == python_eax == expected, f"SHL: Rust={rust_eax}, Python={python_eax}, expected={expected}"


@rust_vex_execution
class TestMinimalEdgeCases:
    """Minimal tests for edge cases."""

    def test_add_overflow(self, skip_if_no_rust):
        """Test ADD with overflow."""
        code = bytes([0x01, 0xD8])  # add eax, ebx
        code_base = 0x1000

        # Rust execution
        rust = RustVEXEngineWrapper("x86")
        rust.map_memory_data(code_base, code)
        rust.set_register("eax", 0xFFFFFFFF)
        rust.set_register("ebx", 1)
        rust.pc = code_base
        rust.step()
        rust_eax = rust.get_register("eax")

        # Python execution
        proj = angr.load_shellcode(code, arch="X86", load_address=code_base)
        state = proj.factory.blank_state(
            addr=code_base,
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        state.regs.eax = 0xFFFFFFFF
        state.regs.ebx = 1
        succ = state.step(num_inst=1)
        python_state = succ.successors[0]
        python_eax = python_state.solver.eval(python_state.regs.eax)

        expected = 0  # Wraps around
        assert rust_eax == python_eax == expected, f"ADD overflow: Rust={rust_eax}, Python={python_eax}"

    def test_sub_underflow(self, skip_if_no_rust):
        """Test SUB with underflow."""
        code = bytes([0x29, 0xD8])  # sub eax, ebx
        code_base = 0x1000

        # Rust execution
        rust = RustVEXEngineWrapper("x86")
        rust.map_memory_data(code_base, code)
        rust.set_register("eax", 0)
        rust.set_register("ebx", 1)
        rust.pc = code_base
        rust.step()
        rust_eax = rust.get_register("eax")

        # Python execution
        proj = angr.load_shellcode(code, arch="X86", load_address=code_base)
        state = proj.factory.blank_state(
            addr=code_base,
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        state.regs.eax = 0
        state.regs.ebx = 1
        succ = state.step(num_inst=1)
        python_state = succ.successors[0]
        python_eax = python_state.solver.eval(python_state.regs.eax)

        expected = 0xFFFFFFFF  # Wraps around
        assert rust_eax == python_eax == expected, f"SUB underflow: Rust={rust_eax:#x}, Python={python_eax:#x}"

    def test_mul_large_values(self, skip_if_no_rust):
        """Test MUL with large values that overflow into EDX."""
        code = bytes([0xF7, 0xE3])  # mul ebx
        code_base = 0x1000

        # Rust execution
        rust = RustVEXEngineWrapper("x86")
        rust.map_memory_data(code_base, code)
        rust.set_register("eax", 0x10000000)
        rust.set_register("ebx", 0x10000000)
        rust.set_register("edx", 0)
        rust.pc = code_base
        rust.step()
        rust_eax = rust.get_register("eax")
        rust_edx = rust.get_register("edx")

        # Python execution
        proj = angr.load_shellcode(code, arch="X86", load_address=code_base)
        state = proj.factory.blank_state(
            addr=code_base,
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        state.regs.eax = 0x10000000
        state.regs.ebx = 0x10000000
        state.regs.edx = 0
        succ = state.step(num_inst=1)
        python_state = succ.successors[0]
        python_eax = python_state.solver.eval(python_state.regs.eax)
        python_edx = python_state.solver.eval(python_state.regs.edx)

        # 0x10000000 * 0x10000000 = 0x100000000000000
        # Low 32 bits = 0x00000000, high 32 bits = 0x01000000
        assert rust_eax == python_eax == 0, f"MUL large eax: Rust={rust_eax:#x}, Python={python_eax:#x}"
        assert rust_edx == python_edx == 0x01000000, f"MUL large edx: Rust={rust_edx:#x}, Python={python_edx:#x}"


@rust_vex_execution
class TestMinimalAMD64:
    """Minimal differential tests for AMD64 architecture."""

    def test_add_64bit(self, skip_if_no_rust):
        """Test 64-bit ADD instruction."""
        # x86-64: add rax, rbx (0x48 0x01 0xD8)
        code = bytes([0x48, 0x01, 0xD8])
        code_base = 0x1000

        # Rust execution
        rust = RustVEXEngineWrapper("amd64")
        rust.map_memory_data(code_base, code)
        rust.set_register("rax", 0x100000000)
        rust.set_register("rbx", 0x200000000)
        rust.pc = code_base
        rust.step()
        rust_rax = rust.get_register("rax")

        # Python execution
        proj = angr.load_shellcode(code, arch="AMD64", load_address=code_base)
        state = proj.factory.blank_state(
            addr=code_base,
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        state.regs.rax = 0x100000000
        state.regs.rbx = 0x200000000
        succ = state.step(num_inst=1)
        python_state = succ.successors[0]
        python_rax = python_state.solver.eval(python_state.regs.rax)

        expected = 0x300000000
        assert rust_rax == python_rax == expected, f"ADD 64-bit: Rust={rust_rax:#x}, Python={python_rax:#x}"

    def test_xor_64bit(self, skip_if_no_rust):
        """Test 64-bit XOR instruction."""
        # x86-64: xor rax, rbx (0x48 0x31 0xD8)
        code = bytes([0x48, 0x31, 0xD8])
        code_base = 0x1000

        # Rust execution
        rust = RustVEXEngineWrapper("amd64")
        rust.map_memory_data(code_base, code)
        rust.set_register("rax", 0xFFFFFFFFFFFFFFFF)
        rust.set_register("rbx", 0x0F0F0F0F0F0F0F0F)
        rust.pc = code_base
        rust.step()
        rust_rax = rust.get_register("rax")

        # Python execution
        proj = angr.load_shellcode(code, arch="AMD64", load_address=code_base)
        state = proj.factory.blank_state(
            addr=code_base,
            add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
        )
        state.regs.rax = 0xFFFFFFFFFFFFFFFF
        state.regs.rbx = 0x0F0F0F0F0F0F0F0F
        succ = state.step(num_inst=1)
        python_state = succ.successors[0]
        python_rax = python_state.solver.eval(python_state.regs.rax)

        expected = 0xFFFFFFFFFFFFFFFF ^ 0x0F0F0F0F0F0F0F0F
        assert rust_rax == python_rax == expected, f"XOR 64-bit: Rust={rust_rax:#x}, Python={python_rax:#x}"


class TestRustEngineBasicFunctionality:
    """Tests for Rust engine functionality that currently works.

    These tests validate register read/write, memory read/write, and
    basic block structure - features that work independently of VEX
    operation execution.
    """

    def test_register_read_write_x86(self, skip_if_no_rust):
        """Test x86 register read/write."""
        rust = RustVEXEngineWrapper("x86")

        test_values = [
            ("eax", 0x12345678),
            ("ebx", 0xDEADBEEF),
            ("ecx", 0),
            ("edx", 0xFFFFFFFF),
        ]

        for reg, val in test_values:
            rust.set_register(reg, val)
            result = rust.get_register(reg)
            assert result == val, f"{reg}: expected {val:#x}, got {result:#x}"

    def test_register_read_write_amd64(self, skip_if_no_rust):
        """Test AMD64 register read/write."""
        rust = RustVEXEngineWrapper("amd64")

        test_values = [
            ("rax", 0x123456789ABCDEF0),
            ("rbx", 0xDEADBEEFCAFEBABE),
            ("rcx", 0),
            ("rdx", 0xFFFFFFFFFFFFFFFF),
        ]

        for reg, val in test_values:
            rust.set_register(reg, val)
            result = rust.get_register(reg)
            assert result == val, f"{reg}: expected {val:#x}, got {result:#x}"

    def test_memory_read_write(self, skip_if_no_rust):
        """Test memory read/write."""
        rust = RustVEXEngineWrapper("x86")

        # Map memory
        rust.map_memory(0x1000, 0x1000)

        # Write and read back
        test_data = b"\xDE\xAD\xBE\xEF\xCA\xFE\xBA\xBE"
        rust.write_memory(0x1000, test_data)
        result = rust.read_memory(0x1000, len(test_data))

        assert result == test_data, f"Memory mismatch: expected {test_data.hex()}, got {result.hex()}"

    def test_memory_map_with_data(self, skip_if_no_rust):
        """Test map_memory_data."""
        rust = RustVEXEngineWrapper("x86")

        test_data = bytes([0x01, 0xD8, 0xCC])  # add eax, ebx; int3
        rust.map_memory_data(0x1000, test_data)

        result = rust.read_memory(0x1000, len(test_data))
        assert result == test_data, f"Memory mismatch: expected {test_data.hex()}, got {result.hex()}"

    def test_pc_read_write(self, skip_if_no_rust):
        """Test PC register read/write."""
        rust = RustVEXEngineWrapper("x86")

        rust.pc = 0x1000
        assert rust.pc == 0x1000

        rust.pc = 0xDEADBEEF
        assert rust.pc == 0xDEADBEEF

    def test_fork_state(self, skip_if_no_rust):
        """Test state forking."""
        rust = RustVEXEngineWrapper("x86")

        rust.set_register("eax", 100)
        rust.pc = 0x1000

        # Fork
        rust2 = rust.fork()

        # Modify original
        rust.set_register("eax", 200)
        rust.pc = 0x2000

        # Forked state should be independent
        assert rust2.get_register("eax") == 100, "Forked state should be independent"
        assert rust2.pc == 0x1000, "Forked PC should be independent"
        assert rust.get_register("eax") == 200, "Original should have new value"

    def test_get_registers(self, skip_if_no_rust):
        """Test getting all registers."""
        rust = RustVEXEngineWrapper("x86")

        rust.set_register("eax", 1)
        rust.set_register("ebx", 2)
        rust.set_register("ecx", 3)

        regs = rust.get_registers()

        assert isinstance(regs, dict)
        assert regs.get("eax") == 1
        assert regs.get("ebx") == 2
        assert regs.get("ecx") == 3


class TestRustEngineBlockStructure:
    """Tests for Rust engine block structure handling.

    The Rust engine can build and navigate block structures even though
    it doesn't execute VEX operations yet.
    """

    def test_block_creation(self, skip_if_no_rust):
        """Test that blocks can be created."""
        from angr.rustylib.vex_engine import RustVEXEngine

        rust = RustVEXEngine("x86")

        rust.start_block(0x1000)
        rust.add_imark(0x1000, 2, 0)
        rust.set_block_exit(0x1000, 0x1002, "Ijk_Boring")

        assert rust.has_block(0x1000)
        assert rust.cached_block_count() == 1

    def test_step_advances_pc(self, skip_if_no_rust):
        """Test that step() advances PC through block structure."""
        from angr.rustylib.vex_engine import RustVEXEngine

        rust = RustVEXEngine("x86")

        # Build a simple block
        rust.start_block(0x1000)
        rust.add_imark(0x1000, 2, 0)
        rust.set_block_exit(0x1000, 0x1002, "Ijk_Boring")

        # Map memory (needed even for empty block)
        rust.map_memory_data(0x1000, b"\x90\x90")  # two NOPs

        rust.pc = 0x1000
        event = rust.step()

        assert event.event_type == "block_end"
        assert event.next_addr == 0x1002
        assert rust.pc == 0x1002
