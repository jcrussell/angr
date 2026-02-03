"""
Core differential test harness for comparing Rust and Python VEX engines.

Provides TestCase dataclass and DifferentialHarness for running shellcode
on both engines and comparing final states.

The harness uses execute_code() to lift shellcode with pyvex and execute it
in the Rust VEX interpreter.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import angr

l = logging.getLogger(__name__)

# Check if Rust engine is available
try:
    from angr.engines.rust_vex import (
        RustVEXEngineWrapper,
        RUST_ENGINE_AVAILABLE,
    )
except ImportError:
    RUST_ENGINE_AVAILABLE = False
    RustVEXEngineWrapper = None


@dataclass
class DifferentialTestCase:
    """
    A single differential test case.

    Attributes:
        name: Descriptive name for the test
        shellcode: Raw instruction bytes to execute
        initial_regs: Register values to set before execution
        compare_regs: List of register names to compare after execution
        seed: Deterministic seed for reproducibility
        initial_mem: Optional memory contents (addr -> bytes)
        arch: Target architecture (default: x86)
    """
    name: str
    shellcode: bytes
    initial_regs: dict[str, int]
    compare_regs: list[str]
    seed: int
    initial_mem: dict[int, bytes] = field(default_factory=dict)
    arch: str = "x86"

    @property
    def shellcode_hex(self) -> str:
        """Return shellcode as hex string."""
        return self.shellcode.hex()

    def __repr__(self) -> str:
        return f"TestCase(name={self.name!r}, seed={self.seed}, shellcode={self.shellcode_hex})"


@dataclass
class ExecutionResult:
    """Result of executing a test case on one engine."""
    registers: dict[str, int]
    memory: dict[int, bytes] = field(default_factory=dict)
    error: str | None = None
    pc: int = 0

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class ComparisonResult:
    """Result of comparing execution between engines."""
    test: DifferentialTestCase
    rust_result: ExecutionResult
    python_result: ExecutionResult
    register_diffs: dict[str, dict[str, int]]  # {reg: {"python": val, "rust": val}}
    memory_diffs: dict[int, dict[str, bytes]]  # {addr: {"python": bytes, "rust": bytes}}
    match: bool

    def __repr__(self) -> str:
        status = "PASS" if self.match else "FAIL"
        if not self.match:
            diffs = ", ".join(f"{k}:P={v['python']}!=R={v['rust']}"
                              for k, v in self.register_diffs.items())
            return f"{status}: {self.test.name} ({self.test.seed}) - {diffs}"
        return f"{status}: {self.test.name} ({self.test.seed})"


class DifferentialHarness:
    """
    Harness for running differential tests between Rust and Python VEX engines.

    The Rust engine only exposes block-level execution via step(), so we use
    shellcode to exercise specific operations and compare final states.
    """

    ARCH_MAPPING = {
        "x86": "X86",
        "amd64": "AMD64",
        "arm": "ARMEL",
        "arm64": "AARCH64",
        "mips32": "MIPS32",
    }

    # Default code base address for shellcode
    CODE_BASE = 0x1000

    # Register widths by name prefix/suffix (in bits)
    REG_WIDTHS = {
        # x86/amd64 GPRs
        "al": 8, "ah": 8, "bl": 8, "bh": 8, "cl": 8, "ch": 8, "dl": 8, "dh": 8,
        "ax": 16, "bx": 16, "cx": 16, "dx": 16, "si": 16, "di": 16, "bp": 16, "sp": 16,
        "eax": 32, "ebx": 32, "ecx": 32, "edx": 32, "esi": 32, "edi": 32, "ebp": 32, "esp": 32,
        "rax": 64, "rbx": 64, "rcx": 64, "rdx": 64, "rsi": 64, "rdi": 64, "rbp": 64, "rsp": 64,
        "r8": 64, "r9": 64, "r10": 64, "r11": 64, "r12": 64, "r13": 64, "r14": 64, "r15": 64,
        "r8d": 32, "r9d": 32, "r10d": 32, "r11d": 32, "r12d": 32, "r13d": 32, "r14d": 32, "r15d": 32,
        # XMM registers (128-bit but typically compared as 32/64 low bits)
        "xmm0": 128, "xmm1": 128, "xmm2": 128, "xmm3": 128,
        "xmm4": 128, "xmm5": 128, "xmm6": 128, "xmm7": 128,
    }

    @staticmethod
    def _to_unsigned(value: int, width: int) -> int:
        """Convert signed value to unsigned representation for given bit width."""
        if value >= 0:
            return value
        # Convert negative to 2's complement unsigned
        return value + (1 << width)

    def _reg_width(self, reg: str) -> int:
        """Get the bit width of a register."""
        reg_lower = reg.lower()
        if reg_lower in self.REG_WIDTHS:
            return self.REG_WIDTHS[reg_lower]
        # Default to 32-bit for x86, 64-bit for amd64
        return 64 if self.arch == "amd64" else 32

    def __init__(self, arch: str = "x86"):
        """
        Initialize the harness for a given architecture.

        Args:
            arch: Architecture name (x86, amd64, arm, arm64, mips32)
        """
        self.arch = arch
        self._angr_arch = self.ARCH_MAPPING.get(arch, arch.upper())

    def run_rust(self, test: DifferentialTestCase) -> ExecutionResult:
        """
        Execute test case on Rust engine.

        Args:
            test: Test case to execute

        Returns:
            ExecutionResult with final register/memory state
        """
        if not RUST_ENGINE_AVAILABLE:
            return ExecutionResult(
                registers={},
                error="Rust engine not available"
            )

        try:
            engine = RustVEXEngineWrapper(test.arch)

            # Map code memory and write shellcode
            engine.map_memory_data(self.CODE_BASE, test.shellcode)

            # Set up initial memory
            for addr, data in test.initial_mem.items():
                # Align to page boundary
                page_addr = addr & ~0xFFF
                engine.map_memory(page_addr, 0x2000)
                engine.write_memory(addr, data)

            # Set initial registers (convert negative to unsigned 2's complement)
            for reg, value in test.initial_regs.items():
                width = self._reg_width(reg)
                unsigned_value = self._to_unsigned(value, width)
                engine.set_register(reg, unsigned_value)

            # Set PC and execute - use execute_code to lift and run
            engine.pc = self.CODE_BASE
            engine.execute_code(test.shellcode)

            # Collect results
            result_regs = {}
            for reg in test.compare_regs:
                try:
                    result_regs[reg] = engine.get_register(reg)
                except Exception as e:
                    l.debug("Failed to read register %s: %s", reg, e)

            return ExecutionResult(
                registers=result_regs,
                pc=engine.pc
            )

        except Exception as e:
            return ExecutionResult(
                registers={},
                error=f"Rust execution error: {e}"
            )

    def run_python(self, test: DifferentialTestCase) -> ExecutionResult:
        """
        Execute test case on Python VEX engine.

        Args:
            test: Test case to execute

        Returns:
            ExecutionResult with final register/memory state
        """
        try:
            proj = angr.load_shellcode(
                test.shellcode,
                arch=self._angr_arch,
                load_address=self.CODE_BASE
            )

            state = proj.factory.blank_state(
                addr=self.CODE_BASE,
                add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
            )

            # Set up initial memory
            for addr, data in test.initial_mem.items():
                state.memory.store(addr, data)

            # Set initial registers
            for reg, value in test.initial_regs.items():
                setattr(state.regs, reg, value)

            # Execute all instructions in the shellcode
            # Count instructions by counting IMark statements in the lifted IRSB
            import pyvex
            irsb = pyvex.lift(test.shellcode, self.CODE_BASE, state.arch)
            num_inst = irsb.instructions

            succ = state.step(num_inst=num_inst)

            if not succ.successors:
                return ExecutionResult(
                    registers={},
                    error="No successors from Python execution"
                )

            final_state = succ.successors[0]

            # Collect results
            result_regs = {}
            for reg in test.compare_regs:
                try:
                    val = getattr(final_state.regs, reg)
                    if hasattr(val, "concrete") and val.concrete:
                        result_regs[reg] = final_state.solver.eval(val)
                    else:
                        # Symbolic - evaluate to concrete
                        result_regs[reg] = final_state.solver.eval(val)
                except Exception as e:
                    l.debug("Failed to read register %s: %s", reg, e)

            return ExecutionResult(
                registers=result_regs,
                pc=final_state.solver.eval(final_state.regs.pc)
            )

        except Exception as e:
            return ExecutionResult(
                registers={},
                error=f"Python execution error: {e}"
            )

    def compare(self, test: DifferentialTestCase) -> ComparisonResult:
        """
        Run test on both engines and compare results.

        Args:
            test: Test case to run

        Returns:
            ComparisonResult with detailed comparison
        """
        rust_result = self.run_rust(test)
        python_result = self.run_python(test)

        # Check for errors
        if rust_result.error or python_result.error:
            return ComparisonResult(
                test=test,
                rust_result=rust_result,
                python_result=python_result,
                register_diffs={},
                memory_diffs={},
                match=False
            )

        # Compare registers
        register_diffs = {}
        for reg in test.compare_regs:
            rust_val = rust_result.registers.get(reg)
            python_val = python_result.registers.get(reg)

            if rust_val is None or python_val is None:
                # Missing register value
                register_diffs[reg] = {
                    "python": python_val if python_val is not None else -1,
                    "rust": rust_val if rust_val is not None else -1
                }
            elif rust_val != python_val:
                register_diffs[reg] = {
                    "python": python_val,
                    "rust": rust_val
                }

        # For now, skip memory comparison in basic implementation
        memory_diffs = {}

        match = len(register_diffs) == 0 and len(memory_diffs) == 0

        return ComparisonResult(
            test=test,
            rust_result=rust_result,
            python_result=python_result,
            register_diffs=register_diffs,
            memory_diffs=memory_diffs,
            match=match
        )

    def run_batch(self, tests: list[DifferentialTestCase]) -> list[ComparisonResult]:
        """
        Run a batch of tests and return results.

        Args:
            tests: List of test cases to run

        Returns:
            List of comparison results
        """
        results = []
        for test in tests:
            result = self.compare(test)
            results.append(result)
        return results
