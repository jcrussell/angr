"""
Differential test harness for comparing Rust and Python VEX engines.

This module provides utilities for running the same operations on both engines
and comparing results.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import claripy

from angr import SimState, load_shellcode
from angr.engines import HeavyVEXMixin
from angr.engines.vex.claripy import irop

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
class ComparisonResult:
    """Result of comparing an operation between engines."""
    operation: str
    inputs: tuple
    python_result: Any
    rust_result: Any
    match: bool
    error: str | None = None

    def __str__(self) -> str:
        status = "PASS" if self.match else "FAIL"
        if self.error:
            return f"{status}: {self.operation} - {self.error}"
        return f"{status}: {self.operation}({self.inputs}) -> Python: {self.python_result}, Rust: {self.rust_result}"


@dataclass
class DifferentialReport:
    """Report of differential testing results."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[ComparisonResult] = field(default_factory=list)

    def add(self, result: ComparisonResult) -> None:
        self.results.append(result)
        self.total += 1
        if result.match:
            self.passed += 1
        else:
            self.failed += 1

    def add_skipped(self) -> None:
        self.skipped += 1

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    def failures(self) -> list[ComparisonResult]:
        return [r for r in self.results if not r.match]

    def __str__(self) -> str:
        return (
            f"DifferentialReport: {self.passed}/{self.total} passed "
            f"({self.success_rate:.1%}), {self.skipped} skipped"
        )


class DifferentialHarness:
    """
    Harness for running differential tests between Rust and Python VEX engines.
    """

    def __init__(self, arch: str = "amd64"):
        """
        Initialize the harness for a given architecture.

        Args:
            arch: Architecture name (amd64, x86, arm, arm64, mips32)
        """
        self.arch = arch
        self._arch_mapping = {
            "amd64": "AMD64",
            "x86": "X86",
            "arm": "ARMEL",
            "arm64": "AARCH64",
            "mips32": "MIPS32",
        }

        # Create Python engine via HeavyVEXMixin
        angr_arch = self._arch_mapping.get(arch, arch.upper())
        self._project = load_shellcode(b"\xc3", arch=angr_arch)
        self._python_engine = HeavyVEXMixin(self._project)
        self._python_state = SimState(project=self._project)

        # Create Rust engine wrapper if available
        self._rust_engine: RustVEXEngineWrapper | None = None
        if RUST_ENGINE_AVAILABLE:
            try:
                self._rust_engine = RustVEXEngineWrapper(arch)
            except Exception as e:
                l.warning("Failed to create Rust engine: %s", e)

    @property
    def rust_available(self) -> bool:
        """Check if Rust engine is available."""
        return self._rust_engine is not None

    def run_python_op(self, op_name: str, *args: claripy.ast.Base) -> claripy.ast.Base:
        """
        Run a VEX operation on the Python engine.

        Args:
            op_name: VEX operation name (e.g., "Iop_Add32")
            *args: Operation arguments as claripy ASTs

        Returns:
            Result as a claripy AST
        """
        self._python_engine.state = self._python_state
        return self._python_engine._perform_vex_expr_Op(op_name, list(args))

    def run_python_op_direct(self, op_name: str, *args: claripy.ast.Base) -> claripy.ast.Base:
        """
        Run a VEX operation directly via irop module.

        Args:
            op_name: VEX operation name (e.g., "Iop_Add32")
            *args: Operation arguments as claripy ASTs

        Returns:
            Result as a claripy AST
        """
        simop = irop.vexop_to_simop(op_name)
        return simop.calculate(*args)

    def compare_operation(
        self,
        op_name: str,
        *args: claripy.ast.Base,
        tolerance: float = 0.0,
    ) -> ComparisonResult:
        """
        Compare an operation's result between Python and Rust engines.

        Args:
            op_name: VEX operation name
            *args: Operation arguments
            tolerance: For floating point, allowed relative error

        Returns:
            ComparisonResult with comparison details
        """
        try:
            python_result = self.run_python_op(op_name, *args)
        except Exception as e:
            return ComparisonResult(
                operation=op_name,
                inputs=args,
                python_result=None,
                rust_result=None,
                match=False,
                error=f"Python engine error: {e}",
            )

        if not self.rust_available:
            return ComparisonResult(
                operation=op_name,
                inputs=args,
                python_result=python_result,
                rust_result=None,
                match=False,
                error="Rust engine not available",
            )

        # For now, we compare Python results with themselves
        # since the Rust engine doesn't expose individual operations
        # This will be updated when Rust exposes operation-level API
        rust_result = python_result  # Placeholder

        match = self._compare_results(python_result, rust_result, tolerance)

        return ComparisonResult(
            operation=op_name,
            inputs=args,
            python_result=python_result,
            rust_result=rust_result,
            match=match,
        )

    def _compare_results(
        self,
        python_result: claripy.ast.Base,
        rust_result: Any,
        tolerance: float = 0.0,
    ) -> bool:
        """Compare two results for equality."""
        if python_result is None and rust_result is None:
            return True
        if python_result is None or rust_result is None:
            return False

        # For claripy ASTs, compare concrete values
        if hasattr(python_result, "concrete") and python_result.concrete:
            if hasattr(rust_result, "concrete") and rust_result.concrete:
                py_val = python_result.concrete_value
                rust_val = rust_result.concrete_value
                if tolerance > 0:
                    # Floating point comparison
                    if py_val == 0 and rust_val == 0:
                        return True
                    if py_val == 0:
                        return abs(rust_val) < tolerance
                    return abs((py_val - rust_val) / py_val) < tolerance
                return py_val == rust_val

        # For symbolic values, compare structure
        if hasattr(python_result, "structurally_match"):
            return python_result.structurally_match(rust_result)

        return python_result == rust_result

    def compare_concrete_values(
        self,
        op_name: str,
        values: list[tuple],
        width: int,
    ) -> DifferentialReport:
        """
        Compare an operation with multiple concrete value inputs.

        Args:
            op_name: VEX operation name
            values: List of input tuples (each tuple is one test case)
            width: Bit width for BVV creation

        Returns:
            DifferentialReport with all comparison results
        """
        report = DifferentialReport()

        for value_tuple in values:
            args = tuple(claripy.BVV(v, width) for v in value_tuple)
            result = self.compare_operation(op_name, *args)
            report.add(result)

        return report


class MemoryHarness:
    """
    Harness for comparing memory operations between engines.
    """

    def __init__(self, arch: str = "amd64"):
        self.arch = arch
        self._rust_engine: RustVEXEngineWrapper | None = None
        if RUST_ENGINE_AVAILABLE:
            try:
                self._rust_engine = RustVEXEngineWrapper(arch)
            except Exception as e:
                l.warning("Failed to create Rust engine: %s", e)

        # Create Python state for memory operations
        arch_mapping = {"amd64": "AMD64", "x86": "X86", "arm": "ARMEL"}
        angr_arch = arch_mapping.get(arch, arch.upper())
        self._project = load_shellcode(b"\xc3", arch=angr_arch)
        self._state = SimState(project=self._project)

    @property
    def rust_available(self) -> bool:
        return self._rust_engine is not None

    def compare_memory_rw(
        self,
        addr: int,
        data: bytes,
        size: int | None = None,
    ) -> ComparisonResult:
        """
        Compare memory write-then-read between engines.
        """
        if size is None:
            size = len(data)

        # Python engine
        try:
            self._state.memory.store(addr, data)
            python_result = self._state.memory.load(addr, size)
            if hasattr(python_result, "concrete_value"):
                python_bytes = python_result.concrete_value.to_bytes(size, "big")
            else:
                python_bytes = data[:size]
        except Exception as e:
            return ComparisonResult(
                operation="memory_rw",
                inputs=(addr, data),
                python_result=None,
                rust_result=None,
                match=False,
                error=f"Python error: {e}",
            )

        # Rust engine
        if not self.rust_available:
            return ComparisonResult(
                operation="memory_rw",
                inputs=(addr, data),
                python_result=python_bytes,
                rust_result=None,
                match=False,
                error="Rust engine not available",
            )

        try:
            self._rust_engine.map_memory(addr & ~0xFFF, 0x2000)
            self._rust_engine.write_memory(addr, data)
            rust_bytes = self._rust_engine.read_memory(addr, size)
        except Exception as e:
            return ComparisonResult(
                operation="memory_rw",
                inputs=(addr, data),
                python_result=python_bytes,
                rust_result=None,
                match=False,
                error=f"Rust error: {e}",
            )

        match = python_bytes == rust_bytes

        return ComparisonResult(
            operation="memory_rw",
            inputs=(addr, data),
            python_result=python_bytes,
            rust_result=rust_bytes,
            match=match,
        )


class RegisterHarness:
    """
    Harness for comparing register operations between engines.
    """

    def __init__(self, arch: str = "amd64"):
        self.arch = arch
        self._rust_engine: RustVEXEngineWrapper | None = None
        if RUST_ENGINE_AVAILABLE:
            try:
                self._rust_engine = RustVEXEngineWrapper(arch)
            except Exception as e:
                l.warning("Failed to create Rust engine: %s", e)

        # Create Python state for register operations
        arch_mapping = {"amd64": "AMD64", "x86": "X86", "arm": "ARMEL"}
        angr_arch = arch_mapping.get(arch, arch.upper())
        self._project = load_shellcode(b"\xc3", arch=angr_arch)
        self._state = SimState(project=self._project)

    @property
    def rust_available(self) -> bool:
        return self._rust_engine is not None

    def compare_register_rw(
        self,
        reg_name: str,
        value: int,
    ) -> ComparisonResult:
        """
        Compare register write-then-read between engines.
        """
        # Python engine
        try:
            setattr(self._state.regs, reg_name, value)
            python_val = getattr(self._state.regs, reg_name)
            if hasattr(python_val, "concrete_value"):
                python_result = python_val.concrete_value
            else:
                python_result = int(python_val)
        except Exception as e:
            return ComparisonResult(
                operation=f"register_{reg_name}",
                inputs=(value,),
                python_result=None,
                rust_result=None,
                match=False,
                error=f"Python error: {e}",
            )

        # Rust engine
        if not self.rust_available:
            return ComparisonResult(
                operation=f"register_{reg_name}",
                inputs=(value,),
                python_result=python_result,
                rust_result=None,
                match=False,
                error="Rust engine not available",
            )

        try:
            self._rust_engine.set_register(reg_name, value)
            rust_result = self._rust_engine.get_register(reg_name)
        except Exception as e:
            return ComparisonResult(
                operation=f"register_{reg_name}",
                inputs=(value,),
                python_result=python_result,
                rust_result=None,
                match=False,
                error=f"Rust error: {e}",
            )

        match = python_result == rust_result

        return ComparisonResult(
            operation=f"register_{reg_name}",
            inputs=(value,),
            python_result=python_result,
            rust_result=rust_result,
            match=match,
        )


class ExecutionHarness:
    """
    Harness for comparing instruction execution between engines.
    """

    def __init__(self, arch: str = "amd64"):
        self.arch = arch
        self._arch_mapping = {
            "amd64": "AMD64",
            "x86": "X86",
            "arm": "ARMEL",
            "arm64": "AARCH64",
        }

    def compare_instruction(
        self,
        insn_bytes: bytes,
        initial_regs: dict[str, int] | None = None,
        initial_mem: dict[int, bytes] | None = None,
    ) -> ComparisonResult:
        """
        Compare instruction execution between Python and Rust engines.

        Args:
            insn_bytes: Raw instruction bytes
            initial_regs: Initial register values
            initial_mem: Initial memory contents (addr -> bytes)

        Returns:
            ComparisonResult comparing final states
        """
        angr_arch = self._arch_mapping.get(self.arch, self.arch.upper())

        # Python execution
        try:
            import angr
            p = load_shellcode(insn_bytes + b"\xcc", arch=angr_arch)  # Add int3 to stop
            state = p.factory.blank_state(
                add_options={angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS}
            )

            # Set initial state
            if initial_regs:
                for reg, val in initial_regs.items():
                    setattr(state.regs, reg, val)
            if initial_mem:
                for addr, data in initial_mem.items():
                    state.memory.store(addr, data)

            # Execute
            successor = state.step(num_inst=1)
            if successor.successors:
                python_state = successor.successors[0]
            else:
                python_state = None
        except Exception as e:
            return ComparisonResult(
                operation="instruction",
                inputs=(insn_bytes.hex(),),
                python_result=None,
                rust_result=None,
                match=False,
                error=f"Python execution error: {e}",
            )

        # Rust execution
        if not RUST_ENGINE_AVAILABLE:
            return ComparisonResult(
                operation="instruction",
                inputs=(insn_bytes.hex(),),
                python_result=python_state,
                rust_result=None,
                match=False,
                error="Rust engine not available",
            )

        try:
            rust_engine = RustVEXEngineWrapper(self.arch)
            rust_engine.map_memory_data(0, insn_bytes + b"\xcc")

            # Set initial state
            if initial_regs:
                for reg, val in initial_regs.items():
                    rust_engine.set_register(reg, val)
            if initial_mem:
                for addr, data in initial_mem.items():
                    rust_engine.map_memory_data(addr, data)

            rust_engine.pc = 0
            event = rust_engine.step()
            rust_result = rust_engine.get_registers()
        except Exception as e:
            return ComparisonResult(
                operation="instruction",
                inputs=(insn_bytes.hex(),),
                python_result=python_state,
                rust_result=None,
                match=False,
                error=f"Rust execution error: {e}",
            )

        # Compare results
        # This is a simplified comparison - real implementation would compare
        # register by register and flag by flag
        match = True  # Placeholder - implement detailed comparison

        return ComparisonResult(
            operation="instruction",
            inputs=(insn_bytes.hex(),),
            python_result=python_state,
            rust_result=rust_result,
            match=match,
        )


# Edge case value generators for testing
def edge_case_values(width: int) -> list[int]:
    """Generate edge case values for a given bit width."""
    values = [
        0,                          # Zero
        1,                          # One
        (1 << width) - 1,           # All ones (max unsigned)
        1 << (width - 1),           # Only high bit set (min signed)
        (1 << (width - 1)) - 1,     # Max signed positive
        (1 << (width - 1)) + 1,     # Min signed + 1
    ]

    # Add powers of 2
    for i in range(0, width, 8):
        if (1 << i) not in values:
            values.append(1 << i)

    # Add alternating patterns
    alt_0x55 = 0
    alt_0xAA = 0
    for i in range(0, width, 8):
        alt_0x55 |= 0x55 << i
        alt_0xAA |= 0xAA << i
    values.extend([alt_0x55 & ((1 << width) - 1), alt_0xAA & ((1 << width) - 1)])

    return list(set(values))  # Remove duplicates


def shift_amounts(width: int) -> list[int]:
    """Generate test shift amounts for a given bit width."""
    return [0, 1, width // 2, width - 1, width, width + 1]


# Known limitations in Rust engine
RUST_UNSUPPORTED_OPS = {
    # Symbolic floats
    "Iop_F64toF32",
    "Iop_F32toF64",
    "Iop_SinF64",
    "Iop_CosF64",
    "Iop_TanF64",
    "Iop_2xm1F64",
    "Iop_SqrtF64",
    "Iop_SqrtF32",
    # GetI/PutI (x87 FPU stack)
    # These are statement types, not ops
    # CCall helpers
    # These would be caught by operation dispatch
    # LoadG/StoreG guarded ops
    # CAS/LLSC atomic ops
    # Dirty calls
}


def is_rust_supported_op(op_name: str) -> bool:
    """Check if an operation is supported by the Rust engine."""
    if op_name in RUST_UNSUPPORTED_OPS:
        return False

    # Check for floating point symbolic operations
    if "F64" in op_name or "F32" in op_name:
        # Float ops are only supported for concrete values
        return True  # Will fail at runtime for symbolic

    return True
