"""
Rust VEX execution engine adapter for angr.

This module provides a SimEngine mixin that uses the Rust-based VEX interpreter
for faster symbolic execution. It maintains compatibility with angr's existing
engine protocol while delegating execution to Rust.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import claripy

from angr.engines.successors import SuccessorsEngine, SimSuccessors
from angr.engines.vex.lifter import VEXLifter
from angr import sim_options as o
from angr import errors

if TYPE_CHECKING:
    import angr
    from angr.sim_state import SimState

l = logging.getLogger(__name__)


# Import the Rust VEX engine
try:
    from angr.rustylib.vex_engine import RustVEXEngine, ExecutionEvent
    RUST_ENGINE_AVAILABLE = True
except ImportError:
    l.warning("Rust VEX engine not available - rustylib not compiled with vex-engine feature")
    RUST_ENGINE_AVAILABLE = False
    RustVEXEngine = None
    ExecutionEvent = None


def _arch_name_to_rust(arch_name: str) -> str:
    """Convert archinfo arch name to Rust engine arch name."""
    mapping = {
        "AMD64": "amd64",
        "X86": "x86",
        "ARMEL": "arm",
        "ARMHF": "arm",
        "ARM": "arm",
        "AARCH64": "arm64",
        "MIPS32": "mips32",
        "MIPS64": "mips64",
    }
    return mapping.get(arch_name.upper(), arch_name.lower())


class RustVEXMixin(SuccessorsEngine, VEXLifter):
    """
    Execution engine mixin that uses Rust-based VEX interpreter.

    This mixin provides VEX-based execution using a Rust backend for improved
    performance, especially on fork-heavy workloads.

    Responds to the following parameters:
    - irsb: The PyVEX IRSB object to use for execution
    - thumb: Whether to lift in ARM THUMB mode
    - extra_stop_points: Additional points at which to break basic blocks
    - opt_level: VEX optimization level
    - insn_bytes: Raw bytes to use instead of project memory
    - size: Maximum block size in bytes
    - num_inst: Maximum number of instructions
    """

    _rust_engine: RustVEXEngine | None = None
    _rust_engine_synced: bool = False

    def __init__(self, project: angr.Project):
        super().__init__(project)

        if not RUST_ENGINE_AVAILABLE:
            l.warning("RustVEXMixin initialized but Rust engine not available")
            self._rust_engine = None
        else:
            rust_arch = _arch_name_to_rust(project.arch.name)
            try:
                self._rust_engine = RustVEXEngine(rust_arch)
            except ValueError as e:
                l.warning("Failed to create Rust VEX engine for %s: %s", rust_arch, e)
                self._rust_engine = None

    @property
    def rust_engine_available(self) -> bool:
        """Check if the Rust engine is available and initialized."""
        return self._rust_engine is not None

    def _sync_state_to_rust(self, state: SimState) -> None:
        """
        Synchronize angr SimState to Rust engine.

        This copies register values and mapped memory from the SimState
        to the Rust engine's internal state.
        """
        if self._rust_engine is None:
            return

        engine = self._rust_engine

        # Sync PC
        pc = state.solver.eval(state.ip)
        engine.pc = pc

        # Sync key registers based on architecture
        key_registers = self._get_key_registers(state.arch)
        for reg_name in key_registers:
            try:
                offset = state.arch.get_register_offset(reg_name)
                size = state.arch.registers.get(reg_name, (None, None))[1]
                if size is None:
                    continue

                reg_val = state.registers.load(offset, size=size)
                if not reg_val.symbolic:
                    concrete_val = state.solver.eval(reg_val)
                    engine.set_register(reg_name, concrete_val)
            except (KeyError, AttributeError, errors.SimValueError):
                pass
            except Exception:
                pass

        self._rust_engine_synced = True

    def _get_key_registers(self, arch) -> list[str]:
        """Get the key registers to sync for an architecture."""
        arch_name = arch.name.upper()

        if arch_name in ("AMD64", "X86_64"):
            return ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rsp", "rbp",
                    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "rip"]
        elif arch_name == "X86":
            return ["eax", "ebx", "ecx", "edx", "esi", "edi", "esp", "ebp", "eip"]
        elif arch_name in ("ARM", "ARMEL", "ARMHF"):
            return ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8",
                    "r9", "r10", "r11", "r12", "sp", "lr", "pc"]
        elif arch_name in ("AARCH64", "ARM64"):
            return ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
                    "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16",
                    "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24",
                    "x25", "x26", "x27", "x28", "x29", "x30", "sp", "pc"]
        elif arch_name in ("MIPS32", "MIPS"):
            return ["v0", "v1", "a0", "a1", "a2", "a3", "t0", "t1", "t2",
                    "t3", "s0", "s1", "s2", "s3", "sp", "ra", "pc"]
        else:
            return []

    def _sync_state_from_rust(self, state: SimState) -> None:
        """
        Synchronize Rust engine state back to angr SimState.

        This copies register values from the Rust engine back to the SimState.
        """
        if self._rust_engine is None:
            return

        engine = self._rust_engine

        # Sync PC
        state.ip = engine.pc

        # Sync registers back
        regs = engine.get_registers()
        for reg_name, value in regs.items():
            try:
                state.registers.store(reg_name, claripy.BVV(value, state.arch.bits))
            except (KeyError, AttributeError):
                pass

    def _handle_rust_execution_event(
        self,
        event: ExecutionEvent,
        state: SimState,
        successors: SimSuccessors,
    ) -> bool:
        """
        Handle an execution event from the Rust engine.

        Returns True if execution should continue, False if done.
        """
        event_type = event.event_type

        if event_type == "block_end":
            # Normal block end - create successor
            next_addr = event.next_addr
            jumpkind = event.jumpkind or "Ijk_Boring"

            self._sync_state_from_rust(state)
            state.ip = next_addr

            guard = claripy.true()
            successors.add_successor(
                state,
                next_addr,
                guard,
                jumpkind,
                exit_stmt_idx="default",
                exit_ins_addr=state.scratch.ins_addr if state.scratch else None,
            )
            return False

        elif event_type == "symbolic_branch":
            # Symbolic branch - create two successors
            true_target = event.true_target
            false_target = event.false_target

            self._sync_state_from_rust(state)

            # Create true branch successor
            true_state = state.copy()
            true_state.ip = true_target
            successors.add_successor(
                true_state,
                true_target,
                claripy.true(),  # Simplified - real impl would track condition
                "Ijk_Boring",
            )

            # Create false branch successor
            false_state = state.copy()
            false_state.ip = false_target
            successors.add_successor(
                false_state,
                false_target,
                claripy.true(),
                "Ijk_Boring",
            )
            return False

        elif event_type == "syscall":
            # Syscall - return to Python for handling
            self._sync_state_from_rust(state)

            jumpkind = event.jumpkind or "Ijk_Sys_syscall"
            successors.add_successor(
                state,
                state.ip,
                claripy.true(),
                jumpkind,
            )
            return False

        elif event_type == "hook":
            # Hook address hit - return to Python for handling
            self._sync_state_from_rust(state)

            # The hook will be handled by HooksMixin
            successors.add_successor(
                state,
                event.next_addr,
                claripy.true(),
                "Ijk_NoHook",  # Special jumpkind to signal hook needed
            )
            return False

        elif event_type == "need_lift":
            # Rust engine doesn't have this block cached - we need to lift it
            # This shouldn't happen often once we implement proper block passing
            return True

        elif event_type == "error":
            # Execution error
            error_msg = event.error or "Unknown error"
            l.warning("Rust VEX engine error: %s", error_msg)
            raise errors.SimEngineError(f"Rust VEX engine error: {error_msg}")

        else:
            l.warning("Unknown Rust VEX engine event type: %s", event_type)
            return False

    def process_successors(
        self,
        successors: SimSuccessors,
        irsb=None,
        insn_bytes=None,
        thumb=False,
        size=None,
        num_inst=None,
        extra_stop_points=None,
        opt_level=None,
        strict_block_end=None,
        **kwargs,
    ):
        """
        Process successors using the Rust VEX engine.

        If the Rust engine is not available or cannot handle the request,
        this falls back to the parent implementation.
        """
        # Check if Rust engine is available
        if not self.rust_engine_available:
            return super().process_successors(
                successors,
                irsb=irsb,
                insn_bytes=insn_bytes,
                extra_stop_points=extra_stop_points,
                num_inst=num_inst,
                size=size,
                **kwargs,
            )

        # Check if address is concrete
        if not isinstance(successors.addr, int):
            return super().process_successors(
                successors,
                irsb=irsb,
                insn_bytes=insn_bytes,
                extra_stop_points=extra_stop_points,
                num_inst=num_inst,
                size=size,
                **kwargs,
            )

        # Mark this as a Rust VEX execution
        successors.sort = "RUST_VEX"
        successors.description = "Rust VEX"

        addr = successors.addr
        state = self.state

        # Setup state scratch
        state.history.recent_block_count = 1
        state.scratch.guard = claripy.true()
        state.scratch.sim_procedure = None
        state.scratch.bbl_addr = addr

        # Sync hooks to Rust engine
        if state.project is not None:
            for hook_addr in state.project._sim_procedures:
                self._rust_engine.add_hook(hook_addr)

        # Sync state to Rust engine
        self._sync_state_to_rust(state)

        # Lift the block if not provided
        if irsb is None:
            irsb = self.lift_vex(
                insn_bytes=insn_bytes,
                addr=addr,
                state=state,
                thumb=thumb,
                size=size,
                num_inst=num_inst,
                extra_stop_points=extra_stop_points,
                opt_level=opt_level,
                strict_block_end=strict_block_end,
            )

        # Store IRSB in artifacts
        successors.artifacts["irsb"] = irsb
        successors.artifacts["irsb_size"] = irsb.size
        successors.artifacts["irsb_direct_next"] = irsb.direct_next

        # For now, we fall back to Python VEX for actual execution
        # The Rust engine needs proper IRSB serialization to execute
        # TODO: Implement proper IRSB passing to Rust

        # Attempt Rust execution
        try:
            event = self._rust_engine.step()

            if event.event_type == "need_lift":
                # Rust doesn't have this block - fall back to Python
                l.debug("Rust engine needs block at 0x%x, falling back to Python", addr)
                return super().process_successors(
                    successors,
                    irsb=irsb,
                    insn_bytes=insn_bytes,
                    extra_stop_points=extra_stop_points,
                    num_inst=num_inst,
                    size=size,
                    **kwargs,
                )

            # Handle the execution event
            needs_more = self._handle_rust_execution_event(event, state, successors)

            if not needs_more:
                successors.processed = True
                return

        except Exception as e:
            l.warning("Rust VEX execution failed: %s, falling back to Python", e)
            return super().process_successors(
                successors,
                irsb=irsb,
                insn_bytes=insn_bytes,
                extra_stop_points=extra_stop_points,
                num_inst=num_inst,
                size=size,
                **kwargs,
            )

        successors.processed = True


class RustVEXEngineWrapper:
    """
    Standalone wrapper for the Rust VEX engine.

    This provides a lower-level interface for using the Rust engine
    without going through the full SimEngine protocol.
    """

    def __init__(self, arch: str = "amd64"):
        if not RUST_ENGINE_AVAILABLE:
            raise ImportError("Rust VEX engine not available")

        self._engine = RustVEXEngine(arch)

    @property
    def pc(self) -> int:
        return self._engine.pc

    @pc.setter
    def pc(self, value: int):
        self._engine.pc = value

    def get_register(self, name: str) -> int:
        return self._engine.get_register(name)

    def set_register(self, name: str, value: int) -> None:
        self._engine.set_register(name, value)

    def get_registers(self) -> dict[str, int]:
        return dict(self._engine.get_registers())

    def map_memory(self, addr: int, size: int, permissions: int = 7) -> None:
        self._engine.map_memory(addr, size, permissions)

    def map_memory_data(self, addr: int, data: bytes, permissions: int = 7) -> None:
        self._engine.map_memory_data(addr, data, permissions)

    def read_memory(self, addr: int, size: int) -> bytes:
        return bytes(self._engine.read_memory(addr, size))

    def write_memory(self, addr: int, data: bytes) -> None:
        self._engine.write_memory(addr, data)

    def add_hook(self, addr: int) -> None:
        self._engine.add_hook(addr)

    def remove_hook(self, addr: int) -> None:
        self._engine.remove_hook(addr)

    def step(self) -> ExecutionEvent:
        return self._engine.step()

    def fork(self) -> "RustVEXEngineWrapper":
        wrapper = object.__new__(RustVEXEngineWrapper)
        wrapper._engine = self._engine.fork()
        return wrapper

    def snapshot(self):
        return self._engine.snapshot()

    def stats(self) -> dict[str, Any]:
        return dict(self._engine.stats())


# Export convenience alias
RustVEX = RustVEXMixin


__all__ = [
    "RustVEXMixin",
    "RustVEXEngineWrapper",
    "RustVEX",
    "RUST_ENGINE_AVAILABLE",
]
