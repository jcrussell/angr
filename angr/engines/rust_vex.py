"""
Rust VEX execution engine adapter for angr.

This module provides a SimEngine mixin that uses the Rust-based VEX interpreter
for faster symbolic execution. It maintains compatibility with angr's existing
engine protocol while delegating execution to Rust.
"""
from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

import claripy


# Profiling stats - can be enabled for performance analysis
class RustVEXProfiler:
    """Collects timing statistics for Rust VEX engine operations."""

    def __init__(self):
        self.enabled = False
        self.reset()

    def reset(self):
        self.serialize_time = 0.0
        self.execute_time = 0.0
        self.sync_to_rust_time = 0.0
        self.sync_from_rust_time = 0.0
        self.lift_time = 0.0
        self.block_count = 0
        self.total_time = 0.0

    def report(self) -> dict:
        """Return profiling statistics as a dictionary."""
        if self.block_count == 0:
            return {"blocks": 0}
        return {
            "blocks": self.block_count,
            "total_time_ms": self.total_time * 1000,
            "serialize_time_ms": self.serialize_time * 1000,
            "execute_time_ms": self.execute_time * 1000,
            "sync_to_rust_time_ms": self.sync_to_rust_time * 1000,
            "sync_from_rust_time_ms": self.sync_from_rust_time * 1000,
            "lift_time_ms": self.lift_time * 1000,
            "avg_per_block_us": (self.total_time / self.block_count) * 1e6,
            "avg_serialize_us": (self.serialize_time / self.block_count) * 1e6,
            "avg_execute_us": (self.execute_time / self.block_count) * 1e6,
            "breakdown_pct": {
                "serialize": (self.serialize_time / self.total_time * 100) if self.total_time > 0 else 0,
                "execute": (self.execute_time / self.total_time * 100) if self.total_time > 0 else 0,
                "sync_to_rust": (self.sync_to_rust_time / self.total_time * 100) if self.total_time > 0 else 0,
                "sync_from_rust": (self.sync_from_rust_time / self.total_time * 100) if self.total_time > 0 else 0,
                "lift": (self.lift_time / self.total_time * 100) if self.total_time > 0 else 0,
            }
        }


# Global profiler instance
_profiler = RustVEXProfiler()

from angr.engines.successors import SuccessorsEngine, SimSuccessors
from angr.engines.vex.lifter import VEXLifter
from angr import sim_options as o
from angr import errors


def enable_profiling(enabled: bool = True) -> None:
    """Enable or disable profiling for the Rust VEX engine."""
    _profiler.enabled = enabled
    if enabled:
        _profiler.reset()


def get_profiler() -> RustVEXProfiler:
    """Get the global profiler instance."""
    return _profiler


if TYPE_CHECKING:
    import angr
    from angr.sim_state import SimState
    import pyvex

l = logging.getLogger(__name__)


# Import the Rust VEX engine
try:
    from angr.rustylib.vex_engine import (
        RustVEXEngine,
        ExecutionEvent,
        PythonCallbacks,
        LoopExecutionEvent,
        ExecutionConfig,
        BranchPolicy,
        DeferredFork,
    )
    RUST_ENGINE_AVAILABLE = True
except ImportError:
    l.warning("Rust VEX engine not available - rustylib not compiled with vex-engine feature")
    RUST_ENGINE_AVAILABLE = False
    RustVEXEngine = None
    ExecutionEvent = None
    PythonCallbacks = None
    LoopExecutionEvent = None


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


def _serialize_irsb(irsb: "pyvex.IRSB") -> str:
    """
    Serialize a pyvex IRSB to JSON for the Rust engine.

    This converts pyvex's IRSB structure to a JSON format that matches
    the Rust pyvex_bridge deserialization format.
    """
    return json.dumps(_irsb_to_dict(irsb))


def _irsb_to_dict(irsb: "pyvex.IRSB") -> dict:
    """Convert pyvex IRSB to dictionary format."""
    return {
        "addr": irsb.addr,
        "arch": irsb.arch.name,
        "statements": [_stmt_to_dict(s) for s in irsb.statements],
        "next": _expr_to_dict(irsb.next),
        "jumpkind": irsb.jumpkind,
        "offsIP": irsb.offsIP,
        "tyenv": _tyenv_to_dict(irsb.tyenv),
    }


def _tyenv_to_dict(tyenv) -> dict:
    """Convert type environment to dictionary."""
    types = []
    for i in range(tyenv.types_used):
        ty = tyenv.lookup(i)
        types.append(ty if ty else "Ity_I64")
    return {"types": types}


def _stmt_to_dict(stmt) -> dict:
    """Convert a pyvex statement to dictionary."""
    import pyvex

    if isinstance(stmt, pyvex.stmt.NoOp):
        return {"tag": "Ist_NoOp"}

    elif isinstance(stmt, pyvex.stmt.IMark):
        return {
            "tag": "Ist_IMark",
            "addr": stmt.addr,
            "len": stmt.len,
            "delta": stmt.delta,
        }

    elif isinstance(stmt, pyvex.stmt.AbiHint):
        return {
            "tag": "Ist_AbiHint",
            "base": _expr_to_dict(stmt.base),
            "len": stmt.len,
            "nia": _expr_to_dict(stmt.nia),
        }

    elif isinstance(stmt, pyvex.stmt.Put):
        return {
            "tag": "Ist_Put",
            "offset": stmt.offset,
            "data": _expr_to_dict(stmt.data),
        }

    elif isinstance(stmt, pyvex.stmt.PutI):
        return {
            "tag": "Ist_PutI",
            "descr": _regarray_to_dict(stmt.descr),
            "ix": _expr_to_dict(stmt.ix),
            "bias": stmt.bias,
            "data": _expr_to_dict(stmt.data),
        }

    elif isinstance(stmt, pyvex.stmt.WrTmp):
        return {
            "tag": "Ist_WrTmp",
            "tmp": stmt.tmp,
            "data": _expr_to_dict(stmt.data),
        }

    elif isinstance(stmt, pyvex.stmt.Store):
        return {
            "tag": "Ist_Store",
            "addr": _expr_to_dict(stmt.addr),
            "data": _expr_to_dict(stmt.data),
            "end": stmt.end,
        }

    elif isinstance(stmt, pyvex.stmt.StoreG):
        return {
            "tag": "Ist_StoreG",
            "addr": _expr_to_dict(stmt.addr),
            "data": _expr_to_dict(stmt.data),
            "guard": _expr_to_dict(stmt.guard),
            "end": stmt.end,
        }

    elif isinstance(stmt, pyvex.stmt.LoadG):
        return {
            "tag": "Ist_LoadG",
            "dst": stmt.dst,
            "addr": _expr_to_dict(stmt.addr),
            "alt": _expr_to_dict(stmt.alt),
            "guard": _expr_to_dict(stmt.guard),
            "cvt": stmt.cvt,
            "end": stmt.end,
        }

    elif isinstance(stmt, pyvex.stmt.CAS):
        result = {
            "tag": "Ist_CAS",
            "oldHi": stmt.oldHi,
            "oldLo": stmt.oldLo,
            "addr": _expr_to_dict(stmt.addr),
            "expdLo": _expr_to_dict(stmt.expdLo),
            "dataLo": _expr_to_dict(stmt.dataLo),
            "end": stmt.end,
        }
        if stmt.expdHi is not None:
            result["expdHi"] = _expr_to_dict(stmt.expdHi)
        if stmt.dataHi is not None:
            result["dataHi"] = _expr_to_dict(stmt.dataHi)
        return result

    elif isinstance(stmt, pyvex.stmt.LLSC):
        result = {
            "tag": "Ist_LLSC",
            "result": stmt.result,
            "addr": _expr_to_dict(stmt.addr),
            "end": stmt.end,
        }
        if stmt.storedata is not None:
            result["storedata"] = _expr_to_dict(stmt.storedata)
        return result

    elif isinstance(stmt, pyvex.stmt.MBE):
        return {
            "tag": "Ist_MBE",
            "event": stmt.event,
        }

    elif isinstance(stmt, pyvex.stmt.Dirty):
        result = {
            "tag": "Ist_Dirty",
            "cee": _callee_to_dict(stmt.cee),
            "tmp": stmt.tmp,
            "mFx": stmt.mFx,
            "mSize": stmt.mSize,
            "nFxState": stmt.nFxState,
            "args": [_expr_to_dict(a) for a in stmt.args],
        }
        if stmt.guard is not None:
            result["guard"] = _expr_to_dict(stmt.guard)
        if stmt.mAddr is not None:
            result["mAddr"] = _expr_to_dict(stmt.mAddr)
        return result

    elif isinstance(stmt, pyvex.stmt.Exit):
        return {
            "tag": "Ist_Exit",
            "guard": _expr_to_dict(stmt.guard),
            "dst": _const_to_dict(stmt.dst),
            "jk": stmt.jumpkind,
            "offsIP": stmt.offsIP,
        }

    else:
        # Fallback for unknown statement types
        l.warning("Unknown pyvex statement type: %s", type(stmt).__name__)
        return {"tag": "Ist_NoOp"}


def _expr_to_dict(expr) -> dict:
    """Convert a pyvex expression to dictionary."""
    import pyvex

    if isinstance(expr, pyvex.expr.Const):
        return {
            "tag": "Iex_Const",
            "con": _const_to_dict(expr.con),
        }

    elif isinstance(expr, pyvex.expr.RdTmp):
        return {
            "tag": "Iex_RdTmp",
            "tmp": expr.tmp,
        }

    elif isinstance(expr, pyvex.expr.Get):
        return {
            "tag": "Iex_Get",
            "offset": expr.offset,
            "ty": expr.ty,
        }

    elif isinstance(expr, pyvex.expr.GetI):
        return {
            "tag": "Iex_GetI",
            "descr": _regarray_to_dict(expr.descr),
            "ix": _expr_to_dict(expr.ix),
            "bias": expr.bias,
        }

    elif isinstance(expr, pyvex.expr.Load):
        return {
            "tag": "Iex_Load",
            "addr": _expr_to_dict(expr.addr),
            "ty": expr.ty,
            "end": expr.end,
        }

    elif isinstance(expr, pyvex.expr.Unop):
        return {
            "tag": "Iex_Unop",
            "op": expr.op,
            "arg": _expr_to_dict(expr.args[0]),
        }

    elif isinstance(expr, pyvex.expr.Binop):
        return {
            "tag": "Iex_Binop",
            "op": expr.op,
            "args": [_expr_to_dict(expr.args[0]), _expr_to_dict(expr.args[1])],
        }

    elif isinstance(expr, pyvex.expr.Triop):
        return {
            "tag": "Iex_Triop",
            "op": expr.op,
            "args": [
                _expr_to_dict(expr.args[0]),
                _expr_to_dict(expr.args[1]),
                _expr_to_dict(expr.args[2]),
            ],
        }

    elif isinstance(expr, pyvex.expr.Qop):
        return {
            "tag": "Iex_Qop",
            "op": expr.op,
            "args": [
                _expr_to_dict(expr.args[0]),
                _expr_to_dict(expr.args[1]),
                _expr_to_dict(expr.args[2]),
                _expr_to_dict(expr.args[3]),
            ],
        }

    elif isinstance(expr, pyvex.expr.ITE):
        return {
            "tag": "Iex_ITE",
            "cond": _expr_to_dict(expr.cond),
            "iftrue": _expr_to_dict(expr.iftrue),
            "iffalse": _expr_to_dict(expr.iffalse),
        }

    elif isinstance(expr, pyvex.expr.CCall):
        return {
            "tag": "Iex_CCall",
            "cee": _callee_to_dict(expr.cee),
            "retty": expr.retty,
            "args": [_expr_to_dict(a) for a in expr.args],
        }

    elif isinstance(expr, pyvex.expr.VECRET):
        return {"tag": "Iex_VECRET"}

    elif isinstance(expr, pyvex.expr.GSPTR):
        return {"tag": "Iex_GSPTR"}

    else:
        # Fallback - treat as a constant 0
        l.warning("Unknown pyvex expression type: %s", type(expr).__name__)
        return {"tag": "Iex_Const", "con": {"tag": "Ico_U64", "value": 0}}


def _const_to_dict(con) -> dict:
    """Convert a pyvex constant to dictionary."""
    import pyvex

    # pyvex constants have a value attribute and a type
    # The tag is based on the size/type
    if hasattr(con, 'value'):
        value = con.value
    else:
        value = 0

    # Determine the constant type from the pyvex constant
    type_name = type(con).__name__

    if "U1" in type_name or (hasattr(con, 'type') and "I1" in str(con.type)):
        return {"tag": "Ico_U1", "value": bool(value)}
    elif "U8" in type_name or (hasattr(con, 'size') and con.size == 8):
        return {"tag": "Ico_U8", "value": value & 0xFF}
    elif "U16" in type_name or (hasattr(con, 'size') and con.size == 16):
        return {"tag": "Ico_U16", "value": value & 0xFFFF}
    elif "U32" in type_name or (hasattr(con, 'size') and con.size == 32):
        return {"tag": "Ico_U32", "value": value & 0xFFFFFFFF}
    elif "U64" in type_name or (hasattr(con, 'size') and con.size == 64):
        return {"tag": "Ico_U64", "value": value & 0xFFFFFFFFFFFFFFFF}
    elif "F32" in type_name:
        return {"tag": "Ico_F32", "value": float(value)}
    elif "F64" in type_name:
        return {"tag": "Ico_F64", "value": float(value)}
    elif "V128" in type_name:
        return {"tag": "Ico_V128", "value": value}
    elif "V256" in type_name:
        # V256 is stored as 4 x u64
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return {"tag": "Ico_V256", "value": list(value)}
        else:
            return {"tag": "Ico_V256", "value": [value & ((1 << 64) - 1), 0, 0, 0]}
    else:
        # Default to U64
        return {"tag": "Ico_U64", "value": int(value) & 0xFFFFFFFFFFFFFFFF}


def _regarray_to_dict(descr) -> dict:
    """Convert a pyvex register array descriptor to dictionary."""
    return {
        "base": descr.base,
        "elemTy": descr.elemTy,
        "nElems": descr.nElems,
    }


def _callee_to_dict(cee) -> dict:
    """Convert a pyvex callee to dictionary."""
    return {
        "name": cee.name if hasattr(cee, 'name') else "",
        "addr": cee.addr if hasattr(cee, 'addr') else 0,
        "mcx_mask": cee.mcx_mask if hasattr(cee, 'mcx_mask') else 0,
    }


class RustVEXCallbacks:
    """
    Provides Python callbacks to the Rust VEX execution engine.

    This class bridges angr's memory model and SimProcedures to the Rust
    engine, allowing Rust to handle the execution loop while Python handles
    memory access and hooks.
    """

    def __init__(self, state: "SimState", project: "angr.Project", lifter: "VEXLifter"):
        """
        Initialize the callbacks.

        Args:
            state: The SimState to operate on.
            project: The angr Project.
            lifter: The VEXLifter for lifting blocks.
        """
        self.state = state
        self.project = project
        self.lifter = lifter
        self._lifted_blocks = {}  # Cache of lifted blocks

    def memory_load(self, addr: int, size: int) -> tuple[bytes, bool, Any]:
        """
        Load from angr's memory model.

        Args:
            addr: Address to load from.
            size: Number of bytes to load.

        Returns:
            Tuple of (concrete_bytes, is_symbolic, symbolic_ast_or_none).
        """
        try:
            val = self.state.memory.load(addr, size, endness='Iend_LE')
            is_sym = val.symbolic

            # FAST PATH: Extract concrete value directly without solver
            if not is_sym:
                if val.op == 'BVV':
                    concrete = val.args[0]
                else:
                    # Fallback for other concrete representations
                    concrete = self.state.solver.eval(val)
            else:
                concrete = self.state.solver.eval(val)

            concrete_bytes = concrete.to_bytes(size, 'little')

            if is_sym:
                return (concrete_bytes, True, val)
            else:
                return (concrete_bytes, False, None)
        except Exception as e:
            l.warning("Memory load failed at 0x%x: %s", addr, e)
            # Return zeros on error
            return (bytes(size), False, None)

    def memory_store(self, addr: int, data: bytes) -> None:
        """
        Store to angr's memory model.

        Args:
            addr: Address to store to.
            data: Bytes to store.
        """
        try:
            size = len(data)
            value = int.from_bytes(data, 'little')
            bv = claripy.BVV(value, size * 8)
            self.state.memory.store(addr, bv, endness='Iend_LE')
        except Exception as e:
            l.warning("Memory store failed at 0x%x: %s", addr, e)

    def on_hook(self, addr: int) -> int:
        """
        Execute a hook at the given address.

        Args:
            addr: Address of the hook.

        Returns:
            New PC after hook execution.
        """
        try:
            # Check if there's a SimProcedure at this address
            if self.project._sim_procedures and addr in self.project._sim_procedures:
                proc_info = self.project._sim_procedures[addr]
                # For now, return the address to let Python handle it
                # TODO: Actually execute the SimProcedure
                return addr

            # No hook found, just return the address
            return addr
        except Exception as e:
            l.warning("Hook execution failed at 0x%x: %s", addr, e)
            return addr

    def on_syscall(self, num: int) -> None:
        """
        Handle a syscall.

        Args:
            num: Syscall number.
        """
        try:
            # Syscall handling is done at the SimOS level
            # For now, just log it - actual handling is done in Python
            l.debug("Syscall %d at PC 0x%x", num, self.state.solver.eval(self.state.ip))
        except Exception as e:
            l.warning("Syscall handling failed for syscall %d: %s", num, e)

    def lift_block(self, addr: int) -> str:
        """
        Lift a block at the given address via pyvex.

        Args:
            addr: Address to lift from.

        Returns:
            IRSB serialized as JSON string.
        """
        try:
            # Check cache first
            if addr in self._lifted_blocks:
                return self._lifted_blocks[addr]

            # Lift the block
            irsb = self.lifter.lift_vex(
                addr=addr,
                state=self.state,
            )

            # Serialize to JSON
            irsb_json = _serialize_irsb(irsb)

            # Cache it
            self._lifted_blocks[addr] = irsb_json

            return irsb_json
        except Exception as e:
            l.warning("Block lifting failed at 0x%x: %s", addr, e)
            raise

    def get_register(self, offset: int, size: int) -> tuple[bytes, bool, Any]:
        """
        Get a register value from angr's state.

        Args:
            offset: Register offset.
            size: Size in bytes.

        Returns:
            Tuple of (concrete_bytes, is_symbolic, symbolic_ast_or_none).
        """
        try:
            val = self.state.registers.load(offset, size=size)
            is_sym = val.symbolic

            # FAST PATH: Extract concrete value directly without solver
            if not is_sym:
                if val.op == 'BVV':
                    concrete = val.args[0]
                else:
                    # Fallback for other concrete representations
                    concrete = self.state.solver.eval(val)
            else:
                concrete = self.state.solver.eval(val)

            concrete_bytes = concrete.to_bytes(size, 'little')

            if is_sym:
                return (concrete_bytes, True, val)
            else:
                return (concrete_bytes, False, None)
        except Exception as e:
            l.warning("Register read failed at offset %d: %s", offset, e)
            return (bytes(size), False, None)

    def put_register(self, offset: int, data: bytes) -> None:
        """
        Set a register value in angr's state.

        Args:
            offset: Register offset.
            data: Bytes to store.
        """
        try:
            size = len(data)
            value = int.from_bytes(data, 'little')
            bv = claripy.BVV(value, size * 8)
            self.state.registers.store(offset, bv)
        except Exception as e:
            l.warning("Register write failed at offset %d: %s", offset, e)

    def setup_rust_callbacks(self) -> "PythonCallbacks":
        """
        Create and configure a Rust PythonCallbacks object.

        Returns:
            Configured PythonCallbacks instance.
        """
        if PythonCallbacks is None:
            raise ImportError("PythonCallbacks not available")

        rust_cbs = PythonCallbacks()
        rust_cbs.set_memory_load(self.memory_load)
        rust_cbs.set_memory_store(self.memory_store)
        rust_cbs.set_on_hook(self.on_hook)
        rust_cbs.set_on_syscall(self.on_syscall)
        rust_cbs.set_lift_block(self.lift_block)
        rust_cbs.set_get_register(self.get_register)
        rust_cbs.set_put_register(self.put_register)

        return rust_cbs


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

    def __init__(self, project: angr.Project, use_deferred_forks: bool = True, max_deferred_forks: int = 50):
        super().__init__(project)

        self._use_deferred_forks = use_deferred_forks
        self._max_deferred_forks = max_deferred_forks
        self._concrete_memory_synced = False

        if not RUST_ENGINE_AVAILABLE:
            l.warning("RustVEXMixin initialized but Rust engine not available")
            self._rust_engine = None
        else:
            rust_arch = _arch_name_to_rust(project.arch.name)
            try:
                self._rust_engine = RustVEXEngine(rust_arch)
                # Configure deferred forks
                self._rust_engine.set_use_deferred_forks(use_deferred_forks)
                self._rust_engine.set_max_deferred_forks(max_deferred_forks)
                self._rust_engine.set_branch_policy(BranchPolicy.take_true())
                # Sync concrete memory regions for fast access
                regions = self._sync_concrete_memory_to_rust()
                self._concrete_memory_synced = True
                if regions > 0:
                    l.debug("Synced %d concrete memory regions to Rust engine", regions)
            except ValueError as e:
                l.warning("Failed to create Rust VEX engine for %s: %s", rust_arch, e)
                self._rust_engine = None

    def configure_deferred_forks(self, enabled: bool = True, max_forks: int = 50, policy: str = "take_true"):
        """
        Configure deferred fork behavior.

        Args:
            enabled: Whether to use deferred forks (default True).
            max_forks: Maximum deferred forks before returning to Python (default 50).
            policy: Branch policy - "take_true", "take_false", "take_fallthrough", or "alternate".
        """
        if self._rust_engine is None:
            return

        self._use_deferred_forks = enabled
        self._max_deferred_forks = max_forks
        self._rust_engine.set_use_deferred_forks(enabled)
        self._rust_engine.set_max_deferred_forks(max_forks)

        policy_map = {
            "take_true": BranchPolicy.take_true(),
            "take_false": BranchPolicy.take_false(),
            "take_fallthrough": BranchPolicy.take_fallthrough(),
            "alternate": BranchPolicy.alternate(),
        }
        if policy in policy_map:
            self._rust_engine.set_branch_policy(policy_map[policy])

    def _sync_concrete_memory_to_rust(self) -> int:
        """
        Map binary's concrete memory regions to Rust engine for fast access.

        This maps read-only sections (like .text, .rodata) directly to Rust,
        allowing memory loads from these regions to skip the Python callback
        entirely.

        Returns:
            Number of regions mapped.
        """
        if not self.rust_engine_available:
            return 0

        regions_mapped = 0

        # Clear any existing memory mappings
        self._rust_engine.clear_memory()

        # Map binary sections that are readable but not writable
        for obj in self.project.loader.all_objects:
            # Map segments from the loader's memory
            for segment in obj.segments:
                # Only map readable, non-writable segments (code/rodata)
                if segment.is_readable and not segment.is_writable:
                    try:
                        # Load the data from the loader's memory
                        data = self.project.loader.memory.load(
                            segment.vaddr,
                            segment.memsize
                        )
                        # Map to Rust engine (permissions: R-X = 5)
                        perms = 4  # R--
                        if segment.is_executable:
                            perms |= 1  # R-X
                        self._rust_engine.map_memory_data(
                            segment.vaddr,
                            bytes(data),
                            perms
                        )
                        regions_mapped += 1
                        l.debug(
                            "Mapped segment 0x%x-0x%x (%d bytes) to Rust",
                            segment.vaddr,
                            segment.vaddr + segment.memsize,
                            segment.memsize
                        )
                    except Exception as e:
                        l.debug("Failed to map segment at 0x%x: %s", segment.vaddr, e)

        return regions_mapped

    @property
    def rust_engine_available(self) -> bool:
        """Check if the Rust engine is available and initialized."""
        return self._rust_engine is not None

    def process(self, state, **kwargs):
        """
        Override to enable loop execution when RUST_VEX_LOOP option is set.

        When the RUST_VEX_LOOP sim_option is enabled and prerequisites are met,
        this uses process_successors_loop() for multi-block execution with
        deferred forks. Otherwise, falls back to standard single-block execution.
        """
        # Check if loop execution is enabled and prerequisites are met
        if (o.RUST_VEX_LOOP in state.options
            and self.rust_engine_available
            and isinstance(state.addr, int)):
            # Use loop execution path with deferred forks
            return self._process_with_loop(state, **kwargs)
        # Fall back to standard execution
        return super().process(state, **kwargs)

    def _process_with_loop(self, state, max_blocks=100, **kwargs):
        """
        Execute using process_successors_loop for multi-block execution.

        This method performs the standard process() setup and then delegates
        to process_successors_loop() for Rust-based multi-block execution
        with deferred fork support.

        Args:
            state: The SimState to execute.
            max_blocks: Maximum blocks to execute before returning (default 100).
            **kwargs: Additional arguments passed to process_successors_loop.

        Returns:
            SimSuccessors containing all execution results.
        """
        inline = kwargs.pop("inline", False)
        force_addr = kwargs.pop("force_addr", None)

        ip = state._ip
        if force_addr is not None:
            addr = force_addr
        elif isinstance(ip, claripy.ast.BV):
            addr = state.solver.eval(ip)
        else:
            addr = ip

        # Copy state if needed
        if not inline and o.COPY_STATES in state.options:
            new_state = state.copy()
        else:
            new_state = state
        old_state = state
        del state
        self.state = new_state

        # Setup history
        new_state.register_plugin("history", old_state.history.make_child())
        new_state.history.recent_bbl_addrs.append(addr)

        # Create successors object
        self.successors = SimSuccessors(addr, old_state)

        # Call process_successors_loop instead of process_successors
        self.process_successors_loop(self.successors, max_blocks=max_blocks, **kwargs)

        self.successors._finalize()
        return self.successors

    def _sync_state_to_rust(self, state: SimState) -> None:
        """
        Synchronize angr SimState to Rust engine.

        This copies register values from the SimState to the Rust engine.
        """
        if self._rust_engine is None:
            return

        engine = self._rust_engine

        # Sync PC
        pc = state.solver.eval(state.ip)
        engine.pc = pc

        # Sync key registers
        self._sync_registers_individual(state, engine)

        self._rust_engine_synced = True

    def _sync_registers_individual(self, state: SimState, engine) -> None:
        """Sync registers individually (fallback path)."""
        key_registers = self._get_key_registers(state.arch)
        for reg_name in key_registers:
            try:
                offset = state.arch.get_register_offset(reg_name)
                size = state.arch.registers.get(reg_name, (None, None))[1]
                if size is None:
                    continue

                reg_val = state.registers.load(offset, size=size)
                if not reg_val.symbolic:
                    # FAST PATH: Extract concrete value directly without solver
                    if reg_val.op == 'BVV':
                        concrete_val = reg_val.args[0]
                    else:
                        concrete_val = state.solver.eval(reg_val)
                    engine.set_register(reg_name, concrete_val)
            except (KeyError, AttributeError, errors.SimValueError):
                pass
            except Exception:
                pass

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

        # Start profiling if enabled
        if _profiler.enabled:
            block_start = time.perf_counter()

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
        if _profiler.enabled:
            t0 = time.perf_counter()
        self._sync_state_to_rust(state)
        if _profiler.enabled:
            _profiler.sync_to_rust_time += time.perf_counter() - t0

        # Lift the block if not provided
        if irsb is None:
            if _profiler.enabled:
                t0 = time.perf_counter()
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
            if _profiler.enabled:
                _profiler.lift_time += time.perf_counter() - t0

        # Store IRSB in artifacts
        successors.artifacts["irsb"] = irsb
        successors.artifacts["irsb_size"] = irsb.size
        successors.artifacts["irsb_direct_next"] = irsb.direct_next

        # Serialize IRSB and execute in Rust
        try:
            if _profiler.enabled:
                t0 = time.perf_counter()
            irsb_json = _serialize_irsb(irsb)
            if _profiler.enabled:
                _profiler.serialize_time += time.perf_counter() - t0
                t0 = time.perf_counter()
            event = self._rust_engine.execute_irsb_json(irsb_json)
            if _profiler.enabled:
                _profiler.execute_time += time.perf_counter() - t0

            # Handle the execution event
            if _profiler.enabled:
                t0 = time.perf_counter()
            needs_more = self._handle_rust_execution_event(event, state, successors)
            if _profiler.enabled:
                _profiler.sync_from_rust_time += time.perf_counter() - t0

            if needs_more:
                # Rust engine returned need_lift or similar - fall back to Python
                l.debug("Rust engine needs more processing at 0x%x, falling back to Python", addr)
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

            # Update profiler stats
            if _profiler.enabled:
                _profiler.block_count += 1
                _profiler.total_time += time.perf_counter() - block_start

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

    def process_successors_loop(
        self,
        successors: SimSuccessors,
        max_blocks: int = 100,
        **kwargs,
    ):
        """
        Process successors using the callback-based execution loop.

        This method uses Rust for the execution loop with Python callbacks
        for memory access, hooks, and syscalls. It can execute multiple
        blocks before returning to Python.

        Args:
            successors: SimSuccessors to populate.
            max_blocks: Maximum blocks to execute before returning.
            **kwargs: Additional arguments (ignored for now).
        """
        if not self.rust_engine_available:
            l.warning("Rust engine not available, falling back to single-step")
            return self.process_successors(successors, **kwargs)

        if not isinstance(successors.addr, int):
            l.warning("Non-concrete address, falling back to single-step")
            return self.process_successors(successors, **kwargs)

        if PythonCallbacks is None:
            l.warning("PythonCallbacks not available, falling back to single-step")
            return self.process_successors(successors, **kwargs)

        # Start profiling if enabled
        if _profiler.enabled:
            loop_start = time.perf_counter()

        # Mark this as a Rust VEX loop execution
        successors.sort = "RUST_VEX_LOOP"
        successors.description = "Rust VEX Loop"

        addr = successors.addr
        state = self.state

        # Setup state scratch
        state.history.recent_block_count = 1
        state.scratch.guard = claripy.true()
        state.scratch.sim_procedure = None
        state.scratch.bbl_addr = addr

        # Create callback object
        cbs = RustVEXCallbacks(state, self.project, self)

        try:
            # Setup Rust callbacks
            rust_cbs = cbs.setup_rust_callbacks()

            # Set callbacks on engine
            self._rust_engine.set_callbacks(rust_cbs)

            # Sync hooks to Rust engine
            if state.project is not None:
                for hook_addr in state.project._sim_procedures:
                    self._rust_engine.add_hook(hook_addr)

            # Sync state to Rust engine
            if _profiler.enabled:
                t0 = time.perf_counter()
            self._sync_state_to_rust(state)
            if _profiler.enabled:
                _profiler.sync_to_rust_time += time.perf_counter() - t0

            # Run the execution loop
            if _profiler.enabled:
                t0 = time.perf_counter()
            event = self._rust_engine.run_loop(max_blocks)
            if _profiler.enabled:
                _profiler.execute_time += time.perf_counter() - t0

            # Handle the result
            if _profiler.enabled:
                t0 = time.perf_counter()
            self._handle_loop_execution_event(event, state, successors)
            if _profiler.enabled:
                _profiler.sync_from_rust_time += time.perf_counter() - t0

            successors.processed = True

            # Update profiler stats
            if _profiler.enabled:
                _profiler.block_count += event.blocks_executed
                _profiler.total_time += time.perf_counter() - loop_start

        except Exception as e:
            l.warning("Rust VEX loop execution failed: %s, falling back", e)
            # Clear callbacks and fall back to single-step
            self._rust_engine.clear_callbacks()
            return self.process_successors(successors, **kwargs)
        finally:
            # Always clear callbacks after use
            self._rust_engine.clear_callbacks()

    def _handle_loop_execution_event(
        self,
        event: "LoopExecutionEvent",
        state: "SimState",
        successors: SimSuccessors,
    ) -> None:
        """
        Handle an execution event from the Rust run_loop.

        Args:
            event: The LoopExecutionEvent from Rust.
            state: The SimState being executed.
            successors: SimSuccessors to populate.
        """
        event_type = event.event_type

        # First, sync state from Rust
        self._sync_state_from_rust(state)

        if event_type == "max_blocks":
            # Reached max blocks - create a successor to continue
            next_addr = event.pc
            state.ip = next_addr
            successors.add_successor(
                state,
                next_addr,
                claripy.true(),
                "Ijk_Boring",
            )

        elif event_type == "max_deferred_forks":
            # Reached max deferred forks limit - create successor for current state
            # and handle deferred forks below
            next_addr = event.pc
            state.ip = next_addr
            successors.add_successor(
                state,
                next_addr,
                claripy.true(),
                "Ijk_Boring",
            )

        elif event_type == "block_end":
            # Normal block end
            next_addr = event.pc
            jumpkind = event.jumpkind or "Ijk_Boring"
            state.ip = next_addr
            successors.add_successor(
                state,
                next_addr,
                claripy.true(),
                jumpkind,
            )

        elif event_type == "hook":
            # Hook hit - let Python handle via HooksMixin
            hook_addr = event.addr
            state.ip = hook_addr
            successors.add_successor(
                state,
                hook_addr,
                claripy.true(),
                "Ijk_NoHook",
            )

        elif event_type == "syscall":
            # Syscall - return for Python handling
            jumpkind = event.jumpkind or "Ijk_Sys_syscall"
            successors.add_successor(
                state,
                state.ip,
                claripy.true(),
                jumpkind,
            )

        elif event_type == "symbolic_branch":
            # Symbolic branch - fork states
            true_target = event.true_target
            false_target = event.false_target

            # Create true branch successor
            true_state = state.copy()
            true_state.ip = true_target
            successors.add_successor(
                true_state,
                true_target,
                claripy.true(),  # TODO: track actual condition
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

        elif event_type == "need_lift":
            # Need to lift a block - shouldn't happen with callbacks
            l.warning("Unexpected need_lift event at 0x%x", event.addr)
            raise errors.SimEngineError(f"Unexpected need_lift at 0x{event.addr:x}")

        elif event_type == "error":
            # Error during execution
            error_msg = event.error or "Unknown error"
            raise errors.SimEngineError(f"Rust VEX engine error: {error_msg}")

        else:
            l.warning("Unknown loop execution event type: %s", event_type)
            raise errors.SimEngineError(f"Unknown event type: {event_type}")

        # Process deferred forks - create additional successors for unexplored paths
        # These are branches where Rust took one path and deferred the other
        if hasattr(event, 'deferred_forks') and event.deferred_forks:
            l.debug("Processing %d deferred forks", len(event.deferred_forks))
            for fork in event.deferred_forks:
                # Create a state for the unexplored branch
                fork_state = state.copy()
                fork_state.ip = fork.unexplored_target

                # Add constraint for the unexplored path
                # If we took true, the unexplored path has NOT(condition)
                # If we took false, the unexplored path has the condition
                # Note: Without full constraint tracking, we can't add the actual constraint
                # but we can still schedule the unexplored path for execution

                successors.add_successor(
                    fork_state,
                    fork.unexplored_target,
                    claripy.true(),  # TODO: track actual negated condition
                    "Ijk_Boring",
                )


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

    def execute_irsb_json(self, irsb_json: str) -> ExecutionEvent:
        """Execute a serialized IRSB JSON string."""
        return self._engine.execute_irsb_json(irsb_json)

    def execute_code(self, code: bytes, arch_name: str = None) -> ExecutionEvent:
        """
        Lift code with pyvex and execute in Rust.

        This is a convenience method that handles the full pipeline:
        1. Lift code with pyvex
        2. Serialize IRSB to JSON
        3. Execute in Rust
        """
        import pyvex
        import archinfo

        # Get architecture
        if arch_name is None:
            arch_name = self._engine.arch

        arch_mapping = {
            "x86": archinfo.ArchX86,
            "amd64": archinfo.ArchAMD64,
            "arm": archinfo.ArchARM,
            "arm64": archinfo.ArchAArch64,
            "mips32": archinfo.ArchMIPS32,
            "mips64": archinfo.ArchMIPS64,
        }
        arch_cls = arch_mapping.get(arch_name.lower())
        if arch_cls is None:
            raise ValueError(f"Unsupported architecture: {arch_name}")

        arch = arch_cls()

        # Lift with pyvex
        irsb = pyvex.lift(code, self._engine.pc, arch)

        # Serialize and execute
        irsb_json = _serialize_irsb(irsb)
        return self._engine.execute_irsb_json(irsb_json)

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
    "RustVEXCallbacks",
    "RustVEX",
    "RUST_ENGINE_AVAILABLE",
    "enable_profiling",
    "get_profiler",
    "RustVEXProfiler",
]
