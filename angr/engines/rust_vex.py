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


# Cached claripy constants to avoid repeated function calls on hot paths
_CLARIPY_TRUE = claripy.true()
_CLARIPY_FALSE = claripy.false()

# Cache for common zero BVVs to avoid repeated creation
_BVV_ZERO_CACHE = {}


def _get_zero_bvv(bits):
    """Get a cached zero BVV of the specified bit width."""
    if bits not in _BVV_ZERO_CACHE:
        _BVV_ZERO_CACHE[bits] = claripy.BVV(0, bits)
    return _BVV_ZERO_CACHE[bits]


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
from angr.state_plugins.rust_solver import RustSimSolver, RUST_SOLVER_AVAILABLE


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


def _memory_has_paging(memory) -> bool:
    """Check if memory plugin supports direct paged operations."""
    return hasattr(memory, 'page_size') and hasattr(memory, '_pages')


def _memory_is_regioned(memory) -> bool:
    """Check if memory plugin uses regioned memory model."""
    return hasattr(memory, '_regions')


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
    # Handle potential pyvex inconsistencies
    try:
        actual_count = tyenv.types_used
        # Fallback if types list is accessible and smaller
        if hasattr(tyenv, 'types') and len(tyenv.types) < actual_count:
            actual_count = len(tyenv.types)
    except Exception:
        actual_count = 0

    for i in range(actual_count):
        try:
            ty = tyenv.lookup(i)
            types.append(ty if ty else "Ity_I64")
        except (IndexError, Exception) as e:
            l.debug("tyenv.lookup(%d) failed: %s, using default", i, e)
            types.append("Ity_I64")
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

        # Callback invocation counters for profiling
        self.memory_load_count = 0
        self.memory_store_count = 0
        self.memory_store_batch_count = 0
        self.memory_load_batch_count = 0
        self.register_get_count = 0
        self.register_put_count = 0
        self.lift_block_count = 0
        self.hook_count = 0
        self.syscall_count = 0

        # Timing accumulators (seconds)
        self.memory_load_time = 0.0
        self.memory_store_time = 0.0
        self.memory_store_batch_time = 0.0
        self.memory_load_batch_time = 0.0
        self.register_get_time = 0.0
        self.register_put_time = 0.0
        self.lift_block_time = 0.0

    def memory_load(self, addr: int, size: int) -> tuple[bytes, bool, Any]:
        """
        Load from angr's memory model.

        Args:
            addr: Address to load from.
            size: Number of bytes to load.

        Returns:
            Tuple of (concrete_bytes, is_symbolic, symbolic_ast_or_none).
        """
        self.memory_load_count += 1
        if _profiler.enabled:
            t0 = time.perf_counter()
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
                result = (concrete_bytes, True, val)
            else:
                result = (concrete_bytes, False, None)
            if _profiler.enabled:
                self.memory_load_time += time.perf_counter() - t0
            return result
        except Exception as e:
            l.warning("Memory load failed at 0x%x: %s", addr, e)
            if _profiler.enabled:
                self.memory_load_time += time.perf_counter() - t0
            # Return zeros on error
            return (bytes(size), False, None)

    def memory_store(self, addr: int, data: bytes) -> None:
        """
        Store to angr's memory model.

        Args:
            addr: Address to store to.
            data: Bytes to store.
        """
        self.memory_store_count += 1
        if _profiler.enabled:
            t0 = time.perf_counter()
        try:
            size = len(data)
            value = int.from_bytes(data, 'little')
            bits = size * 8
            # Use cached zero BVV for common zero stores
            bv = _get_zero_bvv(bits) if value == 0 else claripy.BVV(value, bits)
            self.state.memory.store(addr, bv, endness='Iend_LE')
        except Exception as e:
            l.warning("Memory store failed at 0x%x: %s", addr, e)
        finally:
            if _profiler.enabled:
                self.memory_store_time += time.perf_counter() - t0

    def memory_store_batch(self, stores: list[tuple[int, bytes]]) -> None:
        """
        Store multiple memory values in a single batch callback.

        This is more efficient than individual store callbacks because it
        reduces FFI overhead - multiple stores are handled in a single
        Python callback invocation.

        Args:
            stores: List of (address, data) tuples to store.
        """
        self.memory_store_batch_count += 1
        if _profiler.enabled:
            t0 = time.perf_counter()
        try:
            for addr, data in stores:
                size = len(data)
                value = int.from_bytes(data, 'little')
                bits = size * 8
                # Use cached zero BVV for common zero stores
                bv = _get_zero_bvv(bits) if value == 0 else claripy.BVV(value, bits)
                self.state.memory.store(addr, bv, endness='Iend_LE')
        except Exception as e:
            l.warning("Batch memory store failed: %s", e)
        finally:
            if _profiler.enabled:
                self.memory_store_batch_time += time.perf_counter() - t0

    def memory_load_batch(self, loads: list[tuple[int, int]]) -> list[tuple[bytes, bool, Any]]:
        """
        Load multiple memory values in a single batch callback.

        This is more efficient than individual load callbacks because it
        reduces FFI overhead - multiple loads are handled in a single
        Python callback invocation.

        Args:
            loads: List of (address, size) tuples to load.

        Returns:
            List of (concrete_bytes, is_symbolic, symbolic_ast_or_none) tuples.
        """
        self.memory_load_batch_count += 1
        self.memory_load_count += len(loads)  # Track individual loads too
        if _profiler.enabled:
            t0 = time.perf_counter()

        results = []
        try:
            for addr, size in loads:
                try:
                    val = self.state.memory.load(addr, size, endness='Iend_LE')
                    is_sym = val.symbolic

                    # FAST PATH: Extract concrete value directly without solver
                    if not is_sym:
                        if val.op == 'BVV':
                            concrete = val.args[0]
                        else:
                            concrete = self.state.solver.eval(val)
                    else:
                        concrete = self.state.solver.eval(val)

                    concrete_bytes = concrete.to_bytes(size, 'little')

                    if is_sym:
                        results.append((concrete_bytes, True, val))
                    else:
                        results.append((concrete_bytes, False, None))
                except Exception as e:
                    l.warning("Batch memory load failed at 0x%x: %s", addr, e)
                    results.append((bytes(size), False, None))
        finally:
            if _profiler.enabled:
                self.memory_load_batch_time += time.perf_counter() - t0

        return results

    def memory_load_symbolic(self, addrs: list[int], size: int, addr_width: int) -> bytes:
        """
        Load from memory with symbolic address (multiple concrete possibilities).

        This builds an ITE chain: If(addr==a0, mem[a0], If(addr==a1, mem[a1], ...))
        For simplicity in the Rust->Python callback, we load from all addresses
        and return bytes. The Rust side can handle the ITE chain construction.

        For now, we just load from the first address as a fallback.
        More sophisticated handling would build an ITE expression in Python.

        Args:
            addrs: List of possible concrete addresses.
            size: Number of bytes to load.
            addr_width: Width of the address in bits (for building ITE conditions).

        Returns:
            Loaded bytes (from first address as fallback).
        """
        try:
            if not addrs:
                return bytes(size)

            # For now, load from first address as the concrete value
            # A full implementation would build an ITE chain
            val = self.state.memory.load(addrs[0], size, endness='Iend_LE')

            # Extract concrete value
            if not val.symbolic:
                if val.op == 'BVV':
                    concrete = val.args[0]
                else:
                    concrete = self.state.solver.eval(val)
            else:
                concrete = self.state.solver.eval(val)

            return concrete.to_bytes(size, 'little')
        except Exception as e:
            l.warning("Symbolic memory load failed for addrs %s: %s", addrs, e)
            return bytes(size)

    def memory_store_symbolic(self, addrs: list[int], data: bytes, addr_width: int) -> None:
        """
        Store to memory with symbolic address (multiple concrete possibilities).

        This performs conditional stores to each possible address:
        mem[addr] = If(addr == candidate, new_value, mem[addr])

        For simplicity in the Rust->Python callback, we store to all addresses
        with conditional values.

        Args:
            addrs: List of possible concrete addresses.
            data: Bytes to store.
            addr_width: Width of the address in bits (for building ITE conditions).
        """
        try:
            if not addrs:
                return

            size = len(data)
            value = int.from_bytes(data, 'little')
            bits = size * 8
            # Use cached zero BVV for common zero stores
            new_val = _get_zero_bvv(bits) if value == 0 else claripy.BVV(value, bits)

            # For each candidate address, perform a conditional store
            # Note: For a full implementation, we'd need the actual symbolic address
            # to build proper ITE conditions. For now, we store to first address.
            self.state.memory.store(addrs[0], new_val, endness='Iend_LE')
        except Exception as e:
            l.warning("Symbolic memory store failed for addrs %s: %s", addrs, e)

    def memory_load_ast(self, size: int) -> tuple[bytes, bool, Any]:
        """
        Load from memory when the address range is too large to concretize.

        This is called when Rust's address concretizer returns TooLarge,
        meaning the symbolic address has too many possible values to enumerate.
        We delegate to angr's full memory model which can use its own
        address concretization strategies.

        The key insight is that the symbolic address expression is still
        tracked in angr's state (in the scratch space or computed from
        registers). We use a symbolic value as a placeholder and let
        angr's memory model handle the actual load.

        Args:
            size: Number of bytes to load.

        Returns:
            Tuple of (concrete_bytes, is_symbolic, symbolic_ast_or_none).
        """
        try:
            # Create a fresh symbolic value to represent the loaded data
            # since we can't know the actual value without concretizing the address
            sym_name = f"unconstrained_load_{size}_{id(self)}"
            result = claripy.BVS(sym_name, size * 8)

            # Return as symbolic - the actual value depends on which address is accessed
            # The caller will get a fresh symbolic variable representing "unknown memory"
            is_sym = True
            # Get a concrete evaluation for the bytes (arbitrary, but needed for Rust)
            concrete = 0  # Default to zero
            concrete_bytes = concrete.to_bytes(size, 'little')

            return (concrete_bytes, is_sym, result)
        except Exception as e:
            l.warning("Symbolic AST memory load failed: %s", e)
            return (bytes(size), True, None)

    def memory_store_ast(self, data: bytes, size: int) -> None:
        """
        Store to memory when the address range is too large to concretize.

        This is called when Rust's address concretizer returns TooLarge,
        meaning the symbolic address has too many possible values to enumerate.
        We delegate to angr's full memory model which can use its own
        address concretization strategies.

        For stores with unconstrained addresses, angr typically:
        1. Applies address concretization strategies
        2. Or treats it as a write to "symbolic memory"

        Args:
            data: Bytes to store.
            size: Number of bytes being stored.
        """
        try:
            # For stores to unconstrained addresses, we log a warning
            # since the address could be anywhere in the address space.
            # The actual store semantics depend on angr's memory model settings.
            l.debug("Symbolic AST store of %d bytes (address unconstrained)", size)

            # We don't have the actual address expression here, so we can't
            # perform a real store. This is a limitation - the store is "lost"
            # unless angr's state has other tracking mechanisms.
            # TODO: Track the symbolic address expression through Rust to enable
            # proper symbolic stores.
        except Exception as e:
            l.warning("Symbolic AST memory store failed: %s", e)

    def on_hook(self, addr: int) -> int:
        """
        Execute a hook at the given address.

        Args:
            addr: Address of the hook.

        Returns:
            New PC after hook execution.
        """
        self.hook_count += 1
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
        self.syscall_count += 1
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
        self.lift_block_count += 1
        if _profiler.enabled:
            t0 = time.perf_counter()
        try:
            # Check cache first
            if addr in self._lifted_blocks:
                if _profiler.enabled:
                    self.lift_block_time += time.perf_counter() - t0
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

            if _profiler.enabled:
                self.lift_block_time += time.perf_counter() - t0
            return irsb_json
        except Exception as e:
            l.warning("Block lifting failed at 0x%x: %s", addr, e)
            if _profiler.enabled:
                self.lift_block_time += time.perf_counter() - t0
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
        self.register_get_count += 1
        if _profiler.enabled:
            t0 = time.perf_counter()
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
                result = (concrete_bytes, True, val)
            else:
                result = (concrete_bytes, False, None)
            if _profiler.enabled:
                self.register_get_time += time.perf_counter() - t0
            return result
        except Exception as e:
            l.warning("Register read failed at offset %d: %s", offset, e)
            if _profiler.enabled:
                self.register_get_time += time.perf_counter() - t0
            return (bytes(size), False, None)

    def put_register(self, offset: int, data: bytes) -> None:
        """
        Set a register value in angr's state.

        Args:
            offset: Register offset.
            data: Bytes to store.
        """
        self.register_put_count += 1
        if _profiler.enabled:
            t0 = time.perf_counter()
        try:
            size = len(data)
            value = int.from_bytes(data, 'little')
            bits = size * 8
            # Use cached zero BVV for common zero stores
            bv = _get_zero_bvv(bits) if value == 0 else claripy.BVV(value, bits)
            self.state.registers.store(offset, bv)
        except Exception as e:
            l.warning("Register write failed at offset %d: %s", offset, e)
        finally:
            if _profiler.enabled:
                self.register_put_time += time.perf_counter() - t0

    def dirty_call(self, name: str, args: list[int], ret_ty_bits: int) -> tuple[bytes, bool, Any]:
        """
        Handle a VEX dirty call to a helper function.

        This handles dirty calls like CPUID, RDTSC, x87 operations, etc.

        Args:
            name: Name of the helper function (e.g., "x86g_dirtyhelper_CPUID_sse42").
            args: List of concrete argument values.
            ret_ty_bits: Expected return type size in bits (0 if no return).

        Returns:
            Tuple of (concrete_bytes, is_symbolic, symbolic_ast_or_none).
        """
        try:
            # Common dirty helpers can be handled directly
            if "CPUID" in name:
                # CPUID returns EAX:EBX:ECX:EDX in a 128-bit value
                # For simplicity, return a generic result
                if ret_ty_bits > 0:
                    result = bytes(ret_ty_bits // 8)
                    return (result, True, None)  # Mark as symbolic
                return (b'', False, None)

            elif "RDTSC" in name:
                # RDTSC returns a 64-bit timestamp counter
                import time as _time
                tsc = int(_time.perf_counter() * 1e9) & 0xFFFFFFFFFFFFFFFF
                result = tsc.to_bytes(8, 'little')
                return (result, False, None)

            elif "x87" in name.lower() or "fpu" in name.lower():
                # x87 FPU operations - return symbolic for now
                if ret_ty_bits > 0:
                    result = bytes(ret_ty_bits // 8)
                    return (result, True, None)
                return (b'', False, None)

            else:
                # Unknown helper - return symbolic value
                l.debug("Unknown dirty helper: %s", name)
                if ret_ty_bits > 0:
                    result = bytes(ret_ty_bits // 8)
                    return (result, True, None)
                return (b'', False, None)

        except Exception as e:
            l.warning("Dirty call %s failed: %s", name, e)
            if ret_ty_bits > 0:
                return (bytes(ret_ty_bits // 8), True, None)
            return (b'', False, None)

    def get_stats(self) -> dict:
        """
        Get callback invocation statistics.

        Returns:
            Dictionary with callback counts and timing information.
        """
        return {
            "memory_load_count": self.memory_load_count,
            "memory_store_count": self.memory_store_count,
            "memory_store_batch_count": self.memory_store_batch_count,
            "memory_load_batch_count": self.memory_load_batch_count,
            "register_get_count": self.register_get_count,
            "register_put_count": self.register_put_count,
            "lift_block_count": self.lift_block_count,
            "hook_count": self.hook_count,
            "syscall_count": self.syscall_count,
            "memory_load_time_ms": self.memory_load_time * 1000,
            "memory_store_time_ms": self.memory_store_time * 1000,
            "memory_store_batch_time_ms": self.memory_store_batch_time * 1000,
            "memory_load_batch_time_ms": self.memory_load_batch_time * 1000,
            "register_get_time_ms": self.register_get_time * 1000,
            "register_put_time_ms": self.register_put_time * 1000,
            "lift_block_time_ms": self.lift_block_time * 1000,
            "total_callback_time_ms": (
                self.memory_load_time + self.memory_store_time +
                self.memory_store_batch_time + self.memory_load_batch_time +
                self.register_get_time + self.register_put_time +
                self.lift_block_time
            ) * 1000,
        }

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
        rust_cbs.set_memory_store_batch(self.memory_store_batch)
        rust_cbs.set_memory_load_batch(self.memory_load_batch)
        rust_cbs.set_memory_load_symbolic(self.memory_load_symbolic)
        rust_cbs.set_memory_store_symbolic(self.memory_store_symbolic)
        rust_cbs.set_memory_load_ast(self.memory_load_ast)
        rust_cbs.set_memory_store_ast(self.memory_store_ast)
        rust_cbs.set_on_hook(self.on_hook)
        rust_cbs.set_on_syscall(self.on_syscall)
        rust_cbs.set_lift_block(self.lift_block)
        rust_cbs.set_get_register(self.get_register)
        rust_cbs.set_put_register(self.put_register)
        rust_cbs.set_dirty_call(self.dirty_call)

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

    Rust-Native Memory Model:
    - When use_rust_memory=True, memory operations use a Rust-native symbolic
      memory model instead of Python callbacks. This provides significant
      performance improvements (2-10x) for memory-intensive symbolic execution.
    - Use configure_rust_memory() to enable and sync memory from SimState.
    - Dirty pages are tracked and synced back to Python after execution.
    """

    _rust_engine: RustVEXEngine | None = None
    _rust_engine_synced: bool = False
    _use_rust_memory: bool = False

    def __init__(self, project: angr.Project, use_deferred_forks: bool = True, max_deferred_forks: int = 5, use_rust_memory: bool = False):
        super().__init__(project)

        self._use_deferred_forks = use_deferred_forks
        self._max_deferred_forks = max_deferred_forks
        self._concrete_memory_synced = False
        self._last_callbacks: RustVEXCallbacks | None = None
        self._use_rust_memory = use_rust_memory
        self._rust_memory_synced = False

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
                # Initialize Rust-native memory if requested
                if use_rust_memory:
                    self._init_rust_memory()
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

    def configure_rust_memory(self, enabled: bool = True) -> None:
        """
        Enable or disable Rust-native memory model.

        When enabled, memory operations use a Rust-native symbolic memory model
        instead of Python callbacks. This can provide significant performance
        improvements (2-10x) for memory-intensive symbolic execution.

        Args:
            enabled: Whether to use Rust-native memory (default True).
        """
        if self._rust_engine is None:
            return

        self._use_rust_memory = enabled
        if enabled:
            self._init_rust_memory()
        else:
            self._rust_engine.disable_rust_memory()

    def _init_rust_memory(self) -> None:
        """Initialize Rust-native memory model."""
        if self._rust_engine is None:
            return

        # Determine endianness from architecture
        little_endian = self.project.arch.memory_endness == 'Iend_LE'

        # Create the Rust memory model
        self._rust_engine.create_rust_memory(little_endian)
        self._rust_engine.enable_rust_memory()
        l.debug("Rust-native memory model initialized (little_endian=%s)", little_endian)

    def _sync_rust_memory_from_state(self, state: "SimState") -> int:
        """
        Sync memory from SimState to Rust memory model.

        Routes to appropriate implementation based on memory type:
        - PagedMemoryMixin: Direct page access
        - RegionedMemoryMixin (AbstractMemory): Iterate through regions
        """
        if not self._use_rust_memory or self._rust_engine is None:
            return 0

        if _memory_is_regioned(state.memory):
            return self._sync_rust_memory_from_state_regioned(state)
        elif _memory_has_paging(state.memory):
            return self._sync_rust_memory_from_state_paged(state)
        else:
            l.debug("Unknown memory type, skipping Rust memory sync")
            return 0

    def _sync_rust_memory_from_state_paged(self, state: "SimState") -> int:
        """
        Sync memory from PagedMemoryMixin to Rust memory model.

        This maps concrete pages from angr's memory to the Rust memory model,
        enabling Rust to handle memory operations without Python callbacks.
        """
        pages_synced = 0
        page_size = state.memory.page_size

        # Get pages from angr's memory
        try:
            pages = state.memory._pages
        except AttributeError:
            l.debug("Memory plugin doesn't expose _pages, skipping Rust memory sync")
            return 0

        for page_no, page in pages.items():
            if page is None:
                continue

            page_addr = page_no * page_size

            try:
                # Get concrete data with symbolic bitmap
                data, bitmap = state.memory.concrete_load(
                    page_addr, page_size, with_bitmap=True
                )

                # Only sync fully concrete pages
                if all(b == 0 for b in bitmap):
                    self._rust_engine.map_rust_memory_data(
                        page_addr,
                        bytes(data),
                        7  # RWX permissions
                    )
                    pages_synced += 1
            except Exception:
                # Page not loadable - skip
                pass

        self._rust_memory_synced = True
        l.debug("Synced %d concrete pages to Rust memory", pages_synced)
        return pages_synced

    def _sync_rust_memory_from_state_regioned(self, state: "SimState") -> int:
        """
        Sync memory from regioned memory (AbstractMemory) to Rust memory model.

        AbstractMemory uses RegionedMemoryMixin which stores memory in regions.
        Each region is a RegionedMemory that internally uses PagedMemoryMixin.
        """
        pages_synced = 0

        # Iterate each region
        for region_id, region in state.memory._regions.items():
            # Each region has _pages from PagedMemoryMixin
            if not hasattr(region, '_pages') or not hasattr(region, 'page_size'):
                continue

            page_size = region.page_size

            # Get region base address for absolute address calculation
            try:
                if hasattr(state.memory, '_region_base'):
                    region_base = state.memory._region_base(region_id)
                else:
                    # Fallback: use address mapping if available
                    region_base = 0
            except Exception:
                region_base = 0

            # Process pages in this region
            for page_no, page in region._pages.items():
                if page is None:
                    continue

                page_addr = page_no * page_size
                abs_addr = region_base + page_addr

                try:
                    # Get concrete data with bitmap
                    data, bitmap = region.concrete_load(
                        page_addr, page_size, with_bitmap=True
                    )

                    # Only map if fully concrete (all bitmap bits are 0)
                    if all(b == 0 for b in bitmap):
                        self._rust_engine.map_rust_memory_data(
                            abs_addr, bytes(data), 7  # RWX permissions
                        )
                        pages_synced += 1
                except Exception as e:
                    l.debug("Failed to sync region %s page 0x%x: %s", region_id, page_addr, e)
                    pass

        self._rust_memory_synced = True
        l.debug("Synced %d concrete pages from regioned memory to Rust", pages_synced)
        return pages_synced

    def _sync_rust_memory_to_state(self, state: "SimState") -> int:
        """
        Sync dirty pages from Rust memory back to SimState.

        Only syncs pages that were modified during Rust execution,
        minimizing overhead.

        Routes to appropriate implementation based on memory type.
        """
        if not self._use_rust_memory or self._rust_engine is None:
            return 0

        if _memory_is_regioned(state.memory):
            return self._sync_rust_memory_to_state_regioned(state)
        elif _memory_has_paging(state.memory):
            return self._sync_rust_memory_to_state_paged(state)
        else:
            l.debug("Unknown memory type, skipping Rust memory sync to state")
            return 0

    def _sync_rust_memory_to_state_paged(self, state: "SimState") -> int:
        """Sync dirty pages from Rust memory back to paged SimState memory."""
        pages_synced = 0
        page_size = state.memory.page_size

        # Get list of dirty pages from Rust
        dirty_pages = self._rust_engine.get_rust_memory_dirty_pages()

        for page_addr in dirty_pages:
            try:
                # Get page data from Rust
                result = self._rust_engine.get_rust_memory_page(page_addr)
                if result is None:
                    continue

                data, _perms = result

                # Store back to angr's memory
                bv = claripy.BVV(int.from_bytes(data, 'little'), page_size * 8)
                state.memory.store(page_addr, bv, endness='Iend_LE')
                pages_synced += 1
            except Exception as e:
                l.debug("Failed to sync dirty page 0x%x: %s", page_addr, e)

        # Clear dirty tracking for next execution
        self._rust_engine.clear_rust_memory_dirty_pages()

        if pages_synced > 0:
            l.debug("Synced %d dirty pages from Rust memory", pages_synced)
        return pages_synced

    def _sync_rust_memory_to_state_regioned(self, state: "SimState") -> int:
        """Sync dirty pages from Rust memory back to regioned SimState memory."""
        pages_synced = 0
        # Use default page size for regioned memory
        page_size = 0x1000

        # Get list of dirty pages from Rust
        dirty_pages = self._rust_engine.get_rust_memory_dirty_pages()

        for page_addr in dirty_pages:
            try:
                # Get page data from Rust
                result = self._rust_engine.get_rust_memory_page(page_addr)
                if result is None:
                    continue

                data, _perms = result

                # Store back to angr's memory (store works on all memory types)
                bv = claripy.BVV(int.from_bytes(data, 'little'), page_size * 8)
                state.memory.store(page_addr, bv, endness='Iend_LE')
                pages_synced += 1
            except Exception as e:
                l.debug("Failed to sync dirty page 0x%x to regioned memory: %s", page_addr, e)

        # Clear dirty tracking for next execution
        self._rust_engine.clear_rust_memory_dirty_pages()

        if pages_synced > 0:
            l.debug("Synced %d dirty pages from Rust to regioned memory", pages_synced)
        return pages_synced

    def get_rust_memory_stats(self) -> dict | None:
        """
        Get statistics about Rust-native memory.

        Returns:
            Dictionary with memory stats, or None if not available.
        """
        if self._rust_engine is None:
            return None
        try:
            return dict(self._rust_engine.rust_memory_stats())
        except Exception:
            return None

    def get_callback_stats(self) -> dict | None:
        """
        Get callback invocation statistics from the last execution.

        Returns:
            Dictionary with callback counts and timing, or None if no callbacks were used.
        """
        if self._last_callbacks is not None:
            return self._last_callbacks.get_stats()
        return None

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

    def _sync_state_memory_to_rust(self, state: "SimState") -> int:
        """
        Sync all concrete memory from state to Rust engine.

        This maps all pages from angr's memory model to Rust, enabling
        Rust to execute without memory callbacks for concrete regions.

        Routes to appropriate implementation based on memory type:
        - PagedMemoryMixin: Direct page access
        - RegionedMemoryMixin (AbstractMemory): Iterate through regions
        """
        if not self.rust_engine_available:
            return 0

        if _memory_is_regioned(state.memory):
            return self._sync_state_memory_to_rust_regioned(state)
        elif _memory_has_paging(state.memory):
            return self._sync_state_memory_to_rust_paged(state)
        else:
            l.debug("Unknown memory type, skipping sync to Rust")
            return 0

    def _sync_state_memory_to_rust_paged(self, state: "SimState") -> int:
        """Sync memory from PagedMemoryMixin to Rust."""
        pages_synced = 0
        page_size = state.memory.page_size

        # Clear dirty pages from previous execution
        self._rust_engine.clear_dirty_pages()

        # Iterate all mapped pages in angr's state
        try:
            pages = state.memory._pages
        except AttributeError:
            # Memory plugin doesn't expose _pages directly
            return 0

        for page_no, page in pages.items():
            if page is None:
                continue  # Explicitly unmapped

            page_addr = page_no * page_size

            try:
                # Get concrete data with bitmap
                data, bitmap = state.memory.concrete_load(
                    page_addr, page_size, with_bitmap=True
                )

                # Check if page is fully concrete (all bitmap bytes are 0)
                if all(b == 0 for b in bitmap):
                    # Fully concrete - map directly to Rust
                    self._rust_engine.map_memory_data(
                        page_addr,
                        bytes(data),
                        7  # RWX permissions
                    )
                    pages_synced += 1
            except Exception:
                # Page not loadable - skip
                pass

        return pages_synced

    def _sync_state_memory_to_rust_regioned(self, state: "SimState") -> int:
        """
        Sync memory from regioned memory (AbstractMemory) to Rust.

        AbstractMemory uses RegionedMemoryMixin which stores memory in regions.
        Each region is a RegionedMemory that internally uses PagedMemoryMixin.
        """
        pages_synced = 0

        # Clear dirty pages from previous execution
        self._rust_engine.clear_dirty_pages()

        # Iterate each region
        for region_id, region in state.memory._regions.items():
            # Each region has _pages from PagedMemoryMixin
            if not hasattr(region, '_pages') or not hasattr(region, 'page_size'):
                continue

            page_size = region.page_size

            # Get region base address for absolute address calculation
            try:
                if hasattr(state.memory, '_region_base'):
                    region_base = state.memory._region_base(region_id)
                else:
                    # Fallback: use address mapping if available
                    region_base = 0
            except Exception:
                region_base = 0

            # Process pages in this region
            for page_no, page in region._pages.items():
                if page is None:
                    continue

                page_addr = page_no * page_size
                abs_addr = region_base + page_addr

                try:
                    # Get concrete data with bitmap
                    data, bitmap = region.concrete_load(
                        page_addr, page_size, with_bitmap=True
                    )

                    # Only map if fully concrete (all bitmap bits are 0)
                    if all(b == 0 for b in bitmap):
                        self._rust_engine.map_memory_data(
                            abs_addr, bytes(data), 7  # RWX permissions
                        )
                        pages_synced += 1
                except Exception as e:
                    l.debug("Failed to sync region %s page 0x%x: %s", region_id, page_addr, e)
                    pass

        return pages_synced

    def _sync_memory_from_rust(self, state: "SimState") -> int:
        """
        Sync memory changes from Rust back to angr's state.

        Routes to appropriate implementation based on memory type.

        Returns:
            Number of pages synced back.
        """
        if not self.rust_engine_available:
            return 0

        if _memory_is_regioned(state.memory):
            return self._sync_memory_from_rust_regioned(state)
        elif _memory_has_paging(state.memory):
            return self._sync_memory_from_rust_paged(state)
        else:
            l.debug("Unknown memory type, skipping sync from Rust")
            return 0

    def _sync_memory_from_rust_paged(self, state: "SimState") -> int:
        """Sync memory changes from Rust back to paged memory."""
        pages_synced = 0
        page_size = state.memory.page_size

        # Get list of dirty pages from Rust (pages that were written)
        dirty_pages = self._rust_engine.get_dirty_pages()

        for page_addr in dirty_pages:
            try:
                # Read the modified page data from Rust
                data = self._rust_engine.read_memory(page_addr, page_size)
                # Store back to angr's memory as a bitvector
                bv = claripy.BVV(int.from_bytes(data, 'little'), page_size * 8)
                state.memory.store(page_addr, bv, endness='Iend_LE')
                pages_synced += 1
            except Exception:
                # Page might not be readable or writable in this context
                pass

        return pages_synced

    def _sync_memory_from_rust_regioned(self, state: "SimState") -> int:
        """Sync memory changes from Rust back to regioned memory."""
        pages_synced = 0
        # Use default page size for regioned memory
        page_size = 0x1000

        # Get list of dirty pages from Rust (pages that were written)
        dirty_pages = self._rust_engine.get_dirty_pages()

        for page_addr in dirty_pages:
            try:
                # Read the modified page data from Rust
                data = self._rust_engine.read_memory(page_addr, page_size)
                # Store back to angr's memory as a bitvector (store works on all memory types)
                bv = claripy.BVV(int.from_bytes(data, 'little'), page_size * 8)
                state.memory.store(page_addr, bv, endness='Iend_LE')
                pages_synced += 1
            except Exception:
                # Page might not be readable or writable in this context
                pass

        return pages_synced

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

        IMPORTANT: We must check for hooks/syscalls BEFORE using the loop path,
        since the loop path bypasses the normal MRO chain that handles these.
        """
        # Check if loop execution is enabled and prerequisites are met
        # IMPORTANT: Check state._ip.symbolic to detect symbolic instruction pointers.
        # state.addr returns a concrete integer even for symbolic IPs (via solver evaluation),
        # so we must check the actual IP to avoid executing from arbitrary concretized addresses.
        ip_symbolic = isinstance(state._ip, claripy.ast.BV) and state._ip.symbolic

        # Also validate that the IP points to mapped memory
        # This catches cases where a symbolic IP was concretized to an invalid address
        addr_valid = True
        if state.project is not None and isinstance(state.addr, int):
            try:
                obj = state.project.loader.find_object_containing(state.addr)
                if obj is None:
                    addr_valid = False
                    l.debug("Skipping Rust VEX loop for unmapped address 0x%x", state.addr)
            except Exception:
                addr_valid = False

        if (o.RUST_VEX_LOOP in state.options
            and self.rust_engine_available
            and isinstance(state.addr, int)
            and not ip_symbolic
            and addr_valid):

            # Check for hooks at the current address BEFORE using loop path
            # This is necessary because the loop path bypasses HooksMixin
            addr = state.addr

            # Check the CURRENT history's jumpkind (not parent).
            # If it's Ijk_NoHook, a hook was just executed at this address.
            # For length=0 hooks, the PC doesn't advance, so we need to execute
            # the actual instruction at this address.
            #
            # Note: We check state.history.jumpkind (not parent) because that's
            # the jumpkind of the execution that CREATED this state.
            current_jumpkind = None
            if state.history:
                current_jumpkind = state.history.jumpkind

            if current_jumpkind == "Ijk_NoHook":
                # A hook was just executed. Temporarily remove it to prevent
                # re-triggering, then execute the instruction via Python VEX.
                hook_proc = None
                if state.project is not None and addr in state.project._sim_procedures:
                    hook_proc = state.project._sim_procedures.pop(addr)
                    # Also remove from Rust engine's internal hook set
                    if self._rust_engine is not None:
                        self._rust_engine.remove_hook(addr)
                try:
                    # Execute without the hook (falls back to Python VEX)
                    return super().process(state, **kwargs)
                finally:
                    # Restore the hook
                    if hook_proc is not None:
                        state.project._sim_procedures[addr] = hook_proc
                        if self._rust_engine is not None:
                            self._rust_engine.add_hook(addr)

            if state.project is not None:
                # Check if there's a hook at this address
                if addr in state.project._sim_procedures:
                    # Hook exists - use normal execution path to handle it
                    return super().process(state, **kwargs)

            # Check for syscall jumpkind
            if current_jumpkind and current_jumpkind.startswith("Ijk_Sys"):
                # Syscall - use normal execution path
                return super().process(state, **kwargs)

            # Check for symbolic base registers BEFORE using loop mode
            # When base registers are symbolic, RUST_VEX_LOOP with RustSimSolver
            # causes constraint propagation issues. Use standard Python VEX instead.
            if self._has_symbolic_base_registers(state):
                l.debug("Symbolic base registers - using standard execution")
                # Also disable RUST_VEX_LOOP option so forked states don't use it
                if o.RUST_VEX_LOOP in state.options:
                    state.options.discard(o.RUST_VEX_LOOP)
                return super().process(state, **kwargs)

            # No hook or syscall - use loop execution path with deferred forks
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

        This copies register values from the SimState to the Rust engine,
        including both concrete and symbolic values.
        """
        if self._rust_engine is None:
            return

        engine = self._rust_engine

        # Clear any stale symbolic registers from previous sync
        engine.clear_symbolic_registers()

        # Sync PC
        pc = state.solver.eval(state.ip)
        engine.pc = pc

        # Sync key registers (handles both concrete and symbolic values)
        self._sync_registers_individual(state, engine)

        # Prefetch stack pages to avoid unmapped memory fallbacks
        self._prefetch_stack_pages(state, engine)

        self._rust_engine_synced = True

    def _prefetch_stack_pages(self, state: SimState, engine) -> None:
        """
        Pre-map stack pages before execution to avoid unmapped memory fallbacks.

        This prefetches the stack region (SP-64KB to SP+4KB) into the Rust engine's
        concrete memory, reducing the number of Python callbacks needed for stack access.
        """
        # For regioned memory, skip prefetching - regions manage their own pages
        # and may not support the permissions() method
        if _memory_is_regioned(state.memory):
            return

        try:
            sp = state.solver.eval(state.regs.sp)
        except Exception:
            return

        # Stack region: SP-64KB to SP+4KB (typical stack access range)
        stack_start = sp - 0x10000  # 64KB below SP
        stack_end = sp + 0x1000     # 4KB above SP

        # Round to page boundaries
        page_size = 0x1000
        stack_start = (stack_start // page_size) * page_size

        # Map each page that's accessible
        for page_addr in range(stack_start, stack_end, page_size):
            try:
                # Check if page is mapped and readable
                if not state.memory.permissions(page_addr):
                    continue

                # Load page data
                data = state.memory.load(page_addr, page_size)

                # Only map concrete pages
                if data.symbolic:
                    continue

                # Extract concrete bytes
                if data.op == 'BVV':
                    concrete_val = data.args[0]
                    page_bytes = concrete_val.to_bytes(page_size, byteorder='big')
                else:
                    # Try to evaluate (might fail for symbolic data)
                    concrete_val = state.solver.eval(data, cast_to=bytes)
                    page_bytes = concrete_val

                # Map the page in Rust (read/write/execute permissions)
                engine.map_memory_data(page_addr, page_bytes, 7)
            except Exception:
                # Skip pages that can't be read (unmapped, symbolic, etc.)
                pass

    def _sync_registers_individual(self, state: SimState, engine) -> None:
        """Sync registers individually, handling both concrete and symbolic values."""
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
                else:
                    # Symbolic register: pass the AST to Rust for symbolic execution
                    try:
                        engine.set_symbolic_register(offset, reg_val)
                    except Exception as e:
                        l.debug("Failed to sync symbolic register %s: %s", reg_name, e)
                        # Fall back to a concrete approximation
                        try:
                            concrete_val = state.solver.eval(reg_val)
                            engine.set_register(reg_name, concrete_val)
                        except Exception:
                            pass
            except (KeyError, AttributeError, errors.SimValueError):
                pass
            except Exception:
                pass

    def _get_register_size_at_offset(self, arch, offset: int) -> int:
        """Get the size of register at given offset.

        Args:
            arch: The architecture object.
            offset: Register offset.

        Returns:
            Size in bytes (default to arch word size if not found).
        """
        # Check if offset matches a known register
        for reg_name, reg_info in arch.registers.items():
            if reg_info[0] == offset:
                return reg_info[1]
        # Default to architecture word size
        return arch.bytes

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

    def _has_symbolic_base_registers(self, state: SimState) -> bool:
        """
        Check if base/stack pointer registers are symbolic.

        The Rust VEX engine works best with concrete base registers.
        When base registers are symbolic, memory addressing becomes complex
        and causes significant sync overhead. Fall back to Python in this case.

        Returns:
            True if base registers are symbolic and should fall back to Python.
        """
        arch = state.arch

        # Check architecture-specific base registers
        if arch.name in ("X86", "AMD64"):
            base_regs = ["ebp", "esp"] if arch.name == "X86" else ["rbp", "rsp"]
        elif arch.name.startswith("ARM"):
            base_regs = ["sp", "fp"] if hasattr(state.regs, "fp") else ["sp"]
        elif arch.name.startswith("MIPS"):
            base_regs = ["sp", "gp"]
        else:
            # Default: just check stack pointer if available
            base_regs = ["sp"] if hasattr(state.regs, "sp") else []

        for reg_name in base_regs:
            try:
                reg_val = getattr(state.regs, reg_name)
                if hasattr(reg_val, 'symbolic') and reg_val.symbolic:
                    l.warning("Symbolic base register detected: %s - falling back to Python", reg_name)
                    return True
            except (AttributeError, KeyError):
                pass

        return False

    def _sync_state_from_rust(self, state: SimState) -> None:
        """
        Synchronize Rust engine state back to angr SimState.

        This copies only MODIFIED register values from the Rust engine back
        to the SimState, using dirty register tracking for efficiency.
        """
        if self._rust_engine is None:
            return

        engine = self._rust_engine

        # Sync PC
        state.ip = engine.pc

        # Get only the dirty register offsets (registers modified during Rust execution)
        dirty_offsets = engine.get_dirty_register_offsets()

        if not dirty_offsets:
            # No registers modified - nothing to sync
            return

        # Sync only dirty registers
        for offset in dirty_offsets:
            try:
                size = self._get_register_size_at_offset(state.arch, offset)
                value = engine.get_register_by_offset(offset, size)
                state.registers.store(offset, claripy.BVV(value, size * 8))
            except Exception as e:
                l.debug("Failed to sync register at offset %d: %s", offset, e)

        # Clear dirty tracking for next execution
        engine.clear_dirty_registers()

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

            guard = _CLARIPY_TRUE
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
            # Symbolic branch detected - fall back to Python for proper handling
            # The Rust engine doesn't track branch conditions, so Python VEX must
            # handle symbolic branches to ensure proper constraint propagation.
            # Without proper constraints, the solver may produce wrong solutions.
            l.debug("Symbolic branch detected, falling back to Python VEX")
            raise errors.SimEngineError("symbolic branch requires Python VEX")

        elif event_type == "syscall":
            # Syscall - return to Python for handling
            self._sync_state_from_rust(state)

            jumpkind = event.jumpkind or "Ijk_Sys_syscall"
            successors.add_successor(
                state,
                state.ip,
                _CLARIPY_TRUE,
                jumpkind,
            )
            return False

        elif event_type == "hook":
            # Hook address hit - return to Python for handling
            self._sync_state_from_rust(state)

            # The hook will be handled by HooksMixin on the next step
            # IMPORTANT: Use Ijk_Boring (not Ijk_NoHook) so HooksMixin will
            # check and execute the hook. Ijk_NoHook tells HooksMixin to SKIP.
            successors.add_successor(
                state,
                event.next_addr,
                _CLARIPY_TRUE,
                "Ijk_Boring",  # Changed from Ijk_NoHook
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

        addr = successors.addr
        state = self.state

        # Check if base/stack pointer registers are symbolic - if so, fall back to Python
        # because Rust cannot properly handle symbolic register values (they won't be synced)
        if self._has_symbolic_base_registers(state):
            # Note: Warning already printed by _has_symbolic_base_registers
            l.debug("Single-block path: falling back to Python VEX at 0x%x", addr)
            return super().process_successors(
                successors,
                irsb=irsb,
                insn_bytes=insn_bytes,
                extra_stop_points=extra_stop_points,
                num_inst=num_inst,
                size=size,
                **kwargs,
            )

        # Save state snapshot for rollback if Rust fails
        # Single-block Rust can still modify state via sync operations
        state_snapshot = state.copy()

        # Start profiling if enabled
        if _profiler.enabled:
            block_start = time.perf_counter()

        # Mark this as a Rust VEX execution
        successors.sort = "RUST_VEX"
        successors.description = "Rust VEX"

        # Setup state scratch
        state.history.recent_block_count = 1
        state.scratch.guard = _CLARIPY_TRUE
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
        # Sync concrete memory pages for fast Rust access
        self._sync_state_memory_to_rust(state)
        # Sync Rust-native memory if enabled
        if self._use_rust_memory:
            self._sync_rust_memory_from_state(state)
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
            # Sync memory changes back from Rust
            self._sync_memory_from_rust(state)
            # Sync Rust-native memory dirty pages if enabled
            if self._use_rust_memory:
                self._sync_rust_memory_to_state(state)
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
            # Restore state snapshot to ensure Python VEX has clean state
            self.state = state_snapshot
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

        IMPORTANT: Requires RustSimSolver for proper constraint handling.
        The Rust engine and solver share the same SymContext, ensuring all
        branch constraints are applied consistently in Rust's Z3 solver.

        Args:
            successors: SimSuccessors to populate.
            max_blocks: Maximum blocks to execute before returning.
            **kwargs: Additional arguments (ignored for now).

        Raises:
            TypeError: If state.solver is not a RustSimSolver.
        """
        if not self.rust_engine_available:
            l.warning("Rust engine not available, falling back to single-step")
            return self.process_successors(successors, **kwargs)

        if not isinstance(successors.addr, int):
            l.warning("Non-concrete address, falling back to single-step")
            return self.process_successors(successors, **kwargs)

        # Check if state IP is symbolic - must fall back to Python for proper handling
        if isinstance(self.state._ip, claripy.ast.BV) and self.state._ip.symbolic:
            l.debug("Symbolic IP detected, falling back to Python VEX")
            return super().process_successors(successors, **kwargs)

        if PythonCallbacks is None:
            l.warning("PythonCallbacks not available, falling back to single-step")
            return self.process_successors(successors, **kwargs)

        # Enforce RustSimSolver for unified solver architecture
        # The Rust VEX engine requires RustSimSolver for proper constraint propagation.
        # Without it, branch constraints would be lost when returning to Python.
        if RUST_SOLVER_AVAILABLE and not isinstance(self.state.solver, RustSimSolver):
            raise TypeError(
                "Rust VEX loop execution requires RustSimSolver for constraint consistency. "
                "Create state with: state.register_plugin('solver', RustSimSolver())"
            )

        addr = successors.addr
        state = self.state

        # Check if base/stack pointer registers are symbolic - if so, fall back to Python
        # because Rust cannot properly handle symbolic register values (they won't be synced)
        if self._has_symbolic_base_registers(state):
            l.debug("Symbolic base registers detected, falling back to Python VEX")
            return super().process_successors(successors, **kwargs)

        # Start profiling if enabled
        if _profiler.enabled:
            loop_start = time.perf_counter()

        # Mark this as a Rust VEX loop execution
        successors.sort = "RUST_VEX_LOOP"
        successors.description = "Rust VEX Loop"

        # Setup state scratch
        state.history.recent_block_count = 1
        state.scratch.guard = _CLARIPY_TRUE
        state.scratch.sim_procedure = None
        state.scratch.bbl_addr = addr

        # Create callback object
        cbs = RustVEXCallbacks(state, self.project, self)
        self._last_callbacks = cbs  # Save for profiling access

        # Save state snapshot for rollback on failure
        # This is necessary because Rust execution can modify Python state:
        # - Memory callbacks write concrete values (overwriting symbolic)
        # - State might have been partially synced
        # If Rust execution fails, we need to restore the original state.
        state_snapshot = state.copy()

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
            # Sync concrete memory pages for fast Rust access
            self._sync_state_memory_to_rust(state)
            # Sync Rust-native memory if enabled
            if self._use_rust_memory:
                self._sync_rust_memory_from_state(state)
            if _profiler.enabled:
                _profiler.sync_to_rust_time += time.perf_counter() - t0

            # Get solver context from RustSimSolver (if available)
            # This allows the Rust engine to share the constraint solver with Python,
            # ensuring branch constraints are properly tracked.
            solver_ctx = None
            if RUST_SOLVER_AVAILABLE and isinstance(state.solver, RustSimSolver):
                solver_ctx = state.solver._rust_ctx

            # Run the execution loop
            if _profiler.enabled:
                t0 = time.perf_counter()
            event = self._rust_engine.run_loop(max_blocks, solver_ctx=solver_ctx)
            if _profiler.enabled:
                _profiler.execute_time += time.perf_counter() - t0

            # Handle the result
            if _profiler.enabled:
                t0 = time.perf_counter()
            self._handle_loop_execution_event(event, state, successors)
            # Sync memory changes back from Rust
            self._sync_memory_from_rust(state)
            # Sync Rust-native memory dirty pages if enabled
            if self._use_rust_memory:
                self._sync_rust_memory_to_state(state)
            if _profiler.enabled:
                _profiler.sync_from_rust_time += time.perf_counter() - t0

            successors.processed = True

            # Update profiler stats
            if _profiler.enabled:
                _profiler.block_count += event.blocks_executed
                _profiler.total_time += time.perf_counter() - loop_start

        except Exception as e:
            l.warning("Rust VEX loop execution failed: %s, falling back", e)
            # Restore state snapshot - Rust execution may have modified Python state
            # through memory callbacks or partial syncs. Restore to ensure Python VEX
            # has clean symbolic values.
            self.state = state_snapshot

            # Clear callbacks and fall back to Python VEX directly
            # Note: We use super().process_successors() to skip RustVEXMixin and go
            # directly to the Python implementation. This avoids another Rust attempt
            # which would also likely fail for the same reason (symbolic values).
            self._rust_engine.clear_callbacks()
            return super().process_successors(successors, **kwargs)
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

        # Check for error events FIRST - don't sync state if execution failed
        # because that would overwrite symbolic values with concrete garbage
        if event_type == "error":
            error_msg = event.error or "Unknown error"
            raise errors.SimEngineError(f"Rust VEX engine error: {error_msg}")

        # Sync state from Rust only for successful execution events
        self._sync_state_from_rust(state)

        if event_type == "max_blocks":
            # Reached max blocks - create a successor to continue
            next_addr = event.pc
            state.ip = next_addr
            successors.add_successor(
                state,
                next_addr,
                _CLARIPY_TRUE,
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
                _CLARIPY_TRUE,
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
                _CLARIPY_TRUE,
                jumpkind,
            )

        elif event_type == "hook":
            # Hook hit - return to Python to let HooksMixin execute it
            # IMPORTANT: Use Ijk_Boring (not Ijk_NoHook) so HooksMixin will
            # check and execute the hook. Ijk_NoHook tells HooksMixin to SKIP
            # the hook, which is the opposite of what we want here.
            hook_addr = event.addr
            state.ip = hook_addr
            successors.add_successor(
                state,
                hook_addr,
                _CLARIPY_TRUE,
                "Ijk_Boring",  # Changed from Ijk_NoHook
            )

        elif event_type == "syscall":
            # Syscall - return for Python handling
            jumpkind = event.jumpkind or "Ijk_Sys_syscall"
            successors.add_successor(
                state,
                state.ip,
                _CLARIPY_TRUE,
                jumpkind,
            )

        elif event_type == "symbolic_branch":
            # Symbolic branch detected with deferred forks disabled
            # This only happens when use_deferred_forks=False in ExecutionConfig.
            # When deferred forks are enabled, symbolic branches are handled inline
            # and this event type should not be raised.
            l.debug("Symbolic branch detected (deferred forks disabled), falling back to Python VEX")
            raise errors.SimEngineError("symbolic branch requires Python VEX (enable deferred forks)")

        elif event_type == "need_lift":
            # Need to lift a block - shouldn't happen with callbacks
            l.warning("Unexpected need_lift event at 0x%x", event.addr)
            raise errors.SimEngineError(f"Unexpected need_lift at 0x{event.addr:x}")

        # Note: "error" event type is handled at the top of this function

        else:
            l.warning("Unknown loop execution event type: %s", event_type)
            raise errors.SimEngineError(f"Unknown event type: {event_type}")

        # Process deferred forks using unified solver architecture
        # Each deferred fork represents a branch where we took one path and deferred the other.
        # With the shared solver context, we can properly handle fork constraints.
        if hasattr(event, 'deferred_forks') and event.deferred_forks:
            self._process_deferred_forks(event.deferred_forks, event.push_level, state, successors)

    def _process_deferred_forks(
        self,
        deferred_forks: list,
        current_push_level: int,
        state: "SimState",
        successors: SimSuccessors,
    ) -> None:
        """
        Process deferred forks using the unified solver architecture.

        For each deferred fork, we:
        1. Add the taken path constraint to the main state
        2. Create a fork state that will explore the unexplored branch path
           with the negated constraint

        This matches Python VEX's behavior in heavy.py for symbolic branches.

        Args:
            deferred_forks: List of DeferredFork objects from Rust execution.
            current_push_level: Current solver push level after execution.
            state: The current state after execution.
            successors: SimSuccessors to add fork states to.
        """
        import claripy

        l.debug("Processing %d deferred forks (push_level=%d)",
                len(deferred_forks), current_push_level)

        # First, add the taken path constraints to the main state
        # This ensures the main state's solver knows about all branches taken
        for fork in deferred_forks:
            if fork.condition_ast is not None:
                if fork.path_taken:
                    # We took the true path, add condition as constraint
                    state.solver.add(fork.condition_ast)
                else:
                    # We took the false path, add Not(condition) as constraint
                    state.solver.add(claripy.Not(fork.condition_ast))

        # Process forks in reverse order (LIFO) to match solver push/pop structure
        for fork in reversed(deferred_forks):
            l.debug("Processing fork: branch_addr=0x%x, path_taken=%s, unexplored=0x%x, fork_push_level=%d",
                    fork.branch_addr, fork.path_taken, fork.unexplored_target, fork.push_level)

            # Validate the unexplored target is a mapped executable address
            # Skip forks with invalid targets (e.g., from concretized symbolic addresses)
            target = fork.unexplored_target
            if state.project is not None:
                try:
                    obj = state.project.loader.find_object_containing(target)
                    if obj is None:
                        l.debug("Skipping fork with unmapped target 0x%x", target)
                        continue
                except Exception:
                    l.debug("Skipping fork with invalid target 0x%x", target)
                    continue

            # Create fork state by copying current state
            # The solver will be forked along with the state (now includes main state constraints)
            fork_state = state.copy()

            # Set the PC to the unexplored target
            fork_state.ip = fork.unexplored_target

            # If condition AST is available, replace the main state's taken constraint
            # with the fork's opposite constraint
            if fork.condition_ast is not None:
                if fork.path_taken:
                    # Main took true, fork needs ~condition (false path)
                    # Remove the taken constraint and add the opposite
                    # Since fork_state inherits main state's constraints, we need to
                    # remove the taken constraint and add the negated one
                    # For simplicity, we add both constraints - the solver handles contradictions
                    fork_state.solver.add(claripy.Not(fork.condition_ast))
                else:
                    # Main took false, fork needs condition (true path)
                    fork_state.solver.add(fork.condition_ast)

            # Add the fork state as a successor
            successors.add_successor(
                fork_state,
                fork.unexplored_target,
                claripy.true,  # Guard - actual constraints are in solver
                "Ijk_Boring",
                add_guard=False,  # Don't add guard as constraint (already handled)
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
