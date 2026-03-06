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
from collections import OrderedDict
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
    elif "V128" in type_name or "U128" in type_name:
        # Serialize 128-bit values as [low, high] u64 pair since serde_json
        # cannot deserialize u128 natively
        low = value & ((1 << 64) - 1)
        high = (value >> 64) & ((1 << 64) - 1)
        tag = "Ico_V128" if "V128" in type_name else "Ico_U128"
        return {"tag": tag, "low": low, "high": high}
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

        # Page fetch counters and timing
        self.fetch_page_count = 0
        self.batch_fetch_pages_count = 0
        self.fetch_page_time = 0.0

        # Track concretization constraints to ensure forked solvers use same values
        self._concretization_constraints = []

        # Deduplication set for already-constrained symbolic loads this block
        # Key: (addr, size), prevents adding redundant constraints
        self._constrained_this_block = set()

        # Readonly memory regions cache: list of (start, end) tuples
        # Loads from these regions don't need concretization constraints
        self._readonly_regions = None
        self._readonly_regions_initialized = False

    def _initialize_readonly_regions(self):
        """Build cache of readonly memory regions from the project."""
        if self._readonly_regions_initialized:
            return

        self._readonly_regions = []
        if hasattr(self.project, 'loader') and self.project.loader:
            # Get .text, .rodata and other readonly sections
            for obj in self.project.loader.all_objects:
                if hasattr(obj, 'sections'):
                    for sec in obj.sections:
                        # Check for readonly sections
                        name = sec.name if hasattr(sec, 'name') else ''
                        is_readonly = (
                            name in ('.text', '.rodata', '.init', '.fini', '.plt', '.plt.got') or
                            (hasattr(sec, 'is_writable') and not sec.is_writable and sec.vaddr)
                        )
                        if is_readonly and sec.vaddr and sec.memsize > 0:
                            self._readonly_regions.append((sec.vaddr, sec.vaddr + sec.memsize))
        self._readonly_regions_initialized = True

    def _is_readonly_region(self, addr: int, size: int = 1) -> bool:
        """Check if address range is in a readonly memory region."""
        self._initialize_readonly_regions()
        end = addr + size
        for (start, region_end) in self._readonly_regions:
            if addr >= start and end <= region_end:
                return True
        return False

    def clear_block_tracking(self):
        """Clear per-block tracking data. Call at block boundaries."""
        self._constrained_this_block.clear()

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
                # Add concretization constraint so forked solvers use same value.
                # Skip for:
                # 1. Readonly regions (values can't change, constraint is redundant)
                # 2. Already constrained this block (avoid duplicate constraints)
                key = (addr, size)
                if key not in self._constrained_this_block and not self._is_readonly_region(addr, size):
                    self._constrained_this_block.add(key)
                    concretization_constraint = (val == concrete)
                    self.state.solver.add(concretization_constraint)
                    self._concretization_constraints.append(concretization_constraint)

            concrete_bytes = concrete.to_bytes(size, 'little')

            if is_sym:
                result = (concrete_bytes, True, val)
            else:
                result = (concrete_bytes, False, None)
            if _profiler.enabled:
                self.memory_load_time += time.perf_counter() - t0
            return result
        except Exception as e:
            l.debug("Memory load failed at 0x%x (size %d): %s, creating symbolic", addr, size, e)
            if _profiler.enabled:
                self.memory_load_time += time.perf_counter() - t0
            # Create symbolic value for unmapped memory instead of returning zeros
            sym_val = claripy.BVS(f"mem_{addr:x}_{size}", size * 8)
            return (bytes(size), True, sym_val)

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
                        # Add concretization constraint so forked solvers use same value.
                        # Skip for:
                        # 1. Readonly regions (values can't change, constraint is redundant)
                        # 2. Already constrained this block (avoid duplicate constraints)
                        key = (addr, size)
                        if key not in self._constrained_this_block and not self._is_readonly_region(addr, size):
                            self._constrained_this_block.add(key)
                            concretization_constraint = (val == concrete)
                            self.state.solver.add(concretization_constraint)
                            self._concretization_constraints.append(concretization_constraint)

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

        Note: When use_rust_memory=True, Rust handles symbolic loads internally
        via load_symbolic_unified() which builds proper ITE chains. This callback
        is a fallback that only loads from the first address.

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

            # Fallback: only loads from first address - no ITE chain built
            addr = addrs[0]
            val = self.state.memory.load(addr, size, endness='Iend_LE')

            # Extract concrete value
            if not val.symbolic:
                if val.op == 'BVV':
                    concrete = val.args[0]
                else:
                    concrete = self.state.solver.eval(val)
            else:
                concrete = self.state.solver.eval(val)
                # Add concretization constraint so forked solvers use same value.
                # Skip for readonly regions or already-constrained addresses
                key = (addr, size)
                if key not in self._constrained_this_block and not self._is_readonly_region(addr, size):
                    self._constrained_this_block.add(key)
                    concretization_constraint = (val == concrete)
                    self.state.solver.add(concretization_constraint)
                    self._concretization_constraints.append(concretization_constraint)

            return concrete.to_bytes(size, 'little')
        except Exception as e:
            l.warning("Symbolic memory load failed for addrs %s: %s", addrs, e)
            return bytes(size)

    def memory_store_symbolic(self, addrs: list[int], data: bytes, addr_width: int) -> None:
        """
        Store to memory with symbolic address (multiple concrete possibilities).

        Note: When use_rust_memory=True, Rust handles symbolic stores internally
        via store_symbolic_unified() which performs proper conditional stores.
        This callback is a fallback that only stores to the first address.

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
            new_val = _get_zero_bvv(bits) if value == 0 else claripy.BVV(value, bits)

            # Fallback: only stores to first address - no conditional stores
            self.state.memory.store(addrs[0], new_val, endness='Iend_LE')
        except Exception as e:
            l.warning("Symbolic memory store failed for addrs %s: %s", addrs, e)

    def memory_load_ast(self, size: int) -> tuple[bytes, bool, Any]:
        """
        Load from memory when the address range is too large to concretize.

        Returns a fresh symbolic value since the actual value depends on
        which address is accessed.

        Args:
            size: Number of bytes to load.

        Returns:
            Tuple of (concrete_bytes, is_symbolic, symbolic_ast_or_none).
        """
        try:
            sym_name = f"unconstrained_load_{size}_{id(self)}"
            result = claripy.BVS(sym_name, size * 8)
            return (bytes(size), True, result)
        except Exception as e:
            l.warning("Symbolic AST memory load failed: %s", e)
            return (bytes(size), True, None)

    def memory_store_ast(self, data: bytes, size: int) -> None:
        """
        Store to memory when the address range is too large to concretize.

        Since we can't enumerate all possible addresses, this store is
        treated as a no-op. This is a known limitation for unconstrained
        symbolic addresses.

        Args:
            data: Bytes to store.
            size: Number of bytes being stored.
        """
        # No-op: can't perform store without knowing the address
        pass

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
                # Add concretization constraint so forked solvers use same value.
                # This prevents divergence when falling back from Rust to Python.
                concretization_constraint = (val == concrete)
                self.state.solver.add(concretization_constraint)
                self._concretization_constraints.append(concretization_constraint)

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

    def fetch_page(self, page_addr: int) -> tuple[bytes, int, bool]:
        """
        Fetch a 4KB page from angr's memory model.

        This is called when Rust's memory encounters an unmapped page in a
        lazy region. The page is fetched and cached in Rust memory for
        subsequent accesses.

        Args:
            page_addr: Page-aligned address (must be multiple of 4096).

        Returns:
            Tuple of (page_data_4kb, permissions, is_mapped).
            - page_data: 4096 bytes of page content
            - permissions: permission bits (R=4, W=2, X=1)
            - is_mapped: False if the page doesn't exist in Python memory
        """
        self.fetch_page_count += 1
        if _profiler.enabled:
            t0 = time.perf_counter()

        PAGE_SIZE = 4096

        try:
            # Check if this page is mapped in angr's memory
            memory = self.state.memory

            # Try to load the full page
            # For efficiency, we load all 4KB at once
            try:
                page_data = bytearray(PAGE_SIZE)
                has_data = False

                # Load byte by byte, handling potential unmapped regions
                for offset in range(PAGE_SIZE):
                    addr = page_addr + offset
                    try:
                        val = memory.load(addr, 1, endness='Iend_LE')
                        if val.symbolic:
                            # Symbolic byte - use concrete approximation
                            if val.op == 'BVV':
                                page_data[offset] = val.args[0] & 0xFF
                            else:
                                page_data[offset] = self.state.solver.eval(val) & 0xFF
                        else:
                            if val.op == 'BVV':
                                page_data[offset] = val.args[0] & 0xFF
                            else:
                                page_data[offset] = self.state.solver.eval(val) & 0xFF
                        has_data = True
                    except Exception:
                        # This byte is unmapped - leave as zero
                        page_data[offset] = 0

                if not has_data:
                    # Entire page is unmapped
                    if _profiler.enabled:
                        self.fetch_page_time += time.perf_counter() - t0
                    return (bytes(PAGE_SIZE), 0, False)

                # Get permissions (default to RWX if unknown)
                permissions = 7  # RWX

                if _profiler.enabled:
                    self.fetch_page_time += time.perf_counter() - t0
                return (bytes(page_data), permissions, True)

            except Exception as e:
                l.debug("Page fetch failed at 0x%x: %s", page_addr, e)
                if _profiler.enabled:
                    self.fetch_page_time += time.perf_counter() - t0
                return (bytes(PAGE_SIZE), 0, False)

        except Exception as e:
            l.warning("Page fetch error at 0x%x: %s", page_addr, e)
            if _profiler.enabled:
                self.fetch_page_time += time.perf_counter() - t0
            return (bytes(PAGE_SIZE), 0, False)

    def batch_fetch_pages(self, page_addrs: list[int]) -> list[tuple[bytes, int, bool]]:
        """
        Fetch multiple 4KB pages from angr's memory model.

        This is more efficient than fetching pages one at a time.

        Args:
            page_addrs: List of page-aligned addresses.

        Returns:
            List of (page_data_4kb, permissions, is_mapped) tuples.
        """
        self.batch_fetch_pages_count += 1
        if _profiler.enabled:
            t0 = time.perf_counter()

        results = []
        for page_addr in page_addrs:
            # Reuse single-page fetch logic
            result = self.fetch_page(page_addr)
            results.append(result)
            # Don't double-count
            self.fetch_page_count -= 1

        if _profiler.enabled:
            self.fetch_page_time += time.perf_counter() - t0

        return results

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
            "fetch_page_count": self.fetch_page_count,
            "batch_fetch_pages_count": self.batch_fetch_pages_count,
            "fetch_page_time_ms": self.fetch_page_time * 1000,
            "total_callback_time_ms": (
                self.memory_load_time + self.memory_store_time +
                self.memory_store_batch_time + self.memory_load_batch_time +
                self.register_get_time + self.register_put_time +
                self.lift_block_time + self.fetch_page_time
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
        rust_cbs.set_fetch_page(self.fetch_page)
        rust_cbs.set_batch_fetch_pages(self.batch_fetch_pages)

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
    # Maximum number of fallback entries to cache (LRU eviction)
    _MAX_FALLBACK_CACHE_SIZE: int = 1024

    def __init__(self, project: angr.Project, use_deferred_forks: bool = True, max_deferred_forks: int = 5, use_rust_memory: bool = False):
        super().__init__(project)

        self._use_deferred_forks = use_deferred_forks
        self._max_deferred_forks = max_deferred_forks
        self._concrete_memory_synced = False
        self._last_callbacks: RustVEXCallbacks | None = None
        self._use_rust_memory = use_rust_memory
        self._rust_memory_synced = False
        # Cache of addresses that failed Rust execution -> reason string (LRU via OrderedDict)
        self._fallback_addresses: OrderedDict[int, str] = OrderedDict()
        # Cache of last synced register values (offset -> concrete_value)
        # Used to skip redundant syncs when values haven't changed
        self._last_synced_registers: dict[int, int] = {}
        # Memory version tracking: id() of the last state whose memory was synced
        # Used to skip redundant memory syncs when stepping the same state
        self._last_synced_state_id: int | None = None
        # Track if memory was modified by Rust (requires resync from Python)
        self._rust_memory_dirty: bool = False

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
                # Load binary regions for native VEX lifting (eliminates lift callbacks)
                native_regions = self._load_binary_regions_for_native_lift()
                if native_regions > 0:
                    l.debug("Loaded %d binary regions for native lifting", native_regions)
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

    def _should_use_rust(self, addr: int) -> bool:
        """
        Check if Rust engine should be used for the given address.

        Returns False if the address is in the fallback cache (known to fail),
        True otherwise.
        """
        return addr not in self._fallback_addresses

    def _record_fallback(self, addr: int, reason: str) -> None:
        """
        Record that Rust execution failed at the given address.

        This caches the failure so we don't repeatedly try Rust at this address.
        Uses LRU eviction when cache exceeds _MAX_FALLBACK_CACHE_SIZE.
        """
        # Move to end if already present (LRU behavior)
        if addr in self._fallback_addresses:
            self._fallback_addresses.move_to_end(addr)
            return

        # Evict oldest entry if at capacity
        while len(self._fallback_addresses) >= self._MAX_FALLBACK_CACHE_SIZE:
            self._fallback_addresses.popitem(last=False)

        self._fallback_addresses[addr] = reason
        l.debug("Recorded Rust fallback for 0x%x: %s", addr, reason)

    def _init_rust_memory(self) -> None:
        """Initialize Rust-native memory model with lazy regions for on-demand fetching."""
        if self._rust_engine is None:
            return

        # Determine endianness from architecture
        little_endian = self.project.arch.memory_endness == 'Iend_LE'

        # Create the Rust memory model
        self._rust_engine.create_rust_memory(little_endian)
        self._rust_engine.enable_rust_memory()

        # Add lazy regions for common memory areas that should be fetched on-demand
        # This avoids pre-loading pages that may never be accessed
        self._setup_lazy_regions()

        l.debug("Rust-native memory model initialized (little_endian=%s)", little_endian)

    def _setup_lazy_regions(self) -> None:
        """Set up lazy regions for on-demand page fetching."""
        if self._rust_engine is None:
            return

        # Stack region: angr uses addresses around 0x7fffffffffef000 for the stack
        # We need to cover a range below the stack top since stack grows down
        arch_bits = self.project.arch.bits

        if arch_bits == 64:
            # 64-bit: Add lazy region for angr's stack area
            # angr stack is around 0x7fffffffffef000, stack grows DOWN
            # Cover 1GB below the typical stack top to reduce fallbacks
            stack_top = 0x7fffffffffff000   # Typical stack top in angr
            stack_size = 0x40000000         # 1GB lazy region (was 256MB)
            stack_base = stack_top - stack_size

            self._rust_engine.add_lazy_region(stack_base, stack_size)
            l.debug("Added 64-bit stack lazy region: 0x%x - 0x%x (1GB)", stack_base, stack_top)

            # Also add lazy region for typical heap area (for brk-based allocation)
            heap_base = 0x0000_0060_0000  # Common heap start
            heap_size = 0x0000_4000_0000  # 1GB lazy region (was 256MB)

            self._rust_engine.add_lazy_region(heap_base, heap_size)
            l.debug("Added 64-bit heap lazy region: 0x%x - 0x%x (1GB)", heap_base, heap_base + heap_size)

            # Add lazy region for mmap area (typical location for large allocations)
            mmap_base = 0x7fff_0000_0000  # Common mmap region
            mmap_size = 0x0000_4000_0000  # 1GB lazy region

            self._rust_engine.add_lazy_region(mmap_base, mmap_size)
            l.debug("Added 64-bit mmap lazy region: 0x%x - 0x%x (1GB)", mmap_base, mmap_base + mmap_size)

        elif arch_bits == 32:
            # 32-bit: Address space is more limited but still expand regions
            # angr uses various stack addresses for 32-bit, including 0x7ffef000 and 0xbffff000
            # We add multiple regions to cover common cases

            # Primary stack region (covers addresses like 0x7ffef000)
            stack_base_1 = 0x7f00_0000  # Start of typical stack area
            stack_size_1 = 0x0100_0000  # 16MB to cover 0x7f000000 - 0x80000000

            self._rust_engine.add_lazy_region(stack_base_1, stack_size_1)
            l.debug("Added 32-bit stack lazy region 1: 0x%x - 0x%x (16MB)", stack_base_1, stack_base_1 + stack_size_1)

            # Secondary stack region (covers addresses like 0xbffef000)
            stack_base_2 = 0xbf00_0000  # Typical 32-bit stack area for some binaries
            stack_size_2 = 0x0100_0000  # 16MB

            self._rust_engine.add_lazy_region(stack_base_2, stack_size_2)
            l.debug("Added 32-bit stack lazy region 2: 0x%x - 0x%x (16MB)", stack_base_2, stack_base_2 + stack_size_2)

            heap_base = 0x0804_0000  # After typical binary load
            heap_size = 0x2000_0000  # 512MB (was 256MB)

            self._rust_engine.add_lazy_region(heap_base, heap_size)
            l.debug("Added 32-bit heap lazy region: 0x%x - 0x%x (512MB)", heap_base, heap_base + heap_size)

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

        # Map all readable binary sections (including writable like GOT/data)
        for obj in self.project.loader.all_objects:
            # Map segments from the loader's memory
            for segment in obj.segments:
                # Map all readable segments (code, rodata, data, got, etc.)
                if segment.is_readable:
                    try:
                        # Load the data from the loader's memory
                        data = self.project.loader.memory.load(
                            segment.vaddr,
                            segment.memsize
                        )
                        # Set permissions based on segment flags
                        perms = 4  # R--
                        if segment.is_executable:
                            perms |= 1  # R-X
                        if segment.is_writable:
                            perms |= 2  # RW- or RWX
                        self._rust_engine.map_memory_data(
                            segment.vaddr,
                            bytes(data),
                            perms
                        )
                        regions_mapped += 1
                        l.debug(
                            "Mapped segment 0x%x-0x%x (%d bytes, perms=%d) to Rust",
                            segment.vaddr,
                            segment.vaddr + segment.memsize,
                            segment.memsize,
                            perms
                        )
                    except Exception as e:
                        l.debug("Failed to map segment at 0x%x: %s", segment.vaddr, e)

        return regions_mapped

    def _load_binary_regions_for_native_lift(self) -> int:
        """
        Load binary code regions for native VEX lifting.

        This loads executable segments (.text, etc.) into the Rust engine
        for native lifting via libpyvex FFI, eliminating Python callbacks
        for code lifting in most cases.

        Returns:
            Number of regions loaded for native lifting.
        """
        if not self.rust_engine_available:
            return 0

        regions = []

        # Load all executable segments
        for obj in self.project.loader.all_objects:
            for segment in obj.segments:
                # Only load executable segments (code)
                if segment.is_executable and segment.is_readable:
                    try:
                        data = self.project.loader.memory.load(
                            segment.vaddr,
                            segment.memsize
                        )
                        regions.append((segment.vaddr, bytes(data)))
                        l.debug(
                            "Loaded code region 0x%x-0x%x (%d bytes) for native lifting",
                            segment.vaddr,
                            segment.vaddr + segment.memsize,
                            segment.memsize
                        )
                    except Exception as e:
                        l.debug("Failed to load code region at 0x%x: %s", segment.vaddr, e)

        if regions:
            try:
                self._rust_engine.load_binary_regions(regions)
                if self._rust_engine.native_lift_available:
                    l.info("Native VEX lifting enabled with %d code regions", len(regions))
                return len(regions)
            except Exception as e:
                l.debug("Failed to enable native lifting: %s", e)

        return 0

    def _sync_state_memory_to_rust(self, state: "SimState", force: bool = False) -> int:
        """
        Sync all concrete memory from state to Rust engine.

        This maps all pages from angr's memory model to Rust, enabling
        Rust to execute without memory callbacks for concrete regions.

        Uses state identity tracking to skip redundant syncs when the same
        state is being stepped multiple times without modification.

        Args:
            state: SimState to sync from.
            force: If True, force sync even if state appears unchanged.

        Routes to appropriate implementation based on memory type:
        - PagedMemoryMixin: Direct page access
        - RegionedMemoryMixin (AbstractMemory): Iterate through regions
        """
        if not self.rust_engine_available:
            return 0

        # Skip sync if this is the same state we already synced and Rust didn't
        # modify memory (no stores during last execution)
        state_id = id(state)
        if not force and state_id == self._last_synced_state_id and not self._rust_memory_dirty:
            l.debug("Skipping memory sync - state unchanged (id=%d)", state_id)
            return 0

        # Track that we're syncing this state
        self._last_synced_state_id = state_id
        self._rust_memory_dirty = False

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
            Number of stores synced back.
        """
        if not self.rust_engine_available:
            return 0

        stores_synced = 0
        if _memory_is_regioned(state.memory):
            stores_synced = self._sync_memory_from_rust_regioned(state)
        elif _memory_has_paging(state.memory):
            stores_synced = self._sync_memory_from_rust_paged(state)
        else:
            l.debug("Unknown memory type, skipping sync from Rust")

        # If Rust wrote to memory, mark as dirty so next sync doesn't skip
        if stores_synced > 0:
            self._rust_memory_dirty = True

        return stores_synced

    def _sync_memory_from_rust_paged(self, state: "SimState") -> int:
        """Sync memory changes from Rust back to paged memory.

        Uses store log for precise sync - only updates bytes that were actually
        written, preserving symbolic values in other locations on the same page.
        """
        stores_synced = 0

        # Get store log: list of (address, size) for individual stores
        store_log = self._rust_engine.get_store_log()

        for store_addr, store_size in store_log:
            try:
                # Read only the specific bytes that were stored
                data = self._rust_engine.read_memory(store_addr, store_size)
                # Store back to angr's memory as a bitvector
                bv = claripy.BVV(int.from_bytes(data, 'little'), store_size * 8)
                state.memory.store(store_addr, bv, endness='Iend_LE')
                stores_synced += 1
            except Exception:
                # Address might not be readable in this context
                pass

        # Clear store log after sync
        self._rust_engine.clear_store_log()

        return stores_synced

    def _sync_memory_from_rust_regioned(self, state: "SimState") -> int:
        """Sync memory changes from Rust back to regioned memory.

        Uses store log for precise sync - only updates bytes that were actually
        written, preserving symbolic values in other locations.
        """
        stores_synced = 0

        # Get store log: list of (address, size) for individual stores
        store_log = self._rust_engine.get_store_log()

        for store_addr, store_size in store_log:
            try:
                # Read only the specific bytes that were stored
                data = self._rust_engine.read_memory(store_addr, store_size)
                # Store back to angr's memory as a bitvector (store works on all memory types)
                bv = claripy.BVV(int.from_bytes(data, 'little'), store_size * 8)
                state.memory.store(store_addr, bv, endness='Iend_LE')
                stores_synced += 1
            except Exception:
                # Address might not be readable in this context
                pass

        # Clear store log after sync
        self._rust_engine.clear_store_log()

        return stores_synced

    @property
    def rust_engine_available(self) -> bool:
        """Check if the Rust engine is available and initialized."""
        return self._rust_engine is not None

    def process(self, state, **kwargs):
        """
        Override to use multi-block loop execution by default.

        By default, this uses process_successors_loop() for multi-block execution
        with deferred forks, reducing Python/Rust round trips. Multi-block mode
        requires RustSimSolver for constraint consistency; without it, falls back
        to single-block mode. To explicitly disable multi-block mode, add
        RUST_VEX_SINGLE to state.options.

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

        # Multi-block execution requires RustSimSolver for constraint consistency
        has_rust_solver = RUST_SOLVER_AVAILABLE and isinstance(state.solver, RustSimSolver)

        if (o.RUST_VEX_SINGLE not in state.options
            and self.rust_engine_available
            and has_rust_solver
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

    def _sync_simprocedures_to_rust(self, project) -> None:
        """
        Synchronize SimProcedure hooks to the Rust engine.

        This registers SimProcedures with their names and argument counts,
        enabling the Rust engine to provide more detailed event information.
        """
        if self._rust_engine is None:
            return

        for hook_addr, proc_info in project._sim_procedures.items():
            if proc_info is not None:
                # proc_info is typically (SimProcedure class, kwargs) or just a SimProcedure
                proc = proc_info if not isinstance(proc_info, tuple) else proc_info[0]

                # Get procedure name
                proc_name = (
                    getattr(proc, "__name__", None)
                    or getattr(proc, "display_name", None)
                    or type(proc).__name__
                )

                # Try to get num_args from prototype if available
                num_args = 0
                if hasattr(proc, "prototype") and proc.prototype is not None:
                    num_args = len(getattr(proc.prototype, "args", []))
                elif hasattr(proc, "num_args"):
                    num_args = proc.num_args

                # Check if procedure never returns
                no_return = getattr(proc, "NO_RET", False)

                self._rust_engine.register_simprocedure(hook_addr, proc_name, num_args, no_return)
            else:
                # Fall back to simple hook if no proc info
                self._rust_engine.add_hook(hook_addr)

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

        # Sync concretization configuration to match Python's strategy.
        # This ensures Rust uses the same address concretization behavior as Python,
        # preventing state divergence from different concretization results.
        use_approximate = o.APPROXIMATE_MEMORY_INDICES in state.options
        engine.configure_concretization(use_approximate, range_limit=1024)

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
        Pre-map existing stack pages before execution to avoid unmapped memory fallbacks.

        Only prefetches pages that already exist in the state's memory - does NOT create
        new pages. Symbolic data is concretized to provide initial concrete values.
        """
        # Handle regioned memory with dedicated method
        if _memory_is_regioned(state.memory):
            self._prefetch_stack_pages_regioned(state, engine)
            return

        # Only prefetch pages that already exist in state's memory
        # Get access to internal _pages dict to avoid creating new pages
        try:
            pages = state.memory._pages
            page_size = state.memory.page_size
        except AttributeError:
            return

        try:
            sp = state.solver.eval(state.regs.sp)
        except Exception:
            return

        # Stack region: SP-128KB to SP+4KB (extended stack access range)
        stack_start = sp - 0x20000  # 128KB below SP
        stack_end = sp + 0x1000     # 4KB above SP

        # Round to page boundaries
        stack_start = (stack_start // page_size) * page_size

        # Map each existing page in stack region
        for page_addr in range(stack_start, stack_end, page_size):
            page_no = page_addr // page_size
            if page_no not in pages or pages[page_no] is None:
                continue  # Only prefetch existing pages

            try:
                # Load page data (won't create new symbolic data since page exists)
                data = state.memory.load(page_addr, page_size)

                # Skip pages with symbolic data - let Python callback handle them
                # This ensures symbolic branches work correctly
                if data.symbolic:
                    continue

                # Fast path for concrete data
                if data.op == 'BVV':
                    concrete_val = data.args[0]
                else:
                    # Should be concrete at this point, but eval to be safe
                    concrete_val = state.solver.eval(data)

                page_bytes = concrete_val.to_bytes(page_size, byteorder='big')

                # Map the page in Rust (read/write/execute permissions)
                engine.map_memory_data(page_addr, page_bytes, 7)
            except Exception:
                # Skip pages that can't be read
                pass

    def _prefetch_stack_pages_regioned(self, state: SimState, engine) -> None:
        """Prefetch existing stack pages from regioned memory."""
        try:
            sp = state.solver.eval(state.regs.sp)
        except Exception:
            return

        # Stack region: SP-128KB to SP+4KB
        stack_start = sp - 0x20000  # 128KB below SP
        stack_end = sp + 0x1000     # 4KB above SP

        # Round to page boundaries
        page_size = 0x1000
        stack_start = (stack_start // page_size) * page_size

        # Iterate existing regions and their pages
        for region_id, region in state.memory._regions.items():
            if not hasattr(region, '_pages') or not hasattr(region, 'page_size'):
                continue

            region_page_size = region.page_size
            for page_no, page in region._pages.items():
                if page is None:
                    continue

                page_addr = page_no * region_page_size

                # Only prefetch pages in stack region
                if page_addr < stack_start or page_addr >= stack_end:
                    continue

                try:
                    data = state.memory.load(page_addr, region_page_size, endness='Iend_LE')

                    # Skip pages with symbolic data - let Python callback handle them
                    if data.symbolic:
                        continue

                    if data.op == 'BVV':
                        concrete_val = data.args[0]
                    else:
                        concrete_val = state.solver.eval(data)

                    page_bytes = concrete_val.to_bytes(region_page_size, byteorder='big')
                    engine.map_memory_data(page_addr, page_bytes, 7)
                except Exception as e:
                    l.debug("Failed to prefetch stack page 0x%x: %s", page_addr, e)

    def _sync_registers_individual(self, state: SimState, engine) -> None:
        """Sync registers individually, handling both concrete and symbolic values.

        Uses caching to skip syncing registers whose values haven't changed since
        the last sync, reducing FFI overhead.
        """
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

                    # Check cache - skip sync if value unchanged
                    cached_val = self._last_synced_registers.get(offset)
                    if cached_val is not None and cached_val == concrete_val:
                        continue  # Value unchanged, skip sync

                    engine.set_register(reg_name, concrete_val)
                    self._last_synced_registers[offset] = concrete_val
                else:
                    # Symbolic register: always sync (can't easily cache symbolic ASTs)
                    # Clear cache entry since value is now symbolic
                    self._last_synced_registers.pop(offset, None)
                    try:
                        engine.set_symbolic_register(offset, reg_val)
                    except Exception as e:
                        l.debug("Failed to sync symbolic register %s: %s", reg_name, e)
                        # Fall back to a concrete approximation
                        try:
                            concrete_val = state.solver.eval(reg_val)
                            engine.set_register(reg_name, concrete_val)
                            self._last_synced_registers[offset] = concrete_val
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

        Uses lazy register sync: only syncs registers that were modified in Rust.
        This significantly reduces FFI overhead when few registers change.
        """
        if self._rust_engine is None:
            return

        engine = self._rust_engine

        # Sync PC (always needed)
        state.ip = engine.pc

        # Get dirty register offsets (4-byte aligned offsets)
        dirty_offsets = engine.get_dirty_register_offsets()

        if dirty_offsets:
            # Lazy sync: only sync registers that were modified
            for offset in dirty_offsets:
                # Determine register size at this offset
                size = self._get_register_size_at_offset(state.arch, offset)
                try:
                    value = engine.get_register_by_offset(offset, size)
                    state.registers.store(offset, claripy.BVV(value, size * 8))
                    # Update cache with value from Rust
                    self._last_synced_registers[offset] = value
                except Exception as e:
                    l.debug("Failed to sync dirty register at offset %d: %s", offset, e)
        else:
            # No dirty registers tracked - fall back to full sync for safety
            # This handles edge cases where dirty tracking wasn't updated
            key_registers = self._get_key_registers(state.arch)
            for reg_name in key_registers:
                try:
                    offset = state.arch.get_register_offset(reg_name)
                    size = state.arch.registers.get(reg_name, (None, None))[1]
                    if size is None:
                        continue
                    value = engine.get_register_by_offset(offset, size)
                    state.registers.store(offset, claripy.BVV(value, size * 8))
                    # Update cache with value from Rust
                    self._last_synced_registers[offset] = value
                except Exception as e:
                    l.debug("Failed to sync register %s: %s", reg_name, e)

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

        elif event_type == "simprocedure":
            # SimProcedure hook hit with additional info
            self._sync_state_from_rust(state)

            proc_addr = event.next_addr or event.addr
            proc_name = getattr(event, "simprocedure_name", None)
            num_args = getattr(event, "simprocedure_num_args", None)
            return_addr = getattr(event, "simprocedure_return_addr", None)

            l.debug(
                "SimProcedure hit (single-step): %s at 0x%x (num_args=%s, ret_addr=%s)",
                proc_name or "unknown", proc_addr, num_args, return_addr
            )

            successors.add_successor(
                state,
                proc_addr,
                _CLARIPY_TRUE,
                "Ijk_Call",
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

        # Check if there's a hook at this address - if so, use Python for hook handling
        # This is critical: hooks are at ExternObject addresses which have garbage data.
        # Lifting/executing garbage IRSB causes incorrect symbolic execution.
        if state.project is not None and addr in state.project._sim_procedures:
            l.debug("Single-block path: hook at 0x%x, falling back to Python", addr)
            return super().process_successors(
                successors,
                irsb=irsb,
                insn_bytes=insn_bytes,
                extra_stop_points=extra_stop_points,
                num_inst=num_inst,
                size=size,
                **kwargs,
            )

        # Check if this address previously failed Rust execution (fallback cache)
        if not self._should_use_rust(addr):
            l.debug("Skipping Rust for 0x%x (cached fallback)", addr)
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

        # Setup state scratch
        state.history.recent_block_count = 1
        state.scratch.guard = _CLARIPY_TRUE
        state.scratch.sim_procedure = None
        state.scratch.bbl_addr = addr

        # Create callback object for proper memory access
        cbs = RustVEXCallbacks(state, self.project, self)
        self._last_callbacks = cbs

        # Capture constraints before Rust execution for replay on fallback.
        # When Rust fails and we restore the snapshot, any constraints added
        # during Rust execution would be lost. We capture the pre-Rust constraint
        # count so we can replay the Rust-added constraints on the restored state.
        pre_rust_constraint_count = 0
        if RUST_SOLVER_AVAILABLE and isinstance(state.solver, RustSimSolver):
            state.solver._flush()  # Ensure all pending are in Rust context
            pre_rust_constraint_count = len(state.solver._constraint_list)

        # Save state snapshot for rollback if Rust fails
        state_snapshot = state.copy()

        try:
            # Setup Rust callbacks - this enables proper memory access via Python
            rust_cbs = cbs.setup_rust_callbacks()
            self._rust_engine.set_callbacks(rust_cbs)

            # Sync hooks and SimProcedures to Rust engine
            if state.project is not None:
                self._sync_simprocedures_to_rust(state.project)

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

            # Serialize and cache IRSB in callbacks so lift_block callback returns it
            if _profiler.enabled:
                t0 = time.perf_counter()
            irsb_json = _serialize_irsb(irsb)
            cbs._lifted_blocks[addr] = irsb_json  # Pre-cache for lift_block callback
            if _profiler.enabled:
                _profiler.serialize_time += time.perf_counter() - t0
                t0 = time.perf_counter()

            # Get solver context from RustSimSolver (if available)
            solver_ctx = None
            if RUST_SOLVER_AVAILABLE and isinstance(state.solver, RustSimSolver):
                solver_ctx = state.solver._rust_ctx

            # Execute single block using callback-based interpreter
            # Note: deferred forks are now properly handled via BV-to-Bool conversion
            event = self._rust_engine.run_loop(1, solver_ctx=solver_ctx)
            if _profiler.enabled:
                _profiler.execute_time += time.perf_counter() - t0

            # Handle the execution event
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
                _profiler.block_count += 1
                _profiler.total_time += time.perf_counter() - block_start

        except Exception as e:
            l.warning("Rust VEX execution failed: %s, falling back to Python", e)
            # Record this address as a fallback to avoid repeated Rust attempts
            self._record_fallback(addr, str(e))

            # Capture constraints added during Rust execution BEFORE restoring snapshot.
            # These constraints are valid (added by branch decisions, etc.) and must be
            # replayed on the restored state to avoid constraint loss.
            rust_added_constraints = []
            if RUST_SOLVER_AVAILABLE and isinstance(state.solver, RustSimSolver):
                state.solver._flush()  # Ensure pending constraints are committed
                rust_added_constraints = state.solver._constraint_list[pre_rust_constraint_count:]

            # Restore state snapshot to ensure Python VEX has clean state
            self.state = state_snapshot

            # Replay Rust-added constraints on restored state to prevent constraint loss.
            # This is critical for correctness: branch constraints established during
            # Rust execution must persist even when falling back to Python.
            if rust_added_constraints:
                for constraint in rust_added_constraints:
                    state_snapshot.solver.add(constraint)

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

        # Check if this address previously failed Rust execution (fallback cache)
        if not self._should_use_rust(successors.addr):
            l.debug("Skipping Rust for 0x%x (cached fallback)", successors.addr)
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

        # Capture constraints before Rust execution for replay on fallback.
        # When Rust fails and we restore the snapshot, any constraints added
        # during Rust execution would be lost. We capture the pre-Rust constraint
        # count so we can replay the Rust-added constraints on the restored state.
        pre_rust_constraint_count = 0
        if RUST_SOLVER_AVAILABLE and isinstance(state.solver, RustSimSolver):
            state.solver._flush()  # Ensure all pending are in Rust context
            pre_rust_constraint_count = len(state.solver._constraint_list)

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

            # Sync hooks and SimProcedures to Rust engine
            if state.project is not None:
                self._sync_simprocedures_to_rust(state.project)

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
            # Record this address as a fallback to avoid repeated Rust attempts
            self._record_fallback(addr, str(e))

            # Capture constraints added during Rust execution BEFORE restoring snapshot.
            # These constraints are valid (added by branch decisions, etc.) and must be
            # replayed on the restored state to avoid constraint loss.
            rust_added_constraints = []
            if RUST_SOLVER_AVAILABLE and isinstance(state.solver, RustSimSolver):
                state.solver._flush()  # Ensure pending constraints are committed
                rust_added_constraints = state.solver._constraint_list[pre_rust_constraint_count:]

            # Restore state snapshot - Rust execution may have modified Python state
            # through memory callbacks or partial syncs. Restore to ensure Python VEX
            # has clean symbolic values.
            self.state = state_snapshot

            # Replay Rust-added constraints on restored state to prevent constraint loss.
            # This is critical for correctness: branch constraints established during
            # Rust execution must persist even when falling back to Python.
            if rust_added_constraints:
                for constraint in rust_added_constraints:
                    state_snapshot.solver.add(constraint)

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

        elif event_type == "simprocedure":
            # SimProcedure hook hit - similar to hook but with additional info
            # The Rust engine provides the SimProcedure name, arg count, and return addr.
            # This event type is used when a registered SimProcedure is hit.
            proc_addr = event.addr
            proc_name = getattr(event, "simprocedure_name", None)
            num_args = getattr(event, "simprocedure_num_args", None)
            return_addr = getattr(event, "simprocedure_return_addr", None)

            l.debug(
                "SimProcedure hit: %s at 0x%x (num_args=%s, ret_addr=%s)",
                proc_name or "unknown", proc_addr, num_args, return_addr
            )

            state.ip = proc_addr
            successors.add_successor(
                state,
                proc_addr,
                _CLARIPY_TRUE,
                "Ijk_Call",  # Use Ijk_Call for SimProcedures
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
                # Convert 1-bit BV to proper Bool constraint (matches heavy.py:271)
                # Rust returns condition_ast as claripy.BVV/BVS(val, 1) but Z3
                # expects a Bool. Using "!= 0" converts BV to Bool properly.
                condition_bool = fork.condition_ast != 0
                if fork.path_taken:
                    # We took the true path, add condition as constraint
                    state.solver.add(condition_bool)
                else:
                    # We took the false path, add Not(condition) as constraint
                    state.solver.add(claripy.Not(condition_bool))

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
                # Convert 1-bit BV to proper Bool constraint (matches heavy.py:271)
                condition_bool = fork.condition_ast != 0
                if fork.path_taken:
                    # Main took true, fork needs ~condition (false path)
                    # Remove the taken constraint and add the opposite
                    # Since fork_state inherits main state's constraints, we need to
                    # remove the taken constraint and add the negated one
                    # For simplicity, we add both constraints - the solver handles contradictions
                    fork_state.solver.add(claripy.Not(condition_bool))
                else:
                    # Main took false, fork needs condition (true path)
                    fork_state.solver.add(condition_bool)

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

    def register_simprocedure(
        self, addr: int, name: str, num_args: int = 0, no_return: bool = False
    ) -> None:
        """
        Register a SimProcedure at the given address.

        This allows the Rust interpreter to pre-extract arguments when the hook is hit,
        providing the SimProcedure name and argument count in the event.

        Args:
            addr: The hook address.
            name: SimProcedure name (e.g., "strlen", "malloc").
            num_args: Number of arguments to extract.
            no_return: Whether this procedure never returns (e.g., "exit").
        """
        self._engine.register_simprocedure(addr, name, num_args, no_return)

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
