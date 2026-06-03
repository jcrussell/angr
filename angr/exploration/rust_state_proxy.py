"""
Lightweight proxy objects that expose Rust symex state via PyO3 bindings.

Instead of creating full angr SimState objects and syncing bidirectionally,
these proxies delegate reads directly to the Rust engine. This eliminates
caching, sync bugs, and the dual-solver problem for non-SimProcedure paths.

Full SimState creation is only needed for SimProcedure execution (which
requires angr plugins like posix, filesystem, etc.).
"""

import logging

import claripy

from angr.rustylib.vex_engine import register_size_for_arch

l = logging.getLogger(__name__)


class RustSolverProxy:
    """
    Wraps a RustSolverContext to present a claripy-compatible solver interface.

    Delegates satisfiability checks, evaluation, and constraint operations
    directly to the Rust Z3 solver — no claripy frontend sync needed.
    """

    def __init__(self, rust_mgr, state_id):
        self._mgr = rust_mgr
        self._state_id = state_id
        self._solver_ctx = None  # lazy — forked on first access

    def _ensure_solver(self):
        if self._solver_ctx is None:
            self._solver_ctx = self._mgr.fork_state_solver(self._state_id)

    def satisfiable(self, extra_constraints=(), **kwargs):
        """Check if the state's constraints are satisfiable."""
        self._ensure_solver()
        if extra_constraints:
            self._solver_ctx.push()
            try:
                for c in extra_constraints:
                    self._solver_ctx.add_constraint_ast(c)
                return self._solver_ctx.satisfiable()
            finally:
                self._solver_ctx.pop()
        return self._solver_ctx.satisfiable()

    def eval(self, expr, n=1, cast_to=None, extra_constraints=(), **kwargs):
        """Evaluate a symbolic expression to a concrete value.

        Matches angr SimSolver.eval: returns a single value (not a tuple).
        """
        self._ensure_solver()
        # Fast path: concrete expression
        if hasattr(expr, 'concrete') and expr.concrete:
            val = expr.concrete_value if hasattr(expr, 'concrete_value') else expr.args[0]
            return self._cast_result(expr, val, cast_to)
        if extra_constraints:
            self._solver_ctx.push()
            try:
                for c in extra_constraints:
                    self._solver_ctx.add_constraint_ast(c)
                result = self._solver_ctx.eval(expr)
            finally:
                self._solver_ctx.pop()
        else:
            result = self._solver_ctx.eval(expr)
        if result is None:
            raise claripy.errors.UnsatError("unsat")
        return self._cast_result(expr, result, cast_to)

    def _cast_result(self, expr, result, cast_to):
        """Cast eval result matching angr's SimSolver._cast_to behavior."""
        if cast_to is None:
            return result
        if cast_to is bytes:
            if hasattr(expr, '__len__'):
                nbits = len(expr)
            elif hasattr(expr, 'size'):
                nbits = expr.size()
            else:
                nbits = 64
            if nbits == 0:
                return b""
            return result.to_bytes(nbits // 8, byteorder="big")
        return cast_to(result)

    def _eval_inner(self, expr, n, cast_to):
        if n == 1:
            result = self._solver_ctx.eval(expr)
            if result is None:
                raise claripy.errors.UnsatError("unsat")
            result = self._cast_result(expr, result, cast_to)
            return (result,)
        else:
            results = self._solver_ctx.eval_upto(expr, n)
            results = tuple(self._cast_result(expr, r, cast_to) for r in results)
            return results

    def eval_one(self, expr, **kwargs):
        """Evaluate expression expecting exactly one solution."""
        results = self.eval_upto(expr, 2, **kwargs)
        if len(results) != 1:
            raise claripy.errors.ClaripyError(
                f"expected 1 solution, got {len(results)}"
            )
        return results[0]

    def eval_upto(self, expr, n, cast_to=None, extra_constraints=(), **kwargs):
        """Evaluate expression for up to n solutions. Returns a tuple."""
        self._ensure_solver()
        if extra_constraints:
            self._solver_ctx.push()
            try:
                for c in extra_constraints:
                    self._solver_ctx.add_constraint_ast(c)
                return self._eval_inner(expr, n, cast_to)
            finally:
                self._solver_ctx.pop()
        return self._eval_inner(expr, n, cast_to)

    def eval_exact(self, expr, n, **kwargs):
        """Evaluate expression expecting exactly n solutions."""
        results = self.eval_upto(expr, n + 1, **kwargs)
        if len(results) != n:
            raise claripy.errors.ClaripyError(
                f"expected {n} solutions, got {len(results)}"
            )
        return results

    def eval_atleast(self, expr, n, **kwargs):
        """Evaluate expression expecting at least n solutions."""
        results = self.eval_upto(expr, n, **kwargs)
        if len(results) < n:
            raise claripy.errors.ClaripyError(
                f"expected at least {n} solutions, got {len(results)}"
            )
        return results

    def min(self, expr, extra_constraints=(), signed=False, **kwargs):
        """Get minimum value of expression."""
        self._ensure_solver()
        if extra_constraints:
            self._solver_ctx.push()
            try:
                for c in extra_constraints:
                    self._solver_ctx.add_constraint_ast(c)
                result = self._solver_ctx.min(expr, signed=signed)
            finally:
                self._solver_ctx.pop()
        else:
            result = self._solver_ctx.min(expr, signed=signed)
        if result is None:
            raise claripy.errors.UnsatError("unsat")
        return result

    def max(self, expr, extra_constraints=(), signed=False, **kwargs):
        """Get maximum value of expression."""
        self._ensure_solver()
        if extra_constraints:
            self._solver_ctx.push()
            try:
                for c in extra_constraints:
                    self._solver_ctx.add_constraint_ast(c)
                result = self._solver_ctx.max(expr, signed=signed)
            finally:
                self._solver_ctx.pop()
        else:
            result = self._solver_ctx.max(expr, signed=signed)
        if result is None:
            raise claripy.errors.UnsatError("unsat")
        return result

    def add(self, *constraints):
        """Add constraint(s) to the solver."""
        self._ensure_solver()
        for c in constraints:
            if isinstance(c, (list, tuple)):
                for cc in c:
                    self._solver_ctx.add_constraint_ast(cc)
            else:
                self._solver_ctx.add_constraint_ast(c)

    def is_true(self, expr, **kwargs):
        """Check if expression is definitely true."""
        self._ensure_solver()
        return self._solver_ctx.is_true(expr)

    def is_false(self, expr, **kwargs):
        """Check if expression is definitely false."""
        self._ensure_solver()
        return self._solver_ctx.is_false(expr)

    def symbolic(self, expr):
        """Check if expression contains symbolic variables."""
        if isinstance(expr, claripy.ast.Base):
            return expr.symbolic
        return False

    def solution(self, expr, value, **kwargs):
        """Check if value is a valid solution for expr."""
        self._ensure_solver()
        return self._solver_ctx.solution(expr, value)

    @property
    def constraints(self):
        """Get all constraints as claripy ASTs."""
        return self._mgr.export_state_constraints(self._state_id)

    @property
    def timeout(self):
        """Z3 solver timeout in milliseconds for this state's solver context."""
        try:
            return self._mgr.get_state_solver_timeout(self._state_id)
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: state may have been GC'd or backend
            # lacks Z3 — return 0 as default. (No log: this property is read
            # frequently from solver hot paths.)
            return 0

    @timeout.setter
    def timeout(self, value):
        """Forward `state.solver.timeout = N` to the Rust solver context.

        Updates the underlying state's SymContext so future forks inherit
        the timeout, and also propagates to the proxy's already-forked
        solver (if any) so the next satisfiable()/eval() honors it.
        """
        if value is None:
            return
        timeout_ms = int(value)
        self._mgr.set_state_solver_timeout(self._state_id, timeout_ms)
        if self._solver_ctx is not None:
            self._solver_ctx.set_timeout(timeout_ms)


class RustRegisterProxy:
    """
    Provides `state.regs.rax`-style access by delegating to Rust.

    Register values are returned as claripy BVVs for compatibility
    with code that expects symbolic bitvectors.
    """

    def __init__(self, rust_mgr, state_id, arch):
        # Use object.__setattr__ to bypass our own __setattr__ during init
        # (which routes name= writes through to Rust). Without this, the
        # first attribute assignment below would try to look up self._mgr
        # before it exists.
        object.__setattr__(self, "_mgr", rust_mgr)
        object.__setattr__(self, "_state_id", state_id)
        object.__setattr__(self, "_arch", arch)
        object.__setattr__(self, "_cache", {})  # name -> claripy BVV/BVS

    def prefetch(self, names):
        """Batch-fetch multiple registers in one FFI call and cache them."""
        try:
            values = self._mgr.get_state_registers_batch(self._state_id, names)
        except Exception:
            # cat-(a) EXPECTED CONTROL FLOW: prefetch is a best-effort
            # optimization; missing prefetch falls back to per-register
            # __getattr__ on first access.
            return
        for name, val in zip(names, values):
            width = self._get_register_width(name)
            if val is None:
                self._cache[name] = self._recover_symbolic_register_ast(name, width)
            else:
                self._cache[name] = claripy.BVV(val, width)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._cache:
            return self._cache[name]
        try:
            val = self._mgr.get_state_register(self._state_id, name)
        except Exception:
            # cat-(b) FALLBACK WITH LOSS: re-raise as AttributeError so callers
            # using hasattr()/getattr() see "no such register". Note: this
            # masks transient FFI errors as missing-attribute — log debug so
            # they're visible under --debug.
            l.debug("get_state_register(sid=%d, name=%r) failed; reporting as AttributeError",
                    self._state_id, name)
            raise AttributeError(f"register '{name}' not found")
        width = self._get_register_width(name)
        if val is None:
            result = self._recover_symbolic_register_ast(name, width)
        else:
            result = claripy.BVV(val, width)
        self._cache[name] = result
        return result

    def _recover_symbolic_register_ast(self, name, width):
        """Recover Rust's claripy AST for a symbolic register (angr-4pm1).

        ``get_state_register`` returns ``None`` when the register holds a
        symbolic ``RustBV``. Previously we minted a fresh, orphan ``claripy.BVS``
        here — but the orphan symbol has no identity link to Rust's internal
        Z3 AST, so ``state.solver.add(state.regs.<reg> == K)`` constrained a
        ghost symbol and never affected the actual register value.

        Now we ask the manager for the register's claripy AST via
        ``get_state_register_ast`` (which round-trips through
        ``rustbv_to_claripy`` and preserves the underlying Z3 AST), so
        downstream solver operations land on the correct symbol. Only when
        the manager genuinely has no value for the register (unknown name or
        old build without the FFI shim) do we fall back to the orphan BVS —
        same loss-of-identity behavior as before, but at least with a debug
        log so the gap is visible.
        """
        if hasattr(self._mgr, "get_state_register_ast"):
            try:
                ast = self._mgr.get_state_register_ast(self._state_id, name)
            except (RuntimeError, ValueError, KeyError, AttributeError):
                # cat-(b) FALLBACK WITH LOSS: AST recovery failed; we mint
                # an orphan BVS below. Solver writes through that BVS will
                # not affect the actual register value.
                ast = None
            if ast is not None:
                return ast
        l.debug(
            "RustRegisterProxy: no AST recovered for symbolic register %r "
            "(sid=%d); minting orphan BVS — solver writes through the proxy "
            "will not affect this register",
            name,
            self._state_id,
        )
        return claripy.BVS(f"reg_{name}_{self._state_id}", width)

    def _get_register_width(self, name):
        """Get the bit width for a named register.

        Looks up the size in BYTES from the Rust arch table (single source of
        truth) and converts to bits. Falls back to ``arch.bits`` for registers
        Rust doesn't model — the same default the old hand-rolled prefix
        matcher used.
        """
        size_bytes = register_size_for_arch(self._arch.name, name)
        if size_bytes is not None:
            return size_bytes * 8
        return self._arch.bits

    def __setattr__(self, name, value):
        """Write-through register assignment (angr-j28e).

        ``proxy.regs.<name> = value`` from a state.inspect callback,
        find/avoid predicate, or external user code is forwarded immediately
        to the Rust state via ``set_state_register_symbolic_ast``. Rust is
        the single source of truth — there is no Python-side shadow store.

        ``value`` may be a claripy AST (BVV/BVS/any expression), an int, or
        ``bytes``. Ints/bytes are wrapped in a BVV at the register's native
        width before forwarding. After a successful write, the proxy cache
        is updated so the next ``proxy.regs.<name>`` read returns the new
        value without an FFI round-trip.

        Underscore-prefixed names (``_mgr``, ``_state_id``, ``_cache``,
        ``_arch``) and Python protocol attributes are stored as ordinary
        Python attributes via ``object.__setattr__`` — only public register
        names route to Rust.
        """
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        width = self._get_register_width(name)
        ast = self._coerce_to_ast(value, width)
        self._mgr.set_state_register_symbolic_ast(self._state_id, name, ast)
        # Keep the cache coherent so a subsequent __getattr__ returns the
        # AST we just wrote (matches the post-write read invariant the
        # caller would otherwise see if no cache existed).
        self._cache[name] = ast

    @staticmethod
    def _coerce_to_ast(value, width):
        """Coerce a Python int / bytes / claripy AST to a claripy AST.

        The FFI shim wants a claripy AST so it can route through
        ``claripy_to_rustbv`` and register the symbol in the shared cache.
        Plain ints and bytes are wrapped in a ``BVV`` at the requested
        width — same convention as ``state.regs.<name> = 0x41`` on a
        regular SimState.
        """
        if isinstance(value, claripy.ast.Base):
            return value
        if isinstance(value, (bytes, bytearray)):
            return claripy.BVV(bytes(value), width)
        if isinstance(value, int):
            return claripy.BVV(value, width)
        raise TypeError(
            f"register write expects claripy AST, int, or bytes; "
            f"got {type(value).__name__}"
        )

    def load(self, reg_name_or_offset, size=None):
        """Load register by name or by ``(offset, size)`` tuple.

        Mirrors ``SimRegisters.load``: string names route through
        ``__getattr__``; integer offsets are resolved via the architecture's
        ``register_size_names[(offset, size)]`` map (size defaults to
        ``arch.bytes``, matching angr's SimMemory default).
        """
        if isinstance(reg_name_or_offset, str):
            return getattr(self, reg_name_or_offset)
        if isinstance(reg_name_or_offset, int):
            if size is None:
                size = self._arch.bytes
            try:
                name = self._arch.register_size_names[(reg_name_or_offset, size)]
            except KeyError as e:
                raise NotImplementedError(
                    f"no register for offset {reg_name_or_offset} size {size} on {self._arch.name}"
                ) from e
            return getattr(self, name)
        raise TypeError(
            f"register load expects str name or int offset, got {type(reg_name_or_offset).__name__}"
        )


class RustMemoryProxy:
    """
    Provides `state.memory.load(addr, size)`-style access via Rust.

    Returns bytes as claripy BVVs for compatibility.
    """

    def __init__(self, rust_mgr, state_id, arch):
        self._mgr = rust_mgr
        self._state_id = state_id
        self._arch = arch
        self._solver_ctx = None  # lazy — forked on first symbolic-addr load

    def _ensure_solver(self):
        if self._solver_ctx is None:
            self._solver_ctx = self._mgr.fork_state_solver(self._state_id)

    def load(self, addr, size=None, endness=None, **kwargs):
        """Load memory from the Rust state.

        Concrete addresses (int or concrete claripy AST) issue a direct
        FFI load. Symbolic addresses are concretized to a single solution
        under the state's constraints by forking a Rust solver context.
        Unsat addresses raise ``claripy.errors.UnsatError``.
        """
        if isinstance(addr, claripy.ast.Base):
            if addr.concrete:
                addr = addr.concrete_value
            else:
                self._ensure_solver()
                resolved = self._solver_ctx.eval(addr)
                if resolved is None:
                    raise claripy.errors.UnsatError(
                        "symbolic memory load addr is unsat"
                    )
                addr = resolved
        if size is None:
            size = self._arch.bytes
        if isinstance(size, claripy.ast.Base):
            size = size.concrete_value

        data = self._mgr.get_state_memory(self._state_id, addr, size)
        if data is None:
            return claripy.BVV(0, size * 8)

        # Convert bytes to BVV — angr's memory.load() defaults to big-endian
        # regardless of architecture (caller must explicitly pass Iend_LE)
        if endness is None:
            endness = "Iend_BE"
        if endness == "Iend_LE":
            val = int.from_bytes(data, "little")
        else:
            val = int.from_bytes(data, "big")
        return claripy.BVV(val, size * 8)

    def store(self, addr, data, endness=None, **kwargs):
        """Write-through memory store to the Rust state (angr-j28e).

        ``proxy.memory.store(addr, value)`` from a state.inspect callback,
        find/avoid predicate, or external user code is forwarded immediately
        to the Rust state. Rust is the single source of truth — there is no
        Python-side shadow store.

        Address handling:

        * Concrete ``int`` or concrete claripy AST → direct FFI store.
        * Symbolic claripy AST → ``NotImplementedError``. Symbolic-address
          writes from a callback would need the lazy Multi-cell path
          (which exists for the in-engine store but requires solver
          coordination that the proxy lacks); the symmetric refusal here
          matches how ``solver.add`` does not accept symbolic-AST
          constraints with no boolean structure.

        Value handling:

        * ``int`` → wrapped in a ``BVV`` at ``size * 8`` bits (caller must
          pass ``size``).
        * ``bytes`` / ``bytearray`` → wrapped at ``len(value) * 8`` bits.
        * concrete claripy ``BVV`` → fast path through the concrete-bytes
          FFI shim.
        * symbolic claripy AST → routes through ``set_state_memory_ast``
          so the symbol is registered in the shared cache.

        Endianness defaults to ``Iend_BE`` (matches angr's ``memory.store``
        default for raw bytes); pass ``endness='Iend_LE'`` to mirror VEX's
        little-endian convention.
        """
        if isinstance(addr, claripy.ast.Base):
            if addr.concrete:
                addr = addr.concrete_value
            else:
                raise NotImplementedError(
                    "symbolic-address memory store is not supported on "
                    "RustStateProxy. Workaround: use a SimProcedure-style hook "
                    "(proj.hook(addr, fn)) — SimProcedure callbacks receive "
                    "a full SimState and writes are synced back via the "
                    "lazy Multi-cell path. See docs/advanced-topics/"
                    "rust_engine.rst for details."
                )
        size = kwargs.pop('size', None)
        if endness is None:
            endness = "Iend_BE"

        if isinstance(data, claripy.ast.Base):
            width_bits = data.length if hasattr(data, 'length') else data.size()
            if data.concrete:
                value = data.concrete_value
                nbytes = width_bits // 8
                byteorder = "little" if endness == "Iend_LE" else "big"
                payload = value.to_bytes(nbytes, byteorder)
                self._mgr.set_state_memory_concrete(self._state_id, addr, payload)
                return
            # Symbolic AST — route through the AST FFI so the symbol is
            # registered in the shared cache. Endianness on a symbolic AST
            # is the caller's responsibility (claripy ASTs don't carry an
            # endianness flag); we forward verbatim.
            self._mgr.set_state_memory_ast(self._state_id, addr, data)
            return

        if isinstance(data, (bytes, bytearray)):
            payload = bytes(data)
            if endness == "Iend_LE":
                payload = payload[::-1]
            self._mgr.set_state_memory_concrete(self._state_id, addr, payload)
            return

        if isinstance(data, int):
            if size is None:
                raise TypeError(
                    "memory store with an int value requires size=N (bytes)"
                )
            byteorder = "little" if endness == "Iend_LE" else "big"
            payload = data.to_bytes(size, byteorder)
            self._mgr.set_state_memory_concrete(self._state_id, addr, payload)
            return

        raise TypeError(
            f"memory store expects claripy AST, int, or bytes; "
            f"got {type(data).__name__}"
        )


class RustHeapProxy:
    """Proxy for state.heap — exposes mmap_base and allocation metadata
    backed by RustSimState.

    The Rust state owns mmap_base (used by the native mmap syscall handler)
    and a HeapMetadata struct tracking malloc/free regions. Reads/writes go
    through PyO3 accessors so Python user code sees the live Rust value.
    """

    def __init__(self, rust_mgr, state_id):
        self._mgr = rust_mgr
        self._state_id = state_id

    @property
    def mmap_base(self):
        """Current mmap base pointer (mirrors `state.heap.mmap_base`)."""
        try:
            return self._mgr.get_state_mmap_base(self._state_id)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: returning None on FFI error mirrors
            # angr's behavior when state.heap is unavailable.
            l.debug("get_state_mmap_base(sid=%d) failed: %s: %s",
                    self._state_id, type(e).__name__, e)
            return None

    @mmap_base.setter
    def mmap_base(self, value):
        self._mgr.set_state_mmap_base(self._state_id, int(value))

    @property
    def allocations(self):
        """List of (addr, size) for currently-live heap allocations."""
        try:
            allocated, _freed = self._mgr.get_state_heap_metadata(self._state_id)
            return list(allocated)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: empty list when heap metadata is
            # unavailable (e.g., state already cleaned up).
            l.debug("get_state_heap_metadata(sid=%d) failed: %s: %s",
                    self._state_id, type(e).__name__, e)
            return []

    @property
    def freed(self):
        """List of freed addresses in free-call order."""
        try:
            _allocated, freed = self._mgr.get_state_heap_metadata(self._state_id)
            return list(freed)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: empty list when heap metadata is
            # unavailable (e.g., state already cleaned up).
            l.debug("get_state_heap_metadata(sid=%d) failed: %s: %s",
                    self._state_id, type(e).__name__, e)
            return []


class RustScratchProxy:
    """Read-only proxy for `state.scratch` over a Rust state.

    Python's SimStateScratch carries per-block transient values used by the
    Python VEX engine (irsb, bbl_addr, ins_addr, stmt_idx, jumpkind, temps,
    tyenv, ...). The Rust engine doesn't store most of those because they
    belong to the in-flight VEXInterpreter, not the persistent state.

    This proxy exposes the subset that *is* recoverable from the Rust state:
    the most recently entered block address (mirrors `state.pc`) and the
    jumpkind that led to the current state (last detailed-history entry).
    The rest (irsb, temps, tyenv, stmt_idx) are exposed as None — the Rust
    interpreter clears its per-block buffers between blocks, so there are no
    stable post-block values to read.
    """

    # Mirrors angr's Ijk_* string convention. Indices match the u8 values
    # produced by RustSimState::detailed_history (see exploration/mod.rs:1726).
    _JUMPKIND_NAMES = (
        "Ijk_Boring",
        "Ijk_Call",
        "Ijk_Ret",
        "Ijk_Sys_syscall",
        "Ijk_Other",
    )

    def __init__(self, rust_mgr, state_id):
        self._mgr = rust_mgr
        self._state_id = state_id

    @property
    def bbl_addr(self):
        """Address of the most recently entered block (mirrors state.pc)."""
        try:
            return self._mgr.get_state_pc_by_id(self._state_id)
        except Exception as e:
            # cat-(b) FALLBACK WITH LOSS: best-effort read; None mirrors
            # SimStateScratch's pre-block default.
            l.debug("get_state_pc_by_id(sid=%d) failed: %s: %s",
                    self._state_id, type(e).__name__, e)
            return None

    @property
    def ins_addr(self):
        """Address of the most recently executed instruction.

        The Rust interpreter tracks this in VEXInterpreter.current_insn_addr
        but doesn't persist it on the state — once the block finishes, the
        interpreter is dropped. We surface state.pc as a best-effort proxy:
        for a found state, pc is the find address (the last IMark seen).
        """
        return self.bbl_addr

    @property
    def jumpkind(self):
        """Jumpkind that led to the current state, e.g. "Ijk_Boring".

        Pulled from the last entry of detailed_history. Returns None for a
        freshly-created state with no recorded transitions.
        """
        try:
            history = self._mgr.get_state_detailed_history(self._state_id)
        except (RuntimeError, AttributeError, KeyError) as e:
            # cat-(b) FALLBACK WITH LOSS: history fetch failed; caller sees
            # jumpkind=None as if there were no recorded transitions.
            l.debug("get_state_detailed_history(sid=%d) failed: %s: %s",
                    self._state_id, type(e).__name__, e)
            return None
        if not history:
            return None
        _addr, kind_u8, _target = history[-1]
        if 0 <= kind_u8 < len(self._JUMPKIND_NAMES):
            return self._JUMPKIND_NAMES[kind_u8]
        return "Ijk_Other"

    # SimStateScratch attributes the Rust engine doesn't persist between
    # blocks. Returning None matches how Python's plugin reads them before
    # the first block executes.
    @property
    def irsb(self):
        return None

    @property
    def stmt_idx(self):
        return None

    @property
    def temps(self):
        return []

    @property
    def tyenv(self):
        return None

    @property
    def sim_procedure(self):
        return None


class RustHistoryProxy:
    """Provides state.history.recent_bbl_addrs and similar."""

    # Tail length used when materializing recent_bbl_addrs. Matches the
    # spirit of angr's SimStateHistory.recent_bbl_addrs ("most recent" rather
    # than the full lineage chain). Long-running benches accumulate 100k+ bbl
    # entries; cloning the full Vec across FFI was the regression risk
    # called out in angr-kwpi.1.
    _RECENT_TAIL_DEFAULT = 256

    def __init__(self, rust_mgr, state_id):
        self._mgr = rust_mgr
        self._state_id = state_id
        self._bbl_addrs = None

    @property
    def recent_bbl_addrs(self):
        if self._bbl_addrs is None:
            tail = self._mgr.get_state_bbl_history_tail(
                self._state_id, self._RECENT_TAIL_DEFAULT,
            )
            self._bbl_addrs = list(tail) if tail is not None else []
        return self._bbl_addrs

    @property
    def bbl_addrs(self):
        return self.recent_bbl_addrs

    @property
    def block_count(self):
        return len(self.recent_bbl_addrs)


class RustPosixProxy:
    """
    Provides state.posix.dumps(fd) for stdin/stdout extraction.

    For stdout (fd=1): reads from a Rust-side or Python-side buffer
    of accumulated write/puts/printf output.

    For stdin (fd=0): evaluates stdin symbolic variables under the
    state's constraints to extract the concrete input.
    """

    def __init__(self, rust_mgr, state_id, stdin_vars=None, stdout_data=None):
        self._mgr = rust_mgr
        self._state_id = state_id
        self._stdin_vars = stdin_vars or []  # list of (claripy BVS, offset) for stdin
        self._stdout_data = stdout_data or b""  # accumulated stdout bytes

    def dumps(self, fd):
        """Dump file descriptor contents."""
        if fd == 0:
            # stdin — evaluate symbolic variables under constraints
            return self._eval_stdin()
        elif fd == 1:
            # stdout — return accumulated output (cached on proxy init)
            return self._stdout_data
        else:
            # Other fds (stderr, opened files) — query Rust engine
            try:
                return bytes(self._mgr.get_state_fd_output(self._state_id, fd))
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: empty bytes when the requested
                # fd has no Rust-side buffer. Tools that expect specific
                # output should distinguish "no buffer" from "empty buffer";
                # log so they're visible under --debug.
                l.debug("get_state_fd_output(sid=%d, fd=%d) failed: %s: %s",
                        self._state_id, fd, type(e).__name__, e)
                return b""

    def _eval_stdin(self):
        """Evaluate stdin symbolic variables to concrete bytes."""
        if not self._stdin_vars:
            return b""
        try:
            solver_ctx = self._mgr.fork_state_solver(self._state_id)
            result = bytearray()
            for var, _offset in sorted(self._stdin_vars, key=lambda x: x[1]):
                val = solver_ctx.eval(var)
                if val is not None:
                    # Convert to bytes
                    byte_len = max(1, var.length // 8)
                    result.extend(val.to_bytes(byte_len, "big"))
            return bytes(result)
        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: returning empty bytes when stdin eval
            # fails could be misread as "the input was empty" by callers
            # comparing solution bytes. Logged at warn so the failure is loud.
            l.warning("RustPosixProxy: failed to evaluate stdin: %s", e)
            return b""


class RustCallStackFrameProxy:
    """A single frame view backed by a Rust CallStackEntry tuple.

    The Rust engine returns frames as (call_site_addr, callee_addr, return_addr,
    stack_ptr) tuples. This proxy presents the same attribute names as
    angr's CallStack plugin so user code can read frames uniformly.
    """

    __slots__ = ("call_site_addr", "func_addr", "ret_addr", "stack_ptr",
                 "_index", "_owner")

    def __init__(self, frame_tuple, index, owner):
        call_site_addr, callee_addr, return_addr, stack_ptr = frame_tuple
        self.call_site_addr = call_site_addr
        self.func_addr = callee_addr
        self.ret_addr = return_addr
        self.stack_ptr = stack_ptr
        self._index = index
        self._owner = owner

    @property
    def current_function_address(self):
        return self.func_addr

    @property
    def current_return_target(self):
        return self.ret_addr

    @property
    def current_stack_pointer(self):
        return self.stack_ptr

    @property
    def jumpkind(self):
        return "Ijk_Call"

    @property
    def next(self):
        """Walk one frame down the stack (toward the bottom)."""
        next_index = self._index + 1
        if next_index >= len(self._owner._frames):
            return None
        return RustCallStackFrameProxy(
            self._owner._frames[next_index], next_index, self._owner
        )

    def __repr__(self):
        return (f"<RustCallStackFrame func=0x{self.func_addr:x} "
                f"ret=0x{self.ret_addr:x} sp=0x{self.stack_ptr:x}>")


class RustCallStackProxy:
    """Iterable callstack view of a Rust state.

    Frames are stored as a list with the most-recent (top) frame first,
    matching angr's CallStack iteration order. The Rust engine stores
    frames in push order (top last), so we reverse on construction.
    """

    def __init__(self, rust_mgr, state_id):
        self._mgr = rust_mgr
        self._state_id = state_id
        self._frames_cache = None

    @property
    def _frames(self):
        if self._frames_cache is None:
            try:
                raw = self._mgr.get_state_call_stack(self._state_id)
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: empty callstack when FFI fails.
                # Distinguishable from a real empty stack only via the log.
                l.debug("get_state_call_stack(sid=%d) failed: %s: %s",
                        self._state_id, type(e).__name__, e)
                raw = []
            # Rust pushes onto the end → most recent is last → reverse.
            self._frames_cache = list(reversed(raw))
        return self._frames_cache

    def __iter__(self):
        for i, frame in enumerate(self._frames):
            yield RustCallStackFrameProxy(frame, i, self)

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, k):
        if k < 0:
            k += len(self._frames)
        if k < 0 or k >= len(self._frames):
            raise IndexError(k)
        return RustCallStackFrameProxy(self._frames[k], k, self)

    @property
    def top(self):
        if not self._frames:
            return None
        return RustCallStackFrameProxy(self._frames[0], 0, self)

    @property
    def current_function_address(self):
        if not self._frames:
            return 0
        return self._frames[0][1]  # callee_addr

    @property
    def current_return_target(self):
        if not self._frames:
            return 0
        return self._frames[0][2]  # return_addr

    @property
    def current_stack_pointer(self):
        if not self._frames:
            return 0
        return self._frames[0][3]  # stack_ptr

    @property
    def func_addr(self):
        return self.current_function_address

    @property
    def ret_addr(self):
        return self.current_return_target

    @property
    def stack_ptr(self):
        return self.current_stack_pointer

    @property
    def call_site_addr(self):
        if not self._frames:
            return 0
        return self._frames[0][0]

    def __repr__(self):
        return f"<RustCallStackProxy depth={len(self)}>"


_INSPECT_NOT_IMPLEMENTED_MSG = (
    "state.inspect breakpoints are not dispatched by the Rust symex engine. "
    "Registering a breakpoint here would silently never fire. "
    "Switch to the Python engine (use_rust_engine=False) for breakpoint-driven "
    "analyses. See docs/advanced-topics/rust_engine.rst for details."
)

_COPY_NOT_IMPLEMENTED_MSG = (
    "RustStateProxy.copy() is not supported in v1.0. The Rust engine owns "
    "this state's solver/memory/registers, and a faithful CoW deep copy "
    "requires a Rust-side fork plus per-state metadata duplication that "
    "has not yet landed (tracked under bd angr-2zwy). The previous shallow "
    "copy aliased _state_id with the source and silently corrupted the "
    "parent on any mutation (see docs/advanced-topics/rust_engine.rst — "
    "the Veritesting row under 'Analyses compatibility'). To work around: "
    "drop to the Python engine (use_rust_engine=False) for code that needs "
    "state.copy()."
)


# Single source of truth for the Rust engine's state.inspect dispatch.
#
# Each entry maps an angr `event_types` member to a spec dict with:
#   - bit: position in the Rust callbacks `inspect_enabled` u32 bitmask
#   - attrs: SimInspector attribute names this event populates
#   - when_fired: 'before' or 'after' — when in the Rust pipeline it fires
#   - dispatch_origin: 'rust' (default) when the Rust engine invokes the
#     callback via PythonCallbacks; 'python' when dispatch is fired from
#     within an existing Python-side callback handler in
#     `rust_callback_dispatch.py` / `rust_manager.py`. Python-dispatched
#     events do not need a PythonCallbacks slot because no Rust code path
#     ever calls into the inspect callback for them.
#
# Bits 0..=5 mirror `crate::state::InspectEvent` ordering; bit 4 (fork) is
# reserved for future wiring; bits 6 and 7 are custom (instruction / irsb)
# with no InspectionManager enum slot in Rust; bits 8 and 9 are custom for
# call / return (angr-4ai9 widened the bitmask from u8 to u16 to make
# room — the InspectEvent enum is unchanged). Bits 10..=12 are
# Python-dispatched (simprocedure / syscall / dirty) — angr-xmfj wired
# them from the existing Python callback handlers in
# `rust_callback_dispatch.py` / `rust_manager._cb_dirty_call`.
#
# Wiring a new Rust-origin event requires (mirror the angr-d46u 5-touchpoint
# pattern):
#   1. Add a row here with a unique bit, attrs, when_fired.
#   2. Add a PythonCallbacks slot + setter + dispatch helper in
#      native/angr/src/callbacks.rs and the interpreter dispatch
#      helpers.
#   3. Instrument the corresponding interpreter site in
#      native/angr/src/interpreter/.
#   4. Add a `_cb_inspect_<name>` method on RustExplorationManager
#      that builds the attrs dict and calls `_dispatch_inspect_event`.
#   5. Register the callback in RustExplorationManager.set_callbacks().
#
# For a Python-dispatched event (`dispatch_origin: 'python'`), steps 2/3
# collapse to "call self._cb_inspect_<name>(...) from the existing Python
# handler site". No Rust changes are needed.
#
# The CI test `test_inspect_allowlist_complete_and_consistent` enforces
# that every event in `angr.state_plugins.inspect.event_types` is either
# present here with a `_cb_inspect_<name>` method, or registering a
# breakpoint for it raises NotImplementedError. No silent pass-through.
_INSPECT_EVENT_SPECS: dict = {
    "mem_read": {
        "bit": 0,
        "attrs": (
            "mem_read_address",
            "mem_read_length",
            "mem_read_expr",
            "mem_read_condition",
            "mem_read_endness",
        ),
        "when_fired": "after",
    },
    "mem_write": {
        "bit": 1,
        "attrs": (
            "mem_write_address",
            "mem_write_length",
            "mem_write_expr",
            "mem_write_condition",
            "mem_write_endness",
        ),
        "when_fired": "after",
    },
    "reg_read": {
        "bit": 2,
        "attrs": (
            "reg_read_offset",
            "reg_read_length",
            "reg_read_expr",
            "reg_read_condition",
            "reg_read_endness",
        ),
        "when_fired": "after",
    },
    "reg_write": {
        "bit": 3,
        "attrs": (
            "reg_write_offset",
            "reg_write_length",
            "reg_write_expr",
            "reg_write_condition",
            "reg_write_endness",
        ),
        "when_fired": "after",
    },
    # angr-ysml: fork dispatch fires from `exploration/stepping.rs` for
    # each forked state created by the deferred-fork processing in
    # `handle_block_end` and `process_deferred_forks_into`. Matches
    # Python's `engines/successors.py:203` where `state._inspect("fork",
    # BP_AFTER)` fires on the newly-added successor after constraints +
    # ip are applied. The BP fires BEFORE the Rust-side satisfiability
    # check so UNSAT-pruned forks still surface — same pre-discard
    # intent as Python's add_successor flow. Has NO attrs in
    # `inspect_attributes` (verified against state_plugins/inspect.py);
    # the BP just sees the forked state's id via the proxy.
    "fork": {
        "bit": 4,
        "attrs": (),
        "when_fired": "after",
    },
    "exit": {
        "bit": 5,
        "attrs": (
            "exit_target",
            "exit_guard",
            "exit_jumpkind",
        ),
        "when_fired": "before",
    },
    "instruction": {
        "bit": 6,
        "attrs": (
            "instruction",
        ),
        "when_fired": "before",
    },
    "irsb": {
        "bit": 7,
        "attrs": (
            "address",
        ),
        "when_fired": "before",
    },
    # angr-4ai9: call/return dispatched from the BlockEnd path of
    # interpreter/execution.rs (Ijk_Call / Ijk_Ret). Fires `before` and
    # `after` around the Rust call_stack push/pop, matching Python
    # callstack.py:386/419 (call) and :430/432 (return). Only attribute is
    # `function_address` (the resolved call target on call; the popped
    # frame's callee on return).
    "call": {
        "bit": 8,
        "attrs": (
            "function_address",
        ),
        "when_fired": "before",
    },
    "return": {
        "bit": 9,
        "attrs": (
            "function_address",
        ),
        "when_fired": "before",
    },
    # angr-xmfj: simprocedure / syscall / dirty dispatch fires from the
    # existing Python callback sites in `rust_callback_dispatch.py`
    # (_handle_simprocedure_callback, _handle_syscall_callback_inner) and
    # `rust_manager._cb_dirty_call`. The Rust engine never invokes these
    # inspect callbacks directly — `dispatch_origin: 'python'` flags that
    # no PythonCallbacks slot is needed. The bit is still flipped in
    # `_update_inspect_bitmask` for consistency, even though Rust doesn't
    # read it. Attrs mirror the Python engine's _inspect call signatures
    # (sim_procedure.py:246/314 for simprocedure, procedure.py:34/50 for
    # syscall, engines/vex/heavy/inspect.py:10/22 for dirty). MVP scope:
    # the BP fires with the engine's chosen handler/args/result; user
    # mutations to those attrs in BP_BEFORE actions do NOT influence the
    # engine (Python's _inspect_getattr override path is not honored).
    "simprocedure": {
        "bit": 10,
        "attrs": (
            "simprocedure_name",
            "simprocedure_addr",
            "simprocedure",
            "simprocedure_result",
        ),
        "when_fired": "before",
        "dispatch_origin": "python",
    },
    "syscall": {
        "bit": 11,
        "attrs": (
            "syscall_name",
            "simprocedure",
        ),
        "when_fired": "before",
        "dispatch_origin": "python",
    },
    "dirty": {
        "bit": 12,
        "attrs": (
            "dirty_name",
            "dirty_handler",
            "dirty_args",
            "dirty_result",
        ),
        "when_fired": "before",
        "dispatch_origin": "python",
    },
    # angr-64pi: tmp_read / tmp_write dispatch fires from the Rust
    # interpreter's `RdTmp` / `WrTmp` arms (interpreter/expressions.rs
    # and interpreter/statements.rs). Each `RdTmp` evaluation pays a
    # single bitmask test in the no-BP case; with a BP the tmp's value
    # is round-tripped to a claripy AST and dispatched as
    # `tmp_read_expr`. `tmp_write` follows the same pattern in the
    # `WrTmp` arm, firing `when='after'` after the value is computed
    # but BEFORE the slot is mutated so the BP sees the value going in.
    # Both events fire many times per IRSB (every binop arg goes
    # through `RdTmp`); the bitmask short-circuit keeps overhead at
    # one `AtomicU16::load` per dispatch site when no BP is set.
    "tmp_read": {
        "bit": 13,
        "attrs": (
            "tmp_read_num",
            "tmp_read_expr",
        ),
        "when_fired": "after",
    },
    "tmp_write": {
        "bit": 14,
        "attrs": (
            "tmp_write_num",
            "tmp_write_expr",
        ),
        "when_fired": "after",
    },
    # angr-t8vf: statement dispatch fires from `execute_block_with_callbacks`
    # in `interpreter/execution.rs`, once per VEX IR statement, before the
    # statement runs. The only attr is `statement` (the integer index into
    # `irsb.statements`) — matches Python's `SimInspectMixin._handle_vex_stmt`
    # BP_BEFORE call signature. BP_AFTER is not wired (same MVP gap as
    # `instruction` BP_AFTER). Bit 15 is the LAST free slot in the
    # `AtomicU16` bitmask; the companion `expr` event widened
    # `inspect_enabled` to `AtomicU32` in angr-lge2.
    "statement": {
        "bit": 15,
        "attrs": (
            "statement",
        ),
        "when_fired": "before",
    },
    # angr-lge2: expr dispatch fires from `eval_expr_with_callbacks` in
    # `interpreter/expressions.rs`, once per IR expression evaluation
    # (every constant, RdTmp, Get, Load, unop/binop arg, ITE, etc.).
    # `when='after'`, with `expr_result` set to the claripy-reconstructed
    # value of the expression. `expr` itself is passed as `None` — Rust
    # IRExpr doesn't round-trip cleanly into `pyvex.IRExpr`, and the BP
    # would otherwise need an expensive lift on every eval. Bit 16
    # required widening the bitmask from `AtomicU16` to `AtomicU32` (the
    # u16 had filled at bit 15 with `statement`). The bitmask short-circuit
    # is load-bearing for this dispatch site: it's the highest-frequency
    # call in the engine, so the no-BP cost MUST stay at one atomic load.
    # User mutations to `expr_result` are NOT honored (MVP gap pattern).
    "expr": {
        "bit": 16,
        "attrs": (
            "expr",
            "expr_result",
        ),
        "when_fired": "after",
    },
    # angr-vfst: address_concretization dispatch fires from
    # `interpreter/expressions.rs::load_symbolic_addr` (read path) and
    # `interpreter/statements.rs::try_rust_memory_store` (write path) when
    # the load/store address is symbolic. Fires both `before` (with the
    # symbolic addr AST, `result=None`) and `after` (with the concretized
    # `addr_concretization_result`). Strategy / memory / add_constraints
    # attrs are passed as None — the Rust engine has no SimMemory instance
    # or strategy stack to surface to the BP (MVP gap; user mutations to
    # those attrs in BP_BEFORE are not honored, matching the wider Rust
    # inspect MVP scope documented in `rust_engine.rst`).
    "address_concretization": {
        "bit": 17,
        "attrs": (
            "address_concretization_strategy",
            "address_concretization_action",
            "address_concretization_memory",
            "address_concretization_expr",
            "address_concretization_result",
            "address_concretization_add_constraints",
        ),
        "when_fired": "before",
    },
    # angr-vfst: symbolic_variable dispatch fires from
    # `interpreter/mod.rs::load_from_callback` when the engine mints a
    # fresh BVS for an unconstrained memory load (Python returned
    # `is_symbolic=True` with no AST). Fires `when='after'` with
    # `symbolic_name`, `symbolic_size`, and `symbolic_expr` mirroring
    # Python's `solver.py:432-439` BP_AFTER signature. The user-callable
    # `state.solver.BVS()` path still fires the same event from Python
    # natively, independent of this Rust dispatch — the Rust path covers
    # only fresh-BVS minting that originates inside the engine.
    "symbolic_variable": {
        "bit": 18,
        "attrs": (
            "symbolic_name",
            "symbolic_size",
            "symbolic_expr",
        ),
        "when_fired": "after",
    },
}

# Derived views — DO NOT add entries here; edit _INSPECT_EVENT_SPECS instead.
_RUST_INSPECT_SUPPORTED_EVENTS = frozenset(_INSPECT_EVENT_SPECS)
_RUST_INSPECT_ATTRS_BY_EVENT = {
    name: spec["attrs"] for name, spec in _INSPECT_EVENT_SPECS.items()
}
_RUST_INSPECT_EVENT_BITS = {
    name: spec["bit"] for name, spec in _INSPECT_EVENT_SPECS.items()
}
# Events whose dispatch fires from Python (not from the Rust engine's
# PythonCallbacks invocation path). Used by `_setup_callbacks` to skip
# Rust slot registration and by the allowlist consistency test to relax
# the "must have a `set_inspect_<event>` PyO3 slot" requirement.
_RUST_INSPECT_PYTHON_DISPATCHED_EVENTS = frozenset(
    name for name, spec in _INSPECT_EVENT_SPECS.items()
    if spec.get("dispatch_origin") == "python"
)


def _format_unsupported_event_msg(event_type: str) -> str:
    """Build the NotImplementedError message for a non-honored inspect event."""
    supported = ", ".join(sorted(_RUST_INSPECT_SUPPORTED_EVENTS))
    return (
        f"Rust engine inspect dispatches {supported} events; "
        f"got event_type={event_type!r}. "
        "Drop to use_rust_engine=False for full state.inspect coverage."
    )


class _NoOpInspectProxy:
    """Stand-in for state.inspect on a RustStateProxy detached from any manager.

    Used only when a RustStateProxy is constructed without a python_mgr
    (mostly low-level unit tests). With a manager attached, the proxy
    routes `.inspect` to the manager's `RustInspectProxy` instead, which
    actually dispatches mem_read / mem_write events from the Rust engine.
    """

    def b(self, *args, **kwargs):
        raise NotImplementedError(_INSPECT_NOT_IMPLEMENTED_MSG)

    def make_breakpoint(self, *args, **kwargs):
        raise NotImplementedError(_INSPECT_NOT_IMPLEMENTED_MSG)

    def add_breakpoint(self, *args, **kwargs):
        raise NotImplementedError(_INSPECT_NOT_IMPLEMENTED_MSG)

    def remove_breakpoint(self, *args, **kwargs):
        raise NotImplementedError(_INSPECT_NOT_IMPLEMENTED_MSG)

    def action(self, *args, **kwargs):
        raise NotImplementedError(_INSPECT_NOT_IMPLEMENTED_MSG)


class RustInspectProxy:
    """state.inspect for Rust-engine states.

    Implements the subset of `SimInspector` needed for the mem_read /
    mem_write MVP. Other event types raise NotImplementedError at
    registration time to make the gap loud.

    Storage is centralized on the owning RustExplorationManager (a single
    set of breakpoints applies across all that manager's states), to keep
    the bitmask the Rust engine reads in sync with what's registered.
    The proxy itself is shared across all per-state RustStateProxy
    instances; `set_state` rebinds the `state` field that BP.check / fire
    receive when dispatch occurs for that state.

    This is an MVP deviation from Python's per-state inspect plugin
    semantics — see `docs/advanced-topics/rust_engine.rst`.
    """

    SUPPORTED_EVENTS = _RUST_INSPECT_SUPPORTED_EVENTS

    def __init__(self, mgr):
        # Manager-owned BP storage; the proxy is a thin facade so all
        # state proxies share the same dispatch set.
        self._mgr = mgr
        self.state = None
        self.action_attrs_set = False
        # Initialize every known inspect attribute to None so BP.check
        # doesn't AttributeError when reading attrs not set by the
        # current event.
        for attrs in _RUST_INSPECT_ATTRS_BY_EVENT.values():
            for a in attrs:
                setattr(self, a, None)

    @property
    def _breakpoints(self):
        """Compat with code that pokes into SimInspector._breakpoints."""
        return self._mgr._inspect_breakpoints

    def set_state(self, state):
        """Bind the state passed to BP.check / BP.fire during the next action().

        Called by the manager-side dispatcher right before invoking
        `action(...)` so the user's BP callable sees the per-state proxy.
        """
        self.state = state

    def b(self, event_type, *args, **kwargs):
        """Alias for make_breakpoint, matching SimInspector.b."""
        return self.make_breakpoint(event_type, *args, **kwargs)

    def make_breakpoint(self, event_type, *args, **kwargs):
        self._check_event(event_type)
        from angr.state_plugins.inspect import BP
        bp = BP(*args, **kwargs)
        self.add_breakpoint(event_type, bp)
        return bp

    def add_breakpoint(self, event_type, bp):
        self._check_event(event_type)
        self._mgr._inspect_breakpoints[event_type].append(bp)
        self._mgr._update_inspect_bitmask()

    def remove_breakpoint(self, event_type, bp=None, filter_func=None):
        self._check_event(event_type)
        if bp is None and filter_func is None:
            raise ValueError(
                'remove_breakpoint(): You must specify either "bp" or "filter".'
            )
        bps = self._mgr._inspect_breakpoints[event_type]
        if bp is not None:
            try:
                bps.remove(bp)
            except ValueError:
                pass
        else:
            self._mgr._inspect_breakpoints[event_type] = [
                b for b in bps if not filter_func(b)
            ]
        self._mgr._update_inspect_bitmask()

    def action(self, event_type, when, **kwargs):
        """Mirror SimInspector.action: stage attrs, walk BPs, fire matches.

        BP.fire() invokes the user action with `self.state` (set by the
        dispatcher just before this call). On reentrant calls (user action
        triggering another inspect event), staging clobbers attrs — same
        as Python's SimInspector. The Rust dispatch site guards against
        reentrant dispatch (uq4n.4).
        """
        self._check_event(event_type)
        self._set_inspect_attrs(**kwargs)
        self.action_attrs_set = True
        state = self.state
        try:
            for bp in list(self._mgr._inspect_breakpoints[event_type]):
                if not self.action_attrs_set:
                    self._set_inspect_attrs(**kwargs)
                    self.action_attrs_set = True
                if bp.check(state, when):
                    bp.fire(state)
        finally:
            self.action_attrs_set = False

    def _check_event(self, event_type):
        if event_type not in self.SUPPORTED_EVENTS:
            raise NotImplementedError(
                _format_unsupported_event_msg(event_type)
            )

    def _set_inspect_attrs(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class RustStateProxy:
    """
    Lightweight read-through proxy to a Rust symex state.

    Delegates property reads to Rust via PyO3 bindings. No caching,
    no state sync — reads directly from the Rust engine.

    Use this for:
    - ExplorationTechnique filter()/complete() callbacks
    - Callable find/avoid predicates
    - Found-state export and solution extraction

    Full SimState is only needed for SimProcedure execution.
    """

    def __init__(self, rust_mgr, state_id, project=None, stdin_vars=None,
                 stdout_data=None, python_mgr=None):
        self._mgr = rust_mgr
        self._state_id = state_id
        self._project = project
        self._stdin_vars = stdin_vars
        self._stdout_data = stdout_data or b""
        # High-level RustExplorationManager — used for options/globals lookup.
        # None when constructed standalone (e.g., low-level unit tests); in
        # that case options/globals fall back to empty stand-ins.
        self._python_mgr = python_mgr
        # Lazy-initialized sub-proxies
        self._solver_proxy = None
        self._regs_proxy = None
        self._mem_proxy = None
        self._history_proxy = None
        self._posix_proxy = None
        self._callstack_proxy = None
        self._inspect_proxy = None
        self._heap_proxy = None
        self._scratch_proxy = None

    @property
    def state_id(self):
        """Rust-side state identifier."""
        return self._state_id

    @property
    def addr(self):
        """Current program counter (O(1) via state index, no full export)."""
        if hasattr(self, '_override_addr') and self._override_addr is not None:
            return self._override_addr
        try:
            pc = self._mgr.get_state_pc_by_id(self._state_id)
            if pc is not None:
                return pc
        except Exception as e:
            # cat-(c) WRONG-ANSWER RISK: returning 0 when PC lookup fails can
            # spuriously match a find/avoid predicate that includes addr 0.
            # Log at warn so the failure is loud; callers reading proxy.addr
            # in critical paths will see the silent zero behavior in stderr.
            l.warning("RustStateProxy.addr(sid=%d) lookup failed: %s: %s",
                      self._state_id, type(e).__name__, e)
        return 0

    @property
    def ip(self):
        """Alias for addr, as claripy BVV."""
        return claripy.BVV(self.addr, self.arch.bits)

    @property
    def arch(self):
        """Architecture object from the project."""
        if self._project is not None:
            return self._project.arch
        # Fallback: get arch name from Rust and resolve
        arch_name = self._mgr.arch
        import archinfo
        return archinfo.arch_from_id(arch_name)

    @property
    def project(self):
        """The angr Project."""
        return self._project

    @property
    def solver(self):
        """Solver proxy — delegates to Rust Z3 via PyO3."""
        if self._solver_proxy is None:
            self._solver_proxy = RustSolverProxy(self._mgr, self._state_id)
        return self._solver_proxy

    @property
    def se(self):
        """Legacy alias for solver."""
        return self.solver

    @property
    def regs(self):
        """Register proxy — reads registers from Rust state."""
        if self._regs_proxy is None:
            self._regs_proxy = RustRegisterProxy(
                self._mgr, self._state_id, self.arch
            )
        return self._regs_proxy

    @property
    def registers(self):
        """Alias for regs."""
        return self.regs

    @property
    def memory(self):
        """Memory proxy — reads memory from Rust state."""
        if self._mem_proxy is None:
            self._mem_proxy = RustMemoryProxy(
                self._mgr, self._state_id, self.arch
            )
        return self._mem_proxy

    @property
    def mem(self):
        """Alias for memory."""
        return self.memory

    @property
    def history(self):
        """History proxy."""
        if self._history_proxy is None:
            self._history_proxy = RustHistoryProxy(
                self._mgr, self._state_id
            )
        return self._history_proxy

    @property
    def posix(self):
        """Posix proxy for stdin/stdout dumps."""
        if self._posix_proxy is None:
            self._posix_proxy = RustPosixProxy(
                self._mgr, self._state_id,
                stdin_vars=self._stdin_vars,
                stdout_data=self._stdout_data,
            )
        return self._posix_proxy

    @property
    def callstack(self):
        """Callstack proxy — iterable view of the Rust state's call frames.

        Top frame (most recent) first. Frame attrs match angr's CallStack
        plugin (call_site_addr, func_addr, ret_addr, stack_ptr).
        """
        if self._callstack_proxy is None:
            self._callstack_proxy = RustCallStackProxy(self._mgr, self._state_id)
        return self._callstack_proxy

    @property
    def inspect(self):
        """state.inspect for the Rust engine.

        If this proxy is attached to a RustExplorationManager, returns the
        manager-wide RustInspectProxy that supports mem_read / mem_write
        BP registration. Without a manager (low-level proxy construction),
        registration raises NotImplementedError via _NoOpInspectProxy.
        """
        if self._python_mgr is not None:
            return self._python_mgr._get_inspect_proxy()
        if self._inspect_proxy is None:
            self._inspect_proxy = _NoOpInspectProxy()
        return self._inspect_proxy

    @property
    def heap(self):
        """Heap proxy — exposes mmap_base and allocation tracking."""
        if self._heap_proxy is None:
            self._heap_proxy = RustHeapProxy(self._mgr, self._state_id)
        return self._heap_proxy

    @property
    def scratch(self):
        """Read-only scratch proxy — exposes bbl_addr / ins_addr / jumpkind.

        The Rust engine doesn't store the per-block VEX-temp/tyenv/stmt_idx
        values that the Python SimStateScratch plugin maintains, so those
        attributes return None / empty. The values that survive between
        blocks (block address and last jumpkind) read live from the Rust
        state.
        """
        if self._scratch_proxy is None:
            self._scratch_proxy = RustScratchProxy(self._mgr, self._state_id)
        return self._scratch_proxy

    @property
    def options(self):
        """Per-state options set, backed by the high-level manager.

        The Rust engine doesn't honor SimOptions (LAZY_SOLVES /
        STRICT_PAGE_ACCESS are mirrored separately on the Rust state), but
        user code reads `X in state.options` and writes `state.options.add(X)`.
        Returns a live set; mutations persist for this state_id.
        """
        if self._python_mgr is not None:
            return self._python_mgr.get_state_options_py(self._state_id)
        return set()

    @property
    def globals(self):
        """Per-state globals dict, backed by the high-level manager."""
        if self._python_mgr is not None:
            return self._python_mgr.get_state_globals_py(self._state_id)
        return {}

    def add_constraints(self, *constraints):
        """Add constraints to the solver."""
        self.solver.add(*constraints)

    def satisfiable(self, **kwargs):
        """Check if this state is satisfiable."""
        return self.solver.satisfiable(**kwargs)

    def copy(self):
        """Forking a Rust-engine state via the proxy is unsupported in v1.0.

        The Python angr contract for ``SimState.copy()`` is a CoW deep fork:
        mutations on the copy must not affect the source. The Rust engine
        owns the per-state solver/memory/registers; producing a faithful
        deep copy would require routing through ``RustSimState::fork`` and
        plumbing the new state ID back through the manager's bookkeeping
        (stash, options dict, globals dict, stdout tracker). That work is
        tracked under bd ``angr-2zwy`` for a future iteration.

        Until then this method raises rather than silently returning the
        original-aliased shallow proxy that previously caused state
        corruption under ``Veritesting`` (see
        ``docs/advanced-topics/rust_engine.rst`` — Analyses compatibility
        row, item 1).
        """
        raise NotImplementedError(_COPY_NOT_IMPLEMENTED_MSG)

    def __repr__(self):
        try:
            stash = self._mgr.state_stash(self._state_id)
        except Exception:
            # cat-(a) EXPECTED CONTROL FLOW: __repr__ is best-effort; skip
            # missing stash field rather than fail repr().
            stash = None
        try:
            n_constraints = self._mgr.state_constraint_count(self._state_id)
        except Exception:
            # cat-(a) EXPECTED CONTROL FLOW: __repr__ is best-effort; skip
            # missing constraint count rather than fail repr().
            n_constraints = None
        parts = [f"id={self._state_id}", f"addr={hex(self.addr)}"]
        if stash is not None:
            parts.append(f"stash={stash}")
        if n_constraints is not None:
            parts.append(f"constraints={n_constraints}")
        return f"<RustStateProxy {' '.join(parts)}>"


class RustSimulationManagerProxy:
    """
    Presents a SimulationManager-like interface backed by Rust stashes.

    Used by ExplorationTechniques that need simgr.stashes, simgr.found, etc.
    States are returned as RustStateProxy objects (O(1) creation, no sync).
    """

    def __init__(self, rust_mgr, project=None, stdin_vars=None,
                 stdout_tracker=None, python_mgr=None):
        self._mgr = rust_mgr
        self._project = project
        self._stdin_vars = stdin_vars
        self._stdout_tracker = stdout_tracker or {}  # state_id -> bytes
        self._python_mgr = python_mgr
        self._errored = []
        # Wired by RustExplorationManager when dispatching ExplorationTechnique
        # step() hooks. The callback advances the Rust engine by one batch and
        # records the resulting ExplorationEvent. When None, step()/successors()
        # raise — they must not silently no-op once a tech overrides them.
        self._step_callback = None

    def _wrap_state(self, state_id):
        """Wrap a Rust state ID in a RustStateProxy."""
        return RustStateProxy(
            self._mgr, state_id,
            project=self._project,
            stdin_vars=self._stdin_vars,
            stdout_data=self._stdout_tracker.get(state_id, b""),
            python_mgr=self._python_mgr,
        )

    def _get_stash(self, name):
        """Get all states in a stash as proxy objects."""
        state_ids = self._mgr.get_state_ids(name)
        return [self._wrap_state(sid) for sid in state_ids]

    @property
    def active(self):
        return self._get_stash("active")

    @active.setter
    def active(self, states):
        l.warning("RustSimulationManagerProxy: setting active is not yet supported")

    @property
    def found(self):
        return self._get_stash("found")

    @property
    def deadended(self):
        return self._get_stash("deadended")

    @property
    def avoid(self):
        return self._get_stash("avoid")

    @property
    def errored(self):
        return self._errored

    @property
    def one_found(self):
        """Return first found state or None."""
        found = self.found
        return found[0] if found else None

    @property
    def one_active(self):
        """Return first active state or None."""
        active = self.active
        return active[0] if active else None

    @property
    def stashes(self):
        """Dict-like access to all stashes."""
        return _StashDict(self)

    @property
    def _stashes(self):
        """Direct stash dict access (for techniques that access simgr._stashes)."""
        return self.stashes

    def filter(self, state, filter_func=None):
        """Default filter — delegates to filter_func or returns None.

        Techniques call simgr.filter(state) to delegate to the next technique.
        """
        if filter_func is not None:
            return filter_func(state)
        return None

    def step(self, stash="active", **kwargs):
        """Advance the Rust engine by one batch.

        This is the base step impl that ExplorationTechnique.step() hooks
        wrap. The RustExplorationManager installs `_step_callback` before
        dispatching, so calling this directly without the callback set is a
        programming error — raise rather than silently no-op.
        """
        if self._step_callback is None:
            raise NotImplementedError(
                "RustSimulationManagerProxy.step() can only be called from "
                "inside an ExplorationTechnique step() dispatch (the owning "
                "RustExplorationManager wires the callback)."
            )
        self._step_callback(stash=stash, **kwargs)
        return self

    def step_state(self, state, **kwargs):
        """Categorise successors into stashes.

        Not supported on the Rust manager: producing a SimSuccessors object
        requires re-running the step from Python, which defeats the engine.
        Techniques that override step_state() should fall back to the Python
        engine (``use_rust_engine=False``).
        """
        raise NotImplementedError(
            "step_state() is not supported by RustSimulationManagerProxy "
            "(would require a Python-side re-run). Use use_rust_engine=False "
            "if a registered ExplorationTechnique relies on step_state()."
        )

    def successors(self, state, **kwargs):
        """Run one state forward, returning SimSuccessors.

        Not supported on the Rust manager — see step_state() docstring.
        """
        raise NotImplementedError(
            "successors() is not supported by RustSimulationManagerProxy "
            "(would require a Python-side re-run). Use use_rust_engine=False "
            "if a registered ExplorationTechnique relies on successors()."
        )

    def move(self, from_stash="active", to_stash="stashed", filter_func=None):
        """Move states between stashes."""
        if filter_func is None:
            count = self._mgr.move_states(from_stash, to_stash)
        else:
            # Move states that match filter
            state_ids = self._mgr.get_state_ids(from_stash)
            moved = 0
            for sid in state_ids:
                proxy = self._wrap_state(sid)
                if filter_func(proxy):
                    self._mgr.move_state(sid, from_stash, to_stash)
                    moved += 1
            count = moved
        return count

    def __repr__(self):
        counts = self._mgr.stash_counts()
        parts = [f"<RustSimulationManagerProxy"]
        for name, count in sorted(counts.items()):
            if count > 0:
                parts.append(f" {name}:{count}")
        parts.append(">")
        return "".join(parts)


class _StashDict:
    """Dict-like wrapper for stash access: simgr.stashes['found']."""

    def __init__(self, simgr_proxy):
        self._simgr = simgr_proxy

    def __getitem__(self, key):
        return self._simgr._get_stash(key)

    def __setitem__(self, key, value):
        if isinstance(value, list) and len(value) == 0:
            # Clear the stash — common pattern: simgr.stashes['found'] = []
            try:
                self._simgr._mgr.clear_stash(key)
            except Exception as e:
                # cat-(b) FALLBACK WITH LOSS: stash clear failed; user expects
                # the stash to be empty afterward but it may not be. Log so
                # this is visible under --debug.
                l.debug("clear_stash(%r) failed: %s: %s", key, type(e).__name__, e)
        else:
            l.warning("_StashDict.__setitem__ only supports clearing (empty list) for key '%s'", key)

    def __contains__(self, key):
        counts = self._simgr._mgr.stash_counts()
        return key in counts

    def keys(self):
        return self._simgr._mgr.stash_counts().keys()

    def values(self):
        return [self._simgr._get_stash(k) for k in self.keys()]

    def items(self):
        return [(k, self._simgr._get_stash(k)) for k in self.keys()]

    def get(self, key, default=None):
        try:
            return self[key]
        except Exception:
            # cat-(a) EXPECTED CONTROL FLOW: dict.get() contract — return the
            # caller-supplied default when the key is missing or unfetchable.
            return default
