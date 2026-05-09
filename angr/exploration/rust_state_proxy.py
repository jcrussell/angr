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
        self._mgr = rust_mgr
        self._state_id = state_id
        self._arch = arch
        self._cache = {}  # name -> claripy BVV/BVS

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
                self._cache[name] = claripy.BVS(f"reg_{name}_{self._state_id}", width)
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
            result = claripy.BVS(f"reg_{name}_{self._state_id}", width)
        else:
            result = claripy.BVV(val, width)
        self._cache[name] = result
        return result

    def _get_register_width(self, name):
        """Get the bit width for a named register."""
        # Common register widths by name pattern
        if self._arch.name in ("AMD64", "X86_64"):
            if name in ("rax", "rbx", "rcx", "rdx", "rsi", "rdi",
                        "rbp", "rsp", "rip", "r8", "r9", "r10",
                        "r11", "r12", "r13", "r14", "r15"):
                return 64
            if name in ("eax", "ebx", "ecx", "edx", "esi", "edi",
                        "ebp", "esp", "eip"):
                return 32
            if name in ("ax", "bx", "cx", "dx", "si", "di", "bp", "sp"):
                return 16
            if name in ("al", "ah", "bl", "bh", "cl", "ch", "dl", "dh"):
                return 8
        elif self._arch.name in ("X86",):
            if name in ("eax", "ebx", "ecx", "edx", "esi", "edi",
                        "ebp", "esp", "eip"):
                return 32
            if name in ("ax", "bx", "cx", "dx", "si", "di", "bp", "sp"):
                return 16
            if name in ("al", "ah", "bl", "bh", "cl", "ch", "dl", "dh"):
                return 8
        elif "ARM" in self._arch.name or "AARCH" in self._arch.name:
            if name.startswith("x") or name in ("sp", "lr", "pc"):
                return 64 if "64" in self._arch.name else 32
            if name.startswith("r") or name.startswith("w"):
                return 32
        elif "MIPS" in self._arch.name:
            if name.startswith("v") or name.startswith("a") or name.startswith("t") or name.startswith("s"):
                return 64 if "64" in self._arch.name else 32
        # Default: use architecture word size
        return self._arch.bits

    def load(self, reg_name_or_offset, size=None):
        """Load register by name."""
        if isinstance(reg_name_or_offset, str):
            return getattr(self, reg_name_or_offset)
        raise NotImplementedError("register load by offset not yet supported in proxy")


class RustMemoryProxy:
    """
    Provides `state.memory.load(addr, size)`-style access via Rust.

    Returns bytes as claripy BVVs for compatibility.
    """

    def __init__(self, rust_mgr, state_id, arch):
        self._mgr = rust_mgr
        self._state_id = state_id
        self._arch = arch

    def load(self, addr, size=None, endness=None, **kwargs):
        """Load memory from the Rust state."""
        if isinstance(addr, claripy.ast.Base):
            # Concrete address extraction
            if addr.concrete:
                addr = addr.concrete_value
            else:
                raise NotImplementedError(
                    "symbolic memory load not supported in proxy"
                )
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

    def store(self, addr, data, **kwargs):
        """Store not supported on proxy (read-only view)."""
        raise NotImplementedError(
            "memory store not supported on RustStateProxy (read-only)"
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


class RustHistoryProxy:
    """Provides state.history.recent_bbl_addrs and similar."""

    def __init__(self, rust_mgr, state_id):
        self._mgr = rust_mgr
        self._state_id = state_id
        self._bbl_addrs = None

    @property
    def recent_bbl_addrs(self):
        if self._bbl_addrs is None:
            snapshot = self._mgr.export_state(self._state_id)
            self._bbl_addrs = snapshot.get_history()
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
    "Switch to use_rust_engine=False or track this in beads angr-osuu."
)


class _NoOpInspectProxy:
    """Stand-in for state.inspect on RustStateProxy.

    The Rust engine does not surface inspect events, so any breakpoint
    registered here would silently never fire. Rather than letting that
    fail invisibly, registration methods raise NotImplementedError.
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
        """No-op inspect proxy. Breakpoint registration silently succeeds
        but never fires — the Rust engine doesn't surface inspect events.
        """
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
        """Create a shallow copy of the proxy (same Rust state)."""
        return RustStateProxy(
            self._mgr, self._state_id,
            project=self._project,
            stdin_vars=self._stdin_vars,
            stdout_data=self._stdout_data,
            python_mgr=self._python_mgr,
        )

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
