//! Python callback infrastructure for Rust VEX engine.
//!
//! This module provides the callback holder that allows Rust to call back into Python
//! for operations like memory access, hook execution, and syscall handling.
//!
//! # Callback + sync invariants
//!
//! This is the Rust side of the cross-language callback contract. Several
//! cross-cutting rules are documented in bd memories; the most load-bearing
//! ones for any future edit in this module are mirrored here so they live
//! next to their enforcement sites (mirroring pattern documented in bd
//! memory `invariant-doc-mirror-pattern-2026-06-01`).
//!
//! 1. **`avoid-silent-no-op-callback-fallbacks`** — every `call_*` method
//!    on [`PythonCallbacks`] MUST hard-error when the hook is `None`.
//!    The exemplar is [`PythonCallbacks::call_on_hook`]: it `ok_or_else`'s
//!    into `PyRuntimeError::new_err("on_hook callback not set")`. Silent
//!    `Ok(())` fallbacks (the removed `call_memory_store_symbolic_ast`)
//!    mask wiring bugs by making the engine appear to run while stores
//!    are silently dropped, which produces divergent Rust↔Python memory
//!    that is painful to debug. When adding a new `call_*`, copy the
//!    `ok_or_else` idiom — do NOT return `Ok(())` or default values when
//!    the hook is unset.
//!
//! 2. **`drop-terminal-vs-predicates`** — when callable `find` /
//!    `avoid` predicates are active, the Python-side
//!    `drop_terminal_states` MUST be `False`. This is enforced in
//!    `angr.exploration.RustExplorationManager` (Python) and is invisible
//!    to Rust, but the predicate callback paths in this module
//!    ([`RunResult::SymbolicBranch`] forks, and find/avoid predicate
//!    dispatch in `exploration/run_loop.rs`) assume the Python side
//!    keeps stdout-carrying exit() states alive until predicate
//!    evaluation runs. Examples like `sym-write` produce zero found
//!    states without this rule.
//!
//! 3. **`invariant-3tek2-replay-ordering`** — the Python-side
//!    `_create_state_for_callback` replays Rust-recorded dirty-page
//!    mutations in a strict order around `_install_rust_memory_proxy`
//!    and `_restore_symbolic_pages`. Rust signals the change set through
//!    the dirty-page bookkeeping but does NOT enforce ordering;
//!    re-ordering the Python helper without updating both ends will
//!    clobber NativeRead/NativeWrite symbolic bytes with concrete
//!    pointer-slot copies. Future changes to dirty-page tracking
//!    (`pending_store.rs`, prefetch invalidation) need to consider the
//!    Python replay sequencing.
//!
//! 4. **`invariant-callstack-sync-export-pipeline`** — any per-state
//!    sync helper added to the Rust manager (memory, registers,
//!    callstack, mmap_base, posix_brk) must be wired into all four
//!    Python export paths in `_materialize_single_state`
//!    (cached / parent-root / stepping / snapshot fallback). The
//!    callstack sync added 2026-05-07 follows that pattern. If a new
//!    FFI sync method is exposed here without updating all four paths,
//!    some stash configurations will silently miss the sync.
//!
//! 5. **`invariant-rust-solver-fallback-class`** — Python's
//!    `RustSolverFallback` (in `angr/exploration/rust_state_export.py`)
//!    owns the per-state Rust solver fallback wiring: cached forked
//!    context, original method handles, constraint sync counter, and a
//!    `_rust_fallback_attached` flag that guards against double-patching
//!    (which would cause infinite recursion since the second wrapper's
//!    "originals" would be the first wrapper). Any new FFI solver entry
//!    point added here should be considered for inclusion in
//!    `RustSolverFallback.attach()`.
//!
//! 6. **`avoid-hasattr-lazy-init`** — `hasattr(state, plugin_name)` on
//!    angr `SimState` triggers `__getattr__` and lazy-initializes the
//!    plugin (`heap` plugin = ~80 ms first-touch). The Python callback
//!    paths use `plugin_name in state.plugins` for the O(1) dict check
//!    instead. Anything Rust does that prompts Python plugin probing
//!    (e.g., new SimProcedure dispatch helpers) should preserve that
//!    discipline on the Python side.

use pyo3::class::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::symbolic::RustBV;

/// One entry in the result of a batched memory-load callback:
/// `(data_bytes, is_symbolic, symbolic_ast)`. Shared with
/// `interpreter::expressions` which consumes these slices to mint
/// per-load `RustBV` values.
pub(crate) type BatchLoadEntry = (Vec<u8>, bool, Option<Py<PyAny>>);

mod config;
mod dispatch;
mod events;
mod inspect;

pub(crate) use config::{BranchPolicy, DeferredFork, ExecutionConfig};
pub(crate) use events::{LoopExecutionEvent, RunErrorKind, RunResult};

/// Python callback holder for the Rust VEX engine.
///
/// This struct holds references to Python callback functions that the Rust
/// engine calls during execution for memory access, hooks, syscalls, etc.
///
/// # Invariants enforced by every `call_*` method
///
/// * **No silent `Ok(())` fallback.** See module-level invariant 1
///   (`avoid-silent-no-op-callback-fallbacks`). When a hook is `None`,
///   the `call_*` method MUST return `Err(PyRuntimeError)` rather than
///   no-op. The reference pattern is [`Self::call_on_hook`].
/// * **`Clone` + shared atomics.** `PythonCallbacks` is `Clone` because the
///   Rust exploration manager keeps a cloned copy of the original passed
///   in via `set_callbacks`. Any mutable state shared with Python after
///   clone (e.g. [`Self::inspect_enabled`]) MUST be wrapped in
///   `Arc<Atomic*>` so updates from the Python side remain visible to
///   the Rust copy.
#[pyclass]
#[derive(Clone)]
pub struct PythonCallbacks {
    /// Callback for memory loads: fn(addr: u64, size: u32) -> (bytes, is_symbolic, symbolic_ast?)
    pub memory_load: Option<Py<PyAny>>,
    /// Callback for memory stores: fn(addr: u64, data: bytes) -> None
    pub memory_store: Option<Py<PyAny>>,
    /// Callback for batched memory stores: fn(stores: list[tuple[int, bytes]]) -> None
    /// This is more efficient than individual stores when multiple stores can be batched.
    pub memory_store_batch: Option<Py<PyAny>>,
    /// Callback for batched memory loads: fn(loads: list[tuple[int, int]]) -> list[tuple[bytes, bool, object | None]]
    /// Each tuple in input is (address, size). Returns list of (data, is_symbolic, ast_or_none).
    /// This is more efficient than individual loads when multiple loads can be batched.
    pub memory_load_batch: Option<Py<PyAny>>,
    /// Callback for hook execution: fn(addr: u64) -> new_pc
    pub on_hook: Option<Py<PyAny>>,
    /// Callback for syscall handling: fn(num: u64) -> None
    pub on_syscall: Option<Py<PyAny>>,
    /// Callback for lifting a block: fn(addr: u64) -> irsb_json
    pub lift_block: Option<Py<PyAny>>,
    /// Callback for getting register value: fn(offset: u32, size: u32) -> (bytes, is_symbolic, symbolic_ast?)
    pub get_register: Option<Py<PyAny>>,
    /// Callback for setting register value: fn(offset: u32, data: bytes) -> None
    pub put_register: Option<Py<PyAny>>,
    /// Callback for dirty helper calls: fn(name: str, args: list\[int\], ret_ty_bits: int) -> (bytes, bool, object | None)
    /// This handles VEX dirty calls to helper functions (CPUID, RDTSC, etc.)
    pub dirty_call: Option<Py<PyAny>>,
    /// Callback for fetching a single 4KB page: fn(page_addr: u64) -> (bytes, permissions: u8, is_mapped: bool)
    /// This is used for on-demand page loading when Rust memory encounters an unmapped page.
    pub fetch_page: Option<Py<PyAny>>,
    /// Callback for batched page fetching: fn(page_addrs: list[u64]) -> list[(bytes, u8, bool)]
    /// Returns list of (data, permissions, is_mapped) for each requested page.
    pub batch_fetch_pages: Option<Py<PyAny>>,
    /// Callback for storing a symbolic value with full expression tree: fn(addr: int, ast: claripy.AST) -> None
    /// This is called when storing a symbolic value to memory. The AST is reconstructed from
    /// the Rust expression tree, preserving the original symbolic expression structure.
    /// This allows symbolic values to be properly stored without data loss.
    pub memory_store_symbolic_value: Option<Py<PyAny>>,
    /// Callback for storing symbolic data at a symbolic address.
    /// Called when the address cannot be concretized (too many possibilities).
    /// Takes (addr_ast: claripy.AST, data_ast: claripy.AST) -> None
    /// Python should use state.memory.store(addr_ast, data_ast).
    pub memory_store_symbolic_full: Option<Py<PyAny>>,
    /// Callback for loading data at a symbolic address.
    /// Called when the address cannot be concretized (too many possibilities).
    /// Takes (addr_ast: claripy.AST, size: int) -> claripy.AST
    /// Python should use state.memory.load(addr_ast, size) and return the result.
    pub memory_load_symbolic_full: Option<Py<PyAny>>,
    /// Callback for resolving unmodeled function calls.
    /// Called when execution reaches a function that isn't hooked or modeled.
    /// Takes (addr: int, name: str | None) -> (name: str, num_args: int, no_return: bool) | None
    /// If returns None, the function is truly unmodeled and state should be deadended.
    /// If returns info, Rust will register the function as a SimProcedure and retry.
    pub resolve_function: Option<Py<PyAny>>,
    /// Callback for state.inspect mem_read events.
    /// Signature:
    ///   fn(state_id: int, when: str, addr: int, size: int,
    ///      value_ast: object | None, endness: str) -> None
    /// `when` is "before" or "after"; on BEFORE `value_ast` is None,
    /// on AFTER it holds the loaded value (concrete value as int or
    /// claripy AST). `endness` is "Iend_LE" or "Iend_BE".
    /// Only dispatched when bit InspectEvent::MemRead in `inspect_enabled` is set.
    pub inspect_mem_read: Option<Py<PyAny>>,
    /// Callback for state.inspect mem_write events.
    /// Signature mirrors `inspect_mem_read` but `value_ast` is the data
    /// being stored (set on BEFORE and AFTER).
    pub inspect_mem_write: Option<Py<PyAny>>,
    /// Callback for state.inspect reg_read events.
    /// Signature:
    ///   fn(state_id: int, when: str, offset: int, size: int,
    ///      value_ast: object | None) -> None
    /// `value_ast` carries the loaded register value on AFTER (None on BEFORE).
    /// Only dispatched when bit InspectEvent::RegRead in `inspect_enabled` is set.
    pub inspect_reg_read: Option<Py<PyAny>>,
    /// Callback for state.inspect reg_write events.
    /// Signature mirrors `inspect_reg_read`; `value_ast` is the value being
    /// written to the register file.
    pub inspect_reg_write: Option<Py<PyAny>>,
    /// Callback for state.inspect instruction events (one per IMark).
    /// Signature: fn(state_id: int, when: str, addr: int) -> None.
    /// Fired BEFORE the instruction's VEX statements execute.
    pub inspect_instruction: Option<Py<PyAny>>,
    /// Callback for state.inspect irsb events (one per basic block).
    /// Signature: fn(state_id: int, when: str, addr: int) -> None.
    /// Fired BEFORE the first statement of the IRSB.
    pub inspect_irsb: Option<Py<PyAny>>,
    /// Callback for state.inspect exit events (conditional VEX exits).
    /// Signature: fn(state_id: int, when: str, target: int, jumpkind: str,
    ///                guard_ast: object) -> None.
    /// Fired BEFORE the branch is taken. `guard_ast` is the symbolic guard
    /// condition (claripy AST). `jumpkind` is the VEX Ijk_* tag name.
    pub inspect_exit: Option<Py<PyAny>>,
    /// Callback for state.inspect call events (Ijk_Call dispatch).
    /// Signature: `fn(state_id: int, when: str, function_address: int) -> None`.
    /// Fires twice per call — once `when="before"` (with the resolved call
    /// target), once `when="after"` (after the Rust call_stack has been
    /// pushed). Matches Python's callstack.py:386/419 semantics.
    pub inspect_call: Option<Py<PyAny>>,
    /// Callback for state.inspect return events (Ijk_Ret dispatch).
    /// Signature: `fn(state_id: int, when: str, function_address: int) -> None`.
    /// Fires twice per return — once `when="before"` with the func_addr of
    /// the frame about to be popped, once `when="after"` after the pop.
    /// Matches Python's callstack.py:430/432 semantics.
    pub inspect_return: Option<Py<PyAny>>,
    /// Callback for state.inspect tmp_read events (VEX `RdTmp`).
    /// Signature: `fn(state_id: int, when: str, tmp_num: int,
    ///                value_ast: object | None) -> None`.
    /// Fired `when="after"` with the tmp's stored value as `tmp_read_expr`.
    /// Gated on `inspect_event_enabled(13)` so the no-breakpoint case costs
    /// one bitmask test per `RdTmp` evaluation.
    pub inspect_tmp_read: Option<Py<PyAny>>,
    /// Callback for state.inspect tmp_write events (VEX `WrTmp`).
    /// Signature mirrors `inspect_tmp_read`; `value_ast` is the value being
    /// stored into the tmp. Fired `when="after"` once the tmp slot has
    /// been written.
    pub inspect_tmp_write: Option<Py<PyAny>>,
    /// Callback for state.inspect statement events (per VEX IR statement).
    /// Signature: `fn(state_id: int, when: str, stmt_idx: int) -> None`.
    /// Fired `when="before"` from `execute_block_with_callbacks` just before
    /// each statement runs. Gated on `inspect_event_enabled(15)`.
    pub inspect_statement: Option<Py<PyAny>>,
    /// Callback for state.inspect expr events (per VEX IR expression eval).
    /// Signature: `fn(state_id: int, when: str, expr_result: object | None)
    /// -> None`. Fired `when="after"` from `eval_expr_with_callbacks` once
    /// the expression has been reduced to a value. `expr` itself is passed
    /// as `None` because Rust IRExpr does not round-trip cleanly into a
    /// pyvex.IRExpr; the BP receives only the computed `expr_result` (the
    /// RustBV reconstructed as a claripy AST). User mutations to
    /// `expr_result` in BP_AFTER actions are not honored — same MVP gap
    /// as the other inspect events. Gated on `inspect_event_enabled(16)`;
    /// this dispatch site fires more often than any other (every VEX
    /// expression evaluation), so the bitmask short-circuit is critical.
    pub inspect_expr: Option<Py<PyAny>>,
    /// Callback for state.inspect address_concretization events.
    /// Signature: `fn(state_id: int, when: str, action: str,
    ///                addr_ast: object, result: list\[int\] | None) -> None`.
    /// `action` is "load" or "store"; `addr_ast` is the symbolic address
    /// AST (claripy reconstruction); `result` is the list of concrete
    /// addresses the concretizer produced (None on BEFORE). MVP gap: the
    /// `address_concretization_strategy` / `_memory` / `_add_constraints`
    /// attrs from the Python event are passed through as `None` — the
    /// Rust engine doesn't expose strategy objects or the SimMemory
    /// instance to the BP. Gated on `inspect_event_enabled(17)`.
    pub inspect_address_concretization: Option<Py<PyAny>>,
    /// Callback for state.inspect symbolic_variable events.
    /// Signature: `fn(state_id: int, when: str, name: str, size: int,
    ///                expr_ast: object) -> None`. Fires `when='after'`
    /// when the Rust engine mints a fresh BVS for an unconstrained
    /// memory load (`load_from_callback` fallback path). The user-visible
    /// `state.solver.BVS()` path still fires the event from Python
    /// natively; this Rust dispatch is for the BVS minted internally by
    /// the engine when Python returned `is_symbolic=True` with no AST.
    /// Gated on `inspect_event_enabled(18)`.
    pub inspect_symbolic_variable: Option<Py<PyAny>>,
    /// Callback for state.inspect fork events.
    /// Signature: `fn(state_id: int, when: str) -> None`. Fires
    /// `when='after'` for each forked state created by the deferred-fork
    /// processing in `exploration/stepping.rs` (both `handle_block_end`
    /// and `process_deferred_forks_into`). The dispatch fires on the
    /// FORKED state's id (matching Python `engines/successors.py:203`
    /// where `state` is the newly-added successor), not the original
    /// state being forked from. UNSAT-pruned forks still fire the BP
    /// before the satisfiability check so the user sees every fork
    /// attempt — same intent as Python's pre-discard fire. The `fork`
    /// event takes NO attrs in `inspect_attributes`; the dispatch is
    /// state_id + when only. Gated on `inspect_event_enabled(4)`.
    pub inspect_fork: Option<Py<PyAny>>,
    /// `state.inspect.constraints` dispatcher (angr-op0dn.14.4.1).
    ///
    /// Signature: `fn(state_id: int, when: str, added_constraints: list) -> Any`.
    /// Fired from the native fork-guard add sites (see
    /// `exploration::helpers::add_fork_guard_constraint`) around the
    /// `assume_true` / `assume_false` that installs a branch guard on the
    /// continuing state, mirroring Python's `state_plugins/solver.py::add`.
    /// The Python endpoint is `RustExplorationManager._cb_inspect_constraints`,
    /// shared with the `RustSolverProxyPlugin.add` dispatch. Gated on
    /// `inspect_event_enabled(19)`. The BP's return value (mutated
    /// `added_constraints`) is honored by the proxy-add path but NOT by this
    /// native path — the guard is already lowered into a `RustBV`.
    pub inspect_constraints: Option<Py<PyAny>>,
    /// `state.inspect.vex_lift` dispatcher for the NATIVE lift path
    /// (angr-op0dn.14.4.2).
    ///
    /// Signature: `fn(state_id: int, when: str, addr: int, size: int | None,
    /// buff: bytes | None) -> None`. Fired from `try_native_lift`
    /// (`interpreter/execution.rs`) when the feature-gated in-process libVEX
    /// lifter serves a block — that path bypasses `_cb_lift_block`, which is
    /// where the Python-lift dispatch of this event lives. The Python
    /// endpoint, `RustExplorationManager._cb_inspect_vex_lift`, is shared by
    /// both origins. Gated on `inspect_event_enabled(20)`. The BP_BEFORE pair
    /// member is fired only once the native lift has succeeded, so user
    /// mutation of `vex_lift_buff` / `vex_lift_addr` is not honored here (the
    /// bytes are already lifted); firing it eagerly would double-fire BEFORE
    /// whenever the native lift misses and the Python callback re-fires it.
    pub inspect_vex_lift: Option<Py<PyAny>>,
    /// Bitmask of enabled inspect events. Bit N = `InspectEvent` variant N.
    /// VEX dispatch sites read this with a single `& != 0` check before
    /// touching any payload — keeps the cost of inspect-disabled
    /// execution at one branch per Load/Store.
    /// Python writes via `set_inspect_enabled`; defaults to 0 (off).
    ///
    /// Wrapped in `Arc<AtomicU32>` because `PythonCallbacks` is `Clone` and
    /// the Rust exploration manager stores a CLONED copy after Python
    /// passes the original in via `set_callbacks`. Bitmask updates from
    /// Python (`mgr._callbacks.set_inspect_enabled(...)`) must be visible
    /// to the Rust side; sharing the atomic makes both copies read/write
    /// the same word. Widened from `AtomicU8` in angr-4ai9 (added bits 8/9
    /// for call/return), then from `AtomicU16` in angr-lge2 so `expr`
    /// (bit 16) fits after `statement` filled bit 15.
    pub inspect_enabled: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// True when the callback-memory-proxy gate is on, i.e. the Python
    /// callback state's `state.memory` *is* Rust memory (`RustMemoryProxy`).
    ///
    /// Under that gate the `memory_store` / `memory_store_symbolic_value`
    /// callbacks are deliberate no-ops — re-entering `run()` through the proxy
    /// would double-borrow the manager — so any store the interpreter
    /// dispatches *only* to a callback is dropped outright (angr-5rjbq:
    /// flareon2015_5's base64 output vanished and the solve went UNSAT). The
    /// store paths consult this flag and buffer such stores for `rust_memory`
    /// instead. Ungated, the Python shadow really does absorb the store, so
    /// the extra write is pure cost — hence the flag rather than always
    /// buffering.
    ///
    /// `Arc<AtomicBool>` for the same reason as `inspect_enabled`: Python sets
    /// it after the manager has already cloned this struct.
    pub memory_is_rust_proxy: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// The set of page addresses `fetch_page` / `batch_fetch_pages` could
    /// actually serve, snapshotted from the Python state at setup (angr-gorvf.4.6).
    ///
    /// `None` means "unknown" — every fetch crosses into Python, the legacy
    /// behaviour. When `Some`, a page outside the set is DECLINED natively and
    /// the callback is never invoked: Python's page universe is frozen after
    /// `RustStateSyncMixin._sync_extra_python_pages` hands Rust every page it
    /// can serve, so a page not in the snapshot is one Python would decline
    /// (it is symbolic, or default-fill under a non-zero-fill state) anyway.
    /// Python installs the snapshot only when it can prove the verdict for
    /// every page (UltraPage backend, no `ZERO_FILL_UNCONSTRAINED_MEMORY`);
    /// otherwise it leaves this `None`.
    ///
    /// Declining is the conservative answer — Rust then serves the page from
    /// its own `SymbolicMemory`, which already holds the symbolic regions
    /// imported at setup. It must never be turned into a native zero-fill:
    /// these pages carry symbolic stdin (see the `fetch-page-symbolic-not-concrete`
    /// memory).
    ///
    /// `Arc<RwLock<..>>` for the same reason as `inspect_enabled`: Python
    /// installs it after the manager has already cloned this struct.
    pub python_servable_pages:
        std::sync::Arc<std::sync::RwLock<Option<std::collections::HashSet<u64>>>>,

    /// Every page Python holds a page object for (angr-gorvf.4.7) — a strict
    /// superset of `python_servable_pages`.
    ///
    /// The two answer different questions and must not be conflated.
    /// `python_servable_pages` is "could `fetch_page` hand back concrete bytes
    /// for this whole page", so a page carrying symbolic data is *declined*
    /// there even though Python can answer a `memory_load` from it perfectly
    /// well. This set is the load-side oracle: a page absent from it is one
    /// Python has no data for at all, so a load can only produce an
    /// unconstrained filler — which Rust can mint itself.
    pub python_page_universe:
        std::sync::Arc<std::sync::RwLock<Option<std::collections::HashSet<u64>>>>,
}

#[pymethods]
impl PythonCallbacks {
    /// Create a new empty callback holder.
    #[new]
    pub fn new() -> Self {
        PythonCallbacks {
            memory_load: None,
            memory_store: None,
            memory_store_batch: None,
            memory_load_batch: None,
            on_hook: None,
            on_syscall: None,
            lift_block: None,
            get_register: None,
            put_register: None,
            dirty_call: None,
            fetch_page: None,
            batch_fetch_pages: None,
            memory_store_symbolic_value: None,
            memory_store_symbolic_full: None,
            memory_load_symbolic_full: None,
            resolve_function: None,
            inspect_mem_read: None,
            inspect_mem_write: None,
            inspect_reg_read: None,
            inspect_reg_write: None,
            inspect_instruction: None,
            inspect_irsb: None,
            inspect_exit: None,
            inspect_call: None,
            inspect_return: None,
            inspect_tmp_read: None,
            inspect_tmp_write: None,
            inspect_statement: None,
            inspect_expr: None,
            inspect_address_concretization: None,
            inspect_symbolic_variable: None,
            inspect_fork: None,
            inspect_constraints: None,
            inspect_vex_lift: None,
            inspect_enabled: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            memory_is_rust_proxy: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            python_servable_pages: std::sync::Arc::new(std::sync::RwLock::new(None)),
            python_page_universe: std::sync::Arc::new(std::sync::RwLock::new(None)),
        }
    }

    /// Set the memory load callback.
    ///
    /// The callback should have signature:
    /// `fn(addr: int, size: int) -> tuple[bytes, bool, object | None]`
    ///
    /// Returns (concrete_bytes, is_symbolic, symbolic_ast_or_none).
    pub fn set_memory_load(&mut self, cb: Py<PyAny>) {
        self.memory_load = Some(cb);
    }

    /// Set the memory store callback.
    ///
    /// The callback should have signature:
    /// `fn(addr: int, data: bytes) -> None`
    pub fn set_memory_store(&mut self, cb: Py<PyAny>) {
        self.memory_store = Some(cb);
    }

    /// Set the batched memory store callback.
    ///
    /// The callback should have signature:
    /// `fn(stores: list[tuple[int, bytes]]) -> None`
    ///
    /// This is called with a batch of stores for efficiency. Each element is
    /// a (address, data) tuple. If not set, falls back to individual stores.
    pub fn set_memory_store_batch(&mut self, cb: Py<PyAny>) {
        self.memory_store_batch = Some(cb);
    }

    /// Set the batched memory load callback.
    ///
    /// The callback should have signature:
    /// `fn(loads: list[tuple[int, int]]) -> list[tuple[bytes, bool, object | None]]`
    ///
    /// Each tuple in input is (address, size). Returns list of (data, is_symbolic, ast_or_none).
    /// This is called with a batch of loads for efficiency, reducing FFI overhead.
    /// If not set, falls back to individual loads.
    pub fn set_memory_load_batch(&mut self, cb: Py<PyAny>) {
        self.memory_load_batch = Some(cb);
    }

    /// Set the hook execution callback.
    ///
    /// The callback should have signature:
    /// `fn(addr: int) -> int`
    ///
    /// Returns the new PC after hook execution.
    pub fn set_on_hook(&mut self, cb: Py<PyAny>) {
        self.on_hook = Some(cb);
    }

    /// Set the syscall handling callback.
    ///
    /// The callback should have signature:
    /// `fn(num: int) -> None`
    pub fn set_on_syscall(&mut self, cb: Py<PyAny>) {
        self.on_syscall = Some(cb);
    }

    /// Set the block lifting callback.
    ///
    /// The callback should have signature:
    /// `fn(addr: int) -> str`
    ///
    /// Returns the IRSB as a JSON string.
    pub fn set_lift_block(&mut self, cb: Py<PyAny>) {
        self.lift_block = Some(cb);
    }

    /// Set the register get callback.
    ///
    /// The callback should have signature:
    /// `fn(offset: int, size: int) -> tuple[bytes, bool, object | None]`
    ///
    /// Returns (concrete_bytes, is_symbolic, symbolic_ast_or_none).
    pub fn set_get_register(&mut self, cb: Py<PyAny>) {
        self.get_register = Some(cb);
    }

    /// Set the register put callback.
    ///
    /// The callback should have signature:
    /// `fn(offset: int, data: bytes) -> None`
    pub fn set_put_register(&mut self, cb: Py<PyAny>) {
        self.put_register = Some(cb);
    }

    /// Set the dirty call callback.
    ///
    /// The callback should have signature:
    /// `fn(name: str, args: list\[int\], ret_ty_bits: int) -> tuple[bytes, bool, object | None]`
    ///
    /// This handles VEX dirty calls to helper functions like CPUID, RDTSC, etc.
    /// Returns (concrete_bytes, is_symbolic, symbolic_ast_or_none).
    pub fn set_dirty_call(&mut self, cb: Py<PyAny>) {
        self.dirty_call = Some(cb);
    }

    /// Set the page fetch callback.
    ///
    /// The callback should have signature:
    /// `fn(page_addr: int) -> tuple[bytes, int, bool]`
    ///
    /// Returns (page_data_4kb, permissions, is_mapped).
    /// If is_mapped is False, the page doesn't exist in Python memory.
    pub fn set_fetch_page(&mut self, cb: Py<PyAny>) {
        self.fetch_page = Some(cb);
    }

    /// Set the batched page fetch callback.
    ///
    /// The callback should have signature:
    /// `fn(page_addrs: list\[int\]) -> list[tuple[bytes, int, bool]]`
    ///
    /// Each result is (page_data_4kb, permissions, is_mapped).
    pub fn set_batch_fetch_pages(&mut self, cb: Py<PyAny>) {
        self.batch_fetch_pages = Some(cb);
    }

    /// Set the symbolic value store callback.
    ///
    /// The callback should have signature:
    /// `fn(addr: int, ast: claripy.AST) -> None`
    ///
    /// This is called when storing a symbolic value to memory. The AST
    /// is the fully reconstructed claripy expression from the Rust engine,
    /// preserving the original symbolic structure (e.g., `x + 10 ^ 0x42`
    /// instead of a fresh symbolic variable).
    pub fn set_memory_store_symbolic_value(&mut self, cb: Py<PyAny>) {
        self.memory_store_symbolic_value = Some(cb);
    }

    /// Set the callback for storing symbolic data at a symbolic address.
    /// Used when the address cannot be concretized (too many possibilities).
    pub fn set_memory_store_symbolic_full(&mut self, cb: Py<PyAny>) {
        self.memory_store_symbolic_full = Some(cb);
    }

    /// Set the callback for loading data at a symbolic address.
    /// Used when the address cannot be concretized (too many possibilities).
    pub fn set_memory_load_symbolic_full(&mut self, cb: Py<PyAny>) {
        self.memory_load_symbolic_full = Some(cb);
    }

    /// Set the callback for resolving unmodeled function calls.
    ///
    /// The callback should have signature:
    /// `fn(addr: int, name: str | None) -> tuple[str, int, bool] | None`
    ///
    /// If the function can be resolved, return (name, num_args, no_return).
    /// If not, return None to deadend the state.
    ///
    /// This is called when execution reaches a function that isn't hooked.
    /// The callback should check project._sim_procedures and angr's procedure registry.
    pub fn set_resolve_function(&mut self, cb: Py<PyAny>) {
        self.resolve_function = Some(cb);
    }

    /// Set the inspect mem_read callback.
    ///
    /// The callback should have signature:
    /// `fn(state_id: int, when: str, addr: int, size: int, value_ast: object | None, endness: str) -> None`
    pub fn set_inspect_mem_read(&mut self, cb: Py<PyAny>) {
        self.inspect_mem_read = Some(cb);
    }

    /// Set the inspect mem_write callback.
    ///
    /// The callback should have signature:
    /// `fn(state_id: int, when: str, addr: int, size: int, value_ast: object | None, endness: str) -> None`
    pub fn set_inspect_mem_write(&mut self, cb: Py<PyAny>) {
        self.inspect_mem_write = Some(cb);
    }

    /// Set the inspect reg_read callback.
    ///
    /// Signature: `fn(state_id: int, when: str, offset: int, size: int,
    ///                value_ast: object | None) -> None`
    pub fn set_inspect_reg_read(&mut self, cb: Py<PyAny>) {
        self.inspect_reg_read = Some(cb);
    }

    /// Set the inspect reg_write callback.
    pub fn set_inspect_reg_write(&mut self, cb: Py<PyAny>) {
        self.inspect_reg_write = Some(cb);
    }

    /// Set the inspect instruction callback.
    ///
    /// Signature: `fn(state_id: int, when: str, addr: int) -> None`.
    pub fn set_inspect_instruction(&mut self, cb: Py<PyAny>) {
        self.inspect_instruction = Some(cb);
    }

    /// Set the inspect irsb (block) callback.
    pub fn set_inspect_irsb(&mut self, cb: Py<PyAny>) {
        self.inspect_irsb = Some(cb);
    }

    /// Set the inspect exit (conditional branch) callback.
    ///
    /// Signature: `fn(state_id: int, when: str, target: int, jumpkind: str,
    ///                guard_ast: object) -> None`.
    pub fn set_inspect_exit(&mut self, cb: Py<PyAny>) {
        self.inspect_exit = Some(cb);
    }

    /// Set the inspect call (function-entry) callback.
    ///
    /// Signature: `fn(state_id: int, when: str, function_address: int) -> None`.
    /// Fires twice per Ijk_Call exit (before/after the frame push).
    pub fn set_inspect_call(&mut self, cb: Py<PyAny>) {
        self.inspect_call = Some(cb);
    }

    /// Set the inspect return (function-exit) callback.
    ///
    /// Signature: `fn(state_id: int, when: str, function_address: int) -> None`.
    /// Fires twice per Ijk_Ret exit (before/after the frame pop).
    pub fn set_inspect_return(&mut self, cb: Py<PyAny>) {
        self.inspect_return = Some(cb);
    }

    /// Set the inspect tmp_read (VEX `RdTmp`) callback.
    ///
    /// Signature: `fn(state_id: int, when: str, tmp_num: int,
    ///                value_ast: object | None) -> None`.
    pub fn set_inspect_tmp_read(&mut self, cb: Py<PyAny>) {
        self.inspect_tmp_read = Some(cb);
    }

    /// Set the inspect tmp_write (VEX `WrTmp`) callback.
    ///
    /// Signature: `fn(state_id: int, when: str, tmp_num: int,
    ///                value_ast: object | None) -> None`.
    pub fn set_inspect_tmp_write(&mut self, cb: Py<PyAny>) {
        self.inspect_tmp_write = Some(cb);
    }

    /// Set the inspect statement (per VEX IR statement) callback.
    ///
    /// Signature: `fn(state_id: int, when: str, stmt_idx: int) -> None`.
    pub fn set_inspect_statement(&mut self, cb: Py<PyAny>) {
        self.inspect_statement = Some(cb);
    }

    /// Set the inspect expr (per VEX IR expression eval) callback.
    ///
    /// Signature: `fn(state_id: int, when: str, expr_result: object | None)
    ///                 -> None`.
    pub fn set_inspect_expr(&mut self, cb: Py<PyAny>) {
        self.inspect_expr = Some(cb);
    }

    /// Set the inspect address_concretization callback.
    ///
    /// Signature: `fn(state_id: int, when: str, action: str,
    ///                addr_ast: object, result: list\[int\] | None) -> None`.
    pub fn set_inspect_address_concretization(&mut self, cb: Py<PyAny>) {
        self.inspect_address_concretization = Some(cb);
    }

    /// Set the inspect symbolic_variable callback.
    ///
    /// Signature: `fn(state_id: int, when: str, name: str, size: int,
    ///                expr_ast: object) -> None`.
    pub fn set_inspect_symbolic_variable(&mut self, cb: Py<PyAny>) {
        self.inspect_symbolic_variable = Some(cb);
    }

    /// Set the inspect fork callback.
    ///
    /// Signature: `fn(state_id: int, when: str) -> None`.
    pub fn set_inspect_fork(&mut self, cb: Py<PyAny>) {
        self.inspect_fork = Some(cb);
    }

    /// Set the inspect constraints callback.
    ///
    /// Signature: `fn(state_id: int, when: str, added_constraints: list) -> Any`.
    pub fn set_inspect_constraints(&mut self, cb: Py<PyAny>) {
        self.inspect_constraints = Some(cb);
    }

    /// Set the inspect vex_lift callback (native-lift dispatch origin).
    ///
    /// Signature: `fn(state_id: int, when: str, addr: int, size: int | None,
    ///                buff: bytes | None) -> None`.
    pub fn set_inspect_vex_lift(&mut self, cb: Py<PyAny>) {
        self.inspect_vex_lift = Some(cb);
    }

    /// Set the inspect-enabled bitmask. Bit N = `InspectEvent` variant N.
    /// Python aggregates registered breakpoints into this single value;
    /// VEX dispatch sites do a single AND test before any payload work.
    ///
    /// `inspect_enabled` is `Arc<AtomicU32>` so this write is visible to
    /// the cloned PythonCallbacks held by the Rust manager. 32 bits leave
    /// ample headroom; current layout: 0..=3 mem/reg read/write,
    /// 4 fork (deferred-fork dispatch in exploration/stepping.rs),
    /// 5 exit, 6/7 custom for instruction/irsb, 8/9 call/return,
    /// 10..=12 Python-dispatched (simprocedure/syscall/dirty),
    /// 13/14 tmp_read/tmp_write, 15 statement, 16 expr,
    /// 17 address_concretization, 18 symbolic_variable.
    #[pyo3(name = "set_inspect_enabled")]
    pub fn py_set_inspect_enabled(&self, mask: u32) {
        self.inspect_enabled
            .store(mask, std::sync::atomic::Ordering::Relaxed);
    }

    /// Tell Rust that the callback state's `state.memory` is a
    /// `RustMemoryProxy`, so the memory-store callbacks are no-ops and the
    /// interpreter must keep such stores in `rust_memory` itself (angr-5rjbq).
    #[pyo3(name = "set_memory_is_rust_proxy")]
    pub fn py_set_memory_is_rust_proxy(&self, on: bool) {
        self.memory_is_rust_proxy
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// Install the snapshot of pages Python's `fetch_page` can serve
    /// (angr-gorvf.4.6). Pages outside it are declined natively, so the fetch
    /// callback is never invoked for them. Pass an empty list to say "Python
    /// can serve nothing" — that is the common case and drives the run-loop
    /// `batch_fetch_pages` GIL cost to zero.
    #[pyo3(name = "set_python_servable_pages")]
    pub fn py_set_python_servable_pages(&self, pages: Vec<u64>) {
        if let Ok(mut guard) = self.python_servable_pages.write() {
            *guard = Some(pages.into_iter().collect());
        }
    }

    /// Drop the servable-page snapshot: every fetch crosses into Python again.
    #[pyo3(name = "clear_python_servable_pages")]
    pub fn py_clear_python_servable_pages(&self) {
        if let Ok(mut guard) = self.python_servable_pages.write() {
            *guard = None;
        }
    }

    /// Install the snapshot of every page Python holds a page object for
    /// (angr-gorvf.4.7). Loads that miss both this set and Rust's own memory
    /// are served natively with an unconstrained filler instead of crossing.
    #[pyo3(name = "set_python_page_universe")]
    pub fn py_set_python_page_universe(&self, pages: Vec<u64>) {
        if let Ok(mut guard) = self.python_page_universe.write() {
            *guard = Some(pages.into_iter().collect());
        }
    }

    /// Drop the page-universe snapshot: every unbacked load crosses again.
    #[pyo3(name = "clear_python_page_universe")]
    pub fn py_clear_python_page_universe(&self) {
        if let Ok(mut guard) = self.python_page_universe.write() {
            *guard = None;
        }
    }

    /// Read the inspect-enabled bitmask (Python-side, mostly for tests).
    #[pyo3(name = "get_inspect_enabled")]
    pub fn py_get_inspect_enabled(&self) -> u32 {
        self.inspect_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Test entry point: invoke the registered mem_read callback directly.
    /// Lets the marshalling round-trip be exercised before VEX dispatch
    /// sites are wired (uq4n.3). Returns whatever Python returned.
    #[pyo3(name = "call_inspect_mem_read")]
    #[pyo3(signature = (state_id, when, addr, size, value_ast, endness))]
    pub fn py_call_inspect_mem_read(
        &self,
        state_id: i64,
        when: &str,
        addr: u64,
        size: u32,
        value_ast: Option<Py<PyAny>>,
        endness: &str,
    ) -> PyResult<Option<Py<PyAny>>> {
        self.call_inspect_mem_read(state_id, when, addr, size, value_ast.as_ref(), endness)
    }

    /// Test entry point: invoke the registered mem_write callback directly.
    #[pyo3(name = "call_inspect_mem_write")]
    #[pyo3(signature = (state_id, when, addr, size, value_ast, endness))]
    pub fn py_call_inspect_mem_write(
        &self,
        state_id: i64,
        when: &str,
        addr: u64,
        size: u32,
        value_ast: Option<Py<PyAny>>,
        endness: &str,
    ) -> PyResult<Option<Py<PyAny>>> {
        self.call_inspect_mem_write(state_id, when, addr, size, value_ast.as_ref(), endness)
    }

    /// Test entry point: invoke the registered reg_read callback directly.
    #[pyo3(name = "call_inspect_reg_read")]
    #[pyo3(signature = (state_id, when, offset, size, value_ast))]
    pub fn py_call_inspect_reg_read(
        &self,
        state_id: i64,
        when: &str,
        offset: u32,
        size: u32,
        value_ast: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        self.call_inspect_reg_read(state_id, when, offset, size, value_ast.as_ref())
    }

    /// Test entry point: invoke the registered reg_write callback directly.
    #[pyo3(name = "call_inspect_reg_write")]
    #[pyo3(signature = (state_id, when, offset, size, value_ast))]
    pub fn py_call_inspect_reg_write(
        &self,
        state_id: i64,
        when: &str,
        offset: u32,
        size: u32,
        value_ast: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        self.call_inspect_reg_write(state_id, when, offset, size, value_ast.as_ref())
    }

    /// Test entry point: invoke the registered instruction callback directly.
    #[pyo3(name = "call_inspect_instruction")]
    pub fn py_call_inspect_instruction(
        &self,
        state_id: i64,
        when: &str,
        addr: u64,
    ) -> PyResult<()> {
        self.call_inspect_instruction(state_id, when, addr)
    }

    /// Test entry point: invoke the registered irsb callback directly.
    #[pyo3(name = "call_inspect_irsb")]
    pub fn py_call_inspect_irsb(&self, state_id: i64, when: &str, addr: u64) -> PyResult<()> {
        self.call_inspect_irsb(state_id, when, addr)
    }

    /// Test entry point: invoke the registered exit callback directly.
    #[pyo3(name = "call_inspect_exit")]
    #[pyo3(signature = (state_id, when, target, jumpkind, guard_ast))]
    pub fn py_call_inspect_exit(
        &self,
        state_id: i64,
        when: &str,
        target: u64,
        jumpkind: &str,
        guard_ast: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        self.call_inspect_exit(state_id, when, target, jumpkind, guard_ast.as_ref())
    }

    /// Test entry point: invoke the registered call callback directly.
    #[pyo3(name = "call_inspect_call")]
    pub fn py_call_inspect_call(
        &self,
        state_id: i64,
        when: &str,
        function_address: u64,
    ) -> PyResult<()> {
        self.call_inspect_call(state_id, when, function_address)
    }

    /// Test entry point: invoke the registered return callback directly.
    #[pyo3(name = "call_inspect_return")]
    pub fn py_call_inspect_return(
        &self,
        state_id: i64,
        when: &str,
        function_address: u64,
    ) -> PyResult<()> {
        self.call_inspect_return(state_id, when, function_address)
    }

    /// Test entry point: invoke the registered tmp_read callback directly.
    #[pyo3(name = "call_inspect_tmp_read")]
    #[pyo3(signature = (state_id, when, tmp_num, value_ast))]
    pub fn py_call_inspect_tmp_read(
        &self,
        state_id: i64,
        when: &str,
        tmp_num: u32,
        value_ast: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        self.call_inspect_tmp_read(state_id, when, tmp_num, value_ast.as_ref())
    }

    /// Test entry point: invoke the registered tmp_write callback directly.
    #[pyo3(name = "call_inspect_tmp_write")]
    #[pyo3(signature = (state_id, when, tmp_num, value_ast))]
    pub fn py_call_inspect_tmp_write(
        &self,
        state_id: i64,
        when: &str,
        tmp_num: u32,
        value_ast: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        self.call_inspect_tmp_write(state_id, when, tmp_num, value_ast.as_ref())
    }

    /// Test entry point: invoke the registered statement callback directly.
    #[pyo3(name = "call_inspect_statement")]
    pub fn py_call_inspect_statement(
        &self,
        state_id: i64,
        when: &str,
        stmt_idx: u32,
    ) -> PyResult<()> {
        self.call_inspect_statement(state_id, when, stmt_idx)
    }

    /// Test entry point: invoke the registered expr callback directly.
    #[pyo3(name = "call_inspect_expr")]
    #[pyo3(signature = (state_id, when, expr_result))]
    pub fn py_call_inspect_expr(
        &self,
        state_id: i64,
        when: &str,
        expr_result: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        self.call_inspect_expr(state_id, when, expr_result.as_ref())
    }

    /// Test entry point: invoke the address_concretization callback directly.
    #[pyo3(name = "call_inspect_address_concretization")]
    #[pyo3(signature = (state_id, when, action, addr_ast, result))]
    pub fn py_call_inspect_address_concretization(
        &self,
        state_id: i64,
        when: &str,
        action: &str,
        addr_ast: Py<PyAny>,
        result: Option<Vec<u64>>,
    ) -> PyResult<()> {
        self.call_inspect_address_concretization(state_id, when, action, &addr_ast, result)
    }

    /// Test entry point: invoke the symbolic_variable callback directly.
    #[pyo3(name = "call_inspect_symbolic_variable")]
    #[pyo3(signature = (state_id, when, name, size, expr_ast))]
    pub fn py_call_inspect_symbolic_variable(
        &self,
        state_id: i64,
        when: &str,
        name: &str,
        size: u32,
        expr_ast: Py<PyAny>,
    ) -> PyResult<()> {
        self.call_inspect_symbolic_variable(state_id, when, name, size, &expr_ast)
    }

    /// Test entry point: invoke the registered fork callback directly.
    #[pyo3(name = "call_inspect_fork")]
    pub fn py_call_inspect_fork(&self, state_id: i64, when: &str) -> PyResult<()> {
        self.call_inspect_fork(state_id, when)
    }

    /// Check if all required callbacks are set.
    pub fn is_ready(&self) -> bool {
        self.memory_load.is_some() && self.memory_store.is_some() && self.lift_block.is_some()
    }

    /// GC traversal: visit each held Python callback so the cycle
    /// `mgr -> _callbacks -> bound method -> mgr` is GC-collectible.
    /// Without this, the manager (and its _state_cache, ~4030 angr pages
    /// per call in mma_howtouse) leaks permanently.
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        Self::traverse_fields(self, &visit)
    }

    /// GC clear: drop all Python callback refs. After clear, calls to any
    /// callback method will fail with "callback not set", but by then the
    /// containing object is being collected anyway.
    fn __clear__(&mut self) {
        self.clear_fields();
    }
}

impl PythonCallbacks {
    /// Visit every `Py<PyAny>` field. Used by __traverse__ on this type and
    /// by RustExplorationManager.__traverse__ which holds a cloned copy of
    /// PythonCallbacks (and so participates in the same cycle).
    pub(crate) fn traverse_fields(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for obj in [
            &self.memory_load,
            &self.memory_store,
            &self.memory_store_batch,
            &self.memory_load_batch,
            &self.on_hook,
            &self.on_syscall,
            &self.lift_block,
            &self.get_register,
            &self.put_register,
            &self.dirty_call,
            &self.fetch_page,
            &self.batch_fetch_pages,
            &self.memory_store_symbolic_value,
            &self.memory_store_symbolic_full,
            &self.memory_load_symbolic_full,
            &self.resolve_function,
            &self.inspect_mem_read,
            &self.inspect_mem_write,
            &self.inspect_reg_read,
            &self.inspect_reg_write,
            &self.inspect_instruction,
            &self.inspect_irsb,
            &self.inspect_exit,
            &self.inspect_call,
            &self.inspect_return,
            &self.inspect_tmp_read,
            &self.inspect_tmp_write,
            &self.inspect_statement,
            &self.inspect_expr,
            &self.inspect_address_concretization,
            &self.inspect_symbolic_variable,
            &self.inspect_fork,
            &self.inspect_constraints,
            &self.inspect_vex_lift,
        ]
        .into_iter()
        .flatten()
        {
            visit.call(obj)?;
        }
        Ok(())
    }

    /// Drop every `Py<PyAny>` field. Used by __clear__ on this type and on
    /// RustExplorationManager (which has a cloned copy in its `callbacks` field).
    pub(crate) fn clear_fields(&mut self) {
        self.memory_load = None;
        self.memory_store = None;
        self.memory_store_batch = None;
        self.memory_load_batch = None;
        self.on_hook = None;
        self.on_syscall = None;
        self.lift_block = None;
        self.get_register = None;
        self.put_register = None;
        self.dirty_call = None;
        self.fetch_page = None;
        self.batch_fetch_pages = None;
        self.memory_store_symbolic_value = None;
        self.memory_store_symbolic_full = None;
        self.memory_load_symbolic_full = None;
        self.resolve_function = None;
        self.inspect_mem_read = None;
        self.inspect_mem_write = None;
        self.inspect_reg_read = None;
        self.inspect_reg_write = None;
        self.inspect_instruction = None;
        self.inspect_irsb = None;
        self.inspect_exit = None;
        self.inspect_call = None;
        self.inspect_return = None;
        self.inspect_tmp_read = None;
        self.inspect_tmp_write = None;
        self.inspect_statement = None;
        self.inspect_expr = None;
        self.inspect_address_concretization = None;
        self.inspect_symbolic_variable = None;
        self.inspect_fork = None;
        self.inspect_constraints = None;
        self.inspect_vex_lift = None;
        self.inspect_enabled
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Default for PythonCallbacks {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "../callbacks_tests.rs"]
mod tests;
