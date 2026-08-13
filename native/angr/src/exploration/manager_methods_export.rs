//! `#[pymethods]` for [`RustExplorationManager`]: snapshot export plus the
//! by-state-id read/write accessors the Python state proxy is built on.
//!
//! One of several `#[pymethods]` blocks for the pyclass; see the
//! `manager_methods` module doc for the split rationale (angr-9ke6b.50).
//!
//! Only three methods here are `export_*`-prefixed, so — following the
//! `manager_methods_constraints` precedent — the contents are spelled out
//! rather than left to the file name (angr-03vl4.11):
//!
//! - **Snapshot export**: `export_state`, `export_state_flushed`,
//!   `export_found_states`, plus `step_state`, which returns per-stash
//!   snapshots of one state's successors.
//! - **Symbolic introspection**: `state_symbolic_info`,
//!   `get_state_symbolic_z3_asts`, `state_satisfiable`.
//! - **SimOption queries**: `state_has_option` and the named shorthands
//!   `state_enforce_{permissions,nx}`, `state_no_ip_concretization`,
//!   `state_no_symbolic_jump_resolution`, `state_keep_ip_symbolic`.
//! - **Registers**: `get_state_register{,s_batch,_ast}`,
//!   `set_state_register_symbolic_ast`.
//! - **Memory**: `get_state_memory{,_ast}`,
//!   `set_state_memory_{concrete,ast}{,_automap}`,
//!   `state_memory_store_symbolic_multi`.
//! - **Files and stdio**: `get_state_stdout`, `{get,has,append}_state_fd_output`,
//!   `{has,get}_state_stdin_symbols`, `eval_stdin_symbol`,
//!   `get_state_open_fds`, `has_state_extra_fds`, `get_state_fd_content`,
//!   `register_state_fd`.
//! - **History and heap**: `get_state_call_stack{,_depth}`,
//!   `get_state_detailed_history`, `get_state_heap_metadata`.
//!
//! Five `export_*`-named methods live *elsewhere*, in
//! `manager_methods_constraints`: `export_pending_{constraints,state}`,
//! `export_state_constraints`, `export_callback_bundle` and
//! `export_z3_constraint_ptrs`. They are grouped by subject (the pending
//! state and its constraint plumbing), not by name prefix; look for an
//! `export_*` method there before assuming it is missing.
//!
//! The line against `manager_methods_state.rs` is *addressing*, not subject
//! matter: that module owns state lifecycle and stash membership, while every
//! method here reads or writes the contents of one already-existing state
//! named by `state_id`.
//!
//! This is a Python-boundary module; `unwrap`/`expect` are denied here so a
//! future panic-on-input landmine cannot be reintroduced without a reviewed,
//! reasoned `#[allow]` (angr-qwyti.11 enforcement layer).
//!
//! **No `test_submod!` here, by design** (angr-03vl4.13). Every method in this
//! file forwards to a `_`-prefixed body in a sibling module — see the
//! `See ... for the body` line on each — and those modules carry the test
//! coverage (`state_api.rs`, `state_lifecycle.rs`,
//! `stats_api.rs`). What is left at this
//! layer is the PyO3 signature and the `#[angr_macros::steady_guarded]` /
//! `#[angr_macros::steady_guard_exempt]` choice — the signature defaults
//! only apply to a call made *from Python*, so no Rust-level unit test can
//! observe those, but making *some* explicit choice (not necessarily the
//! correct one) is a compile-time obligation enforced by
//! `#[angr_macros::steady_guard_checked]` on the `impl` block below.
//! Sibling `manager_methods_{procedures,techniques,state}.rs` do have test
//! modules because their methods carry filtering / stash-declaration logic of
//! their own rather than delegating outright.
#![deny(clippy::unwrap_used, clippy::expect_used)]
use super::*;

#[angr_macros::steady_guard_checked]
#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
#[pymethods]
impl RustExplorationManager {
    /// Export a state by ID as a full snapshot.
    ///
    /// This searches all stashes for the state with the given ID and returns
    /// a complete snapshot that can be used to reconstruct an angr SimState.
    /// Deferred writes (pending writes and Multi cells) are flushed first, so
    /// this is now an alias for `export_state_flushed` (angr-9ke6b.101).
    #[angr_macros::steady_guard_exempt(
        reason = "exports one state_id-scoped state's snapshot (flushing its own deferred \
                  writes first); does not mutate exploration config or the active stash."
    )]
    pub fn export_state(
        &mut self,
        state_id: u64,
    ) -> PyResult<crate::state::ExplorationStateSnapshot> {
        self._export_state(state_id)
    }

    /// Step a single state out-of-band and return its successors bucketed by
    /// category, WITHOUT stashing them (E1.a).
    ///
    /// `extra_stop_points` are stop addresses honored for this call only, on
    /// top of the manager-level stop addresses. Every returned state is parked
    /// in the `_step_out` stash; the caller places it with `move_state()`.
    /// See `stepping::_step_state` for the body.
    #[pyo3(signature = (state_id, extra_stop_points=None))]
    #[angr_macros::steady_guard_exempt(
        reason = "out-of-band per-technique dispatch helper (rust_techniques.py's step_state-hook \
                  path), same routine-per-step rationale as move_states; takes the source state \
                  out of whichever stash it was in (including active) but parks results in the \
                  private `_step_out` stash, and mutates no exploration config."
    )]
    pub fn step_state(
        &mut self,
        state_id: u64,
        extra_stop_points: Option<Vec<u64>>,
    ) -> PyResult<HashMap<String, Vec<crate::state::ExplorationStateSnapshot>>> {
        self._step_state(state_id, extra_stop_points)
    }

    /// Export a state by ID, flushing pending writes first.
    #[angr_macros::steady_guard_exempt(
        reason = "exports one state_id-scoped state's snapshot (flushing its own deferred \
                  writes first); does not mutate exploration config or the active stash."
    )]
    pub fn export_state_flushed(
        &mut self,
        state_id: u64,
    ) -> PyResult<crate::state::ExplorationStateSnapshot> {
        self._export_state_flushed(state_id)
    }

    /// Export all found states as snapshots.
    ///
    /// Flushes each state's deferred writes first (angr-9ke6b.101) — this is
    /// the primary `explore(find=...)` result API, so an unflushed export here
    /// silently drops Multi-cell bytes.
    #[angr_macros::steady_guard_exempt(
        reason = "exports the found stash's states (flushing each one's own deferred writes \
                  first); does not mutate exploration config or the active stash."
    )]
    pub fn export_found_states(&mut self) -> Vec<crate::state::ExplorationStateSnapshot> {
        self._export_found_states()
    }

    /// Debug: Get symbolic object info for a state.
    pub fn state_symbolic_info(&self, state_id: u64, addr: u64) -> PyResult<String> {
        self._state_symbolic_info(state_id, addr)
    }

    /// Get Z3 AST pointers for all symbolic objects in a state's memory.
    ///
    /// Returns Vec<(addr, z3_ast_ptr_as_usize, width_bits)> for each symbolic
    /// object. The Z3 ASTs are built in the shared Z3 context, so Python can
    /// directly wrap them as z3.BitVecRef and convert to claripy ASTs.
    ///
    /// This is used to export Rust-computed symbolic expressions (e.g., flag
    /// computations in asisctf) to Python state memory during state export.
    #[cfg(feature = "vex-engine-z3")]
    #[angr_macros::steady_guard_exempt(
        reason = "read-only export of one state_id-scoped state's Z3 AST pointers; does not \
                  mutate exploration config or the active stash."
    )]
    pub fn get_state_symbolic_z3_asts(
        &mut self,
        state_id: u64,
    ) -> PyResult<Vec<(u64, usize, u32)>> {
        self._get_state_symbolic_z3_asts(state_id)
    }

    /// Check if constraints are satisfiable for a state.
    pub fn state_satisfiable(&self, state_id: u64) -> PyResult<bool> {
        self._state_satisfiable(state_id)
    }

    /// Whether strict memory permission enforcement is enabled on a state.
    /// Mirrors angr's STRICT_PAGE_ACCESS option.
    pub fn state_enforce_permissions(&self, state_id: u64) -> PyResult<bool> {
        self._state_enforce_permissions(state_id)
    }

    /// Whether non-executable page enforcement is enabled on a state.
    /// Mirrors angr's ENABLE_NX option.
    pub fn state_enforce_nx(&self, state_id: u64) -> PyResult<bool> {
        self._state_enforce_nx(state_id)
    }

    /// Whether NO_IP_CONCRETIZATION is active on a state.
    /// When set, symbolic jump targets short-circuit to the unconstrained
    /// stash without enumeration.
    pub fn state_no_ip_concretization(&self, state_id: u64) -> PyResult<bool> {
        self._state_no_ip_concretization(state_id)
    }

    /// Whether NO_SYMBOLIC_JUMP_RESOLUTION is active on a state.
    /// Same Rust effect as `state_no_ip_concretization` — symbolic jump
    /// targets route to the unconstrained stash without enumeration.
    pub fn state_no_symbolic_jump_resolution(&self, state_id: u64) -> PyResult<bool> {
        self._state_no_symbolic_jump_resolution(state_id)
    }

    /// Whether KEEP_IP_SYMBOLIC is active on a state.
    /// When set, the IP register on each post-concretization successor stays
    /// holding the original symbolic next-pc expression (no `target == addr`
    /// narrowing constraint is added). The next block lift still drives from
    /// the concretized `state.pc` value.
    pub fn state_keep_ip_symbolic(&self, state_id: u64) -> PyResult<bool> {
        self._state_keep_ip_symbolic(state_id)
    }

    /// Whether the named symex-relevant SimOption (e.g. `"SHORT_READS"`) is
    /// active on a state (angr-kzjv6). Mirrors the option subset that
    /// `_add_rust_state` threads onto the Rust state for native SimProcedures.
    pub fn state_has_option(&self, state_id: u64, name: &str) -> PyResult<bool> {
        self._state_has_option(state_id, name)
    }

    /// Get a register value from a state.
    ///
    /// Returns `None` both for a register name this architecture does not
    /// model and for a register holding a symbolic value — unlike
    /// `get_pending_register`, which raises `ValueError` on an unknown name.
    /// The asymmetry is deliberate; see `state_api::_get_state_register` for
    /// the body and the rationale.
    pub fn get_state_register(&self, state_id: u64, name: &str) -> PyResult<Option<u128>> {
        self._get_state_register(state_id, name)
    }

    /// Get multiple register values from a state in one FFI call.
    /// Returns a list of `Option<u128>` in the same order as the input names,
    /// each with the same two-cause `None` contract as `get_state_register`.
    pub fn get_state_registers_batch(
        &self,
        state_id: u64,
        names: Vec<String>,
    ) -> PyResult<Vec<Option<u128>>> {
        self._get_state_registers_batch(state_id, names)
    }

    /// Get the claripy AST for a register on a state (angr-4pm1).
    ///
    /// Mirrors `get_pending_register_ast` for an arbitrary `state_id`.
    /// Returns the claripy AST built from Rust's stored `RustBV`, preserving
    /// identity for symbolic values so constraints added by the proxy land on
    /// the same Z3 symbol Rust is tracking. Returns `None` if the register
    /// name is unknown or the state holds no value for it.
    /// See `state_api::_get_state_register_ast` for the body.
    pub fn get_state_register_ast(
        &self,
        py: Python<'_>,
        state_id: u64,
        name: &str,
    ) -> PyResult<Option<Py<PyAny>>> {
        self._get_state_register_ast(py, state_id, name)
    }

    /// Set a state's register to a symbolic value from a claripy AST
    /// (angr-4pm1). Mirrors `set_pending_register_symbolic_ast` for an
    /// arbitrary `state_id`, routing through `claripy_to_rustbv` so the
    /// symbol is registered in the shared cache and the inverse
    /// `get_state_register_ast` round-trip preserves identity.
    /// See `state_api::_set_state_register_symbolic_ast` for the body.
    #[angr_macros::steady_guard_exempt(
        reason = "mutates one state_id-scoped state's own register content; does not mutate \
                  exploration config or the active stash."
    )]
    pub fn set_state_register_symbolic_ast(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        reg_name: &str,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self._set_state_register_symbolic_ast(py, state_id, reg_name, ast)
    }

    /// Get memory from a state.
    pub fn get_state_memory(
        &self,
        state_id: u64,
        addr: u64,
        size: u32,
    ) -> PyResult<Option<Vec<u8>>> {
        self._get_state_memory(state_id, addr, size)
    }

    /// Get memory from a state as a claripy AST (angr-8dop.1). Returns the
    /// symbolic AST verbatim — never concretizes via the solver. Used by
    /// `RustMemoryProxy.load` when the gate is on so symbolic libc
    /// SimProcedures (strlen/strchr/memchr/...) see real symbolic bytes
    /// instead of an arbitrary solver witness.
    /// See `state_api::_get_state_memory_ast` for the body.
    pub fn get_state_memory_ast(
        &self,
        py: Python<'_>,
        state_id: u64,
        addr: u64,
        size: u32,
    ) -> PyResult<Option<Py<PyAny>>> {
        self._get_state_memory_ast(py, state_id, addr, size)
    }

    /// Set memory on a state from concrete bytes (angr-j28e write-through).
    /// See `state_api::_set_state_memory_concrete` for the body.
    #[angr_macros::steady_guard_exempt(
        reason = "mutates one state_id-scoped state's own memory content; does not mutate \
                  exploration config or the active stash."
    )]
    pub fn set_state_memory_concrete(
        &mut self,
        state_id: u64,
        addr: u64,
        data: &[u8],
    ) -> PyResult<()> {
        self._set_state_memory_concrete(state_id, addr, data)
    }

    /// Like `set_state_memory_concrete`, but widens the state's lazy region to
    /// cover the target page so a store to an otherwise-unmapped address
    /// auto-maps instead of erroring `Unmapped` (angr-ijwp0). Used by
    /// `RustMemoryProxy.store` to mirror angr's map-on-write memory when
    /// STRICT_PAGE_ACCESS is off.
    /// See `state_api::_set_state_memory_concrete_automap` for the body.
    #[angr_macros::steady_guard_exempt(
        reason = "mutates one state_id-scoped state's own memory content (plus its lazy-region \
                  map); does not mutate exploration config or the active stash."
    )]
    pub fn set_state_memory_concrete_automap(
        &mut self,
        state_id: u64,
        addr: u64,
        data: &[u8],
    ) -> PyResult<()> {
        self._set_state_memory_concrete_automap(state_id, addr, data)
    }

    /// Set memory on a state from a claripy AST (angr-j28e write-through).
    /// Used when the value is symbolic (e.g., a BVS or expression). The
    /// address is concrete; symbolic addresses are not supported on the
    /// proxy write path — callers fall back to the Python engine.
    /// See `state_api::_set_state_memory_ast` for the body.
    #[angr_macros::steady_guard_exempt(
        reason = "mutates one state_id-scoped state's own memory content; does not mutate \
                  exploration config or the active stash."
    )]
    pub fn set_state_memory_ast(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        addr: u64,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self._set_state_memory_ast(py, state_id, addr, ast)
    }

    /// Like `set_state_memory_ast`, but widens the state's lazy region to cover
    /// the target page so a store to an address outside any existing lazy
    /// region auto-maps instead of erroring `Unmapped` (angr-5rjbq). Used by
    /// the callback-memory-proxy symbolic-address store fallback.
    #[angr_macros::steady_guard_exempt(
        reason = "mutates one state_id-scoped state's own memory content (plus its lazy-region \
                  map); does not mutate exploration config or the active stash."
    )]
    pub fn set_state_memory_ast_automap(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        addr: u64,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self._set_state_memory_ast_automap(py, state_id, addr, ast)
    }

    /// Phase 1.4 (angr-5zw8): route a symbolic-address store through the
    /// Multi-cell lazy path on the given state. Used by
    /// `_cb_memory_store_symbolic_full` when the address AST carries a
    /// `MultiwriteAnnotation` — the SimProcedures in `libc/strchr.py`,
    /// `libc/gets.py`, `libc/fgets.py` tag returned addresses with this
    /// annotation so Range concretization picks up >1 candidate.
    ///
    /// Returns `true` on success. Returns `false` if conversion or store
    /// fails (caller should fall back to the existing Python state path
    /// to keep progress).
    #[angr_macros::steady_guard_exempt(
        reason = "mutates one state_id-scoped state's own memory content; does not mutate \
                  exploration config or the active stash."
    )]
    pub fn state_memory_store_symbolic_multi<'py>(
        &mut self,
        py: Python<'py>,
        state_id: u64,
        addr_ast: &Bound<'py, PyAny>,
        data_ast: &Bound<'py, PyAny>,
    ) -> PyResult<bool> {
        self._state_memory_store_symbolic_multi(py, state_id, addr_ast, data_ast)
    }

    /// Get the stdout buffer for a state by ID.
    ///
    /// Returns the accumulated output from native puts/printf calls.
    pub fn get_state_stdout(&self, state_id: u64) -> PyResult<Vec<u8>> {
        self._get_state_stdout(state_id)
    }

    /// Get the output buffer for a specific file descriptor.
    ///
    /// Returns the accumulated output from native write/puts/printf calls.
    pub fn get_state_fd_output(&self, state_id: u64, fd: u32) -> PyResult<Vec<u8>> {
        self._get_state_fd_output(state_id, fd)
    }

    /// Check whether a file descriptor has any output (no allocation).
    pub fn has_state_fd_output(&self, state_id: u64, fd: u32) -> bool {
        self._has_state_fd_output(state_id, fd)
    }

    /// Append bytes to a file descriptor's output buffer.
    ///
    /// The write-back half of the SimProcedure-callback posix channel
    /// (angr-op0dn.14.1.4): a bounced fwrite/fputc/fprintf writes the Python
    /// callback state's posix stream, and this pushes the new suffix into the
    /// Rust state so `posix.dumps(fd)` on a later export sees it.
    ///
    /// Returns `false` when the write was refused because the fd carried
    /// bounded symbolic content (see `RustSimState::write_fd`); the caller
    /// should treat that as a lossy fallback, not an error.
    #[angr_macros::steady_guard_exempt(
        reason = "mutates one state_id-scoped state's own fd-output buffer; does not mutate \
                  exploration config or the active stash."
    )]
    pub fn append_state_fd_output(
        &mut self,
        state_id: u64,
        fd: u32,
        data: Vec<u8>,
    ) -> PyResult<bool> {
        self._append_state_fd_output(state_id, fd, data)
    }

    /// Check if a state has recorded stdin symbols from native fgets/fgetc/getchar.
    pub fn has_state_stdin_symbols(&self, state_id: u64) -> bool {
        self._has_state_stdin_symbols(state_id)
    }

    /// Get the stdin symbols for a state by ID.
    ///
    /// Returns list of (name, bit_width) tuples for symbolic variables
    /// created by native fgets/fgetc/getchar. Used to reconstruct stdin
    /// data in Python's posix plugin for posix.dumps(0).
    pub fn get_state_stdin_symbols(&self, state_id: u64) -> PyResult<Vec<(String, u32)>> {
        self._get_state_stdin_symbols(state_id)
    }

    /// Get the call stack for a state by ID.
    ///
    /// Returns list of (call_site_addr, callee_addr, return_addr, stack_ptr) tuples.
    pub fn get_state_call_stack(&self, state_id: u64) -> PyResult<Vec<(u64, u64, u64, u64)>> {
        self._get_state_call_stack(state_id)
    }

    /// Get the call stack depth for a state by ID.
    pub fn get_state_call_stack_depth(&self, state_id: u64) -> PyResult<usize> {
        self._get_state_call_stack_depth(state_id)
    }

    /// Get the detailed execution history for a state by ID.
    ///
    /// Returns list of (addr, jumpkind, jump_target) tuples.
    /// jumpkind: 0=Boring, 1=Call, 2=Ret, 3=Syscall, 4=Other
    pub fn get_state_detailed_history(&self, state_id: u64) -> PyResult<Vec<(u64, u8, u64)>> {
        self._get_state_detailed_history(state_id)
    }

    /// Get heap metadata for a state by ID.
    ///
    /// Returns dict with:
    /// - allocated: list of (addr, size) tuples for active allocations
    /// - freed: list of distinct freed addresses (a set — see `HeapMetadata::freed`)
    /// - alloc_count: number of active allocations
    /// - free_count: number of distinct freed addresses
    pub fn get_state_heap_metadata(&self, state_id: u64) -> PyResult<HeapMetadataReturn> {
        self._get_state_heap_metadata(state_id)
    }

    /// Get the list of open file descriptors for a state.
    ///
    /// Returns list of (fd, name, position, flags, content_len, is_open) tuples.
    pub fn get_state_open_fds(&self, state_id: u64) -> PyResult<Vec<OpenFdInfo>> {
        self._get_state_open_fds(state_id)
    }

    /// True when the state has an open fd above stderr.
    ///
    /// Fast-path gate for the inbound callback fd sync (angr-op0dn.14.1.6) so
    /// the common bounce (nothing opened natively) never pays for the
    /// `get_state_open_fds` tuple list.
    pub fn has_state_extra_fds(&self, state_id: u64) -> bool {
        self._has_state_extra_fds(state_id)
    }

    /// Get the content of a file descriptor for a state.
    pub fn get_state_fd_content(&self, state_id: u64, fd: u32) -> PyResult<Vec<u8>> {
        self._get_state_fd_content(state_id, fd)
    }

    /// Adopt an fd that a bounced Python SimProcedure opened, at the fd number
    /// Python chose. `flags` is a POSIX open(2) bitfield; `content` the
    /// concrete bytes of the backing SimFile; `position` its seek offset.
    ///
    /// Returns False when the fd is already known natively (nothing changes),
    /// which makes the caller's fd-table diff idempotent across the repeated
    /// callbacks that share one cached state.
    #[angr_macros::steady_guard_exempt(
        reason = "mutates one state_id-scoped state's own fd table; does not mutate \
                  exploration config or the active stash."
    )]
    pub fn register_state_fd(
        &mut self,
        state_id: u64,
        fd: u32,
        name: String,
        flags: u32,
        content: Vec<u8>,
        position: u64,
    ) -> PyResult<bool> {
        self._register_state_fd(state_id, fd, name, flags, content, position)
    }

    /// Evaluate a stdin symbol by name and width using the state's solver.
    ///
    /// `width` is the symbol's recorded bit-width (8 for byte reads, 32/64 for
    /// scanf numeric conversions). Passing the wrong width mints a distinct,
    /// unconstrained Z3 const and yields a garbage model value (angr-ph300.18).
    /// Returns the concrete value as `Option<u64>`, or None if the symbol
    /// cannot be found or evaluated.
    pub fn eval_stdin_symbol(&self, state_id: u64, name: &str, width: u32) -> Option<u64> {
        self._eval_stdin_symbol(state_id, name, width)
    }
}
