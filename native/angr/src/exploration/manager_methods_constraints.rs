//! `#[pymethods]` for [`RustExplorationManager`]: constraint / solver plumbing, per-state metadata, stash mutation and stats dicts.
//!
//! One of several `#[pymethods]` blocks for the pyclass; see the
//! `manager_methods` module doc for the split rationale (angr-9ke6b.50).
//!
//! The full contents, in file order — this is the widest of the
//! `manager_methods_*` blocks, so the list is spelled out rather than
//! summarised:
//!
//! - **Constraints / solvers**: `export_pending_constraints`,
//!   `export_state_constraints`, `add_constraints_to_{pending,state}`,
//!   `{export,import}_z3_constraint_ptrs`, `state_unsat_core`,
//!   `{fork,borrow}_pending_solver`, `fork_state_solver`,
//!   `{get,set}_state_solver_timeout`, `debug_solver_info`.
//! - **Pending-state export/inspection**: `export_pending_state`,
//!   `export_callback_bundle`, `get_active_handle_ids`,
//!   `get_pending_{root_state_id,ancestry,mapped_pages}`,
//!   `pending_memory_load{,_page,_symbolic_page}`.
//! - **Per-state metadata**: mmap base, POSIX brk, heap brk,
//!   `{get,set}_state_symbolic_pages`, `{get,set}_state_hook_symbolic_memory`,
//!   `{get,set}_state_addr_to_ast`, `clear_state_metadata`.
//! - **Skip-hook addresses and error queue**: `set_skip_hook_addr`,
//!   `clear_skip_hook{,_for_addr}`, `get_errors`, `clear_errors`.
//! - **Stash mutation**: `move_state{,s}`, `clear_stash`,
//!   `drop_state_from_stash`, `reset_for_stage`.
//! - **Stats dicts**: `stats`, `get_fallback_stats`.
//!
//! The line against `manager_methods_state.rs` is *granularity*, not subject
//! matter: that module works on a single state's identity and contents
//! (create/add/fork/merge, registers, memory, files, stash membership
//! queries), whereas the stash-level mutators — moving states between
//! stashes, emptying a stash, resetting for the next `find` stage — live
//! here. Both boundaries are inherited from the monolith's section banners
//! rather than derived from a taxonomy, so a few members sit on either side
//! for historical reasons only. When adding a method, match the banner group
//! above; do not re-derive the split from first principles.
//!
//! This is a Python-boundary module; `unwrap`/`expect` are denied here so a
//! future panic-on-input landmine cannot be reintroduced without a reviewed,
//! reasoned `#[allow]` (angr-qwyti.11 enforcement layer).
//!
//! **No `test_submod!` here, by design** (angr-03vl4.13). Every method in this
//! file forwards to a `_`-prefixed body in a sibling module — see the
//! `See ... for the body` line on each — and those modules carry the test
//! coverage (`pending_api.rs`, `constraints.rs`,
//! `state_api.rs`, `stats_api.rs`). What is left at this
//! layer is the PyO3 signature and the `#[angr_macros::steady_guarded]`
//! placement, neither of which a Rust-level unit test can observe: the
//! signature defaults only apply to a call made *from Python*, and guard
//! coverage is gated mechanically by `tools/audit_steady_guard_coverage.py`.
//! Sibling `manager_methods_{procedures,techniques,state}.rs` do have test
//! modules because their methods carry filtering / stash-declaration logic of
//! their own rather than delegating outright.
#![deny(clippy::unwrap_used, clippy::expect_used)]
use super::*;

#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
#[pymethods]
impl RustExplorationManager {
    /// Export pending constraints as a list of claripy ASTs.
    ///
    /// Returns constraints that can be added to Python state.solver.
    /// This exports stored branch conditions accumulated during Rust execution.
    /// See `pending_api::_export_pending_constraints` for the body.
    pub fn export_pending_constraints(
        &self,
        py: Python<'_>,
        state_id: u64,
    ) -> PyResult<Vec<Py<PyAny>>> {
        self._export_pending_constraints(py, state_id)
    }

    /// Get handle IDs that are actively referenced in the pending state.
    ///
    /// Returns handle IDs used in stored conditions and deferred forks.
    /// These should not be evicted from the AST handle cache.
    /// See `pending_api::_get_active_handle_ids` for the body.
    pub fn get_active_handle_ids(&self) -> Vec<u64> {
        self._get_active_handle_ids()
    }

    /// Export the pending state as a full snapshot.
    ///
    /// This allows Python to get a complete snapshot of the pending state
    /// including all registers, memory pages, and metadata.
    /// See `pending_api::_export_pending_state` for the body.
    pub fn export_pending_state(
        &mut self,
        state_id: u64,
    ) -> PyResult<crate::state::ExplorationStateSnapshot> {
        self._export_pending_state(state_id)
    }

    /// Get the root state ID for the pending callback state.
    ///
    /// When Rust forks states internally, Python only has cached data for the
    /// original state that was added via Python. This method returns the root
    /// state ID (the original state) for any forked descendant.
    ///
    /// Returns:
    ///     The root state ID if available, or None if the state has no tracked root.
    /// See `pending_api::_get_pending_root_state_id` for the body.
    pub fn get_pending_root_state_id(&self, state_id: u64) -> PyResult<Option<u64>> {
        self._get_pending_root_state_id(state_id)
    }

    /// Get the full ancestry chain for the pending callback state.
    ///
    /// Returns a list of state IDs starting with the current state and walking
    /// up the parent chain: [state_id, parent_id, grandparent_id, ...].
    ///
    /// This is used by Python to find cached state data when the current state
    /// is a multi-level fork of an original state.
    ///
    /// The walk is best-effort: it follows parent links only through states the
    /// manager still holds (pending callbacks + stashes) and stops at the first
    /// ancestor that has been consumed or dropped. The lineage root from
    /// `sm.roots()` is always appended when not already present, so the list is
    /// never shorter than the previous `[state, parent, root]` behaviour.
    /// See `pending_api::_get_pending_ancestry` for the body.
    pub fn get_pending_ancestry(&self, state_id: u64) -> PyResult<Vec<u64>> {
        self._get_pending_ancestry(state_id)
    }

    /// Fork the pending state's solver context for Python callbacks.
    ///
    /// This creates a new RustSolverContext that inherits all constraints
    /// accumulated during Rust exploration. The forked context can be
    /// attached to the Python callback state, ensuring SimProcedures
    /// see the full constraint context.
    ///
    /// This is critical for proper constraint propagation: without it,
    /// callbacks would create fresh solver contexts without parent
    /// constraints, leading to incorrect symbolic evaluation.
    /// Export a callback bundle: registers, solver context, history, jumpkind
    /// in a single FFI call. Reduces ~20 individual calls to 1.
    ///
    /// Returns a Python dict with:
    /// - "registers": dict of register_name -> concrete u128 value (None if symbolic)
    /// - "solver": forked RustSolverContext
    /// - "history": list of u64 BBL addresses
    /// - "jumpkind": string
    /// - "constraint_count": u64
    /// - "stdout": bytes (accumulated stdout buffer)
    /// See `pending_api::_export_callback_bundle` for the body.
    #[pyo3(signature = (state_id, register_names, shared_solver=true))]
    pub fn export_callback_bundle<'py>(
        &self,
        py: Python<'py>,
        state_id: u64,
        register_names: Vec<String>,
        shared_solver: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        self._export_callback_bundle(py, state_id, register_names, shared_solver)
    }

    /// See `pending_api::_fork_pending_solver` for the body.
    pub fn fork_pending_solver(&self, state_id: u64) -> PyResult<RustSolverContext> {
        self._fork_pending_solver(state_id)
    }

    /// Borrow the pending state's solver context without forking.
    ///
    /// Returns a RustSolverContext that shares the same Z3 solver as the
    /// pending state via Rc reference counting. This is O(1) instead of
    /// the ~3ms Z3 solver clone in fork_pending_solver().
    ///
    /// Constraints added through this solver go directly to the pending state,
    /// so post-callback constraint sync via add_constraints_to_pending() should
    /// be skipped to avoid double-adding.
    /// See `pending_api::_borrow_pending_solver` for the body.
    pub fn borrow_pending_solver(&self, state_id: u64) -> PyResult<RustSolverContext> {
        self._borrow_pending_solver(state_id)
    }

    /// Add constraints from Python callbacks back to the pending state.
    ///
    /// This is called after a SimProcedure executes to sync any new
    /// constraints added during the callback back to the Rust solver.
    /// This ensures bidirectional constraint flow between Rust and Python.
    ///
    /// Args:
    ///     constraints: List of claripy AST constraints to add
    /// See `pending_api::_add_constraints_to_pending` for the body.
    pub fn add_constraints_to_pending(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        constraints: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<()> {
        self._add_constraints_to_pending(py, state_id, constraints)
    }

    /// Add constraints from Python to a state in a stash by state ID.
    /// This is used to sync initial constraints from the Python state.
    pub fn add_constraints_to_state(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        constraints: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<bool> {
        self._add_constraints_to_state(py, state_id, constraints)
    }

    /// Export raw Z3 assertion pointers from a state's solver.
    /// Lossless — captures ALL Z3 assertions, not just those tracked
    /// in assumed_constraints (which drops constraints where claripy_to_rustbv fails).
    #[cfg(feature = "vex-engine-z3")]
    pub fn export_z3_constraint_ptrs(&self, state_id: u64) -> PyResult<Vec<usize>> {
        self._export_z3_constraint_ptrs(state_id)
    }

    /// Import raw Z3 assertion pointers to a state's solver.
    #[cfg(feature = "vex-engine-z3")]
    pub fn import_z3_constraint_ptrs(&mut self, state_id: u64, ptrs: Vec<usize>) -> PyResult<bool> {
        self._import_z3_constraint_ptrs(state_id, ptrs)
    }

    /// Debug: dump solver state for a given state.
    ///
    /// Interactive-debugging tool — it has no production caller in `angr/` by
    /// design (angr-sqfj8.59). The key set it emits is still a de-facto API,
    /// so `TestDebugSolverInfo` in `tests/engines/rust/test_misc.py` pins it
    /// and cross-checks `exported_ptrs` against `export_z3_constraint_ptrs`.
    #[cfg(feature = "vex-engine-z3")]
    pub fn debug_solver_info(&self, state_id: u64) -> PyResult<String> {
        self._debug_solver_info(state_id)
    }

    /// Export constraints from a state in any stash as claripy ASTs.
    pub fn export_state_constraints(
        &self,
        py: Python<'_>,
        state_id: u64,
    ) -> PyResult<Vec<Py<PyAny>>> {
        self._export_state_constraints(py, state_id)
    }

    /// Unsat core for a state, as the subset of `export_state_constraints`
    /// that Z3 blames for the contradiction. Empty when the state is SAT.
    pub fn state_unsat_core(
        &self,
        py: Python<'_>,
        state_id: u64,
        extra_constraints: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        self._state_unsat_core(py, state_id, extra_constraints)
    }

    /// Set the Z3 solver timeout (ms) on a specific state's solver context.
    ///
    /// Future forks of this state inherit the new timeout.  Used by
    /// RustSolverProxy to honor `state.solver.timeout = N` assignments.
    pub fn set_state_solver_timeout(&self, state_id: u64, timeout_ms: u32) -> PyResult<()> {
        self._set_state_solver_timeout(state_id, timeout_ms)
    }

    /// Get the Z3 solver timeout (ms) on a specific state's solver context.
    pub fn get_state_solver_timeout(&self, state_id: u64) -> PyResult<u32> {
        self._get_state_solver_timeout(state_id)
    }

    /// Get the per-state mmap base pointer (mirrors Python's
    /// `state.heap.mmap_base`). The native mmap syscall handler bumps this
    /// on `addr=0` calls; Python imports it on stash export to keep the two
    /// engines from handing out overlapping mmap regions.
    pub fn get_state_mmap_base(&self, state_id: u64) -> PyResult<u64> {
        self._get_state_mmap_base(state_id)
    }

    /// Set the per-state mmap base pointer. Used by tests and by Python-side
    /// fallbacks that allocate from `state.heap.mmap_base` and need to push
    /// the advance back into Rust so subsequent native mmaps don't collide.
    pub fn set_state_mmap_base(&mut self, state_id: u64, addr: u64) -> PyResult<()> {
        self._set_state_mmap_base(state_id, addr)
    }

    /// Get the per-state posix brk pointer (mirrors Python's
    /// `state.posix.brk`). The native brk syscall handler bumps this on
    /// concrete `brk(addr)` calls; Python imports it on stash export to keep
    /// a Python-side `set_brk` fallback from handing out heap addresses that
    /// overlap a Rust-allocated region.
    pub fn get_state_posix_brk(&self, state_id: u64) -> PyResult<u64> {
        self._get_state_posix_brk(state_id)
    }

    /// Set the per-state posix brk pointer. Used by tests and by Python-side
    /// fallbacks (symbolic `brk` argument, collision retry) that bump
    /// `state.posix.brk` and need to push the advance back into Rust.
    pub fn set_state_posix_brk(&mut self, state_id: u64, addr: u64) -> PyResult<()> {
        self._set_state_posix_brk(state_id, addr)
    }

    /// Get the per-state heap brk pointer (mirrors Python's
    /// `state.heap.heap_location`, the malloc bump allocator). Native
    /// heap-allocating procedures (malloc/calloc/realloc/strdup/fopen) bump
    /// this via `heap_alloc`; Python imports it on stash export so a Python
    /// fallback SimProcedure doesn't hand out an address Rust already
    /// allocated. See bead angr-um39j.
    pub fn get_state_heap_brk(&self, state_id: u64) -> PyResult<u64> {
        self._get_state_heap_brk(state_id)
    }

    /// Set the per-state heap brk pointer. Used by tests and on import to
    /// push a Python-side `state.heap.heap_location` advance back into Rust
    /// so subsequent native allocations don't collide.
    pub fn set_state_heap_brk(&mut self, state_id: u64, addr: u64) -> PyResult<()> {
        self._set_state_heap_brk(state_id, addr)
    }

    // ----- Per-state Python-AST metadata (symbolic_pages /
    //       hook_symbolic_memory / addr_to_ast). Storage now lives in
    //       RustSimState; these methods are the FFI surface that replaces the
    //       old Python `_state_metadata: Dict[int, StateMetadata]` map.

    /// Replace the whole `symbolic_pages` map for a state. Mirrors the
    /// previous Python-side `_state_md(sid).symbolic_pages = pages` write.
    pub fn set_state_symbolic_pages<'py>(
        &mut self,
        py: Python<'py>,
        state_id: u64,
        pages: &Bound<'py, PyDict>,
    ) -> PyResult<()> {
        self._set_state_symbolic_pages(py, state_id, pages)
    }

    /// Snapshot the `symbolic_pages` map for a state as a Python dict.
    /// Returns an empty dict if the state is unknown or has no entries — the
    /// truthiness check at call sites already handles both cases.
    pub fn get_state_symbolic_pages<'py>(
        &self,
        py: Python<'py>,
        state_id: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        self._get_state_symbolic_pages(py, state_id)
    }

    /// Insert/replace an entry in `hook_symbolic_memory` for a state.
    pub fn set_state_hook_symbolic_memory(
        &mut self,
        state_id: u64,
        addr: u64,
        ast: Py<PyAny>,
        size: u32,
    ) -> PyResult<()> {
        self._set_state_hook_symbolic_memory(state_id, addr, ast, size)
    }

    /// Snapshot the `hook_symbolic_memory` map as a `dict[int, (ast, size)]`.
    pub fn get_state_hook_symbolic_memory<'py>(
        &self,
        py: Python<'py>,
        state_id: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        self._get_state_hook_symbolic_memory(py, state_id)
    }

    /// Insert/replace an entry in `addr_to_ast` for a state.
    pub fn set_state_addr_to_ast(
        &mut self,
        state_id: u64,
        addr: u64,
        ast: Py<PyAny>,
        size: u32,
    ) -> PyResult<()> {
        self._set_state_addr_to_ast(state_id, addr, ast, size)
    }

    /// Snapshot the `addr_to_ast` map as a `dict[int, (ast, size)]`.
    pub fn get_state_addr_to_ast<'py>(
        &self,
        py: Python<'py>,
        state_id: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        self._get_state_addr_to_ast(py, state_id)
    }

    /// Drop all per-state metadata (`symbolic_pages`, `hook_symbolic_memory`,
    /// `addr_to_ast`) for a state. No-op if the state is unknown — matches the
    /// `_state_metadata.pop(state_id, None)` semantics it replaces.
    pub fn clear_state_metadata(&mut self, state_id: u64) -> PyResult<()> {
        self._clear_state_metadata(state_id)
    }

    /// Fork the solver context of an arbitrary state (by ID).
    ///
    /// Returns a new RustSolverContext with all of the state's constraints,
    /// allowing Python to evaluate/solve against any state — not just the
    /// pending callback state.  This is used by RustStateProxy.
    pub fn fork_state_solver(&self, state_id: u64) -> PyResult<RustSolverContext> {
        self._fork_state_solver(state_id)
    }

    /// Load from pending callback state's Rust memory.
    /// Used by SimProcedure callbacks to read the correct per-state memory.
    /// Get all mapped page addresses from pending callback state's memory.
    /// See `pending_api::_get_pending_mapped_pages` for the body.
    pub fn get_pending_mapped_pages(&self, state_id: u64) -> PyResult<Vec<u64>> {
        self._get_pending_mapped_pages(state_id)
    }

    /// Load an entire page (4096 bytes) from pending callback state's memory.
    /// See `pending_api::_pending_memory_load_page` for the body.
    pub fn pending_memory_load_page(&self, state_id: u64, page_addr: u64) -> PyResult<Vec<u8>> {
        self._pending_memory_load_page(state_id, page_addr)
    }

    /// Symbolic counterpart of `pending_memory_load_page`: returns the
    /// (addr, claripy AST) pairs for every multi-byte symbolic object whose
    /// base address falls on the given page.
    /// See `pending_api::_pending_memory_load_symbolic_page` for the body.
    pub fn pending_memory_load_symbolic_page<'py>(
        &mut self,
        py: Python<'py>,
        state_id: u64,
        page_addr: u64,
    ) -> PyResult<Vec<(u64, Py<PyAny>)>> {
        self._pending_memory_load_symbolic_page(py, state_id, page_addr)
    }

    /// See `pending_api::_pending_memory_load` for the body.
    pub fn pending_memory_load(&self, state_id: u64, addr: u64, size: u32) -> PyResult<Vec<u8>> {
        self._pending_memory_load(state_id, addr, size)
    }

    /// Set address to skip hook check for on next step.
    ///
    /// This is used to prevent infinite loops with zero-length hooks.
    /// When a hook with length=0 runs, it returns to the same address.
    /// Without this skip mechanism, the hook would trigger again immediately.
    ///
    /// The skip is automatically cleared after one step or when the address is used.
    /// GAP 6: Stack-based tracking allows for nested zero-length hooks.
    /// See `pending_api::_set_skip_hook_addr` for the body.
    pub fn set_skip_hook_addr(&mut self, addr: u64) {
        self._set_skip_hook_addr(addr)
    }

    /// Clear all pending skip_hook entries.
    /// See `pending_api::_clear_skip_hook_addr` for the body.
    pub fn clear_skip_hook_addr(&mut self) {
        self._clear_skip_hook_addr()
    }

    /// Clear skip entry for a specific address.
    /// See `pending_api::_clear_skip_hook_for_addr` for the body.
    pub fn clear_skip_hook_for_addr(&mut self, addr: u64) {
        self._clear_skip_hook_for_addr(addr)
    }

    /// Get errors encountered during exploration.
    pub fn get_errors(&self) -> Vec<(u64, String, u64)> {
        self.errors.clone()
    }

    /// Clear error log.
    pub fn clear_errors(&mut self) {
        self.errors.clear();
    }

    /// Move states between stashes.
    ///
    /// `filter_fn` is called with the state id and its return value is tested
    /// for truth the way Python's `if` would; an exception it raises (directly
    /// or from `__bool__`) propagates rather than counting as "no match".
    /// See `state_lifecycle::_move_states` for the body.
    pub fn move_states(
        &mut self,
        from_stash: &str,
        to_stash: &str,
        filter_fn: Option<Py<PyAny>>,
    ) -> PyResult<usize> {
        self._move_states(from_stash, to_stash, filter_fn)
    }

    /// Move a single state by ID between stashes.
    /// See `state_lifecycle::_move_state` for the body.
    pub fn move_state(
        &mut self,
        state_id: u64,
        from_stash: &str,
        to_stash: &str,
    ) -> PyResult<bool> {
        self._move_state(state_id, from_stash, to_stash)
    }

    /// Clear all states from a stash.
    pub fn clear_stash(&mut self, stash: &str) {
        self.sm.clear(stash);
    }

    /// Drop a single state from a specific stash by ID (angr-yhe0).
    ///
    /// Removes the state from the stash's `VecDeque`, drops the lineage-root
    /// and state-index entries, and lets the `RustSimState` destructor free
    /// the Z3 solver clone. No-op (returns `false`) when the state is not in
    /// the named stash — the caller is responsible for picking the right
    /// stash (today this is only ever `"_copies"`, the holding area for
    /// `RustStateProxy.copy()` clones).
    ///
    /// Backs `RustStateProxy.__del__` — when a copy-proxy is GC'd by Python,
    /// the Rust-side state can be reclaimed without waiting for the whole
    /// manager to drop. Returns `true` when a state was actually dropped.
    pub fn drop_state_from_stash(&mut self, state_id: u64, stash: &str) -> bool {
        // Drop clears the root too (the state is gone for good); take_state_from
        // handles the stash removal + unindex (angr-ph300.27).
        if self.sm.take_state_from(state_id, stash).is_some() {
            // Dropping from STASH_ACTIVE outside `policy.select` — notify so a
            // memoizing policy doesn't leak a memo entry (angr-myzjx.25). Today
            // the caller only passes "_copies", so this is a no-op guard now,
            // but it keeps the invariant robust if the API gains callers.
            if stash == STASH_ACTIVE {
                self.policy.on_state_removed(state_id);
            }
            self.sm.remove_root(state_id);
            true
        } else {
            false
        }
    }

    /// Prepare for a new exploration stage: move a specific found state
    /// to active and clear all other stashes. Returns the state ID of the
    /// moved state. This avoids constraint transfer between managers.
    /// See `state_lifecycle::_reset_for_stage` for the body.
    pub fn reset_for_stage(&mut self, found_state_id: u64) -> PyResult<u64> {
        self._reset_for_stage(found_state_id)
    }

    /// Get statistics.
    /// See `stats_api::_stats` for the body.
    pub fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self._stats(py)
    }

    /// Get VEX fallback statistics: total count and per-address reasons.
    ///
    /// Returns a dict with:
    ///   "count": total number of VEX fallbacks
    ///   "addresses": dict mapping hex address string -> reason string
    ///   "dcas_unsupported_count": subset of fallbacks driven by double-CAS
    ///   "simprocedure_python_fallback_count": SimProcedures dispatched to Python
    ///   "syscall_python_fallback_count": syscalls dispatched to Python
    ///   "syscall_native_count": syscalls handled natively (no Python round-trip)
    /// See `stats_api::_get_fallback_stats` for the body.
    pub fn get_fallback_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self._get_fallback_stats(py)
    }
}
