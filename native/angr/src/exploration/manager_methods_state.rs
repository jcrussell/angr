//! `#[pymethods]` for [`RustExplorationManager`]: state lifecycle, stash queries and pending-state accessors.
//!
//! One of several `#[pymethods]` blocks for the pyclass; see the
//! `manager_methods` module doc for the split rationale (angr-9ke6b.50).
//!
//! This is a Python-boundary module; `unwrap`/`expect` are denied here so a
//! future panic-on-input landmine cannot be reintroduced without a reviewed,
//! reasoned `#[allow]` (angr-qwyti.11 enforcement layer).
#![deny(clippy::unwrap_used, clippy::expect_used)]
use super::*;

#[angr_macros::steady_guard_checked]
#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
#[pymethods]
impl RustExplorationManager {
    /// Create a new RustSimState and add it to a stash.
    /// See `state_lifecycle::_create_state` for the body.
    #[pyo3(signature = (stash="active"))]
    #[angr_macros::steady_guarded]
    pub fn create_state(&mut self, stash: &str) -> PyResult<u64> {
        self._create_state(stash)
    }

    /// Add an existing RustSimState to a stash.
    /// See `state_lifecycle::_add_state` for the body.
    #[pyo3(signature = (stash, state))]
    #[angr_macros::steady_guarded]
    pub fn add_state(&mut self, stash: &str, state: &crate::state::PyRustSimState) {
        self._add_state(stash, state)
    }

    /// Merge multiple states into one using symbolic merge conditions.
    ///
    /// Each state's constraints are guarded by a fresh 1-bit merge flag.
    /// Registers and memory that differ between states become ITE expressions.
    /// The merged state is placed into `dest_stash`.
    ///
    /// Returns the merged state's ID.
    /// See `state_lifecycle::_merge_states` for the body.
    #[pyo3(signature = (state_ids, dest_stash="active"))]
    #[angr_macros::steady_guard_exempt(
        reason = "driven by the MergePoint native technique's per-group merge, which can fire \
                  during ordinary exploration (not just at stage boundaries); guarding it \
                  finalizes a live steady session on every merge and starves the bounce/resume \
                  protocol of a resident frontier, same failure mode move_states had."
    )]
    pub fn merge_states(&mut self, state_ids: Vec<u64>, dest_stash: &str) -> PyResult<u64> {
        self._merge_states(state_ids, dest_stash)
    }

    /// Fork an existing state (including the pending callback state) and add
    /// the fork to `stash`. Returns the new state's ID. Inherits the parent's
    /// lineage root.
    ///
    /// Write-through SimProc fork API (angr-t3mr). See
    /// `state_lifecycle::_fork_state_to_stash` for the body.
    #[pyo3(signature = (parent_id, stash="active"))]
    #[angr_macros::steady_guard_exempt(
        reason = "called from the ordinary per-callback fork dispatch path \
                  (_add_forked_state_via_rust) and from RustStateProxy.copy(); guarding it \
                  finalizes a live steady session on every callback-driven fork and starves the \
                  bounce/resume protocol of a resident frontier, same failure mode move_states \
                  had."
    )]
    pub fn fork_state_to_stash(&mut self, parent_id: u64, stash: &str) -> PyResult<u64> {
        self._fork_state_to_stash(parent_id, stash)
    }

    /// Get the PC of a state in a stash by index.
    ///
    /// Applies the same angr-4rq7 pc==0 IP fallback as `get_state_pc_by_id` so
    /// the two accessors agree for a stale-pc forked state (angr-ph300.23).
    #[pyo3(signature = (stash="active", index=0))]
    pub fn get_state_pc(&self, stash: &str, index: usize) -> Option<u64> {
        self.sm
            .get(stash)
            .and_then(|s| s.get(index))
            .map(super::native_proc_dispatch::effective_pc)
    }

    /// Get the PC of a state by its ID (O(1) via state index, no full export).
    pub fn get_state_pc_by_id(&self, state_id: u64) -> Option<u64> {
        // find_state already checks pending_callback first.
        // angr-4rq7 (root cause #2): see `native_proc_dispatch::effective_pc` for why a
        // pc==0 state falls back to the IP register.
        self.find_state(state_id)
            .map(super::native_proc_dispatch::effective_pc)
    }

    /// Get the tail of a state's bbl history (last `n` addresses).
    /// Avoids cloning the full Vec on long benches where max_history=0 means
    /// `history` may grow to 100k+ entries.
    /// `n == 0` is treated as "all entries". Returns None if state not found.
    #[pyo3(signature = (state_id, n=256))]
    pub fn get_state_bbl_history_tail(&self, state_id: u64, n: usize) -> Option<Vec<u64>> {
        let hist = self.find_state(state_id)?.history();
        let start = if n == 0 {
            0
        } else {
            hist.len().saturating_sub(n)
        };
        Some(hist.range(start..).copied().collect())
    }

    /// Get state IDs in a stash.
    #[pyo3(signature = (stash="active"))]
    pub fn get_state_ids(&self, stash: &str) -> Vec<u64> {
        self.sm
            .get(stash)
            .map(|s| s.iter().map(crate::state::RustSimState::state_id).collect())
            .unwrap_or_default()
    }

    /// State ids of the callbacks currently parked in `pending_callbacks`
    /// (angr-op0dn.13.14). A parked callback's state lives in NO stash, so it
    /// is invisible to `get_state_ids` / `stash_counts` — this is the only way
    /// for Python (and the snapshot tests) to observe it.
    pub fn pending_callback_ids(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.pending_callbacks.keys().map(|k| k.raw()).collect();
        ids.sort_unstable();
        ids
    }

    /// Get (state_id, addr, stdout_len) tuples for states in a stash.
    /// Used by Python predicate caching to skip re-evaluation when
    /// a state's address and stdout haven't changed.
    ///
    /// `addr` goes through `native_proc_dispatch::effective_pc`: a stale-pc forked successor
    /// would otherwise be cached under key `(sid, 0)` and its find predicate
    /// evaluated at address 0, missing the genuine find (angr-ph300.23).
    #[pyo3(signature = (stash="active"))]
    pub fn get_state_predicate_info(&self, stash: &str) -> Vec<(u64, u64, usize)> {
        self.sm
            .get(stash)
            .map(|s| {
                s.iter()
                    .map(|state| {
                        (
                            state.state_id(),
                            super::native_proc_dispatch::effective_pc(state),
                            state.stdout_buffer().len(),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check if there are any active states (O(1), no allocation).
    pub fn has_active_states(&self) -> bool {
        self.sm.has_active()
    }

    /// Get the number of states in a stash (O(1), no allocation).
    #[pyo3(signature = (stash="active"))]
    pub fn stash_count(&self, stash: &str) -> usize {
        self.sm
            .get(stash)
            .map_or(0, std::collections::VecDeque::len)
    }

    /// Get the stash name a state currently belongs to (O(1) via state index).
    /// Returns None if the state isn't found in any stash. Used by
    /// RustStateProxy.__repr__ for cheap REPL debugging output.
    pub fn state_stash(&self, state_id: u64) -> Option<String> {
        self.sm
            .stash_of(state_id)
            .map(std::string::ToString::to_string)
    }

    /// Get the number of solver constraints for a state (O(1), reads
    /// the SymContext's atomic counter — no Z3 traversal). Returns None
    /// if the state isn't found. Used by RustStateProxy.__repr__.
    pub fn state_constraint_count(&self, state_id: u64) -> Option<usize> {
        let state = self.find_state(state_id)?;
        Some(state.solver().borrow().num_constraints())
    }

    /// Rebuild the state index after run() modifies stashes internally.
    /// Call from Python after run() returns to keep index up to date.
    #[angr_macros::steady_guard_exempt(
        reason = "rebuilds an internal lookup index from existing stash state; called after \
                  run() completes, mutates no exploration config or stash membership."
    )]
    pub fn sync_state_index(&mut self) {
        self.rebuild_state_index();
    }

    /// Get the root state ID for any state.
    /// Returns the original (initial) state from which this state was forked.
    pub fn get_state_root(&self, state_id: u64) -> Option<u64> {
        self.sm.roots().get(&state_id).copied()
    }

    /// Set the PC of the pending callback state (for external initialization).
    /// See `pending_api::_set_pending_state_pc` for the body.
    #[angr_macros::steady_guard_exempt(
        reason = "mutates one state_id-scoped state's own PC; does not mutate exploration \
                  config or the active stash."
    )]
    pub fn set_pending_state_pc(&mut self, state_id: u64, pc: u64) -> PyResult<()> {
        self._set_pending_state_pc(state_id, pc)
    }

    /// Map memory in active states.
    /// See `pending_api::_active_states_map_memory` for the body.
    #[pyo3(signature = (addr, data, permissions=7))]
    #[angr_macros::steady_guarded]
    pub fn active_states_map_memory(&mut self, addr: u64, data: &[u8], permissions: u8) {
        // Steady-state: this iterates STASH_ACTIVE, which misses a resident
        // frontier — finalize first so every live state is back in the stash.
        self._active_states_map_memory(addr, data, permissions)
    }

    /// Get the branch condition from the pending symbolic branch callback.
    ///
    /// Returns the condition as a claripy AST that Python can use for forking.
    /// See `pending_api::_get_pending_branch_condition` for the body.
    pub fn get_pending_branch_condition(
        &self,
        py: Python<'_>,
        state_id: u64,
    ) -> PyResult<Py<PyAny>> {
        self._get_pending_branch_condition(py, state_id)
    }

    /// Get register value from pending state (concrete only).
    /// Raises `ValueError` for an unknown register name, so `None` means
    /// "known register, symbolic value" — unlike `get_state_register`, which
    /// folds both cases into `None`.
    /// See `pending_api::_get_pending_register` for the body.
    pub fn get_pending_register(&self, state_id: u64, name: &str) -> PyResult<Option<u128>> {
        self._get_pending_register(state_id, name)
    }

    /// Get register as claripy AST from pending state (handles symbolic).
    /// See `pending_api::_get_pending_register_ast` for the body.
    pub fn get_pending_register_ast(
        &self,
        py: Python<'_>,
        state_id: u64,
        name: &str,
    ) -> PyResult<Py<PyAny>> {
        self._get_pending_register_ast(py, state_id, name)
    }

    /// Get history (BBL addresses) and jumpkind from pending callback state
    /// in a single FFI call. Fetching both in one crossing avoids the GIL +
    /// boundary-crossing cost of two separate calls from the callback
    /// dispatcher hot path. Used by Python to initialize history + callstack
    /// on callback states, preventing IndexError when hooks access
    /// `state.history.recent_bbl_addrs[-1]`.
    /// See `pending_api::_get_pending_history_and_jumpkind` for the body.
    pub fn get_pending_history_and_jumpkind(&self, state_id: u64) -> PyResult<(Vec<u64>, String)> {
        self._get_pending_history_and_jumpkind(state_id)
    }

    /// Set register value in pending state.
    /// See `pending_api::_set_pending_register` for the body.
    #[angr_macros::steady_guard_exempt(
        reason = "mutates one state_id-scoped state's own register content; does not mutate \
                  exploration config or the active stash."
    )]
    pub fn set_pending_register(&mut self, state_id: u64, name: &str, value: u128) -> PyResult<()> {
        self._set_pending_register(state_id, name, value)
    }

    /// Set register to a symbolic value from a handle ID.
    ///
    /// Used for syncing symbolic return values from SimProcedures.
    /// The handle_id should reference a RustBV in the solver's symbol table.
    /// See `pending_api::_set_pending_register_symbolic` for the body.
    #[angr_macros::steady_guard_exempt(
        reason = "mutates one state_id-scoped state's own register content; does not mutate \
                  exploration config or the active stash."
    )]
    pub fn set_pending_register_symbolic(
        &mut self,
        state_id: u64,
        name: &str,
        handle_id: u64,
    ) -> PyResult<()> {
        self._set_pending_register_symbolic(state_id, name, handle_id)
    }

    /// Set a symbolic register value in the pending state from claripy AST.
    ///
    /// This allows direct sync of symbolic register values from Python callbacks.
    /// The claripy AST is converted to RustBV and stored in the pending state.
    /// See `pending_api::_set_pending_register_symbolic_ast` for the body.
    #[angr_macros::steady_guard_exempt(
        reason = "mutates one state_id-scoped state's own register content; does not mutate \
                  exploration config or the active stash."
    )]
    pub fn set_pending_register_symbolic_ast(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        reg_name: &str,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self._set_pending_register_symbolic_ast(py, state_id, reg_name, ast)
    }

    /// Import symbolic memory from Python hook into Rust's symbolic_objects.
    ///
    /// Called after a hook writes symbolic memory. Converts the claripy AST
    /// to RustBV and imports it into the pending state's SymbolicMemory.
    /// Import symbolic memory into a state by ID (for init-time symbolic data).
    /// See `pending_api::_import_symbolic_to_state` for the body.
    #[pyo3(signature = (state_id, addr, ast))]
    #[angr_macros::steady_guard_exempt(
        reason = "mutates one state_id-scoped state's own symbolic memory; does not mutate \
                  exploration config or the active stash."
    )]
    pub fn import_symbolic_to_state(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        addr: u64,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self._import_symbolic_to_state(py, state_id, addr, ast)
    }

    /// See `pending_api::_import_symbolic_memory` for the body.
    #[angr_macros::steady_guard_exempt(
        reason = "mutates one state_id-scoped state's own symbolic memory; does not mutate \
                  exploration config or the active stash."
    )]
    pub fn import_symbolic_memory(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        addr: u64,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self._import_symbolic_memory(py, state_id, addr, ast)
    }

    /// Replace a state's filesystem current working directory (angr-0xyq2
    /// Phase 3). Python `state.fs._files` keys are cwd-normalized (default
    /// `/home/user`) while the Rust `FileSystem` cwd defaults to `/`, so the
    /// init-time export must push the Python cwd BEFORE registering file
    /// content — otherwise a relative guest `open()` never matches the
    /// registry keys. Non-UTF-8 cwds are gated out by the Python caller
    /// (the Rust path model is UTF-8-lossy).
    /// See `pending_api::_set_fs_cwd` for the body.
    #[angr_macros::steady_guard_exempt(
        reason = "mutates one state_id-scoped state's own filesystem cwd; does not mutate \
                  exploration config or the active stash."
    )]
    pub fn set_fs_cwd(&mut self, state_id: u64, cwd: &str) -> PyResult<()> {
        self._set_fs_cwd(state_id, cwd)
    }

    /// Register bounded symbolic file content for `path` on a state's
    /// filesystem (angr-0xyq2 Phase 3): one 8-bit claripy AST per byte,
    /// converted to `RustBV` via the claripy bridge (which preserves BVS
    /// identity by hash and name+width, so constraints added natively on
    /// these bytes evaluate correctly against the original Python ASTs on
    /// the found state — no sync-back injection needed). A subsequent
    /// native `open()` of the (cwd-normalized) path attaches the content
    /// and reads are served natively. Errors (not panics) on unknown
    /// `state_id`, a non-convertible AST, or a non-8-bit entry.
    /// See `pending_api::_register_file_content` for the body.
    #[angr_macros::steady_guard_exempt(
        reason = "mutates one state_id-scoped state's own filesystem content; does not mutate \
                  exploration config or the active stash."
    )]
    pub fn register_file_content(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        path: &str,
        byte_asts: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<()> {
        self._register_file_content(py, state_id, path, byte_asts)
    }

    /// Seed fd 0 with the harness's own symbolic stdin bytes (angr-mb09c):
    /// one 8-bit claripy AST per byte, in file order, taken from a
    /// Python-filled `state.posix.stdin.content`. Native `read(0, ...)`
    /// serves these instead of minting fresh `stdin_*` symbols, so the path
    /// condition on an exported found state references the harness's BVS and
    /// `solver.eval(bvs)` / `posix.dumps(0)` return the real solution. Reads
    /// past the seeded content fall back to fresh symbols. Errors (not
    /// panics) on unknown `state_id`, a non-convertible AST, or a non-8-bit
    /// entry. See `pending_api::_seed_stdin_content` for the body.
    #[angr_macros::steady_guard_exempt(
        reason = "mutates one state_id-scoped state's own stdin fd content; does not mutate \
                  exploration config or the active stash."
    )]
    pub fn seed_stdin_content(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        byte_asts: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<()> {
        self._seed_stdin_content(py, state_id, byte_asts)
    }

    /// The cwd-normalized paths a state's lineage has demoted to Python
    /// ownership via native writes (angr-qluof). The Python re-add path
    /// (merge / legacy-fork push) queries these before re-registering
    /// exported content so a demoted path is not re-armed on the new
    /// state. Errors on unknown `state_id`.
    /// See `pending_api::_get_demoted_paths` for the body.
    pub fn get_demoted_paths(&self, state_id: u64) -> PyResult<Vec<String>> {
        self._get_demoted_paths(state_id)
    }

    /// Re-apply a symbolic-content demotion for `path` on `state_id`
    /// (angr-qluof): drops any registered content the export just re-armed,
    /// WITHOUT bumping the native write-demotion counter. Returns `true`
    /// when registered content or fd state was cleared. Errors on unknown
    /// `state_id`. See `pending_api::_demote_file_path` for the body.
    #[angr_macros::steady_guard_exempt(
        reason = "mutates one state_id-scoped state's own filesystem content; does not mutate \
                  exploration config or the active stash."
    )]
    pub fn demote_file_path(&mut self, state_id: u64, path: &str) -> PyResult<bool> {
        self._demote_file_path(state_id, path)
    }

    /// Get memory from pending state.
    /// See `pending_api::_get_pending_memory` for the body.
    pub fn get_pending_memory(&self, state_id: u64, addr: u64, size: u32) -> PyResult<Vec<u8>> {
        self._get_pending_memory(state_id, addr, size)
    }

    /// Get dirty page addresses from pending state.
    ///
    /// This returns the list of page-aligned addresses that have been
    /// modified in the pending callback state.
    /// See `pending_api::_get_pending_dirty_pages` for the body.
    pub fn get_pending_dirty_pages(&self, state_id: u64) -> PyResult<Vec<u64>> {
        self._get_pending_dirty_pages(state_id)
    }

    /// Clear dirty page tracking in pending state.
    /// See `pending_api::_clear_pending_dirty_tracking` for the body.
    #[angr_macros::steady_guard_exempt(
        reason = "clears one state_id-scoped state's own dirty-page tracking; does not mutate \
                  exploration config or the active stash."
    )]
    pub fn clear_pending_dirty_tracking(&mut self, state_id: u64) -> PyResult<()> {
        self._clear_pending_dirty_tracking(state_id)
    }
}

test_submod!("manager_methods_state_tests.rs" => tests);
