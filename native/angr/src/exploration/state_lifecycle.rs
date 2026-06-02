//! State lifecycle and stash-mutation methods.
//!
//! Bodies for the pyclass-exposed methods that create, add, merge, and move
//! `RustSimState` instances across stashes — `create_state`, `add_state`,
//! `merge_states`, `move_states`, `move_state`, and `reset_for_stage`. The
//! pyclass-facing thin wrappers live in `mod.rs` and forward to the
//! `pub(crate)` bodies in this module.
//!
//! PyO3 0.27.2 in this project does not enable `multiple-pymethods`, so each
//! pyclass is limited to a single `#[pymethods]` impl block — see
//! `invariant-pyo3-single-pymethods-impl`. This module mirrors the
//! `helpers.rs` / `stepping.rs` / `run_loop.rs` / `resume.rs` /
//! `pending_api.rs` / `state_api.rs` extension-impl pattern used elsewhere in
//! `exploration/`.
//!
//! # Cross-cutting invariants enforced here
//!
//! - **`state-id-never-reused`** — every state added or merged into a stash
//!   gets a fresh monotonic ID from `next_state_id()` (allocated inside
//!   `RustSimState::fork`/`new`/`merge`). This is what makes it safe for
//!   `_cleanup_state_cache` (Python) to drop shadow mappings keyed by
//!   `state_id` against `any_stash` membership. See `state.rs`
//!   module-level header.
//! - **`state-lifecycle-stats-api`** — the bodies for `create_state` /
//!   `add_state` / `merge_states` / `move_states` / `move_state` /
//!   `reset_for_stage` all live in THIS file (Rust core), with thin PyO3
//!   wrappers in `mod.rs`. Python-side dispatcher is
//!   `angr/exploration/rust_manager.py`. When extending lifecycle, edit
//!   here first; the wrapper in `mod.rs` should remain a single-line
//!   `self._method_name(...)`.
//! - **`state-cache-pinning`** (Python-side) — `_cleanup_state_cache` must
//!   pin `_state_roots`, `_current_callback_state_id`, and
//!   `_current_stepping_state_id`. Rust participates only by owning the
//!   per-state `RustSimState` (Drop on eviction) and by maintaining
//!   `state_roots`/`state_index` in `StashManager` — see `stash.rs`.

use super::*;
use crate::symbolic::DEFAULT_SOLVER_TIMEOUT_MS;

impl RustExplorationManager {
    pub(crate) fn _create_state(&mut self, stash: &str) -> PyResult<u64> {
        let mut state = RustSimState::new_with_endian(
            &self.environment.arch_name,
            self.environment.little_endian,
        )
        .map_err(PyValueError::new_err)?;
        let state_id = state.state_id();

        // Propagate memory options
        if self.memory_config.zero_fill_unconstrained {
            state.memory_mut().set_zero_fill_unconstrained(true);
        }

        // Propagate solver timeout
        if self.constraint_solver.solver_timeout_ms != DEFAULT_SOLVER_TIMEOUT_MS {
            state
                .solver()
                .borrow()
                .set_timeout(self.constraint_solver.solver_timeout_ms);
        }

        // Propagate per-state history cap
        state.set_max_history(self.environment.max_history);

        self.index_state(state_id, stash);
        self.sm.ensure_stash(stash).push_back(state);

        Ok(state_id)
    }

    pub(crate) fn _add_state(&mut self, stash: &str, state: &crate::state::PyRustSimState) {
        // Fork the state to get our own copy
        let mut forked = state.inner().fork();
        let state_id = forked.state_id();
        // `state-id-never-reused`: the fork must have allocated a fresh ID
        // distinct from the source PyRustSimState's ID. Tautological today
        // (fork() always calls next_state_id()); the assert catches a
        // future refactor that tried to "reuse" the source ID to avoid
        // breaking a Python-side mapping.
        debug_assert_ne!(
            state_id,
            state.inner().state_id(),
            "fork() must mint a fresh state_id, got duplicate {}",
            state_id,
        );

        // Propagate memory options
        if self.memory_config.zero_fill_unconstrained {
            forked.memory_mut().set_zero_fill_unconstrained(true);
        }

        // Propagate solver timeout
        if self.constraint_solver.solver_timeout_ms != DEFAULT_SOLVER_TIMEOUT_MS {
            forked
                .solver()
                .borrow()
                .set_timeout(self.constraint_solver.solver_timeout_ms);
        }

        // Propagate per-state history cap
        forked.set_max_history(self.environment.max_history);

        // Track this state as its own root (it was added via Python).
        // See `state-id-never-reused`: the root entry is keyed by the
        // monotonic ID, so it survives every subsequent fork descendant
        // (which inherit via `index_state` calls during the run loop).
        self.sm.set_root(state_id, state_id);

        self.index_state(state_id, stash);
        self.sm.ensure_stash(stash).push_back(forked);
    }

    pub(crate) fn _merge_states(&mut self, state_ids: Vec<u64>, dest_stash: &str) -> PyResult<u64> {
        if state_ids.len() < 2 {
            return Err(PyValueError::new_err(
                "merge_states requires at least 2 state IDs",
            ));
        }

        // Look up all states by ID across all stashes
        let mut states: Vec<RustSimState> = Vec::new();
        for &sid in &state_ids {
            let mut found = false;
            for (_stash_name, stash) in self.sm.stashes() {
                for state in stash.iter() {
                    if state.state_id() == sid {
                        states.push(state.fork());
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }
            if !found {
                return Err(PyValueError::new_err(format!("state {} not found", sid)));
            }
        }

        // Create merge conditions: one 1-bit BVS per state
        let solver = states[0].solver();
        let merge_conditions: Vec<RustBV> = (0..states.len())
            .map(|i| {
                let name = format!("merge_flag_{}", i);
                solver.borrow().new_bv(&name, 1)
            })
            .collect();

        // Perform the merge
        let others: Vec<&RustSimState> = states[1..].iter().collect();
        let merged = states[0].merge(&others, &merge_conditions);
        let merged_id = merged.state_id();

        // Track state root
        self.sm.set_root(merged_id, merged_id);
        self.index_state(merged_id, dest_stash);
        self.sm.ensure_stash(dest_stash).push_back(merged);

        Ok(merged_id)
    }

    pub(crate) fn _move_states(
        &mut self,
        from_stash: &str,
        to_stash: &str,
        filter_fn: Option<Py<PyAny>>,
    ) -> PyResult<usize> {
        // If no filter, move all
        if filter_fn.is_none() {
            if let Some(mut from) = self.sm.remove(from_stash) {
                let count = from.len();
                // Update index for all moved states
                for state in from.iter() {
                    self.sm.index(state.state_id(), to_stash);
                }
                let to = self.sm.ensure_stash(to_stash);
                to.append(&mut from);
                self.sm.insert(from_stash, VecDeque::new());
                return Ok(count);
            }
            return Ok(0);
        }

        // With filter - evaluate Python filter_fn per state
        let filter_fn = filter_fn.expect("filter_fn checked before call");
        let from = match self.sm.stashes().get(from_stash) {
            Some(s) if !s.is_empty() => s,
            _ => return Ok(0),
        };

        // First pass: determine which states pass the filter (immutable borrow)
        let mut move_indices = Vec::new();
        Python::attach(|py| -> PyResult<()> {
            for (i, state) in from.iter().enumerate() {
                let result = filter_fn.call1(py, (state.state_id(),))?;
                if result.extract::<bool>(py).unwrap_or(false) {
                    move_indices.push(i);
                }
            }
            Ok(())
        })?;

        if move_indices.is_empty() {
            return Ok(0);
        }

        // Second pass: move matching states (mutable borrow)
        let mut moved = Vec::new();
        if let Some(from) = self.sm.get_mut(from_stash) {
            for &idx in move_indices.iter().rev() {
                if let Some(state) = from.remove(idx) {
                    moved.push(state);
                }
            }
        }
        // Update index and destination stash after releasing from-stash borrow
        for state in &moved {
            self.sm.index(state.state_id(), to_stash);
        }
        let count = moved.len();
        let to = self.sm.ensure_stash(to_stash);
        for state in moved.into_iter().rev() {
            to.push_back(state);
        }
        Ok(count)
    }

    pub(crate) fn _move_state(
        &mut self,
        state_id: u64,
        from_stash: &str,
        to_stash: &str,
    ) -> PyResult<bool> {
        // Find and remove the state from the source stash
        let mut found_state = None;
        if let Some(stash) = self.sm.get_mut(from_stash) {
            let mut idx = None;
            for (i, state) in stash.iter().enumerate() {
                if state.state_id() == state_id {
                    idx = Some(i);
                    break;
                }
            }
            if let Some(i) = idx {
                found_state = stash.remove(i);
            }
        }

        // Add to destination stash if found
        if let Some(state) = found_state {
            self.index_state(state_id, to_stash);
            self.sm.ensure_stash(to_stash).push_back(state);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub(crate) fn _reset_for_stage(&mut self, found_state_id: u64) -> PyResult<u64> {
        // Move the found state from 'found' to 'active'
        let moved = self._move_state(found_state_id, "found", "active")?;
        if !moved {
            return Err(PyValueError::new_err(format!(
                "state {} not found in 'found' stash",
                found_state_id
            )));
        }

        // Clear all other stashes
        for stash in &[
            STASH_FOUND,
            STASH_AVOID,
            STASH_DEADENDED,
            STASH_ERRORED,
            STASH_UNCONSTRAINED,
        ] {
            self.sm.clear(stash);
        }

        // Remove all other active states (keep only the moved one)
        if let Some(active) = self.sm.get_mut("active") {
            active.retain(|s| s.state_id() == found_state_id);
        }

        Ok(found_state_id)
    }
}
