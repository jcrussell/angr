//! State lifecycle and stash-mutation methods.
//!
//! Bodies for the pyclass-exposed methods that create, add, merge, and move
//! `RustSimState` instances across stashes — `create_state`, `add_state`,
//! `merge_states`, `move_states`, `move_state`, and `reset_for_stage`. The
//! pyclass-facing thin wrappers live in the `manager_methods_*.rs` family —
//! split between `manager_methods_state.rs` and
//! `manager_methods_constraints.rs` — and forward to the `pub(crate)` bodies
//! in this module. (They were in `mod.rs` until the `#[pymethods]` surface
//! was split out of it per angr-nbim4.1 / angr-9ke6b.50; see `mod.rs`'s own
//! module doc.)
//!
//! The extension-impl split here predates PyO3's `multiple-pymethods`
//! feature, which is now enabled (angr-9ke6b.50, see
//! `invariant-pyo3-multiple-pymethods-enabled`), so the single-block rule no
//! longer forces it. It is kept as a style choice: thin `#[pyo3]` wrappers
//! stay next to their siblings while the substantial bodies live here. This module mirrors the
//! `helpers.rs` / `stepping.rs` / `run_loop.rs` / `resume.rs` /
//! `pending_api.rs` / `state_api.rs` extension-impl pattern used elsewhere in
//! `exploration/`.
//!
//! # Cross-cutting invariants enforced here
//!
//! Each bullet is stated in full here; the backticked labels are anchors
//! for cross-referencing within the Rust tree, not bd memory keys.
//!
//! - **`state-id-never-reused`** — every state added or merged into a stash
//!   gets a fresh monotonic ID from `next_state_id()` (allocated inside
//!   `RustSimState::fork`/`new`/`merge`). This is what makes it safe for
//!   `_cleanup_state_cache` (Python) to drop shadow mappings keyed by
//!   `state_id` against `any_stash` membership. See the `state/mod.rs`
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
//!
//! This is a Python-boundary module; `unwrap`/`expect` are denied here so a
//! future panic-on-input landmine cannot be reintroduced without a reviewed,
//! reasoned `#[allow]` (angr-qwyti.11 enforcement layer).
#![deny(clippy::unwrap_used, clippy::expect_used)]

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

        // Propagate strict-deterministic witness selection (angr-op0dn.10.3)
        if self.constraint_solver.deterministic {
            super::constraints::apply_state_deterministic(&state, true);
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
        // breaking a Python-side mapping. Always-on (angr-9ke6b.220): a reused
        // ID silently clobbers the stash index, and `_add_state` is an
        // init-time path, not per-step.
        assert_ne!(
            state_id,
            state.inner().state_id(),
            "fork() must mint a fresh state_id, got duplicate {state_id}",
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

        // Propagate strict-deterministic witness selection (angr-op0dn.10.3).
        // The fork copies the source state's flag, so an explicit set is only
        // needed when the manager is deterministic and the source was not.
        if self.constraint_solver.deterministic {
            super::constraints::apply_state_deterministic(&forked, true);
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

    /// Fork an existing state (looked up by ID, including the pending callback
    /// state) into a new active stash entry and return the new state's ID.
    ///
    /// This is the write-through path for SimProcedure forks (angr-t3mr).
    /// Previously `_add_forked_state` (`rust_callback_dispatch.py`) called the
    /// Python-side `_add_rust_state` which eagerly re-pushed every register,
    /// memory page, and constraint from the post-callback `succ_state`. Under
    /// the write-through model the parent state already lives in Rust as the
    /// pending callback, so a fresh `RustSimState::fork()` is both correct and
    /// strictly cheaper than rebuilding from the Python `succ_state`.
    ///
    /// Path-specific constraints (e.g. branch conditions a SimProc added to
    /// one successor but not the other) are NOT applied here; the caller
    /// should follow up with `add_constraints_to_state(new_id, ...)`.
    pub(crate) fn _fork_state_to_stash(&mut self, parent_id: u64, stash: &str) -> PyResult<u64> {
        let forked = {
            let parent = self.find_state(parent_id).ok_or_else(|| {
                PyValueError::new_err(format!("fork_state_to_stash: state {parent_id} not found"))
            })?;
            parent.fork()
        };
        let new_id = forked.state_id();
        // `state-id-never-reused`: fork() must mint a fresh monotonic ID.
        // Always-on (angr-9ke6b.220), same rationale as `_add_state`.
        assert_ne!(
            new_id, parent_id,
            "fork() must mint a fresh state_id, got duplicate {new_id}"
        );

        // Inherit the parent's lineage root so descendants of a SimProc fork
        // stay grouped with their seed state. Falls back to the parent's own
        // ID if the parent had no explicit root (matches the run_loop fork
        // pattern in resume.rs / stepping.rs).
        let root_state_id = self.sm.get_root(parent_id).unwrap_or(parent_id);
        self.sm.set_root(new_id, root_state_id);

        self.index_state(new_id, stash);
        self.sm.ensure_stash(stash).push_back(forked);

        Ok(new_id)
    }

    pub(crate) fn _merge_states(&mut self, state_ids: Vec<u64>, dest_stash: &str) -> PyResult<u64> {
        if state_ids.len() < 2 {
            return Err(PyValueError::new_err(
                "merge_states requires at least 2 state IDs",
            ));
        }

        // Look up all states by ID via the state_index fast-path, copying each
        // (the sources stay in place for Python's _merge_drop). Was an O(n^2)
        // nested all-stash scan that ignored state_index entirely
        // (angr-ph300.27).
        //
        // `clone_for_merge`, not `fork`: fork resets the removal tombstones, so
        // forking here threw away the very sets `merge` needs to keep a
        // branch's `remove_hook`/`set_option(_, false)`/`unsetenv` from being
        // resurrected (angr-sqfj8.33).
        let mut states: Vec<RustSimState> = Vec::new();
        for &sid in &state_ids {
            match self.sm.find_state(sid) {
                Some(state) => states.push(state.clone_for_merge()),
                None => return Err(PyValueError::new_err(format!("state {sid} not found"))),
            }
        }

        // Create merge conditions: one 1-bit BVS per state
        let solver = states[0].solver();
        let merge_conditions: Vec<RustBV> = (0..states.len())
            .map(|i| {
                let name = format!("merge_flag_{i}");
                solver.borrow().new_bv(&name, 1)
            })
            .collect();

        // Perform the merge
        let others: Vec<&RustSimState> = states[1..].iter().collect();
        let merged = states[0].merge(&others, &merge_conditions);
        let merged_id = merged.state_id();
        // `state-id-never-reused`: the merge must mint a fresh monotonic ID
        // distinct from *every* input state's ID (not just one), matching the
        // guard on the sibling minting sites _add_state / _fork_state_to_stash.
        // Tautological today (RustSimState::merge() allocates via
        // next_state_id()); the assert catches a future refactor that tried to
        // reuse an input ID for the merged state. Always-on (angr-9ke6b.220):
        // merging is rare, so the linear scan over the (small) input ID list
        // is off any hot path.
        assert!(
            !state_ids.contains(&merged_id),
            "merge() must mint a fresh state_id, got duplicate {merged_id}",
        );

        // Track state root. `push_to_stash`, not a raw `ensure_stash().push_back`:
        // a merged state landing in active is a fork-insertion and must go
        // through `policy.on_fork` like every other one (angr-0jh0j.19).
        self.sm.set_root(merged_id, merged_id);
        self.push_to_stash(dest_stash, merged);

        // M3-4 (angr-op0dn.11.4): count the states this native merge consumed,
        // so the Python fast path's `states_merged_native` stat reflects how
        // many states were merged in-Rust without an export round trip.
        self.states_merged_native += state_ids.len() as u64;

        Ok(merged_id)
    }

    pub(crate) fn _move_states(
        &mut self,
        from_stash: &str,
        to_stash: &str,
        filter_fn: Option<Py<PyAny>>,
    ) -> PyResult<usize> {
        // Moving a stash onto itself is a no-op. The no-filter path below
        // removes the source deque and then overwrites the (identical) key
        // with an empty VecDeque, dropping every state it just re-appended
        // and dangling their index entries (angr-ph300.17). Guard here so all
        // paths agree on same-stash semantics.
        if from_stash == to_stash {
            return Ok(0);
        }

        // If no filter, move all
        if filter_fn.is_none() {
            if let Some(mut from) = self.sm.remove(from_stash) {
                let count = from.len();
                // Update index for all moved states
                for state in from.iter() {
                    self.sm.index(state.state_id(), to_stash);
                }
                // Leaving STASH_ACTIVE outside `policy.select` — notify so a
                // memoizing policy (LoopHeadRoundRobin's key_cache) doesn't leak
                // a memo entry (angr-myzjx.25).
                if from_stash == STASH_ACTIVE {
                    for state in from.iter() {
                        self.policy.on_state_removed(state.state_id());
                    }
                }
                let to = self.sm.ensure_stash(to_stash);
                to.append(&mut from);
                self.sm.insert(from_stash, VecDeque::new());
                return Ok(count);
            }
            return Ok(0);
        }

        // With filter - evaluate Python filter_fn per state
        #[allow(
            clippy::expect_used,
            reason = "internal invariant: this branch is only reached when filter_fn.is_some(), checked by the caller above"
        )]
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
                // Python predicates are conventionally truthy/falsy rather than
                // strictly `bool`, so mirror `if filter_fn(sid):`. The former
                // `extract::<bool>(py).unwrap_or(false)` both mis-read a truthy
                // non-bool return as "no match" and swallowed an error raised
                // from `__bool__`; `is_truthy()?` propagates instead of
                // silently matching nothing (angr-sqfj8.31).
                if result.bind(py).is_truthy()? {
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
        // Leaving STASH_ACTIVE outside `policy.select` — notify so a memoizing
        // policy doesn't leak a memo entry (angr-myzjx.25).
        if from_stash == STASH_ACTIVE {
            for state in &moved {
                self.policy.on_state_removed(state.state_id());
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
        // NB: unlike _move_states, _move_state deliberately does NOT short-
        // circuit from_stash == to_stash. A same-stash move removes the state
        // and re-appends it to the back — a reorder-to-back. This is load-
        // bearing: _StashDict.__setitem__ (rust_state_proxy.py) rebuilds a
        // stash in caller-specified order by iterating the desired order and
        // calling move_state(sid, cur, key) with cur == key for each state
        // already in `key` (angr-wxuo). _move_states' same-stash guard is
        // about avoiding a bulk double-insert/drop bug (angr-ph300.17), which
        // does not arise on this single-state remove-then-push_back path; the
        // two methods intentionally differ on same-stash semantics (audit
        // angr-04tw3.4: confirmed intentional, not a missing guard).
        //
        // Move keeps the state's root; take_state_from removes it from the
        // source stash + unindexes, then we re-index onto the destination
        // (angr-ph300.27).
        if let Some(state) = self.sm.take_state_from(state_id, from_stash) {
            // Leaving STASH_ACTIVE outside `policy.select` — notify so a
            // memoizing policy doesn't leak a memo entry (angr-myzjx.25).
            if from_stash == STASH_ACTIVE {
                self.policy.on_state_removed(state_id);
            }
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
                "state {found_state_id} not found in 'found' stash"
            )));
        }

        // Clear every stash except 'active'. A fixed name list used to be
        // spelled out here, which silently omitted STASH_PRUNED and every
        // technique stash (cut/spinning/timeout/not_unique/_copies/
        // merge_waiting_*): those stage-1 states — and their Z3 solver
        // clones — stayed alive across all later stages (angr-ph300.26).
        let others: Vec<String> = self
            .sm
            .iter()
            .map(|(name, _)| name.clone())
            .filter(|name| name != STASH_ACTIVE)
            .collect();
        for stash in &others {
            self.sm.clear(stash);
        }

        // Remove all other active states (keep only the moved one). A bare
        // retain() drops the states without touching the bookkeeping maps,
        // so state_stash(dropped_id) would keep answering 'active' forever;
        // unindex + remove_root each dropped id, mirroring
        // drop_state_from_stash (angr-ph300.26).
        let dropped: Vec<u64> = match self.sm.get(STASH_ACTIVE) {
            Some(active) => active
                .iter()
                .map(|s| s.state_id())
                .filter(|id| *id != found_state_id)
                .collect(),
            None => Vec::new(),
        };
        if let Some(active) = self.sm.get_mut(STASH_ACTIVE) {
            active.retain(|s| s.state_id() == found_state_id);
        }
        for id in dropped {
            // Dropped from STASH_ACTIVE outside `policy.select` — notify so a
            // memoizing policy doesn't leak a memo entry (angr-myzjx.25).
            self.policy.on_state_removed(id);
            self.sm.unindex(id);
            self.sm.remove_root(id);
        }

        Ok(found_state_id)
    }
}

test_submod!("state_lifecycle_tests.rs" => tests);
