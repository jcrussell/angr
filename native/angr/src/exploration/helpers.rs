use super::*;

impl RustExplorationManager {
    /// Push a new state to the active stash, respecting max_active_states limit.
    /// If the limit is reached, the state is dropped (pruned) instead.
    /// Returns true if the state was added, false if dropped.
    #[inline]
    pub(crate) fn push_to_active_or_drop(&mut self, state: RustSimState) -> bool {
        if let Some(limit) = self.max_active_states {
            if self.sm.active_count() >= limit {
                log::debug!(
                    "max_active_states limit ({}) reached, dropping state {}",
                    limit,
                    state.state_id()
                );
                self.push_or_drop_terminal(STASH_PRUNED, state);
                return false;
            }
        }
        self.sm.stashes_mut()
            .entry(STASH_ACTIVE.to_string())
            .or_insert_with(VecDeque::new)
            .push_back(state);
        true
    }

    /// Track a state in the state_index.
    #[inline]
    pub(crate) fn index_state(&mut self, state_id: u64, stash: &str) {
        self.sm.index(state_id, stash);
    }

    /// Remove a state from the state_index.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn unindex_state(&mut self, state_id: u64) {
        self.sm.unindex(state_id);
    }

    /// Rebuild the state_index from scratch by scanning all stashes.
    /// Called after run() to ensure index is up to date for Python API calls.
    pub(crate) fn rebuild_state_index(&mut self) {
        self.sm.rebuild_index();
    }

    /// Find an immutable reference to a state by ID using the index.
    /// Falls back to linear scan if the index is stale.
    pub(crate) fn find_state(&self, state_id: u64) -> Option<&RustSimState> {
        // Check pending callback state first (during find_predicate evaluation)
        if let Some(ref pending) = self.pending_callback {
            if pending.state.state_id() == state_id {
                return Some(&pending.state);
            }
        }
        self.sm.find_state(state_id)
    }

    /// Find a mutable reference to a state by ID using the index.
    /// Falls back to linear scan if the index is stale.
    pub(crate) fn find_state_mut(&mut self, state_id: u64) -> Option<&mut RustSimState> {
        self.sm.find_state_mut(state_id)
    }

    /// Compute a hash for a state's register tuple for uniqueness checking.
    ///
    /// For each register in uniqueness_registers, gets the concrete value
    /// (or uses u64::MAX as sentinel for symbolic). Hashes the tuple.
    pub(crate) fn compute_register_tuple_hash(&self, state: &RustSimState) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        for reg_name in &self.uniqueness_registers {
            match state.get_register(reg_name) {
                Some(bv) => {
                    if let Some(val) = bv.as_u64() {
                        val.hash(&mut hasher);
                    } else {
                        // Symbolic register — use sentinel
                        u64::MAX.hash(&mut hasher);
                        // Also hash 1 to distinguish from concrete u64::MAX
                        1u8.hash(&mut hasher);
                    }
                }
                None => {
                    // Register not found — hash 0
                    0u64.hash(&mut hasher);
                }
            }
        }
        hasher.finish()
    }

    /// Apply the native uniqueness filter to the active stash.
    ///
    /// Moves states with duplicate register tuples to 'not_unique'.
    /// Called after each step in the run() loop.
    pub(crate) fn apply_uniqueness_filter(&mut self) {
        if self.uniqueness_registers.is_empty() {
            return;
        }

        // First pass: compute hashes and find duplicates (immutable borrow of active)
        let to_remove = {
            let active = match self.sm.get(STASH_ACTIVE) {
                Some(s) => s,
                None => return,
            };

            let mut remove_indices = Vec::new();
            for (i, state) in active.iter().enumerate() {
                let hash = self.compute_register_tuple_hash(state);
                if !self.uniqueness_set.insert(hash) {
                    remove_indices.push(i);
                }
            }
            remove_indices
        };

        if to_remove.is_empty() {
            return;
        }

        // Second pass: remove duplicates (mutable borrow of stashes)
        let active = match self.sm.get_mut(STASH_ACTIVE) {
            Some(s) => s,
            None => return,
        };
        let mut removed_states = Vec::new();
        for &idx in to_remove.iter().rev() {
            if let Some(state) = active.remove(idx) {
                removed_states.push(state);
            }
        }

        if !self.sm.drop_terminal_states() {
            let not_unique = self.sm.stashes_mut()
                .entry("not_unique".to_string())
                .or_insert_with(VecDeque::new);
            for state in removed_states {
                not_unique.push_back(state);
            }
        }
        // else dropped states are simply discarded
    }

    /// Push a state to a terminal stash (avoid/pruned/deadended), or drop it
    /// if `drop_terminal_states` is enabled. Increments the appropriate counter.
    pub(crate) fn push_or_drop_terminal(&mut self, stash_name: &str, state: RustSimState) {
        self.sm.push_or_drop_terminal(stash_name, state);
    }

    /// Sync constraints from Python callbacks back to the Rust state's solver.
    ///
    /// This is the critical piece for bidirectional constraint flow:
    /// - Rust syncs constraints TO Python before callbacks (via sync_before_callback)
    /// - Python SimProcedures may add new constraints (e.g., strcmp conditions)
    /// - This method syncs those new constraints BACK to Rust after the callback
    ///
    /// Without this, constraints added by SimProcedures would be lost when
    /// Rust resumes execution, leading to incorrect symbolic evaluation.
    ///
    /// Returns:
    ///   - Ok(true): Constraints synced and state is SAT (satisfiable)
    ///   - Ok(false): State became UNSAT after syncing - should be pruned (P12)
    ///   - Err: Python error during sync
    pub(crate) fn sync_constraints_from_python(
        &self,
        py: Python<'_>,
        state: &RustSimState,
        constraints: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<bool> {
        use pyo3::types::PyListMethods;

        let solver_ref = state.solver();
        let sym_ctx = solver_ref.borrow();
        let ctx_ref: &SymContext = &*sym_ctx;

        let mut success_count = 0usize;
        let mut failed_count = 0usize;

        // Extract list items - we need to convert each to RustBV
        let len = constraints.len();
        for i in 0..len {
            // Use get_item with usize index
            if let Ok(constraint) = constraints.get_item(i) {
                // Convert claripy AST to RustBV
                match claripy_to_rustbv(py, &constraint, ctx_ref) {
                    Ok(bv) => {
                        // Add constraint to solver
                        #[cfg(feature = "vex-engine-z3")]
                        {
                            if bv.width() == 1 {
                                sym_ctx.assume_true(&bv);
                                success_count += 1;
                            } else {
                                // For wider values, interpret as "value != 0"
                                let zero = RustBV::concrete(0, bv.width());
                                let neq = bv.ne(&zero, ctx_ref);
                                sym_ctx.assume_true(&neq);
                                success_count += 1;
                            }
                        }
                        #[cfg(not(feature = "vex-engine-z3"))]
                        {
                            // Without Z3, constraints are tracked but not solved
                            success_count += 1;
                        }
                    }
                    Err(e) => {
                        failed_count += 1;
                        log::warn!(
                            "Constraint {} conversion failed: {}. Solver state may diverge.",
                            i, e
                        );
                    }
                }
            }
        }

        if failed_count > 0 {
            log::warn!(
                "sync_constraints_from_python: {}/{} constraints failed to convert",
                failed_count, failed_count + success_count
            );
        }

        if success_count > 0 {
            log::debug!("Synced {} constraints from Python to Rust", success_count);
        }

        // P12: Check satisfiability and return status so callers can prune UNSAT states
        #[cfg(feature = "vex-engine-z3")]
        {
            let is_sat = sym_ctx.is_sat();
            if !is_sat {
                log::debug!(
                    "P12: Constraints are UNSAT after syncing {} from Python (failed={}). \
                     Returning false to trigger pruning.",
                    success_count, failed_count
                );
                return Ok(false);
            }
        }

        // P14: If many constraints failed to convert, do explicit SAT check
        // Failed conversions can leave state in divergent state
        #[cfg(feature = "vex-engine-z3")]
        if failed_count > 0 && success_count > 0 {
            let is_sat = sym_ctx.is_sat();
            if !is_sat {
                log::debug!(
                    "P14: State became UNSAT with partial constraint sync ({}/{} failed). Pruning.",
                    failed_count, failed_count + success_count
                );
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Extract procedure arguments from state registers.
    pub(crate) fn extract_procedure_args(&self, state: &RustSimState, num_args: usize) -> Vec<RustBV> {
        let arg_regs = self.calling_convention.arg_registers();
        let ptr_size = self.calling_convention.pointer_size();
        let mut args = Vec::with_capacity(num_args);

        let ctx = state.solver().borrow();

        // Extract from registers first
        for &offset in arg_regs.iter().take(num_args) {
            let value = state.get_register_by_offset(offset, ptr_size);
            args.push(value);
        }

        // If we need more args from stack, get them
        if args.len() < num_args {
            if let Some(sp) = state.get_sp().as_u64() {
                let stack_start = sp + self.calling_convention.stack_arg_offset();
                for i in 0..(num_args - args.len()) {
                    let addr = stack_start + (i as u64 * ptr_size as u64);
                    if let Ok(value) = state.memory_load(addr, ptr_size) {
                        args.push(value);
                    } else {
                        // Can't read stack - push zero
                        args.push(RustBV::zero(ptr_size * 8));
                    }
                }
            }
        }

        drop(ctx);
        args
    }

    /// Get return address from stack.
    pub(crate) fn get_return_addr(&self, state: &RustSimState) -> Option<u64> {
        let sp = state.get_sp().as_u64()?;
        let ptr_size = self.calling_convention.pointer_size();

        // On x86/AMD64, return address is at [rsp] after call
        state.memory_load(sp, ptr_size).ok()?.as_u64()
    }
}
