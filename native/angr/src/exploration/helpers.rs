use super::*;

impl RustExplorationManager {
    /// Run a closure with an immutable borrow of the pending callback state.
    /// Returns Err(PyRuntimeError) when no callback is pending.
    #[inline]
    pub(crate) fn with_pending<T, F>(&self, f: F) -> PyResult<T>
    where
        F: FnOnce(&PendingCallback) -> PyResult<T>,
    {
        let pending = self
            .pending_callback
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no pending callback state"))?;
        f(pending)
    }

    /// Run a closure with a mutable borrow of the pending callback state.
    /// Returns Err(PyRuntimeError) when no callback is pending.
    #[inline]
    pub(crate) fn with_pending_mut<T, F>(&mut self, f: F) -> PyResult<T>
    where
        F: FnOnce(&mut PendingCallback) -> PyResult<T>,
    {
        let pending = self
            .pending_callback
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("no pending callback state"))?;
        f(pending)
    }

    /// Run a closure with an immutable borrow of a state by ID.
    /// Looks up the pending callback first (matching `find_state` semantics),
    /// then the stashes. Returns Err(PyValueError) when not found.
    #[inline]
    pub(crate) fn with_state<T, F>(&self, state_id: u64, f: F) -> PyResult<T>
    where
        F: FnOnce(&RustSimState) -> PyResult<T>,
    {
        let state = self
            .find_state(state_id)
            .ok_or_else(|| PyValueError::new_err(format!("state {} not found", state_id)))?;
        f(state)
    }

    /// Run a closure with a mutable borrow of a state by ID.
    /// Note: unlike `with_state`, this does NOT check the pending callback —
    /// it follows the existing `find_state_mut` semantics (stashes only).
    /// Returns Err(PyValueError) when not found.
    #[inline]
    pub(crate) fn with_state_mut<T, F>(&mut self, state_id: u64, f: F) -> PyResult<T>
    where
        F: FnOnce(&mut RustSimState) -> PyResult<T>,
    {
        let state = self
            .find_state_mut(state_id)
            .ok_or_else(|| PyValueError::new_err(format!("state {} not found", state_id)))?;
        f(state)
    }

    /// Push a new state to the active stash, respecting `max_active_states`.
    /// If the limit is reached, the state is sent to the pruned stash (or
    /// dropped, if `drop_terminal_states` is enabled). Returns true if the
    /// state was added to the active stash, false if pruned.
    #[inline]
    pub(crate) fn push_to_active_or_drop(&mut self, state: RustSimState) -> bool {
        if let Some(limit) = self.max_active_states {
            if self.sm.active_count() >= limit {
                log::debug!(
                    "max_active_states limit ({}) reached, pruning state {}",
                    limit,
                    state.state_id()
                );
                self.push_or_drop_terminal(STASH_PRUNED, state);
                return false;
            }
        }
        self.sm.push(STASH_ACTIVE, state);
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
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for reg_name in &self.constraint_tracker.uniqueness_registers {
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
        if self.constraint_tracker.uniqueness_registers.is_empty() {
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
                if !self.constraint_tracker.uniqueness_set.insert(hash) {
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
            let not_unique = self
                .sm
                .stashes_mut()
                .entry("not_unique".to_string())
                .or_insert_with(VecDeque::new);
            for state in removed_states {
                not_unique.push_back(state);
            }
        }
        // else dropped states are simply discarded
    }

    /// Apply native exploration techniques to the active stash.
    ///
    /// Runs after each step in the exploration loop. Checks each technique
    /// and moves/stops states as needed, entirely in Rust.
    pub(crate) fn apply_native_techniques(&mut self) -> bool {
        if self.native_techniques.is_empty() {
            return false;
        }

        let mut complete = false;

        for tech_idx in 0..self.native_techniques.len() {
            match &mut self.native_techniques[tech_idx] {
                NativeTechnique::Timeout {
                    timeout_secs,
                    start_time,
                } => {
                    let start = start_time.get_or_insert_with(std::time::Instant::now);
                    if start.elapsed().as_secs_f64() > *timeout_secs {
                        log::info!(
                            "Native Timeout: exploration timed out after {:.1}s",
                            timeout_secs
                        );
                        // Move all active states to "timeout" stash
                        if let Some(active) = self.sm.get_mut(STASH_ACTIVE) {
                            let states: Vec<_> = active.drain(..).collect();
                            let timeout_stash = self
                                .sm
                                .stashes_mut()
                                .entry("timeout".to_string())
                                .or_insert_with(VecDeque::new);
                            for s in states {
                                timeout_stash.push_back(s);
                            }
                        }
                        complete = true;
                    }
                }
                NativeTechnique::LengthLimiter { max_length, drop } => {
                    let max_len = *max_length;
                    let do_drop = *drop;

                    // Find states exceeding the length limit
                    let to_remove = {
                        let active = match self.sm.get(STASH_ACTIVE) {
                            Some(s) => s,
                            None => continue,
                        };
                        let mut indices = Vec::new();
                        for (i, state) in active.iter().enumerate() {
                            if state.history().len() > max_len {
                                indices.push(i);
                            }
                        }
                        indices
                    };

                    if to_remove.is_empty() {
                        continue;
                    }

                    let active = match self.sm.get_mut(STASH_ACTIVE) {
                        Some(s) => s,
                        None => continue,
                    };
                    let mut removed_states = Vec::new();
                    for &idx in to_remove.iter().rev() {
                        if let Some(state) = active.remove(idx) {
                            removed_states.push(state);
                        }
                    }

                    if do_drop {
                        // States are simply discarded
                    } else {
                        let cut_stash = self
                            .sm
                            .stashes_mut()
                            .entry("cut".to_string())
                            .or_insert_with(VecDeque::new);
                        for state in removed_states {
                            cut_stash.push_back(state);
                        }
                    }
                }
                NativeTechnique::LoopBound {
                    bound,
                    discard_stash,
                } => {
                    let max_bound = *bound;
                    let stash_name = discard_stash.clone();

                    // Find states where any address appears more than `bound` times
                    let to_remove = {
                        let active = match self.sm.get(STASH_ACTIVE) {
                            Some(s) => s,
                            None => continue,
                        };
                        let mut indices = Vec::new();
                        for (i, state) in active.iter().enumerate() {
                            let history = state.history();
                            if Self::exceeds_loop_bound(history, max_bound) {
                                indices.push(i);
                            }
                        }
                        indices
                    };

                    if to_remove.is_empty() {
                        continue;
                    }

                    let active = match self.sm.get_mut(STASH_ACTIVE) {
                        Some(s) => s,
                        None => continue,
                    };
                    let mut removed_states = Vec::new();
                    for &idx in to_remove.iter().rev() {
                        if let Some(state) = active.remove(idx) {
                            removed_states.push(state);
                        }
                    }

                    if !self.sm.drop_terminal_states() {
                        let target = self
                            .sm
                            .stashes_mut()
                            .entry(stash_name)
                            .or_insert_with(VecDeque::new);
                        for state in removed_states {
                            target.push_back(state);
                        }
                    }
                }
            }
        }

        complete
    }

    /// Check if any address in the history exceeds the loop bound.
    fn exceeds_loop_bound(history: &[u64], bound: usize) -> bool {
        // Use a small HashMap to count address frequencies.
        // For typical histories this is fast since most addresses appear once.
        let mut counts: HashMap<u64, usize> = HashMap::new();
        for &addr in history {
            let count = counts.entry(addr).or_insert(0);
            *count += 1;
            if *count > bound {
                return true;
            }
        }
        false
    }

    /// Push a state to a terminal stash (avoid/pruned/deadended), or drop it
    /// if `drop_terminal_states` is enabled. Increments the appropriate counter.
    pub(crate) fn push_or_drop_terminal(&mut self, stash_name: &str, state: RustSimState) {
        self.sm.push_or_drop_terminal(stash_name, state);
    }

    /// Sync constraints from Python callbacks back to the Rust state's solver.
    ///
    /// Python SimProcedures may add new constraints (e.g., strcmp conditions);
    /// this method syncs those new constraints BACK to Rust after the callback.
    /// The reverse direction (Rust→Python) is handled by attaching a
    /// `RustSolverContext` to the Python state in
    /// `rust_callback_dispatch._install_rust_solver_on_callback_state`, so
    /// constraints never need to be replayed across the FFI.
    ///
    /// Without this, constraints added by SimProcedures would be lost when
    /// Rust resumes execution, leading to incorrect symbolic evaluation.
    ///
    /// Conversion strategy: claripy_to_rustbv first (preserves assumed_constraints
    /// tracking that Python re-import depends on), then fall back to lossless Z3
    /// pointer extraction when claripy_to_rustbv hits an unsupported op (FP, etc).
    /// Both paths share the same Z3 context with claripy via the shared backend.
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
        let mut z3_ptr_fallback_count = 0usize;
        let mut failed_count = 0usize;

        // Lazily resolve claripy.backends.z3 once for the Z3 ptr fallback path.
        // None on first failure so we don't keep paying the import cost.
        #[cfg(feature = "vex-engine-z3")]
        let mut z3_backend: Option<Bound<'_, PyAny>> = None;

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
                        // Fallback: extract the constraint's underlying Z3 AST
                        // pointer via claripy.backends.z3 and assert it on the
                        // solver directly. claripy and the Rust solver share a
                        // Z3 context, so the pointer is valid here. This rescues
                        // constraints that use ops claripy_to_rustbv doesn't
                        // model (e.g. FP), keeping the round-trip lossless.
                        #[cfg(feature = "vex-engine-z3")]
                        {
                            let backend = match z3_backend.as_ref() {
                                Some(b) => Some(b.clone()),
                                None => match Self::resolve_claripy_z3_backend(py) {
                                    Ok(b) => {
                                        z3_backend = Some(b.clone());
                                        Some(b)
                                    }
                                    Err(import_err) => {
                                        log::debug!(
                                            "claripy.backends.z3 unavailable for fallback: {}",
                                            import_err
                                        );
                                        None
                                    }
                                },
                            };

                            let z3_ast = backend.as_ref().and_then(|b| {
                                Self::extract_z3_ptr_from_claripy(b, &constraint).ok().flatten()
                            });

                            if let Some(ast) = z3_ast {
                                sym_ctx.add_constraint_raw(ast);
                                z3_ptr_fallback_count += 1;
                                success_count += 1;
                                log::debug!(
                                    "Constraint {} fell back to Z3 ptr (claripy_to_rustbv: {})",
                                    i,
                                    e
                                );
                                continue;
                            }
                        }

                        failed_count += 1;
                        log::warn!(
                            "Constraint {} conversion failed: {}. Solver state may diverge.",
                            i,
                            e
                        );
                    }
                }
            }
        }

        if z3_ptr_fallback_count > 0 {
            log::debug!(
                "sync_constraints_from_python: {} constraints rescued via Z3 ptr fallback",
                z3_ptr_fallback_count
            );
        }

        if failed_count > 0 {
            log::warn!(
                "sync_constraints_from_python: {}/{} constraints failed to convert",
                failed_count,
                failed_count + success_count
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
                    success_count,
                    failed_count
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
                    failed_count,
                    failed_count + success_count
                );
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Resolve `claripy.backends.z3` once per call to `sync_constraints_from_python`.
    /// Returned as a `Bound<PyAny>` because that's what `convert(...)` needs.
    #[cfg(feature = "vex-engine-z3")]
    fn resolve_claripy_z3_backend(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        let claripy = py.import("claripy")?;
        let backends = claripy.getattr("backends")?;
        backends.getattr("z3")
    }

    /// Extract a typed [`Z3AstPtr`] from a claripy AST by going through
    /// `claripy.backends.z3.convert(ast).as_ast().value`. Returns `Ok(None)`
    /// if the conversion succeeded but the extracted pointer is null;
    /// returns `Err` if the Python conversion path itself failed.
    ///
    /// The returned handle has its own `Z3_inc_ref` ref; callers can drop
    /// it without affecting claripy's cached AST.
    #[cfg(feature = "vex-engine-z3")]
    fn extract_z3_ptr_from_claripy(
        z3_backend: &Bound<'_, PyAny>,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Z3AstPtr>> {
        let z3_obj = z3_backend.call_method1("convert", (ast,))?;
        let raw_ast = z3_obj.call_method0("as_ast")?;
        let ptr = raw_ast.getattr("value")?.extract::<usize>()?;
        let ctx = z3::Context::thread_local();
        // SAFETY: claripy's z3 backend returned this pointer for a live
        // AST it holds in its own cache; the AST is in the process-global
        // Z3 context, which matches our thread-local context.
        Ok(unsafe { Z3AstPtr::from_borrowed_raw(&ctx, ptr) })
    }

    /// Extract procedure arguments from state registers (and stack, when
    /// `num_args` exceeds the register portion of the calling convention).
    ///
    /// Returns an [`ExtractionError`] instead of silently zero-padding when
    /// the stack pointer is symbolic or a stack slot cannot be read. Callers
    /// (the native-procedure dispatchers in `stepping.rs` / `run_loop.rs`)
    /// treat any error as a signal to skip the native fast path and fall
    /// through to the Python SimProcedure callback rather than handing the
    /// handler a fabricated `RustBV::zero` that would silently mask a real
    /// stack-setup bug (mirror of the `angr-ydli` fix to the trait method).
    pub(crate) fn extract_procedure_args(
        &self,
        state: &RustSimState,
        num_args: usize,
    ) -> Result<Vec<RustBV>, ExtractionError> {
        let arg_regs = self.environment.calling_convention.arg_registers();
        let ptr_size = self.environment.calling_convention.pointer_size();
        let mut args = Vec::with_capacity(num_args);

        let ctx = state.solver().borrow();

        for &offset in arg_regs.iter().take(num_args) {
            args.push(state.get_register_by_offset(offset, ptr_size));
        }

        if args.len() < num_args {
            let sp = state.get_sp().as_u64().ok_or(ExtractionError::SpSymbolic)?;
            let stack_start = sp + self.environment.calling_convention.stack_arg_offset();
            let already = args.len();
            for i in 0..(num_args - already) {
                let addr = stack_start + (i as u64 * ptr_size as u64);
                let value = state.memory_load(addr, ptr_size).map_err(|_| {
                    ExtractionError::StackUnmapped {
                        arg_index: already + i,
                        addr,
                    }
                })?;
                args.push(value);
            }
        }

        drop(ctx);
        Ok(args)
    }

    /// Extract syscall arguments from state registers.
    ///
    /// Uses the calling convention's `syscall_arg_registers()` rather than
    /// `arg_registers()`. On Linux amd64 these differ at the 4th argument
    /// (R10 vs RCX). Syscalls do not pull args from the stack: a request for
    /// more arguments than the ABI exposes via registers returns
    /// [`ExtractionError::RegisterOverflow`] so callers can fall through to
    /// the Python syscall callback instead of running a native handler with
    /// fabricated zero arguments.
    pub(crate) fn extract_syscall_args(
        &self,
        state: &RustSimState,
        num_args: usize,
    ) -> Result<Vec<RustBV>, ExtractionError> {
        let arg_regs = self.environment.calling_convention.syscall_arg_registers();
        if num_args > arg_regs.len() {
            return Err(ExtractionError::RegisterOverflow {
                requested: num_args,
                available: arg_regs.len(),
            });
        }
        let ptr_size = self.environment.calling_convention.pointer_size();
        let mut args = Vec::with_capacity(num_args);
        for &offset in arg_regs.iter().take(num_args) {
            args.push(state.get_register_by_offset(offset, ptr_size));
        }
        Ok(args)
    }

    /// Get return address from stack.
    pub(crate) fn get_return_addr(&self, state: &RustSimState) -> Option<u64> {
        let sp = state.get_sp().as_u64()?;
        let ptr_size = self.environment.calling_convention.pointer_size();

        // On x86/AMD64, return address is at [rsp] after call
        state.memory_load(sp, ptr_size).ok()?.as_u64()
    }
}
