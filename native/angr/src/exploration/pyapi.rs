// This file is included into mod.rs inside a #[pymethods] impl block.
// Do not add `use`, `impl`, or `#[pymethods]` wrappers here.

    /// Get the architecture name.
    #[getter]
    pub fn arch(&self) -> &str {
        &self.arch_name
    }

    /// Get the total number of steps executed.
    #[getter]
    pub fn step_count(&self) -> u64 {
        self.steps
    }

    /// Get active state count.
    pub fn active_count(&self) -> usize {
        self.sm.get(STASH_ACTIVE).map(|s| s.len()).unwrap_or(0)
    }

    /// Get found state count.
    pub fn found_count(&self) -> usize {
        self.sm.stashes().get(STASH_FOUND).map(|s| s.len()).unwrap_or(0)
    }

    /// Get stash counts as a dictionary.
    pub fn stash_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (name, stash) in self.sm.stashes() {
            dict.set_item(name, stash.len())?;
        }
        Ok(dict)
    }

    /// Set find addresses.
    pub fn set_find_addrs(&mut self, addrs: Vec<u64>) {
        self.find_addrs = addrs.into_iter().collect();
        self.find_needs_python = false;
    }

    /// Set avoid addresses.
    pub fn set_avoid_addrs(&mut self, addrs: Vec<u64>) {
        self.avoid_addrs = addrs.into_iter().collect();
        self.avoid_needs_python = false;
    }

    /// Mark that find condition has callable predicates (needs Python).
    pub fn set_find_needs_python(&mut self, needs: bool) {
        self.find_needs_python = needs;
    }

    /// Mark that avoid condition has callable predicates (needs Python).
    pub fn set_avoid_needs_python(&mut self, needs: bool) {
        self.avoid_needs_python = needs;
    }

    /// P9 fix: Set state selection to LIFO (DFS - depth-first search).
    pub fn set_state_selection_lifo(&mut self) {
        self.use_lifo = true;
        log::debug!("State selection set to LIFO (DFS)");
    }

    /// P9 fix: Set state selection to FIFO (BFS - breadth-first search).
    pub fn set_state_selection_fifo(&mut self) {
        self.use_lifo = false;
        log::debug!("State selection set to FIFO (BFS)");
    }

    /// Set the number of solutions to find before stopping.
    pub fn set_num_find(&mut self, n: usize) {
        self.num_find = n;
    }

    /// Set maximum steps per run iteration.
    pub fn set_max_steps_per_run(&mut self, n: u32) {
        self.max_steps_per_run = n;
    }

    /// Enable lazy solves mode (skip satisfiability checks on forks).
    pub fn set_lazy_solves(&mut self, enabled: bool) {
        self.lazy_solves = enabled;
    }

    /// Enable zero-fill for unconstrained memory reads.
    /// When true, unmapped memory returns zero instead of fresh symbolic values.
    pub fn set_zero_fill_unconstrained(&mut self, enabled: bool) {
        self.zero_fill_unconstrained = enabled;
    }

    /// Set the Z3 solver timeout in milliseconds (default: 30000).
    pub fn set_solver_timeout(&mut self, timeout_ms: u32) {
        self.solver_timeout_ms = timeout_ms;
    }

    /// Set whether to drop terminal states (avoid/pruned/deadended) immediately.
    /// When true (default), terminal states are dropped to save memory.
    /// Set to false when states need to be recovered (e.g., factory.callable()).
    pub fn set_drop_terminal_states(&mut self, enabled: bool) {
        self.sm.set_drop_terminal_states(enabled);
    }

    /// Enable or disable Rust-side profiling.
    /// When enabled, per-step timing and counters are accumulated.
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling_enabled = enabled;
    }

    /// Get accumulated execution statistics as a dict.
    pub fn get_execution_stats(&self) -> HashMap<String, u64> {
        self.accumulated_stats.to_hashmap()
    }

    /// Reset accumulated execution statistics.
    pub fn reset_execution_stats(&mut self) {
        self.accumulated_stats.reset();
    }

    /// Set Python callbacks for memory/lifting.
    pub fn set_callbacks(&mut self, callbacks: PythonCallbacks) {
        self.callbacks = Some(callbacks);
    }

    /// Clear callbacks.
    pub fn clear_callbacks(&mut self) {
        self.callbacks = None;
    }

    /// Add a hook address.
    pub fn add_hook(&mut self, addr: u64) {
        self.hooks.insert(addr);
    }

    /// Add multiple hook addresses.
    pub fn add_hooks(&mut self, addrs: Vec<u64>) {
        for addr in addrs {
            self.hooks.insert(addr);
        }
    }

    /// Clear all hooks.
    pub fn clear_hooks(&mut self) {
        self.hooks.clear();
    }

    /// Register a SimProcedure.
    #[pyo3(signature = (addr, name, num_args=0, no_return=false))]
    pub fn register_simprocedure(
        &mut self,
        addr: u64,
        name: String,
        num_args: usize,
        no_return: bool,
    ) {
        self.hooks.insert(addr);
        self.simprocedures.insert(addr, (name, num_args, no_return));
    }

    /// Register multiple SimProcedures.
    pub fn register_simprocedures(&mut self, procs: Vec<(u64, String, usize, bool)>) {
        for (addr, name, num_args, no_return) in procs {
            self.hooks.insert(addr);
            self.simprocedures.insert(addr, (name, num_args, no_return));
        }
    }

    /// Load binary code regions.
    pub fn load_binary_regions(&mut self, regions: Vec<(u64, Vec<u8>)>) {
        self.binary_regions = regions.into_iter()
            .map(|(base, data)| (base, Arc::new(data)))
            .collect();
    }

    /// Create a new RustSimState and add it to a stash.
    #[pyo3(signature = (stash="active"))]
    pub fn create_state(&mut self, stash: &str) -> PyResult<u64> {
        let mut state = RustSimState::new_with_endian(&self.arch_name, self.little_endian)
            .map_err(|e| PyValueError::new_err(e))?;
        let state_id = state.state_id();

        // Propagate memory options
        if self.zero_fill_unconstrained {
            state.memory_mut().set_zero_fill_unconstrained(true);
        }

        // Copy hooks to state
        for &addr in &self.hooks {
            // State hooks are checked during execution
        }

        self.index_state(state_id, stash);
        self.sm.stashes_mut()
            .entry(stash.to_string())
            .or_insert_with(VecDeque::new)
            .push_back(state);

        Ok(state_id)
    }

    /// Add an existing RustSimState to a stash.
    #[pyo3(signature = (stash, state))]
    pub fn add_state(&mut self, stash: &str, state: &crate::state::PyRustSimState) {
        // Fork the state to get our own copy
        let mut forked = state.inner().fork();
        let state_id = forked.state_id();

        // Propagate memory options
        if self.zero_fill_unconstrained {
            forked.memory_mut().set_zero_fill_unconstrained(true);
        }

        // Track this state as its own root (it was added via Python)
        self.sm.set_root(state_id, state_id);

        self.index_state(state_id, stash);
        self.sm.stashes_mut()
            .entry(stash.to_string())
            .or_insert_with(VecDeque::new)
            .push_back(forked);
    }

    /// Get the PC of a state in a stash by index.
    #[pyo3(signature = (stash="active", index=0))]
    pub fn get_state_pc(&self, stash: &str, index: usize) -> Option<u64> {
        self.sm.get(stash).and_then(|s| s.get(index)).map(|s| s.pc())
    }

    /// Get the PC of a state by its ID (O(1) via state index, no full export).
    pub fn get_state_pc_by_id(&self, state_id: u64) -> Option<u64> {
        if let Some(state) = self.find_state(state_id) {
            return Some(state.pc());
        }
        if let Some(ref cb) = self.pending_callback {
            if cb.state.state_id() == state_id {
                return Some(cb.state.pc());
            }
        }
        None
    }

    /// Get state IDs in a stash.
    #[pyo3(signature = (stash="active"))]
    pub fn get_state_ids(&self, stash: &str) -> Vec<u64> {
        self.sm.get(stash)
            .map(|s| s.iter().map(|state| state.state_id()).collect())
            .unwrap_or_default()
    }

    /// Check if there are any active states (O(1), no allocation).
    pub fn has_active_states(&self) -> bool {
        self.sm.get(STASH_ACTIVE).map_or(false, |s| !s.is_empty())
    }

    /// Get the number of states in a stash (O(1), no allocation).
    #[pyo3(signature = (stash="active"))]
    pub fn stash_count(&self, stash: &str) -> usize {
        self.sm.get(stash).map_or(0, |s| s.len())
    }

    /// Rebuild the state index after run() modifies stashes internally.
    /// Call from Python after run() returns to keep index up to date.
    pub fn sync_state_index(&mut self) {
        self.rebuild_state_index();
    }

    /// Get the root state ID for any state.
    /// Returns the original (initial) state from which this state was forked.
    pub fn get_state_root(&self, state_id: u64) -> Option<u64> {
        self.sm.roots().get(&state_id).copied()
    }

    /// Set the PC of the pending callback state (for external initialization).
    pub fn set_pending_state_pc(&mut self, pc: u64) -> PyResult<()> {
        if let Some(ref mut pending) = self.pending_callback {
            pending.state.set_pc(pc);
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Map memory in the pending state.
    #[pyo3(signature = (addr, data, permissions=7))]
    pub fn pending_state_map_memory(
        &mut self,
        addr: u64,
        data: &[u8],
        permissions: u8,
    ) -> PyResult<()> {
        if let Some(ref mut pending) = self.pending_callback {
            pending.state.map_memory_data(addr, data, Permission::from_bits(permissions));
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Map memory in active states.
    #[pyo3(signature = (addr, data, permissions=7))]
    pub fn active_states_map_memory(&mut self, addr: u64, data: &[u8], permissions: u8) {
        if let Some(stash) = self.sm.get_mut(STASH_ACTIVE) {
            for state in stash.iter_mut() {
                state.map_memory_data(addr, data, Permission::from_bits(permissions));
            }
        }
    }

    /// Get the branch condition from the pending symbolic branch callback.
    ///
    /// Returns the condition as a claripy AST that Python can use for forking.
    pub fn get_pending_branch_condition(&self, py: Python<'_>) -> PyResult<PyObject> {
        let pending = self.pending_callback.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no pending callback state"))?;

        // Get the condition ID from the callback reason
        let condition_id = match &pending.reason {
            CallbackReason::SymbolicBranch { condition_id, .. } => *condition_id,
            _ => return Err(PyValueError::new_err("pending callback is not a symbolic branch")),
        };

        // Look up the condition in stored_conditions
        let condition = pending.stored_conditions.get(&condition_id)
            .ok_or_else(|| PyValueError::new_err(
                format!("condition {} not found in stored_conditions", condition_id)
            ))?;

        // Convert to claripy AST
        let claripy = py.import("claripy")?;
        rustbv_to_claripy(py, condition, claripy.as_any())
            .map_err(|e| PyRuntimeError::new_err(format!("failed to convert condition: {}", e)))
    }

    /// Get register value from pending state (concrete only).
    pub fn get_pending_register(&self, name: &str) -> PyResult<Option<u128>> {
        if let Some(ref pending) = self.pending_callback {
            pending.state.get_register(name)
                .map(|bv| bv.as_u128())
                .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Get register as claripy AST from pending state (handles symbolic).
    pub fn get_pending_register_ast(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        let pending = self.pending_callback.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no pending callback state"))?;
        let bv = pending.state.get_register(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))?;
        let claripy = py.import("claripy")?;
        rustbv_to_claripy(py, &bv, claripy.as_any())
            .map_err(|e| PyRuntimeError::new_err(format!("register conversion: {}", e)))
    }

    /// Get history (BBL addresses) from pending callback state.
    ///
    /// This is used by Python to initialize history on callback states,
    /// preventing IndexError when hooks access `state.history.recent_bbl_addrs[-1]`.
    pub fn get_pending_history(&self) -> PyResult<Vec<u64>> {
        if let Some(ref pending) = self.pending_callback {
            Ok(pending.state.history().to_vec())
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Get jumpkind for pending callback.
    ///
    /// Returns the jumpkind that led to this callback (e.g., "Ijk_Call", "Ijk_Boring").
    /// This is used by Python to properly initialize callstack management.
    pub fn get_pending_jumpkind(&self) -> PyResult<String> {
        if let Some(ref pending) = self.pending_callback {
            Ok(pending.jumpkind.clone().unwrap_or_else(|| "Ijk_Boring".to_string()))
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Set register value in pending state.
    pub fn set_pending_register(&mut self, name: &str, value: u128) -> PyResult<()> {
        if let Some(ref mut pending) = self.pending_callback {
            let size = pending.state.arch().register_size(name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))?;
            let bv = crate::symbolic::RustBV::concrete(value, size * 8);
            if pending.state.set_register(name, bv) {
                Ok(())
            } else {
                Err(PyValueError::new_err(format!("failed to set register: {}", name)))
            }
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Set register to a symbolic value from a handle ID.
    ///
    /// Used for syncing symbolic return values from SimProcedures.
    /// The handle_id should reference a RustBV in the solver's symbol table.
    pub fn set_pending_register_symbolic(&mut self, name: &str, handle_id: u64) -> PyResult<()> {
        if let Some(ref mut pending) = self.pending_callback {
            // Look up the RustBV from the symbol table
            let bv = if let Some(ref solver) = pending.solver_ctx {
                solver.symbol_table().get(handle_id)
                    .ok_or_else(|| PyValueError::new_err(format!(
                        "invalid handle id: {}", handle_id
                    )))?
            } else {
                return Err(PyRuntimeError::new_err("no solver context in pending state"));
            };

            if pending.state.set_register(name, bv) {
                Ok(())
            } else {
                Err(PyValueError::new_err(format!("failed to set register: {}", name)))
            }
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Set a symbolic register value in the pending state from claripy AST.
    ///
    /// This allows direct sync of symbolic register values from Python callbacks.
    /// The claripy AST is converted to RustBV and stored in the pending state.
    pub fn set_pending_register_symbolic_ast(
        &mut self,
        py: Python<'_>,
        reg_name: &str,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if let Some(ref mut pending) = self.pending_callback {
            let solver_ref = pending.state.solver();
            let sym_ctx = solver_ref.borrow();
            let ctx_ref: &SymContext = &*sym_ctx;

            // Convert claripy AST to RustBV
            let bv = claripy_to_rustbv(py, ast, ctx_ref)
                .map_err(|e| PyValueError::new_err(format!("AST conversion failed: {}", e)))?;

            drop(sym_ctx);

            if pending.state.set_register(reg_name, bv) {
                log::debug!("Set symbolic register {} from claripy AST", reg_name);
                Ok(())
            } else {
                Err(PyValueError::new_err(format!("failed to set register: {}", reg_name)))
            }
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Import symbolic memory from Python hook into Rust's symbolic_objects.
    ///
    /// Called after a hook writes symbolic memory. Converts the claripy AST
    /// to RustBV and imports it into the pending state's SymbolicMemory.
    /// Import symbolic memory into a state by ID (for init-time symbolic data).
    #[pyo3(signature = (state_id, addr, ast))]
    pub fn import_symbolic_to_state(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        addr: u64,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if let Some(state) = self.find_state_mut(state_id) {
            let solver_ref = state.solver();
            let sym_ctx = solver_ref.borrow();
            let bv = claripy_to_rustbv(py, ast, &*sym_ctx)
                .map_err(|e| PyValueError::new_err(format!("AST conversion: {}", e)))?;
            drop(sym_ctx);
            state.memory_mut().import_symbolic_value(addr, bv, None);
            return Ok(());
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    pub fn import_symbolic_memory(
        &mut self,
        py: Python<'_>,
        addr: u64,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let pending = self.pending_callback.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("no pending callback state")
        })?;

        // Convert claripy AST to RustBV (caches original AST for round-trip)
        let solver_ref = pending.state.solver();
        let sym_ctx = solver_ref.borrow();
        let bv = claripy_to_rustbv(py, ast, &*sym_ctx)
            .map_err(|e| PyValueError::new_err(format!("AST conversion failed: {}", e)))?;
        drop(sym_ctx);

        // Import into symbolic memory via existing infrastructure
        // Symbol ID is not used by import_symbolic_value, so pass None
        pending.state.memory_mut().import_symbolic_value(addr, bv, None);
        log::debug!("Imported symbolic memory at 0x{:x}", addr);
        Ok(())
    }

    /// Get memory from pending state.
    pub fn get_pending_memory(&self, addr: u64, size: u32) -> PyResult<Vec<u8>> {
        if let Some(ref pending) = self.pending_callback {
            let bv = pending.state.memory_load(addr, size)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            let value = bv.to_u128();
            let bytes: Vec<u8> = (0..size as usize)
                .map(|i| (value >> (i * 8)) as u8)
                .collect();
            Ok(bytes)
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Store memory in pending state.
    pub fn set_pending_memory(&mut self, addr: u64, data: &[u8]) -> PyResult<()> {
        if let Some(ref mut pending) = self.pending_callback {
            let width = (data.len() * 8) as u32;
            let mut value: u128 = 0;
            for (i, &b) in data.iter().enumerate() {
                value |= (b as u128) << (i * 8);
            }
            let bv = crate::symbolic::RustBV::concrete(value, width);
            pending.state.memory_store(addr, bv)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Get dirty page addresses from pending state.
    ///
    /// This returns the list of page-aligned addresses that have been
    /// modified in the pending callback state.
    pub fn get_pending_dirty_pages(&self) -> PyResult<Vec<u64>> {
        if let Some(ref pending) = self.pending_callback {
            Ok(pending.state.get_dirty_pages())
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Get dirty register offsets from pending state.
    pub fn get_pending_dirty_registers(&self) -> PyResult<Vec<u32>> {
        if let Some(ref pending) = self.pending_callback {
            Ok(pending.state.get_dirty_registers())
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Clear dirty tracking in pending state.
    pub fn clear_pending_dirty_tracking(&mut self) -> PyResult<()> {
        if let Some(ref mut pending) = self.pending_callback {
            pending.state.clear_dirty_pages();
            pending.state.clear_dirty_registers();
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Export pending constraints as a list of claripy ASTs.
    ///
    /// Returns constraints that can be added to Python state.solver.
    /// This exports stored branch conditions accumulated during Rust execution.
    pub fn export_pending_constraints(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        if let Some(ref pending) = self.pending_callback {
            let mut result = Vec::new();

            // Import claripy for AST conversion
            let claripy_mod = py.import("claripy")?;

            // Export stored branch conditions as claripy ASTs
            for (_condition_id, rustbv) in &pending.stored_conditions {
                match rustbv_to_claripy(py, rustbv, &claripy_mod) {
                    Ok(ast) => {
                        result.push(ast);
                    }
                    Err(e) => {
                        log::debug!("Could not convert stored condition to claripy: {}", e);
                    }
                }
            }

            log::debug!("Exported {} pending constraints", result.len());
            Ok(result)
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Get handle IDs that are actively referenced in the pending state.
    ///
    /// Returns handle IDs used in stored conditions and deferred forks.
    /// These should not be evicted from the AST handle cache.
    pub fn get_active_handle_ids(&self) -> Vec<u64> {
        let mut ids = Vec::new();
        if let Some(ref pending) = self.pending_callback {
            // Add condition IDs from stored_conditions
            for (id, _) in &pending.stored_conditions {
                ids.push(*id);
            }
            // Add condition IDs from deferred forks
            for fork in &pending.deferred_forks {
                ids.push(fork.condition_id);
            }
        }
        ids
    }

    /// Export the pending state as a full snapshot.
    ///
    /// This allows Python to get a complete snapshot of the pending state
    /// including all registers, memory pages, and metadata.
    pub fn export_pending_state(&self) -> PyResult<crate::state::ExplorationStateSnapshot> {
        if let Some(ref pending) = self.pending_callback {
            Ok(pending.state.export_full())
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Get the root state ID for the pending callback state.
    ///
    /// When Rust forks states internally, Python only has cached data for the
    /// original state that was added via Python. This method returns the root
    /// state ID (the original state) for any forked descendant.
    ///
    /// Returns:
    ///     The root state ID if available, or None if the state has no tracked root.
    pub fn get_pending_root_state_id(&self) -> PyResult<Option<u64>> {
        if let Some(ref pending) = self.pending_callback {
            let state_id = pending.state.state_id();
            Ok(self.sm.roots().get(&state_id).copied())
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Get the full ancestry chain for the pending callback state.
    ///
    /// Returns a list of state IDs starting with the current state and walking
    /// up the parent chain: [state_id, parent_id, grandparent_id, ...].
    ///
    /// This is used by Python to find cached state data when the current state
    /// is a multi-level fork of an original state.
    pub fn get_pending_ancestry(&self) -> PyResult<Vec<u64>> {
        if let Some(ref pending) = self.pending_callback {
            let mut ancestry = vec![pending.state.state_id()];

            // Walk the parent chain
            let mut current_parent = pending.state.parent_id();
            while let Some(parent_id) = current_parent {
                ancestry.push(parent_id);
                // We can't traverse further without access to parent state objects,
                // but we can include the root state if known
                break;
            }

            // Add root state if not already in ancestry
            let state_id = pending.state.state_id();
            if let Some(&root_id) = self.sm.roots().get(&state_id) {
                if !ancestry.contains(&root_id) {
                    ancestry.push(root_id);
                }
            }

            Ok(ancestry)
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
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
    #[pyo3(signature = (register_names, shared_solver=true))]
    pub fn export_callback_bundle<'py>(
        &self,
        py: Python<'py>,
        register_names: Vec<String>,
        shared_solver: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let pending = self.pending_callback.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no pending callback state"))?;

        let dict = PyDict::new(py);

        // Registers: batch export all requested registers
        let reg_dict = PyDict::new(py);
        for name in &register_names {
            match pending.state.get_register(name) {
                Some(bv) => {
                    if let Some(val) = bv.as_u128() {
                        reg_dict.set_item(name, val)?;
                    } else {
                        // Symbolic — set to None, Python will fetch AST if needed
                        reg_dict.set_item(name, py.None())?;
                    }
                }
                None => {
                    reg_dict.set_item(name, py.None())?;
                }
            }
        }
        dict.set_item("registers", reg_dict)?;

        // Solver context: shared (O(1) Rc clone) or forked (~3ms Z3 clone)
        let solver_ref = pending.state.solver();
        if shared_solver {
            let rust_ctx = RustSolverContext::from_shared_sym_context(solver_ref.clone());
            let constraint_count = rust_ctx.num_constraints();
            dict.set_item("solver", Py::new(py, rust_ctx)?)?;
            dict.set_item("constraint_count", constraint_count)?;
        } else {
            let forked_ctx = solver_ref.borrow().fork();
            let rust_ctx = RustSolverContext::from_sym_context(forked_ctx);
            let constraint_count = rust_ctx.num_constraints();
            dict.set_item("solver", Py::new(py, rust_ctx)?)?;
            dict.set_item("constraint_count", constraint_count)?;
        }

        // History
        dict.set_item("history", pending.state.history().to_vec())?;

        // Jumpkind
        dict.set_item("jumpkind",
            pending.jumpkind.clone().unwrap_or_else(|| "Ijk_Boring".to_string()))?;

        // Stdout buffer
        dict.set_item("stdout", pending.state.stdout_buffer().to_vec())?;

        Ok(dict)
    }

    pub fn fork_pending_solver(&self) -> PyResult<RustSolverContext> {
        if let Some(ref pending) = self.pending_callback {
            // Fork the pending state's solver context
            let solver_ref = pending.state.solver();
            let forked_ctx = solver_ref.borrow().fork();
            // Create a new RustSolverContext wrapping the forked SymContext
            Ok(RustSolverContext::from_sym_context(forked_ctx))
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
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
    pub fn borrow_pending_solver(&self) -> PyResult<RustSolverContext> {
        if let Some(ref pending) = self.pending_callback {
            let solver_rc = pending.state.solver().clone();
            Ok(RustSolverContext::from_shared_sym_context(solver_rc))
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Add constraints from Python callbacks back to the pending state.
    ///
    /// This is called after a SimProcedure executes to sync any new
    /// constraints added during the callback back to the Rust solver.
    /// This ensures bidirectional constraint flow between Rust and Python.
    ///
    /// Args:
    ///     constraints: List of claripy AST constraints to add
    pub fn add_constraints_to_pending(
        &mut self,
        py: Python<'_>,
        constraints: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<()> {
        use pyo3::types::PyListMethods;

        if let Some(ref mut pending) = self.pending_callback {
            let solver_ref = pending.state.solver();
            let sym_ctx = solver_ref.borrow();
            let ctx_ref: &SymContext = &*sym_ctx;

            let len = constraints.len();
            for i in 0..len {
                // Use get_item with usize index
                if let Ok(constraint) = constraints.get_item(i) {
                    // Convert claripy AST to RustBV
                    if let Ok(bv) = claripy_to_rustbv(py, &constraint, ctx_ref) {
                        // Add constraint to solver
                        #[cfg(feature = "vex-engine-z3")]
                        {
                            if bv.width() == 1 {
                                sym_ctx.assume_true(&bv);
                            } else {
                                // For wider values, interpret as "value != 0"
                                let zero = RustBV::concrete(0, bv.width());
                                let neq = bv.ne(&zero, ctx_ref);
                                sym_ctx.assume_true(&neq);
                            }
                        }
                    } else {
                        log::debug!("Could not convert constraint {} from Python", i);
                    }
                }
            }
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Add constraints from Python to a state in a stash by state ID.
    /// This is used to sync initial constraints from the Python state.
    pub fn add_constraints_to_state(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        constraints: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<bool> {
        if let Some(state) = self.find_state_mut(state_id) {
            let solver_ref = state.solver();
            let sym_ctx = solver_ref.borrow();
            let ctx_ref: &SymContext = &*sym_ctx;

            // Pre-fetch Z3 backend for fast path
            #[cfg(feature = "vex-engine-z3")]
            let z3_backend = py.import("claripy")
                .and_then(|c| c.getattr("backends"))
                .and_then(|b| b.getattr("z3"))
                .ok();

            let mut added = 0u32;
            for item in constraints.iter() {
                // Fast path: extract raw Z3 AST and assert directly.
                // This preserves the exact Z3 structure from the previous
                // exploration, avoiding lossy RustBV round-trips that can
                // drop constraints (critical for multi-stage explore).
                #[cfg(feature = "vex-engine-z3")]
                {
                    if let Some(ref backend) = z3_backend {
                        if let Ok(z3_obj) = backend.call_method1("convert", (&item,)) {
                            if let Ok(ast_ref) = z3_obj.call_method0("as_ast") {
                                if let Ok(ptr) = ast_ref.getattr("value").and_then(|v| v.extract::<usize>()) {
                                    if ptr != 0 {
                                        unsafe { ctx_ref.add_constraint_raw(ptr); }
                                        // Track in assumed_constraints for re-export
                                        if let Ok(bv) = claripy_to_rustbv(py, &item, ctx_ref) {
                                            ctx_ref.assumed_constraints_push(bv, true);
                                        }
                                        added += 1;
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }

                // Slow path: convert via RustBV
                match claripy_to_rustbv(py, &item, ctx_ref) {
                    Ok(bv) => {
                        if bv.width() == 1 {
                            sym_ctx.assume_true(&bv);
                        } else {
                            let zero = RustBV::concrete(0, bv.width());
                            let neq = bv.ne(&zero, ctx_ref);
                            sym_ctx.assume_true(&neq);
                        }
                        added += 1;
                    }
                    Err(e) => {
                        log::debug!("Could not convert initial constraint: {}", e);
                    }
                }
            }
            log::debug!("Added {} initial constraints to state {}", added, state_id);
            return Ok(state.satisfiable());
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Export raw Z3 assertion pointers from a state's solver.
    /// Returns a list of usize pointers that can be passed to import_z3_constraints.
    /// This is lossless — captures ALL Z3 assertions, not just those tracked
    /// in assumed_constraints (which drops constraints where claripy_to_rustbv fails).
    #[cfg(feature = "vex-engine-z3")]
    pub fn export_z3_constraint_ptrs(&self, state_id: u64) -> PyResult<Vec<usize>> {
        if let Some(state) = self.find_state(state_id) {
            let solver_ref = state.solver();
            let ctx = solver_ref.borrow();
            return Ok(ctx.export_z3_assertion_ptrs());
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Import raw Z3 assertion pointers to a state's solver.
    /// Uses add_constraint_raw for each pointer, preserving exact Z3 structure.
    #[cfg(feature = "vex-engine-z3")]
    pub fn import_z3_constraint_ptrs(&mut self, state_id: u64, ptrs: Vec<usize>) -> PyResult<bool> {
        if let Some(state) = self.find_state_mut(state_id) {
            let solver_ref = state.solver();
            let ctx = solver_ref.borrow();
            for ptr in &ptrs {
                if *ptr != 0 {
                    unsafe { ctx.add_constraint_raw(*ptr); }
                }
            }
            log::debug!("Imported {} Z3 constraints to state {}", ptrs.len(), state_id);
            return Ok(state.satisfiable());
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Debug: dump solver state for a given state.
    #[cfg(feature = "vex-engine-z3")]
    pub fn debug_solver_info(&self, state_id: u64) -> PyResult<String> {
        if let Some(state) = self.find_state(state_id) {
            let solver_ref = state.solver();
            let ctx = solver_ref.borrow();
            let push_level = ctx.debug_push_level();
            let solver_str = ctx.debug_solver_string();
            let n_assertions = solver_str.matches('\n').count();
            return Ok(format!("push_level={}, assertions_lines={}, solver:\n{}", push_level, n_assertions, &solver_str[..solver_str.len().min(2000)]));
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Export constraints from a state in any stash as claripy ASTs.
    pub fn export_state_constraints(
        &self,
        py: Python<'_>,
        state_id: u64,
    ) -> PyResult<Vec<PyObject>> {
        let claripy = py.import("claripy")?;
        if let Some(state) = self.find_state(state_id) {
            let solver_ref = state.solver();
            let ctx = solver_ref.borrow();
            let assumed = ctx.get_assumed_constraints();
            let mut results = Vec::new();
            for (bv, is_true) in &assumed {
                match rustbv_to_claripy(py, bv, claripy.as_any()) {
                    Ok(ast) => {
                        if *is_true {
                            results.push(ast);
                        } else {
                            match claripy.call_method1("Not", (ast,)) {
                                Ok(negated) => results.push(negated.unbind()),
                                Err(_) => {}
                            }
                        }
                    }
                    Err(_) => {}
                }
            }
            return Ok(results);
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Fork the solver context of an arbitrary state (by ID).
    ///
    /// Returns a new RustSolverContext with all of the state's constraints,
    /// allowing Python to evaluate/solve against any state — not just the
    /// pending callback state.  This is used by RustStateProxy.
    pub fn fork_state_solver(&self, state_id: u64) -> PyResult<RustSolverContext> {
        if let Some(state) = self.find_state(state_id) {
            let solver_ref = state.solver();
            let forked_ctx = solver_ref.borrow().fork();
            return Ok(RustSolverContext::from_sym_context(forked_ctx));
        }
        Err(PyValueError::new_err(format!(
            "fork_state_solver: state {} not found",
            state_id
        )))
    }

    /// Get the number of constraints in the pending state's solver.
    pub fn pending_constraint_count(&self) -> PyResult<usize> {
        if let Some(ref pending) = self.pending_callback {
            let solver_ref = pending.state.solver();
            Ok(solver_ref.borrow().num_constraints())
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Get the ID of the state currently being stepped.
    pub fn get_current_stepping_state_id(&self) -> Option<u64> {
        self.current_stepping_state_id
    }

    /// Load from pending callback state's Rust memory.
    /// Used by SimProcedure callbacks to read the correct per-state memory.
    /// Get all mapped page addresses from pending callback state's memory.
    pub fn get_pending_mapped_pages(&self) -> PyResult<Vec<u64>> {
        let pending = self.pending_callback.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no pending callback state"))?;
        Ok(pending.state.memory().pages().keys().map(|&pn| pn << 12).collect())
    }

    /// Load an entire page (4096 bytes) from pending callback state's memory.
    pub fn pending_memory_load_page(&self, page_addr: u64) -> PyResult<Vec<u8>> {
        let pending = self.pending_callback.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no pending callback state"))?;
        pending.state.memory().load_page_concrete(page_addr)
            .map_err(|e| PyValueError::new_err(format!("page load failed: {}", e)))
    }

    pub fn pending_memory_load(&self, addr: u64, size: u32) -> PyResult<Vec<u8>> {
        let pending = self.pending_callback.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no pending callback state"))?;
        let solver_ref = pending.state.solver();
        let ctx = solver_ref.borrow();
        match pending.state.memory().load_concrete(addr, size, &*ctx) {
            Ok(bv) => {
                if let Some(val) = bv.as_u128() {
                    let byte_count = (size as usize).min(16);
                    Ok(val.to_le_bytes()[..byte_count].to_vec())
                } else {
                    let solver = pending.state.solver();
                    let ctx = solver.borrow();
                    if let Some(val) = ctx.eval(&bv) {
                        let byte_count = (size as usize).min(16);
                        Ok(val.to_le_bytes()[..byte_count].to_vec())
                    } else {
                        Ok(vec![0u8; size as usize])
                    }
                }
            }
            Err(_) => Ok(vec![0u8; size as usize]),
        }
    }

    /// Store to pending callback state's Rust memory.
    pub fn pending_memory_store(&mut self, addr: u64, data: &[u8]) -> PyResult<()> {
        let pending = self.pending_callback.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("no pending callback state"))?;
        let mut value: u128 = 0;
        for (i, &byte) in data.iter().enumerate() {
            if i < 16 { value |= (byte as u128) << (i * 8); }
        }
        let bv = RustBV::concrete(value, (data.len() * 8) as u32);
        pending.state.memory_mut().store_concrete(addr, bv)
            .map_err(|e| PyRuntimeError::new_err(format!("memory store error: {}", e)))
    }

    /// Map memory with data in pending callback state.
    pub fn pending_memory_map_data(&mut self, addr: u64, data: &[u8], perm: u8) -> PyResult<()> {
        let pending = self.pending_callback.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("no pending callback state"))?;
        pending.state.map_memory_data(addr, data, crate::memory::Permission::from_bits(perm));
        Ok(())
    }

    /// Set address to skip hook check for on next step.
    ///
    /// This is used to prevent infinite loops with zero-length hooks.
    /// When a hook with length=0 runs, it returns to the same address.
    /// Without this skip mechanism, the hook would trigger again immediately.
    ///
    /// The skip is automatically cleared after one step or when the address is used.
    /// GAP 6: Stack-based tracking allows for nested zero-length hooks.
    pub fn set_skip_hook_addr(&mut self, addr: u64) {
        // Set expiry to current_step + 2 to account for step increment
        // This ensures the skip persists through the next step
        let expiry = self.steps + 2;
        self.skip_hook_stack.push((addr, expiry));
        log::debug!("Added skip hook 0x{:x} with expiry step {}", addr, expiry);
    }

    /// Clear all pending skip_hook entries.
    pub fn clear_skip_hook_addr(&mut self) {
        self.skip_hook_stack.clear();
    }

    /// Clear skip entry for a specific address.
    pub fn clear_skip_hook_for_addr(&mut self, addr: u64) {
        self.skip_hook_stack.retain(|&(a, _)| a != addr);
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
    pub fn move_states(&mut self, from_stash: &str, to_stash: &str, filter_fn: Option<PyObject>) -> PyResult<usize> {
        // If no filter, move all
        if filter_fn.is_none() {
            if let Some(mut from) = self.sm.remove(from_stash) {
                let count = from.len();
                // Update index for all moved states
                for state in from.iter() {
                    self.sm.index(state.state_id(), to_stash);
                }
                let to = self.sm.stashes_mut().entry(to_stash.to_string()).or_insert_with(VecDeque::new);
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
        Python::with_gil(|py| -> PyResult<()> {
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
        let to = self.sm.stashes_mut().entry(to_stash.to_string()).or_insert_with(VecDeque::new);
        for state in moved.into_iter().rev() {
            to.push_back(state);
        }
        Ok(count)
    }

    /// P8 fix: Move a single state by ID between stashes.
    pub fn move_state(&mut self, state_id: u64, from_stash: &str, to_stash: &str) -> PyResult<bool> {
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
            let to = self.sm.stashes_mut().entry(to_stash.to_string()).or_insert_with(VecDeque::new);
            to.push_back(state);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// P8 fix: Clear all states from a stash.
    pub fn clear_stash(&mut self, stash: &str) {
        self.sm.clear(stash);
    }

    /// Get statistics.
    pub fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("steps", self.steps)?;
        dict.set_item("active", self.active_count())?;
        dict.set_item("found", self.found_count())?;
        dict.set_item("errors", self.errors.len())?;
        dict.set_item("hooks", self.hooks.len())?;
        dict.set_item("simprocedures", self.simprocedures.len())?;
        dict.set_item("find_addrs", self.find_addrs.len())?;
        dict.set_item("avoid_addrs", self.avoid_addrs.len())?;
        dict.set_item("block_cache_size", self.block_cache.len())?;
        dict.set_item("native_proc_calls", self.native_proc_stats.native_calls)?;
        dict.set_item("native_proc_fallbacks", self.native_proc_stats.python_fallbacks)?;
        dict.set_item("avoided_count", self.sm.avoided_count)?;
        dict.set_item("pruned_count", self.sm.pruned_count)?;
        dict.set_item("deadended_count", self.sm.deadended_count)?;
        dict.set_item("drop_terminal_states", self.sm.drop_terminal_states())?;
        dict.set_item("state_roots_size", self.sm.roots().len())?;
        Ok(dict)
    }

    // =========================================================================
    // Native Procedure Management
    // =========================================================================

    /// Disable all native procedures (always use Python).
    pub fn disable_native_procedures(&mut self) {
        self.native_procedures.disable_all();
    }

    /// Enable all native procedures.
    pub fn enable_native_procedures(&mut self) {
        self.native_procedures.enable_all();
    }

    /// Check if native procedures are enabled.
    pub fn native_procedures_enabled(&self) -> bool {
        self.native_procedures.is_enabled()
    }

    /// Disable a specific native procedure (fall back to Python).
    pub fn disable_native_procedure(&mut self, name: &str) {
        self.native_procedures.disable(name);
    }

    /// Enable a specific native procedure.
    pub fn enable_native_procedure(&mut self, name: &str) {
        self.native_procedures.enable(name);
    }

    /// Set a Python override for a procedure.
    ///
    /// When set, the native implementation is never called.
    pub fn set_python_override(&mut self, name: &str) {
        self.native_procedures.set_python_override(name);
    }

    /// Remove a Python override.
    pub fn remove_python_override(&mut self, name: &str) {
        self.native_procedures.remove_python_override(name);
    }

    /// Get list of available native procedures.
    pub fn list_native_procedures(&self) -> Vec<String> {
        self.native_procedures.procedure_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Check if a procedure has a native implementation.
    pub fn has_native_procedure(&self, name: &str) -> bool {
        self.native_procedures.has_native(name)
    }

    /// Get native procedure statistics.
    pub fn native_procedure_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("native_calls", self.native_proc_stats.native_calls)?;
        dict.set_item("python_fallbacks", self.native_proc_stats.python_fallbacks)?;

        let call_counts = PyDict::new(py);
        for (name, count) in &self.native_proc_stats.call_counts {
            call_counts.set_item(name, *count)?;
        }
        dict.set_item("call_counts", call_counts)?;

        Ok(dict)
    }

    // =========================================================================
    // Native Uniqueness Filter
    // =========================================================================

    /// Enable native uniqueness filter with given register names.
    ///
    /// After each step in run(), states with duplicate register tuples
    /// are moved to 'not_unique' stash. This replaces the Python
    /// CheckUniqueness technique with zero FFI overhead.
    pub fn register_uniqueness_filter(&mut self, register_names: Vec<String>) {
        self.uniqueness_registers = register_names;
        self.uniqueness_set.clear();
        // Ensure not_unique stash exists
        self.sm.stashes_mut().entry("not_unique".to_string()).or_insert_with(VecDeque::new);
    }

    /// Disable the native uniqueness filter.
    pub fn disable_uniqueness_filter(&mut self) {
        self.uniqueness_registers.clear();
        self.uniqueness_set.clear();
    }

    /// Check if native uniqueness filter is enabled.
    pub fn uniqueness_filter_enabled(&self) -> bool {
        !self.uniqueness_registers.is_empty()
    }

    /// Get the number of unique register tuples seen.
    pub fn uniqueness_set_size(&self) -> usize {
        self.uniqueness_set.len()
    }

    // =========================================================================
    // State Export Methods
    // =========================================================================

    /// Export a state by ID as a full snapshot.
    ///
    /// This searches all stashes for the state with the given ID and returns
    /// a complete snapshot that can be used to reconstruct an angr SimState.
    pub fn export_state(&self, state_id: u64) -> PyResult<crate::state::ExplorationStateSnapshot> {
        if let Some(state) = self.find_state(state_id) {
            return Ok(state.export_full());
        }

        // Also check pending callback state
        if let Some(ref pending) = self.pending_callback {
            if pending.state.state_id() == state_id {
                return Ok(pending.state.export_full());
            }
        }

        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Export a state by ID, flushing pending writes first.
    pub fn export_state_flushed(&mut self, state_id: u64) -> PyResult<crate::state::ExplorationStateSnapshot> {
        if let Some(state) = self.find_state_mut(state_id) {
            return Ok(state.flush_and_export_full());
        }

        // Also check pending callback state
        if let Some(ref mut pending) = self.pending_callback {
            if pending.state.state_id() == state_id {
                return Ok(pending.state.flush_and_export_full());
            }
        }

        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Export all states in a stash as snapshots.
    pub fn export_stash(&self, stash: &str) -> Vec<crate::state::ExplorationStateSnapshot> {
        self.sm.get(stash)
            .map(|s| s.iter().map(|state| state.export_full()).collect())
            .unwrap_or_default()
    }

    /// Export all found states as snapshots (flushing pending writes).
    pub fn export_found_states_flushed(&mut self) -> Vec<crate::state::ExplorationStateSnapshot> {
        if let Some(states) = self.sm.get_mut(STASH_FOUND) {
            states.iter_mut().map(|s| s.flush_and_export_full()).collect()
        } else {
            Vec::new()
        }
    }

    /// Export all found states as snapshots.
    pub fn export_found_states(&self) -> Vec<crate::state::ExplorationStateSnapshot> {
        self.export_stash(STASH_FOUND)
    }

    /// Evaluate a symbolic value in a state's solver context.
    ///
    /// This allows Python to get concrete values for symbolic inputs
    /// that were found during exploration.
    pub fn eval_in_state(&self, state_id: u64, addr: u64, size: u32) -> PyResult<Option<Vec<u8>>> {
        if let Some(state) = self.find_state(state_id) {
            match state.memory_load(addr, size) {
                Ok(bv) => {
                    if let Some(val) = state.eval(&bv) {
                        let bytes: Vec<u8> = (0..size as usize)
                            .map(|i| (val >> (i * 8)) as u8)
                            .collect();
                        return Ok(Some(bytes));
                    }
                    return Ok(None);
                }
                Err(_) => return Ok(None),
            }
        }

        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Debug: Get symbolic object info for a state.
    pub fn state_symbolic_info(&self, state_id: u64, addr: u64) -> PyResult<String> {
        if let Some(state) = self.find_state(state_id) {
            let mem = state.memory();
            let total = mem.symbolic_object_count();
            let has_at_addr = mem.get_symbolic_object(addr).map(|bv| bv.width());
            let page_num = addr >> 12;
            let offset = (addr & 0xFFF) as u16;
            let page_info = if let Some(page) = mem.pages().get(&page_num) {
                format!("page=mapped sym_at_offset={}", page.is_symbolic(offset))
            } else {
                "page=unmapped".to_string()
            };
            return Ok(format!(
                "total_sym_objs={} at_0x{:x}={:?} {}",
                total, addr, has_at_addr, page_info
            ));
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
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
    pub fn get_state_symbolic_z3_asts(&self, state_id: u64) -> PyResult<Vec<(u64, usize, u32)>> {
        use z3::ast::Ast;
        if let Some(state) = self.find_state(state_id) {
            let mem = state.memory();
            let mut result = Vec::new();
            for (&addr, bv) in mem.symbolic_objects_iter() {
                // Only export Expression values (Rust-computed).
                // Skip Symbolic values (imported from Python) — Python already
                // has those with proper claripy identity.
                if matches!(bv, RustBV::Expression { .. }) {
                    let z3_ast = bv.to_z3_ast();
                    let raw_ptr = z3_ast.get_z3_ast().as_ptr() as usize;
                    // Prevent z3::ast::BV destructor from decrementing the ref count.
                    // Python takes ownership of this pointer.
                    std::mem::forget(z3_ast);
                    result.push((addr, raw_ptr, bv.width()));
                }
            }
            return Ok(result);
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Check if constraints are satisfiable for a state.
    pub fn state_satisfiable(&self, state_id: u64) -> PyResult<bool> {
        if let Some(state) = self.find_state(state_id) {
            return Ok(state.satisfiable());
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Get a register value from a state.
    pub fn get_state_register(&self, state_id: u64, name: &str) -> PyResult<Option<u128>> {
        if let Some(state) = self.find_state(state_id) {
            return Ok(state.get_register(name).and_then(|bv| bv.as_u128()));
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Get multiple register values from a state in one FFI call.
    /// Returns a list of Option<u128> in the same order as the input names.
    pub fn get_state_registers_batch(&self, state_id: u64, names: Vec<String>) -> PyResult<Vec<Option<u128>>> {
        if let Some(state) = self.find_state(state_id) {
            let results: Vec<Option<u128>> = names.iter()
                .map(|name| state.get_register(name).and_then(|bv| bv.as_u128()))
                .collect();
            return Ok(results);
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Get memory from a state.
    pub fn get_state_memory(&self, state_id: u64, addr: u64, size: u32) -> PyResult<Option<Vec<u8>>> {
        if let Some(state) = self.find_state(state_id) {
            match state.memory_load(addr, size) {
                Ok(bv) => {
                    if let Some(val) = bv.as_u128() {
                        let bytes: Vec<u8> = (0..size as usize)
                            .map(|i| (val >> (i * 8)) as u8)
                            .collect();
                        return Ok(Some(bytes));
                    }
                    if let Some(val) = state.eval(&bv) {
                        let bytes: Vec<u8> = (0..size as usize)
                            .map(|i| (val >> (i * 8)) as u8)
                            .collect();
                        return Ok(Some(bytes));
                    }
                    return Ok(None);
                }
                Err(_) => return Ok(None),
            }
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Check if a state has stdout output (dirty flag check, no allocation).
    pub fn has_state_stdout(&self, state_id: u64) -> bool {
        if let Some(state) = self.find_state(state_id) {
            return state.has_stdout();
        }
        if let Some(ref cb) = self.pending_callback {
            if cb.state.state_id() == state_id {
                return cb.state.has_stdout();
            }
        }
        false
    }

    /// Get the stdout buffer for a state by ID.
    ///
    /// Returns the accumulated output from native puts/printf calls.
    pub fn get_state_stdout(&self, state_id: u64) -> PyResult<Vec<u8>> {
        self.get_state_fd_output(state_id, 1)
    }

    /// Get the output buffer for a specific file descriptor.
    ///
    /// Returns the accumulated output from native write/puts/printf calls.
    pub fn get_state_fd_output(&self, state_id: u64, fd: u32) -> PyResult<Vec<u8>> {
        if let Some(state) = self.find_state(state_id) {
            return Ok(state.fd_buffer(fd).to_vec());
        }
        // Also check pending callback state
        if let Some(ref cb) = self.pending_callback {
            if cb.state.state_id() == state_id {
                return Ok(cb.state.fd_buffer(fd).to_vec());
            }
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }
