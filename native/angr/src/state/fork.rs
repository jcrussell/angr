//! Fork / merge family for `RustSimState`.

use super::*;

impl RustSimState {
    /// Clone the three Python-AST metadata maps. Each PyObject ref-count is
    /// incremented under the GIL so the parent and fork share strong refs.
    #[allow(clippy::type_complexity)] // single-use private fn; aliases would obscure intent
    fn clone_py_metadata(
        &self,
    ) -> (
        HashMap<u64, Py<PyAny>>,
        HashMap<u64, (Py<PyAny>, u32)>,
        HashMap<u64, (Py<PyAny>, u32)>,
    ) {
        // Common case for binaries with no symbolic pages or hooks: all three
        // maps are empty, so there is nothing to clone_ref. Skip the GIL
        // acquire entirely (Python::attach is not free on the exploration
        // worker thread, where the GIL is not already held). Guard MUST check
        // all three maps — any non-empty map needs the clone_ref pass.
        if self.symbolic_pages.is_empty()
            && self.hook_symbolic_memory.is_empty()
            && self.addr_to_ast.is_empty()
        {
            return (HashMap::default(), HashMap::default(), HashMap::default());
        }
        Python::attach(|py| {
            let pages = self
                .symbolic_pages
                .iter()
                .map(|(k, v)| (*k, v.clone_ref(py)))
                .collect();
            let hook = self
                .hook_symbolic_memory
                .iter()
                .map(|(k, (v, sz))| (*k, (v.clone_ref(py), *sz)))
                .collect();
            let addr_map = self
                .addr_to_ast
                .iter()
                .map(|(k, (v, sz))| (*k, (v.clone_ref(py), *sz)))
                .collect();
            (pages, hook, addr_map)
        })
    }
    // =========================================================================
    // Forking
    // =========================================================================

    /// Fork the state (O(1) copy-on-write).
    ///
    /// Creates a new state that shares memory pages via CoW.
    /// The solver context is forked to preserve constraints.
    ///
    /// See module-level `state-id-never-reused` (child gets a fresh
    /// monotonic ID, parent's ID is preserved on the parent),
    /// `arc-make-mut-cow` (registers/memory/hooks/environment/fs share Arc
    /// or persistent backing with the parent), and `state-metadata-dataclass`
    /// (the three `Py<PyAny>` metadata maps are cloned under the GIL).
    ///
    /// # Returns
    /// A new state with the same register/memory/constraint state.
    pub fn fork(&self) -> Self {
        let forked_solver = Rc::new(RefCell::new(self.solver.borrow().fork()));
        let (symbolic_pages, hook_symbolic_memory, addr_to_ast) = self.clone_py_metadata();

        let child_id = next_state_id();
        // `state-id-never-reused`: monotonic counter must produce a value
        // strictly greater than the parent's ID. Tautological today; this
        // assert fires if a future refactor reorders the allocation or
        // (worse) introduces ID recycling.
        debug_assert!(
            child_id > self.state_id,
            "next_state_id() must monotonically increase; got child={} parent={}",
            child_id,
            self.state_id,
        );

        RustSimState {
            arch: self.arch.clone(),
            vex_arch: self.vex_arch,
            registers: self.registers.fork(),
            memory: self.memory.fork(),
            solver: forked_solver,
            pc: self.pc,
            state_id: child_id,
            parent_id: Some(self.state_id),
            history: self.history.clone(),
            detailed_history: self.detailed_history.clone(),
            max_history: self.max_history,
            hooks: self.hooks.clone(),
            concretizer: self.concretizer.clone(),
            track_history: self.track_history,
            fs: self.fs.clone(),
            heap_brk: self.heap_brk,
            posix_brk: self.posix_brk,
            mmap_base: self.mmap_base,
            getopt_optind: self.getopt_optind,
            getopt_optchar: self.getopt_optchar,
            ctype_loc: self.ctype_loc,
            stdin_symbols: self.stdin_symbols.clone(),
            call_stack: self.call_stack.clone(),
            heap_metadata: self.heap_metadata.clone(),
            inspection: self.inspection.clone(),
            environment: self.environment.clone(),
            symbolic_pages,
            hook_symbolic_memory,
            addr_to_ast,
            last_time: self.last_time.clone(),
            no_ip_concretization: self.no_ip_concretization,
            no_symbolic_jump_resolution: self.no_symbolic_jump_resolution,
            keep_ip_symbolic: self.keep_ip_symbolic,
            force_eager_forks: self.force_eager_forks,
            cgc_allocation_base: self.cgc_allocation_base,
            cgc_sinkholes: self.cgc_sinkholes.clone(),
            sim_options: self.sim_options.clone(),
        }
    }

    /// Cross-context twin of [`Self::fork`] (angr-ahypj): produce a copy of
    /// this state whose every context-bound `RustBV` — register overlays,
    /// symbolic memory (`symbolic_objects` / `multi_objects` / pending
    /// writes), path constraints, and `last_time` — has been `Z3_translate`d
    /// into `target_ctx`. Every other field is context-independent and cloned
    /// exactly as `fork` does.
    ///
    /// Unlike `fork`, identity is preserved: `state_id` and `parent_id` carry
    /// over unchanged because this is the *same* state observed in a different
    /// worker's Z3 context, not a new path.
    ///
    /// The `Py<PyAny>` overlay maps (`symbolic_pages`, `hook_symbolic_memory`,
    /// `addr_to_ast`) hold Python claripy ASTs, which are context-independent;
    /// they are `clone_ref`'d under the GIL exactly as in `fork`.
    ///
    /// # Preconditions
    ///
    /// `target_ctx` must be the **active thread-local Z3 context** and a
    /// *different* context from this state's — see
    /// [`SymContext::translate_into`] and [`RustBV::translate_into`]. In the
    /// Option-A parallel model this runs on the target worker's thread after
    /// `set_thread_local(target_ctx)`.
    #[cfg(feature = "vex-engine-z3")]
    pub fn translate_state(&self, target_ctx: &z3::Context) -> Self {
        let translated_solver = Rc::new(RefCell::new(
            self.solver.borrow().translate_into(target_ctx),
        ));
        let (symbolic_pages, hook_symbolic_memory, addr_to_ast) = self.clone_py_metadata();

        RustSimState {
            arch: self.arch.clone(),
            vex_arch: self.vex_arch,
            registers: self.registers.translate_into(target_ctx),
            memory: self.memory.translate_into(target_ctx),
            solver: translated_solver,
            pc: self.pc,
            state_id: self.state_id,
            parent_id: self.parent_id,
            history: self.history.clone(),
            detailed_history: self.detailed_history.clone(),
            max_history: self.max_history,
            hooks: self.hooks.clone(),
            concretizer: self.concretizer.clone(),
            track_history: self.track_history,
            fs: self.fs.clone(),
            heap_brk: self.heap_brk,
            posix_brk: self.posix_brk,
            mmap_base: self.mmap_base,
            getopt_optind: self.getopt_optind,
            getopt_optchar: self.getopt_optchar,
            ctype_loc: self.ctype_loc,
            stdin_symbols: self.stdin_symbols.clone(),
            call_stack: self.call_stack.clone(),
            heap_metadata: self.heap_metadata.clone(),
            inspection: self.inspection.clone(),
            environment: self.environment.clone(),
            symbolic_pages,
            hook_symbolic_memory,
            addr_to_ast,
            last_time: self
                .last_time
                .as_ref()
                .map(|bv| bv.translate_into(target_ctx)),
            no_ip_concretization: self.no_ip_concretization,
            no_symbolic_jump_resolution: self.no_symbolic_jump_resolution,
            keep_ip_symbolic: self.keep_ip_symbolic,
            force_eager_forks: self.force_eager_forks,
            cgc_allocation_base: self.cgc_allocation_base,
            cgc_sinkholes: self.cgc_sinkholes.clone(),
            sim_options: self.sim_options.clone(),
        }
    }

    /// Fork with a constraint on the true branch.
    pub fn fork_true(&self, condition: &RustBV) -> Self {
        let forked_solver = Rc::new(RefCell::new(self.solver.borrow().fork_true(condition)));
        let (symbolic_pages, hook_symbolic_memory, addr_to_ast) = self.clone_py_metadata();

        RustSimState {
            arch: self.arch.clone(),
            vex_arch: self.vex_arch,
            registers: self.registers.fork(),
            memory: self.memory.fork(),
            solver: forked_solver,
            pc: self.pc,
            state_id: next_state_id(),
            parent_id: Some(self.state_id),
            history: self.history.clone(),
            detailed_history: self.detailed_history.clone(),
            max_history: self.max_history,
            hooks: self.hooks.clone(),
            concretizer: self.concretizer.clone(),
            track_history: self.track_history,
            fs: self.fs.clone(),
            heap_brk: self.heap_brk,
            posix_brk: self.posix_brk,
            mmap_base: self.mmap_base,
            getopt_optind: self.getopt_optind,
            getopt_optchar: self.getopt_optchar,
            ctype_loc: self.ctype_loc,
            stdin_symbols: self.stdin_symbols.clone(),
            call_stack: self.call_stack.clone(),
            heap_metadata: self.heap_metadata.clone(),
            inspection: self.inspection.clone(),
            environment: self.environment.clone(),
            symbolic_pages,
            hook_symbolic_memory,
            addr_to_ast,
            last_time: self.last_time.clone(),
            no_ip_concretization: self.no_ip_concretization,
            no_symbolic_jump_resolution: self.no_symbolic_jump_resolution,
            keep_ip_symbolic: self.keep_ip_symbolic,
            force_eager_forks: self.force_eager_forks,
            cgc_allocation_base: self.cgc_allocation_base,
            cgc_sinkholes: self.cgc_sinkholes.clone(),
            sim_options: self.sim_options.clone(),
        }
    }

    /// Fork with a constraint on the false branch.
    pub fn fork_false(&self, condition: &RustBV) -> Self {
        let forked_solver = Rc::new(RefCell::new(self.solver.borrow().fork_false(condition)));
        let (symbolic_pages, hook_symbolic_memory, addr_to_ast) = self.clone_py_metadata();

        RustSimState {
            arch: self.arch.clone(),
            vex_arch: self.vex_arch,
            registers: self.registers.fork(),
            memory: self.memory.fork(),
            solver: forked_solver,
            pc: self.pc,
            state_id: next_state_id(),
            parent_id: Some(self.state_id),
            history: self.history.clone(),
            detailed_history: self.detailed_history.clone(),
            max_history: self.max_history,
            hooks: self.hooks.clone(),
            concretizer: self.concretizer.clone(),
            track_history: self.track_history,
            fs: self.fs.clone(),
            heap_brk: self.heap_brk,
            posix_brk: self.posix_brk,
            mmap_base: self.mmap_base,
            getopt_optind: self.getopt_optind,
            getopt_optchar: self.getopt_optchar,
            ctype_loc: self.ctype_loc,
            stdin_symbols: self.stdin_symbols.clone(),
            call_stack: self.call_stack.clone(),
            heap_metadata: self.heap_metadata.clone(),
            inspection: self.inspection.clone(),
            environment: self.environment.clone(),
            symbolic_pages,
            hook_symbolic_memory,
            addr_to_ast,
            last_time: self.last_time.clone(),
            no_ip_concretization: self.no_ip_concretization,
            no_symbolic_jump_resolution: self.no_symbolic_jump_resolution,
            keep_ip_symbolic: self.keep_ip_symbolic,
            force_eager_forks: self.force_eager_forks,
            cgc_allocation_base: self.cgc_allocation_base,
            cgc_sinkholes: self.cgc_sinkholes.clone(),
            sim_options: self.sim_options.clone(),
        }
    }

    /// Replace the solver context with a different one.
    /// Used for deferred fork processing where the alternate path needs a solver
    /// snapshot from before the branch constraint was added.
    pub fn replace_solver(&mut self, ctx: crate::symbolic::SymContext) {
        self.solver = Rc::new(RefCell::new(ctx));
    }

    /// Create a forked state using a full branch snapshot (solver + registers + memory).
    /// The resulting state has the correct state from the branch point, not from
    /// the continuation of the taken path.
    pub fn fork_from_snapshot(&self, snapshot: crate::interpreter::BranchSnapshot) -> Self {
        let forked_solver = Rc::new(RefCell::new(snapshot.solver));
        let (symbolic_pages, hook_symbolic_memory, addr_to_ast) = self.clone_py_metadata();
        RustSimState {
            arch: self.arch.clone(),
            vex_arch: self.vex_arch,
            registers: snapshot.registers,
            memory: snapshot.memory.unwrap_or_else(|| self.memory.fork()),
            solver: forked_solver,
            pc: self.pc,
            state_id: next_state_id(),
            parent_id: Some(self.state_id),
            history: self.history.clone(),
            detailed_history: self.detailed_history.clone(),
            max_history: self.max_history,
            hooks: self.hooks.clone(),
            concretizer: self.concretizer.clone(),
            track_history: self.track_history,
            fs: self.fs.clone(),
            heap_brk: self.heap_brk,
            posix_brk: self.posix_brk,
            mmap_base: self.mmap_base,
            getopt_optind: self.getopt_optind,
            getopt_optchar: self.getopt_optchar,
            ctype_loc: self.ctype_loc,
            stdin_symbols: self.stdin_symbols.clone(),
            call_stack: self.call_stack.clone(),
            heap_metadata: self.heap_metadata.clone(),
            inspection: self.inspection.clone(),
            environment: self.environment.clone(),
            symbolic_pages,
            hook_symbolic_memory,
            addr_to_ast,
            last_time: self.last_time.clone(),
            no_ip_concretization: self.no_ip_concretization,
            no_symbolic_jump_resolution: self.no_symbolic_jump_resolution,
            keep_ip_symbolic: self.keep_ip_symbolic,
            force_eager_forks: self.force_eager_forks,
            cgc_allocation_base: self.cgc_allocation_base,
            cgc_sinkholes: self.cgc_sinkholes.clone(),
            sim_options: self.sim_options.clone(),
        }
    }

    /// Merge this state with one or more other states using symbolic merge conditions.
    ///
    /// Creates a new merged state where registers and memory that differ between
    /// states are represented as ITE expressions guarded by merge conditions.
    /// The solver receives guarded constraints from all input states.
    ///
    /// `merge_conditions` has one entry per state: self first, then each of `others`.
    /// Each condition is a fresh 1-bit symbolic variable indicating that path is active.
    ///
    /// Returns a new merged RustSimState.
    pub fn merge(&self, others: &[&RustSimState], merge_conditions: &[RustBV]) -> Self {
        assert_eq!(
            others.len() + 1,
            merge_conditions.len(),
            "merge_conditions must have one entry per state (self + others)"
        );

        // Merge solver contexts
        let other_solvers: Vec<_> = others.iter().map(|s| s.solver.borrow()).collect();
        let other_solver_refs: Vec<&SymContext> = other_solvers.iter().map(|s| &**s).collect();
        let merged_solver = self
            .solver
            .borrow()
            .merge(&other_solver_refs, merge_conditions);

        // Start with a clone of self's registers and merge each other into it
        let mut merged_regs = self.registers.fork();
        for (i, other) in others.iter().enumerate() {
            let cond = &merge_conditions[i + 1]; // skip self's condition
            merged_regs.merge(&other.registers, cond, &merged_solver);
        }

        // Start with a clone of self's memory and merge each other into it
        let mut merged_mem = self.memory.fork();
        for (i, other) in others.iter().enumerate() {
            let cond = &merge_conditions[i + 1];
            merged_mem.merge(&other.memory, cond, &merged_solver);
        }

        // Merge stdout buffers: pick the longest (heuristic — full merge would need ITE on bytes)
        let mut best_fs = self.fs.clone();
        let mut best_len = self.stdout_buffer().len();
        for other in others {
            let other_len = other.stdout_buffer().len();
            if other_len > best_len {
                best_fs = other.fs.clone();
                best_len = other_len;
            }
        }

        // Merge stdin symbols (union)
        let mut merged_stdin = self.stdin_symbols.clone();
        for other in others {
            for sym in &other.stdin_symbols {
                if !merged_stdin.iter().any(|(n, _)| n == &sym.0) {
                    merged_stdin.push(sym.clone());
                }
            }
        }

        let (symbolic_pages, hook_symbolic_memory, addr_to_ast) = self.clone_py_metadata();
        RustSimState {
            arch: self.arch.clone(),
            vex_arch: self.vex_arch,
            registers: merged_regs,
            memory: merged_mem,
            solver: Rc::new(RefCell::new(merged_solver)),
            pc: self.pc,
            state_id: next_state_id(),
            parent_id: Some(self.state_id),
            history: self.history.clone(),
            detailed_history: self.detailed_history.clone(),
            max_history: self.max_history,
            hooks: self.hooks.clone(),
            concretizer: self.concretizer.clone(),
            track_history: self.track_history,
            fs: best_fs,
            heap_brk: self.heap_brk,
            posix_brk: self.posix_brk,
            mmap_base: self.mmap_base,
            getopt_optind: self.getopt_optind,
            getopt_optchar: self.getopt_optchar,
            ctype_loc: self.ctype_loc,
            stdin_symbols: merged_stdin,
            call_stack: self.call_stack.clone(),
            heap_metadata: self.heap_metadata.clone(),
            inspection: self.inspection.clone(),
            environment: self.environment.clone(),
            symbolic_pages,
            hook_symbolic_memory,
            addr_to_ast,
            last_time: self.last_time.clone(),
            no_ip_concretization: self.no_ip_concretization,
            no_symbolic_jump_resolution: self.no_symbolic_jump_resolution,
            keep_ip_symbolic: self.keep_ip_symbolic,
            force_eager_forks: self.force_eager_forks,
            cgc_allocation_base: self.cgc_allocation_base,
            cgc_sinkholes: self.cgc_sinkholes.clone(),
            sim_options: self.sim_options.clone(),
        }
    }
}

impl Clone for RustSimState {
    fn clone(&self) -> Self {
        self.fork()
    }
}
