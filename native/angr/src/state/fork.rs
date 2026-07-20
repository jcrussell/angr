//! Fork / merge family for `RustSimState`.

use super::*;

impl RustSimState {
    /// Clone the three Python-AST metadata maps so the parent and the fork hold
    /// independent maps over shared AST handles.
    ///
    /// **GIL-free by construction** (angr-gorvf.4.2). The values are
    /// [`SharedPyAst`] (`Arc<Py<PyAny>>`), so this is a map copy plus an atomic
    /// refcount bump per entry — no `Python::attach`, hence no `GilWorkGuard`
    /// here and `gil_work_ns_fork_metadata` stays 0. Storing bare `Py` handles
    /// instead would force a `clone_ref` pass (which needs a `Python` token) on
    /// every fork of a state carrying any overlay, which measured as the sole
    /// GIL holder on four otherwise Python-free corpus benches. Do not "simplify"
    /// the `Arc` away.
    #[allow(clippy::type_complexity)] // single-use private fn; aliases would obscure intent
    fn clone_py_metadata(
        &self,
    ) -> (
        HashMap<u64, SharedPyAst>,
        HashMap<u64, (SharedPyAst, u32)>,
        HashMap<u64, (SharedPyAst, u32)>,
    ) {
        (
            self.symbolic_pages.clone(),
            self.hook_symbolic_memory.clone(),
            self.addr_to_ast.clone(),
        )
    }
    // =========================================================================
    // Forking
    // =========================================================================

    /// Shared constructor for the plain-fork family (`fork`, `fork_true`,
    /// `fork_false`, `fork_from_snapshot`). These four differ *only* in how
    /// they derive `registers`, `memory`, `solver`, and the child `state_id`;
    /// every other field is carried from `self` by the identical
    /// clone/copy/`clone_py_metadata` logic. Extracting that logic here keeps
    /// the ~30 per-field carries in one place (DRY, angr-0mqkc.7) so a new
    /// `RustSimState` field only has to be threaded through one struct literal
    /// on this path instead of four. `merge` and `translate_state` keep their
    /// own literals because they merge/`Z3_translate` most fields rather than
    /// plain-carrying them.
    ///
    /// The caller supplies the already-derived divergent fields; `child_id`
    /// must come from [`next_state_id`] (the monotonic-increase invariant is
    /// asserted here). `parent_id` is always `Some(self.state_id)`.
    fn fork_with(
        &self,
        registers: RegisterFile,
        memory: SymbolicMemory,
        solver: SymContext,
        child_id: u64,
    ) -> Self {
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
        let (symbolic_pages, hook_symbolic_memory, addr_to_ast) = self.clone_py_metadata();
        RustSimState {
            arch: self.arch.clone(),
            vex_arch: self.vex_arch,
            registers,
            memory,
            solver: Rc::new(RefCell::new(solver)),
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
            getopt_extern: self.getopt_extern,
            native_resume_stack: self.native_resume_stack.clone(),
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
        self.fork_with(
            self.registers.fork(),
            self.memory.fork(),
            self.solver.borrow().fork(),
            next_state_id(),
        )
    }

    /// Cross-context twin of [`Self::fork`] (angr-ahypj): produce a copy of
    /// this state whose every context-bound `RustBV` — register overlays,
    /// symbolic memory (`symbolic_objects` / `multi_objects` / pending
    /// writes), path constraints, filesystem symbolic content (fd
    /// `content_sym` + the `file_contents` registry, angr-0xyq2), and
    /// `last_time` — has been `Z3_translate`d into `target_ctx`. Every
    /// other field is context-independent and cloned exactly as `fork`
    /// does.
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
            fs: self.fs.translate_into(target_ctx),
            heap_brk: self.heap_brk,
            posix_brk: self.posix_brk,
            mmap_base: self.mmap_base,
            getopt_optind: self.getopt_optind,
            getopt_optchar: self.getopt_optchar,
            getopt_extern: self.getopt_extern,
            // angr-1ilq.2: every other context-bound field above is
            // Z3_translate'd into target_ctx; the resume stack's
            // `saved_args` carries context-bound ASTs too and must be
            // translated, not plain-cloned, or a symbolic saved_arg becomes a
            // dangling foreign-context AST under threading.
            native_resume_stack: self
                .native_resume_stack
                .iter()
                .map(|frame| frame.translate_into(target_ctx))
                .collect(),
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
        self.fork_with(
            self.registers.fork(),
            self.memory.fork(),
            self.solver.borrow().fork_true(condition),
            next_state_id(),
        )
    }

    /// Fork with a constraint on the false branch.
    pub fn fork_false(&self, condition: &RustBV) -> Self {
        self.fork_with(
            self.registers.fork(),
            self.memory.fork(),
            self.solver.borrow().fork_false(condition),
            next_state_id(),
        )
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
        let memory = snapshot.memory.unwrap_or_else(|| self.memory.fork());
        self.fork_with(snapshot.registers, memory, snapshot.solver, next_state_id())
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

        // Union the three Python-AST overlay maps across self + others
        // (angr-ph300.51). `clone_py_metadata` seeds from self; each other's
        // entries are folded in, but a key already present is kept from the
        // earlier state (self wins over others; an earlier `others` entry wins
        // over a later one) and the conflict is warned. A byte-level ITE merge
        // of overlay ASTs is out of scope here; without this union an
        // other-only symbolic overlay byte was silently lost on merge.
        let (mut symbolic_pages, mut hook_symbolic_memory, mut addr_to_ast) =
            self.clone_py_metadata();
        for other in others {
            for (addr, ast) in other.symbolic_pages() {
                if symbolic_pages.contains_key(addr) {
                    log::warn!(
                        "merge: conflicting symbolic_pages overlay at {addr:#x}; keeping earlier state's AST"
                    );
                } else {
                    symbolic_pages.insert(*addr, ast.clone());
                }
            }
            for (addr, entry) in other.hook_symbolic_memory() {
                if hook_symbolic_memory.contains_key(addr) {
                    log::warn!(
                        "merge: conflicting hook_symbolic_memory overlay at {addr:#x}; keeping earlier state's AST"
                    );
                } else {
                    hook_symbolic_memory.insert(*addr, entry.clone());
                }
            }
            for (addr, entry) in other.addr_to_ast() {
                if addr_to_ast.contains_key(addr) {
                    log::warn!(
                        "merge: conflicting addr_to_ast overlay at {addr:#x}; keeping earlier state's AST"
                    );
                } else {
                    addr_to_ast.insert(*addr, entry.clone());
                }
            }
        }
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
            // Take the furthest-advanced allocator watermarks (angr-ph300.51):
            // if any branch bumped its brk/mmap, keeping only self's would let
            // the merged state's next malloc alias live allocations from that
            // branch's ITE arm.
            heap_brk: others.iter().fold(self.heap_brk, |m, o| m.max(o.heap_brk)),
            posix_brk: others
                .iter()
                .fold(self.posix_brk, |m, o| m.max(o.posix_brk)),
            mmap_base: others
                .iter()
                .fold(self.mmap_base, |m, o| m.max(o.mmap_base)),
            getopt_optind: self.getopt_optind,
            getopt_optchar: self.getopt_optchar,
            getopt_extern: self.getopt_extern,
            native_resume_stack: self.native_resume_stack.clone(),
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
