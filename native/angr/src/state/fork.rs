//! Fork / merge family for `RustSimState`.

use super::*;

/// Fold one branch's Python-AST overlay map into the accumulating merged map.
///
/// Shared by `merge`'s three overlay unions (`symbolic_pages`,
/// `hook_symbolic_memory`, `addr_to_ast`). Entries already present in `target`
/// are kept (earlier state wins) and the conflict is warned with `field` in the
/// message; other-only keys are cloned in. Collapsing the three copy-pasted
/// loops into one implementation means the conflict/insert logic is written —
/// and tested — exactly once (angr-n0irt.10).
fn union_overlay<'a, V>(
    target: &mut HashMap<u64, V>,
    source: impl IntoIterator<Item = (&'a u64, &'a V)>,
    field: &str,
) where
    V: Clone + 'a,
{
    for (addr, value) in source {
        if target.contains_key(addr) {
            log::warn!(
                "merge: conflicting {field} overlay at {addr:#x}; keeping earlier state's AST"
            );
        } else {
            target.insert(*addr, value.clone());
        }
    }
}

/// Union an `Arc<HashSet<_>>` config field across every merge branch, then
/// drop anything any branch explicitly removed since its fork point (that
/// branch's tombstone companion set — see `RustSimState::removed_hooks` and
/// its siblings `removed_sim_options` / `removed_env_keys`).
///
/// Shared by `merge`'s `hooks` and `sim_options` unions (angr-9ke6b.121, bug
/// fix follow-up). A plain union of the live sets can't distinguish "never
/// touched" from "explicitly removed": `Arc::make_mut` makes a removal
/// diverge the `Arc` exactly like an addition would, so a naive union
/// resurrects anything a sibling branch still has — e.g. `a.remove_hook(x)`
/// then `a.merge(&[b])` where `b` never touched `x` would silently bring `x`
/// back. Subtracting the union of every branch's tombstones fixes that:
/// "removed on some branch, not re-added by that same branch" wins over
/// "still present on another branch that never touched it" — any branch's
/// removal wins, mirroring the union-favors-presence policy this function
/// already uses for additions.
///
/// A branch whose live `Arc` is pointer-equal to `base`'s AND whose
/// tombstone `Arc` is pointer-equal to `base`'s tombstone contributes
/// nothing, so the common case (nobody mutated the set after the fork)
/// returns `base`'s `Arc` unchanged and allocates nothing.
fn union_arc_set<'a, T>(
    base: &Arc<HashSet<T>>,
    base_removed: &Arc<HashSet<T>>,
    others: impl IntoIterator<Item = (&'a Arc<HashSet<T>>, &'a Arc<HashSet<T>>)>,
) -> Arc<HashSet<T>>
where
    T: Clone + Eq + std::hash::Hash + 'a,
{
    let mut merged_live: Option<HashSet<T>> = None;
    let mut merged_removed: Option<HashSet<T>> = None;
    for (other, other_removed) in others {
        if !Arc::ptr_eq(base, other) {
            merged_live
                .get_or_insert_with(|| (**base).clone())
                .extend(other.iter().cloned());
        }
        // Tombstone sets start fresh (empty) at every fork (see
        // `RustSimState::removed_hooks`), so they are essentially never
        // pointer-equal across siblings even when neither side removed
        // anything — checking `is_empty()` instead of `Arc::ptr_eq` is both
        // correct (an empty tombstone contributes nothing to the union
        // regardless of which allocation it is) and the actually-common fast
        // path (most forks never call remove_hook/set_option(false)/unsetenv).
        if !other_removed.is_empty() {
            merged_removed
                .get_or_insert_with(|| (**base_removed).clone())
                .extend(other_removed.iter().cloned());
        }
    }
    if merged_live.is_none() && merged_removed.is_none() {
        return Arc::clone(base);
    }
    let mut live = merged_live.unwrap_or_else(|| (**base).clone());
    let removed = merged_removed.unwrap_or_else(|| (**base_removed).clone());
    if !removed.is_empty() {
        live.retain(|item| !removed.contains(item));
    }
    Arc::new(live)
}

/// Warn when a per-path config scalar disagrees across merge branches.
///
/// The merged state carries `self`'s value for the fields routed through here
/// (see [`RustSimState::merge`]); there is no meaningful union for a cursor or
/// an init-time pointer, so this makes the drop *loud* instead of silent —
/// exactly the treatment the `fs` merge already gives its dropped branches
/// (angr-9ke6b.121).
fn warn_config_divergence(field: &str, diverged: bool) {
    if diverged {
        log::warn!(
            "merge: branches disagree on `{field}`; merged state keeps the first \
             branch's value and the other branches' values are dropped"
        );
    }
}

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
        // (worse) introduces ID recycling. Always-on (angr-9ke6b.220): ID
        // recycling silently clobbers the stash index and every `state_id`-keyed
        // shadow map on the Python side, and the check is one `u64` compare
        // per fork.
        assert!(
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
            // Fresh divergence point: this child hasn't removed anything
            // relative to itself yet (see `removed_hooks` field doc).
            removed_hooks: Arc::new(HashSet::new()),
            concretizer: self.concretizer.clone(),
            track_history: self.track_history,
            fs: self.fs.clone(),
            heap_brk: self.heap_brk,
            posix_brk: self.posix_brk,
            mmap_base: self.mmap_base,
            tsc_counter: self.tsc_counter,
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
            // Fresh divergence point — see `removed_hooks` above.
            removed_env_keys: Arc::new(HashSet::new()),
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
            // Fresh divergence point — see `removed_hooks` above.
            removed_sim_options: Arc::new(HashSet::new()),
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
            // Same logical state, different Z3 context — carry over unchanged.
            removed_hooks: self.removed_hooks.clone(),
            concretizer: self.concretizer.clone(),
            track_history: self.track_history,
            fs: self.fs.translate_into(target_ctx),
            heap_brk: self.heap_brk,
            posix_brk: self.posix_brk,
            mmap_base: self.mmap_base,
            tsc_counter: self.tsc_counter,
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
            // Same logical state, different Z3 context — carry over unchanged.
            removed_env_keys: self.removed_env_keys.clone(),
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
            // Same logical state, different Z3 context — carry over unchanged.
            removed_sim_options: self.removed_sim_options.clone(),
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

        // Merge the FileSystem by the longest-stdout heuristic (angr-ph300.75).
        // This is a *documented stdout-only* merge contract: the merged state
        // keeps exactly one branch's fd table, file offsets, writes, and
        // symbolic-content registry — a true per-file merge would need
        // byte-level ITE on every file's content keyed by the merge condition
        // (out of scope, and fd numbers collide across branches so a naive
        // union is not well-defined). Ties keep the earlier branch (self >
        // earlier-other > later-other), matching the prior strictly-greater
        // loop. When a *dropped* branch carried open fds above stderr, its
        // offsets / writes are silently discarded, so warn loudly: a post-merge
        // read of a file written only on the losing branch will not see it.
        let mut best_idx = 0usize;
        let mut best_len = self.stdout_buffer().len();
        for (i, other) in others.iter().enumerate() {
            let other_len = other.stdout_buffer().len();
            if other_len > best_len {
                best_idx = i + 1;
                best_len = other_len;
            }
        }
        let best_fs = if best_idx == 0 {
            self.fs.clone()
        } else {
            others[best_idx - 1].fs.clone()
        };
        for (i, fs) in std::iter::once(&self.fs)
            .chain(others.iter().map(|o| &o.fs))
            .enumerate()
        {
            if i != best_idx && fs.has_fds_above_stderr() {
                log::warn!(
                    "merge: dropping branch {i}'s non-stdout filesystem state \
                     (open fds above stderr); merged state keeps only branch \
                     {best_idx}'s fd table by the longest-stdout heuristic — \
                     file writes/offsets on the dropped branch are lost"
                );
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
            union_overlay(
                &mut symbolic_pages,
                other.symbolic_pages(),
                "symbolic_pages",
            );
            union_overlay(
                &mut hook_symbolic_memory,
                other.hook_symbolic_memory(),
                "hook_symbolic_memory",
            );
            union_overlay(&mut addr_to_ast, other.addr_to_ast(), "addr_to_ast");
        }

        // ---- Config-like fields (angr-9ke6b.121) ----------------------------
        // Each of these can be mutated per-state *after* a fork (native procs:
        // setenv/getopt/__ctype_*_loc; Python setters: set_option, add_hook,
        // set_concretizer, inspection), so carrying only `self`'s value silently
        // loses a divergent branch's change. Set-like fields union; genuinely
        // per-path or init-time scalars keep `self`'s value but warn, the same
        // contract the `fs` merge above uses when it drops a branch.
        let merged_hooks = union_arc_set(
            &self.hooks,
            &self.removed_hooks,
            others.iter().map(|o| (&o.hooks, &o.removed_hooks)),
        );
        // `sim_options` and the four booleans below are two views of the same
        // Python option set, so they merge the same way: union / logical-OR. An
        // option a branch switched on stays on in the merged state rather than
        // being reverted by whichever branch happened to be `self`.
        let merged_sim_options = union_arc_set(
            &self.sim_options,
            &self.removed_sim_options,
            others
                .iter()
                .map(|o| (&o.sim_options, &o.removed_sim_options)),
        );
        let merged_no_ip_concretization =
            self.no_ip_concretization || others.iter().any(|o| o.no_ip_concretization);
        let merged_no_symbolic_jump_resolution = self.no_symbolic_jump_resolution
            || others.iter().any(|o| o.no_symbolic_jump_resolution);
        let merged_keep_ip_symbolic =
            self.keep_ip_symbolic || others.iter().any(|o| o.keep_ip_symbolic);
        let merged_force_eager_forks =
            self.force_eager_forks || others.iter().any(|o| o.force_eager_forks);
        // Environment: union with self-wins on a conflicting *value*, mirroring
        // `union_overlay`'s earlier-state-wins rule. A key only one branch set
        // (setenv on that path) would otherwise vanish. As with `merged_hooks`
        // / `merged_sim_options` above, a plain union can't tell "never
        // touched" from "explicitly unset" — a key any branch removed (and
        // didn't re-`setenv` on that same branch) is dropped from the merged
        // map even if another branch never touched it, mirroring
        // `union_arc_set`'s tombstone policy (angr-9ke6b.121 bug fix follow-up).
        let merged_removed_env_keys: HashSet<Vec<u8>> = {
            let mut removed: HashSet<Vec<u8>> = (*self.removed_env_keys).clone();
            for other in others {
                // Tombstone sets start fresh at every fork, so `is_empty()`
                // is the actually-common fast path — see `union_arc_set`.
                if !other.removed_env_keys.is_empty() {
                    removed.extend(other.removed_env_keys.iter().cloned());
                }
            }
            removed
        };
        let merged_environment = {
            let mut merged: Option<HashMap<Vec<u8>, Vec<u8>>> = None;
            for other in others {
                if Arc::ptr_eq(&self.environment, &other.environment) {
                    continue;
                }
                let acc = merged.get_or_insert_with(|| (*self.environment).clone());
                for (key, value) in other.environment.iter() {
                    match acc.get(key) {
                        Some(existing) if existing != value => log::warn!(
                            "merge: conflicting environment value for `{}`; keeping the \
                             earlier branch's value",
                            String::from_utf8_lossy(key)
                        ),
                        Some(_) => {}
                        None => {
                            acc.insert(key.clone(), value.clone());
                        }
                    }
                }
            }
            if merged_removed_env_keys.is_empty() {
                merged.map_or_else(|| Arc::clone(&self.environment), Arc::new)
            } else {
                let mut env = merged.unwrap_or_else(|| (*self.environment).clone());
                env.retain(|key, _| !merged_removed_env_keys.contains(key));
                Arc::new(env)
            }
        };
        // No union exists for these: a getopt cursor is a per-path scan
        // position, and the concretizer / inspection / extern-pointer fields are
        // init-time config that is *not expected* to diverge — a warning here
        // means something mutated one branch's config mid-run.
        warn_config_divergence(
            "getopt_optind",
            others.iter().any(|o| o.getopt_optind != self.getopt_optind),
        );
        warn_config_divergence(
            "getopt_optchar",
            others
                .iter()
                .any(|o| o.getopt_optchar != self.getopt_optchar),
        );
        warn_config_divergence(
            "getopt_extern",
            others.iter().any(|o| o.getopt_extern != self.getopt_extern),
        );
        warn_config_divergence(
            "ctype_loc",
            others.iter().any(|o| o.ctype_loc != self.ctype_loc),
        );
        warn_config_divergence(
            "concretizer",
            others.iter().any(|o| o.concretizer != self.concretizer),
        );
        warn_config_divergence(
            "inspection",
            others
                .iter()
                .any(|o| o.inspection.enabled_mask() != self.inspection.enabled_mask()),
        );

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
            hooks: merged_hooks,
            // Fresh baseline going forward: `merged_hooks` above already
            // resolved every branch's removal, so nothing is "removed since"
            // this new merged state yet — see `removed_hooks` field doc.
            removed_hooks: Arc::new(HashSet::new()),
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
            // Simulated time is a monotonic watermark like the allocator
            // bases above: take the furthest-advanced branch so the merged
            // state's next RDTSC can't read earlier than one an arm already
            // observed (which would make time appear to run backwards).
            tsc_counter: others
                .iter()
                .fold(self.tsc_counter, |m, o| m.max(o.tsc_counter)),
            getopt_optind: self.getopt_optind,
            getopt_optchar: self.getopt_optchar,
            getopt_extern: self.getopt_extern,
            native_resume_stack: self.native_resume_stack.clone(),
            ctype_loc: self.ctype_loc,
            stdin_symbols: merged_stdin,
            call_stack: self.call_stack.clone(),
            // Union every branch's heap bookkeeping, not just self's: heap_brk
            // is maxed above so a branch-only allocation stays reachable in the
            // merged memory, and dropping its alloc_size entry makes
            // NativeRealloc over-read past the true old size (angr-n0irt.3).
            heap_metadata: {
                let mut hm = self.heap_metadata.clone();
                for o in others {
                    hm.union_from(&o.heap_metadata);
                }
                hm
            },
            inspection: self.inspection.clone(),
            environment: merged_environment,
            // Fresh baseline going forward — see `removed_hooks` above.
            removed_env_keys: Arc::new(HashSet::new()),
            symbolic_pages,
            hook_symbolic_memory,
            addr_to_ast,
            last_time: self.last_time.clone(),
            no_ip_concretization: merged_no_ip_concretization,
            no_symbolic_jump_resolution: merged_no_symbolic_jump_resolution,
            keep_ip_symbolic: merged_keep_ip_symbolic,
            force_eager_forks: merged_force_eager_forks,
            // CGC allocate(2) bumps DOWNWARD from cgc_allocation_base
            // (checked_sub in syscalls/cgc.rs::NativeAllocateSyscall), so —
            // unlike the up-growing heap_brk/posix_brk/mmap_base watermarks
            // above — the anti-alias merge takes the *minimum* (furthest-
            // advanced) base across branches. memory::merge unions pages from
            // every branch, so keeping only self's higher base would let a
            // later allocate() hand out an address aliasing a live allocation
            // from a branch that bumped lower (angr-n0irt.2; same bug class as
            // angr-ph300.51's heap_brk/mmap_base fix).
            cgc_allocation_base: others.iter().fold(self.cgc_allocation_base, |m, o| {
                m.min(o.cgc_allocation_base)
            }),
            // A sinkhole is a freed (unmapped) region eligible for allocate()
            // reuse. After a page-unioning merge a region is only safe to reuse
            // if it was freed in *every* branch: a region freed in one branch
            // but still live in another survives in the merged memory, so
            // reusing it would alias that live data. Intersect by exact
            // (addr,len) rather than union — union would reintroduce exactly
            // that aliasing (angr-n0irt.2). Order is irrelevant to
            // cgc_take_max_sinkhole (it scans for the highest address), so
            // preserving self's order via filter is fine.
            cgc_sinkholes: self
                .cgc_sinkholes
                .iter()
                .copied()
                .filter(|hole| others.iter().all(|o| o.cgc_sinkholes.contains(hole)))
                .collect(),
            sim_options: merged_sim_options,
            // Fresh baseline going forward — see `removed_hooks` above.
            removed_sim_options: Arc::new(HashSet::new()),
        }
    }
}

impl Clone for RustSimState {
    fn clone(&self) -> Self {
        self.fork()
    }
}
