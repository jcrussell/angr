//! Main exploration run loop.
//!
//! `RustExplorationManager::run_loop` is a thin **driver**: it owns the loop
//! frame (pop a state from the active stash, the I8 termination checks, and the
//! post-step bookkeeping) and delegates the entire per-state body to
//! `step_one`, matching on the `StepOutcome` it returns to route results.
//!
//! `step_one` is the **single stepping decision point**: it evaluates find/avoid
//! predicates, dispatches SimProcedures (native fast path or Python fallback),
//! steps the interpreter, handles deferred forks at find/avoid addresses, and
//! classifies the result into a `StepOutcome`. Extracting it (bead 1ilq.6) gives
//! the work-stealing scheduler (`scheduler.rs`) one place to share, and keeps the
//! driver small. It is a pure refactor — behavior is byte-identical to the former
//! monolithic `run_loop`.
//!
//! Note: `step_one` is `&mut self` (it mutates stashes, counters, profiling, and
//! `pending_callback`), so a `Send + Sync` scheduler worker cannot call it
//! directly — reconciling that is the 1ilq.7 GIL-strategy spike. This bead only
//! lands the seam.
//!
//! The pyclass-facing `run` thin wrapper lives in `mod.rs` and just calls
//! `self.run_loop(py, n)`. PyO3 0.27.2 without `multiple-pymethods` only
//! permits a single `#[pymethods]` impl per class, so the body is extracted
//! here as `pub(crate)` methods on `RustExplorationManager`, mirroring the
//! `helpers.rs` / `stepping.rs` extension-impl pattern used elsewhere in
//! `exploration/`.
//!
//! **Invariant I8 (cross-mixin termination, mirror of
//! rust_manager.py:98):** the run loop must terminate on EITHER (a)
//! `found_count() >= num_find` (checked at the top of every iteration),
//! OR (b) the active stash exhausting itself (`pop_*` returns `None`,
//! emitting an `active_empty` event). `found_count()` covers both
//! Rust-native finds (forks routed to the found stash by the address
//! check) and Python-predicate-derived finds (added via the need_callback
//! resume path). The earlier predicate-only termination check infinite-
//! looped when `find=int` was combined with a non-predicate technique
//! like DFS — the technique made `_active_techniques` non-empty, routing
//! through the Python predicate path, which never saw the Rust find.
//! See module-level I8 in `state.rs`.

use super::*;

/// Outcome of stepping a single state in `step_one`. The thin `run_loop` driver
/// matches on this both to route results and to gate the post-step bookkeeping
/// (`steps += 1`, uniqueness filter, native techniques, reconvergence sample),
/// which runs ONLY on `Successors`/`Terminal` — never on `Routed`/`NeedCallback`
/// (mirroring the former `continue`/`return` paths before the bookkeeping block).
///
/// Carries `PendingCallback` / `RustSimState` inline by value, exactly like
/// `StepError` (which these variants are reclassified from). Boxing to satisfy
/// `large_enum_variant` would add a heap allocation on every step termination —
/// the wrong call on the hot path; the variants are intentionally inline. See
/// the matching rationale on `StepError` in `stepping.rs`.
#[allow(clippy::large_enum_variant)]
pub(crate) enum StepOutcome {
    /// `step_one` already pushed the state into its terminal/found/active stash
    /// via pre-step or find/avoid policy (avoid-addr, find-addr, native
    /// return/subcall/no_return, NeedCallback→find/avoid routing incl. deferred
    /// forks). Driver just advances — NO post-step bookkeeping.
    Routed,
    /// Live successors from a completed interpreter step. Driver routes each via
    /// `route_successor`, THEN runs post-step bookkeeping.
    Successors(Vec<RustSimState>),
    /// A terminal disposition from a completed step. Driver applies it via
    /// `apply_terminal`, THEN runs post-step bookkeeping.
    Terminal(TerminalDisposition),
    /// A Python callback is pending. The `PendingCallback` is returned as a value
    /// (NOT yet stored in `self.pending_callback`) so the driver can build the
    /// event from this local — letting the `PythonVEXFallback` counter mutations
    /// touch `self` while only `pending` is borrowed (no E0502) — before storing.
    NeedCallback(PendingCallback),
}

/// The three terminal step outcomes, each carrying the data the driver needs to
/// reproduce the original per-stash push (and its side effects) byte-for-byte.
pub(crate) enum TerminalDisposition {
    /// `push_or_drop_terminal(STASH_DEADENDED, state)`.
    Deadended(RustSimState),
    /// `errors.push((pc, message, state_id))` then a direct `push_back` into
    /// `STASH_ERRORED` (errored states are never dropped — not push_or_drop).
    Errored {
        state: RustSimState,
        pc: u64,
        message: String,
        state_id: u64,
    },
    /// `sm.push_or_drop_terminal(STASH_UNCONSTRAINED, state)` then route each
    /// loop-exit fork like a normal successor.
    Unconstrained {
        state: RustSimState,
        forks: Vec<RustSimState>,
    },
}

impl RustExplorationManager {
    /// Inner body of the pymethods-exposed `run`. See `run` in `mod.rs`.
    ///
    /// Thin driver over `step_one`: owns the loop frame (I8 termination,
    /// state pop, post-step bookkeeping) and routes each `StepOutcome`.
    pub(crate) fn run_loop(
        &mut self,
        py: Python<'_>,
        n: Option<u32>,
    ) -> PyResult<ExplorationEvent> {
        let max_steps = n.unwrap_or(self.max_steps_per_run);

        // Ensure callbacks are set and clone to avoid borrow issues
        let callbacks = self
            .callbacks
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("callbacks not set"))?
            .clone();

        if !callbacks.is_ready() {
            return Err(PyRuntimeError::new_err("callbacks not ready"));
        }

        let run_loop_start = if self.profiling.profiling_enabled {
            Some(std::time::Instant::now())
        } else {
            None
        };

        for _ in 0..max_steps {
            // angr-v5ht: runtime thrash detection for the
            // `use_shared_lineage_solver` opt-in. Hooks at the TOP of
            // the for-loop iteration (BEFORE any early-return path)
            // because callback-heavy workloads (e.g.
            // google2016_unbreakable_0: every iteration returns via
            // `need_simprocedure`) never reach `self.steps += 1` and
            // would otherwise never sample (`bd recall
            // v5ht-sampler-tick-bottleneck`). `tick_and_sample_for_thrash`
            // uses its own internal tick counter so the sampling cadence
            // is independent of `self.steps`. Always-on: cheap (single
            // atomic load + branch on `LINEAGE_DISMANTLED`) and a no-op
            // until the kill switch is turned on. Sample every 10 ticks;
            // dismantle when ≥20 lineage_switch events show <35% hot
            // ratio over a window — threshold calibrated on N=4 workloads
            // (`bd recall v5ht-threshold-justification-2026-05-25`).
            #[cfg(feature = "vex-engine-z3")]
            crate::symbolic::lineage::tick_and_sample_for_thrash(10, 20, 35);

            // Check if we have enough solutions.
            // I8 termination path (a): `found_count()` covers both
            // Rust-native finds (`found` stash via address check) and
            // Python-predicate finds (need_callback resume). See module
            // header for the full contract.
            if self.found_count() >= self.num_find {
                return Ok(ExplorationEvent::found(
                    self.found_count(),
                    self.active_count(),
                    self.steps,
                ));
            }

            // Get next state from active stash
            // P9 fix: Use LIFO (pop_back) for DFS or FIFO (pop_front) for BFS
            let state = match self.sm.get_mut(STASH_ACTIVE).and_then(|s| {
                if self.use_lifo {
                    s.pop_back() // DFS: LIFO (most recent state first)
                } else {
                    s.pop_front() // BFS: FIFO (oldest state first)
                }
            }) {
                Some(s) => {
                    let sid = s.state_id();
                    self.current_stepping_state_id = Some(sid.into());
                    STEPPING_STATE_ID.with(|cell| cell.set(Some(sid)));
                    // angr-panhl.1: model a work-stealing migration at dispatch.
                    self.record_migration_sample(sid);
                    s
                }
                None => {
                    // No active states.
                    // I8 termination path (b): active stash exhausted —
                    // emit `found` if we picked up any solutions, else
                    // `active_empty`. Either way the loop exits here
                    // rather than spinning. See module header.
                    if self.found_count() > 0 {
                        return Ok(ExplorationEvent::found(self.found_count(), 0, self.steps));
                    } else {
                        return Ok(ExplorationEvent::active_empty(
                            self.found_count(),
                            self.steps,
                        ));
                    }
                }
            };

            match self.step_one(py, &callbacks, state)? {
                // Pre-step / find-avoid policy already routed the state. Advance
                // without post-step bookkeeping (former `continue` paths).
                StepOutcome::Routed => continue,
                // Build the event from the LOCAL `pending` (so the
                // PythonVEXFallback counter mutations are free of a borrow
                // conflict), THEN store it. Mirrors the former order exactly.
                StepOutcome::NeedCallback(pending) => {
                    let event = self.callback_event(&pending);
                    self.pending_callback = Some(pending);
                    return Ok(event);
                }
                // A real interpreter step completed — route successors / apply
                // the terminal disposition, then fall through to bookkeeping.
                StepOutcome::Successors(successors) => {
                    // Add successors to appropriate stashes, checking find/avoid
                    for successor in successors {
                        self.route_successor(successor, true);
                    }
                }
                StepOutcome::Terminal(disposition) => self.apply_terminal(disposition),
            }

            self.steps += 1;

            // Apply native uniqueness filter if enabled
            self.apply_uniqueness_filter();
            // Apply native techniques (LengthLimiter, Timeout, LoopBound)
            self.apply_native_techniques();
            // DS-instr (angr-11djq.16): sample (pc, callstack) reconvergence
            // over the post-filter active frontier. Counters only.
            self.record_reconvergence_sample();
        }

        // Record run loop timing and active state count
        if let Some(start) = run_loop_start {
            self.profiling.accumulated_stats.run_loop_time_ns += start.elapsed().as_nanos() as u64;
            self.profiling.accumulated_stats.active_states_count = self.active_count() as u64;
        }

        // Max steps reached
        Ok(ExplorationEvent::step_complete(
            self.found_count(),
            self.active_count(),
            self.steps,
        ))
    }

    /// Step a single popped state to its next outcome — the shared stepping
    /// decision point. Returns a `StepOutcome` the driver routes; this method
    /// performs all the per-state work (find/avoid checks, SimProcedure
    /// dispatch, interpreter step, deferred-fork routing) but leaves loop-frame
    /// concerns (state pop, post-step bookkeeping, event storage) to the driver.
    pub(crate) fn step_one(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        mut state: RustSimState,
    ) -> PyResult<StepOutcome> {
        // Check find/avoid before stepping
        let pc = state.pc();

        // P7 fix: Check if callable avoid predicate needs Python evaluation
        // When avoid is a callable (lambda/function), we must return to Python
        // to evaluate it for each state, not just check addresses.
        // Skip if this state was just checked (resume_avoid_predicate(false)
        // sets skip_avoid_predicate_states to prevent infinite loop).
        if self.avoid_needs_python {
            let state_id = state.state_id();
            if self
                .constraint_tracker
                .skip_avoid_predicate_states
                .remove(&state_id)
            {
                // Fall through — predicate already checked at this PC
            } else {
                let pending = PendingCallback::lightweight(
                    state,
                    CallbackReason::AvoidPredicate { addr: pc },
                );
                return Ok(StepOutcome::NeedCallback(pending));
            }
        }

        // Check avoid addresses (address-based, only when NOT using callable predicate)
        if self.avoid_addrs.contains(&pc) {
            self.push_or_drop_terminal(STASH_AVOID, state);
            return Ok(StepOutcome::Routed);
        }

        // P2 fix: Check if callable find predicate needs Python evaluation
        // When find is a callable (lambda/function), we must return to Python
        // to evaluate it for each state, not just check addresses.
        // Skip if this state was just checked (resume_find_predicate(false)
        // sets skip_find_predicate_state to avoid infinite loop).
        if self.find_needs_python {
            let state_id = state.state_id();
            if self
                .constraint_tracker
                .skip_find_predicate_states
                .remove(&state_id)
            {
                // Fall through to hooks/stepping — predicate already checked
            } else {
                let pending =
                    PendingCallback::lightweight(state, CallbackReason::FindPredicate { addr: pc });
                return Ok(StepOutcome::NeedCallback(pending));
            } // else (not skip_find_predicate_state)
        }

        // Check find addresses (address-based, only when NOT using callable predicate)
        if self.find_addrs.contains(&pc) {
            // Only add to found if the state is satisfiable
            // (UNSAT states reached the address via infeasible paths)
            if self.constraint_solver.lazy_solves || state.satisfiable() {
                self.sm
                    .stashes_mut()
                    .entry(STASH_FOUND.to_string())
                    .or_default()
                    .push_back(state);
            } else {
                log::debug!("State at find address 0x{:x} is UNSAT, pruning", pc);
                self.push_or_drop_terminal(STASH_PRUNED, state);
            }
            return Ok(StepOutcome::Routed);
        }

        // Check hooks (SimProcedures)
        // GAP 6: Stack-based skip tracking for zero-length hooks
        // Clean up expired skip entries before checking
        self.skip_hook_stack
            .retain(|&(_, expiry)| expiry > self.steps);

        // Check if this address is in the skip stack
        let should_skip_hook = self.skip_hook_stack.iter().any(|&(addr, _)| addr == pc);
        if should_skip_hook {
            // Remove this address from the skip stack (consumed)
            self.skip_hook_stack.retain(|&(addr, _)| addr != pc);
            log::debug!(
                "Skipping hook at 0x{:x} (zero-length hook, step {})",
                pc,
                self.steps
            );
        }
        if self.hooks.contains(&pc) && !should_skip_hook {
            // Check if this is a registered SimProcedure
            if let Some((name, num_args, no_return)) = self.simprocedures.get(&pc).cloned() {
                // Skip native for addresses inside the binary (user-placed hooks)
                let is_in_binary = self
                    .environment
                    .binary_regions
                    .iter()
                    .any(|(base, data)| pc >= *base && pc < *base + data.len() as u64);
                // Try native procedure first (only for external/library hooks)
                if !is_in_binary && let Some(native_proc) = self.native_procedures.get(&name) {
                    // Extract arguments from state registers (and stack
                    // when num_args exceeds the register count). On
                    // failure (symbolic SP, unmapped stack slot) skip
                    // the native fast path and let the Python
                    // SimProcedure callback below handle it — handing
                    // the native handler fabricated zeros would mask
                    // the underlying stack-setup bug.
                    match self.extract_procedure_args(&state, num_args) {
                        Err(e) => {
                            log::debug!(
                                "Skipping native procedure {} (arg extraction failed: {:?})",
                                name,
                                e
                            );
                            self.profiling.native_proc_stats.python_fallbacks += 1;
                            *self
                                .profiling
                                .native_proc_stats
                                .other_fallbacks_by_name
                                .entry(name.clone())
                                .or_insert(0) += 1;
                        }
                        Ok(args) => match native_proc.call_ex(&mut state, &args) {
                            // Borrow note: `native_proc` borrows
                            // `self.native_procedures`; that borrow ends at
                            // the `call_ex` call above (NLL), freeing
                            // `&mut self` for `setup_native_subcall` /
                            // `push_to_active_or_drop` below. These arms must
                            // NOT reference `native_proc` again. (Path B's
                            // `handle_simprocedure` uses an explicit
                            // `NativeProcDisposition` enum for the same reason.)
                            Ok(ProcOutcome::Return(ret_val)) => {
                                // Native execution succeeded
                                self.profiling.native_proc_stats.native_calls += 1;
                                *self
                                    .profiling
                                    .native_proc_stats
                                    .call_counts
                                    .entry(name.clone())
                                    .or_insert(0) += 1;

                                // For no-return procedures (exit/abort), skip
                                // the return-address dance and deadend directly.
                                // Setting PC to a stack-derived return address
                                // can produce a spurious successor (e.g. when
                                // exit is called from rejected() in fauxware,
                                // the post-call address happens to overlap
                                // main's start, causing infinite re-entry).
                                if no_return {
                                    self.push_or_drop_terminal(STASH_DEADENDED, state);
                                    return Ok(StepOutcome::Routed);
                                }

                                // Set return value if present
                                if let Some(rv) = ret_val {
                                    let ret_reg =
                                        self.environment.calling_convention.return_register();
                                    state.set_register_by_offset(ret_reg, rv);
                                }

                                // Get return address and set PC. Use the
                                // state's real register file so that LR/X30/$ra
                                // overrides see actual values; passing a blank
                                // RegisterFile here used to make ARM/ARM64/MIPS
                                // read LR=0 and set PC to 0.
                                let ctx = state.solver().borrow();
                                let ret_addr_opt = self
                                    .environment
                                    .calling_convention
                                    .get_return_addr(state.registers(), None, &ctx);
                                let pops_return_addr =
                                    self.environment.calling_convention.pops_return_addr();
                                drop(ctx);
                                if let Some(ret_addr) = ret_addr_opt {
                                    // Only adjust SP for stack-based ABIs
                                    // (x86/AMD64). ARM/ARM64/MIPS keep ret addr
                                    // in a register and leave SP untouched.
                                    if pops_return_addr {
                                        let sp = state.get_sp().as_u64().unwrap_or(0);
                                        let ptr_size = state.arch().bytes() as u64;
                                        state.set_sp(RustBV::concrete(
                                            (sp + ptr_size) as u128,
                                            state.arch().bits(),
                                        ));
                                    }
                                    state.set_pc(ret_addr);
                                } else if pops_return_addr {
                                    // Fallback: read ret addr from [sp] for
                                    // stack-based ABIs (only useful when the
                                    // calling convention's get_return_addr
                                    // declined to read memory itself).
                                    if let Some(sp) = state.get_sp().as_u64()
                                        && let Ok(ret_bv) =
                                            state.memory_load(sp, state.arch().bytes())
                                        && let Some(ret_addr) = ret_bv.as_u64()
                                    {
                                        let ptr_size = state.arch().bytes() as u64;
                                        state.set_sp(RustBV::concrete(
                                            (sp + ptr_size) as u128,
                                            state.arch().bits(),
                                        ));
                                        state.set_pc(ret_addr);
                                    }
                                }

                                self.push_to_active_or_drop(state);
                                return Ok(StepOutcome::Routed);
                            }
                            Ok(ProcOutcome::CallAndResume {
                                target,
                                args: sub_args,
                                resume_tag,
                            }) => {
                                // The proc requested a guest sub-call. Capture
                                // the caller return address from [sp] BEFORE
                                // `setup_native_subcall` overwrites that slot
                                // with the resume sentinel. A symbolic SP /
                                // unmapped slot (None) or a setup failure falls
                                // back to Python (the state is left untouched
                                // by setup on Err). Do NOT pop SP or honour
                                // `no_return` here: the guest's own `ret`
                                // advances SP, and `handle_native_resume`
                                // finishes without re-adjusting it.
                                let setup = match self.get_return_addr(&state) {
                                    Some(caller_return_addr) => self
                                        .setup_native_subcall(
                                            &mut state,
                                            name.clone(),
                                            args,
                                            caller_return_addr,
                                            target,
                                            sub_args,
                                            resume_tag,
                                        )
                                        .map_err(|e| format!("{e:?}")),
                                    None => Err("no concrete return address".to_string()),
                                };
                                match setup {
                                    Ok(()) => {
                                        self.profiling.native_proc_stats.native_calls += 1;
                                        *self
                                            .profiling
                                            .native_proc_stats
                                            .call_counts
                                            .entry(name.clone())
                                            .or_insert(0) += 1;
                                        self.push_to_active_or_drop(state);
                                        return Ok(StepOutcome::Routed);
                                    }
                                    Err(reason) => {
                                        log::debug!(
                                            "native sub-call setup failed ({}); \
                                             falling back to Python for {}",
                                            reason,
                                            name
                                        );
                                        self.profiling.native_proc_stats.python_fallbacks += 1;
                                        *self
                                            .profiling
                                            .native_proc_stats
                                            .other_fallbacks_by_name
                                            .entry(name.clone())
                                            .or_insert(0) += 1;
                                    }
                                }
                            }
                            Err(e) => {
                                // Native execution failed, fall back to Python
                                self.profiling.native_proc_stats.python_fallbacks += 1;
                                let bucket = match e {
                                    ProcedureError::SymbolicArgument(_) => {
                                        &mut self
                                            .profiling
                                            .native_proc_stats
                                            .symbolic_fallbacks_by_name
                                    }
                                    ProcedureError::NotImplemented => {
                                        &mut self
                                            .profiling
                                            .native_proc_stats
                                            .not_implemented_fallbacks_by_name
                                    }
                                    _ => {
                                        &mut self
                                            .profiling
                                            .native_proc_stats
                                            .other_fallbacks_by_name
                                    }
                                };
                                *bucket.entry(name.clone()).or_insert(0) += 1;
                            }
                        },
                    }
                } // if !is_in_binary

                // Fall back to Python for SimProcedure execution
                self.simprocedure_python_fallback_count += 1;
                *self
                    .simprocedure_fallback_by_name
                    .entry(name.clone())
                    .or_insert(0) += 1;
                let return_addr = self.get_return_addr(&state).unwrap_or(0);

                // No deferred forks in run-loop path, so pre_callback_snapshot
                // is unnecessary (it's only used as fork base for deferred forks).
                // Use shared solver (O(1) Rc clone) instead of fork (~3-40ms Z3 clone).
                let solver_ref = state.solver();
                let shared_ctx = RustSolverContext::from_shared_sym_context(solver_ref.clone());

                let pending = PendingCallback::with_context(
                    state,
                    None,
                    CallbackReason::SimProcedure {
                        addr: pc,
                        name: name.clone(),
                        num_args,
                        return_addr,
                    },
                    "Ijk_Call",
                    Some(shared_ctx),
                    Vec::new(),
                    FxHashMap::default(),
                    FxHashMap::default(),
                );

                return Ok(StepOutcome::NeedCallback(pending));
            }
        }

        // Step the state, passing the skip_hook_addr if we just skipped
        let skip_addr_for_step = if should_skip_hook { Some(pc) } else { None };
        match self.step_state_with_skip(py, callbacks, state, skip_addr_for_step) {
            Ok(successors) => Ok(StepOutcome::Successors(successors)),
            Err(StepError::NeedCallback(pending)) => {
                // Check if the callback address is a find/avoid address
                // (these were added as interpreter hooks to stop execution)
                let callback_addr = match &pending.reason {
                    CallbackReason::SimProcedure { addr, .. } => Some(*addr),
                    _ => None,
                };
                if let Some(addr) = callback_addr
                    && (self.find_addrs.contains(&addr) || self.avoid_addrs.contains(&addr))
                {
                    let is_find = self.find_addrs.contains(&addr);

                    // Process deferred forks BEFORE handling the find/avoid state.
                    // These represent unexplored branches that diverged before
                    // reaching the find/avoid address and must not be dropped.
                    let fork_base = pending
                        .pre_callback_snapshot
                        .unwrap_or_else(|| pending.state.fork());
                    let root_state_id = self.sm.root_or_self(pending.state.state_id());

                    let mut snapshots = pending.fork_snapshots;
                    let cb_fork_start = if self.profiling.profiling_enabled {
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                    let cb_fork_total = pending.deferred_forks.len() as u64;
                    for fork in pending.deferred_forks {
                        let condition = pending.stored_conditions.get(&fork.condition_id);
                        let reconstructed = if condition.is_none() {
                            if let Some(ref py_ast) = fork.condition_ast {
                                Python::attach(|py| {
                                    let ast = py_ast.bind(py);
                                    let solver_ref = fork_base.solver();
                                    let ctx: &SymContext = &solver_ref.borrow();
                                    claripy_to_rustbv(py, ast, ctx).ok()
                                })
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        if let Some(cond) = condition.or(reconstructed.as_ref()) {
                            // Add the taken-path constraint to the main state
                            // (mirrors the BlockEnd handling at line 3560-3564).
                            if fork.path_taken {
                                pending.state.solver().borrow().assume_true(cond);
                            } else {
                                pending.state.solver().borrow().assume_false(cond);
                            }
                            let fork_op_start = if self.profiling.profiling_enabled {
                                Some(std::time::Instant::now())
                            } else {
                                None
                            };
                            let forked = super::helpers::build_unexplored_fork(
                                &fork_base,
                                &fork,
                                cond,
                                &mut snapshots,
                            );
                            if let Some(start) = fork_op_start {
                                self.profiling.accumulated_stats.solver_fork_time_ns +=
                                    start.elapsed().as_nanos() as u64;
                                self.profiling.accumulated_stats.solver_fork_count += 1;
                            }
                            self.sm.set_root(forked.state_id(), root_state_id);
                            let sat_start = if self.profiling.profiling_enabled {
                                Some(std::time::Instant::now())
                            } else {
                                None
                            };
                            if self.constraint_solver.lazy_solves || forked.satisfiable() {
                                if let Some(start) = sat_start {
                                    self.profiling.accumulated_stats.solver_sat_time_ns +=
                                        start.elapsed().as_nanos() as u64;
                                    self.profiling.accumulated_stats.solver_sat_count += 1;
                                }
                                self.push_to_active_or_drop(forked);
                            } else if let Some(start) = sat_start {
                                self.profiling.accumulated_stats.solver_sat_time_ns +=
                                    start.elapsed().as_nanos() as u64;
                                self.profiling.accumulated_stats.solver_sat_count += 1;
                            }
                        }
                    }
                    if let Some(start) = cb_fork_start {
                        self.profiling.accumulated_stats.deferred_fork_time_ns +=
                            start.elapsed().as_nanos() as u64;
                        self.profiling.accumulated_stats.deferred_fork_count += cb_fork_total;
                    }

                    // Now handle the main state
                    if is_find {
                        if self.constraint_solver.lazy_solves || pending.state.satisfiable() {
                            self.sm
                                .stashes_mut()
                                .entry(STASH_FOUND.to_string())
                                .or_default()
                                .push_back(pending.state);
                        } else {
                            log::debug!("State at find address 0x{:x} is UNSAT, pruning", addr);
                            self.push_or_drop_terminal(STASH_PRUNED, pending.state);
                        }
                    } else {
                        self.push_or_drop_terminal(STASH_AVOID, pending.state);
                    }
                    return Ok(StepOutcome::Routed);
                }

                // Need Python callback — hand the pending back to the driver,
                // which builds the event and stores it.
                Ok(StepOutcome::NeedCallback(pending))
            }
            Err(StepError::Deadended(state)) => {
                Ok(StepOutcome::Terminal(TerminalDisposition::Deadended(state)))
            }
            Err(StepError::Error(state, message)) => {
                let pc = state.pc();
                let state_id = state.state_id();
                Ok(StepOutcome::Terminal(TerminalDisposition::Errored {
                    state,
                    pc,
                    message,
                    state_id,
                }))
            }
            Err(StepError::Unconstrained(state, forks)) => {
                Ok(StepOutcome::Terminal(TerminalDisposition::Unconstrained {
                    state,
                    forks,
                }))
            }
        }
    }

    /// Apply a terminal disposition to the stashes, reproducing each original
    /// per-stash push path (and its side effects) byte-for-byte. Called by the
    /// driver, which then runs post-step bookkeeping.
    pub(crate) fn apply_terminal(&mut self, disposition: TerminalDisposition) {
        match disposition {
            TerminalDisposition::Deadended(state) => {
                self.push_or_drop_terminal(STASH_DEADENDED, state);
            }
            TerminalDisposition::Errored {
                state,
                pc,
                message,
                state_id,
            } => {
                self.errors.push((pc, message, state_id));
                self.sm
                    .stashes_mut()
                    .entry(STASH_ERRORED.to_string())
                    .or_default()
                    .push_back(state);
            }
            TerminalDisposition::Unconstrained { state, forks } => {
                // State has too many symbolic jump targets - move to unconstrained stash
                log::debug!("State {} moved to unconstrained stash", state.state_id());
                self.sm.push_or_drop_terminal(STASH_UNCONSTRAINED, state);
                // angr-027h: loop-exit deferred forks materialized in eager
                // mode at the unconstrained jump. Route them to active (or
                // found/avoid) exactly like normal successors so a
                // find-guided search can reach a target behind the loop.
                for fork in forks {
                    self.route_successor(fork, true);
                }
            }
        }
    }

    /// Build the `ExplorationEvent` for a pending Python callback from its
    /// reason. Single place events are constructed (DRY): absorbs the former
    /// inline predicate/simproc constructions and the post-step match. Takes the
    /// LOCAL `pending` by ref so the `PythonVEXFallback` counter mutations can
    /// touch `self` without a borrow conflict; the driver stores `pending`
    /// afterward.
    pub(crate) fn callback_event(&mut self, pending: &PendingCallback) -> ExplorationEvent {
        let state_id = pending.state.state_id();
        match &pending.reason {
            CallbackReason::SimProcedure {
                addr,
                name,
                num_args,
                return_addr,
            } => ExplorationEvent::need_simprocedure(
                state_id,
                *addr,
                name.clone(),
                *num_args,
                *return_addr,
                self.found_count(),
                self.active_count(),
                self.steps,
            ),
            CallbackReason::Syscall { num } => ExplorationEvent::need_syscall(
                state_id,
                *num,
                self.found_count(),
                self.active_count(),
                self.steps,
            ),
            CallbackReason::FindPredicate { addr } => ExplorationEvent::need_predicate(
                state_id,
                *addr,
                "find_predicate",
                self.found_count(),
                self.active_count(),
                self.steps,
            ),
            CallbackReason::AvoidPredicate { addr } => ExplorationEvent::need_predicate(
                state_id,
                *addr,
                "avoid_predicate",
                self.found_count(),
                self.active_count(),
                self.steps,
            ),
            CallbackReason::SymbolicBranch {
                condition_id,
                true_target,
                false_target,
            } => ExplorationEvent::need_symbolic_branch(
                state_id,
                *condition_id,
                *true_target,
                *false_target,
                self.found_count(),
                self.active_count(),
                self.steps,
            ),
            CallbackReason::PythonVEXFallback { addr, reason } => {
                self.vex_fallback_count += 1;
                self.vex_fallback_addrs
                    .entry(*addr)
                    .or_insert_with(|| reason.clone());
                if reason.contains(DCAS_UNSUPPORTED_REASON) {
                    self.dcas_unsupported_count += 1;
                    if self.dcas_warned_states.insert(state_id) {
                        log::warn!(
                            "DCAS (cmpxchg16b) unsupported in Rust interpreter at \
                             0x{:x} (state {}); falling back to Python VEX engine",
                            addr,
                            state_id
                        );
                    }
                }
                if reason.contains(VECRET_GSPTR_REASON) {
                    self.vecret_gsptr_fallback_count += 1;
                }
                ExplorationEvent::need_python_vex(
                    state_id,
                    *addr,
                    reason,
                    self.found_count(),
                    self.active_count(),
                    self.steps,
                )
            }
            CallbackReason::Error { message } => ExplorationEvent::error(
                message.clone(),
                self.found_count(),
                self.active_count(),
                self.steps,
            ),
        }
    }
}
