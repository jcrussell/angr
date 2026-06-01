//! Main exploration run loop.
//!
//! `RustExplorationManager::run_loop` drives the inner state-stepping loop:
//! pop a state from the active stash, evaluate find/avoid predicates, dispatch
//! SimProcedures (native fast path or Python fallback), step the interpreter,
//! handle deferred forks at find/avoid addresses, and route Python callbacks.
//!
//! The pyclass-facing `run` thin wrapper lives in `mod.rs` and just calls
//! `self.run_loop(py, n)`. PyO3 0.27.2 without `multiple-pymethods` only
//! permits a single `#[pymethods]` impl per class, so the body is extracted
//! here as a `pub(crate)` method on `RustExplorationManager`, mirroring the
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

impl RustExplorationManager {
    /// Inner body of the pymethods-exposed `run`. See `run` in `mod.rs`.
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
            let mut state = match self.sm.get_mut(STASH_ACTIVE).and_then(|s| {
                if self.use_lifo {
                    s.pop_back() // DFS: LIFO (most recent state first)
                } else {
                    s.pop_front() // BFS: FIFO (oldest state first)
                }
            }) {
                Some(s) => {
                    let sid = s.state_id();
                    self.current_stepping_state_id = Some(sid);
                    STEPPING_STATE_ID.with(|cell| cell.set(Some(sid)));
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
                    self.pending_callback = Some(PendingCallback::lightweight(
                        state,
                        CallbackReason::AvoidPredicate { addr: pc },
                    ));

                    return Ok(ExplorationEvent {
                        event_type: "need_callback".to_string(),
                        callback_reason: Some("avoid_predicate".to_string()),
                        callback_addr: Some(pc),
                        callback_state_id: Some(state_id),
                        found_count: self.found_count(),
                        active_count: self.active_count(),
                        steps_taken: self.steps,
                        callback_name: None,
                        callback_syscall_num: None,
                        callback_return_addr: None,
                        callback_num_args: None,
                        branch_true_target: None,
                        branch_false_target: None,
                        branch_condition_id: None,
                    });
                }
            }

            // Check avoid addresses (address-based, only when NOT using callable predicate)
            if self.avoid_addrs.contains(&pc) {
                self.push_or_drop_terminal(STASH_AVOID, state);
                continue;
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
                    self.pending_callback = Some(PendingCallback::lightweight(
                        state,
                        CallbackReason::FindPredicate { addr: pc },
                    ));

                    return Ok(ExplorationEvent {
                        event_type: "need_callback".to_string(),
                        callback_reason: Some("find_predicate".to_string()),
                        callback_addr: Some(pc),
                        callback_state_id: Some(state_id),
                        found_count: self.found_count(),
                        active_count: self.active_count(),
                        steps_taken: self.steps,
                        callback_name: None,
                        callback_syscall_num: None,
                        callback_return_addr: None,
                        callback_num_args: None,
                        branch_true_target: None,
                        branch_false_target: None,
                        branch_condition_id: None,
                    });
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
                        .or_insert_with(VecDeque::new)
                        .push_back(state);
                } else {
                    log::debug!("State at find address 0x{:x} is UNSAT, pruning", pc);
                    self.push_or_drop_terminal(STASH_PRUNED, state);
                }
                continue;
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
                    if !is_in_binary {
                        if let Some(native_proc) = self.native_procedures.get(&name) {
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
                                Ok(args) => match native_proc.call(&mut state, &args) {
                                    Ok(ret_val) => {
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
                                            continue;
                                        }

                                        // Set return value if present
                                        if let Some(rv) = ret_val {
                                            let ret_reg = self
                                                .environment
                                                .calling_convention
                                                .return_register();
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
                                            if let Some(sp) = state.get_sp().as_u64() {
                                                if let Ok(ret_bv) =
                                                    state.memory_load(sp, state.arch().bytes())
                                                {
                                                    if let Some(ret_addr) = ret_bv.as_u64() {
                                                        let ptr_size = state.arch().bytes() as u64;
                                                        state.set_sp(RustBV::concrete(
                                                            (sp + ptr_size) as u128,
                                                            state.arch().bits(),
                                                        ));
                                                        state.set_pc(ret_addr);
                                                    }
                                                }
                                            }
                                        }

                                        self.push_to_active_or_drop(state);
                                        continue;
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
                        }
                    } // if !is_in_binary

                    // Fall back to Python for SimProcedure execution
                    self.simprocedure_python_fallback_count += 1;
                    *self
                        .simprocedure_fallback_by_name
                        .entry(name.clone())
                        .or_insert(0) += 1;
                    let state_id = state.state_id();
                    let return_addr = self.get_return_addr(&state).unwrap_or(0);

                    // No deferred forks in run-loop path, so pre_callback_snapshot
                    // is unnecessary (it's only used as fork base for deferred forks).
                    // Use shared solver (O(1) Rc clone) instead of fork (~3-40ms Z3 clone).
                    let solver_ref = state.solver();
                    let shared_ctx = RustSolverContext::from_shared_sym_context(solver_ref.clone());

                    self.pending_callback = Some(PendingCallback::with_context(
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
                    ));

                    return Ok(ExplorationEvent::need_simprocedure(
                        state_id,
                        pc,
                        name,
                        num_args,
                        return_addr,
                        self.found_count(),
                        self.active_count(),
                        self.steps,
                    ));
                }
            }

            // Step the state, passing the skip_hook_addr if we just skipped
            let skip_addr_for_step = if should_skip_hook { Some(pc) } else { None };
            match self.step_state_with_skip(py, &callbacks, state, skip_addr_for_step) {
                Ok(successors) => {
                    // Add successors to appropriate stashes, checking find/avoid
                    for successor in successors {
                        let spc = successor.pc();
                        if self.find_addrs.contains(&spc) {
                            if self.constraint_solver.lazy_solves || successor.satisfiable() {
                                self.sm
                                    .stashes_mut()
                                    .entry(STASH_FOUND.to_string())
                                    .or_insert_with(VecDeque::new)
                                    .push_back(successor);
                            }
                        } else if self.avoid_addrs.contains(&spc) {
                            self.push_or_drop_terminal(STASH_AVOID, successor);
                        } else {
                            self.push_to_active_or_drop(successor);
                        }
                    }
                }
                Err(StepError::NeedCallback(pending)) => {
                    // Check if the callback address is a find/avoid address
                    // (these were added as interpreter hooks to stop execution)
                    let callback_addr = match &pending.reason {
                        CallbackReason::SimProcedure { addr, .. } => Some(*addr),
                        _ => None,
                    };
                    if let Some(addr) = callback_addr {
                        if self.find_addrs.contains(&addr) || self.avoid_addrs.contains(&addr) {
                            let is_find = self.find_addrs.contains(&addr);

                            // Process deferred forks BEFORE handling the find/avoid state.
                            // These represent unexplored branches that diverged before
                            // reaching the find/avoid address and must not be dropped.
                            let fork_base = pending
                                .pre_callback_snapshot
                                .unwrap_or_else(|| pending.state.fork());
                            let original_state_id = pending.state.state_id();
                            let root_state_id = self
                                .sm
                                .roots()
                                .get(&original_state_id)
                                .copied()
                                .unwrap_or(original_state_id);

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
                                            let ctx: &SymContext = &*solver_ref.borrow();
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
                                    let forked = if let Some(snapshot) =
                                        snapshots.remove(&fork.condition_id)
                                    {
                                        let mut f = fork_base.fork_from_snapshot(snapshot);
                                        if fork.path_taken {
                                            f.solver().borrow().assume_false(cond);
                                        } else {
                                            f.solver().borrow().assume_true(cond);
                                        }
                                        f.set_pc(fork.unexplored_target);
                                        f
                                    } else if fork.path_taken {
                                        let mut f = fork_base.fork_false(cond);
                                        f.set_pc(fork.unexplored_target);
                                        f
                                    } else {
                                        let mut f = fork_base.fork_true(cond);
                                        f.set_pc(fork.unexplored_target);
                                        f
                                    };
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
                                self.profiling.accumulated_stats.deferred_fork_count +=
                                    cb_fork_total;
                            }

                            // Now handle the main state
                            if is_find {
                                if self.constraint_solver.lazy_solves || pending.state.satisfiable()
                                {
                                    self.sm
                                        .stashes_mut()
                                        .entry(STASH_FOUND.to_string())
                                        .or_insert_with(VecDeque::new)
                                        .push_back(pending.state);
                                } else {
                                    log::debug!(
                                        "State at find address 0x{:x} is UNSAT, pruning",
                                        addr
                                    );
                                    self.push_or_drop_terminal(STASH_PRUNED, pending.state);
                                }
                            } else {
                                self.push_or_drop_terminal(STASH_AVOID, pending.state);
                            }
                            continue;
                        }
                    }

                    // Need Python callback
                    let state_id = pending.state.state_id();
                    let event = match &pending.reason {
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
                        _ => ExplorationEvent::error(
                            "unhandled callback reason".to_string(),
                            self.found_count(),
                            self.active_count(),
                            self.steps,
                        ),
                    };

                    self.pending_callback = Some(pending);
                    return Ok(event);
                }
                Err(StepError::Deadended(state)) => {
                    self.push_or_drop_terminal(STASH_DEADENDED, state);
                }
                Err(StepError::Error(state, message)) => {
                    let pc = state.pc();
                    let state_id = state.state_id();
                    self.errors.push((pc, message, state_id));
                    self.sm
                        .stashes_mut()
                        .entry(STASH_ERRORED.to_string())
                        .or_insert_with(VecDeque::new)
                        .push_back(state);
                }
                Err(StepError::Unconstrained(state)) => {
                    // State has too many symbolic jump targets - move to unconstrained stash
                    log::debug!("State {} moved to unconstrained stash", state.state_id());
                    self.sm.push_or_drop_terminal(STASH_UNCONSTRAINED, state);
                }
            }

            self.steps += 1;

            // Apply native uniqueness filter if enabled
            self.apply_uniqueness_filter();
            // Apply native techniques (LengthLimiter, Timeout, LoopBound)
            self.apply_native_techniques();
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
}
