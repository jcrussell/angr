use super::*;
use crate::arch::RegisterFile;
use crate::interpreter::{BLOCK_CACHE_CAPACITY, BranchSnapshot};
use crate::memory::SymbolicMemory;
use crate::state::{CallStackEntry, HistoryEntry};
use crate::vex::IRSB;
use lru::LruCache;
use std::num::NonZeroUsize;

/// Error during state stepping.
///
/// `Err` is used as a non-error control-flow signal: `Deadended` /
/// `Unconstrained` / `Error` carry the terminated `RustSimState` back
/// to the run loop for stash placement, and `NeedCallback` shuttles a
/// `PendingCallback` to the resume path. Boxing the inner state would
/// add a heap allocation on every step termination (`large_enum_variant`)
/// or every `?` propagation (`result_large_err`); the variants are
/// intentionally inline. See iter 59/60 handoff: clippy's "fix" is the
/// wrong call here — the design is the size.
#[allow(clippy::large_enum_variant)]
pub(crate) enum StepError {
    /// Need Python callback.
    NeedCallback(PendingCallback),
    /// State deadended (no successors).
    Deadended(RustSimState),
    /// Error during execution.
    Error(RustSimState, String),
    /// Unconstrained state - too many symbolic jump targets.
    Unconstrained(RustSimState),
}

/// Output of one interpreter run, packaged for the post-execution phase.
///
/// Replaces a 12-element tuple destructure that became unreadable as fields
/// were added. All fields are owned (taken from the interpreter before drop).
struct InterpreterStepResult {
    result: RunResult,
    deferred_forks: Vec<DeferredFork>,
    last_condition: Option<RustBV>,
    stored_conditions: FxHashMap<u64, RustBV>,
    fork_snapshots: FxHashMap<u64, BranchSnapshot>,
    new_registers: RegisterFile,
    new_pc: u64,
    new_call_stack: Vec<CallStackEntry>,
    new_detailed_history: Vec<HistoryEntry>,
    recovered_memory: Option<SymbolicMemory>,
    step_stats: ExecutionStats,
    updated_block_cache: LruCache<u64, Arc<IRSB>>,
    /// Set only when `state.keep_ip_symbolic()` was true and the interpreter
    /// concretized a symbolic default-exit next-pc. The manager writes this
    /// back to the state's IP register after `state.set_pc(new_pc)`.
    symbolic_ip_at_exit: Option<RustBV>,
}

// `StepError` carries an inline `RustSimState` (see enum doc above) so
// `Result<_, StepError>` is intentionally large. Every step function below
// uses Err for control flow, not failures — boxing would add allocs on the
// hot path. Suppress at the impl level rather than repeating the rationale
// per-function.
#[allow(clippy::result_large_err)]
impl RustExplorationManager {
    /// Step a state, optionally skipping a hook address.
    ///
    /// The skip_addr parameter is used for zero-length hooks: after the hook
    /// runs but returns to the same address, we skip adding that hook to the
    /// interpreter so the underlying instruction can execute.
    pub(crate) fn step_state_with_skip(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        mut state: RustSimState,
        skip_addr: Option<u64>,
    ) -> Result<Vec<RustSimState>, StepError> {
        let setup_start = if self.profiling.profiling_enabled {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let initial_pc = state.pc();

        // Run the VEX interpreter to its next event.
        let step = self.run_interpreter_step(
            py,
            callbacks,
            &mut state,
            initial_pc,
            skip_addr,
            setup_start,
        );

        // Restore the shared block cache (now populated with any newly-lifted blocks)
        self.environment.block_cache = step.updated_block_cache;

        // Accumulate profiling stats
        if self.profiling.profiling_enabled {
            let mut stats = step.step_stats;
            stats.step_count = 1;
            self.profiling.accumulated_stats.merge(&stats);
        }

        // Restore memory from interpreter back to state FIRST.
        // This must happen before any PendingCallback creation
        // because the state's memory was taken by set_rust_memory().
        if let Some(mem) = step.recovered_memory {
            state.replace_memory(mem);
        }

        // Update state from interpreter results
        // Restore registers (including symbolic values) from interpreter
        state.set_registers(step.new_registers);
        state.set_pc(step.new_pc);
        // KEEP_IP_SYMBOLIC: overwrite the IP register (just concretized by
        // set_pc above) with the original symbolic next-pc expression. The
        // `state.pc` u64 still points to the concretized address so the next
        // block lift drives from there, but the IP register reads as the
        // unpinned symbolic expression — matching Python's
        // `split_state.regs.ip = target` at engines/successors.py:328.
        if let Some(sym_ip) = step.symbolic_ip_at_exit {
            state.set_ip(sym_ip);
        }
        // Restore call stack and detailed history from interpreter
        state.set_call_stack(step.new_call_stack);
        state.set_detailed_history(step.new_detailed_history);

        // Add to history
        state.add_to_history(state.pc());

        let deferred_forks = step.deferred_forks;
        let last_condition = step.last_condition;
        let stored_conditions = step.stored_conditions;
        let fork_snapshots = step.fork_snapshots;

        // Process result
        match step.result {
            RunResult::MaxBlocks { pc }
            | RunResult::MaxDeferredForks { pc }
            | RunResult::BlockEnd { next_addr: pc, .. } => {
                self.handle_block_end(state, pc, deferred_forks, stored_conditions, fork_snapshots)
            }
            RunResult::Hook { addr } => {
                state.set_pc(addr);
                // P1 Fix: Add to history BEFORE callback so Python can access recent_bbl_addrs[-1]
                state.add_to_history(addr);
                // Only create pre-callback snapshot if deferred forks need it.
                // state.fork() clones the Z3 solver (~3-40ms), so skip when not needed.
                let hook_fork_start = if self.profiling.profiling_enabled {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let pre_callback_snapshot = if !deferred_forks.is_empty() {
                    Some(state.fork())
                } else {
                    None
                };
                // Use shared solver (O(1) Rc clone) instead of fork (~3-40ms Z3 clone)
                let solver_ref = state.solver();
                let shared_ctx = RustSolverContext::from_shared_sym_context(solver_ref.clone());
                if let Some(start) = hook_fork_start {
                    let fork_count = if pre_callback_snapshot.is_some() {
                        1u64
                    } else {
                        0u64
                    };
                    self.profiling.accumulated_stats.solver_fork_time_ns +=
                        start.elapsed().as_nanos() as u64;
                    self.profiling.accumulated_stats.solver_fork_count += fork_count;
                }
                // Return to Python for hook - store deferred forks for later processing
                Err(StepError::NeedCallback(PendingCallback::with_context(
                    state,
                    pre_callback_snapshot,
                    CallbackReason::SimProcedure {
                        addr,
                        name: "unknown".to_string(),
                        num_args: 0,
                        return_addr: 0,
                    },
                    "Ijk_Boring",
                    Some(shared_ctx),
                    deferred_forks,
                    stored_conditions,
                    fork_snapshots,
                )))
            }
            RunResult::SimProcedure {
                addr,
                name,
                num_args,
                return_addr,
            } => self.handle_simprocedure(
                state,
                addr,
                name,
                num_args,
                return_addr,
                deferred_forks,
                stored_conditions,
                fork_snapshots,
            ),
            RunResult::Syscall { num, pc } => {
                state.set_pc(pc);
                // P1 Fix: Add to history BEFORE callback so Python can access recent_bbl_addrs[-1]
                state.add_to_history(pc);

                // Try native syscall dispatch first. On success we skip the
                // Python `_handle_syscall_callback` round-trip entirely.
                //
                // angr-gffd: when `num` is `None` the syscall register is
                // symbolic. Skip the native registry entirely and force the
                // Python fallback below — Python's `_handle_syscall_callback`
                // calls `engine.process(state)` which reads the still-symbolic
                // register through `_resolve_syscall` and either enumerates
                // (default) or routes to the unknown-syscall stub
                // (NO_SYMBOLIC_SYSCALL_RESOLUTION). Dispatching to a native
                // handler at concrete `0` would silently invoke `read` on
                // amd64.
                let arch_name = state.arch().name();
                let native_handler = num.and_then(|n| self.native_syscalls.get(arch_name, n));
                if let Some(handler) = native_handler {
                    let n_args = handler.num_args();
                    let args = if n_args == 0 {
                        Ok(Vec::new())
                    } else {
                        self.extract_syscall_args(&state, n_args)
                    };
                    let Ok(args) = args else {
                        // Arg extraction failed (RegisterOverflow on a
                        // misconfigured handler). Skip the native fast path
                        // — falling through to the Python syscall callback
                        // below is safer than invoking the native handler
                        // with fabricated zeros.
                        log::debug!(
                            "Skipping native syscall (arg extraction failed): {:?}",
                            args.unwrap_err()
                        );
                        self.syscall_python_fallback_count += 1;
                        *self
                            .syscall_python_fallback_by_num
                            .entry(num.map(|n| n as i64).unwrap_or(-1))
                            .or_insert(0) += 1;
                        let pre_callback_snapshot = if !deferred_forks.is_empty() {
                            Some(state.fork())
                        } else {
                            None
                        };
                        let solver_ref = state.solver();
                        let shared_ctx =
                            RustSolverContext::from_shared_sym_context(solver_ref.clone());
                        return Err(StepError::NeedCallback(PendingCallback::with_context(
                            state,
                            pre_callback_snapshot,
                            CallbackReason::Syscall { num },
                            "Ijk_Sys_syscall",
                            Some(shared_ctx),
                            deferred_forks,
                            stored_conditions,
                            fork_snapshots,
                        )));
                    };
                    match handler.call(&mut state, &args) {
                        Ok(SyscallOutcome::Continue { ret }) => {
                            let ret_reg = self.environment.calling_convention.return_register();
                            let bits = state.arch().bits();
                            state.set_register_by_offset(
                                ret_reg,
                                RustBV::concrete(ret as u128, bits),
                            );
                            let mut successors = vec![state];
                            self.process_deferred_forks_into(
                                &mut successors,
                                deferred_forks,
                                &stored_conditions,
                                fork_snapshots,
                            );
                            return Ok(successors);
                        }
                        Ok(SyscallOutcome::ContinueSymbolic { ret }) => {
                            let ret_reg = self.environment.calling_convention.return_register();
                            state.set_register_by_offset(ret_reg, ret);
                            let mut successors = vec![state];
                            self.process_deferred_forks_into(
                                &mut successors,
                                deferred_forks,
                                &stored_conditions,
                                fork_snapshots,
                            );
                            return Ok(successors);
                        }
                        Ok(SyscallOutcome::Exit) => {
                            let mut successors = vec![state];
                            self.process_deferred_forks_into(
                                &mut successors,
                                deferred_forks,
                                &stored_conditions,
                                fork_snapshots,
                            );
                            let main_state = successors.remove(0);
                            self.push_or_drop_terminal(STASH_DEADENDED, main_state);
                            return Ok(successors);
                        }
                        Err(_) => {
                            // Fall through to Python callback path below.
                        }
                    }
                }

                // Falling through to Python syscall callback (no native handler
                // matched, or native handler returned Err).
                self.syscall_python_fallback_count += 1;
                *self
                    .syscall_python_fallback_by_num
                    .entry(num.map(|n| n as i64).unwrap_or(-1))
                    .or_insert(0) += 1;
                // Only snapshot if deferred forks need it
                let pre_callback_snapshot = if !deferred_forks.is_empty() {
                    Some(state.fork())
                } else {
                    None
                };
                // Use shared solver (O(1) Rc clone) instead of fork
                let solver_ref = state.solver();
                let shared_ctx = RustSolverContext::from_shared_sym_context(solver_ref.clone());
                Err(StepError::NeedCallback(PendingCallback::with_context(
                    state,
                    pre_callback_snapshot,
                    CallbackReason::Syscall { num },
                    "Ijk_Sys_syscall",
                    Some(shared_ctx),
                    deferred_forks,
                    stored_conditions,
                    fork_snapshots,
                )))
            }
            RunResult::SymbolicBranch {
                condition_id,
                true_target,
                false_target,
            } => {
                // Return to Python for proper state forking with constraints.
                // No pre_callback_snapshot or solver_ctx fork needed here —
                // resume_after_symbolic_branch forks from pending.state directly.
                // Skipping these 2 unnecessary state forks eliminates O(n)
                // constraint replay per symbolic branch.
                let mut branch_conditions = stored_conditions;
                if let Some(cond) = last_condition {
                    branch_conditions.insert(condition_id, cond);
                }

                Err(StepError::NeedCallback(PendingCallback::with_context(
                    state,
                    None,
                    CallbackReason::SymbolicBranch {
                        condition_id,
                        true_target,
                        false_target,
                    },
                    "Ijk_Boring",
                    None,
                    deferred_forks,
                    branch_conditions,
                    fork_snapshots,
                )))
            }
            RunResult::Error { message, addr } => {
                state.set_pc(addr);
                // Treat lift errors at unmapped addresses as deadends, not errors.
                // This matches Python engine behavior where states that reach
                // invalid code addresses (e.g., 0x0 after exit) are deadended.
                if message.contains("No bytes in memory") || message.contains("lift") || addr == 0 {
                    Err(StepError::Deadended(state))
                } else {
                    Err(StepError::Error(state, message))
                }
            }
            RunResult::NeedPythonVEX { addr, reason } => {
                // Rust interpreter hit an unsupported operation (CAS, dirty call, SIMD, etc.)
                // Fall back to Python's SimEngineVEX to handle this block
                log::debug!("VEX fallback at 0x{:x}: {}", addr, reason);
                state.set_pc(addr);
                Err(StepError::NeedCallback(PendingCallback::with_context(
                    state,
                    None,
                    CallbackReason::PythonVEXFallback { addr, reason },
                    "Ijk_Boring",
                    None,
                    deferred_forks,
                    stored_conditions,
                    fork_snapshots,
                )))
            }
            RunResult::NeedLift { addr } => {
                // This shouldn't happen if callbacks are properly set
                state.set_pc(addr);
                Err(StepError::Error(
                    state,
                    format!("need lift at 0x{:x}", addr),
                ))
            }
            RunResult::SymbolicJumpTarget {
                targets,
                condition_id,
                jumpkind: _,
            } => self.handle_symbolic_jump_target(
                state,
                targets,
                condition_id,
                deferred_forks,
                stored_conditions,
                fork_snapshots,
            ),
            RunResult::UnconstrainedJump {
                min_target: _,
                max_target: _,
                limit: _,
                jumpkind: _,
            } => {
                // Too many symbolic jump targets - move to unconstrained stash
                Err(StepError::Unconstrained(state))
            }
            RunResult::UnmodeledCall {
                addr,
                return_addr,
                symbol_name,
            } => self.handle_unmodeled_call(
                py,
                callbacks,
                state,
                addr,
                return_addr,
                symbol_name,
                deferred_forks,
                stored_conditions,
                fork_snapshots,
            ),
        }
    }

    /// Handle MaxBlocks / MaxDeferredForks / BlockEnd: continue at `pc` and
    /// process accumulated deferred forks with constraint handling and UNSAT
    /// pruning. Profiling instrumentation here is intentionally distinct from
    /// `process_deferred_forks_into` (per-fork sat/fork timers, deferred-fork
    /// total timer).
    fn handle_block_end(
        &mut self,
        mut state: RustSimState,
        pc: u64,
        deferred_forks: Vec<DeferredFork>,
        stored_conditions: FxHashMap<u64, RustBV>,
        mut fork_snapshots: FxHashMap<u64, BranchSnapshot>,
    ) -> Result<Vec<RustSimState>, StepError> {
        state.set_pc(pc);

        // Track root state ID for lineage
        let original_state_id = state.state_id();
        let root_state_id = self
            .sm
            .roots()
            .get(&original_state_id)
            .copied()
            .unwrap_or(original_state_id);

        // Process deferred forks with proper constraint handling
        // P13: Track UNSAT states for pruning
        let mut successors = vec![state];
        let mut pruned_states = Vec::new();
        let deferred_fork_start = if self.profiling.profiling_enabled {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let deferred_fork_total = deferred_forks.len() as u64;

        for fork in deferred_forks {
            // Look up the condition for this deferred fork
            if let Some(condition) = stored_conditions.get(&fork.condition_id) {
                // Add the taken-path constraint to the main state.
                // This was NOT done during block execution to avoid
                // polluting subsequent feasibility checks within the
                // same IRSB.
                if fork.path_taken {
                    successors[0].solver().borrow().assume_true(condition);
                } else {
                    successors[0].solver().borrow().assume_false(condition);
                }

                // Create forked state for the unexplored path.
                // Use solver snapshot (from before branch constraint) if available
                // to avoid inheriting the taken-path constraint (which would make
                // the opposite constraint UNSAT).
                let fork_start = if self.profiling.profiling_enabled {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let forked = if let Some(snapshot) = fork_snapshots.remove(&fork.condition_id) {
                    let mut f = successors[0].fork_from_snapshot(snapshot);
                    if fork.path_taken {
                        f.solver().borrow().assume_false(condition);
                    } else {
                        f.solver().borrow().assume_true(condition);
                    }
                    f.set_pc(fork.unexplored_target);
                    f
                } else if fork.path_taken {
                    let mut f = successors[0].fork_false(condition);
                    f.set_pc(fork.unexplored_target);
                    f
                } else {
                    let mut f = successors[0].fork_true(condition);
                    f.set_pc(fork.unexplored_target);
                    f
                };
                if let Some(start) = fork_start {
                    self.profiling.accumulated_stats.solver_fork_time_ns +=
                        start.elapsed().as_nanos() as u64;
                    self.profiling.accumulated_stats.solver_fork_count += 1;
                }
                // Track root state ID for this forked state
                self.sm.set_root(forked.state_id(), root_state_id);

                // P13: Check satisfiability before adding to successors
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
                    successors.push(forked);
                } else {
                    if let Some(start) = sat_start {
                        self.profiling.accumulated_stats.solver_sat_time_ns +=
                            start.elapsed().as_nanos() as u64;
                        self.profiling.accumulated_stats.solver_sat_count += 1;
                    }
                    log::debug!(
                        "P13: Deferred fork at 0x{:x} is UNSAT, will be pruned",
                        fork.unexplored_target
                    );
                    pruned_states.push(forked);
                }
            } else {
                // P15: Create conservative fork to explore the path even without condition
                log::warn!(
                    "P15: Missing condition for deferred fork at 0x{:x} (condition_id={}). \
                     Creating conservative fork.",
                    fork.branch_addr,
                    fork.condition_id
                );
                let mut forked = successors[0].fork();
                forked.set_pc(fork.unexplored_target);
                self.sm.set_root(forked.state_id(), root_state_id);

                // P13: Still check satisfiability
                if self.constraint_solver.lazy_solves || forked.satisfiable() {
                    successors.push(forked);
                } else {
                    log::debug!(
                        "P13: Unconstrained fork at 0x{:x} is UNSAT, will be pruned",
                        fork.unexplored_target
                    );
                    pruned_states.push(forked);
                }
            }
        }
        if let Some(start) = deferred_fork_start {
            self.profiling.accumulated_stats.deferred_fork_time_ns +=
                start.elapsed().as_nanos() as u64;
            self.profiling.accumulated_stats.deferred_fork_count += deferred_fork_total;
        }

        // Add pruned states to pruned stash
        for s in pruned_states {
            self.push_or_drop_terminal(STASH_PRUNED, s);
        }

        Ok(successors)
    }

    /// Handle SimProcedure: try native first, fall back to Python callback.
    /// Native success continues at the return address with deferred forks
    /// processed via the unified path; Python fallback returns a NeedCallback.
    #[allow(clippy::too_many_arguments)]
    fn handle_simprocedure(
        &mut self,
        mut state: RustSimState,
        addr: u64,
        name: String,
        num_args: usize,
        return_addr: u64,
        deferred_forks: Vec<DeferredFork>,
        stored_conditions: FxHashMap<u64, RustBV>,
        fork_snapshots: FxHashMap<u64, BranchSnapshot>,
    ) -> Result<Vec<RustSimState>, StepError> {
        // Try native procedure first — avoids Python callback overhead.
        // Skip native for addresses inside the binary — these are user-placed
        // hooks where the Python SimProcedure should always run (the user hooked
        // a specific function for a reason, e.g., hooking strings_not_equal with strcmp).
        let is_in_binary = self
            .environment
            .binary_regions
            .iter()
            .any(|(base, data)| addr >= *base && addr < *base + data.len() as u64);
        // Result of native execution: None = fall back to Python, Some(bool) =
        // succeeded with no_return flag indicating whether to deadend the main state.
        let native_no_return: Option<bool> = if !is_in_binary {
            if let Some(native_proc) = self.native_procedures.get(&name) {
                let proc_no_return = native_proc.no_return();
                match self.extract_procedure_args(&state, num_args) {
                    Err(e) => {
                        // SP symbolic / stack unmapped — silently zero-padding
                        // here would hand the native handler fabricated zeros
                        // and mask the underlying stack-setup bug. Skip the
                        // native fast path; `None` falls through to the
                        // Python SimProcedure callback at the bottom of this
                        // function.
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
                        None
                    }
                    Ok(args) => match native_proc.call(&mut state, &args) {
                        Ok(ret_val) => {
                            self.profiling.native_proc_stats.native_calls += 1;
                            *self
                                .profiling
                                .native_proc_stats
                                .call_counts
                                .entry(name.clone())
                                .or_insert(0) += 1;

                            if !proc_no_return {
                                if let Some(rv) = ret_val {
                                    let ret_reg =
                                        self.environment.calling_convention.return_register();
                                    state.set_register_by_offset(ret_reg, rv);
                                }

                                // Set PC to return address and pop stack
                                state.set_pc(return_addr);
                                let sp = state.get_sp().as_u64().unwrap_or(0);
                                let ptr_size = state.arch().bytes() as u64;
                                state.set_sp(RustBV::concrete(
                                    (sp + ptr_size) as u128,
                                    state.arch().bits(),
                                ));
                            }
                            Some(proc_no_return)
                        }
                        Err(e) => {
                            self.profiling.native_proc_stats.python_fallbacks += 1;
                            let bucket = match e {
                                ProcedureError::SymbolicArgument(_) => {
                                    &mut self.profiling.native_proc_stats.symbolic_fallbacks_by_name
                                }
                                ProcedureError::NotImplemented => {
                                    &mut self
                                        .profiling
                                        .native_proc_stats
                                        .not_implemented_fallbacks_by_name
                                }
                                _ => &mut self.profiling.native_proc_stats.other_fallbacks_by_name,
                            };
                            *bucket.entry(name.clone()).or_insert(0) += 1;
                            None
                        }
                    },
                }
            } else {
                None
            }
        } else {
            None
        };

        if let Some(no_return) = native_no_return {
            // Unified deferred-fork handling: identical semantics to
            // MaxBlocks/BlockEnd (snapshot-based forks restore solver
            // state from the branch point; UNSAT forks go to STASH_PRUNED).
            let mut successors = vec![state];
            self.process_deferred_forks_into(
                &mut successors,
                deferred_forks,
                &stored_conditions,
                fork_snapshots,
            );
            if no_return {
                // For no-return procedures (exit/abort): the main state must
                // not continue at the call's return address (which would re-enter
                // the caller and loop). Deadend it. Deferred forks (already in
                // successors[1..]) are kept so unexplored branches still run.
                let main_state = successors.remove(0);
                self.push_or_drop_terminal(STASH_DEADENDED, main_state);
            }
            Ok(successors)
        } else {
            // Fall through to Python callback
            self.simprocedure_python_fallback_count += 1;
            *self
                .simprocedure_fallback_by_name
                .entry(name.clone())
                .or_insert(0) += 1;
            state.set_pc(addr);
            state.add_to_history(addr);
            // Only snapshot if deferred forks need it
            let pre_callback_snapshot = if !deferred_forks.is_empty() {
                Some(state.fork())
            } else {
                None
            };
            // Use shared solver (O(1) Rc clone) instead of fork
            let solver_ref = state.solver();
            let shared_ctx = RustSolverContext::from_shared_sym_context(solver_ref.clone());
            Err(StepError::NeedCallback(PendingCallback::with_context(
                state,
                pre_callback_snapshot,
                CallbackReason::SimProcedure {
                    addr,
                    name,
                    num_args,
                    return_addr,
                },
                "Ijk_Call",
                Some(shared_ctx),
                deferred_forks,
                stored_conditions,
                fork_snapshots,
            )))
        }
    }

    /// Handle SymbolicJumpTarget: a symbolic jump concretized to a bounded
    /// set of addresses. Single target adds a constraint and continues; multiple
    /// targets fork from an unconstrained base so each fork only carries its
    /// own target constraint.
    fn handle_symbolic_jump_target(
        &mut self,
        state: RustSimState,
        targets: Vec<u64>,
        condition_id: u64,
        deferred_forks: Vec<DeferredFork>,
        stored_conditions: FxHashMap<u64, RustBV>,
        fork_snapshots: FxHashMap<u64, BranchSnapshot>,
    ) -> Result<Vec<RustSimState>, StepError> {
        // Look up the condition for constraint addition
        let target_expr = stored_conditions.get(&condition_id).cloned();

        if targets.is_empty() {
            // No targets - deadended
            return Err(StepError::Deadended(state));
        }

        // KEEP_IP_SYMBOLIC: mirror engines/successors.py:326-331 — skip the
        // per-fork `add_constraints(cond)` narrowing and leave each fork's
        // IP register holding the symbolic `target` expression. The concrete
        // `addr` still drives the next block lift via `state.pc`.
        let keep_ip_symbolic = state.keep_ip_symbolic();

        if targets.len() == 1 {
            // Single target - just continue
            let mut state = state;
            let addr = targets[0];
            if let Some(ref expr) = target_expr {
                if keep_ip_symbolic {
                    state.set_pc(addr);
                    state.set_ip(expr.clone());
                } else {
                    // Add constraint: target_expr == addr
                    let concrete = RustBV::concrete(addr as u128, expr.width());
                    let constraint = expr.eq(&concrete, &state.solver().borrow());
                    state.add_constraint(constraint);
                    state.set_pc(addr);
                }
            } else {
                state.set_pc(addr);
            }
            let mut successors = vec![state];
            self.process_deferred_forks_into(
                &mut successors,
                deferred_forks,
                &stored_conditions,
                fork_snapshots,
            );
            return Ok(successors);
        }

        // Multiple targets - fork for each from the UNCONSTRAINED original
        // CRITICAL: Save unconstrained base state BEFORE adding any target constraints
        // This ensures each fork only has its own target constraint, not all previous ones
        let base_state = state.fork(); // Save unconstrained clone

        // Track root state ID for lineage
        let original_state_id = state.state_id();
        let root_state_id = self
            .sm
            .roots()
            .get(&original_state_id)
            .copied()
            .unwrap_or(original_state_id);

        let mut successors = Vec::with_capacity(targets.len());

        // Handle first target - use the original state (moved here)
        let first_addr = targets[0];
        let mut first_state = state; // Move state into first_state
        if let Some(ref expr) = target_expr {
            if keep_ip_symbolic {
                first_state.set_pc(first_addr);
                first_state.set_ip(expr.clone());
            } else {
                let concrete = RustBV::concrete(first_addr as u128, expr.width());
                let constraint = expr.eq(&concrete, &first_state.solver().borrow());
                first_state.add_constraint(constraint);
                first_state.set_pc(first_addr);
            }
        } else {
            first_state.set_pc(first_addr);
        }
        successors.push(first_state);

        // Handle remaining targets - fork from unconstrained base
        for &addr in targets.iter().skip(1) {
            let mut forked = base_state.fork();

            // Add constraint: target_expr == addr (only this target's constraint)
            if let Some(ref expr) = target_expr {
                if keep_ip_symbolic {
                    forked.set_pc(addr);
                    forked.set_ip(expr.clone());
                } else {
                    let concrete = RustBV::concrete(addr as u128, expr.width());
                    let constraint = expr.eq(&concrete, &forked.solver().borrow());
                    forked.add_constraint(constraint);
                    forked.set_pc(addr);
                }
            } else {
                forked.set_pc(addr);
            }
            // Track root state ID for this forked state
            self.sm.set_root(forked.state_id(), root_state_id);
            successors.push(forked);
        }

        // Process any deferred forks accumulated during execution
        self.process_deferred_forks_into(
            &mut successors,
            deferred_forks,
            &stored_conditions,
            fork_snapshots,
        );

        Ok(successors)
    }

    /// Handle UnmodeledCall: try to resolve via Python callback. Resolved
    /// functions are registered as SimProcedures and dispatched via callback;
    /// unresolved calls use P21 generic skip (set return register to 0,
    /// continue at return address) instead of deadending.
    #[allow(clippy::too_many_arguments)]
    fn handle_unmodeled_call(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        mut state: RustSimState,
        addr: u64,
        return_addr: u64,
        symbol_name: Option<String>,
        deferred_forks: Vec<DeferredFork>,
        stored_conditions: FxHashMap<u64, RustBV>,
        fork_snapshots: FxHashMap<u64, BranchSnapshot>,
    ) -> Result<Vec<RustSimState>, StepError> {
        // Unhooked CALL target - try to resolve via Python callback
        state.set_pc(addr);
        // P1 Fix: Add to history BEFORE callback so Python can access recent_bbl_addrs[-1]
        state.add_to_history(addr);

        // Try to resolve the function via callback
        if callbacks.has_resolve_function() {
            match callbacks.call_resolve_function(py, addr, symbol_name.as_deref()) {
                Ok(Some((name, num_args, no_return))) => {
                    // Function resolved! Register it and return to Python for execution
                    log::debug!(
                        "Resolved unmodeled call at 0x{:x} -> {} (args={}, no_return={})",
                        addr,
                        name,
                        num_args,
                        no_return
                    );

                    // Register the procedure so future calls are hooked
                    self.hooks.insert(addr);
                    self.simprocedures
                        .insert(addr, (name.clone(), num_args, no_return));

                    // Only snapshot if deferred forks need it
                    let pre_callback_snapshot = if !deferred_forks.is_empty() {
                        Some(state.fork())
                    } else {
                        None
                    };
                    // Use shared solver (O(1) Rc clone) instead of fork
                    let solver_ref = state.solver();
                    let shared_ctx = RustSolverContext::from_shared_sym_context(solver_ref.clone());

                    // Return to Python for SimProcedure execution
                    Err(StepError::NeedCallback(PendingCallback::with_context(
                        state,
                        pre_callback_snapshot,
                        CallbackReason::SimProcedure {
                            addr,
                            name,
                            num_args,
                            return_addr,
                        },
                        "Ijk_Call",
                        Some(shared_ctx),
                        deferred_forks,
                        stored_conditions,
                        fork_snapshots,
                    )))
                }
                Ok(None) => {
                    // P21: Function could not be resolved - use generic skip instead of deadending
                    self.unmodeled_call_generic_skip(
                        state,
                        addr,
                        return_addr,
                        deferred_forks,
                        stored_conditions,
                        fork_snapshots,
                    )
                }
                Err(e) => {
                    // Callback error - treat as execution error
                    log::warn!("resolve_function callback error at 0x{:x}: {}", addr, e);
                    Err(StepError::Error(
                        state,
                        format!("resolve_function error: {}", e),
                    ))
                }
            }
        } else {
            // P21: No resolve_function callback - use generic skip instead of deadending
            self.unmodeled_call_generic_skip(
                state,
                addr,
                return_addr,
                deferred_forks,
                stored_conditions,
                fork_snapshots,
            )
        }
    }

    /// P21 generic skip for unmodeled calls: set return register to 0,
    /// continue at return_addr, and process any deferred forks. Used both
    /// when resolve_function returns None and when no callback is registered.
    fn unmodeled_call_generic_skip(
        &mut self,
        mut state: RustSimState,
        addr: u64,
        return_addr: u64,
        deferred_forks: Vec<DeferredFork>,
        stored_conditions: FxHashMap<u64, RustBV>,
        fork_snapshots: FxHashMap<u64, BranchSnapshot>,
    ) -> Result<Vec<RustSimState>, StepError> {
        log::debug!(
            "P21: Unmodeled call at 0x{:x} - generic skip (ret=0) to return_addr=0x{:x}",
            addr,
            return_addr
        );

        // Set return register to 0 (symbolic unconstrained would be better but
        // concrete 0 is simpler and often sufficient)
        let ret_reg_offset = self.environment.calling_convention.return_register();
        let ptr_size = self.environment.calling_convention.pointer_size();
        let zero_val = RustBV::zero(ptr_size * 8);
        state.set_register_by_offset(ret_reg_offset, zero_val);

        // Continue at return address
        state.set_pc(return_addr);

        // Process any deferred forks from the interpreter step
        let mut successors = vec![state];
        self.process_deferred_forks_into(
            &mut successors,
            deferred_forks,
            &stored_conditions,
            fork_snapshots,
        );
        Ok(successors)
    }

    /// Run the VEX interpreter for one step and recover all owned state from it.
    ///
    /// This wraps the borrow scope around the state's solver: a `VEXInterpreter`
    /// is constructed against the borrowed solver, run until its next event, then
    /// fully drained (registers, memory, history, block cache, profiling stats)
    /// before being dropped at scope end.
    fn run_interpreter_step(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        state: &mut RustSimState,
        initial_pc: u64,
        skip_addr: Option<u64>,
        setup_start: Option<std::time::Instant>,
    ) -> InterpreterStepResult {
        let solver_rc = state.solver().clone();
        let solver_ref = solver_rc.borrow();

        // Create interpreter with the state's solver
        let mut interp = VEXInterpreter::with_config(
            self.environment.vex_arch,
            &solver_ref,
            self.exec_config.clone(),
        );

        // Propagate lazy_solves to skip Z3 feasibility checks
        interp.lazy_solves = self.constraint_solver.lazy_solves;
        // Propagate NO_IP_CONCRETIZATION from the state. Unlike lazy_solves
        // which is a manager-level flag, this is a per-state SimOption.
        interp.no_ip_concretization = state.no_ip_concretization();
        // NO_SYMBOLIC_JUMP_RESOLUTION sibling — same routing, applied at the
        // same short-circuit in eval_next_addr_concretized.
        interp.no_symbolic_jump_resolution = state.no_symbolic_jump_resolution();
        // KEEP_IP_SYMBOLIC: per-state SimOption that tells eval_next_addr_concretized
        // to stash the original symbolic next-pc expression (for restore via
        // set_ip after the manager's set_pc) and to skip the
        // `assume_true(next_val == addr)` narrowing constraint.
        interp.keep_ip_symbolic = state.keep_ip_symbolic();
        interp.set_profiling(self.profiling.profiling_enabled);
        // Propagate concretization strategy config
        interp.set_concretizer(self.memory_config.concretizer_config.clone());
        // Propagate VEX optimization level settings
        interp.vex_opt_level = self.memory_config.vex_opt_level;
        // Take a fresh Arc snapshot of the manager's overrides; interp will
        // share until a setter mutates (none do during step execution).
        interp.vex_opt_level_overrides =
            Arc::new(self.memory_config.vex_opt_level_overrides.clone());

        // Copy state registers to interpreter (including symbolic values)
        interp.registers = state.registers().fork();
        interp.set_pc(initial_pc);
        // Forward state_id so inspect dispatch sites can identify which
        // state owns the firing event (angr-uq4n.3/.4).
        interp.current_state_id = state.state_id() as i64;
        // Transfer call stack and detailed history to interpreter
        interp.call_stack = state.call_stack().to_vec();
        interp.detailed_history = state.detailed_history().to_vec();

        // Set up hooks, skipping the one we just processed (for zero-length hooks)
        for &addr in &self.hooks {
            if Some(addr) != skip_addr {
                interp.add_hook(addr);
            }
        }

        // Register SimProcedures, also skipping the one we just processed
        for (addr, (name, num_args, no_return)) in &self.simprocedures {
            if Some(*addr) != skip_addr {
                interp.register_simprocedure(*addr, name.clone(), *num_args, *no_return);
            }
        }

        // Add find/avoid addresses as hooks so the interpreter stops there
        for &addr in &self.find_addrs {
            interp.add_hook(addr);
        }
        for &addr in &self.avoid_addrs {
            interp.add_hook(addr);
        }

        // Copy binary regions for code lifting (O(1) Arc clone per region)
        for (base, data) in &self.environment.binary_regions {
            interp.add_concrete_memory_shared(*base, Arc::clone(data));
        }

        // Transfer state's SymbolicMemory into the interpreter.
        // This makes Rust the source of truth for all memory during
        // VEX execution. Loads/stores go to SymbolicMemory directly
        // instead of calling back to Python.
        interp.set_rust_memory(state.take_memory());

        // Share the exploration-level block cache with the interpreter
        // so lifted blocks persist across steps (avoids re-lifting).
        // Swap exploration's populated cache into interp, stash interp's empty one.
        let interp_empty_cache = interp.swap_block_cache(std::mem::replace(
            &mut self.environment.block_cache,
            LruCache::new(
                NonZeroUsize::new(BLOCK_CACHE_CAPACITY).expect("BLOCK_CACHE_CAPACITY is non-zero"),
            ),
        ));
        // interp now has the exploration's cache; self.environment.block_cache is a temporary empty placeholder
        let _ = interp_empty_cache; // drop the empty cache

        // Record setup time before execution
        if let Some(start) = setup_start {
            interp.stats_mut().step_setup_time_ns += start.elapsed().as_nanos() as u64;
        }

        // Run until event.
        // When callable predicates are active (find_needs_python), limit to
        // 1 block so the run loop can check the predicate at each PC.
        // Otherwise the interpreter would execute many blocks, skipping past
        // the target address without the predicate ever seeing it.
        let steps_limit = if self.find_needs_python || self.avoid_needs_python {
            1
        } else {
            self.max_steps_per_run
        };
        let (result, _blocks_executed, deferred_forks) =
            interp.run_until_event(py, callbacks, steps_limit);

        // Drain interpreter state into owned values before drop.
        let last_condition = interp.take_last_branch_condition();
        let stored_conditions = interp.take_stored_conditions();
        let fork_snapshots = interp.take_fork_snapshots();
        let symbolic_ip_at_exit = interp.take_symbolic_ip_at_exit();
        let new_registers = interp.registers.fork();
        let new_pc = interp.get_pc();
        let new_call_stack = std::mem::take(&mut interp.call_stack);
        let new_detailed_history = std::mem::take(&mut interp.detailed_history);

        // Flush any remaining pending stores to rust_memory before recovery.
        interp.flush_stores_to_rust_memory();
        let recovered_memory = interp.take_rust_memory();

        // Return shared block cache to exploration before interpreter is dropped
        let updated_block_cache = interp.swap_block_cache(LruCache::new(
            NonZeroUsize::new(BLOCK_CACHE_CAPACITY).expect("BLOCK_CACHE_CAPACITY is non-zero"),
        ));

        let step_stats = interp.take_stats();

        InterpreterStepResult {
            result,
            deferred_forks,
            last_condition,
            stored_conditions,
            fork_snapshots,
            new_registers,
            new_pc,
            new_call_stack,
            new_detailed_history,
            recovered_memory,
            step_stats,
            updated_block_cache,
            symbolic_ip_at_exit,
        }
    }

    /// Process deferred forks and add the resulting forked states to the successor list.
    /// This is used by code paths (like P21 generic skip) that don't go through
    /// the main MaxBlocks/BlockEnd deferred fork processing.
    pub(crate) fn process_deferred_forks_into(
        &mut self,
        successors: &mut Vec<RustSimState>,
        deferred_forks: Vec<DeferredFork>,
        stored_conditions: &FxHashMap<u64, RustBV>,
        mut fork_snapshots: FxHashMap<u64, crate::interpreter::BranchSnapshot>,
    ) {
        if deferred_forks.is_empty() {
            return;
        }

        let root_state_id = {
            let original_state_id = successors[0].state_id();
            self.sm
                .roots()
                .get(&original_state_id)
                .copied()
                .unwrap_or(original_state_id)
        };

        for fork in &deferred_forks {
            if let Some(condition) = stored_conditions.get(&fork.condition_id) {
                // Add the taken-path constraint to the main state
                if fork.path_taken {
                    successors[0].solver().borrow().assume_true(condition);
                } else {
                    successors[0].solver().borrow().assume_false(condition);
                }

                // Create forked state for the unexplored path
                let forked = if let Some(snapshot) = fork_snapshots.remove(&fork.condition_id) {
                    let mut f = successors[0].fork_from_snapshot(snapshot);
                    if fork.path_taken {
                        f.solver().borrow().assume_false(condition);
                    } else {
                        f.solver().borrow().assume_true(condition);
                    }
                    f.set_pc(fork.unexplored_target);
                    f
                } else if fork.path_taken {
                    let mut f = successors[0].fork_false(condition);
                    f.set_pc(fork.unexplored_target);
                    f
                } else {
                    let mut f = successors[0].fork_true(condition);
                    f.set_pc(fork.unexplored_target);
                    f
                };

                self.sm.set_root(forked.state_id(), root_state_id);

                if self.constraint_solver.lazy_solves || forked.satisfiable() {
                    successors.push(forked);
                } else {
                    self.push_or_drop_terminal(STASH_PRUNED, forked);
                }
            } else {
                // Conservative fork without condition
                let mut forked = successors[0].fork();
                forked.set_pc(fork.unexplored_target);
                self.sm.set_root(forked.state_id(), root_state_id);
                if self.constraint_solver.lazy_solves || forked.satisfiable() {
                    successors.push(forked);
                } else {
                    self.push_or_drop_terminal(STASH_PRUNED, forked);
                }
            }
        }

        self.profiling.accumulated_stats.deferred_fork_count += deferred_forks.len() as u64;
    }
}

#[cfg(test)]
mod sizes {
    use super::*;
    use std::mem::size_of;

    /// Documentation-grade size probe. Captured to validate the iter-61
    /// decision to suppress `clippy::result_large_err` / `large_enum_variant`
    /// instead of boxing the inline `RustSimState` / `PendingCallback`. Run
    /// with `cargo test --release -p angr -- --nocapture sizes::print`.
    #[test]
    fn print_step_error_sizes() {
        let step_err = size_of::<StepError>();
        let result_unit = size_of::<Result<(), StepError>>();
        let pending = size_of::<PendingCallback>();
        let sim_state = size_of::<RustSimState>();
        let boxed_result = size_of::<Result<(), Box<StepError>>>();
        println!("StepError                       = {step_err} bytes");
        println!("Result<(), StepError>           = {result_unit} bytes");
        println!("Result<(), Box<StepError>>      = {boxed_result} bytes");
        println!("PendingCallback                 = {pending} bytes");
        println!("RustSimState                    = {sim_state} bytes");
        println!(
            "boxing-would-save               = {} bytes per Err return",
            result_unit.saturating_sub(boxed_result)
        );
    }
}
