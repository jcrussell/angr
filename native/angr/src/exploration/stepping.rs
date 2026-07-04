use super::core_outcome::{
    BounceKind, CoreCounters, CoreOutcome, CoreReturn, ParallelProfiling, PendingBounce,
    PostStepInputs, run_post_step_core,
};
use super::*;
use crate::arch::RegisterFile;
use crate::interpreter::BranchSnapshot;
use crate::memory::SymbolicMemory;
use crate::state::{CallStackEntry, HistoryEntry};
use crate::vex::IRSB;
use lru::LruCache;

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
    /// Unconstrained state - too many symbolic jump targets. The second field
    /// carries any loop-exit deferred forks materialized in EAGER mode at the
    /// unconstrained jump (angr-027h): the main state goes to the unconstrained
    /// stash but these forks are routed back to active so a find-guided search
    /// can still reach a target that lies behind the loop exit. Empty in the
    /// common case (no deferred forks pending, or deferred forks disabled).
    Unconstrained(RustSimState, Vec<RustSimState>),
}

/// Why a native sub-call could not be set up; the dispatcher falls back to the
/// Python SimProcedure path on any of these. Fields are carried for the
/// `Debug` diagnostic in the fallback log line (dead-code analysis ignores
/// `Debug`-only reads, hence the allow).
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum SubcallSetupError {
    /// More guest arguments than the ABI exposes in registers. Stack-spilled
    /// guest args are not yet supported (S2 scope; defers to Python).
    TooManyArgs { requested: usize, available: usize },
    /// Stack pointer is symbolic — cannot place the sentinel return slot.
    SpSymbolic,
    /// Writing the sentinel return slot to the stack failed (unmapped / perms).
    Memory(crate::memory::MemoryError),
    /// Link-register ABI with no `link_register()` wired up — cannot redirect
    /// the guest routine's return to the sentinel.
    UnsupportedAbi,
}

/// Output of one interpreter run, packaged for the post-execution phase.
///
/// Replaces a 12-element tuple destructure that became unreadable as fields
/// were added. All fields are owned (taken from the interpreter before drop).
pub(crate) struct InterpreterStepResult {
    pub(crate) result: RunResult,
    pub(crate) deferred_forks: Vec<DeferredFork>,
    pub(crate) last_condition: Option<RustBV>,
    pub(crate) stored_conditions: FxHashMap<u64, RustBV>,
    pub(crate) fork_snapshots: FxHashMap<u64, BranchSnapshot>,
    pub(crate) new_registers: RegisterFile,
    pub(crate) new_pc: u64,
    pub(crate) new_call_stack: Vec<CallStackEntry>,
    pub(crate) new_detailed_history: Vec<HistoryEntry>,
    pub(crate) recovered_memory: Option<SymbolicMemory>,
    pub(crate) step_stats: ExecutionStats,
    pub(crate) updated_block_cache: LruCache<u64, Arc<IRSB>>,
    /// Set only when `state.keep_ip_symbolic()` was true and the interpreter
    /// concretized a symbolic default-exit next-pc. The manager writes this
    /// back to the state's IP register after `state.set_pc(new_pc)`.
    pub(crate) symbolic_ip_at_exit: Option<RustBV>,
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

        // Snapshot the read-only step config once (angr-vh834); reused by both
        // the interpreter step and the post-step core so the single-threaded
        // path takes exactly one `step_context()` clone per step.
        let ctx = self.step_context();

        // Run the VEX interpreter to its next event.
        let step = super::step_core::run_interpreter_step_core(
            &ctx,
            callbacks,
            &mut state,
            initial_pc,
            skip_addr,
            setup_start,
            &mut self.environment.block_cache,
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

        // Post-interpreter classification + fork materialization, run without
        // touching `&mut self` (angr-vh834, Phase 1). The coordinator (this
        // method) then applies every deferred mutation the core recorded.
        let prof = ParallelProfiling::default();
        let root_hint = self.sm.root_or_self(state.state_id());
        let inputs = PostStepInputs {
            result: step.result,
            deferred_forks,
            last_condition,
            stored_conditions,
            fork_snapshots,
        };
        let outcome = run_post_step_core(
            &ctx,
            &prof,
            &self.native_procedures,
            &self.native_syscalls,
            state,
            inputs,
            root_hint,
        );
        self.apply_core_outcome(callbacks, &prof, outcome)
    }

    /// Apply the deferred mutations the `&mut self`-free post-step core recorded
    /// (angr-vh834), reproducing the single-threaded firing order byte-for-byte:
    /// fold profiling/counters, stamp `set_root` on every fork, fire each
    /// `dispatch_fork_inspect`, push pruned + side-effect terminal states, then
    /// translate the routing decision into the `Result` the run loop consumes.
    fn apply_core_outcome(
        &mut self,
        callbacks: &PythonCallbacks,
        prof: &ParallelProfiling,
        outcome: CoreOutcome,
    ) -> Result<Vec<RustSimState>, StepError> {
        let CoreOutcome {
            ret,
            pruned,
            fork_ids,
            terminal_pushes,
            counters,
            root_hint,
        } = outcome;

        // 1. Fold solver fork/sat/deferred timing into accumulated_stats. Adds
        //    zero for fields that never fired (profiling disabled), except the
        //    UNCONDITIONAL `deferred_fork_count` from the process path — matching
        //    the legacy gating exactly.
        prof.fold_into(&mut self.profiling.accumulated_stats);

        // 2. Fold manager-level counter deltas.
        self.fold_core_counters(counters);

        // 3. set_root for every newly minted fork (SAT successors flagged
        //    `is_fork`, all pruned, and the eager unconstrained forks). Order
        //    independent (keyed map insert) — matches the inline legacy value.
        match &ret {
            CoreReturn::Continue(succ) => {
                for (s, tag) in succ {
                    if let Some(rh) = tag.root_hint {
                        debug_assert!(tag.is_fork);
                        self.sm.set_root(s.state_id(), rh);
                    }
                }
            }
            CoreReturn::Unconstrained(_, forks) => {
                for f in forks {
                    self.sm.set_root(f.state_id(), root_hint);
                }
            }
            _ => {}
        }
        for s in &pruned {
            self.sm.set_root(s.state_id(), root_hint);
        }

        // 4. Fire `dispatch_fork_inspect` in the legacy firing order.
        for fid in fork_ids {
            self.dispatch_fork_inspect(fid);
        }

        // 5. Push UNSAT forks to STASH_PRUNED.
        for s in pruned {
            self.push_or_drop_terminal(STASH_PRUNED, s);
        }

        // 6. Side-effect terminal pushes (no-return main state from a native
        //    `exit` / no-return proc, deadended while its forks continue).
        for (s, stash) in terminal_pushes {
            self.push_or_drop_terminal(stash, s);
        }

        // 7. Translate the routing decision.
        match ret {
            CoreReturn::Continue(succ) => Ok(succ.into_iter().map(|(s, _)| s).collect()),
            CoreReturn::Deadended(state) => Err(StepError::Deadended(state)),
            CoreReturn::Errored(state, message) => Err(StepError::Error(state, message)),
            CoreReturn::Unconstrained(state, forks) => Err(StepError::Unconstrained(state, forks)),
            CoreReturn::NeedsPython(bounce) => self.dispatch_bounce(callbacks, bounce),
        }
    }

    /// Fold a `CoreOutcome`'s manager-level counter deltas (native-proc /
    /// syscall / simproc fallback counters, `deferred_forks_dropped`) into the
    /// manager fields. Extracted from `apply_core_outcome` (step 2) so the
    /// parallel wave loop can fold the counters its workers accumulated
    /// (angr-vh834 Phase 5) through the exact same path.
    pub(crate) fn fold_core_counters(&mut self, counters: CoreCounters) {
        let nps = &mut self.profiling.native_proc_stats;
        nps.native_calls += counters.native_calls;
        nps.python_fallbacks += counters.native_python_fallbacks;
        for (k, v) in counters.call_counts {
            *nps.call_counts.entry(k).or_insert(0) += v;
        }
        for (k, v) in counters.symbolic_fallbacks_by_name {
            *nps.symbolic_fallbacks_by_name.entry(k).or_insert(0) += v;
        }
        for (k, v) in counters.not_implemented_fallbacks_by_name {
            *nps.not_implemented_fallbacks_by_name.entry(k).or_insert(0) += v;
        }
        for (k, v) in counters.other_fallbacks_by_name {
            *nps.other_fallbacks_by_name.entry(k).or_insert(0) += v;
        }
        self.syscall_native_count += counters.syscall_native_count;
        for (k, v) in counters.syscall_native_by_num {
            *self.syscall_native_by_num.entry(k).or_insert(0) += v;
        }
        self.syscall_python_fallback_count += counters.syscall_python_fallback_count;
        for (k, v) in counters.syscall_python_fallback_by_num {
            *self.syscall_python_fallback_by_num.entry(k).or_insert(0) += v;
        }
        self.simprocedure_python_fallback_count += counters.simprocedure_python_fallback_count;
        for (k, v) in counters.simprocedure_fallback_by_name {
            *self.simprocedure_fallback_by_name.entry(k).or_insert(0) += v;
        }
        self.deferred_forks_dropped += counters.deferred_forks_dropped;
    }

    /// Run the legacy Python-bouncing arm for a `NeedsPython` core outcome. The
    /// core already did any native dispatch and recorded its counters; these
    /// tails build the exact `PendingCallback` (and, for `UnmodeledCall`, run the
    /// `&mut self` resolve path) the single-threaded engine produced inline.
    pub(crate) fn dispatch_bounce(
        &mut self,
        callbacks: &PythonCallbacks,
        bounce: PendingBounce,
    ) -> Result<Vec<RustSimState>, StepError> {
        let PendingBounce {
            kind,
            mut state,
            deferred_forks,
            last_condition,
            stored_conditions,
            fork_snapshots,
        } = bounce;

        match kind {
            BounceKind::Hook { addr } => {
                state.set_pc(addr);
                // P1 Fix: history BEFORE callback so Python can read recent_bbl_addrs[-1].
                state.add_to_history(addr);
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
                let solver_ref = state.solver();
                let shared_ctx = RustSolverContext::from_shared_sym_context(solver_ref.clone());
                if let Some(start) = hook_fork_start {
                    let fork_count = u64::from(pre_callback_snapshot.is_some());
                    self.profiling.accumulated_stats.solver_fork_time_ns +=
                        start.elapsed().as_nanos() as u64;
                    self.profiling.accumulated_stats.solver_fork_count += fork_count;
                }
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
            BounceKind::SimProcedurePython {
                addr,
                name,
                num_args,
                return_addr,
            } => {
                state.set_pc(addr);
                state.add_to_history(addr);
                let (pre_callback_snapshot, shared_ctx) =
                    super::helpers::prepare_shared_callback_solver(&state, &deferred_forks);
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
            BounceKind::SyscallPython { num } => {
                let (pre_callback_snapshot, shared_ctx) =
                    super::helpers::prepare_shared_callback_solver(&state, &deferred_forks);
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
            BounceKind::SymbolicBranch {
                condition_id,
                true_target,
                false_target,
            } => {
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
            BounceKind::UnmodeledCall {
                addr,
                return_addr,
                symbol_name,
            } => self.handle_unmodeled_call(
                callbacks,
                state,
                addr,
                return_addr,
                symbol_name,
                deferred_forks,
                stored_conditions,
                fork_snapshots,
            ),
            BounceKind::PythonVEXFallback { addr, reason } => {
                // state.set_pc(addr) was applied by the core before bouncing.
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
        }
    }

    /// Set up a native sub-call (`ProcOutcome::CallAndResume`): make the guest
    /// routine `target` run with `sub_args`, then return to the resume sentinel
    /// so the proc's continuation re-enters via [`handle_native_resume`].
    ///
    /// All feasibility checks happen before any state mutation, so on `Err` the
    /// caller can cleanly fall back to the Python SimProcedure path. `S2`,
    /// bead angr-5gf0s. See `tools/decisions/native_subcall_dispatcher_design.md`.
    ///
    /// Stack-return ABI (x86/amd64): at proc entry `[sp]` holds the caller's
    /// return address; we overwrite it with the sentinel so the guest `ret`
    /// lands on the sentinel (SP unchanged here — the guest's own `ret` advances
    /// it). The original caller address rides in the frame's
    /// `caller_return_addr`, not the stack. Link-register ABI: write the
    /// sentinel into the link register (requires `link_register()`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn setup_native_subcall(
        &self,
        state: &mut RustSimState,
        proc_name: String,
        saved_args: Vec<RustBV>,
        caller_return_addr: u64,
        target: u64,
        sub_args: Vec<RustBV>,
        resume_tag: u32,
    ) -> Result<(), SubcallSetupError> {
        let cc = &self.environment.calling_convention;
        let arg_regs = cc.arg_registers();
        if sub_args.len() > arg_regs.len() {
            return Err(SubcallSetupError::TooManyArgs {
                requested: sub_args.len(),
                available: arg_regs.len(),
            });
        }
        let ptr_bits = cc.pointer_size() * 8;
        let sentinel = crate::procedures::native_resume_sentinel(cc.pointer_size());

        // --- feasibility checks (no mutation yet) ---
        let lr_offset = if cc.pops_return_addr() {
            None
        } else {
            Some(
                cc.link_register()
                    .ok_or(SubcallSetupError::UnsupportedAbi)?,
            )
        };
        let sp_val = if cc.pops_return_addr() {
            Some(
                state
                    .get_sp()
                    .as_u64()
                    .ok_or(SubcallSetupError::SpSymbolic)?,
            )
        } else {
            None
        };

        // --- mutation: redirect the guest routine's return to the sentinel ---
        if let Some(sp) = sp_val {
            // Overwrite the caller return slot at [sp] with the sentinel. This
            // is the only fallible mutation; do it first so an unmapped stack
            // leaves the state untouched for the Python fallback.
            state
                .memory_mut()
                .store_concrete(sp, RustBV::concrete(sentinel as u128, ptr_bits))
                .map_err(SubcallSetupError::Memory)?;
        } else if let Some(lr) = lr_offset {
            state.set_register_by_offset(lr, RustBV::concrete(sentinel as u128, ptr_bits));
        }

        // --- record the continuation and enter the guest routine ---
        state.push_native_resume_frame(crate::state::NativeResumeFrame {
            proc_name,
            resume_tag,
            saved_args,
            caller_return_addr,
        });
        for (reg, val) in arg_regs.iter().zip(sub_args.into_iter()) {
            state.set_register_by_offset(*reg, val);
        }
        state.set_pc(target);
        Ok(())
    }

    /// Re-enter a native proc's continuation after a `CallAndResume` sub-call
    /// returns to the resume sentinel. Pops the top resume frame, calls the
    /// proc's [`crate::procedures::NativeSimProcedure::resume`], and applies the
    /// resulting [`ProcOutcome`]. `S2`, bead angr-5gf0s.
    ///
    /// Retained as a focused direct-call test harness (`subcall_tests.rs`); the
    /// production single-threaded path now resumes via
    /// `core_outcome::handle_native_resume_core` (angr-vh834). Gated to test
    /// builds — its only caller is the `#[cfg(test)]` `subcall_tests` module —
    /// so it carries no `dead_code` allow (angr-0mqkc.2).
    #[cfg(test)]
    fn handle_native_resume(
        &mut self,
        mut state: RustSimState,
        deferred_forks: Vec<DeferredFork>,
        stored_conditions: FxHashMap<u64, RustBV>,
        fork_snapshots: FxHashMap<u64, BranchSnapshot>,
    ) -> Result<Vec<RustSimState>, StepError> {
        let frame = match state.pop_native_resume_frame() {
            Some(f) => f,
            None => {
                // Sentinel reached with no pending frame: a corrupt state we
                // cannot resume. Deadend defensively rather than guess a PC.
                log::error!("native resume sentinel hit with empty resume stack; deadending");
                return Err(StepError::Deadended(state));
            }
        };

        // Run the continuation. The `self.native_procedures` borrow is released
        // once `outcome` is bound, freeing `&mut self` for the sub-call setup.
        let outcome = match self.native_procedures.get(&frame.proc_name) {
            Some(proc) => proc.resume(&mut state, frame.resume_tag, &frame.saved_args),
            None => {
                log::error!(
                    "native resume: proc {} not in registry; deadending",
                    frame.proc_name
                );
                return Err(StepError::Deadended(state));
            }
        };

        match outcome {
            Ok(ProcOutcome::Return(ret_val)) => {
                if let Some(rv) = ret_val {
                    let ret_reg = self.environment.calling_convention.return_register();
                    state.set_register_by_offset(ret_reg, rv);
                }
                // Resume the original caller. The guest routine's `ret` already
                // consumed the sentinel return slot (stack-return ABI advances
                // SP), so unlike the fresh-entry return path we do NOT adjust SP.
                state.set_pc(frame.caller_return_addr);
            }
            Ok(ProcOutcome::CallAndResume {
                target,
                args: sub_args,
                resume_tag,
            }) => {
                // Nested sub-call: the original caller and saved args carry
                // forward so the final return still lands at `caller_return_addr`.
                if let Err(e) = self.setup_native_subcall(
                    &mut state,
                    frame.proc_name.clone(),
                    frame.saved_args.clone(),
                    frame.caller_return_addr,
                    target,
                    sub_args,
                    resume_tag,
                ) {
                    log::error!("native resume nested sub-call setup failed ({e:?}); deadending");
                    return Err(StepError::Deadended(state));
                }
            }
            Err(e) => {
                // resume() should never fail when reached via the sentinel (the
                // proc opted into sub-calls). Deadend defensively.
                log::error!(
                    "native resume: {} resume() failed: {:?}; deadending",
                    frame.proc_name,
                    e
                );
                return Err(StepError::Deadended(state));
            }
        }

        // Deferred-fork handling identical to the native return path.
        let mut successors = vec![state];
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
            match callbacks.call_resolve_function(addr, symbol_name.as_deref()) {
                Ok(Some((name, num_args, no_return))) => {
                    // Function resolved! Register it and return to Python for execution
                    log::debug!(
                        "Resolved unmodeled call at 0x{addr:x} -> {name} (args={num_args}, no_return={no_return})"
                    );

                    // Register the procedure so future calls are hooked
                    self.hooks.insert(addr);
                    self.simprocedures
                        .insert(addr, (name.clone(), num_args, no_return));

                    let (pre_callback_snapshot, shared_ctx) =
                        super::helpers::prepare_shared_callback_solver(&state, &deferred_forks);

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
                    log::warn!("resolve_function callback error at 0x{addr:x}: {e}");
                    Err(StepError::Error(
                        state,
                        format!("resolve_function error: {e}"),
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
            "P21: Unmodeled call at 0x{addr:x} - generic skip (ret=0) to return_addr=0x{return_addr:x}"
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

        let root_state_id = self.sm.root_or_self(successors[0].state_id());

        for fork in &deferred_forks {
            if let Some(condition) = stored_conditions.get(&fork.condition_id) {
                // Add the taken-path constraint to the main state
                if fork.path_taken {
                    successors[0].solver().borrow().assume_true(condition);
                } else {
                    successors[0].solver().borrow().assume_false(condition);
                }

                // Create forked state for the unexplored path
                let forked = super::helpers::build_unexplored_fork(
                    &successors[0],
                    fork,
                    condition,
                    &mut fork_snapshots,
                );

                self.sm.set_root(forked.state_id(), root_state_id);

                // state.inspect fork BP — see handle_block_end for rationale.
                self.dispatch_fork_inspect(forked.state_id());

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

                self.dispatch_fork_inspect(forked.state_id());

                if self.constraint_solver.lazy_solves || forked.satisfiable() {
                    successors.push(forked);
                } else {
                    self.push_or_drop_terminal(STASH_PRUNED, forked);
                }
            }
        }

        self.profiling.accumulated_stats.deferred_fork_count += deferred_forks.len() as u64;
    }

    /// Fire a `state.inspect.fork` BP for the given forked state id.
    /// Bit-gated on `InspectEvent::Fork` (bit 4) — single atomic load in
    /// the common no-BP case. Dispatches `when='after'` with no attrs,
    /// matching Python `engines/successors.py:203` where the BP fires
    /// on the newly-added successor after constraints + ip are applied
    /// but before satisfiability is checked downstream. Errors from the
    /// user's BP action are swallowed (logged at debug) — same MVP
    /// pattern as the other Rust-side inspect dispatchers.
    #[inline]
    pub(crate) fn dispatch_fork_inspect(&self, forked_state_id: u64) {
        let cb = match self.callbacks.as_ref() {
            Some(c) => c,
            None => return,
        };
        // Fork = bit 4 (reserved slot mirrored in
        // `_INSPECT_EVENT_SPECS["fork"]`).
        if !cb.inspect_event_enabled(4) {
            return;
        }
        // call_inspect_fork self-attaches the GIL (angr-vh834 Phase 4), so no
        // explicit Python::attach wrapper is needed here.
        if let Err(e) = cb.call_inspect_fork(forked_state_id as i64, "after") {
            log::debug!("fork inspect dispatch raised (state {forked_state_id}): {e}");
        }
    }
}

#[cfg(test)]
#[path = "sizes_tests.rs"]
mod sizes;

#[cfg(test)]
#[path = "subcall_tests.rs"]
mod subcall_tests;
