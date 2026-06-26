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
    /// Unconstrained state - too many symbolic jump targets. The second field
    /// carries any loop-exit deferred forks materialized in EAGER mode at the
    /// unconstrained jump (angr-027h): the main state goes to the unconstrained
    /// stash but these forks are routed back to active so a find-guided search
    /// can still reach a target that lies behind the loop exit. Empty in the
    /// common case (no deferred forks pending, or deferred forks disabled).
    Unconstrained(RustSimState, Vec<RustSimState>),
}

/// What the native fast path in `handle_simprocedure` decided, captured so the
/// follow-up action runs *after* the `self.native_procedures` borrow is
/// released (a `SubCall` setup needs `&mut self`). See the S2 design in
/// `tools/decisions/native_subcall_dispatcher_design.md` (bead angr-5gf0s).
enum NativeProcDisposition {
    /// Native ran and produced a plain return; `no_return` deadends the state.
    /// The return value has not been written yet.
    Returned {
        no_return: bool,
        ret_val: Option<RustBV>,
    },
    /// Native requested a guest sub-call (`ProcOutcome::CallAndResume`).
    /// `saved_args` are the original proc args (re-handed to the continuation);
    /// `sub_args` are the arguments for the guest routine `target`.
    SubCall {
        proc_name: String,
        saved_args: Vec<RustBV>,
        target: u64,
        sub_args: Vec<RustBV>,
        resume_tag: u32,
    },
    /// No native handler (or it errored / declined) — use the Python path.
    Fallback,
}

/// Why a native sub-call could not be set up; the dispatcher falls back to the
/// Python SimProcedure path on any of these. Fields are carried for the
/// `Debug` diagnostic in the fallback log line (dead-code analysis ignores
/// `Debug`-only reads, hence the allow).
#[derive(Debug)]
#[allow(dead_code)]
enum SubcallSetupError {
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
    /// Write a native syscall's return value into the return register,
    /// applying the Linux error-register semantics for ABIs that carry a
    /// separate success/failure flag (MIPS `$a3`).
    ///
    /// Mirrors `CC.linux_syscall_update_error_reg` in
    /// `angr/calling_conventions.py`: when the (unsigned) return value is at
    /// or above `errno_start` it is treated as `-errno`, the error register is
    /// set to all-ones, and the return register holds the *positive* errno;
    /// otherwise the error register is cleared to 0 and the return value is
    /// passed through unchanged. On arches with no error register (the common
    /// case) only the return register is written.
    fn write_syscall_return(&self, state: &mut RustSimState, ret_reg: u32, ret: RustBV) {
        let Some((err_reg, errno_start)) =
            self.environment.calling_convention.syscall_error_register()
        else {
            state.set_register_by_offset(ret_reg, ret);
            return;
        };

        let bits = ret.width();
        let (ret_val, err_val) = {
            let ctx = state.solver().borrow();
            let errno_start_bv = RustBV::concrete(errno_start as u128, bits);
            let error_cond = ret.uge(&errno_start_bv, &ctx);
            let err_val = error_cond.ite(&RustBV::ones(bits), &RustBV::zero(bits), &ctx);
            let ret_val = error_cond.ite(&ret.neg(&ctx), &ret, &ctx);
            (ret_val, err_val)
        };
        state.set_register_by_offset(ret_reg, ret_val);
        state.set_register_by_offset(err_reg, err_val);
    }

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
                // CGC binaries use x86 syscall numbers 1-7 that collide with
                // Linux i386 (1=exit/_terminate, 2=fork/transmit, ...). When
                // os_name=="cgc", dispatch through the CGC table instead of
                // the arch table so the DECREE ABI handlers fire. Other OSes
                // (default "linux") fall through to per-arch dispatch.
                let dispatch_key: &str = if self.environment.os_name == "cgc" {
                    "CGC"
                } else {
                    state.arch().name()
                };
                let native_handler = num.and_then(|n| self.native_syscalls.get(dispatch_key, n));
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
                        let (pre_callback_snapshot, shared_ctx) =
                            super::helpers::prepare_shared_callback_solver(&state, &deferred_forks);
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
                    let outcome = handler.call(&mut state, &args);
                    if outcome.is_ok() {
                        // Native fast path fired (no Python round-trip). An
                        // `Err` outcome falls through to the Python callback
                        // below and is counted as a fallback instead.
                        self.syscall_native_count += 1;
                        *self
                            .syscall_native_by_num
                            .entry(num.map(|n| n as i64).unwrap_or(-1))
                            .or_insert(0) += 1;
                    }
                    match outcome {
                        Ok(SyscallOutcome::Continue { ret }) => {
                            let ret_reg = self.environment.calling_convention.return_register();
                            let bits = state.arch().bits();
                            let ret_bv = RustBV::concrete(ret as u128, bits);
                            self.write_syscall_return(&mut state, ret_reg, ret_bv);
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
                            self.write_syscall_return(&mut state, ret_reg, ret);
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
            RunResult::Error {
                message,
                addr,
                kind,
            } => {
                state.set_pc(addr);
                // Route on the typed error kind rather than substrings of the
                // message (angr-zzju9). A `Deadend` kind is an unliftable block
                // (the Python lift callback returned the empty-IRSB sentinel) —
                // gracefully deadended, matching the Python engine. A jump to
                // 0x0 (e.g. after a clean exit) is likewise a deadend, not an
                // error. Everything else — including a genuinely malformed IRSB,
                // now classified `Fatal` — moves to the errored stash.
                match kind {
                    RunErrorKind::Deadend => Err(StepError::Deadended(state)),
                    RunErrorKind::Fatal if addr == 0 => Err(StepError::Deadended(state)),
                    RunErrorKind::Fatal => Err(StepError::Error(state, message)),
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
                // Too many symbolic jump targets - the main state goes to the
                // unconstrained stash. But its accumulated deferred forks are
                // NOT dropped: they are materialized in EAGER mode and routed
                // back to active so a find-guided search can still reach a
                // target behind the loop exit.
                //
                // Why this matters for the CADET easter-egg find (confirmed
                // iter64-65, full CFG of sub_80481a0 — see bd memories
                // `benchmark-cadet-eggphase-rootcause-cfg` and
                // `benchmark-cadet-eager-reaches-egg`): the find target
                // 0x804833E sits behind a symbolic strlen loop whose exit
                // (je 0x804826f) is a FORWARD branch, so deferred-fork mode
                // takes the loop-continuation as the main chain and DEFERS every
                // loop exit. The main chain dives the loop, overflows the saved
                // return address (receive reads 0x80 bytes), and ret goes
                // unconstrained. The accumulated loop-exit forks are the only
                // paths that can reach the egg.
                //
                // iter63 materialized them in DEFERRED mode: they re-dive the
                // loop nest, re-overflow, re-go-unconstrained, and recursively
                // diverge. iter65 PROVED eager forking converges (found at
                // step 38, active bounded ~27). So we materialize them with
                // `force_eager` so the resumed subtree BFSes cleanly instead of
                // re-deferring. The flag is per-state, so the default deferred
                // (fast) path on every other bench is untouched.
                // angr-027h two-phase explore: in deferred mode (phase 1) the
                // accumulated loop-exit forks are DROPPED so the active stash
                // collapses to `active_empty`, which is the signal Python's
                // `_explore_with_addresses` uses to re-seed the initial states
                // in eager mode (phase 2). In eager mode (phase 2) no forks are
                // deferred in the first place (every fork materialized during
                // the loop dive), so `deferred_forks` is empty here and
                // materialize is a no-op — but we still route through it so the
                // egg-reaching subtree is handled identically if a future caller
                // mixes the two. Routing eager forks back to active while still
                // in deferred mode would prevent `active_empty` and defeat the
                // phase-2 trigger (iter66 regression — see bd memory
                // `benchmark-cadet-single-step-loop-unroll-defeats-latch`).
                let forks = if self.exec_config.use_deferred_forks
                    && !self.materialize_unconstrained_forks
                {
                    // angr-ckdy: record that egg-reaching loop-exit forks were
                    // discarded so a step-driven loop (CADET solve.py phase 3)
                    // can tell `active_empty` apart from a genuine exhaustion
                    // and re-seed in eager mode.
                    self.deferred_forks_dropped += deferred_forks.len() as u64;
                    Vec::new()
                } else {
                    // Either we are already in eager mode, OR the opt-in
                    // `materialize_unconstrained_forks` flag (angr-ckdy) asks us
                    // to keep the loop-exit forks alive even in deferred mode so
                    // a bare step-loop keeps progressing toward a target behind
                    // the loop exit instead of collapsing to `active_empty`.
                    let root_state_id = self.sm.root_or_self(state.state_id());
                    self.materialize_deferred_forks(
                        &state,
                        deferred_forks,
                        &stored_conditions,
                        fork_snapshots,
                        root_state_id,
                        true,
                    )
                };
                Err(StepError::Unconstrained(state, forks))
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
        fork_snapshots: FxHashMap<u64, BranchSnapshot>,
    ) -> Result<Vec<RustSimState>, StepError> {
        state.set_pc(pc);

        // Track root state ID for lineage
        let root_state_id = self.sm.root_or_self(state.state_id());

        // Materialize the deferred forks (continuing the main chain, deferred
        // mode preserved) and prepend the main state as successors[0].
        let forks = self.materialize_deferred_forks(
            &state,
            deferred_forks,
            &stored_conditions,
            fork_snapshots,
            root_state_id,
            false,
        );
        let mut successors = Vec::with_capacity(forks.len() + 1);
        successors.push(state);
        successors.extend(forks);

        Ok(successors)
    }

    /// Materialize a batch of `DeferredFork`s into concrete successor states.
    ///
    /// `base` is the main state the forks diverge from; the taken-path
    /// constraint of each fork is accumulated onto it (mirroring Python's
    /// per-branch narrowing) so later in-block forks inherit earlier branch
    /// decisions. The returned vec holds the SAT forks only — UNSAT forks are
    /// routed to the pruned stash here. `base` itself is NOT included.
    ///
    /// `force_eager` (angr-027h): when true, each materialized fork is flagged
    /// `force_eager_forks` so its subsequent steps fork eagerly regardless of
    /// the manager `use_deferred_forks` setting. Used when resuming loop-exit
    /// forks at an UnconstrainedJump, where leaving them in deferred mode makes
    /// them re-dive the symbolic loop nest and recursively diverge.
    fn materialize_deferred_forks(
        &mut self,
        base: &RustSimState,
        deferred_forks: Vec<DeferredFork>,
        stored_conditions: &FxHashMap<u64, RustBV>,
        mut fork_snapshots: FxHashMap<u64, BranchSnapshot>,
        root_state_id: u64,
        force_eager: bool,
    ) -> Vec<RustSimState> {
        let mut forks = Vec::new();
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
                    base.solver().borrow().assume_true(condition);
                } else {
                    base.solver().borrow().assume_false(condition);
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
                let mut forked = super::helpers::build_unexplored_fork(
                    base,
                    &fork,
                    condition,
                    &mut fork_snapshots,
                );
                if force_eager {
                    forked.set_force_eager_forks(true);
                }
                if let Some(start) = fork_start {
                    self.profiling.accumulated_stats.solver_fork_time_ns +=
                        start.elapsed().as_nanos() as u64;
                    self.profiling.accumulated_stats.solver_fork_count += 1;
                }
                // Track root state ID for this forked state
                self.sm.set_root(forked.state_id(), root_state_id);

                // state.inspect fork BP fires BEFORE the satisfiability
                // check (matching Python successors.py:203 which fires
                // before downstream pruning). UNSAT forks still get the
                // BP — same intent as Python's pre-discard fire.
                self.dispatch_fork_inspect(forked.state_id());

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
                    forks.push(forked);
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
                let mut forked = base.fork();
                forked.set_pc(fork.unexplored_target);
                if force_eager {
                    forked.set_force_eager_forks(true);
                }
                self.sm.set_root(forked.state_id(), root_state_id);

                self.dispatch_fork_inspect(forked.state_id());

                // P13: Still check satisfiability
                if self.constraint_solver.lazy_solves || forked.satisfiable() {
                    forks.push(forked);
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

        forks
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
        // Native sub-call resume sentinel (S2, bead angr-5gf0s): a guest routine
        // invoked via `ProcOutcome::CallAndResume` returns here. Pop the top
        // resume frame and re-enter its proc's continuation — recognized by name
        // before any native-registry or Python lookup.
        if name == crate::procedures::NATIVE_RESUME_SENTINEL_NAME {
            return self.handle_native_resume(
                state,
                deferred_forks,
                stored_conditions,
                fork_snapshots,
            );
        }

        // Try native procedure first — avoids Python callback overhead.
        // Skip native for addresses inside the binary — these are user-placed
        // hooks where the Python SimProcedure should always run (the user hooked
        // a specific function for a reason, e.g., hooking strings_not_equal with strcmp).
        let is_in_binary = self
            .environment
            .binary_regions
            .iter()
            .any(|(base, data)| addr >= *base && addr < *base + data.len() as u64);
        // Run the native fast path (if any) and capture *what to do* without
        // acting yet: the action (especially a `CallAndResume` sub-call setup,
        // which needs `&mut self`) must run after the `self.native_procedures`
        // borrow is released. `Fallback` routes to the Python SimProcedure path.
        let disposition: NativeProcDisposition = if !is_in_binary {
            if let Some(native_proc) = self.native_procedures.get(&name) {
                let proc_no_return = native_proc.no_return();
                match self.extract_procedure_args(&state, num_args) {
                    Err(e) => {
                        // SP symbolic / stack unmapped — silently zero-padding
                        // here would hand the native handler fabricated zeros
                        // and mask the underlying stack-setup bug. Skip the
                        // native fast path; `Fallback` routes to the Python
                        // SimProcedure callback at the bottom of this function.
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
                        NativeProcDisposition::Fallback
                    }
                    Ok(args) => match native_proc.call_ex(&mut state, &args) {
                        Ok(outcome) => {
                            self.profiling.native_proc_stats.native_calls += 1;
                            *self
                                .profiling
                                .native_proc_stats
                                .call_counts
                                .entry(name.clone())
                                .or_insert(0) += 1;
                            match outcome {
                                ProcOutcome::Return(ret_val) => NativeProcDisposition::Returned {
                                    no_return: proc_no_return,
                                    ret_val,
                                },
                                ProcOutcome::CallAndResume {
                                    target,
                                    args: sub_args,
                                    resume_tag,
                                } => NativeProcDisposition::SubCall {
                                    proc_name: name.clone(),
                                    saved_args: args,
                                    target,
                                    sub_args,
                                    resume_tag,
                                },
                            }
                        }
                        Err(e) => {
                            log::debug!(
                                "Native procedure {} returned error, falling back to Python: {:?}",
                                name,
                                e
                            );
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
                            NativeProcDisposition::Fallback
                        }
                    },
                }
            } else {
                NativeProcDisposition::Fallback
            }
        } else {
            NativeProcDisposition::Fallback
        };

        let fall_back_to_python = match disposition {
            NativeProcDisposition::Returned { no_return, ret_val } => {
                if !no_return {
                    if let Some(rv) = ret_val {
                        let ret_reg = self.environment.calling_convention.return_register();
                        state.set_register_by_offset(ret_reg, rv);
                    }
                    // Set PC to return address and pop stack (return-only path,
                    // semantics unchanged from the pre-S2 dispatcher).
                    state.set_pc(return_addr);
                    let sp = state.get_sp().as_u64().unwrap_or(0);
                    let ptr_size = state.arch().bytes() as u64;
                    state.set_sp(RustBV::concrete(
                        (sp + ptr_size) as u128,
                        state.arch().bits(),
                    ));
                }
                Some(no_return)
            }
            NativeProcDisposition::SubCall {
                proc_name,
                saved_args,
                target,
                sub_args,
                resume_tag,
            } => {
                // The proc requested a guest sub-call. Set it up (jump to
                // `target`, arrange resume via the sentinel). On any setup
                // failure, leave the state untouched and fall back to Python.
                match self.setup_native_subcall(
                    &mut state,
                    proc_name,
                    saved_args,
                    return_addr,
                    target,
                    sub_args,
                    resume_tag,
                ) {
                    Ok(()) => Some(false),
                    Err(e) => {
                        log::debug!(
                            "native sub-call setup failed ({:?}); falling back to Python for {}",
                            e,
                            name
                        );
                        None
                    }
                }
            }
            NativeProcDisposition::Fallback => None,
        };

        if let Some(no_return) = fall_back_to_python {
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
    fn setup_native_subcall(
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
        let root_state_id = self.sm.root_or_self(state.state_id());

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

        // Create interpreter with the state's solver.
        //
        // angr-027h: a state carrying `force_eager_forks` (a loop-exit fork
        // resumed at an UnconstrainedJump) overrides the manager-level
        // `use_deferred_forks` so it materializes successors eagerly and BFSes
        // to the find target instead of recursively re-deferring.
        let mut exec_config = self.exec_config.clone();
        if state.force_eager_forks() {
            exec_config.use_deferred_forks = false;
        }
        let mut interp =
            VEXInterpreter::with_config(self.environment.vex_arch, &solver_ref, exec_config);

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

        // Register the native sub-call resume sentinel (S2, bead angr-5gf0s):
        // a reserved hook address that a proc returning ProcOutcome::CallAndResume
        // makes the guest routine return to. Recognized by name in
        // handle_simprocedure; never lifted (is_hooked fires first).
        let resume_sentinel = crate::procedures::native_resume_sentinel(
            self.environment.calling_convention.pointer_size(),
        );
        interp.register_simprocedure(
            resume_sentinel,
            crate::procedures::NATIVE_RESUME_SENTINEL_NAME.to_string(),
            0,
            false,
        );

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
        let (result, _blocks_executed, deferred_forks) = interp.run_until_event(
            py,
            callbacks,
            steps_limit,
            &self.stop_addrs,
            self.block_granular,
        );

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
        Python::attach(|py| {
            if let Err(e) = cb.call_inspect_fork(py, forked_state_id as i64, "after") {
                log::debug!(
                    "fork inspect dispatch raised (state {}): {}",
                    forked_state_id,
                    e
                );
            }
        });
    }
}

#[cfg(test)]
#[path = "sizes_tests.rs"]
mod sizes;

#[cfg(test)]
#[path = "subcall_tests.rs"]
mod subcall_tests;
