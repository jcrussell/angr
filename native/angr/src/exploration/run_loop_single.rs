//! The single-threaded run loop: the always-compiled default driver.
//!
//! `run_loop_single_threaded` owns the loop frame (I8 termination, state pop,
//! post-step bookkeeping) and delegates the entire per-state body to `step_one`,
//! the **single stepping decision point** — find/avoid predicates, SimProcedure
//! dispatch (native fast path or Python fallback), the interpreter step,
//! deferred forks at find/avoid addresses — which classifies the result into a
//! [`StepOutcome`] the driver routes.
//!
//! Split out of `run_loop.rs` (angr-9ke6b.49) — no behavior change.
//!
//! **Panic policy / lint enforcement:** identical to [`run_loop`]
//! — the crate ships with `panic = "abort"`, so a `MutexGuard` can never be
//! poisoned by an unwind, and every `.expect()` here is a poison /
//! session-live / pool-set invariant guard carrying a narrow
//! `#[allow(clippy::expect_used, reason = ...)]`. This module re-states the
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]` so a *new* fallible
//! unwrap must still be justified.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::*;

use super::core_outcome::NativeSubcall;
use super::helpers::advance_sp_past_return_addr;
use super::native_proc_dispatch::{
    NativeProcCounters, NativeProcDisposition, dispatch_native_proc,
};

use super::run_loop::{StepOutcome, TerminalStep};

impl RustExplorationManager {
    /// Inner body of the pymethods-exposed `run`. See `run` in `mod.rs`.
    ///
    /// Thin driver over `step_one`: owns the loop frame (I8 termination,
    /// state pop, post-step bookkeeping) and routes each `StepOutcome`.
    pub(crate) fn run_loop_single_threaded(
        &mut self,
        n: Option<u32>,
    ) -> PyResult<ExplorationEvent> {
        // A previous wave may have parked the tail of its bounce queue in
        // `pending_parallel_bounces` (states living in NO stash). Only the two
        // parallel loops drain that queue, so a `run()` that routes here
        // instead — worker count dropped to 1, a native technique registered,
        // or a callable find/avoid predicate set between calls — would strand
        // those states permanently and silently (angr-05kiw). Replay them into
        // STASH_ACTIVE first so every route consumes the queue.
        self.flush_parked_bounces_to_active();

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

        // GIL-work timing (angr-1ilq.7): bracket the whole run-loop wall time —
        // the denominator for the GIL/solver fractions. While live it also arms
        // the GIL-region timer (see `gil_profile`), so only Python-touch work
        // *during stepping* is counted (state export after `run()` is excluded,
        // keeping `gil_work_ns <= run_wall_ns`). The `Drop` fires on every exit
        // path (early `return`, `?`, panic) without borrowing `self`.
        let _gil_wall = crate::gil_profile::RunLoopWallGuard::new(self.profiling.profiling_enabled);

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

            // Get next state from active stash via the selection policy
            // (angr-a32jl.1): Fifo=pop_front (BFS), Lifo=pop_back (DFS).
            // `policy` is bound first so the closure captures only that field,
            // leaving `self.sm` free to borrow mutably.
            let policy = &*self.policy;
            let state = match self.sm.get_mut(STASH_ACTIVE).and_then(|s| policy.select(s)) {
                Some(s) => {
                    let sid = s.state_id();
                    self.current_stepping_state_id = Some(sid.into());
                    STEPPING_STATE_ID.with(|cell| cell.set(Some(sid)));
                    // angr-panhl.1: model a work-stealing migration at dispatch.
                    self.record_migration_sample(sid);
                    // SI-B (angr-1ilq.3 increment 2b'): the REAL analog of the
                    // MODEL sample above — measure the migration serde tax on
                    // a fork of this pre-step state and discard the result.
                    // No-op unless RUST_PARALLEL_SHADOW_PROBE is set; when it
                    // is, `shadow_probe_migrate` forks `s` internally so the
                    // live state itself is never mutated (see its doc).
                    #[cfg(feature = "vex-engine-z3")]
                    self.shadow_probe_migrate(&s);
                    s
                }
                None => {
                    // No active states.
                    // I8 termination path (b): active stash exhausted. Always
                    // signal `active_empty` — `found_count()` is provably <
                    // `num_find` here (path (a) at the loop top returns `found`
                    // for `>= num_find` BEFORE this drain), so emitting `found`
                    // with a partial count spins the Python explore loop forever
                    // (angr-q1mwl): its found-break needs `found_count >=
                    // num_find`, unreachable for a partial, and active never
                    // refills. The found stash still carries any partial
                    // solutions. See module header.
                    return Ok(ExplorationEvent::active_empty(
                        self.found_count(),
                        self.steps,
                    ));
                }
            };

            match self.step_one(&callbacks, state)? {
                // Pre-step / find-avoid policy already routed the state. Advance
                // without post-step bookkeeping (former `continue` paths).
                StepOutcome::Routed => continue,
                // Build the event from the LOCAL `pending` (so the
                // PythonVEXFallback counter mutations are free of a borrow
                // conflict), THEN store it. Mirrors the former order exactly.
                StepOutcome::NeedCallback(pending) => {
                    let event = self.callback_event(&pending);
                    self.pending_callbacks
                        .insert(StateId::new(pending.state.state_id()), pending);
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
        callbacks: &PythonCallbacks,
        mut state: RustSimState,
    ) -> PyResult<StepOutcome> {
        // Check find/avoid before stepping
        let pc = state.pc();

        // Check if callable avoid predicate needs Python evaluation.
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

        // Check if callable find predicate needs Python evaluation.
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
                log::debug!("State at find address 0x{pc:x} is UNSAT, pruning");
                self.push_or_drop_terminal(STASH_PRUNED, state);
            }
            return Ok(StepOutcome::Routed);
        }

        // Check hooks (SimProcedures)
        // GAP 6: stack-based skip tracking for zero-length hooks. Expires
        // stale entries and pops at most one token for `pc`; see
        // `consume_skip_hook`.
        let should_skip_hook = self.consume_skip_hook(pc);
        if self.hooks.contains(&pc) && !should_skip_hook {
            // Check if this is a registered SimProcedure
            if let Some((name, num_args, no_return)) = self.simprocedures.get(&pc).cloned() {
                // Skip native for main-object hooks (user-placed `proj.hook()`
                // overrides); see `execution_env::prefer_native_dispatch`.
                let prefer_native = self.environment.prefer_native_dispatch(pc);
                // Try native procedure first (only for external/library hooks)
                if prefer_native
                    && let Some(native_proc) = self.native_procedures.get(&name).cloned()
                {
                    // Extract arguments from state registers (and stack when
                    // num_args exceeds the register count). On failure (symbolic
                    // SP, unmapped stack slot) `dispatch_native_proc` skips the
                    // native fast path and falls through to the Python
                    // SimProcedure callback below — handing the native handler
                    // fabricated zeros would mask the underlying stack-setup bug.
                    //
                    // `num_args` is the Python SimProcedure's FIXED-arg count
                    // (variadics excluded). Native procs that consume variadic
                    // pointers (scanf family) declare a larger `num_args()`; use
                    // the max so `extract_procedure_args` reads the full window.
                    // Truncating to the Python count made the scanf family a
                    // silent no-op end-to-end (angr-8onrp).
                    //
                    // The registry entry is cloned (cheap `Arc`) so the shared
                    // dispatcher can take `&mut self.profiling` alongside it.
                    let native_num_args = num_args.max(native_proc.num_args());
                    let args = self.extract_procedure_args(&state, native_num_args);
                    let stats = &mut self.profiling.native_proc_stats;
                    let disposition = dispatch_native_proc(
                        native_proc.as_ref(),
                        &mut state,
                        &name,
                        no_return,
                        args,
                        // Unlike the parallel mirror, a strict-page-access fault
                        // still bounces to Python here (angr-ph300.73 tracks
                        // unifying the two).
                        false,
                        &mut NativeProcCounters {
                            native_calls: &mut stats.native_calls,
                            python_fallbacks: &mut stats.python_fallbacks,
                            call_counts: &mut stats.call_counts,
                            symbolic_fallbacks_by_name: &mut stats.symbolic_fallbacks_by_name,
                            not_implemented_fallbacks_by_name: &mut stats
                                .not_implemented_fallbacks_by_name,
                            other_fallbacks_by_name: &mut stats.other_fallbacks_by_name,
                        },
                    );
                    match disposition {
                        NativeProcDisposition::Returned { no_return, ret_val } => {
                            // For no-return procedures (exit/abort), skip the
                            // return-address dance and deadend directly. Setting
                            // PC to a stack-derived return address can produce a
                            // spurious successor (e.g. when exit is called from
                            // rejected() in fauxware, the post-call address
                            // happens to overlap main's start, causing infinite
                            // re-entry).
                            if no_return {
                                self.push_or_drop_terminal(STASH_DEADENDED, state);
                                return Ok(StepOutcome::Routed);
                            }

                            // Set return value if present
                            if let Some(rv) = ret_val {
                                let ret_reg = self.environment.calling_convention.return_register();
                                state.set_register_by_offset(ret_reg, rv);
                            }

                            // Get return address and set PC. Use the state's real
                            // register file so that LR/X30/$ra overrides see
                            // actual values; passing a blank RegisterFile here
                            // used to make ARM/ARM64/MIPS read LR=0 and set PC to
                            // 0.
                            let ctx = state.solver().borrow();
                            let ret_addr_opt = self.environment.calling_convention.get_return_addr(
                                state.registers(),
                                None,
                                &ctx,
                            );
                            let pops_return_addr =
                                self.environment.calling_convention.pops_return_addr();
                            drop(ctx);
                            if let Some(ret_addr) = ret_addr_opt {
                                // Only adjust SP for stack-based ABIs
                                // (x86/AMD64). ARM/ARM64/MIPS keep ret addr in a
                                // register and leave SP untouched — the shared
                                // helper gates that, and is also what keeps this
                                // site and `handle_simprocedure_core` from
                                // drifting apart (angr-c7xno.29).
                                advance_sp_past_return_addr(&mut state, pops_return_addr);
                                state.set_pc(ret_addr);
                            } else if pops_return_addr {
                                // Fallback: read ret addr from [sp] for
                                // stack-based ABIs (only useful when the calling
                                // convention's get_return_addr declined to read
                                // memory itself).
                                if let Some(sp) = state.get_sp().as_u64()
                                    && let Ok(ret_bv) = state.memory_load(sp, state.arch().bytes())
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
                        NativeProcDisposition::SubCall {
                            proc_name,
                            saved_args,
                            target,
                            sub_args,
                            resume_tag,
                        } => {
                            // The proc requested a guest sub-call. Capture the
                            // caller return address from [sp] BEFORE
                            // `setup_native_subcall` overwrites that slot with the
                            // resume sentinel. A symbolic SP / unmapped slot
                            // (None) or a setup failure falls back to Python (the
                            // state is left untouched by setup on Err). Do NOT pop
                            // SP or honour `no_return` here: the guest's own `ret`
                            // advances SP, and `handle_native_resume` finishes
                            // without re-adjusting it.
                            let setup = match self.get_return_addr(&state) {
                                Some(caller_return_addr) => self
                                    .setup_native_subcall(
                                        &mut state,
                                        NativeSubcall {
                                            proc_name,
                                            saved_args,
                                            caller_return_addr,
                                            target,
                                            sub_args,
                                            resume_tag,
                                        },
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
                                        "native sub-call setup failed ({reason}); \
                                         falling back to Python for {name}"
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
                        NativeProcDisposition::Fallback => {}
                        NativeProcDisposition::Segfault(_) => {
                            unreachable!("mirror_segfault=false never yields Segfault")
                        }
                    }
                } // if prefer_native

                // Fall back to Python for SimProcedure execution
                self.simprocedure_python_fallback_count += 1;
                *self
                    .simprocedure_fallback_by_name
                    .entry(name.clone())
                    .or_insert(0) += 1;
                let return_addr =
                    self.get_return_addr_or_log(&state, "Python SimProcedure fallback");

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
                    ForkBundle::empty(),
                );

                return Ok(StepOutcome::NeedCallback(pending));
            }
        }

        // Step the state, passing the skip_hook_addr if we just skipped
        let skip_addr_for_step = if should_skip_hook { Some(pc) } else { None };
        match self.step_state_with_skip(callbacks, state, skip_addr_for_step) {
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
                    let profiling_enabled = self.profiling.profiling_enabled;
                    let lazy_solves = self.constraint_solver.lazy_solves;
                    let materialized = super::fork_materialize::materialize_deferred_forks(
                        pending.deferred_forks,
                        super::fork_materialize::MaterializeForkCtx {
                            fork_base: &fork_base,
                            stored_conditions: &pending.stored_conditions,
                            snapshots: &mut snapshots,
                            lazy_solves,
                            // The taken-path guard lands on the find/avoid state
                            // itself (mirrors the BlockEnd handling).
                            guard_sink: Some(&pending.state),
                            stats: profiling_enabled
                                .then_some(&mut self.profiling.accumulated_stats),
                        },
                    );
                    for forked in &materialized.unsat {
                        // Lineage is registered for UNSAT forks too, then the
                        // state is dropped (this path has never had a pruned
                        // stash push).
                        self.sm.set_root(forked.state_id(), root_state_id);
                    }
                    for forked in materialized.sat {
                        self.sm.set_root(forked.state_id(), root_state_id);
                        self.push_to_active_or_drop(forked);
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
                            log::debug!("State at find address 0x{addr:x} is UNSAT, pruning");
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
                Ok(StepOutcome::Terminal(TerminalStep::Deadended(state)))
            }
            Err(StepError::Error(state, message)) => {
                let pc = state.pc();
                let state_id = state.state_id();
                Ok(StepOutcome::Terminal(TerminalStep::Errored {
                    state,
                    pc,
                    message,
                    state_id,
                }))
            }
            Err(StepError::Unconstrained(state, forks)) => {
                Ok(StepOutcome::Terminal(TerminalStep::Unconstrained {
                    state,
                    forks,
                }))
            }
        }
    }

    /// Apply a terminal disposition to the stashes, reproducing each original
    /// per-stash push path (and its side effects) byte-for-byte. Called by the
    /// driver, which then runs post-step bookkeeping.
    pub(crate) fn apply_terminal(&mut self, disposition: TerminalStep) {
        match disposition {
            TerminalStep::Deadended(state) => {
                self.push_or_drop_terminal(STASH_DEADENDED, state);
            }
            TerminalStep::Errored {
                state,
                pc,
                message,
                state_id,
            } => {
                self.errors.push((pc, message, state_id));
                self.push_errored(state);
            }
            TerminalStep::Unconstrained { state, forks } => {
                // State has too many symbolic jump targets - move to unconstrained stash
                log::debug!("State {} moved to unconstrained stash", state.state_id());
                self.push_or_drop_terminal(STASH_UNCONSTRAINED, state);
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
                SimProcCall {
                    addr: *addr,
                    name: name.clone(),
                    num_args: *num_args,
                    return_addr: *return_addr,
                },
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
                             0x{addr:x} (state {state_id}); falling back to Python VEX engine"
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

test_submod!(z3 "run_loop_single_tests.rs" => tests);
