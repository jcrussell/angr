//! Post-step classification / fork-materialization handler arms.
//!
//! Extracted from `core_outcome.rs` (angr-nbim4.3). Each `*_core` fn is one
//! arm of the `run_post_step_core` dispatcher (kept in the parent module); the
//! fns the dispatcher calls are `pub(super)`, the rest are module-private.
//! `use super::*` inherits the parent's imports plus the private `ForkPayload`
//! / `ForkSink` / `SimProcCall` helper structs (visible to this descendant).
//!
//! **Panic policy / enforcement (angr-qwyti.11, angr-9ke6b.212):** these arms
//! classify guest-derived step results, so the parent [`core_outcome`](super)
//! module's `#![deny(clippy::unwrap_used, clippy::expect_used)]` reaches this
//! file and there are no `unwrap`/`expect` sites left in it; the deny is
//! restated below so the guarantee is visible when reading this file alone.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::*;

/// Package a Python-bouncing outcome (no fork materialization here — the
/// deferred forks ride into the `PendingCallback` the coordinator builds).
pub(super) fn bounce(
    kind: BounceKind,
    state: RustSimState,
    payload: ForkPayload,
    counters: CoreCounters,
    root_hint: u64,
) -> CoreOutcome {
    let ForkPayload {
        deferred_forks,
        stored_conditions,
        fork_snapshots,
        ..
    } = payload;
    CoreOutcome {
        ret: CoreReturn::NeedsPython(PendingBounce {
            kind,
            state,
            deferred_forks,
            stored_conditions,
            fork_snapshots,
        }),
        pruned: Vec::new(),
        fork_ids: Vec::new(),
        terminal_pushes: Vec::new(),
        counters,
        root_hint,
    }
}

/// Mirror of `materialize_deferred_forks`: SAT forks -> `forks_out` (tagged
/// fork), UNSAT -> `pruned_out`; every minted fork id -> `fork_ids_out`
/// (dispatch order). Solver timing -> `prof`.
pub(super) fn materialize_deferred_forks_core(
    cc: &CoreCtx,
    base: &RustSimState,
    payload: ForkPayload,
    root_hint: u64,
    force_eager: bool,
    sink: ForkSink,
) {
    let ctx = cc.ctx;
    let prof = cc.prof;
    let ForkPayload {
        deferred_forks,
        stored_conditions,
        mut fork_snapshots,
        ..
    } = payload;
    let ForkSink {
        forks: forks_out,
        pruned: pruned_out,
        fork_ids: fork_ids_out,
    } = sink;
    let deferred_fork_start = if ctx.profiling_enabled {
        Some(Instant::now())
    } else {
        None
    };
    let deferred_fork_total = deferred_forks.len() as u64;
    // See `PriorGuards` (angr-62ar5): `base` accumulates each taken-path guard
    // below, but a snapshot-built fork does not.
    let mut prior_guards = super::super::fork_materialize::PriorGuards::new(true);

    for fork in deferred_forks {
        if let Some(condition) = stored_conditions.get(&fork.condition_id) {
            super::super::fork_materialize::add_fork_guard_constraint(
                cc.callbacks,
                base,
                condition,
                fork.path_taken,
            );

            let fork_start = if ctx.profiling_enabled {
                Some(Instant::now())
            } else {
                None
            };
            let mut forked = super::super::fork_materialize::build_unexplored_fork(
                base,
                &fork,
                condition,
                &mut fork_snapshots,
                &prior_guards,
            );
            prior_guards.record(condition.clone(), fork.path_taken);
            if force_eager {
                forked.set_force_eager_forks(true);
            }
            if let Some(start) = fork_start {
                ParallelProfiling::add(
                    &prof.solver_fork_time_ns,
                    start.elapsed().as_nanos() as u64,
                );
                ParallelProfiling::add(&prof.solver_fork_count, 1);
            }
            // set_root + dispatch_fork_inspect deferred to the coordinator.
            fork_ids_out.push(forked.state_id());

            let sat_start = if ctx.profiling_enabled {
                Some(Instant::now())
            } else {
                None
            };
            if ctx.lazy_solves || forked.satisfiable() {
                if let Some(start) = sat_start {
                    ParallelProfiling::add(
                        &prof.solver_sat_time_ns,
                        start.elapsed().as_nanos() as u64,
                    );
                    ParallelProfiling::add(&prof.solver_sat_count, 1);
                }
                forks_out.push((forked, RoutingTag::fork(root_hint)));
            } else {
                if let Some(start) = sat_start {
                    ParallelProfiling::add(
                        &prof.solver_sat_time_ns,
                        start.elapsed().as_nanos() as u64,
                    );
                    ParallelProfiling::add(&prof.solver_sat_count, 1);
                }
                log::debug!(
                    "P13: Deferred fork at 0x{:x} is UNSAT, will be pruned",
                    fork.unexplored_target
                );
                pruned_out.push(forked);
            }
        } else {
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
            fork_ids_out.push(forked.state_id());

            if ctx.lazy_solves || forked.satisfiable() {
                forks_out.push((forked, RoutingTag::fork(root_hint)));
            } else {
                log::debug!(
                    "P13: Unconstrained fork at 0x{:x} is UNSAT, will be pruned",
                    fork.unexplored_target
                );
                pruned_out.push(forked);
            }
        }
    }
    if let Some(start) = deferred_fork_start {
        ParallelProfiling::add(
            &prof.deferred_fork_time_ns,
            start.elapsed().as_nanos() as u64,
        );
        ParallelProfiling::add(&prof.deferred_fork_count, deferred_fork_total);
    }
}

/// Mirror of `process_deferred_forks_into` (no fork/sat timers; the final
/// `deferred_fork_count` bump is UNCONDITIONAL, matching the legacy site).
fn process_deferred_forks_into_core(
    cc: &CoreCtx,
    base: &RustSimState,
    payload: ForkPayload,
    root_hint: u64,
    sink: ForkSink,
) {
    let ctx = cc.ctx;
    let prof = cc.prof;
    let ForkPayload {
        deferred_forks,
        stored_conditions,
        mut fork_snapshots,
        ..
    } = payload;
    let ForkSink {
        forks: forks_out,
        pruned: pruned_out,
        fork_ids: fork_ids_out,
    } = sink;
    if deferred_forks.is_empty() {
        return;
    }

    let mut prior_guards = super::super::fork_materialize::PriorGuards::new(true);
    for fork in &deferred_forks {
        if let Some(condition) = stored_conditions.get(&fork.condition_id) {
            super::super::fork_materialize::add_fork_guard_constraint(
                cc.callbacks,
                base,
                condition,
                fork.path_taken,
            );

            let forked = super::super::fork_materialize::build_unexplored_fork(
                base,
                fork,
                condition,
                &mut fork_snapshots,
                &prior_guards,
            );
            prior_guards.record(condition.clone(), fork.path_taken);
            fork_ids_out.push(forked.state_id());

            if ctx.lazy_solves || forked.satisfiable() {
                forks_out.push((forked, RoutingTag::fork(root_hint)));
            } else {
                pruned_out.push(forked);
            }
        } else {
            let mut forked = base.fork();
            forked.set_pc(fork.unexplored_target);
            fork_ids_out.push(forked.state_id());

            if ctx.lazy_solves || forked.satisfiable() {
                forks_out.push((forked, RoutingTag::fork(root_hint)));
            } else {
                pruned_out.push(forked);
            }
        }
    }

    ParallelProfiling::add(&prof.deferred_fork_count, deferred_forks.len() as u64);
}

/// Worker-side helper (angr-vh834 Phase 5): turn the deferred forks that ride
/// into a `NeedsPython` bounce into real, migratable fork states so the parallel
/// wave loop can keep exploring them locally instead of losing them across the
/// bounce boundary (the loose `RustBV` conditions are `!Send` and cannot cross
/// a thread, but the materialized fork *states* can).
///
/// Uses `base` (the bounce state, PC parked at the bounce point, before the
/// Python handler runs) as the fork base — the same logical base the
/// single-threaded resume path uses at simproc-return time, so the forks are
/// identical. Returns `(sat forks, unsat/pruned forks, minted fork ids)`.
#[allow(clippy::type_complexity)]
pub(crate) fn materialize_bounce_forks(
    cc: &CoreCtx,
    base: &RustSimState,
    deferred_forks: Vec<DeferredFork>,
    stored_conditions: FxHashMap<u64, RustBV>,
    fork_snapshots: FxHashMap<u64, BranchSnapshot>,
    root_hint: u64,
) -> (Vec<RustSimState>, Vec<RustSimState>, Vec<u64>) {
    let payload = ForkPayload {
        deferred_forks,
        last_condition: None,
        stored_conditions,
        fork_snapshots,
    };
    let mut forks_out: Vec<(RustSimState, RoutingTag)> = Vec::new();
    let mut pruned_out: Vec<RustSimState> = Vec::new();
    let mut fork_ids_out: Vec<u64> = Vec::new();
    process_deferred_forks_into_core(
        cc,
        base,
        payload,
        root_hint,
        ForkSink {
            forks: &mut forks_out,
            pruned: &mut pruned_out,
            fork_ids: &mut fork_ids_out,
        },
    );
    (
        forks_out.into_iter().map(|(s, _)| s).collect(),
        pruned_out,
        fork_ids_out,
    )
}

/// Resolve an eager-mode symbolic branch natively (angr-gorvf.14).
///
/// The interpreter only emits `RunResult::SymbolicBranch` when deferred forks
/// are off (`ExecutionConfig::use_deferred_forks == false` — the angr-027h
/// phase-2 eager retry), where a symbolic guard forks *both* directions at
/// once instead of deferring one. This used to park the state and bounce to
/// Python, but the Python handler did no work the core cannot do: it re-derived
/// the guard as a claripy AST only to hand back constraints that
/// `resume_after_symbolic_branch` explicitly ignored (the guard is sourced from
/// `stored_conditions`). So we fork here and stay in Rust.
///
/// Both children need a real sat check: the non-deferred `IRStmt::Exit` path in
/// `statements.rs` returns `SymbolicBranch` *without* calling
/// `check_branch_feasibility`, so only the guard's symbolic-ness is established
/// — neither direction is known feasible (angr-3ag1l).
pub(super) fn handle_symbolic_branch_core(
    cc: &CoreCtx,
    state: RustSimState,
    branch: SymBranch,
    mut payload: ForkPayload,
    counters: CoreCounters,
    root_hint: u64,
) -> CoreOutcome {
    let SymBranch {
        condition_id,
        true_target,
        false_target,
    } = branch;
    // The interpreter stores the guard under `condition_id`; `last_condition` is
    // the same value carried out-of-band, so only fill the gap.
    if let Some(cond) = payload.last_condition.take() {
        payload
            .stored_conditions
            .entry(condition_id)
            .or_insert(cond);
    }
    let branch_condition = payload.stored_conditions.get(&condition_id).cloned();

    // Materialize the deferred forks accumulated earlier in this step FIRST, so
    // they fork off the state as it was *before* the branch guard is asserted
    // (their own guards are applied to `state` as we go, which both children
    // then inherit).
    let mut forks_out = Vec::new();
    let mut pruned = Vec::new();
    let mut fork_ids = Vec::new();
    process_deferred_forks_into_core(
        cc,
        &state,
        payload,
        root_hint,
        ForkSink {
            forks: &mut forks_out,
            pruned: &mut pruned,
            fork_ids: &mut fork_ids,
        },
    );

    let mut true_state = state.fork();
    true_state.set_pc(true_target);
    let mut false_state = state;
    false_state.set_pc(false_target);
    if let Some(ref cond) = branch_condition {
        true_state.solver().borrow().assume_true(cond);
        false_state.solver().borrow().assume_false(cond);
    }

    let mut succ = Vec::with_capacity(forks_out.len() + 2);
    // The false child keeps the stepped state's id (it is the fallthrough), so
    // it routes as the main successor; the true child is a freshly minted fork.
    for (child, tag) in [
        (false_state, RoutingTag::main()),
        (true_state, RoutingTag::fork(root_hint)),
    ] {
        if cc.ctx.lazy_solves || child.satisfiable() {
            succ.push((child, tag));
        } else {
            pruned.push(child);
        }
    }
    succ.extend(forks_out);

    CoreOutcome {
        ret: CoreReturn::Continue(succ),
        pruned,
        fork_ids,
        terminal_pushes: Vec::new(),
        counters,
        root_hint,
    }
}

/// Mirror of `handle_symbolic_jump_target`.
pub(super) fn handle_symbolic_jump_target_core(
    cc: &CoreCtx,
    state: RustSimState,
    targets: Vec<u64>,
    condition_id: u64,
    payload: ForkPayload,
    counters: CoreCounters,
    root_hint: u64,
) -> CoreOutcome {
    let target_expr = payload.stored_conditions.get(&condition_id).cloned();

    if targets.is_empty() {
        return CoreOutcome {
            ret: CoreReturn::Deadended(state),
            pruned: Vec::new(),
            fork_ids: Vec::new(),
            terminal_pushes: Vec::new(),
            counters,
            root_hint,
        };
    }

    let keep_ip_symbolic = state.keep_ip_symbolic();
    let mut pruned = Vec::new();
    let mut fork_ids = Vec::new();

    if targets.len() == 1 {
        let mut first = state;
        let addr = targets[0];
        if let Some(ref expr) = target_expr {
            if keep_ip_symbolic {
                first.set_pc(addr);
                first.set_ip(expr.clone());
            } else {
                let concrete = RustBV::concrete(addr as u128, expr.width());
                let constraint = expr.eq(&concrete, &first.solver().borrow());
                first.add_constraint(constraint);
                first.set_pc(addr);
            }
        } else {
            first.set_pc(addr);
        }
        let mut forks_out = Vec::new();
        process_deferred_forks_into_core(
            cc,
            &first,
            payload,
            root_hint,
            ForkSink {
                forks: &mut forks_out,
                pruned: &mut pruned,
                fork_ids: &mut fork_ids,
            },
        );
        let mut succ = Vec::with_capacity(forks_out.len() + 1);
        succ.push((first, RoutingTag::main()));
        succ.extend(forks_out);
        return CoreOutcome {
            ret: CoreReturn::Continue(succ),
            pruned,
            fork_ids,
            terminal_pushes: Vec::new(),
            counters,
            root_hint,
        };
    }

    // Multiple targets - fork each from the UNCONSTRAINED original.
    let base_state = state.fork();

    let first_addr = targets[0];
    let mut first_state = state;
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

    // Target forks (set_root via tag; NOT dispatched in the legacy path).
    let mut target_forks: Vec<(RustSimState, RoutingTag)> = Vec::new();
    for &addr in targets.iter().skip(1) {
        let mut forked = base_state.fork();
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
        target_forks.push((forked, RoutingTag::fork(root_hint)));
    }

    let mut deferred_out = Vec::new();
    process_deferred_forks_into_core(
        cc,
        &first_state,
        payload,
        root_hint,
        ForkSink {
            forks: &mut deferred_out,
            pruned: &mut pruned,
            fork_ids: &mut fork_ids,
        },
    );

    let mut succ = Vec::with_capacity(1 + target_forks.len() + deferred_out.len());
    succ.push((first_state, RoutingTag::main()));
    succ.extend(target_forks);
    succ.extend(deferred_out);
    CoreOutcome {
        ret: CoreReturn::Continue(succ),
        pruned,
        fork_ids,
        terminal_pushes: Vec::new(),
        counters,
        root_hint,
    }
}

// The proc-dispatch decision itself lives in `helpers.rs` so `step_one`'s
// serial arm and this parallel one cannot drift (angr-ph300.73).
#[cfg(test)]
pub(super) use super::super::native_proc_dispatch::segfault_message;
use super::super::native_proc_dispatch::{
    NativeProcCounters, NativeProcDisposition, dispatch_native_proc,
};

/// Mirror of `handle_simprocedure` (native fast path + native resume; Python
/// fallback bounces).
pub(super) fn handle_simprocedure_core(
    cc: &CoreCtx,
    counters: &mut CoreCounters,
    mut state: RustSimState,
    call: SimProcCall,
    payload: ForkPayload,
    root_hint: u64,
) -> CoreOutcome {
    let ctx = cc.ctx;
    let native_procs = cc.native_procs;
    let SimProcCall {
        addr,
        name,
        num_args,
        return_addr,
    } = call;
    // Native sub-call resume sentinel: a guest routine returns here.
    if name == NATIVE_RESUME_SENTINEL_NAME {
        return handle_native_resume_core(cc, state, payload, std::mem::take(counters), root_hint);
    }

    let prefer_native = crate::exploration::execution_env::prefer_native_dispatch(
        &ctx.binary_regions,
        ctx.main_object_range,
        ctx.prefer_native_library_hooks,
        addr,
    );

    // A hook that IS an address-based find/avoid target must never run natively
    // (angr-1i5h7). Native dispatch is inline: it runs the proc and lands the
    // state at `return_addr`, so the target address never surfaces at a step
    // boundary and the run-loop find/avoid check never fires — execution sails
    // straight past `explore(find=<hooked libc symbol>)`. Falling back bounces
    // this call to Python, where `step_one`'s NeedCallback special case (and the
    // parallel Bug-C1 route) short-circuits the SimProcedure bounce to
    // FOUND/AVOIDED *before* any procedure body runs. `step_one`'s own inline
    // native path needs no such guard: its pre-step find/avoid check on `pc`
    // already runs before the hook block.
    let is_find_or_avoid = ctx.find_addrs.contains(&addr) || ctx.avoid_addrs.contains(&addr);

    let disposition: NativeProcDisposition = if prefer_native && !is_find_or_avoid {
        if let Some(native_proc) = native_procs.get(&name) {
            let proc_no_return = native_proc.no_return();
            // `num_args` is the Python SimProcedure's FIXED-arg count (variadics
            // excluded). Native procs consuming variadic pointers (scanf family)
            // declare a larger `num_args()`; use the max so the full arg window
            // is read. Truncating made the scanf family a no-op (angr-8onrp).
            let native_num_args = num_args.max(native_proc.num_args());
            let args = ctx.cc.extract_procedure_args(&state, native_num_args);
            // The sub-call's native_call / fallback bump is deferred to the
            // `SubCall` arm below so it agrees with `step_one`: a native call
            // is booked only once `setup_native_subcall` succeeds; a setup
            // failure books a Python fallback instead.
            dispatch_native_proc(
                native_proc.as_ref(),
                &mut state,
                &name,
                proc_no_return,
                args,
                // This path terminal-errors a strict-page-access fault natively
                // rather than bouncing to Python just to raise.
                true,
                &mut NativeProcCounters {
                    native_calls: &mut counters.native_calls,
                    python_fallbacks: &mut counters.native_python_fallbacks,
                    call_counts: &mut counters.call_counts,
                    symbolic_fallbacks_by_name: &mut counters.symbolic_fallbacks_by_name,
                    not_implemented_fallbacks_by_name: &mut counters
                        .not_implemented_fallbacks_by_name,
                    other_fallbacks_by_name: &mut counters.other_fallbacks_by_name,
                },
            )
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
                    state.set_register_by_offset(ctx.cc.return_register, rv);
                }
                state.set_pc(return_addr);
                // Only stack-return ABIs (x86/AMD64) pop the return address, so
                // only they advance SP here. On link-register ABIs (ARM/ARM64
                // LR/X30, MIPS $ra) the caller's return address never went on
                // the stack, and bumping SP would discard a live stack slot
                // (angr-sqfj8.37). `step_one`'s inline native path in
                // `run_loop_single.rs` gates the same bump the same way; keep
                // the two in sync.
                if ctx.cc.pops_return_addr {
                    let sp = state.get_sp().as_u64().unwrap_or(0);
                    let ptr_size = state.arch().bytes() as u64;
                    state.set_sp(RustBV::concrete(
                        (sp + ptr_size) as u128,
                        state.arch().bits(),
                    ));
                }
            }
            Some(no_return)
        }
        NativeProcDisposition::SubCall {
            proc_name,
            saved_args,
            target,
            sub_args,
            resume_tag,
        } => match ctx.cc.setup_native_subcall(
            &mut state,
            NativeSubcall {
                proc_name,
                saved_args,
                caller_return_addr: return_addr,
                target,
                sub_args,
                resume_tag,
            },
        ) {
            Ok(()) => {
                // The native proc ran and its guest sub-call was set up:
                // book it as a native call (mirrors `step_one`).
                counters.native_calls += 1;
                *counters.call_counts.entry(name.clone()).or_insert(0) += 1;
                Some(false)
            }
            Err(e) => {
                // Setup failed (symbolic SP / unmapped slot); we bounce to
                // Python, so classify this as a native->Python fallback in the
                // `other` bucket rather than a completed native call — matching
                // `step_one`'s inline fast path.
                log::debug!(
                    "native sub-call setup failed ({e:?}); falling back to Python for {name}"
                );
                counters.native_python_fallbacks += 1;
                *counters
                    .other_fallbacks_by_name
                    .entry(name.clone())
                    .or_insert(0) += 1;
                None
            }
        },
        NativeProcDisposition::Fallback => None,
        NativeProcDisposition::Segfault(msg) => {
            // Python would have re-run the proc only to raise SimSegfaultException
            // out of it; land the state at the proc address like the Python bounce
            // does and terminal-error it here. Pending forks are dropped, matching
            // the interpreter's own `RunResult::Error` arm.
            state.set_pc(addr);
            return CoreOutcome {
                ret: CoreReturn::Errored(state, msg),
                pruned: Vec::new(),
                fork_ids: Vec::new(),
                terminal_pushes: Vec::new(),
                counters: std::mem::take(counters),
                root_hint,
            };
        }
    };

    if let Some(no_return) = fall_back_to_python {
        let mut forks_out = Vec::new();
        let mut pruned = Vec::new();
        let mut fork_ids = Vec::new();
        process_deferred_forks_into_core(
            cc,
            &state,
            payload,
            root_hint,
            ForkSink {
                forks: &mut forks_out,
                pruned: &mut pruned,
                fork_ids: &mut fork_ids,
            },
        );
        if no_return {
            // Deadend the main state; surviving forks continue.
            CoreOutcome {
                ret: CoreReturn::Continue(forks_out),
                pruned,
                fork_ids,
                terminal_pushes: vec![(state, STASH_DEADENDED)],
                counters: std::mem::take(counters),
                root_hint,
            }
        } else {
            let mut succ = Vec::with_capacity(forks_out.len() + 1);
            succ.push((state, RoutingTag::main()));
            succ.extend(forks_out);
            CoreOutcome {
                ret: CoreReturn::Continue(succ),
                pruned,
                fork_ids,
                terminal_pushes: Vec::new(),
                counters: std::mem::take(counters),
                root_hint,
            }
        }
    } else {
        // Python SimProcedure fallback (counters recorded; bounce builds the
        // PendingCallback after set_pc(addr)+add_to_history(addr)).
        counters.simprocedure_python_fallback_count += 1;
        *counters
            .simprocedure_fallback_by_name
            .entry(name.clone())
            .or_insert(0) += 1;
        bounce(
            BounceKind::SimProcedurePython {
                addr,
                name,
                num_args,
                return_addr,
            },
            state,
            payload,
            std::mem::take(counters),
            root_hint,
        )
    }
}

/// Mirror of `handle_native_resume`.
fn handle_native_resume_core(
    cc: &CoreCtx,
    mut state: RustSimState,
    payload: ForkPayload,
    counters: CoreCounters,
    root_hint: u64,
) -> CoreOutcome {
    let ctx = cc.ctx;
    let native_procs = cc.native_procs;
    let frame = match state.pop_native_resume_frame() {
        Some(f) => f,
        None => {
            log::error!("native resume sentinel hit with empty resume stack; deadending");
            return deadend(state, counters, root_hint);
        }
    };

    let outcome = match native_procs.get(&frame.proc_name) {
        Some(proc) => proc.resume(&mut state, frame.resume_tag, &frame.saved_args),
        None => {
            log::error!(
                "native resume: proc {} not in registry; deadending",
                frame.proc_name
            );
            return deadend(state, counters, root_hint);
        }
    };

    match outcome {
        Ok(ProcOutcome::Return(ret_val)) => {
            if let Some(rv) = ret_val {
                state.set_register_by_offset(ctx.cc.return_register, rv);
            }
            state.set_pc(frame.caller_return_addr);
        }
        Ok(ProcOutcome::CallAndResume {
            target,
            args: sub_args,
            resume_tag,
        }) => {
            if let Err(e) = ctx.cc.setup_native_subcall(
                &mut state,
                NativeSubcall {
                    proc_name: frame.proc_name.clone(),
                    saved_args: frame.saved_args.clone(),
                    caller_return_addr: frame.caller_return_addr,
                    target,
                    sub_args,
                    resume_tag,
                },
            ) {
                log::error!("native resume nested sub-call setup failed ({e:?}); deadending");
                return deadend(state, counters, root_hint);
            }
        }
        Err(e) => {
            log::error!(
                "native resume: {} resume() failed: {:?}; deadending",
                frame.proc_name,
                e
            );
            return deadend(state, counters, root_hint);
        }
    }

    let mut forks_out = Vec::new();
    let mut pruned = Vec::new();
    let mut fork_ids = Vec::new();
    process_deferred_forks_into_core(
        cc,
        &state,
        payload,
        root_hint,
        ForkSink {
            forks: &mut forks_out,
            pruned: &mut pruned,
            fork_ids: &mut fork_ids,
        },
    );
    let mut succ = Vec::with_capacity(forks_out.len() + 1);
    succ.push((state, RoutingTag::main()));
    succ.extend(forks_out);
    CoreOutcome {
        ret: CoreReturn::Continue(succ),
        pruned,
        fork_ids,
        terminal_pushes: Vec::new(),
        counters,
        root_hint,
    }
}

/// Mirror of the `Syscall` arm (native fast path; Python fallback bounces).
pub(super) fn handle_syscall_core(
    cc: &CoreCtx,
    counters: &mut CoreCounters,
    mut state: RustSimState,
    num: Option<u64>,
    pc: u64,
    payload: ForkPayload,
    root_hint: u64,
) -> CoreOutcome {
    let ctx = cc.ctx;
    let native_syscalls = cc.native_syscalls;
    state.set_pc(pc);
    state.add_to_history(pc);

    let dispatch_key: &str = if ctx.os_name == "cgc" {
        "CGC"
    } else {
        state.arch().name()
    };
    let native_handler = num.and_then(|n| native_syscalls.get(dispatch_key, n));
    if let Some(handler) = native_handler {
        let handler_name = handler.name();
        let n_args = handler.num_args();
        let args = if n_args == 0 {
            Ok(Vec::new())
        } else {
            ctx.cc.extract_syscall_args(&state, n_args)
        };
        let args = match args {
            Ok(args) => args,
            Err(err) => {
                // SILENT(cat-a): expected control flow — a syscall whose args
                // the native CC cannot extract is bounced to the Python
                // syscall implementation, which is authoritative. Logged at
                // debug and counted in `syscall_python_fallback_by_num`.
                log::debug!(
                    "Skipping native syscall {handler_name} (num={num:?}) \
                     (arg extraction failed): {err:?}"
                );
                counters.syscall_python_fallback_count += 1;
                *counters
                    .syscall_python_fallback_by_num
                    .entry(num.map(|n| n as i64).unwrap_or(-1))
                    .or_insert(0) += 1;
                return bounce(
                    BounceKind::SyscallPython { num },
                    state,
                    payload,
                    std::mem::take(counters),
                    root_hint,
                );
            }
        };
        let outcome = handler.call(&mut state, &args);
        if outcome.is_ok() {
            counters.syscall_native_count += 1;
            *counters
                .syscall_native_by_num
                .entry(num.map(|n| n as i64).unwrap_or(-1))
                .or_insert(0) += 1;
        }
        match outcome {
            Ok(SyscallOutcome::Continue { ret }) => {
                let ret_reg = ctx.cc.return_register;
                let bits = state.arch().bits();
                let ret_bv = RustBV::concrete(ret as u128, bits);
                ctx.cc.write_syscall_return(&mut state, ret_reg, ret_bv);
                return syscall_continue(
                    cc,
                    state,
                    payload,
                    std::mem::take(counters),
                    root_hint,
                    false,
                );
            }
            Ok(SyscallOutcome::ContinueSymbolic { ret }) => {
                let ret_reg = ctx.cc.return_register;
                ctx.cc.write_syscall_return(&mut state, ret_reg, ret);
                return syscall_continue(
                    cc,
                    state,
                    payload,
                    std::mem::take(counters),
                    root_hint,
                    false,
                );
            }
            Ok(SyscallOutcome::Exit) => {
                return syscall_continue(
                    cc,
                    state,
                    payload,
                    std::mem::take(counters),
                    root_hint,
                    true,
                );
            }
            Err(err) => {
                // SILENT(cat-a): expected control flow — a native handler that
                // declines (unsupported fd, symbolic size, ...) falls through
                // to the Python syscall implementation below, which is
                // authoritative. Logged here rather than at the bounce so the
                // handler label distinguishes "registered handler declined"
                // from "no native handler for this (arch, num)"; counted in
                // `syscall_python_fallback_by_num`.
                log::debug!(
                    "Native syscall {handler_name} (num={num:?}) declined: \
                     {err:?}; falling back to Python"
                );
            }
        }
    }

    counters.syscall_python_fallback_count += 1;
    *counters
        .syscall_python_fallback_by_num
        .entry(num.map(|n| n as i64).unwrap_or(-1))
        .or_insert(0) += 1;
    bounce(
        BounceKind::SyscallPython { num },
        state,
        payload,
        std::mem::take(counters),
        root_hint,
    )
}

/// Shared continue/exit tail for the three native syscall outcomes.
fn syscall_continue(
    cc: &CoreCtx,
    state: RustSimState,
    payload: ForkPayload,
    counters: CoreCounters,
    root_hint: u64,
    exit: bool,
) -> CoreOutcome {
    let mut forks_out = Vec::new();
    let mut pruned = Vec::new();
    let mut fork_ids = Vec::new();
    process_deferred_forks_into_core(
        cc,
        &state,
        payload,
        root_hint,
        ForkSink {
            forks: &mut forks_out,
            pruned: &mut pruned,
            fork_ids: &mut fork_ids,
        },
    );
    if exit {
        CoreOutcome {
            ret: CoreReturn::Continue(forks_out),
            pruned,
            fork_ids,
            terminal_pushes: vec![(state, STASH_DEADENDED)],
            counters,
            root_hint,
        }
    } else {
        let mut succ = Vec::with_capacity(forks_out.len() + 1);
        succ.push((state, RoutingTag::main()));
        succ.extend(forks_out);
        CoreOutcome {
            ret: CoreReturn::Continue(succ),
            pruned,
            fork_ids,
            terminal_pushes: Vec::new(),
            counters,
            root_hint,
        }
    }
}

#[inline]
fn deadend(state: RustSimState, counters: CoreCounters, root_hint: u64) -> CoreOutcome {
    CoreOutcome {
        ret: CoreReturn::Deadended(state),
        pruned: Vec::new(),
        fork_ids: Vec::new(),
        terminal_pushes: Vec::new(),
        counters,
        root_hint,
    }
}
