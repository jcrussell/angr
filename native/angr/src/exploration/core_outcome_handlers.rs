//! Post-step classification / fork-materialization handler arms.
//!
//! Extracted from `core_outcome.rs` (angr-nbim4.3). Each `*_core` fn is one
//! arm of the `run_post_step_core` dispatcher (kept in the parent module); the
//! fns the dispatcher calls are `pub(super)`, the rest are module-private.
//! `use super::*` inherits the parent's imports plus the private `ForkPayload`
//! / `ForkSink` / `SimProcCall` helper structs (visible to this descendant).

use super::*;
use crate::memory::MemoryError;

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
        last_condition,
        stored_conditions,
        fork_snapshots,
    } = payload;
    CoreOutcome {
        ret: CoreReturn::NeedsPython(PendingBounce {
            kind,
            state,
            deferred_forks,
            last_condition,
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

    for fork in deferred_forks {
        if let Some(condition) = stored_conditions.get(&fork.condition_id) {
            super::super::helpers::add_fork_guard_constraint(
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
            let mut forked = super::super::helpers::build_unexplored_fork(
                base,
                &fork,
                condition,
                &mut fork_snapshots,
            );
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

    for fork in &deferred_forks {
        if let Some(condition) = stored_conditions.get(&fork.condition_id) {
            super::super::helpers::add_fork_guard_constraint(
                cc.callbacks,
                base,
                condition,
                fork.path_taken,
            );

            let forked = super::super::helpers::build_unexplored_fork(
                base,
                fork,
                condition,
                &mut fork_snapshots,
            );
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

/// What the native fast path decided, so the follow-up runs after the borrow of
/// the registry is released (mirror of `stepping::NativeProcDisposition`).
enum NativeProcDisposition {
    Returned {
        no_return: bool,
        ret_val: Option<RustBV>,
    },
    SubCall {
        proc_name: String,
        saved_args: Vec<RustBV>,
        target: u64,
        sub_args: Vec<RustBV>,
        resume_tag: u32,
    },
    Fallback,
    /// The native proc touched an unmapped page while STRICT_PAGE_ACCESS is on:
    /// Python's `PrivilegedPagingMixin._initialize_page` would raise
    /// `SimSegfaultException`, so the state is terminal-errored natively instead
    /// of bouncing to Python just to raise. Carries the message Python formats
    /// (`"{page_addr:#x} (unmapped)"`).
    Segfault(String),
}

/// Map a native procedure error onto the `SimSegfaultException` Python would
/// raise for the same access, or `None` when Python would service it (and we
/// must therefore fall back).
///
/// Only the unmapped-page case is mirrored: with STRICT_PAGE_ACCESS off Python
/// lazily initializes the page and keeps going, so the state must still bounce.
pub(super) fn segfault_message(state: &RustSimState, err: &ProcedureError) -> Option<String> {
    if !state.enforce_permissions() {
        return None;
    }
    match err {
        ProcedureError::Memory(MemoryError::Unmapped { addr, .. }) => {
            let page_addr = addr & !(crate::memory::PAGE_SIZE - 1);
            Some(format!("{page_addr:#x} (unmapped)"))
        }
        _ => None,
    }
}

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
            match ctx.cc.extract_procedure_args(&state, num_args) {
                Err(e) => {
                    log::debug!("Skipping native procedure {name} (arg extraction failed: {e:?})");
                    counters.native_python_fallbacks += 1;
                    *counters
                        .other_fallbacks_by_name
                        .entry(name.clone())
                        .or_insert(0) += 1;
                    NativeProcDisposition::Fallback
                }
                Ok(args) => match native_proc.call_ex(&mut state, &args) {
                    Ok(outcome) => {
                        counters.native_calls += 1;
                        *counters.call_counts.entry(name.clone()).or_insert(0) += 1;
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
                        if let Some(msg) = segfault_message(&state, &e) {
                            log::debug!("Native procedure {name} segfaulted: {msg}");
                            counters.native_calls += 1;
                            *counters.call_counts.entry(name.clone()).or_insert(0) += 1;
                            NativeProcDisposition::Segfault(msg)
                        } else {
                            log::debug!(
                                "Native procedure {name} returned error, falling back to Python: {e:?}"
                            );
                            counters.native_python_fallbacks += 1;
                            let bucket = match e {
                                ProcedureError::SymbolicArgument(_) => {
                                    &mut counters.symbolic_fallbacks_by_name
                                }
                                ProcedureError::NotImplemented => {
                                    &mut counters.not_implemented_fallbacks_by_name
                                }
                                _ => &mut counters.other_fallbacks_by_name,
                            };
                            *bucket.entry(name.clone()).or_insert(0) += 1;
                            NativeProcDisposition::Fallback
                        }
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
                    state.set_register_by_offset(ctx.cc.return_register, rv);
                }
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
            Ok(()) => Some(false),
            Err(e) => {
                log::debug!(
                    "native sub-call setup failed ({e:?}); falling back to Python for {name}"
                );
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
        let n_args = handler.num_args();
        let args = if n_args == 0 {
            Ok(Vec::new())
        } else {
            ctx.cc.extract_syscall_args(&state, n_args)
        };
        let Ok(args) = args else {
            log::debug!(
                "Skipping native syscall (arg extraction failed): {:?}",
                args.unwrap_err()
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
            Err(_) => {
                // Fall through to Python callback path below.
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
