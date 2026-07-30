//! `&mut self`-free post-interpreter step core (angr-vh834, Phase 1).
//!
//! [`run_post_step_core`] performs the post-interpreter classification and fork
//! materialization that `RustExplorationManager::step_state_with_skip`'s
//! `match step.result` arms used to do inline, but WITHOUT touching `&mut self`,
//! `self.sm`, `self.profiling` (directly), the GIL, or constructing Python
//! objects. It returns a [`CoreOutcome`] carrying every deferred mutation the
//! single-threaded coordinator (or, later, a parallel worker's coordinator)
//! must apply itself.
//!
//! This is the load-bearing foundation for a parallel wave loop: each worker
//! must be able to drive the post-step phase from an owned / `Arc`-shared
//! config bundle rather than reaching back into the manager. The single-threaded
//! path in `stepping.rs` now flows through this function and applies the
//! deferred mutations afterward, byte-for-byte identically.
//!
//! ## Deferred-mutation contract
//!
//! The core NEVER calls `sm.set_root`, `dispatch_fork_inspect`,
//! `push_or_drop_terminal`, nor writes `self.profiling.accumulated_stats`. It
//! instead records:
//! - SAT continue states (incl. the main state) in [`CoreReturn::Continue`],
//!   each carrying a [`RoutingTag`] (`is_fork` distinguishes a freshly minted
//!   fork that needs `set_root` from the moved main/first state that does not).
//! - UNSAT forks in [`CoreOutcome::pruned`] (coordinator pushes to STASH_PRUNED;
//!   each also needs `set_root`).
//! - Every newly minted fork id that the single-threaded path fires
//!   `dispatch_fork_inspect` on, in firing order, in [`CoreOutcome::fork_ids`].
//!   NOTE: symbolic-jump-target forks get `set_root` but are NOT dispatched in
//!   the legacy path, so they appear in `successors`/tags but NOT in `fork_ids`.
//! - Side-effect terminal pushes (the deadended main state of a no-return
//!   procedure / syscall `exit`, which the legacy code pushes via
//!   `push_or_drop_terminal` while still returning the surviving forks) in
//!   [`CoreOutcome::terminal_pushes`].
//! - Manager-level counter deltas (native-proc / syscall / simproc fallback
//!   counters, `deferred_forks_dropped`) in [`CoreCounters`].
//! - Solver fork/sat/deferred timing in the [`ParallelProfiling`] atomics the
//!   coordinator folds into `accumulated_stats`.
//!
//! `root_hint` is stamped by the caller (`sm.root_or_self(parent_state_id)`)
//! BEFORE the post-step phase and copied onto every fork's tag; the coordinator
//! calls `sm.set_root(child_id, root_hint)`. This equals the inline value the
//! legacy `materialize_deferred_forks` computed.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rustc_hash::FxHashMap;

use crate::callbacks::{DeferredFork, PythonCallbacks, RunErrorKind, RunResult};
use crate::interpreter::{BranchSnapshot, ExecutionStats};
use crate::procedures::{
    NATIVE_RESUME_SENTINEL_NAME, NativeProcedureRegistry, ProcOutcome, native_resume_sentinel,
};
use crate::stash::STASH_DEADENDED;
use crate::state::{NativeResumeFrame, RustSimState};
use crate::symbolic::RustBV;
use crate::syscalls::{NativeSyscallRegistry, SyscallOutcome};

use super::callback_types::SimProcCall;
use super::step_core::StepContext;
use super::stepping::SubcallSetupError;

#[path = "core_outcome_cc.rs"]
mod cc;
pub(crate) use cc::CcSnapshot;

#[path = "core_outcome_handlers.rs"]
mod handlers;
pub(crate) use handlers::materialize_bounce_forks;
use handlers::{
    bounce, handle_simprocedure_core, handle_symbolic_branch_core,
    handle_symbolic_jump_target_core, handle_syscall_core, materialize_deferred_forks_core,
};

/// `Send + Sync` atomic mirror of the `accumulated_stats` solver-timing fields
/// the extracted post-step arms increment. The coordinator folds these into
/// `self.profiling.accumulated_stats` after the step (single-threaded: folds
/// immediately, so behavior is identical).
#[derive(Default)]
pub(crate) struct ParallelProfiling {
    pub(crate) solver_fork_time_ns: AtomicU64,
    pub(crate) solver_fork_count: AtomicU64,
    pub(crate) solver_sat_time_ns: AtomicU64,
    pub(crate) solver_sat_count: AtomicU64,
    pub(crate) deferred_fork_time_ns: AtomicU64,
    pub(crate) deferred_fork_count: AtomicU64,
    /// Full per-step interpreter `ExecutionStats` accumulated across the wave's
    /// dispatches (angr-qhkye). Each worker merges its step's whole `step_stats`
    /// here so ALL sum-typed interpreter counters — `lift_time_ns`,
    /// `cache_hit_count`/`cache_miss_count`, `blocks_executed`, the load/store/
    /// expr timings, etc. — surface in `mgr.stats()`, mirroring the
    /// single-threaded `accumulated_stats.merge(&step.step_stats)` path
    /// (`stepping.rs`). Solver fork/sat/deferred counters are NOT set by the
    /// interpreter (they come from the post-step arms into the atomics above),
    /// so folding the full `step_stats` here does not double-count them.
    /// A `Mutex` (not per-field atomics) keeps this exhaustive without hand-
    /// listing every field; the per-step lock is negligible against a VEX step.
    pub(crate) step_stats: Mutex<ExecutionStats>,
}

impl ParallelProfiling {
    #[inline]
    fn add(a: &AtomicU64, v: u64) {
        a.fetch_add(v, Ordering::Relaxed);
    }

    /// Merge one interpreter step's full `ExecutionStats` into the shared
    /// accumulator (angr-qhkye). Called by each parallel worker after a
    /// dispatch, mirroring the single-threaded
    /// `accumulated_stats.merge(&step.step_stats)`. Unconditional (not gated on
    /// `profiling_enabled`): timing fields are already zero when profiling is
    /// off, and the always-on cache hit/miss counters must still accumulate.
    pub(crate) fn accumulate_step(&self, step_stats: &ExecutionStats) {
        self.step_stats
            .lock()
            .expect("ParallelProfiling step_stats poisoned")
            .merge(step_stats);
    }

    /// Fold the accumulated atomics into an `ExecutionStats`. Adds zero for any
    /// field that never fired (e.g. profiling disabled), so calling this
    /// unconditionally is byte-identical to the gated legacy increments.
    pub(crate) fn fold_into(&self, stats: &mut ExecutionStats) {
        stats.solver_fork_time_ns += self.solver_fork_time_ns.load(Ordering::Relaxed);
        stats.solver_fork_count += self.solver_fork_count.load(Ordering::Relaxed);
        stats.solver_sat_time_ns += self.solver_sat_time_ns.load(Ordering::Relaxed);
        stats.solver_sat_count += self.solver_sat_count.load(Ordering::Relaxed);
        stats.deferred_fork_time_ns += self.deferred_fork_time_ns.load(Ordering::Relaxed);
        stats.deferred_fork_count += self.deferred_fork_count.load(Ordering::Relaxed);
        // Fold every sum-typed interpreter counter (lift_time_ns, cache
        // hits/misses, blocks_executed, ...) exactly as the single-threaded
        // `merge` path does — see the `step_stats` field doc.
        stats.merge(
            &self
                .step_stats
                .lock()
                .expect("ParallelProfiling step_stats poisoned"),
        );
    }

    /// Like [`fold_into`](Self::fold_into) but SWAPS each accumulator to zero, so
    /// it is safe to call repeatedly against a long-lived accumulator — the
    /// steady-state coordinator folds deltas at every event return while
    /// workers keep adding (angr-nkoct). A wave calling this once is
    /// byte-identical to `fold_into` (the wave's accumulator dies right after).
    pub(crate) fn drain_into(&self, stats: &mut ExecutionStats) {
        stats.solver_fork_time_ns += self.solver_fork_time_ns.swap(0, Ordering::Relaxed);
        stats.solver_fork_count += self.solver_fork_count.swap(0, Ordering::Relaxed);
        stats.solver_sat_time_ns += self.solver_sat_time_ns.swap(0, Ordering::Relaxed);
        stats.solver_sat_count += self.solver_sat_count.swap(0, Ordering::Relaxed);
        stats.deferred_fork_time_ns += self.deferred_fork_time_ns.swap(0, Ordering::Relaxed);
        stats.deferred_fork_count += self.deferred_fork_count.swap(0, Ordering::Relaxed);
        let step_acc = std::mem::take(
            &mut *self
                .step_stats
                .lock()
                .expect("ParallelProfiling step_stats poisoned"),
        );
        stats.merge(&step_acc);
    }
}

// Compile-time proof the profiling accumulator is `Send + Sync`, so the
// persistent worker pool can `Arc`-share one across all workers for a wave
// (angr-vh834 Phase 6 / Work Item 2). All fields are `AtomicU64`, so this holds;
// the assertion fails the build (not a run) if a non-`Send`/`Sync` field lands.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ParallelProfiling>();
};

/// Manager-level (non-`accumulated_stats`) counter deltas the extracted arms
/// produce. Each step touches at most one proc / syscall, so the maps usually
/// hold a single entry. The coordinator folds these into the manager fields.
#[derive(Default)]
pub(crate) struct CoreCounters {
    pub(crate) native_calls: u64,
    pub(crate) native_python_fallbacks: u64,
    pub(crate) call_counts: HashMap<String, u64>,
    pub(crate) symbolic_fallbacks_by_name: HashMap<String, u64>,
    pub(crate) not_implemented_fallbacks_by_name: HashMap<String, u64>,
    pub(crate) other_fallbacks_by_name: HashMap<String, u64>,
    pub(crate) syscall_native_count: u64,
    pub(crate) syscall_native_by_num: HashMap<i64, u64>,
    pub(crate) syscall_python_fallback_count: u64,
    pub(crate) syscall_python_fallback_by_num: HashMap<i64, u64>,
    pub(crate) simprocedure_python_fallback_count: u64,
    pub(crate) simprocedure_fallback_by_name: HashMap<String, u64>,
    pub(crate) deferred_forks_dropped: u64,
}

/// Classification of a routed state (informational for the future parallel
/// coordinator; the single-threaded coordinator routes via the existing run
/// loop). `is_fork` and `root_hint` drive `sm.set_root`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // non-Continue variants are the routing-tag contract for the
// future parallel coordinator (see angr-i40qw); the single-threaded path tags
// continue states.
pub(crate) enum TagKind {
    Continue,
    Found,
    Deadended,
    Errored,
    Unconstrained,
    Pruned,
    NeedsPython,
}

#[derive(Clone, Copy)]
pub(crate) struct RoutingTag {
    #[allow(dead_code)] // consumed by the future parallel coordinator (angr-i40qw)
    pub(crate) kind: TagKind,
    pub(crate) root_hint: Option<u64>,
    pub(crate) is_fork: bool,
}

impl RoutingTag {
    #[inline]
    fn main() -> Self {
        RoutingTag {
            kind: TagKind::Continue,
            root_hint: None,
            is_fork: false,
        }
    }
    #[inline]
    fn fork(root_hint: u64) -> Self {
        RoutingTag {
            kind: TagKind::Continue,
            root_hint: Some(root_hint),
            is_fork: true,
        }
    }
}

/// Which Python-bouncing handler the coordinator must run (with `&mut self` and
/// the GIL). The core re-runs none of these; it hands the (possibly
/// native-dispatch-mutated) state and the deferred-fork data back so the legacy
/// handler builds the exact `PendingCallback`. Carrying the live state + step
/// data (not a serialized `PendingCallback`) keeps the single-threaded path from
/// re-running the interpreter, which would double-count profiling.
#[derive(Clone, Debug)]
pub(crate) enum BounceKind {
    /// Bare hook (zero-arg "unknown" SimProcedure callback).
    Hook { addr: u64 },
    /// Native SimProcedure declined / errored / is in-binary: Python SimProc.
    SimProcedurePython {
        addr: u64,
        name: String,
        num_args: usize,
        return_addr: u64,
    },
    /// Native syscall declined / errored / symbolic-num: Python syscall.
    SyscallPython { num: Option<u64> },
    /// Unhooked CALL needing Python `resolve_function` (mutates hooks/simprocs).
    UnmodeledCall {
        addr: u64,
        return_addr: u64,
        symbol_name: Option<String>,
    },
    /// Unsupported VEX op needing the Python VEX engine.
    PythonVEXFallback { addr: u64, reason: String },
}

/// The eager-mode symbolic branch [`handle_symbolic_branch_core`] resolves:
/// the guard's id in `stored_conditions` plus the two concrete targets.
pub(crate) struct SymBranch {
    pub(crate) condition_id: u64,
    pub(crate) true_target: u64,
    pub(crate) false_target: u64,
}

/// Everything the coordinator's bounce handler needs to reproduce the legacy
/// arm. `state` is owned (handed back); the deferred-fork data rides along.
pub(crate) struct PendingBounce {
    pub(crate) kind: BounceKind,
    pub(crate) state: RustSimState,
    pub(crate) deferred_forks: Vec<DeferredFork>,
    pub(crate) stored_conditions: FxHashMap<u64, RustBV>,
    pub(crate) fork_snapshots: FxHashMap<u64, BranchSnapshot>,
}

/// The terminal/continue routing decision, translated by the coordinator into
/// the `Result<Vec<RustSimState>, StepError>` `step_state_with_skip` returns.
pub(crate) enum CoreReturn {
    /// Live successors (incl. the main state when it continues). Maps to `Ok`.
    Continue(Vec<(RustSimState, RoutingTag)>),
    /// Maps to `Err(StepError::Deadended)`.
    Deadended(RustSimState),
    /// Maps to `Err(StepError::Error)`.
    Errored(RustSimState, String),
    /// Maps to `Err(StepError::Unconstrained)` (main + eager loop-exit forks).
    Unconstrained(RustSimState, Vec<RustSimState>),
    /// Maps to a coordinator bounce (`Err(StepError::NeedCallback)` etc.).
    NeedsPython(PendingBounce),
}

/// The full deferred-mutation payload the coordinator applies after the core
/// returns. See the module docs for the contract.
pub(crate) struct CoreOutcome {
    pub(crate) ret: CoreReturn,
    /// UNSAT forks (coordinator: `set_root` each, then push to STASH_PRUNED).
    pub(crate) pruned: Vec<RustSimState>,
    /// Newly minted fork ids the legacy path fires `dispatch_fork_inspect` on,
    /// in firing order.
    pub(crate) fork_ids: Vec<u64>,
    /// Side-effect terminal pushes (no-return main state) the legacy code
    /// performs while still returning surviving forks.
    pub(crate) terminal_pushes: Vec<(RustSimState, &'static str)>,
    pub(crate) counters: CoreCounters,
    /// Stamped parent root; copied onto every fork tag.
    pub(crate) root_hint: u64,
}

// Compile-time proof that the captured-context references the core fn takes are
// `Send + Sync` — the property the future work-stealing worker path depends on.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ParallelProfiling>();
    assert_send_sync::<&NativeProcedureRegistry>();
    assert_send_sync::<&NativeSyscallRegistry>();
    assert_send_sync::<CcSnapshot>();
};

/// Shared immutable context bundle threaded through every post-step arm.
///
/// Bundling the four references the arms all read retires the
/// `too_many_arguments` allows they used to carry (angr-0mqkc.6). Passed by
/// shared reference, so this is zero-cost (four pointers behind one pointer).
pub(crate) struct CoreCtx<'a> {
    pub(crate) ctx: &'a StepContext,
    pub(crate) prof: &'a ParallelProfiling,
    pub(crate) native_procs: &'a NativeProcedureRegistry,
    pub(crate) native_syscalls: &'a NativeSyscallRegistry,
    /// Python callbacks, for the inspect BPs the post-step core fires itself
    /// (currently `constraints`, from the fork-guard add — angr-op0dn.14.4.1).
    /// `None` in the Rust-only unit tests, which have no Python side.
    pub(crate) callbacks: Option<&'a PythonCallbacks>,
}

/// The deferred-fork payload the post-step arms thread through and, on a Python
/// bounce, hand to the coordinator. Owns all four pieces; the arm consumes it
/// exactly once (either materializing the forks or moving them into a bounce).
struct ForkPayload {
    deferred_forks: Vec<DeferredFork>,
    last_condition: Option<RustBV>,
    stored_conditions: FxHashMap<u64, RustBV>,
    fork_snapshots: FxHashMap<u64, BranchSnapshot>,
}

/// The three fork output vecs the materialization helpers append to.
struct ForkSink<'a> {
    forks: &'a mut Vec<(RustSimState, RoutingTag)>,
    pruned: &'a mut Vec<RustSimState>,
    fork_ids: &'a mut Vec<u64>,
}

/// Arguments for entering a native guest routine via a sub-call continuation
/// (mirror of `NativeProcDisposition::SubCall` plus the caller return address).
pub(crate) struct NativeSubcall {
    pub(crate) proc_name: String,
    pub(crate) saved_args: Vec<RustBV>,
    pub(crate) caller_return_addr: u64,
    pub(crate) target: u64,
    pub(crate) sub_args: Vec<RustBV>,
    pub(crate) resume_tag: u32,
}

/// Post-interpreter inputs the arms consume (the already-drained pieces of the
/// `InterpreterStepResult`). The state-update preamble in `step_state_with_skip`
/// runs before this, so only these fields remain.
pub(crate) struct PostStepInputs {
    pub(crate) result: RunResult,
    pub(crate) deferred_forks: Vec<DeferredFork>,
    pub(crate) last_condition: Option<RustBV>,
    pub(crate) stored_conditions: FxHashMap<u64, RustBV>,
    pub(crate) fork_snapshots: FxHashMap<u64, BranchSnapshot>,
}

/// `&mut self`-free post-interpreter classification + fork materialization.
///
/// Takes the stepped `state` by value (it becomes a successor / terminal /
/// bounce state) and returns a [`CoreOutcome`] of deferred mutations.
pub(crate) fn run_post_step_core(
    cc: &CoreCtx,
    mut state: RustSimState,
    inputs: PostStepInputs,
    root_hint: u64,
) -> CoreOutcome {
    let PostStepInputs {
        result,
        deferred_forks,
        last_condition,
        stored_conditions,
        fork_snapshots,
    } = inputs;
    let payload = ForkPayload {
        deferred_forks,
        last_condition,
        stored_conditions,
        fork_snapshots,
    };

    let mut counters = CoreCounters::default();

    match result {
        RunResult::MaxBlocks { pc }
        | RunResult::MaxDeferredForks { pc }
        | RunResult::BlockEnd { next_addr: pc, .. } => {
            state.set_pc(pc);
            let mut forks_out = Vec::new();
            let mut pruned = Vec::new();
            let mut fork_ids = Vec::new();
            materialize_deferred_forks_core(
                cc,
                &state,
                payload,
                root_hint,
                false,
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

        RunResult::Hook { addr } => bounce(
            BounceKind::Hook { addr },
            state,
            payload,
            counters,
            root_hint,
        ),

        RunResult::SimProcedure {
            addr,
            name,
            num_args,
            return_addr,
        } => handle_simprocedure_core(
            cc,
            &mut counters,
            state,
            SimProcCall {
                addr,
                name,
                num_args,
                return_addr,
            },
            payload,
            root_hint,
        ),

        RunResult::Syscall { num, pc } => {
            handle_syscall_core(cc, &mut counters, state, num, pc, payload, root_hint)
        }

        RunResult::SymbolicBranch {
            condition_id,
            true_target,
            false_target,
        } => handle_symbolic_branch_core(
            cc,
            state,
            SymBranch {
                condition_id,
                true_target,
                false_target,
            },
            payload,
            counters,
            root_hint,
        ),

        RunResult::Error {
            message,
            addr,
            kind,
        } => {
            state.set_pc(addr);
            let ret = match kind {
                RunErrorKind::Deadend => CoreReturn::Deadended(state),
                RunErrorKind::Fatal if addr == 0 => CoreReturn::Deadended(state),
                RunErrorKind::Fatal => CoreReturn::Errored(state, message),
            };
            CoreOutcome {
                ret,
                pruned: Vec::new(),
                fork_ids: Vec::new(),
                terminal_pushes: Vec::new(),
                counters,
                root_hint,
            }
        }

        RunResult::NeedPythonVEX { addr, reason } => {
            log::debug!("VEX fallback at 0x{addr:x}: {reason}");
            state.set_pc(addr);
            bounce(
                BounceKind::PythonVEXFallback { addr, reason },
                state,
                payload,
                counters,
                root_hint,
            )
        }

        RunResult::NeedLift { addr } => {
            state.set_pc(addr);
            CoreOutcome {
                ret: CoreReturn::Errored(state, format!("need lift at 0x{addr:x}")),
                pruned: Vec::new(),
                fork_ids: Vec::new(),
                terminal_pushes: Vec::new(),
                counters,
                root_hint,
            }
        }

        RunResult::SymbolicJumpTarget {
            targets,
            condition_id,
            jumpkind: _,
        } => handle_symbolic_jump_target_core(
            cc,
            state,
            targets,
            condition_id,
            payload,
            counters,
            root_hint,
        ),

        RunResult::UnconstrainedJump { .. } => {
            // Main state -> unconstrained stash; accumulated loop-exit forks are
            // either dropped (deferred mode) or materialized eagerly (angr-027h).
            let forks = if cc.ctx.use_deferred_forks && !cc.ctx.materialize_unconstrained_forks {
                counters.deferred_forks_dropped += payload.deferred_forks.len() as u64;
                Vec::new()
            } else {
                let mut forks_out = Vec::new();
                let mut pruned_tmp = Vec::new();
                let mut fork_ids_tmp = Vec::new();
                materialize_deferred_forks_core(
                    cc,
                    &state,
                    payload,
                    root_hint,
                    true,
                    ForkSink {
                        forks: &mut forks_out,
                        pruned: &mut pruned_tmp,
                        fork_ids: &mut fork_ids_tmp,
                    },
                );
                // The eager forks become Unconstrained's routed forks; their
                // set_root happened via tags, dispatch via fork_ids, pruned via
                // pruned_tmp.
                return CoreOutcome {
                    ret: CoreReturn::Unconstrained(
                        state,
                        forks_out.into_iter().map(|(s, _)| s).collect(),
                    ),
                    pruned: pruned_tmp,
                    fork_ids: fork_ids_tmp,
                    terminal_pushes: Vec::new(),
                    counters,
                    root_hint,
                };
            };
            CoreOutcome {
                ret: CoreReturn::Unconstrained(state, forks),
                pruned: Vec::new(),
                fork_ids: Vec::new(),
                terminal_pushes: Vec::new(),
                counters,
                root_hint,
            }
        }

        RunResult::UnmodeledCall {
            addr,
            return_addr,
            symbol_name,
        } => bounce(
            BounceKind::UnmodeledCall {
                addr,
                return_addr,
                symbol_name,
            },
            state,
            payload,
            counters,
            root_hint,
        ),
    }
}

#[cfg(test)]
#[path = "core_outcome_tests.rs"]
mod tests;
