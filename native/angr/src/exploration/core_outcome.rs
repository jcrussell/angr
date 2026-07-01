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
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rustc_hash::FxHashMap;

use crate::callbacks::{DeferredFork, RunErrorKind, RunResult};
use crate::interpreter::{BranchSnapshot, ExecutionStats};
use crate::procedures::{
    NATIVE_RESUME_SENTINEL_NAME, NativeProcedureRegistry, ProcOutcome, ProcedureError,
    native_resume_sentinel,
};
use crate::stash::STASH_DEADENDED;
use crate::state::{NativeResumeFrame, RustSimState};
use crate::symbolic::RustBV;
use crate::syscalls::{NativeSyscallRegistry, SyscallOutcome};

use super::step_core::StepContext;
use super::stepping::SubcallSetupError;

/// Scalar / vec snapshot of the manager's calling convention.
///
/// The `Box<dyn CallingConvention>` trait object is not `Clone`, so the post-step
/// native arms snapshot exactly the scalar and vec data they read (mirrors the
/// CC reads at the legacy `stepping.rs` sites). Built once per step by
/// `RustExplorationManager::step_context`.
#[derive(Clone)]
pub(crate) struct CcSnapshot {
    pub(crate) arg_registers: Vec<u32>,
    pub(crate) syscall_arg_registers: Vec<u32>,
    pub(crate) return_register: u32,
    pub(crate) link_register: Option<u32>,
    pub(crate) pops_return_addr: bool,
    pub(crate) pointer_size: u32,
    pub(crate) stack_arg_offset: u64,
    pub(crate) syscall_stack_arg_offset: Option<u64>,
    pub(crate) syscall_error_register: Option<(u32, i64)>,
}

impl CcSnapshot {
    /// Mirror of `RustExplorationManager::write_syscall_return`.
    fn write_syscall_return(&self, state: &mut RustSimState, ret_reg: u32, ret: RustBV) {
        let Some((err_reg, errno_start)) = self.syscall_error_register else {
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

    /// Mirror of `RustExplorationManager::extract_procedure_args`.
    fn extract_procedure_args(
        &self,
        state: &RustSimState,
        num_args: usize,
    ) -> Result<Vec<RustBV>, crate::arch::ExtractionError> {
        use crate::arch::ExtractionError;
        let ptr_size = self.pointer_size;
        let mut args = Vec::with_capacity(num_args);

        let ctx = state.solver().borrow();

        for &offset in self.arg_registers.iter().take(num_args) {
            args.push(state.get_register_by_offset(offset, ptr_size));
        }

        if args.len() < num_args {
            let sp = state.get_sp().as_u64().ok_or(ExtractionError::SpSymbolic)?;
            let stack_start = sp + self.stack_arg_offset;
            let already = args.len();
            for i in 0..(num_args - already) {
                let addr = stack_start + (i as u64 * ptr_size as u64);
                let value = state.memory_load(addr, ptr_size).map_err(|_| {
                    ExtractionError::StackUnmapped {
                        arg_index: already + i,
                        addr,
                    }
                })?;
                args.push(value);
            }
        }

        drop(ctx);
        Ok(args)
    }

    /// Mirror of `RustExplorationManager::extract_syscall_args`.
    fn extract_syscall_args(
        &self,
        state: &RustSimState,
        num_args: usize,
    ) -> Result<Vec<RustBV>, crate::arch::ExtractionError> {
        use crate::arch::ExtractionError;
        let ptr_size = self.pointer_size;
        let mut args = Vec::with_capacity(num_args);
        for &offset in self.syscall_arg_registers.iter().take(num_args) {
            args.push(state.get_register_by_offset(offset, ptr_size));
        }

        if args.len() < num_args {
            let stack_offset =
                self.syscall_stack_arg_offset
                    .ok_or(ExtractionError::RegisterOverflow {
                        requested: num_args,
                        available: self.syscall_arg_registers.len(),
                    })?;
            let sp = state.get_sp().as_u64().ok_or(ExtractionError::SpSymbolic)?;
            let stack_start = sp + stack_offset;
            let already = args.len();
            for i in 0..(num_args - already) {
                let addr = stack_start + (i as u64 * ptr_size as u64);
                let value = state.memory_load(addr, ptr_size).map_err(|_| {
                    ExtractionError::StackUnmapped {
                        arg_index: already + i,
                        addr,
                    }
                })?;
                args.push(value);
            }
        }

        Ok(args)
    }

    /// Mirror of `RustExplorationManager::setup_native_subcall` (no `&self`).
    #[allow(clippy::too_many_arguments)] // mirrors the manager method's signature 1:1
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
        let arg_regs = &self.arg_registers;
        if sub_args.len() > arg_regs.len() {
            return Err(SubcallSetupError::TooManyArgs {
                requested: sub_args.len(),
                available: arg_regs.len(),
            });
        }
        let ptr_bits = self.pointer_size * 8;
        let sentinel = native_resume_sentinel(self.pointer_size);

        // --- feasibility checks (no mutation yet) ---
        let lr_offset = if self.pops_return_addr {
            None
        } else {
            Some(
                self.link_register
                    .ok_or(SubcallSetupError::UnsupportedAbi)?,
            )
        };
        let sp_val = if self.pops_return_addr {
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
            state
                .memory_mut()
                .store_concrete(sp, RustBV::concrete(sentinel as u128, ptr_bits))
                .map_err(SubcallSetupError::Memory)?;
        } else if let Some(lr) = lr_offset {
            state.set_register_by_offset(lr, RustBV::concrete(sentinel as u128, ptr_bits));
        }

        // --- record the continuation and enter the guest routine ---
        state.push_native_resume_frame(NativeResumeFrame {
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
}

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
    /// IRSB block-cache hits/misses accumulated across the wave's dispatches
    /// (angr-vh834 Work Item 3). The parallel worker folds each step's
    /// `ExecutionStats::cache_hit_count`/`cache_miss_count` in here so the warm
    /// per-worker cache win surfaces via `stats.cache_hit_count` /
    /// `cache_miss_count` (the `block_cache_hits` / `block_cache_misses` keys in
    /// `mgr.stats()`), consistent with the single-threaded `merge` path.
    pub(crate) cache_hit_count: AtomicU64,
    pub(crate) cache_miss_count: AtomicU64,
}

impl ParallelProfiling {
    #[inline]
    fn add(a: &AtomicU64, v: u64) {
        a.fetch_add(v, Ordering::Relaxed);
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
        // ADD (never clobber) into the same accumulated_stats fields the
        // single-threaded `merge` path populates, so both engines report the
        // block-cache hit/miss counters consistently.
        stats.cache_hit_count += self.cache_hit_count.load(Ordering::Relaxed);
        stats.cache_miss_count += self.cache_miss_count.load(Ordering::Relaxed);
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
// future parallel coordinator; the single-threaded path tags continue states.
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
    #[allow(dead_code)]
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
#[derive(Clone)]
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
    /// Deferred symbolic branch needing Python state forking.
    SymbolicBranch {
        condition_id: u64,
        true_target: u64,
        false_target: u64,
    },
    /// Unhooked CALL needing Python `resolve_function` (mutates hooks/simprocs).
    UnmodeledCall {
        addr: u64,
        return_addr: u64,
        symbol_name: Option<String>,
    },
    /// Unsupported VEX op needing the Python VEX engine.
    PythonVEXFallback { addr: u64, reason: String },
}

/// Everything the coordinator's bounce handler needs to reproduce the legacy
/// arm. `state` is owned (handed back); the deferred-fork data rides along.
pub(crate) struct PendingBounce {
    pub(crate) kind: BounceKind,
    pub(crate) state: RustSimState,
    pub(crate) deferred_forks: Vec<DeferredFork>,
    pub(crate) last_condition: Option<RustBV>,
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_post_step_core(
    ctx: &StepContext,
    prof: &ParallelProfiling,
    native_procs: &NativeProcedureRegistry,
    native_syscalls: &NativeSyscallRegistry,
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
                ctx,
                prof,
                &state,
                deferred_forks,
                &stored_conditions,
                fork_snapshots,
                root_hint,
                false,
                &mut forks_out,
                &mut pruned,
                &mut fork_ids,
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
            deferred_forks,
            last_condition,
            stored_conditions,
            fork_snapshots,
            counters,
            root_hint,
        ),

        RunResult::SimProcedure {
            addr,
            name,
            num_args,
            return_addr,
        } => handle_simprocedure_core(
            ctx,
            prof,
            native_procs,
            &mut counters,
            state,
            addr,
            name,
            num_args,
            return_addr,
            deferred_forks,
            last_condition,
            stored_conditions,
            fork_snapshots,
            root_hint,
        ),

        RunResult::Syscall { num, pc } => handle_syscall_core(
            ctx,
            prof,
            native_syscalls,
            &mut counters,
            state,
            num,
            pc,
            deferred_forks,
            last_condition,
            stored_conditions,
            fork_snapshots,
            root_hint,
        ),

        RunResult::SymbolicBranch {
            condition_id,
            true_target,
            false_target,
        } => {
            // last_condition is folded into stored_conditions by the bounce
            // handler (it needs branch_conditions). Carry it along.
            bounce(
                BounceKind::SymbolicBranch {
                    condition_id,
                    true_target,
                    false_target,
                },
                state,
                deferred_forks,
                last_condition,
                stored_conditions,
                fork_snapshots,
                counters,
                root_hint,
            )
        }

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
            log::debug!("VEX fallback at 0x{:x}: {}", addr, reason);
            state.set_pc(addr);
            bounce(
                BounceKind::PythonVEXFallback { addr, reason },
                state,
                deferred_forks,
                last_condition,
                stored_conditions,
                fork_snapshots,
                counters,
                root_hint,
            )
        }

        RunResult::NeedLift { addr } => {
            state.set_pc(addr);
            CoreOutcome {
                ret: CoreReturn::Errored(state, format!("need lift at 0x{:x}", addr)),
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
            ctx,
            prof,
            state,
            targets,
            condition_id,
            deferred_forks,
            &stored_conditions,
            fork_snapshots,
            counters,
            root_hint,
        ),

        RunResult::UnconstrainedJump { .. } => {
            // Main state -> unconstrained stash; accumulated loop-exit forks are
            // either dropped (deferred mode) or materialized eagerly (angr-027h).
            let forks = if ctx.use_deferred_forks && !ctx.materialize_unconstrained_forks {
                counters.deferred_forks_dropped += deferred_forks.len() as u64;
                Vec::new()
            } else {
                let mut forks_out = Vec::new();
                let mut pruned_tmp = Vec::new();
                let mut fork_ids_tmp = Vec::new();
                materialize_deferred_forks_core(
                    ctx,
                    prof,
                    &state,
                    deferred_forks,
                    &stored_conditions,
                    fork_snapshots,
                    root_hint,
                    true,
                    &mut forks_out,
                    &mut pruned_tmp,
                    &mut fork_ids_tmp,
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
            deferred_forks,
            last_condition,
            stored_conditions,
            fork_snapshots,
            counters,
            root_hint,
        ),
    }
}

/// Package a Python-bouncing outcome (no fork materialization here — the
/// deferred forks ride into the `PendingCallback` the coordinator builds).
#[allow(clippy::too_many_arguments)]
fn bounce(
    kind: BounceKind,
    state: RustSimState,
    deferred_forks: Vec<DeferredFork>,
    last_condition: Option<RustBV>,
    stored_conditions: FxHashMap<u64, RustBV>,
    fork_snapshots: FxHashMap<u64, BranchSnapshot>,
    counters: CoreCounters,
    root_hint: u64,
) -> CoreOutcome {
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
#[allow(clippy::too_many_arguments)]
fn materialize_deferred_forks_core(
    ctx: &StepContext,
    prof: &ParallelProfiling,
    base: &RustSimState,
    deferred_forks: Vec<DeferredFork>,
    stored_conditions: &FxHashMap<u64, RustBV>,
    mut fork_snapshots: FxHashMap<u64, BranchSnapshot>,
    root_hint: u64,
    force_eager: bool,
    forks_out: &mut Vec<(RustSimState, RoutingTag)>,
    pruned_out: &mut Vec<RustSimState>,
    fork_ids_out: &mut Vec<u64>,
) {
    let deferred_fork_start = if ctx.profiling_enabled {
        Some(Instant::now())
    } else {
        None
    };
    let deferred_fork_total = deferred_forks.len() as u64;

    for fork in deferred_forks {
        if let Some(condition) = stored_conditions.get(&fork.condition_id) {
            if fork.path_taken {
                base.solver().borrow().assume_true(condition);
            } else {
                base.solver().borrow().assume_false(condition);
            }

            let fork_start = if ctx.profiling_enabled {
                Some(Instant::now())
            } else {
                None
            };
            let mut forked =
                super::helpers::build_unexplored_fork(base, &fork, condition, &mut fork_snapshots);
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
#[allow(clippy::too_many_arguments)]
fn process_deferred_forks_into_core(
    ctx: &StepContext,
    prof: &ParallelProfiling,
    base: &RustSimState,
    deferred_forks: Vec<DeferredFork>,
    stored_conditions: &FxHashMap<u64, RustBV>,
    mut fork_snapshots: FxHashMap<u64, BranchSnapshot>,
    root_hint: u64,
    forks_out: &mut Vec<(RustSimState, RoutingTag)>,
    pruned_out: &mut Vec<RustSimState>,
    fork_ids_out: &mut Vec<u64>,
) {
    if deferred_forks.is_empty() {
        return;
    }

    for fork in &deferred_forks {
        if let Some(condition) = stored_conditions.get(&fork.condition_id) {
            if fork.path_taken {
                base.solver().borrow().assume_true(condition);
            } else {
                base.solver().borrow().assume_false(condition);
            }

            let forked =
                super::helpers::build_unexplored_fork(base, fork, condition, &mut fork_snapshots);
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
    ctx: &StepContext,
    prof: &ParallelProfiling,
    base: &RustSimState,
    deferred_forks: Vec<DeferredFork>,
    stored_conditions: &FxHashMap<u64, RustBV>,
    fork_snapshots: FxHashMap<u64, BranchSnapshot>,
    root_hint: u64,
) -> (Vec<RustSimState>, Vec<RustSimState>, Vec<u64>) {
    let mut forks_out: Vec<(RustSimState, RoutingTag)> = Vec::new();
    let mut pruned_out: Vec<RustSimState> = Vec::new();
    let mut fork_ids_out: Vec<u64> = Vec::new();
    process_deferred_forks_into_core(
        ctx,
        prof,
        base,
        deferred_forks,
        stored_conditions,
        fork_snapshots,
        root_hint,
        &mut forks_out,
        &mut pruned_out,
        &mut fork_ids_out,
    );
    (
        forks_out.into_iter().map(|(s, _)| s).collect(),
        pruned_out,
        fork_ids_out,
    )
}

/// Mirror of `handle_symbolic_jump_target`.
#[allow(clippy::too_many_arguments)]
fn handle_symbolic_jump_target_core(
    ctx: &StepContext,
    prof: &ParallelProfiling,
    state: RustSimState,
    targets: Vec<u64>,
    condition_id: u64,
    deferred_forks: Vec<DeferredFork>,
    stored_conditions: &FxHashMap<u64, RustBV>,
    fork_snapshots: FxHashMap<u64, BranchSnapshot>,
    counters: CoreCounters,
    root_hint: u64,
) -> CoreOutcome {
    let target_expr = stored_conditions.get(&condition_id).cloned();

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
            ctx,
            prof,
            &first,
            deferred_forks,
            stored_conditions,
            fork_snapshots,
            root_hint,
            &mut forks_out,
            &mut pruned,
            &mut fork_ids,
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
        ctx,
        prof,
        &first_state,
        deferred_forks,
        stored_conditions,
        fork_snapshots,
        root_hint,
        &mut deferred_out,
        &mut pruned,
        &mut fork_ids,
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
}

/// Mirror of `handle_simprocedure` (native fast path + native resume; Python
/// fallback bounces).
#[allow(clippy::too_many_arguments)]
fn handle_simprocedure_core(
    ctx: &StepContext,
    prof: &ParallelProfiling,
    native_procs: &NativeProcedureRegistry,
    counters: &mut CoreCounters,
    mut state: RustSimState,
    addr: u64,
    name: String,
    num_args: usize,
    return_addr: u64,
    deferred_forks: Vec<DeferredFork>,
    last_condition: Option<RustBV>,
    stored_conditions: FxHashMap<u64, RustBV>,
    fork_snapshots: FxHashMap<u64, BranchSnapshot>,
    root_hint: u64,
) -> CoreOutcome {
    // Native sub-call resume sentinel: a guest routine returns here.
    if name == NATIVE_RESUME_SENTINEL_NAME {
        return handle_native_resume_core(
            ctx,
            prof,
            native_procs,
            state,
            deferred_forks,
            &stored_conditions,
            fork_snapshots,
            std::mem::take(counters),
            root_hint,
        );
    }

    let is_in_binary = ctx
        .binary_regions
        .iter()
        .any(|(base, data)| addr >= *base && addr < *base + data.len() as u64);

    let disposition: NativeProcDisposition = if !is_in_binary {
        if let Some(native_proc) = native_procs.get(&name) {
            let proc_no_return = native_proc.no_return();
            match ctx.cc.extract_procedure_args(&state, num_args) {
                Err(e) => {
                    log::debug!(
                        "Skipping native procedure {} (arg extraction failed: {:?})",
                        name,
                        e
                    );
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
                        log::debug!(
                            "Native procedure {} returned error, falling back to Python: {:?}",
                            name,
                            e
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
        },
        NativeProcDisposition::Fallback => None,
    };

    if let Some(no_return) = fall_back_to_python {
        let mut forks_out = Vec::new();
        let mut pruned = Vec::new();
        let mut fork_ids = Vec::new();
        process_deferred_forks_into_core(
            ctx,
            prof,
            &state,
            deferred_forks,
            &stored_conditions,
            fork_snapshots,
            root_hint,
            &mut forks_out,
            &mut pruned,
            &mut fork_ids,
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
            deferred_forks,
            last_condition,
            stored_conditions,
            fork_snapshots,
            std::mem::take(counters),
            root_hint,
        )
    }
}

/// Mirror of `handle_native_resume`.
#[allow(clippy::too_many_arguments)]
fn handle_native_resume_core(
    ctx: &StepContext,
    prof: &ParallelProfiling,
    native_procs: &NativeProcedureRegistry,
    mut state: RustSimState,
    deferred_forks: Vec<DeferredFork>,
    stored_conditions: &FxHashMap<u64, RustBV>,
    fork_snapshots: FxHashMap<u64, BranchSnapshot>,
    counters: CoreCounters,
    root_hint: u64,
) -> CoreOutcome {
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
                frame.proc_name.clone(),
                frame.saved_args.clone(),
                frame.caller_return_addr,
                target,
                sub_args,
                resume_tag,
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
        ctx,
        prof,
        &state,
        deferred_forks,
        stored_conditions,
        fork_snapshots,
        root_hint,
        &mut forks_out,
        &mut pruned,
        &mut fork_ids,
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
#[allow(clippy::too_many_arguments)]
fn handle_syscall_core(
    ctx: &StepContext,
    prof: &ParallelProfiling,
    native_syscalls: &NativeSyscallRegistry,
    counters: &mut CoreCounters,
    mut state: RustSimState,
    num: Option<u64>,
    pc: u64,
    deferred_forks: Vec<DeferredFork>,
    last_condition: Option<RustBV>,
    stored_conditions: FxHashMap<u64, RustBV>,
    fork_snapshots: FxHashMap<u64, BranchSnapshot>,
    root_hint: u64,
) -> CoreOutcome {
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
                deferred_forks,
                last_condition,
                stored_conditions,
                fork_snapshots,
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
                    ctx,
                    prof,
                    state,
                    deferred_forks,
                    &stored_conditions,
                    fork_snapshots,
                    std::mem::take(counters),
                    root_hint,
                    false,
                );
            }
            Ok(SyscallOutcome::ContinueSymbolic { ret }) => {
                let ret_reg = ctx.cc.return_register;
                ctx.cc.write_syscall_return(&mut state, ret_reg, ret);
                return syscall_continue(
                    ctx,
                    prof,
                    state,
                    deferred_forks,
                    &stored_conditions,
                    fork_snapshots,
                    std::mem::take(counters),
                    root_hint,
                    false,
                );
            }
            Ok(SyscallOutcome::Exit) => {
                return syscall_continue(
                    ctx,
                    prof,
                    state,
                    deferred_forks,
                    &stored_conditions,
                    fork_snapshots,
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
        deferred_forks,
        last_condition,
        stored_conditions,
        fork_snapshots,
        std::mem::take(counters),
        root_hint,
    )
}

/// Shared continue/exit tail for the three native syscall outcomes.
#[allow(clippy::too_many_arguments)]
fn syscall_continue(
    ctx: &StepContext,
    prof: &ParallelProfiling,
    state: RustSimState,
    deferred_forks: Vec<DeferredFork>,
    stored_conditions: &FxHashMap<u64, RustBV>,
    fork_snapshots: FxHashMap<u64, BranchSnapshot>,
    counters: CoreCounters,
    root_hint: u64,
    exit: bool,
) -> CoreOutcome {
    let mut forks_out = Vec::new();
    let mut pruned = Vec::new();
    let mut fork_ids = Vec::new();
    process_deferred_forks_into_core(
        ctx,
        prof,
        &state,
        deferred_forks,
        stored_conditions,
        fork_snapshots,
        root_hint,
        &mut forks_out,
        &mut pruned,
        &mut fork_ids,
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

#[cfg(test)]
#[path = "core_outcome_tests.rs"]
mod tests;
