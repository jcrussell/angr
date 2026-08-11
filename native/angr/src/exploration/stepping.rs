//! Single-threaded step driver: everything between "the run loop picked a
//! state" and "here are its successors".
//!
//! [`RustExplorationManager::step_state_inner`] is the entry point. It builds a
//! [`StepContext`](super::step_core::StepContext) from `&self`, runs
//! [`run_interpreter_step_core`](super::step_core::run_interpreter_step_core),
//! and hands the resulting [`CoreOutcome`] to `apply_core_outcome`, which
//! either returns successors, terminates the state via [`StepError`], or
//! forwards a `NeedsPython` outcome to `dispatch_bounce`.
//!
//! `dispatch_bounce` is the Python-bouncing arm: it turns a [`PendingBounce`]
//! into the exact [`PendingCallback`] the caller must service (SimProcedure,
//! hook, syscall, symbolic branch, error), with `handle_unmodeled_call` /
//! `unmodeled_call_generic_skip` covering the unresolved-call path. The
//! matching re-entry points live in `resume.rs`.
//!
//! Also here: the native sub-call ABI setup shared with the parallel path
//! ([`SubcallAbi`] / [`setup_native_subcall_with_abi`]), deferred-fork
//! materialization into a successor list (`process_deferred_forks_into`), core
//! counter folding (`fold_core_counters`), and the out-of-band
//! `_step_state` pymethod body.
//!
//! The parallel drivers do not call into this module: they run the same core
//! through `run_loop_worker` and re-implement the `&mut self` tails against
//! their own snapshots. Anything both arms must agree on belongs in
//! `core_outcome.rs` or `step_core.rs`, not here.
//!
//! **Test-file naming (angr-c7xno.34):** there is deliberately no
//! `stepping_tests.rs`. Every entry point here (`step_state_inner`,
//! `dispatch_bounce`, `handle_unmodeled_call`) needs a lifted `IRSB` and a
//! live Python callback to reach, so a Rust unit test would be a mock of the
//! thing under test; that surface is covered end-to-end by the Python suite in
//! `tests/engines/rust/`. The two `#[path]` submodules cover the slices that
//! *are* reachable without a lift and are named for what they test rather than
//! for this file: `sizes_tests.rs` (the `StepError` / `PendingCallback` size
//! probe backing the `result_large_err` suppression) and `subcall_tests.rs`
//! (`setup_native_subcall` / `handle_native_resume_core`, which stand in for
//! the guest `ret` instead of lifting one). The lift-free routing prefix of
//! the *caller* — `step_one`'s find/avoid and SimProcedure-fallback arms — is
//! covered in `run_loop_single_tests.rs`.

use super::core_outcome::{
    BounceKind, CoreCounters, CoreCtx, CoreOutcome, CoreReturn, NativeSubcall, ParallelProfiling,
    PendingBounce, PostStepInputs, run_post_step_core,
};
use super::*;
use crate::arch::RegisterFile;
use crate::interpreter::BranchSnapshot;
use crate::memory::SymbolicMemory;
use crate::stash::STASH_STEP_OUT;
use crate::state::{CallStackEntry, HistoryEntry};
use crate::vex::IRSB;
use crate::vex::ir::JumpKind;
use lru::LruCache;
use pyo3::exceptions::PyNotImplementedError;

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

/// The ABI facts [`setup_native_subcall_with_abi`] needs, decoupled from where
/// they came from.
///
/// The single-threaded path reads them off the manager's
/// `Box<dyn CallingConvention>`; the parallel post-step path reads them off the
/// scalar `CcSnapshot` (the trait object is not `Clone`). Both funnel through
/// this borrow so the dispatch logic exists once (angr-sqfj8.41).
pub(crate) struct SubcallAbi<'a> {
    pub(crate) arg_registers: &'a [u32],
    pub(crate) pointer_size: u32,
    pub(crate) pops_return_addr: bool,
    pub(crate) link_register: Option<u32>,
}

/// Set up a native sub-call (`ProcOutcome::CallAndResume`): make the guest
/// routine `sub.target` run with `sub.sub_args`, then return to the resume
/// sentinel so the proc's continuation re-enters via `handle_native_resume`.
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
/// sentinel into the link register (requires `SubcallAbi::link_register`).
pub(crate) fn setup_native_subcall_with_abi(
    abi: &SubcallAbi<'_>,
    state: &mut RustSimState,
    sub: NativeSubcall,
) -> Result<(), SubcallSetupError> {
    let NativeSubcall {
        proc_name,
        saved_args,
        caller_return_addr,
        target,
        sub_args,
        resume_tag,
    } = sub;
    let arg_regs = abi.arg_registers;
    if sub_args.len() > arg_regs.len() {
        return Err(SubcallSetupError::TooManyArgs {
            requested: sub_args.len(),
            available: arg_regs.len(),
        });
    }
    let ptr_bits = abi.pointer_size * 8;
    let sentinel = crate::procedures::native_resume_sentinel(abi.pointer_size);

    // --- feasibility checks (no mutation yet) ---
    let lr_offset = if abi.pops_return_addr {
        None
    } else {
        Some(abi.link_register.ok_or(SubcallSetupError::UnsupportedAbi)?)
    };
    let sp_val = if abi.pops_return_addr {
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
    /// Simulated TSC after the step; written back to `RustSimState` so the
    /// counter is per-state rather than process-wide (angr-9ke6b.173).
    pub(crate) new_tsc_counter: u64,
    pub(crate) recovered_memory: Option<SymbolicMemory>,
    pub(crate) step_stats: ExecutionStats,
    pub(crate) updated_block_cache: LruCache<u64, Arc<IRSB>>,
    /// Set only when `state.keep_ip_symbolic()` was true and the interpreter
    /// concretized a symbolic default-exit next-pc. The manager writes this
    /// back to the state's IP register after `state.set_pc(new_pc)`.
    pub(crate) symbolic_ip_at_exit: Option<RustBV>,
}

/// Write the interpreter's post-step results back onto `state`.
///
/// This is the byte-identical 7-statement state-restore sequence that both the
/// single-threaded (`step_state_inner`) and parallel (`parallel_process_state`,
/// run_loop.rs) step paths apply after `run_interpreter_step_core` returns.
/// Extracted so a future new `InterpreterStepResult` field or a reordering fix
/// can't land in one copy and silently miss the other (angr-04tw3.8). Takes the
/// fields by value rather than the whole `InterpreterStepResult` because both
/// callers have already partially moved `step_stats` / `updated_block_cache`
/// out by this point.
// Deliberately by-value and wide (see the doc above): both callers have
// already partially moved `step_stats` / `updated_block_cache` out of the
// `InterpreterStepResult`, so it cannot be re-borrowed as a whole here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_interpreter_step_result(
    state: &mut RustSimState,
    recovered_memory: Option<SymbolicMemory>,
    new_registers: RegisterFile,
    new_pc: u64,
    symbolic_ip_at_exit: Option<RustBV>,
    new_call_stack: Vec<CallStackEntry>,
    new_detailed_history: Vec<HistoryEntry>,
    new_tsc_counter: u64,
) {
    // Restore memory from interpreter back to state FIRST.
    // This must happen before any PendingCallback creation
    // because the state's memory was taken by set_rust_memory().
    if let Some(mem) = recovered_memory {
        state.replace_memory(mem);
    }
    // Restore registers (including symbolic values) from interpreter.
    state.set_registers(new_registers);
    state.set_pc(new_pc);
    // KEEP_IP_SYMBOLIC: overwrite the IP register (just concretized by
    // set_pc above) with the original symbolic next-pc expression. The
    // `state.pc` u64 still points to the concretized address so the next
    // block lift drives from there, but the IP register reads as the
    // unpinned symbolic expression — matching Python's
    // `split_state.regs.ip = target` in the `KEEP_IP_SYMBOLIC` branch of
    // `SimSuccessors._categorize_successor` (`angr/engines/successors.py`).
    if let Some(sym_ip) = symbolic_ip_at_exit {
        state.set_ip(sym_ip);
    }
    // Restore call stack and detailed history from interpreter.
    state.set_call_stack(new_call_stack);
    state.set_detailed_history(new_detailed_history);
    // Carry the simulated TSC forward; forks made below inherit it.
    state.set_tsc_counter(new_tsc_counter);
    // Add to history.
    state.add_to_history(state.pc());
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
        state: RustSimState,
        skip_addr: Option<u64>,
    ) -> Result<Vec<RustSimState>, StepError> {
        self.step_state_inner(callbacks, state, skip_addr, &[], None)
    }

    /// Body of [`step_state_with_skip`], with the two knobs the out-of-band
    /// `step_state()` pymethod (E1.a) needs:
    ///
    /// * `extra_stops` — stop addresses that apply to THIS call only. Unioned
    ///   into the step context's `stop_addrs` (the interpreter's break logic
    ///   already keys off that set); the manager-level `stop_addrs` are left
    ///   alone.
    /// * `terminal_sink` — when `Some`, the pruned/side-effect terminal states
    ///   that `apply_core_outcome` would push into their stashes are handed to
    ///   the caller instead, so a caller that owns state placement does not get
    ///   them stashed behind its back.
    ///
    /// [`step_state_with_skip`]: Self::step_state_with_skip
    pub(crate) fn step_state_inner(
        &mut self,
        callbacks: &PythonCallbacks,
        mut state: RustSimState,
        skip_addr: Option<u64>,
        extra_stops: &[u64],
        terminal_sink: Option<&mut Vec<(String, RustSimState)>>,
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
        let mut ctx = self.step_context();
        ctx.stop_addrs.extend(extra_stops.iter().copied());

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

        // Write the interpreter's step results back onto the state (shared with
        // the parallel path in run_loop.rs — see apply_interpreter_step_result).
        apply_interpreter_step_result(
            &mut state,
            step.recovered_memory,
            step.new_registers,
            step.new_pc,
            step.symbolic_ip_at_exit,
            step.new_call_stack,
            step.new_detailed_history,
            step.new_tsc_counter,
        );

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
        let cc = CoreCtx {
            ctx: &ctx,
            prof: &prof,
            native_procs: &self.native_procedures,
            native_syscalls: &self.native_syscalls,
            callbacks: Some(callbacks),
        };
        let outcome = run_post_step_core(&cc, state, inputs, root_hint);
        self.apply_core_outcome(callbacks, &prof, outcome, terminal_sink)
    }

    /// Apply the deferred mutations the `&mut self`-free post-step core recorded
    /// (angr-vh834), reproducing the single-threaded firing order byte-for-byte:
    /// fold profiling/counters, stamp `set_root` on every fork, fire each
    /// `dispatch_fork_inspect`, push pruned + side-effect terminal states, then
    /// translate the routing decision into the `Result` the run loop consumes.
    ///
    /// `terminal_sink` diverts steps 5-6: with `Some`, the pruned and
    /// side-effect terminal states are handed to the caller (tagged with the
    /// stash they *would* have gone to) instead of being pushed. Only
    /// `step_state()` (E1.a) passes it — the run loop passes `None` and keeps
    /// the legacy push behaviour byte-for-byte.
    fn apply_core_outcome(
        &mut self,
        callbacks: &PythonCallbacks,
        prof: &ParallelProfiling,
        outcome: CoreOutcome,
        terminal_sink: Option<&mut Vec<(String, RustSimState)>>,
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
                        // Kept `debug_assert!` (angr-9ke6b.220): per-successor
                        // per-step hot path, and a root_hint/is_fork mismatch
                        // only mis-groups a state for lineage sharing (a
                        // performance heuristic) — it cannot corrupt results.
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
        // 6. Side-effect terminal pushes (no-return main state from a native
        //    `exit` / no-return proc, deadended while its forks continue).
        //    Both are diverted to the caller when a terminal sink is set.
        let terminals = pruned
            .into_iter()
            .map(|s| (STASH_PRUNED, s))
            .chain(terminal_pushes.into_iter().map(|(s, stash)| (stash, s)));
        if let Some(sink) = terminal_sink {
            sink.extend(terminals.map(|(stash, s)| (stash.to_string(), s)));
        } else {
            for (stash, s) in terminals {
                self.push_or_drop_terminal(stash, s);
            }
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

    /// Body of the `step_state()` pymethod (E1.a): step ONE state out-of-band
    /// and hand its successors back to the caller instead of stashing them.
    ///
    /// The state is taken out of whatever stash holds it, stepped once (with
    /// `extra_stop_points` unioned into the stop-address set for this call
    /// only), and every resulting state — successors, unconstrained, pruned,
    /// deadended, errored — is parked in [`STASH_STEP_OUT`]. The returned map
    /// buckets their snapshots by category (`"flat"` for live successors, the
    /// destination stash name for the terminals) so the Python caller can place
    /// each one with `move_state(id, "_step_out", ...)`. Nothing lands in
    /// `active`/`deadended`/... behind the caller's back.
    ///
    /// A step that needs a Python callback bounce (SimProcedure, hook, syscall)
    /// raises `NotImplementedError` — driving the callback protocol from this
    /// entry point is E1.b's job — but the state is first parked in
    /// [`STASH_STEP_OUT`] (not dropped) so the caller can still find, re-place,
    /// or export it after the error (angr-ph300.22).
    pub(crate) fn _step_state(
        &mut self,
        state_id: u64,
        extra_stop_points: Option<Vec<u64>>,
    ) -> PyResult<HashMap<String, Vec<crate::state::ExplorationStateSnapshot>>> {
        let callbacks = self
            .callbacks
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("callbacks not set"))?
            .clone();
        // If this state is being stepped out of STASH_ACTIVE, it leaves the
        // active deque outside `policy.select` — notify so a memoizing policy
        // (LoopHeadRoundRobin's key_cache) doesn't leak a memo entry
        // (angr-myzjx.25). Capture the stash before the take unindexes it.
        let was_active = self.sm.stash_of(state_id) == Some(STASH_ACTIVE);
        let state = self
            .sm
            .take_state(state_id)
            .ok_or_else(|| PyValueError::new_err(format!("state {state_id} not found")))?;
        if was_active {
            self.policy.on_state_removed(state_id);
        }

        let extra_stops = extra_stop_points.unwrap_or_default();
        let mut terminals: Vec<(String, RustSimState)> = Vec::new();
        let outcome =
            self.step_state_inner(&callbacks, state, None, &extra_stops, Some(&mut terminals));

        let mut buckets: Vec<(String, RustSimState)> = match outcome {
            Ok(successors) => successors
                .into_iter()
                .map(|s| ("flat".to_string(), s))
                .collect(),
            Err(StepError::Deadended(s)) => vec![(STASH_DEADENDED.to_string(), s)],
            Err(StepError::Error(s, message)) => {
                self.errors.push((s.pc(), message, state_id));
                vec![(STASH_ERRORED.to_string(), s)]
            }
            // The unconstrained state goes to its own bucket; the loop-exit
            // forks materialized alongside it are live successors (angr-027h).
            Err(StepError::Unconstrained(s, forks)) => {
                let mut out = vec![(STASH_UNCONSTRAINED.to_string(), s)];
                out.extend(forks.into_iter().map(|f| ("flat".to_string(), f)));
                out
            }
            Err(StepError::NeedCallback(pending)) => {
                // Recover the state instead of dropping it: park it in
                // STASH_STEP_OUT before raising so the caller can find,
                // re-place, or export it (angr-ph300.22). Every other outcome
                // parks its results below; this error path must not silently
                // free the frontier state.
                let reason_desc = format!("{:?}", pending.reason);
                let state = pending.state;
                let id = state.state_id();
                self.index_state(id, STASH_STEP_OUT);
                self.sm.ensure_stash(STASH_STEP_OUT).push_back(state);
                return Err(PyNotImplementedError::new_err(format!(
                    "step_state() cannot drive Python callback bounces yet \
                     (reason: {reason_desc}); state {id} parked in {STASH_STEP_OUT}"
                )));
            }
        };
        buckets.extend(terminals);

        let mut snapshots: HashMap<String, Vec<crate::state::ExplorationStateSnapshot>> =
            HashMap::new();
        for (bucket, mut state) in buckets {
            snapshots
                .entry(bucket)
                .or_default()
                .push(state.flush_and_export_full());
            let id = state.state_id();
            self.index_state(id, STASH_STEP_OUT);
            self.sm.ensure_stash(STASH_STEP_OUT).push_back(state);
        }
        Ok(snapshots)
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
            stored_conditions,
            fork_snapshots,
        } = bounce;

        match kind {
            BounceKind::Hook { addr } => {
                state.set_pc(addr);
                // History BEFORE callback so Python can read recent_bbl_addrs[-1].
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
                    JumpKind::Boring.ijk_name(),
                    Some(shared_ctx),
                    ForkBundle {
                        deferred_forks,
                        stored_conditions,
                        fork_snapshots,
                    },
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
                    ForkBundle {
                        deferred_forks,
                        stored_conditions,
                        fork_snapshots,
                    },
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
                    ForkBundle {
                        deferred_forks,
                        stored_conditions,
                        fork_snapshots,
                    },
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
                ForkBundle {
                    deferred_forks,
                    stored_conditions,
                    fork_snapshots,
                },
            ),
            BounceKind::PythonVEXFallback { addr, reason } => {
                // state.set_pc(addr) was applied by the core before bouncing.
                Err(StepError::NeedCallback(PendingCallback::with_context(
                    state,
                    None,
                    CallbackReason::PythonVEXFallback { addr, reason },
                    JumpKind::Boring.ijk_name(),
                    None,
                    ForkBundle {
                        deferred_forks,
                        stored_conditions,
                        fork_snapshots,
                    },
                )))
            }
        }
    }

    /// Set up a native sub-call (`ProcOutcome::CallAndResume`) from the
    /// manager's live calling convention.
    ///
    /// Thin adapter over [`setup_native_subcall_with_abi`], which holds the
    /// dispatch logic and documents the per-ABI behavior; the parallel path's
    /// `CcSnapshot::setup_native_subcall` adapts the same helper from its
    /// scalar snapshot.
    pub(crate) fn setup_native_subcall(
        &self,
        state: &mut RustSimState,
        sub: NativeSubcall,
    ) -> Result<(), SubcallSetupError> {
        let cc = &self.environment.calling_convention;
        let abi = SubcallAbi {
            arg_registers: cc.arg_registers(),
            pointer_size: cc.pointer_size(),
            pops_return_addr: cc.pops_return_addr(),
            link_register: cc.link_register(),
        };
        setup_native_subcall_with_abi(&abi, state, sub)
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
    /// so it carries no `dead_code` allow (angr-0mqkc.2). The production twin
    /// has its own direct coverage in `core_outcome_tests.rs`
    /// (`native_resume_core_*`, angr-c7xno.32), so the two can no longer drift
    /// unobserved.
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
    fn handle_unmodeled_call(
        &mut self,
        callbacks: &PythonCallbacks,
        mut state: RustSimState,
        addr: u64,
        return_addr: u64,
        symbol_name: Option<String>,
        forks: ForkBundle,
    ) -> Result<Vec<RustSimState>, StepError> {
        let ForkBundle {
            deferred_forks,
            stored_conditions,
            fork_snapshots,
        } = forks;

        // Unhooked CALL target - try to resolve via Python callback
        state.set_pc(addr);
        // Add to history BEFORE callback so Python can access recent_bbl_addrs[-1]
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
                        ForkBundle {
                            deferred_forks,
                            stored_conditions,
                            fork_snapshots,
                        },
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
        // Guards of the forks already materialized. `successors[0]` accumulates
        // them below, but a fork built from a pre-branch *snapshot* does not —
        // see `PriorGuards` (angr-62ar5).
        let mut prior_guards = super::fork_materialize::PriorGuards::new(true);

        for fork in &deferred_forks {
            if let Some(condition) = stored_conditions.get(&fork.condition_id) {
                // Add the taken-path constraint to the main state (fires the
                // constraints inspect BP around the add — angr-op0dn.14.4.1).
                super::fork_materialize::add_fork_guard_constraint(
                    self.callbacks.as_ref(),
                    &successors[0],
                    condition,
                    fork.path_taken,
                );

                // Create forked state for the unexplored path
                let forked = super::fork_materialize::build_unexplored_fork(
                    &successors[0],
                    fork,
                    condition,
                    &mut fork_snapshots,
                    &prior_guards,
                );
                prior_guards.record(condition.clone(), fork.path_taken);

                self.sm.set_root(forked.state_id(), root_state_id);

                // state.inspect fork BP — see `dispatch_fork_inspect` for rationale.
                self.dispatch_fork_inspect(forked.state_id());

                if forked.survives_sat_prune(self.constraint_solver.lazy_solves) {
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

                if forked.survives_sat_prune(self.constraint_solver.lazy_solves) {
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
    /// matching Python `SimSuccessors._preprocess_successor`
    /// (`angr/engines/successors.py`), where the BP fires
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
        if !cb.inspect_event_enabled(crate::callbacks::InspectBit::Fork) {
            return;
        }
        // call_inspect_fork self-attaches the GIL (angr-vh834 Phase 4), so no
        // explicit Python::attach wrapper is needed here.
        if let Err(e) = cb.call_inspect_fork(forked_state_id as i64, "after") {
            log::debug!("fork inspect dispatch raised (state {forked_state_id}): {e}");
        }
    }
}

test_submod!("sizes_tests.rs" => sizes);

test_submod!("subcall_tests.rs" => subcall_tests);
