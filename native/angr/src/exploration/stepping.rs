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
//! `dispatch_bounce` — the Python-bouncing arm, in `stepping_bounce.rs` —
//! turns a [`PendingBounce`](super::core_outcome::PendingBounce) into the exact
//! [`PendingCallback`] the caller must
//! service. The matching re-entry points live in `resume.rs`.
//!
//! Three sibling `#[path]` submodules hold the concerns that used to share
//! this file (angr-5mnx3.71), each reached only through `RustExplorationManager`
//! methods so no import path outside `stepping.rs` changed:
//!
//! - `stepping_subcall.rs` — the native sub-call ABI setup shared with the
//!   parallel path ([`SubcallAbi`] / [`setup_native_subcall_with_abi`], both
//!   re-exported here for `core_outcome.rs`).
//! - `stepping_bounce.rs` — `dispatch_bounce` and the unmodeled-call arm.
//! - `stepping_forks.rs` — `process_deferred_forks_into` /
//!   `dispatch_fork_inspect`.
//!
//! What stays: [`StepError`], [`InterpreterStepResult`] and
//! `apply_interpreter_step_result`, the `step_state_*` / `apply_core_outcome`
//! spine, core counter folding (`fold_core_counters`), and the out-of-band
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
//! `tests/engines/rust/`. The two test submodules cover the slices that *are*
//! reachable without a lift and are named for what they test rather than for
//! the file they hang off: `sizes_tests.rs` here (the `StepError` /
//! `PendingCallback` size probe backing the `result_large_err` suppression) and
//! `subcall_tests.rs` under `stepping_subcall.rs` (`setup_native_subcall` /
//! `handle_native_resume_core`, which stand in for the guest `ret` instead of
//! lifting one). The lift-free routing prefix of the *caller* — `step_one`'s find/avoid and SimProcedure-fallback arms — is
//! covered in `run_loop_single_tests.rs`.

use super::core_outcome::{
    CoreCounters, CoreCtx, CoreOutcome, CoreReturn, ParallelProfiling, PostStepInputs,
    run_post_step_core,
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

#[path = "stepping_subcall.rs"]
mod subcall;
pub(crate) use subcall::{SubcallAbi, SubcallSetupError, setup_native_subcall_with_abi};

#[path = "stepping_bounce.rs"]
mod bounce;

#[path = "stepping_forks.rs"]
mod forks;

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
}

test_submod!("sizes_tests.rs" => sizes);
