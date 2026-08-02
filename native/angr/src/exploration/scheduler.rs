//! Work-stealing scheduler machinery for parallel exploration (angr-1ilq.3,
//! correctness-first isolated increment; anti-migration redesign angr-729vn).
//!
//! # What this is
//!
//! This module holds the *threading + migration + cancellation* machinery for
//! a multi-worker exploration pool, running on real `std::thread::scope`
//! workers. It **is** wired into the run loop:
//! [`run_loop`](super::run_loop) builds a [`PersistentPool`] from this module
//! and drives each wave via `py.detach(|| pool.run_wave(job))` when
//! `RUST_PARALLEL_WORKERS > 1`, and the nightly `run_findall_gate.py` CI lane
//! exercises it for worker-count invariance.
//!
//! The design originally landed as an isolated increment (angr-1ilq.3) because
//! stepping is pervasively GIL-coupled: [`RustExplorationManager::
//! step_state_with_skip`](super::stepping) takes a live `Python<'_>` token and
//! yields back to Python at seven callback points (find/avoid predicates,
//! SimProcedures, syscalls, symbolic branches, Python-VEX fallback, errors). A
//! worker thread cannot run a real engine step without holding the GIL, so the
//! GIL is released only around the Rust-pure inner work and re-acquired for
//! callbacks (the callback-dispatch half was angr-1ilq.4). The scheduler's own
//! contract is pure thread-safety: worker-local live states are explored in a
//! private Z3 context, with cross-worker transport happening *only* on an
//! actual imbalance steal — all under genuine concurrency, with zero `unsafe`.
//!
//! # Transport invariant (angr-729vn — anti-migration)
//!
//! The pivot from `angr-t3l5o` (2026-06-30): the parallel blocker is migration
//! *count*, not per-state transport cost. The residual ~14.5 ms/state migration
//! tax on `codegate` is ~60% an irreducible Z3-AST-rebuild floor that exceeds
//! the per-state work budget, so per-state migration cannot pay off **if every
//! state migrates**. The lever is therefore to migrate as *few* states as
//! possible. This scheduler is built around that:
//!
//! * Each worker keeps its live successors in a thread-private
//!   `VecDeque<RustSimState>` (the *home-context fast path*). These states stay
//!   in the worker's own Z3 context and are **never serialized**.
//! * The only cross-thread channel is a shared [`Injector`] of
//!   [`StateMigrationPayload`]s (`Send` by construction). A state is detached
//!   into a payload — paying the serde + Z3-AST-rebuild tax — **only** when:
//!   1. it is *surplus* offered for stealing on imbalance (see
//!      [`worker::offload_surplus`]), or
//!   2. it is a *materialized* terminal the caller must recover across the
//!      `thread::scope` join (found/matched states).
//! * Deadended/errored terminals are returned as a lightweight
//!   [`TerminalSummary`] built in the worker's context and dropped there — they
//!   pay **no** serde at all. (This is a selective `drop_terminal_states`: the
//!   full symbolic state of dead paths is not recoverable through the parallel
//!   path.)
//! * A stealing worker [`reattach`](StateMigrationPayload::reattach)es a payload
//!   into *its own* thread-local context (every AST minted locally; the source
//!   context is never read cross-thread — this is what makes the design sound
//!   where the rejected `translate_state`-on-steal was not; hazard C on the
//!   bead).
//!
//! The fraction of dispatched states that get serialized — `surplus_offloaded +
//! materialized_terminals` over total dispatches — is the *honest steal
//! fraction* the overhead gate's break-even `f*` bounds. See
//! [`SchedulerStats`] and `tests/benchmarks/run_parallel_overhead_gate.py`.
//!
//! # Panic policy: why the `.expect` sites are invariant guards, not error paths
//!
//! The shipped `.so` is built with `[profile.release] panic = "abort"`
//! (workspace `Cargo.toml`, angr-1cue). That single fact settles the
//! panic-hardening audit (CQ .8) for this module:
//!
//! * **Mutex poisoning cannot happen.** A `Mutex` is only poisoned when a
//!   thread *unwinds* out of a live `MutexGuard`. Under `panic = "abort"` there
//!   is no unwind: a panic while a guard is held aborts the process at the panic
//!   site, before the guard's `Drop` could ever flag the lock. Every
//!   `.lock().expect("… poisoned")` here (results/summaries/done_rx) is
//!   therefore *provably unreachable* — the message names a state that this
//!   build can never produce.
//! * **A worker cannot "die" mid-wave into a live-pool disconnect.** A worker
//!   thread leaves [`worker_thread`] only on `Shutdown`/closed-channel (clean
//!   teardown) or by panicking — and a panic aborts the whole process. So while
//!   a wave is in flight every worker is alive; the coordinator's
//!   `.send(...).expect("persistent worker died …")` and
//!   `.recv().expect("… WaveDone")` can only observe disconnect at pool
//!   teardown, never during [`run_wave`](PersistentPool::run_wave).
//!
//! Consequently there is **no fallible site here to propagate** and **no
//! "surface as a Python exception" path to build**: under `panic = "abort"` a
//! genuine bug in a worker surfaces as a process abort (SIGABRT), by design —
//! the same constraint `symbolic::value_z3::fresh_unconstrained_raw` documents
//! for its own `catch_unwind`-is-useless reasoning. The `.expect` messages are
//! kept as invariant labels: if one ever *did* fire it would mean the
//! panic-strategy assumption changed, which is exactly the signal a future
//! reader wants. A forced-poison test is deliberately **not** added — it cannot
//! observe a Python exception under this profile, only an abort.
//!
//! The one site the argument above does *not* cover is
//! `Builder::spawn(..).expect("failed to spawn persistent worker thread")` in
//! [`PersistentPool::new`], which is genuinely fallible (thread spawn can fail
//! with `EAGAIN` under thread/memory exhaustion). It stays a panic
//! deliberately: `new` returns `Self` and is called from the run loop's
//! pool-construction path, so propagating would ripple a `Result` through the
//! parallel driver, and the obvious local degradation — carry on with fewer
//! workers — silently returns an empty wave if *zero* threads came up, trading
//! a loud abort for lost states. Falling back to serial exploration instead is
//! a real feature, not a lint cleanup; tracked on angr-9ke6b.212.
//!
//! **Enforcement (angr-qwyti.11):** this module carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]`; each function holding
//! one of the guards above carries a narrow
//! `#[allow(clippy::expect_used, reason = ...)]` pointing back at this Panic
//! policy, so a new panic cannot slip in unreviewed.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use crate::exploration::selection_policy::SelectionPolicy;
// `Lifo` is the pre-seam default policy, now only used by the `#[cfg(test)]`
// `Lifo`-default convenience constructors (production threads an explicit policy
// via `*_with_policy`).
#[cfg(test)]
use crate::exploration::selection_policy::Lifo;
use crate::interpreter::BLOCK_CACHE_CAPACITY_NZ;
use crate::state::{RustSimState, StateMigrationPayload};
use crate::vex::IRSB;
use crossbeam_deque::{Injector, Steal};
use lru::LruCache;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use z3::{Config, Context};

/// High-water mark for a worker's local live queue. Above this the worker
/// sheds surplus to the injector even with no idle sibling (Trigger B), bounding
/// per-worker memory. Sized well above any narrow-frontier steady state so it is
/// a safety cap, not the primary load-balancer (that is the idle-gated Trigger
/// A). See [`worker::offload_surplus`].
const LOCAL_HWM: usize = 64;

/// Cooperative cancellation shared across all workers.
///
/// Checked at **task boundaries** — a worker finishes its current task, then
/// stops before pulling the next one. That is the migration granularity the
/// design targets (`rust_parallel_design.rst`: migration is viable only at
/// task boundaries), so task-boundary cancellation is the matching grain.
///
/// This is deliberately NOT a mid-solve Z3 interrupt. `Context::handle()
/// .interrupt()` is available and `ContextHandle` is `Send + Sync` (it is the
/// right tool to abort a long in-flight solve), but calling it safely across
/// threads requires keeping each worker's context alive until every possible
/// interrupter has stopped — coupling that belongs with the run-loop
/// integration increment (where solve durations actually matter). Wiring it
/// here would add a cross-thread raw-pointer lifetime hazard for no benefit at
/// the current granularity.
#[derive(Clone, Default)]
pub(crate) struct CancelToken {
    flag: Arc<AtomicBool>,
    /// Set alongside `flag` when the stop was requested by the wave's dispatch
    /// budget ([`WorkTransport::max_dispatches`], angr-9ke6b.52) rather than by
    /// a find / finalize. The distinction matters for IN-FLIGHT work only: a
    /// find cancel wants peers to drop the state they just picked up
    /// (speculative waste past `num_find`), but a budget cancel must let it
    /// finish — the dispatch was already charged to the budget, and dropping it
    /// unstepped livelocks a small `run(n)`: with `n = 1` and W workers, the
    /// peer that observes the budget spent cancels while the one worker that
    /// dispatched is still mid-step, so the wave returns having advanced
    /// nothing and the next `run(1)` repeats it forever.
    budget: Arc<AtomicBool>,
}

impl CancelToken {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Request that all workers stop at their next task boundary, preempting any
    /// in-flight step (find / finalize semantics).
    pub(crate) fn cancel(&self) {
        // Clear `budget` FIRST: a find cancel always preempts, even if a budget
        // stop got there first.
        self.budget.store(false, Ordering::SeqCst);
        self.flag.store(true, Ordering::SeqCst);
    }

    /// Request that all workers stop at their next task boundary, but let
    /// already-dispatched states finish their step. See the `budget` field docs.
    pub(crate) fn cancel_for_budget(&self) {
        self.budget.store(true, Ordering::SeqCst);
        self.flag.store(true, Ordering::SeqCst);
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }

    /// Whether cancellation should preempt a state that is already dispatched.
    /// True for find/finalize cancels, false for a pure budget stop.
    pub(crate) fn preempts_in_flight(&self) -> bool {
        self.is_cancelled() && !self.budget.load(Ordering::SeqCst)
    }
}

/// How a non-materialized terminal ended, captured in a [`TerminalSummary`].
///
/// Found/matched terminals are *not* represented here — they are materialized
/// (fully serialized) so the caller can recover the satisfying state. These are
/// the dispositions whose full symbolic state the parallel path discards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TerminalDisposition {
    Deadended,
    Errored,
    Avoided,
    Pruned,
}

/// A lightweight record of a terminal state that did **not** need to cross the
/// worker boundary as a full state. Built in the worker's own context from
/// cheap scalar fields, so it pays no serde / Z3-AST-rebuild tax.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TerminalSummary {
    pub state_id: u64,
    pub pc: u64,
    pub disposition: TerminalDisposition,
}

impl TerminalSummary {
    /// Summarize a terminal state without serializing it. The `state` is read
    /// (cheap scalar fields only) and then dropped in its home context.
    pub(crate) fn of(state: &RustSimState, disposition: TerminalDisposition) -> Self {
        Self {
            state_id: state.state_id(),
            pc: state.pc(),
            disposition,
        }
    }
}

/// What a worker produced from processing one state, expressed in that worker's
/// own Z3 context.
///
/// `continue_states` stay live and local (no serde). `terminal_states` are
/// *materialized* — fully serialized to cross the `thread::scope` join, for
/// terminals the caller must recover (found/matched). `terminal_summaries` are
/// lightweight records for dead paths that pay no serde.
pub(crate) struct TaskOutcome {
    /// Successors to keep exploring; pushed live onto the worker's local queue.
    pub continue_states: Vec<RustSimState>,
    /// Terminal states the caller must recover in full — serialized across the
    /// join (found / matched). Keep this set small: it is part of the honest
    /// steal fraction.
    pub terminal_states: Vec<RustSimState>,
    /// Terminal states recorded as cheap summaries (deadended / errored / …);
    /// dropped in the worker context, never serialized.
    pub terminal_summaries: Vec<TerminalSummary>,
    /// Request global cancellation (e.g. the find target was reached).
    pub request_cancel: bool,
}

impl TaskOutcome {
    /// All successors are live; keep exploring, nothing terminal.
    /// Test-only constructor: the synthetic `process` fns in the byte-stable
    /// scheduler unit tests build outcomes with this; production task closures
    /// assemble `TaskOutcome` fields directly.
    #[cfg(test)]
    pub(crate) fn continuing(continue_states: Vec<RustSimState>) -> Self {
        Self {
            continue_states,
            terminal_states: Vec::new(),
            terminal_summaries: Vec::new(),
            request_cancel: false,
        }
    }

    /// All successors are terminal and must be materialized; generate no further
    /// work.
    /// Test-only constructor (see [`TaskOutcome::continuing`]).
    #[cfg(test)]
    pub(crate) fn terminal(terminal_states: Vec<RustSimState>) -> Self {
        Self {
            continue_states: Vec::new(),
            terminal_states,
            terminal_summaries: Vec::new(),
            request_cancel: false,
        }
    }

    /// All successors are dead paths recorded as summaries; no further work, no
    /// serde.
    pub(crate) fn summarized(terminal_summaries: Vec<TerminalSummary>) -> Self {
        Self {
            continue_states: Vec::new(),
            terminal_states: Vec::new(),
            terminal_summaries,
            request_cancel: false,
        }
    }
}

/// Counters accumulated across all workers during a run. Shared as atomics, read
/// back into a [`SchedulerStats`] after the pool joins.
///
/// `pub(crate)` so the coordinator's forthcoming duplex-protocol `RunSession`
/// (run_loop_steady.rs) can own one directly (angr-vh834 steady-state redesign, Phase
/// 1). The scheduler still constructs and reads it here.
#[derive(Default)]
pub(crate) struct SchedulerCounters {
    local_dispatches: AtomicUsize,
    injector_dispatches: AtomicUsize,
    surplus_offloaded: AtomicUsize,
    materialized_terminals: AtomicUsize,
    summarized_terminals: AtomicUsize,
    /// Per-disposition split of `summarized_terminals` (angr-op0dn.13.15). The
    /// summarized states themselves are dropped in-worker, but their *counts*
    /// must still reach the manager's `deadended_count` / `pruned_count` /
    /// `errored_count` / `avoided_count`, or a parallel run reports zero dead
    /// paths where the serial loop reports N. `Avoided` DOES occur here
    /// (angr-pwu71): avoid-routing is usually a coordinator decision, but
    /// `parallel_process_state` also matches a successor's pc against
    /// `avoid_addrs` in-worker, so the fourth slot is required.
    summarized_deadended: AtomicUsize,
    summarized_errored: AtomicUsize,
    summarized_pruned: AtomicUsize,
    summarized_avoided: AtomicUsize,
    /// Payloads pulled from the injector and `reattach`ed into a worker's own Z3
    /// context (the injector-steal path). Observability only — equals
    /// `injector_dispatches` today; the two diverge once the steady-state
    /// coordinator reattaches on paths other than an injector steal (Phase 2+).
    reattaches: AtomicUsize,
    /// Bounce states that made a full worker->coordinator->worker round trip.
    /// Wired by the steady-state coordinator (run_loop_steady.rs); 0 in wave mode.
    pub(crate) bounce_roundtrips: AtomicUsize,
    /// States re-injected after a Python resume callback
    /// ([`RunSession::inject_resumed`]); 0 in wave mode.
    resume_reinjects: AtomicUsize,
    /// Still-live frontier states a worker detached and handed back to the
    /// coordinator on cancel/finalize instead of dropping them (the Bug M1
    /// cancel-drain: [`worker::drain_local_upstream`] in session mode, the
    /// wave loop's `results` push in wave mode), PLUS the never-stolen injector
    /// surplus the coordinator pulls with [`WorkTransport::drain_residual_payloads`].
    /// Distinct from `materialized_terminals` so residual drains never pollute
    /// the honest steal fraction. 0 unless the run was cancelled.
    residual_drains: AtomicUsize,
    /// Steps a worker executed AFTER cancellation was already requested by a
    /// *different* origin (a peer worker's find or the coordinator's num_find
    /// route) — the post-find speculative waste (angr-1ilq.8). A step counts
    /// here when `cancel` is set on `process` return but this step did not
    /// itself raise the cancel (`!request_cancel`). Because both worker loops
    /// re-check `cancel` at the top of every iteration, this captures exactly
    /// the in-flight steps that were committed before the cancel became
    /// visible: the wasted work the find-aware dispatch bead (angr-1ilq.9)
    /// aims to eliminate. 0 until the first find raises a cancel.
    post_cancel_steps: AtomicUsize,
    /// Dispatches per worker id (angr-op0dn.13.9): the load-balance column the
    /// S7 find-all gate needs (`max/min dispatched per worker`). Indexed by
    /// `worker_id`; ids >= [`MAX_TRACKED_WORKERS`] fold into the last slot
    /// (documented lossiness — real runs use <= num_cpus workers). When that
    /// fold does happen it is reported by [`Self::folded_worker_dispatches`]
    /// rather than left silent (angr-9ke6b.67).
    worker_dispatches: [AtomicUsize; MAX_TRACKED_WORKERS],
    /// Dispatches that landed on a worker id >= [`MAX_TRACKED_WORKERS`] and were
    /// therefore folded into the last [`Self::worker_dispatches`] slot
    /// (angr-9ke6b.67). Non-zero means the per-worker load-balance column is
    /// degraded — the tail workers are merged into one bucket, so its max/min
    /// ratio is not trustworthy. Also drives a one-time `log::warn!` on the
    /// first fold so a run producing degraded data says so on stderr.
    folded_worker_dispatches: AtomicUsize,
    /// Step-weighted schedulable-frontier width, bucketed as
    /// `[==1, ==2, 3-4, 5-8, >=9]` — the parallel-path analogue of the serial
    /// model's `parallel_width_hist` (`record_migration_sample` in helpers.rs).
    /// Sampled at dispatch from `pending` (queued + in-flight states), so both
    /// worker loops record it and the frontier-residency mode is no longer
    /// invisible to the width audit.
    width_hist: [AtomicUsize; 5],
    /// Peak `pending` observed at dispatch — the parallel `max_active_width`.
    max_width: AtomicUsize,
    /// Nanoseconds all workers spent inside Z3 migration serde — every
    /// `detach_for_migration` and every `reattach` on the steal path. Paired with
    /// [`Self::step_ns`] to form the serde budget that gates Trigger A offloads
    /// (`worker::offload_is_affordable`, angr-8shhe).
    pub(crate) serde_ns: AtomicU64,
    /// Nanoseconds all workers spent inside the step function itself — the useful
    /// work the serde is a tax on.
    pub(crate) step_ns: AtomicU64,
}

/// Per-worker dispatch slots tracked by [`SchedulerCounters::worker_dispatches`].
/// 32 covers every realistic `RUST_PARALLEL_WORKERS`; higher ids fold into the
/// last slot rather than allocating (the counters live on a hot path and
/// `Default`-derived arrays cap at 32).
pub(crate) const MAX_TRACKED_WORKERS: usize = 32;

impl SchedulerCounters {
    /// Live total of dispatches so far (`local + injector`), the same sum
    /// [`SchedulerStats::dispatches`] reports post-barrier. Read mid-run by
    /// [`WorkTransport::dispatch_budget_exhausted`]; `Relaxed` is sufficient
    /// because the budget it feeds is a soft, task-boundary-checked cap, not a
    /// synchronization edge.
    fn total_dispatches(&self) -> u64 {
        (self.local_dispatches.load(Ordering::Relaxed)
            + self.injector_dispatches.load(Ordering::Relaxed)) as u64
    }

    /// Record one dispatch on `worker_id` observing `width` schedulable states
    /// (queued + in-flight, including the state being dispatched). Two relaxed
    /// bumps plus a max-CAS: cheap enough for the dispatch hot path.
    ///
    /// Buckets match the serial model's `record_migration_sample` (helpers.rs)
    /// so the two width histograms are directly comparable.
    pub(crate) fn record_dispatch(&self, worker_id: usize, width: usize) {
        if worker_id >= MAX_TRACKED_WORKERS {
            // SILENT(cat-b): the per-worker column loses its tail resolution here,
            // but the fold is counted and warned about exactly once, so consumers
            // (the S7 find-all gate) can see the data is degraded instead of
            // reading a merged bucket as a real worker's load.
            if self
                .folded_worker_dispatches
                .fetch_add(1, Ordering::Relaxed)
                == 0
            {
                log::warn!(
                    "worker id {worker_id} exceeds MAX_TRACKED_WORKERS={MAX_TRACKED_WORKERS}; \
                     per-worker dispatch stats fold ids >= {MAX_TRACKED_WORKERS} into the last \
                     slot — the load-balance max/min column is degraded for this run",
                );
            }
        }
        self.worker_dispatches[worker_id.min(MAX_TRACKED_WORKERS - 1)]
            .fetch_add(1, Ordering::Relaxed);
        let bucket = match width {
            0 | 1 => 0,
            2 => 1,
            3..=4 => 2,
            5..=8 => 3,
            _ => 4,
        };
        self.width_hist[bucket].fetch_add(1, Ordering::Relaxed);
        self.max_width.fetch_max(width, Ordering::Relaxed);
    }

    /// Record one task's dead-path summaries: the total plus the per-disposition
    /// split the coordinator folds into the manager's terminal counters. Shared
    /// by BOTH worker loops (wave + session) so neither can grow its own
    /// accounting.
    pub(crate) fn record_summaries(&self, summaries: &[TerminalSummary]) {
        if summaries.is_empty() {
            return;
        }
        self.summarized_terminals
            .fetch_add(summaries.len(), Ordering::SeqCst);
        for s in summaries {
            let slot = match s.disposition {
                TerminalDisposition::Deadended => &self.summarized_deadended,
                TerminalDisposition::Errored => &self.summarized_errored,
                TerminalDisposition::Pruned => &self.summarized_pruned,
                // Reachable from `parallel_process_state`, which checks a
                // successor's pc against `avoid_addrs` on the worker thread
                // (angr-pwu71) — not only the coordinator's avoid-routing.
                TerminalDisposition::Avoided => &self.summarized_avoided,
            };
            slot.fetch_add(1, Ordering::SeqCst);
        }
    }
}

/// Per-run accounting, the basis for the overhead-gate steal-fraction check.
///
/// The only states that pay the serde / Z3-AST-rebuild tax are
/// `surplus_offloaded` (cross-worker steals) plus `materialized_terminals`
/// (found states recovered across the join). [`honest_steal_fraction`] reports
/// that as a fraction of non-seed dispatches — the quantity the gate's
/// break-even `f*` bounds.
///
/// [`honest_steal_fraction`]: SchedulerStats::honest_steal_fraction
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SchedulerStats {
    /// Initial payloads handed to [`ParallelScheduler::run_instrumented`].
    pub seeds: usize,
    /// Nanoseconds all workers spent in Z3 migration serde (detach + reattach),
    /// and in the step function itself. The pair is the serde budget that gates
    /// Trigger A offloads (`worker::offload_is_affordable`); surfaced here so a
    /// migration-dominated frontier is diagnosable from the wave log rather than
    /// from a flamegraph (angr-faorh/8shhe).
    pub serde_ns: u64,
    pub step_ns: u64,
    /// Tasks dispatched from a worker's local live queue (zero serde).
    pub local_dispatches: usize,
    /// Tasks pulled from the injector (each paid one `reattach`). Includes the
    /// initial seeds.
    pub injector_dispatches: usize,
    /// Continue-states detached to the injector on imbalance. The migration
    /// count and the *only* continue-path serde site.
    pub surplus_offloaded: usize,
    /// Found/matched terminals serialized across the join.
    pub materialized_terminals: usize,
    /// Dead-path terminals recorded as summaries (no serde).
    pub summarized_terminals: usize,
    /// Per-disposition split of `summarized_terminals`; the coordinator folds
    /// these into the manager's `deadended_count` / `errored_count` /
    /// `pruned_count` / `avoided_count` so terminal accounting is worker-count
    /// invariant even though the states themselves are dropped in-worker
    /// (angr-op0dn.13.15, extended to `avoided` by angr-pwu71).
    pub summarized_deadended: usize,
    pub summarized_errored: usize,
    pub summarized_pruned: usize,
    pub summarized_avoided: usize,
    /// Payloads reattached into a worker's own Z3 context (injector-steal path).
    /// Observability only; see [`SchedulerCounters::reattaches`].
    pub reattaches: usize,
    /// Bounce states that round-tripped worker->coordinator->worker (0 in wave
    /// mode; counted by the steady-state coordinator).
    pub bounce_roundtrips: usize,
    /// States re-injected after a Python resume (0 in wave mode; counted by
    /// [`RunSession::inject_resumed`]).
    pub resume_reinjects: usize,
    /// Still-live frontier states drained back to the coordinator on cancel /
    /// finalize (both modes; 0 unless the run was cancelled).
    pub residual_drains: usize,
    /// Post-find speculative steps: work committed on a worker after another
    /// origin already requested cancel (angr-1ilq.8). See
    /// [`SchedulerCounters::post_cancel_steps`].
    pub post_cancel_steps: usize,
    /// Dispatches per worker id (angr-op0dn.13.9). Length is always
    /// [`MAX_TRACKED_WORKERS`]; the run loop trims it to the configured worker
    /// count before exporting. See [`SchedulerCounters::worker_dispatches`].
    pub worker_dispatches: Vec<usize>,
    /// Dispatches folded into the last [`Self::worker_dispatches`] slot because
    /// their worker id was >= [`MAX_TRACKED_WORKERS`] (angr-9ke6b.67). Non-zero
    /// marks [`Self::worker_dispatches`] as degraded data.
    pub folded_worker_dispatches: usize,
    /// Step-weighted frontier-width histogram `[==1, ==2, 3-4, 5-8, >=9]`.
    pub width_hist: [usize; 5],
    /// Peak schedulable-frontier width observed at dispatch.
    pub max_width: usize,
}

impl SchedulerStats {
    /// Total states dispatched (the denominator of the steal fraction). Every
    /// processed state is dispatched exactly once, from the local queue or the
    /// injector. Matches the `parallel_tasks` denominator of the run loop's
    /// `f_model` (helpers.rs), so the two steal fractions are comparable.
    pub(crate) fn dispatches(&self) -> usize {
        self.local_dispatches + self.injector_dispatches
    }

    /// The honest fraction of dispatched states that paid the serde tax:
    /// `(surplus_offloaded + materialized_terminals) / dispatches`. This is the
    /// number the overhead gate's break-even `f*` must bound for a GO. Returns
    /// 0.0 when nothing was dispatched.
    ///
    /// Test-only: the overhead-gate assertions in the unit suite check this
    /// ratio; the run loop reads the raw counters directly.
    #[cfg(test)]
    pub(crate) fn honest_steal_fraction(&self) -> f64 {
        let d = self.dispatches();
        if d == 0 {
            return 0.0;
        }
        (self.surplus_offloaded + self.materialized_terminals) as f64 / d as f64
    }
}

/// Snapshot a [`SchedulerCounters`] into a [`SchedulerStats`] — shared by
/// [`WaveJob::stats`] (post-barrier) and [`RunSession::stats`] (any time; the
/// atomics make a mid-session snapshot merely slightly stale, never torn).
fn snapshot_stats(seeds: usize, counters: &SchedulerCounters) -> SchedulerStats {
    SchedulerStats {
        seeds,
        serde_ns: counters.serde_ns.load(Ordering::SeqCst),
        step_ns: counters.step_ns.load(Ordering::SeqCst),
        local_dispatches: counters.local_dispatches.load(Ordering::SeqCst),
        injector_dispatches: counters.injector_dispatches.load(Ordering::SeqCst),
        surplus_offloaded: counters.surplus_offloaded.load(Ordering::SeqCst),
        materialized_terminals: counters.materialized_terminals.load(Ordering::SeqCst),
        summarized_terminals: counters.summarized_terminals.load(Ordering::SeqCst),
        summarized_deadended: counters.summarized_deadended.load(Ordering::SeqCst),
        summarized_errored: counters.summarized_errored.load(Ordering::SeqCst),
        summarized_pruned: counters.summarized_pruned.load(Ordering::SeqCst),
        summarized_avoided: counters.summarized_avoided.load(Ordering::SeqCst),
        reattaches: counters.reattaches.load(Ordering::SeqCst),
        bounce_roundtrips: counters.bounce_roundtrips.load(Ordering::SeqCst),
        resume_reinjects: counters.resume_reinjects.load(Ordering::SeqCst),
        residual_drains: counters.residual_drains.load(Ordering::SeqCst),
        post_cancel_steps: counters.post_cancel_steps.load(Ordering::SeqCst),
        worker_dispatches: counters
            .worker_dispatches
            .iter()
            .map(|c| c.load(Ordering::SeqCst))
            .collect(),
        folded_worker_dispatches: counters.folded_worker_dispatches.load(Ordering::SeqCst),
        width_hist: std::array::from_fn(|i| counters.width_hist[i].load(Ordering::SeqCst)),
        max_width: counters.max_width.load(Ordering::SeqCst),
    }
}

/// A boxed, thread-safe per-state processor. Production (`run_loop_wave.rs` / `run_loop_steady.rs`) wraps
/// `parallel_process_state`, capturing the owned `StepContext` plus `Arc`-shared
/// callbacks / profiling / native registries / `ParallelShared`; the scheduler
/// unit tests wrap synthetic closures. Boxing behind a `dyn` (one indirect call
/// per dispatch — negligible) keeps [`WaveJob`] a single *concrete* `Send + Sync`
/// type, so [`PersistentPool`]'s channels can carry `Arc<WaveJob>` without a
/// generic parameter leaking onto the manager's `parallel_pool` field, and keeps
/// this module fully decoupled from the run-loop's config types.
///
/// The `&mut LruCache` is the worker's WARM per-thread block cache
/// (angr-vh834 Work Item 3): owned by [`worker_thread`], created ONCE alongside
/// the Z3 context and threaded into every dispatch so lifted blocks persist
/// across dispatches AND waves for that worker. `Arc<IRSB>` values are AST-free
/// (no Z3 ASTs) and never cross worker threads, so warming this cache needs NO
/// new Send/Sync bound.
pub(crate) type ProcessFn =
    dyn Fn(RustSimState, &CancelToken, &mut LruCache<u64, Arc<IRSB>>) -> TaskOutcome + Send + Sync;

/// The work-distribution core shared by BOTH scheduling modes: the wave loop's
/// [`WaveJob`] and the steady-state [`RunSession`] embed one, so the dispatch /
/// steal / offload machinery ([`worker::dispatch_next`], [`worker::steal_from_injector`],
/// [`worker::offload_surplus`]) is written once against this struct (DRY) and the two
/// modes differ only in how terminals leave the worker and what a worker does
/// when it runs out of work (wave: return to the barrier; session: park or
/// drain).
pub(crate) struct WorkTransport {
    /// Shared work queue of migratable states — the ONLY cross-thread state
    /// channel.
    injector: Injector<StateMigrationPayload>,
    /// Count of queued + in-flight states (the quiescence detector). A worker
    /// counts spawned children IN before counting the parent task OUT.
    pending: AtomicUsize,
    /// Cooperative cancellation, checked at task boundaries. In session mode
    /// this doubles as the finalize/drain signal: a cancelled session worker
    /// detaches its residual frontier upstream instead of dropping it.
    cancel: CancelToken,
    /// Number of workers currently spinning for work (starvation signal for
    /// [`worker::offload_surplus`] Trigger A).
    idle_workers: AtomicUsize,
    counters: SchedulerCounters,
    /// Active-state selection / fork-insertion policy for the worker-local
    /// frontier (angr-1ilq.9). [`worker::dispatch_next`] routes its local pop through
    /// [`SelectionPolicy::select`] and [`worker::absorb_continues`] through
    /// [`SelectionPolicy::on_fork`], so a find-aware policy can reorder the
    /// per-worker frontier without touching the dispatch skeleton. Defaults to
    /// [`Lifo`] — the pre-seam behavior was an open-coded `pop_back` /
    /// `push_back`, exactly what `Lifo` reproduces, so the default is
    /// byte-for-byte zero-regression. Shared across worker threads as an `Arc`
    /// (the trait is `Send + Sync`).
    policy: Arc<dyn SelectionPolicy>,
    /// The manager's `max_active_states` cap, mirrored onto the parallel
    /// frontier (angr-9ke6b.48). The serial loop bounds `STASH_ACTIVE` in
    /// `RustExplorationManager::push_to_active_or_drop`, but an in-wave /
    /// in-session frontier never round-trips through that stash, so the same
    /// cap is applied by [`worker::absorb_continues`] against `pending`
    /// (queued + in-flight = the resident frontier). `None` = unbounded, which
    /// is what every Rust-side / test construction gets by default; the two
    /// production sites in `run_loop_wave.rs` / `run_loop_steady.rs` thread the manager's value in.
    max_active_states: Option<usize>,
    /// Dispatch budget for this wave — the parallel mirror of the run loop's
    /// `run(n)` step budget (angr-9ke6b.52). The wave coordinator's own budget
    /// check only runs between whole-wave barriers, and a wave runs its frontier
    /// to quiescence, so on a non-terminating frontier a single `run(n)` executed
    /// unboundedly many steps before returning. [`worker::worker_loop`] checks
    /// this at every task boundary and trips `cancel` once it is spent, which
    /// routes through the existing Bug M1 residual-drain path (the un-dispatched
    /// frontier comes back untagged and lands in `STASH_ACTIVE`, exactly as the
    /// single-threaded loop leaves it when it runs out of budget).
    ///
    /// Soft by up to `workers - 1`, same as `max_active_states`: `W` workers can
    /// each observe `limit - 1` simultaneously and dispatch, so the bound is
    /// `limit + W - 1` dispatches. `None` = unbounded (every Rust-side/test
    /// construction default); the wave loop threads its remaining budget in.
    max_dispatches: Option<u64>,
}

impl WorkTransport {
    /// Build a transport with an explicit selection policy. Callers that want a
    /// `Lifo` default (verbatim pre-seam dispatch) pass `Arc::new(Lifo)`;
    /// find-aware dispatch (angr-1ilq.9) supplies its own policy here.
    fn with_policy(policy: Arc<dyn SelectionPolicy>) -> Self {
        Self {
            injector: Injector::new(),
            pending: AtomicUsize::new(0),
            cancel: CancelToken::new(),
            idle_workers: AtomicUsize::new(0),
            counters: SchedulerCounters::default(),
            policy,
            max_active_states: None,
            max_dispatches: None,
        }
    }

    /// Whether this transport has spent its `max_dispatches` budget. Checked at
    /// every task boundary by [`worker::worker_loop`]; `None` (the default)
    /// always answers `false`. See the [`Self::max_dispatches`] field docs for
    /// the soft-bound semantics.
    fn dispatch_budget_exhausted(&self) -> bool {
        self.max_dispatches
            .is_some_and(|limit| self.counters.total_dispatches() >= limit)
    }

    /// Record one dispatch on `worker_id`, sampling the schedulable-frontier
    /// width from `pending` (angr-op0dn.13.9). `pending` counts queued +
    /// in-flight states and is only decremented AFTER the step, so the sample
    /// includes the state being dispatched — the same convention as the serial
    /// model's `active + stepped` width in `record_migration_sample`.
    fn record_dispatch(&self, worker_id: usize) {
        let width = self.pending.load(Ordering::Relaxed);
        self.counters.record_dispatch(worker_id, width);
    }

    /// Pull every payload still sitting on the injector (offloaded surplus that
    /// was never stolen) so a cancelled/finalized run loses nothing — the
    /// injector half of Bug M1 (the worker-local half is
    /// [`worker::drain_local_upstream`]). Balances `pending` and counts the
    /// drained states into `residual_drains`.
    ///
    /// Sound only once no worker can steal again: after the wave barrier
    /// (wave mode) or once every worker has acked `Paused` (session mode). A
    /// stale wake ping on a cancelled session re-acks `Paused` without touching
    /// the injector (the cancel check precedes dispatch), so no worker races it.
    fn drain_residual_payloads(&self) -> Vec<StateMigrationPayload> {
        let mut out = Vec::new();
        loop {
            match self.injector.steal() {
                Steal::Success(payload) => out.push(payload),
                Steal::Retry => continue,
                Steal::Empty => break,
            }
        }
        if !out.is_empty() {
            self.pending.fetch_sub(out.len(), Ordering::SeqCst);
            self.counters
                .residual_drains
                .fetch_add(out.len(), Ordering::SeqCst);
        }
        out
    }
}

/// All per-wave state a worker touches, fed to the persistent pool as an
/// `Arc<WaveJob>`. It bundles the coordination locals the former
/// `run_instrumented` held on its stack (the [`WorkTransport`] core plus
/// `results` / `summaries`) with the boxed [`ProcessFn`] that owns/`Arc`-shares
/// every input the per-state work needs. The coordinator recovers sole
/// ownership after the wave barrier via `Arc::into_inner` (sound because each
/// worker drops its clone before signaling `WaveDone`).
pub(crate) struct WaveJob {
    transport: WorkTransport,
    results: Mutex<Vec<StateMigrationPayload>>,
    /// Test-only accumulation of dead-path summaries. Production reads terminals
    /// via [`take_results`](Self::take_results) and counts summaries through
    /// `CoreCounters::record_summaries`; it never reads this vec, so gating it
    /// (and the `worker_loop` write that fills it) behind `cfg(test)` spares the
    /// production wave path a per-batch mutex lock. Only the `#[cfg(test)]`
    /// [`into_results`](Self::into_results) consumes it (angr-1yge9.8 item 7).
    #[cfg(test)]
    summaries: Mutex<Vec<TerminalSummary>>,
    /// Initial seed count (the `SchedulerStats::seeds` field).
    seeds: usize,
    /// The per-state processor, invoked once per dispatched state.
    process: Box<ProcessFn>,
}

// Compile-time proof the wave payload is `Send + Sync` — the property the
// persistent worker pool depends on to share an `Arc<WaveJob>` across threads.
// Mirrors the assertion on `StepContext` (step_core.rs). A non-`Send`/`Sync`
// field (e.g. a captured `Rc` in the process closure) fails the build, not a run.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<WaveJob>();
};

impl WaveJob {
    /// Build a wave from its seed payloads and a per-state processor. Seeds are
    /// pushed onto the injector and counted into `pending`; the `CancelToken` is
    /// fresh and private to this wave. Defaults the worker-local selection
    /// policy to [`Lifo`] — use [`WaveJob::new_with_policy`] to honor the
    /// run loop's configured policy (angr-x1fya).
    ///
    /// Test-only `Lifo`-default convenience: the only non-test caller is the
    /// `#[cfg(test)]` [`ParallelScheduler`] compat wrapper; the run loop calls
    /// [`WaveJob::new_with_policy`] to thread its configured policy.
    #[cfg(test)]
    pub(crate) fn new(initial: Vec<StateMigrationPayload>, process: Box<ProcessFn>) -> Self {
        Self::new_with_policy(initial, process, Arc::new(Lifo))
    }

    /// Build a wave with an explicit worker-local [`SelectionPolicy`]. The
    /// run loop threads its own `self.policy` here so a FIFO-configured
    /// (BFS) run does not silently drop to the scheduler's LIFO default
    /// under parallel dispatch (angr-x1fya).
    pub(crate) fn new_with_policy(
        initial: Vec<StateMigrationPayload>,
        process: Box<ProcessFn>,
        policy: Arc<dyn SelectionPolicy>,
    ) -> Self {
        let seeds = initial.len();
        let transport = WorkTransport::with_policy(policy);
        for payload in initial {
            transport.injector.push(payload);
        }
        transport.pending.store(seeds, Ordering::SeqCst);
        Self {
            transport,
            results: Mutex::new(Vec::new()),
            #[cfg(test)]
            summaries: Mutex::new(Vec::new()),
            seeds,
            process,
        }
    }

    /// Mirror the manager's `max_active_states` onto this wave's frontier cap
    /// (angr-9ke6b.48). Call before handing the job to the pool; see the
    /// [`WorkTransport::max_active_states`] field docs.
    pub(crate) fn set_max_active_states(&mut self, limit: Option<usize>) {
        self.transport.max_active_states = limit;
    }

    /// Bound this wave to `limit` dispatches (angr-9ke6b.52) — the run loop
    /// passes the `run(n)` budget it has left. Call before handing the job to
    /// the pool; see the [`WorkTransport::max_dispatches`] field docs for the
    /// soft bound and the residual-drain path a spent budget takes.
    pub(crate) fn set_max_dispatches(&mut self, limit: Option<u64>) {
        self.transport.max_dispatches = limit;
    }

    /// Snapshot the accumulated counters into a [`SchedulerStats`]. Call after
    /// the wave barrier, when this thread is the sole owner.
    pub(crate) fn stats(&self) -> SchedulerStats {
        snapshot_stats(self.seeds, &self.transport.counters)
    }

    /// Consume the recovered wave into `(materialized payloads, terminal
    /// summaries, stats)` — the shape the old `run_instrumented` returned.
    ///
    /// Test-only: consumed by the `#[cfg(test)]` [`ParallelScheduler`] wrapper;
    /// the persistent pool recovers results incrementally, not by consuming the
    /// whole `WaveJob`.
    #[cfg(test)]
    #[allow(
        clippy::expect_used,
        reason = "test-only reader; the `results`/`summaries` poison guards are unreachable for the same reason as everywhere else in this module — see the Panic policy header. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out here"
    )]
    pub(crate) fn into_results(
        self,
    ) -> (
        Vec<StateMigrationPayload>,
        Vec<TerminalSummary>,
        SchedulerStats,
    ) {
        let stats = self.stats();
        (
            self.results.into_inner().expect("results mutex poisoned"),
            self.summaries
                .into_inner()
                .expect("summaries mutex poisoned"),
            stats,
        )
    }

    /// Take the materialized terminal payloads out of the recovered wave,
    /// leaving the rest of the job (dropped by the caller). Used by the
    /// coordinator, which reads `summaries` separately (or ignores them).
    #[allow(
        clippy::expect_used,
        reason = "`results` poison guard: poison requires a thread to unwind out of a live guard, which `panic = \"abort\"` forecloses — see the module Panic policy header"
    )]
    pub(crate) fn take_results(&self) -> Vec<StateMigrationPayload> {
        std::mem::take(&mut *self.results.lock().expect("results mutex poisoned"))
    }

    /// Coordinator-side post-barrier helper: pull the surplus a cancelled wave
    /// left on the injector (offloaded but never stolen) so the un-explored
    /// frontier survives a `num_find` early-exit — the injector half of Bug M1
    /// (angr-op0dn.13.8). Untagged, exactly like the worker-local residuals the
    /// wave loop drains into `results`, so the coordinator routes both back to
    /// the active stash. Call only after the wave barrier, where no worker can
    /// steal again.
    pub(crate) fn drain_residual_payloads(&self) -> Vec<StateMigrationPayload> {
        self.transport.drain_residual_payloads()
    }
}

/// Downstream control message: coordinator -> persistent worker.
///
/// The unified channel enum for both scheduling modes (angr-nkoct steady-state
/// Phase B; absorbs the former `WaveMsg`). `Wave` drives the level-synchronous
/// wave loop verbatim; `Run` enters (or re-enters — it doubles as the idempotent
/// wake ping for a parked session worker) the steady-state session loop.
///
/// There is deliberately NO `Pause` variant: a busy session worker cannot poll
/// its control channel per task, so pause/finalize is signalled through the
/// session's [`CancelToken`] (checked at every task boundary) plus a `Run` wake
/// ping for parked workers. A separate message would race the flag path into
/// double `Paused` acks. Likewise no `Resume(payloads)` variant: resumed states
/// re-enter through [`RunSession::inject_resumed`] + a wake ping, so there is
/// exactly one dispatch path.
pub(crate) enum WorkerCtl {
    /// Run one level-synchronous wave to quiescence, then signal `WaveDone`.
    Wave(Arc<WaveJob>),
    /// Enter the steady-state session loop (or wake a parked session worker).
    Run(Arc<RunSession>),
    /// Terminate the worker thread.
    Shutdown,
}

/// Upstream report message: session worker -> coordinator, over the session's
/// mpsc channel. Deliberately UNTAGGED terminals: the coordinator already owns
/// the `kind_map`/`root_map` (its `ParallelShared`, populated by the process
/// closure under a mutex BEFORE the terminal is detached and sent, so the mpsc
/// send→recv edge establishes the happens-before) — carrying `kind`/`root` here
/// would duplicate that source of truth and couple this module to the run
/// loop's routing types.
pub(crate) enum WorkerUp {
    /// A materialized terminal (or, after cancel/finalize, a residual live
    /// frontier state) the coordinator must reattach and route.
    Terminal { payload: StateMigrationPayload },
    /// Ack of a cancel/finalize drain: this worker has detached its entire
    /// residual frontier upstream and parked. Sent once per worker per drain
    /// (dedupe by `worker_id` against late re-entries from stale wake pings).
    Paused { worker_id: usize },
    /// This worker observed global quiescence (no local work, empty injector,
    /// nothing outstanding) and parked. New work needs a `Run` wake ping.
    Quiesced { worker_id: usize },
}

/// The persistent per-run session shared (by `Arc`) across all steady-state
/// workers and the coordinator (angr-nkoct steady-state protocol). Where a
/// [`WaveJob`] lives for one wave and gives its results back through a barrier,
/// a `RunSession` lives for a whole `explore()` — across Python-callback
/// `run()` returns — and streams terminals up its mpsc channel as they
/// materialize. Workers never return states at a barrier; they park (empty-
/// handed) on quiescence and are woken by `WorkerCtl::Run` pings when the
/// coordinator injects more work.
pub(crate) struct RunSession {
    transport: WorkTransport,
    /// Total payloads injected as fresh work (initial frontier drains), the
    /// `SchedulerStats::seeds` analogue. Resume re-injections are counted in
    /// `counters.resume_reinjects` instead.
    seeds: AtomicUsize,
    /// Upstream channel every worker reports on (`std::sync::mpsc::Sender` is
    /// `Send + Sync` since the channel rewrite, so sharing one through the
    /// session `Arc` is sound — proven by the compile-time assertion below).
    up_tx: Sender<WorkerUp>,
    /// The per-state processor, invoked once per dispatched state.
    process: Box<ProcessFn>,
}

impl RunSession {
    /// Build a session around a per-state processor. Returns the session and
    /// the receiving half of its upstream channel (the coordinator keeps it;
    /// dropping it makes late worker sends fail silently — workers ignore send
    /// errors for exactly that shutdown race).
    ///
    /// Test-only `Lifo`-default convenience; the run loop constructs sessions
    /// via [`RunSession::new_with_policy`] to honor its configured policy.
    #[cfg(test)]
    pub(crate) fn new(process: Box<ProcessFn>) -> (Arc<Self>, Receiver<WorkerUp>) {
        Self::new_with_policy(process, Arc::new(Lifo), None)
    }

    /// Build a session with an explicit worker-local [`SelectionPolicy`], so
    /// the steady-state `explore()` path honors the run loop's configured
    /// policy instead of the scheduler's LIFO default (angr-x1fya).
    ///
    /// `max_active_states` mirrors the manager's frontier cap onto the session
    /// (angr-9ke6b.48); it is a constructor parameter rather than a setter
    /// because the session is `Arc`-wrapped on the way out. See the
    /// [`WorkTransport::max_active_states`] field docs.
    pub(crate) fn new_with_policy(
        process: Box<ProcessFn>,
        policy: Arc<dyn SelectionPolicy>,
        max_active_states: Option<usize>,
    ) -> (Arc<Self>, Receiver<WorkerUp>) {
        let (up_tx, up_rx) = mpsc::channel();
        let mut transport = WorkTransport::with_policy(policy);
        transport.max_active_states = max_active_states;
        (
            Arc::new(Self {
                transport,
                seeds: AtomicUsize::new(0),
                up_tx,
                process,
            }),
            up_rx,
        )
    }

    /// Inject fresh frontier work (counted as seeds). `pending` is raised
    /// BEFORE the payloads land on the injector so a spinning worker can never
    /// observe queued work with a zero pending count (the quiescence detector's
    /// invariant).
    pub(crate) fn inject_seeds(&self, payloads: Vec<StateMigrationPayload>) {
        self.seeds.fetch_add(payloads.len(), Ordering::SeqCst);
        self.inject(payloads);
    }

    /// Inject states resumed after a Python callback round-trip (counted into
    /// `resume_reinjects`, the steady-state protocol's accounting of work that
    /// re-entered through the coordinator instead of staying worker-local).
    pub(crate) fn inject_resumed(&self, payloads: Vec<StateMigrationPayload>) {
        self.transport
            .counters
            .resume_reinjects
            .fetch_add(payloads.len(), Ordering::SeqCst);
        self.inject(payloads);
    }

    fn inject(&self, payloads: Vec<StateMigrationPayload>) {
        self.transport
            .pending
            .fetch_add(payloads.len(), Ordering::SeqCst);
        for payload in payloads {
            self.transport.injector.push(payload);
        }
    }

    /// Request cancel/finalize: every worker stops at its next task boundary,
    /// drains its residual frontier upstream as `Terminal`s, acks `Paused`, and
    /// parks. Parked workers need a `Run` wake ping to observe this.
    pub(crate) fn cancel(&self) {
        self.transport.cancel.cancel();
    }

    /// Test-only cancel-state probe; the coordinator drives cancel via
    /// [`RunSession::cancel`] and observes quiescence through `pending`.
    #[cfg(test)]
    pub(crate) fn is_cancelled(&self) -> bool {
        self.transport.cancel.is_cancelled()
    }

    /// Queued + in-flight state count (0 == quiescent, if also no live locals —
    /// workers park only when BOTH hold, so "all workers parked && pending == 0"
    /// is the coordinator's termination condition).
    pub(crate) fn pending(&self) -> usize {
        self.transport.pending.load(Ordering::SeqCst)
    }

    /// Coordinator-side finalize helper: after every worker has acked `Paused`,
    /// pull the injector surplus back ([`WorkTransport::drain_residual_payloads`]).
    pub(crate) fn drain_residual_payloads(&self) -> Vec<StateMigrationPayload> {
        self.transport.drain_residual_payloads()
    }

    /// Bump the bounce-roundtrip counter — called by the coordinator as it
    /// routes each `MatKind::Bounce` terminal it received from this session.
    pub(crate) fn count_bounce_roundtrip(&self) {
        self.transport
            .counters
            .bounce_roundtrips
            .fetch_add(1, Ordering::SeqCst);
    }

    /// Snapshot the session's accumulated counters. Safe mid-session (atomics),
    /// so the coordinator can fold deltas at every event return.
    pub(crate) fn stats(&self) -> SchedulerStats {
        snapshot_stats(self.seeds.load(Ordering::SeqCst), &self.transport.counters)
    }
}

// Compile-time proof the steady-state session is `Send + Sync` and both channel
// payloads are `Send` — the properties the persistent pool depends on to share
// an `Arc<RunSession>` across workers while the coordinator holds a clone. A
// non-thread-safe field fails the build, not a run; do NOT paper over it with
// `unsafe`.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    fn assert_send<T: Send>() {}
    assert_send_sync::<RunSession>();
    assert_send::<WorkerUp>();
    assert_send::<WorkerCtl>();
};

/// A worker's per-wave completion signal — the wave-barrier token the
/// coordinator counts `num_workers` of. The coordinator only counts these
/// (it never inspects which worker sent one), so the token carries no payload.
struct WaveDone;

/// `num_workers` long-lived OS threads, each owning a private Z3 context for the
/// pool's whole life, fed one [`WaveJob`] at a time over per-worker channels.
///
/// This replaces the per-wave `std::thread::scope` + fresh `z3::Context::new()`
/// that taxed every wave with a thread-spawn + Z3-context-creation cost (which
/// erased any parallel speedup). Each worker installs its context ONCE at spawn;
/// because the worker is pinned to one OS thread for the pool's life, that
/// context is stable across waves, so `reattach`'s `target_ctx ==
/// Context::thread_local()` pointer-equality guard keeps holding every wave.
pub(crate) struct PersistentPool {
    job_txs: Vec<Sender<WorkerCtl>>,
    /// Wrapped in a `Mutex` purely so `PersistentPool: Sync` holds — required
    /// because the coordinator runs `run_wave` inside `py.detach`, whose closure
    /// captures `&PersistentPool` and must be `Send` (`std::sync::mpsc::Receiver`
    /// is itself `!Sync`). Only the single coordinator thread ever receives, so
    /// the lock is uncontended.
    done_rx: Mutex<Receiver<WaveDone>>,
    handles: Vec<JoinHandle<()>>,
    num_workers: usize,
}

/// Stack size for each persistent worker thread (angr-h92bx).
///
/// The single-threaded run loop executes on CPython's *main* thread, whose
/// stack is 8 MiB by default. A `std::thread` gets 2 MiB, and the stepping path
/// recurses over AST shape (`claripy_bridge::import` self-recurses per operand,
/// Z3 emission walks the `RustBV` tree). CGC benches with deep symbolic-stdin
/// ASTs (CADET_00001) therefore ran fine at `RUST_PARALLEL_WORKERS=1` but blew
/// the guard page on a worker at `W>=2` — `angr-worker-N ... segfault ... error 6`
/// with the fault address one word below `sp`. Reserve 16 MiB (virtual; pages
/// are committed lazily, so idle workers cost no RSS) to clear the main thread
/// by 2x rather than fall short of it.
const WORKER_STACK_SIZE: usize = 16 * 1024 * 1024;

impl PersistentPool {
    /// Spawn `num_workers` (clamped to >= 1) persistent worker threads. Each
    /// creates its Z3 context once and then blocks waiting for the first wave.
    #[allow(
        clippy::expect_used,
        reason = "`Builder::spawn` is the one genuinely-fallible panic in this module (EAGAIN under thread exhaustion). `new` returns `Self`, so propagating means a `Result` through the whole parallel driver, and degrading to fewer workers loses states outright when none come up — see the module Panic policy header for the full argument"
    )]
    pub(crate) fn new(num_workers: usize) -> Self {
        let num_workers = num_workers.max(1);
        let (done_tx, done_rx) = mpsc::channel::<WaveDone>();
        let mut job_txs = Vec::with_capacity(num_workers);
        let mut handles = Vec::with_capacity(num_workers);
        for worker_id in 0..num_workers {
            let (job_tx, job_rx) = mpsc::channel::<WorkerCtl>();
            let done_tx = done_tx.clone();
            job_txs.push(job_tx);
            handles.push(
                std::thread::Builder::new()
                    .name(format!("angr-worker-{worker_id}"))
                    .stack_size(WORKER_STACK_SIZE)
                    .spawn(move || worker_thread(worker_id, job_rx, done_tx))
                    .expect("failed to spawn persistent worker thread"),
            );
        }
        Self {
            job_txs,
            done_rx: Mutex::new(done_rx),
            handles,
            num_workers,
        }
    }

    pub(crate) fn num_workers(&self) -> usize {
        self.num_workers
    }

    /// Run one wave to quiescence (or cancellation) across all workers and hand
    /// back the recovered [`WaveJob`] + its [`SchedulerStats`].
    ///
    /// Broadcasts an `Arc<WaveJob>` clone to every worker, blocks recv-ing
    /// exactly `num_workers` [`WaveDone`] (the barrier), then `Arc::into_inner`
    /// recovers sole ownership. That is sound *because* each worker drops its
    /// `Arc` clone BEFORE signaling `WaveDone` (see [`worker_thread`]) and the
    /// mpsc send→recv edge establishes the happens-before, so once all N dones
    /// are received the strong count is exactly 1. The `Option` is unwrapped
    /// with an explicit panic (not a silent `.expect`) because a `None` here is
    /// a hard invariant violation, not an expected error.
    #[allow(
        clippy::expect_used,
        reason = "`done_rx` poison guard plus the two live-pool transport guards: a worker leaves `worker_thread` only on clean teardown or by aborting the process, so neither channel can disconnect mid-wave — see the module Panic policy header"
    )]
    pub(crate) fn run_wave(&self, job: WaveJob) -> (WaveJob, SchedulerStats) {
        let job = Arc::new(job);
        for tx in &self.job_txs {
            tx.send(WorkerCtl::Wave(Arc::clone(&job)))
                .expect("persistent worker died before wave dispatch");
        }
        {
            let done_rx = self.done_rx.lock().expect("done_rx mutex poisoned");
            for _ in 0..self.num_workers {
                done_rx
                    .recv()
                    .expect("persistent worker died before signaling WaveDone");
            }
        }
        let job = Arc::into_inner(job).unwrap_or_else(|| {
            panic!(
                "BUG: a worker still holds the WaveJob Arc after the wave barrier \
                 (the drop(job)-before-WaveDone invariant was violated)"
            )
        });
        let stats = job.stats();
        (job, stats)
    }

    /// Broadcast `WorkerCtl::Run` for a steady-state session to every worker
    /// (session start). Also usable as a broadcast wake; prefer
    /// [`wake_worker`](Self::wake_worker) for targeted pings so parked workers
    /// don't accumulate stale `Arc<RunSession>` clones on their channels.
    #[allow(
        clippy::expect_used,
        reason = "live-pool transport guard: every worker is alive for the whole session (it can only leave `worker_thread` on clean teardown or by aborting), so the job channel cannot disconnect here — see the module Panic policy header"
    )]
    pub(crate) fn start_session(&self, session: &Arc<RunSession>) {
        for tx in &self.job_txs {
            tx.send(WorkerCtl::Run(Arc::clone(session)))
                .expect("persistent worker died before session dispatch");
        }
    }

    /// Wake one (presumed parked) session worker with a `Run` ping. Idempotent:
    /// a busy worker consumes the ping when it next parks, re-enters the
    /// session loop, finds nothing to do, and parks again (one cheap loop).
    pub(crate) fn wake_worker(&self, worker_id: usize, session: &Arc<RunSession>) {
        if let Some(tx) = self.job_txs.get(worker_id) {
            let _ = tx.send(WorkerCtl::Run(Arc::clone(session)));
        }
    }
}

impl Drop for PersistentPool {
    /// Broadcast `Shutdown` to every worker, then join. Each worker's Z3 context
    /// drops on its own thread — never freed cross-thread.
    fn drop(&mut self) {
        for tx in &self.job_txs {
            let _ = tx.send(WorkerCtl::Shutdown);
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

/// A persistent worker: install a private Z3 context ONCE, then loop serving
/// control messages — one wave per `WorkerCtl::Wave` (dropping its
/// `Arc<WaveJob>` clone BEFORE sending `WaveDone` so the coordinator's
/// `Arc::into_inner` recovers sole ownership), or one steady-state session
/// entry per `WorkerCtl::Run` (parking back here on quiescence/drain — the
/// blocking `recv` IS the park). On `Shutdown` (or a closed channel at pool
/// teardown) it falls out of the loop, dropping its Z3 context on this same
/// thread (no leak, no cross-thread free).
fn worker_thread(worker_id: usize, job_rx: Receiver<WorkerCtl>, done_tx: Sender<WaveDone>) {
    // Owned for the worker's whole life, installed as the thread-local. Every
    // reattach mints ASTs into a context this thread alone touches; created and
    // destroyed on the same thread. The `local` live queue is created below and
    // threaded into `worker_loop` by `&mut`, so the irreducibly `!Send`
    // `RustSimState`s in it can never escape this thread.
    let z3ctx = Context::new(&Config::new());
    Context::set_thread_local(&z3ctx);
    // Warm per-worker block cache (angr-vh834 Work Item 3), created ONCE and
    // reused across every dispatch AND every wave for this worker. `Arc<IRSB>`
    // values are AST-free and never leave this thread, so this needs no Send/Sync
    // bound and is context-safe: a warm cache returns identical (pure) lifts, so
    // the found set and content fingerprints are unchanged.
    let mut block_cache: LruCache<u64, Arc<IRSB>> = LruCache::new(BLOCK_CACHE_CAPACITY_NZ);
    // Persistent worker-local live frontier (angr-nkoct increment 1). Owned for
    // the worker's WHOLE life alongside the Z3 context and warm block cache, NOT
    // recreated per wave. The `!Send` `RustSimState`s it holds stay stack-local to
    // this OS thread and never escape. Any states a wave did not drain to
    // quiescence are RETAINED here for the next wave and dispatched WITHOUT a
    // detach/reattach round trip — the mechanism that stops the coordinator from
    // re-migrating an already-resident frontier. Today every wave runs its subtree
    // to quiescence, so `local` is empty at each barrier and this is behaviour-
    // neutral for pure-Rust workloads; it sets up the win for the bounce case.
    let mut local: VecDeque<RustSimState> = VecDeque::new();
    loop {
        match job_rx.recv() {
            Ok(WorkerCtl::Wave(job)) => {
                worker_loop(worker_id, &job, &z3ctx, &mut block_cache, &mut local);
                // Release this worker's Arc clone BEFORE signaling done — the
                // invariant that makes the coordinator's `Arc::into_inner` safe.
                drop(job);
                if done_tx.send(WaveDone).is_err() {
                    // Coordinator is gone; nothing to synchronize with. Bail so
                    // the context drops on this thread.
                    break;
                }
            }
            Ok(WorkerCtl::Run(session)) => {
                // Session entry or wake ping. The loop returns with `local`
                // EMPTY in every case (quiesced, drained, or a stale ping on a
                // finalized session), so no `!Send` state is ever parked inside
                // this thread across the coordinator boundary.
                worker_session_loop(&session, worker_id, &z3ctx, &mut block_cache, &mut local);
                // Always-on, NOT `debug_assert!` (angr-9ke6b.68): the release
                // profile leaves `debug-assertions = false`, so a `debug_assert!`
                // here would give the shipped `.so` zero protection for the one
                // invariant that keeps `!Send` states from crossing the
                // coordinator boundary — a violation would silently re-park a
                // worker holding stale states and mis-route them with no
                // diagnostic trail. Under `panic = "abort"` this fails the way
                // the "Panic policy" header describes every other invariant guard
                // in this module: a loud SIGABRT at the violation site. Cost is
                // one `VecDeque::is_empty` per session-loop return, which is off
                // the per-step hot path.
                assert!(local.is_empty(), "session worker parked with live states");
            }
            Ok(WorkerCtl::Shutdown) | Err(_) => break,
        }
    }
    // Drop this worker's bridge thread-local AST caches WHILE `z3ctx` is still
    // the live thread-local Z3 context (angr-bjk8 / angr-1yge9.9). The interpreter
    // paths workers run (`claripy_to_rustbv` in interpreter/{mod,expressions}.rs)
    // populate `AST_CACHE` with `RustBV` values whose z3 ASTs are bound to
    // `z3ctx`. Rust runs `thread_local!` destructors at thread teardown — AFTER
    // this function returns and `z3ctx` (a local) has already dropped — so
    // without this call those `RustBV`s would `dec_ref` against a freed context
    // (UAF). `clear_worker_local_caches` runs here, with `z3ctx` alive, so every
    // cached AST drops against a valid context; it touches only this thread's
    // caches, never the cross-thread global registry (angr-1ilq.2).
    crate::claripy_bridge::clear_worker_local_caches();
    // Shutdown / channel closed / coordinator gone: `z3ctx` drops here.
}

/// A thin compatibility wrapper over a one-shot [`PersistentPool`], preserving
/// the old `ParallelScheduler` surface for the byte-stable scheduler unit tests
/// (`#[cfg(test)]` below). The run loop drives a long-lived `PersistentPool`
/// directly; this spins one up per call, runs a single [`WaveJob`] built from
/// the caller's synthetic `process`, and tears the pool down on return.
///
/// Test-only: this whole surface exists solely for the `#[cfg(test)]` scheduler
/// unit suite below; no production path constructs it.
#[cfg(test)]
pub(super) struct ParallelScheduler {
    num_workers: usize,
}

#[cfg(test)]
impl ParallelScheduler {
    /// Build a scheduler with `num_workers` worker threads (clamped to >= 1).
    pub(super) fn new(num_workers: usize) -> Self {
        Self {
            num_workers: num_workers.max(1),
        }
    }

    /// Run the pool to quiescence (or cancellation) and return the collected
    /// materialized terminal payloads. Convenience wrapper over
    /// [`run_instrumented`](Self::run_instrumented) that drops the summaries and
    /// stats.
    ///
    /// `process` must be `'static` (unlike the former `thread::scope` API, which
    /// let it borrow the caller's stack): the persistent pool hands each worker
    /// an `Arc<WaveJob>` that outlives this frame, so the closure must own its
    /// captures.
    pub(super) fn run<F>(
        &self,
        initial: Vec<StateMigrationPayload>,
        process: F,
    ) -> Vec<StateMigrationPayload>
    where
        F: Fn(RustSimState, &CancelToken, &mut LruCache<u64, Arc<IRSB>>) -> TaskOutcome
            + Send
            + Sync
            + 'static,
    {
        self.run_instrumented(initial, process).0
    }

    /// Run the pool and return `(materialized payloads, terminal summaries,
    /// stats)`.
    ///
    /// `process` runs on a worker thread with that worker's own Z3 context
    /// installed as the thread-local; it receives a state already reattached
    /// into that context (or, on the local fast path, one created there). It
    /// must be `Send + Sync + 'static` (it is stored in the `Arc<WaveJob>` every
    /// worker shares).
    pub(super) fn run_instrumented<F>(
        &self,
        initial: Vec<StateMigrationPayload>,
        process: F,
    ) -> (
        Vec<StateMigrationPayload>,
        Vec<TerminalSummary>,
        SchedulerStats,
    )
    where
        F: Fn(RustSimState, &CancelToken, &mut LruCache<u64, Arc<IRSB>>) -> TaskOutcome
            + Send
            + Sync
            + 'static,
    {
        let pool = PersistentPool::new(self.num_workers);
        let (job, _stats) = pool.run_wave(WaveJob::new(initial, Box::new(process)));
        job.into_results()
        // `pool` drops here → Shutdown broadcast + join of the one-shot workers.
    }
}

#[path = "scheduler_worker.rs"]
mod worker;
use worker::{worker_loop, worker_session_loop};

#[cfg(test)]
#[path = "scheduler_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable"
)]
mod tests;
