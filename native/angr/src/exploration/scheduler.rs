//! Work-stealing scheduler machinery for parallel exploration (angr-1ilq.3,
//! correctness-first isolated increment; anti-migration redesign angr-729vn).
//!
//! # What this is, and what it is NOT (yet)
//!
//! This module lands the *threading + migration + cancellation* machinery for
//! a multi-worker exploration pool, proven in isolation with real
//! `std::thread::scope` workers. It is **not yet wired into the run loop**.
//!
//! The reason for the split is a hard constraint discovered while starting
//! 1ilq.3: stepping is pervasively GIL-coupled. [`RustExplorationManager::
//! step_state_with_skip`](super::stepping) takes a live `Python<'_>` token and
//! yields back to Python at seven callback points (find/avoid predicates,
//! SimProcedures, syscalls, symbolic branches, Python-VEX fallback, errors). A
//! worker thread therefore cannot run a real engine step without holding the
//! GIL. Releasing the GIL only around the Rust-pure inner work
//! (`py.allow_threads`) and re-acquiring it for callbacks is a delicate change
//! to the engine's hottest function and is deferred to a follow-up increment
//! (the callback-dispatch half is 1ilq.4). See the `angr-1ilq.3` bead.
//!
//! So this increment proves the part that has nothing to do with the GIL and
//! everything to do with thread-safety: that worker-local live states can be
//! explored in a private Z3 context, with cross-worker transport happening
//! *only* on an actual imbalance steal — all under genuine concurrency, with
//! zero `unsafe`.
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

use crate::exploration::selection_policy::SelectionPolicy;
// `Lifo` is the pre-seam default policy, now only used by the `#[cfg(test)]`
// `Lifo`-default convenience constructors (production threads an explicit policy
// via `*_with_policy`).
#[cfg(test)]
use crate::exploration::selection_policy::Lifo;
use crate::interpreter::BLOCK_CACHE_CAPACITY;
use crate::state::{RustSimState, StateMigrationPayload};
use crate::vex::IRSB;
use crossbeam_deque::{Injector, Steal};
use lru::LruCache;
use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    /// Request that all workers stop at their next task boundary.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
}

/// How a non-materialized terminal ended, captured in a [`TerminalSummary`].
///
/// Found/matched terminals are *not* represented here — they are materialized
/// (fully serialized) so the caller can recover the satisfying state. These are
/// the dispositions whose full symbolic state the parallel path discards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminalDisposition {
    Deadended,
    Errored,
    Avoided,
    Pruned,
}

/// A lightweight record of a terminal state that did **not** need to cross the
/// worker boundary as a full state. Built in the worker's own context from
/// cheap scalar fields, so it pays no serde / Z3-AST-rebuild tax.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalSummary {
    pub state_id: u64,
    pub pc: u64,
    pub disposition: TerminalDisposition,
}

impl TerminalSummary {
    /// Summarize a terminal state without serializing it. The `state` is read
    /// (cheap scalar fields only) and then dropped in its home context.
    pub fn of(state: &RustSimState, disposition: TerminalDisposition) -> Self {
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
pub struct TaskOutcome {
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
    pub fn continuing(continue_states: Vec<RustSimState>) -> Self {
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
    pub fn terminal(terminal_states: Vec<RustSimState>) -> Self {
        Self {
            continue_states: Vec::new(),
            terminal_states,
            terminal_summaries: Vec::new(),
            request_cancel: false,
        }
    }

    /// All successors are dead paths recorded as summaries; no further work, no
    /// serde.
    pub fn summarized(terminal_summaries: Vec<TerminalSummary>) -> Self {
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
/// (run_loop.rs) can own one directly (angr-vh834 steady-state redesign, Phase
/// 1). The scheduler still constructs and reads it here.
#[derive(Default)]
pub(crate) struct SchedulerCounters {
    local_dispatches: AtomicUsize,
    injector_dispatches: AtomicUsize,
    surplus_offloaded: AtomicUsize,
    materialized_terminals: AtomicUsize,
    summarized_terminals: AtomicUsize,
    /// Payloads pulled from the injector and `reattach`ed into a worker's own Z3
    /// context (the injector-steal path). Observability only — equals
    /// `injector_dispatches` today; the two diverge once the steady-state
    /// coordinator reattaches on paths other than an injector steal (Phase 2+).
    reattaches: AtomicUsize,
    /// Bounce states that made a full worker->coordinator->worker round trip.
    /// Wired by the steady-state coordinator (run_loop.rs); 0 in wave mode.
    pub(crate) bounce_roundtrips: AtomicUsize,
    /// States re-injected after a Python resume callback
    /// ([`RunSession::inject_resumed`]); 0 in wave mode.
    resume_reinjects: AtomicUsize,
    /// Still-live frontier states a session worker detached and streamed back to
    /// the coordinator on cancel/finalize ([`worker::drain_local_upstream`]) instead of
    /// dropping them (the wave loop's Bug M1). Distinct from
    /// `materialized_terminals` so residual drains never pollute the honest
    /// steal fraction. 0 in wave mode.
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
pub struct SchedulerStats {
    /// Initial payloads handed to [`ParallelScheduler::run_instrumented`].
    pub seeds: usize,
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
    /// Payloads reattached into a worker's own Z3 context (injector-steal path).
    /// Observability only; see [`SchedulerCounters::reattaches`].
    pub reattaches: usize,
    /// Bounce states that round-tripped worker->coordinator->worker (0 in wave
    /// mode; counted by the steady-state coordinator).
    pub bounce_roundtrips: usize,
    /// States re-injected after a Python resume (0 in wave mode; counted by
    /// [`RunSession::inject_resumed`]).
    pub resume_reinjects: usize,
    /// Still-live frontier states drained back to the coordinator on session
    /// cancel/finalize (0 in wave mode, which drops them — Bug M1).
    pub residual_drains: usize,
    /// Post-find speculative steps: work committed on a worker after another
    /// origin already requested cancel (angr-1ilq.8). See
    /// [`SchedulerCounters::post_cancel_steps`].
    pub post_cancel_steps: usize,
}

impl SchedulerStats {
    /// Total states dispatched (the denominator of the steal fraction). Every
    /// processed state is dispatched exactly once, from the local queue or the
    /// injector. Matches the `parallel_tasks` denominator of the run loop's
    /// `f_model` (helpers.rs), so the two steal fractions are comparable.
    pub fn dispatches(&self) -> usize {
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
    pub fn honest_steal_fraction(&self) -> f64 {
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
        local_dispatches: counters.local_dispatches.load(Ordering::SeqCst),
        injector_dispatches: counters.injector_dispatches.load(Ordering::SeqCst),
        surplus_offloaded: counters.surplus_offloaded.load(Ordering::SeqCst),
        materialized_terminals: counters.materialized_terminals.load(Ordering::SeqCst),
        summarized_terminals: counters.summarized_terminals.load(Ordering::SeqCst),
        reattaches: counters.reattaches.load(Ordering::SeqCst),
        bounce_roundtrips: counters.bounce_roundtrips.load(Ordering::SeqCst),
        resume_reinjects: counters.resume_reinjects.load(Ordering::SeqCst),
        residual_drains: counters.residual_drains.load(Ordering::SeqCst),
        post_cancel_steps: counters.post_cancel_steps.load(Ordering::SeqCst),
    }
}

/// A boxed, thread-safe per-state processor. Production (`run_loop.rs`) wraps
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
        }
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
            summaries: Mutex::new(Vec::new()),
            seeds,
            process,
        }
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
    pub(crate) fn take_results(&mut self) -> Vec<StateMigrationPayload> {
        std::mem::take(&mut *self.results.lock().expect("results mutex poisoned"))
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
        Self::new_with_policy(process, Arc::new(Lifo))
    }

    /// Build a session with an explicit worker-local [`SelectionPolicy`], so
    /// the steady-state `explore()` path honors the run loop's configured
    /// policy instead of the scheduler's LIFO default (angr-x1fya).
    pub(crate) fn new_with_policy(
        process: Box<ProcessFn>,
        policy: Arc<dyn SelectionPolicy>,
    ) -> (Arc<Self>, Receiver<WorkerUp>) {
        let (up_tx, up_rx) = mpsc::channel();
        (
            Arc::new(Self {
                transport: WorkTransport::with_policy(policy),
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
    /// pull any surplus payloads still sitting on the injector (offloaded but
    /// never stolen) so a finalized session loses nothing — the injector half
    /// of wave-mode Bug M1 (the worker-local half is [`worker::drain_local_upstream`]).
    /// Sound only once all workers are parked: a stale wake ping on a cancelled
    /// session re-acks `Paused` without touching the injector (the cancel check
    /// precedes dispatch), so no worker races this steal.
    pub(crate) fn drain_residual_payloads(&self) -> Vec<StateMigrationPayload> {
        let mut out = Vec::new();
        loop {
            match self.transport.injector.steal() {
                Steal::Success(payload) => out.push(payload),
                Steal::Retry => continue,
                Steal::Empty => break,
            }
        }
        if !out.is_empty() {
            self.transport
                .pending
                .fetch_sub(out.len(), Ordering::SeqCst);
            self.transport
                .counters
                .residual_drains
                .fetch_add(out.len(), Ordering::SeqCst);
        }
        out
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

impl PersistentPool {
    /// Spawn `num_workers` (clamped to >= 1) persistent worker threads. Each
    /// creates its Z3 context once and then blocks waiting for the first wave.
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
    let mut block_cache: LruCache<u64, Arc<IRSB>> = LruCache::new(
        NonZeroUsize::new(BLOCK_CACHE_CAPACITY).expect("BLOCK_CACHE_CAPACITY is non-zero"),
    );
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
                worker_loop(&job, &z3ctx, &mut block_cache, &mut local);
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
                debug_assert!(local.is_empty(), "session worker parked with live states");
            }
            Ok(WorkerCtl::Shutdown) | Err(_) => break,
        }
    }
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
pub struct ParallelScheduler {
    num_workers: usize,
}

#[cfg(test)]
impl ParallelScheduler {
    /// Build a scheduler with `num_workers` worker threads (clamped to >= 1).
    pub fn new(num_workers: usize) -> Self {
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
    pub fn run<F>(
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
    pub fn run_instrumented<F>(
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
mod tests;
