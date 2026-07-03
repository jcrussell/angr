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
//!      [`offload_surplus`]), or
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
/// A). See [`offload_surplus`].
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
    /// the coordinator on cancel/finalize ([`drain_local_upstream`]) instead of
    /// dropping them (the wave loop's Bug M1). Distinct from
    /// `materialized_terminals` so residual drains never pollute the honest
    /// steal fraction. 0 in wave mode.
    residual_drains: AtomicUsize,
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
/// steal / offload machinery ([`dispatch_next`], [`steal_from_injector`],
/// [`offload_surplus`]) is written once against this struct (DRY) and the two
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
    /// [`offload_surplus`] Trigger A).
    idle_workers: AtomicUsize,
    counters: SchedulerCounters,
}

impl WorkTransport {
    fn new() -> Self {
        Self {
            injector: Injector::new(),
            pending: AtomicUsize::new(0),
            cancel: CancelToken::new(),
            idle_workers: AtomicUsize::new(0),
            counters: SchedulerCounters::default(),
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
    /// fresh and private to this wave.
    pub(crate) fn new(initial: Vec<StateMigrationPayload>, process: Box<ProcessFn>) -> Self {
        let seeds = initial.len();
        let transport = WorkTransport::new();
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
    pub(crate) fn new(process: Box<ProcessFn>) -> (Arc<Self>, Receiver<WorkerUp>) {
        let (up_tx, up_rx) = mpsc::channel();
        (
            Arc::new(Self {
                transport: WorkTransport::new(),
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
    /// of wave-mode Bug M1 (the worker-local half is [`drain_local_upstream`]).
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
/// coordinator counts `num_workers` of.
struct WaveDone {
    #[allow(dead_code)] // carried for debuggability / future targeted diagnostics
    worker_id: usize,
}

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
                if done_tx.send(WaveDone { worker_id }).is_err() {
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
pub struct ParallelScheduler {
    num_workers: usize,
}

impl ParallelScheduler {
    /// Build a scheduler with `num_workers` worker threads (clamped to >= 1).
    pub fn new(num_workers: usize) -> Self {
        Self {
            num_workers: num_workers.max(1),
        }
    }

    pub fn num_workers(&self) -> usize {
        self.num_workers
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

/// The per-worker loop: dispatch a live local state (or steal+reattach one),
/// process it, push live successors locally, shed surplus on imbalance, and
/// collect terminals. Reads all per-wave state from `job`; `ctx` is the worker's
/// persistent Z3 context, `block_cache` its warm per-worker IRSB cache, and
/// `local` its persistent live frontier — all owned by [`worker_thread`] and
/// reused across waves (angr-nkoct increment 1).
///
/// `local` may carry states RETAINED from a previous wave (none today: waves run
/// to quiescence, so it is empty at entry). Any such carry-over is counted into
/// `job.pending` up front so quiescence accounting stays balanced — each retained
/// state's terminal `pending.fetch_sub(1)` is matched by this add. With an empty
/// `local` (the invariant today) this is `fetch_add(0)`, a no-op, so the
/// byte-stable scheduler unit tests and single-threaded parity are unaffected.
fn worker_loop(
    job: &WaveJob,
    ctx: &Context,
    block_cache: &mut LruCache<u64, Arc<IRSB>>,
    local: &mut VecDeque<RustSimState>,
) {
    let t = &job.transport;

    // Account for any frontier retained across the wave barrier so `pending`
    // (the quiescence detector) covers every state that will be dispatched and
    // `fetch_sub`'d below. No-op when `local` is empty (today's invariant).
    if !local.is_empty() {
        t.pending.fetch_add(local.len(), Ordering::SeqCst);
    }

    loop {
        if t.cancel.is_cancelled() {
            // Bug M1 (known limitation of WAVE mode; the steady-state session
            // loop drains instead): on cancel (e.g. the run loop hit
            // `num_find`) the worker stops HERE, at a task boundary, dropping
            // whatever live states remain on its `local` queue (and any surplus
            // still sitting on the injector). Those un-dispatched frontier states
            // are NOT materialized back across the join, so the run loop's active
            // stash loses them — `run_loop_parallel`'s post-`num_find`
            // `active_count()` is smaller than the single-threaded loop's and the
            // un-explored frontier is not resumable. The FOUND set is unaffected
            // (every found terminal was materialized before cancel propagated).
            // Draining + materializing the remainder here would pay serde for
            // states we are about to discard, so it is deliberately not done; see
            // the `run_loop_parallel` doc comment. NOTE the states stay ON
            // `local` (not cleared): the persistent frontier (angr-nkoct
            // increment 1) retains them for the next wave, if one runs.
            return;
        }

        let state = match dispatch_next(t, local, ctx) {
            Some(state) => state,
            // No task available and nothing outstanding anywhere (or
            // cancelled): no future task can ever appear this wave.
            None => return,
        };

        let outcome = (job.process)(state, &t.cancel, block_cache);

        // Continue-states stay LIVE and LOCAL — no serde on the fast path.
        absorb_continues(t, local, outcome.continue_states);

        // Materialized terminals must cross the join, so they are detached here
        // (correct context). This is part of the honest steal fraction.
        if !outcome.terminal_states.is_empty() {
            let mut guard = job.results.lock().expect("results mutex poisoned");
            for terminal in outcome.terminal_states {
                guard.push(terminal.detach_for_migration());
                t.counters
                    .materialized_terminals
                    .fetch_add(1, Ordering::SeqCst);
            }
        }

        // Summaries pay no serde — record and drop the full states in-context.
        if !outcome.terminal_summaries.is_empty() {
            let n = outcome.terminal_summaries.len();
            let mut guard = job.summaries.lock().expect("summaries mutex poisoned");
            guard.extend(outcome.terminal_summaries);
            t.counters
                .summarized_terminals
                .fetch_add(n, Ordering::SeqCst);
        }

        // Shed surplus to the injector if a sibling is starving or we are over
        // the high-water mark. This is the only continue-path serde site.
        offload_surplus(local, &t.injector, &t.idle_workers, &t.counters);

        t.pending.fetch_sub(1, Ordering::SeqCst);

        if outcome.request_cancel {
            t.cancel.cancel();
            return;
        }
    }
}

/// The steady-state per-worker loop (angr-nkoct): same dispatch skeleton as
/// [`worker_loop`], but terminals are STREAMED up the session's mpsc channel as
/// they materialize (no barrier, no shared results vec), and running out of
/// work does not end the session:
///
/// * **Quiescence** (empty local + empty injector + `pending == 0`): send
///   `Quiesced` and return — the caller ([`worker_thread`]) parks on its
///   control channel, and a `WorkerCtl::Run` wake ping re-enters this loop
///   against the same session after the coordinator injects more work.
/// * **Cancel** (the session's finalize/num_find/budget signal): DRAIN the
///   residual local frontier upstream ([`drain_local_upstream`] — the steady
///   path's fix for wave-mode Bug M1), ack `Paused`, and return. Re-entry via a
///   stale wake ping on a cancelled session just re-acks `Paused` (empty
///   drain); the coordinator dedupes acks by `worker_id`.
///
/// All upstream sends ignore errors: the coordinator may have dropped the
/// receiver after finalize (session teardown), and a worker must park cleanly
/// through that race rather than panic.
///
/// The loop returns with `local` empty in EVERY case, so no `!Send` state is
/// parked inside the worker across a coordinator boundary — the resident
/// frontier lives only in BUSY workers.
fn worker_session_loop(
    session: &RunSession,
    worker_id: usize,
    ctx: &Context,
    block_cache: &mut LruCache<u64, Arc<IRSB>>,
    local: &mut VecDeque<RustSimState>,
) {
    let t = &session.transport;

    // Absorb any frontier a prior CANCELLED wave retained on `local` (the wave
    // loop's M1 retention path) into this session's quiescence accounting, so
    // mode-mixing on one pool is safe. A normal wake ping enters with `local`
    // empty and this is a no-op.
    if !local.is_empty() {
        t.pending.fetch_add(local.len(), Ordering::SeqCst);
    }

    loop {
        if t.cancel.is_cancelled() {
            drain_local_upstream(session, local);
            let _ = session.up_tx.send(WorkerUp::Paused { worker_id });
            return;
        }

        let state = match dispatch_next(t, local, ctx) {
            Some(state) => state,
            None => {
                if t.cancel.is_cancelled() {
                    // Cancel landed while spinning in the injector: drain (the
                    // local queue is already empty here, but keep the single
                    // drain path) and ack.
                    drain_local_upstream(session, local);
                    let _ = session.up_tx.send(WorkerUp::Paused { worker_id });
                } else {
                    // Global quiescence: park empty-handed until a wake ping.
                    let _ = session.up_tx.send(WorkerUp::Quiesced { worker_id });
                }
                return;
            }
        };

        let outcome = (session.process)(state, &t.cancel, block_cache);

        absorb_continues(t, local, outcome.continue_states);

        // Stream materialized terminals up as they happen — the coordinator
        // routes them (found/unconstrained/bounce) while other states are still
        // being stepped. Detached HERE (correct context), exactly like the wave
        // loop's results push.
        for terminal in outcome.terminal_states {
            let payload = terminal.detach_for_migration();
            t.counters
                .materialized_terminals
                .fetch_add(1, Ordering::SeqCst);
            let _ = session.up_tx.send(WorkerUp::Terminal { payload });
        }

        // Dead paths stay cheap summaries, counted and dropped in-context (the
        // same deadended-content caveat as wave mode).
        if !outcome.terminal_summaries.is_empty() {
            t.counters
                .summarized_terminals
                .fetch_add(outcome.terminal_summaries.len(), Ordering::SeqCst);
        }

        offload_surplus(local, &t.injector, &t.idle_workers, &t.counters);

        t.pending.fetch_sub(1, Ordering::SeqCst);

        if outcome.request_cancel {
            t.cancel.cancel();
            // Loop back to the cancel check, which drains + acks Paused.
        }
    }
}

/// Detach every residual live state on `local` and stream it upstream as an
/// (untagged) `Terminal` — the coordinator routes untagged payloads back to the
/// active stash. Balances `pending` for the states it removes. This is the
/// steady-state fix for wave-mode Bug M1: a cancelled/finalized session returns
/// its un-explored frontier instead of dropping it.
fn drain_local_upstream(session: &RunSession, local: &mut VecDeque<RustSimState>) {
    if local.is_empty() {
        return;
    }
    let t = &session.transport;
    let n = local.len();
    for state in local.drain(..) {
        let payload = state.detach_for_migration();
        t.counters.residual_drains.fetch_add(1, Ordering::SeqCst);
        let _ = session.up_tx.send(WorkerUp::Terminal { payload });
    }
    t.pending.fetch_sub(n, Ordering::SeqCst);
}

/// Pull the next state to process: the live local queue first (LIFO — the
/// freshest child is hottest in cache and the Z3 context; zero serde), else
/// steal + reattach from the shared injector. Returns `None` when no task is
/// available and none can ever appear (quiescence) — or on cancellation.
/// Shared verbatim by [`worker_loop`] and [`worker_session_loop`] (DRY: the
/// dispatch half of the two modes is identical; only terminal transport and
/// end-of-work behavior differ).
fn dispatch_next(
    t: &WorkTransport,
    local: &mut VecDeque<RustSimState>,
    ctx: &Context,
) -> Option<RustSimState> {
    loop {
        match local.pop_back() {
            Some(state) => {
                t.counters.local_dispatches.fetch_add(1, Ordering::SeqCst);
                return Some(state);
            }
            None => {
                match steal_from_injector(&t.injector, &t.pending, &t.cancel, &t.idle_workers) {
                    Some(payload) => {
                        t.counters
                            .injector_dispatches
                            .fetch_add(1, Ordering::SeqCst);
                        // Rebuild the state in THIS worker's context. The ptr-eq
                        // guard inside `reattach` holds because `ctx` is exactly this
                        // thread's thread-local.
                        match payload.reattach(ctx) {
                            Ok(state) => {
                                // Observability only (angr-vh834 Phase 1): count every
                                // injector-steal reattach. No routing change.
                                t.counters.reattaches.fetch_add(1, Ordering::SeqCst);
                                return Some(state);
                            }
                            Err(err) => {
                                // Unreachable in practice (the worker set its own ctx
                                // as thread-local), but never silently keep a phantom
                                // task outstanding.
                                log::error!("scheduler reattach failed, dropping task: {err:?}");
                                t.pending.fetch_sub(1, Ordering::SeqCst);
                                continue;
                            }
                        }
                    }
                    None => return None,
                }
            }
        }
    }
}

/// Push live successors onto the worker's local queue and count them IN before
/// the caller counts the parent task OUT (`pending` must never dip to zero with
/// live descendants queued).
fn absorb_continues(
    t: &WorkTransport,
    local: &mut VecDeque<RustSimState>,
    continue_states: Vec<RustSimState>,
) {
    let spawned = continue_states.len();
    for child in continue_states {
        local.push_back(child);
    }
    if spawned > 0 {
        t.pending.fetch_add(spawned, Ordering::SeqCst);
    }
}

/// Shed surplus live states from `local` to the shared `injector`, detaching
/// each into a `Send` payload. Two triggers:
///
/// * **A — idle-gated (primary):** if any sibling is currently starving
///   (`idle_workers > 0`) and we hold a backlog (`len >= 2`), offload up to half
///   the backlog (hysteresis, never below one local state) so the starving
///   sibling has work to steal. This mirrors the `record_migration_sample`
///   imbalance model (helpers.rs) so the measured steal fraction lines up with
///   the gate's `f_model`.
/// * **B — high-water cap (safety):** if the local queue exceeds [`LOCAL_HWM`],
///   shed down to `HWM/2` regardless of idle siblings, bounding per-worker
///   memory. Rarely fires on a narrow frontier.
///
/// Offload is a relocation of already-counted tasks; it does not touch
/// `pending`.
fn offload_surplus(
    local: &mut VecDeque<RustSimState>,
    injector: &Injector<StateMigrationPayload>,
    idle_workers: &AtomicUsize,
    counters: &SchedulerCounters,
) {
    // Trigger A: idle-gated load sharing. Offload the COLDEST states (front),
    // keeping our hot tail; at most half per production event for hysteresis.
    if idle_workers.load(Ordering::SeqCst) > 0 && local.len() >= 2 {
        let to_offload = local.len() / 2;
        for _ in 0..to_offload {
            if local.len() <= 1 {
                break;
            }
            if let Some(state) = local.pop_front() {
                injector.push(state.detach_for_migration());
                counters.surplus_offloaded.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    // Trigger B: hard memory cap.
    if local.len() > LOCAL_HWM {
        while local.len() > LOCAL_HWM / 2 {
            if let Some(state) = local.pop_front() {
                injector.push(state.detach_for_migration());
                counters.surplus_offloaded.fetch_add(1, Ordering::SeqCst);
            } else {
                break;
            }
        }
    }
}

/// Pull one payload from the shared injector, blocking (cooperative yield) while
/// work may still appear. Returns `None` only when the injector is empty AND no
/// task is outstanding anywhere (a task is only created by an in-flight task, so
/// none can ever appear again) — or when cancellation is requested.
///
/// Maintains the `idle_workers` signal across the wait so producers can detect
/// starvation and offload (Trigger A).
fn steal_from_injector(
    injector: &Injector<StateMigrationPayload>,
    pending: &AtomicUsize,
    cancel: &CancelToken,
    idle_workers: &AtomicUsize,
) -> Option<StateMigrationPayload> {
    idle_workers.fetch_add(1, Ordering::SeqCst);
    let result = loop {
        if cancel.is_cancelled() {
            break None;
        }
        match injector.steal() {
            Steal::Success(task) => break Some(task),
            Steal::Retry => continue,
            Steal::Empty => {
                // Empty right now. If nothing is outstanding anywhere we are
                // done; otherwise another worker is mid-task and may yet offload
                // surplus — yield and retry.
                if pending.load(Ordering::SeqCst) == 0 {
                    break None;
                }
                std::thread::yield_now();
            }
        }
    };
    idle_workers.fetch_sub(1, Ordering::SeqCst);
    result
}

#[cfg(test)]
mod tests {
    use super::{LOCAL_HWM, ParallelScheduler, TaskOutcome, TerminalDisposition, TerminalSummary};
    use crate::state::RustSimState;
    use crate::symbolic::RustBV;
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use z3::Context;

    /// Build a state with `rax` pinned to `witness` by a path constraint, in
    /// the current thread-local Z3 context.
    fn pinned_state(name: &str, witness: u64) -> RustSimState {
        let mut state = RustSimState::new("amd64").unwrap();
        let x = {
            let s = state.solver().borrow();
            RustBV::symbolic(&s, name, 64)
        };
        state.set_register("rax", x.clone());
        let c = {
            let s = state.solver().borrow();
            x.eq(&RustBV::concrete(witness as u128, 64), &s)
        };
        state.add_constraint(c);
        state
    }

    // angr-1ilq.3: N independent states are distributed across W workers, each
    // reattached into that worker's OWN Z3 context, re-proven, detached back to
    // a payload, and collected. Reattaching every collected payload in the main
    // context must recover the exact (state_id -> witness) map — proving the
    // work-stealing transport preserves both identity and constraints across
    // detach -> steal -> reattach(worker) -> detach -> reattach(main), under
    // genuine concurrency.
    #[test]
    fn test_scheduler_distributes_and_reproves() {
        const N: u64 = 96;
        let main_ctx = Context::thread_local();

        let mut payloads = Vec::with_capacity(N as usize);
        let mut expected: BTreeMap<u64, u128> = BTreeMap::new();
        for i in 0..N {
            let witness = 0xA000_0000_u64 + i;
            let state = pinned_state(&format!("sched_{i}"), witness);
            expected.insert(state.state_id(), witness as u128);
            payloads.push(state.detach_for_migration());
        }

        let sched = ParallelScheduler::new(4);
        let collected = sched.run(payloads, |state, _cancel, _cache| {
            // Re-prove in the worker's context before passing it on. The
            // per-witness check happens on the main thread; here we only assert
            // the reattached constraint is still evaluable (a dropped or
            // foreign-context constraint would yield None / panic in eval).
            let rax = state.get_register("rax").expect("rax present on worker");
            assert!(
                state.solver().borrow().eval(&rax).is_some(),
                "worker: reattached rax must be concretizable",
            );
            TaskOutcome::terminal(vec![state])
        });

        assert_eq!(collected.len(), N as usize, "every state must be collected");
        let mut seen: BTreeMap<u64, u128> = BTreeMap::new();
        for payload in collected {
            let state = payload.reattach(&main_ctx).expect("reattach in main ctx");
            let rax = state.get_register("rax").expect("rax present in main");
            let got = state
                .solver()
                .borrow()
                .eval(&rax)
                .expect("rax concretizable in main");
            assert!(
                seen.insert(state.state_id(), got).is_none(),
                "state_id {} collected twice",
                state.state_id(),
            );
        }
        assert_eq!(
            seen, expected,
            "every state must re-prove its witness with its identity intact",
        );
    }

    // angr-1ilq.3: dynamic work generation. One root fans out into a binary
    // tree of 2^DEPTH leaves; non-leaf tasks fork two children (in the worker's
    // context) and re-inject them, leaves are collected. This exercises (a)
    // quiescence detection (the pool must terminate exactly when the whole tree
    // is drained, never early), (b) cross-worker stealing of dynamically
    // produced work, and (c) constraint fidelity through fork-in-worker-context
    // — every leaf must still re-prove the root's path constraint.
    #[test]
    fn test_scheduler_fork_tree_quiesces() {
        const DEPTH: u64 = 7; // 128 leaves, 255 total tasks
        const WITNESS: u64 = 0xDEAD_BEEF;
        let main_ctx = Context::thread_local();

        let mut root = pinned_state("tree_acc", WITNESS);
        root.set_register("rbx", RustBV::concrete(0, 64)); // depth marker

        let sched = ParallelScheduler::new(4);
        let collected = sched.run(
            vec![root.detach_for_migration()],
            |state, _cancel, _cache| {
                let depth = state
                    .get_register("rbx")
                    .and_then(|d| d.as_u64())
                    .expect("depth marker present");
                if depth >= DEPTH {
                    return TaskOutcome::terminal(vec![state]);
                }
                let next = RustBV::concrete((depth + 1) as u128, 64);
                let mut left = state.fork();
                let mut right = state.fork();
                left.set_register("rbx", next.clone());
                right.set_register("rbx", next);
                TaskOutcome::continuing(vec![left, right])
            },
        );

        assert_eq!(
            collected.len(),
            1usize << DEPTH,
            "must collect exactly 2^DEPTH leaves (no lost or duplicated work)",
        );
        for payload in collected {
            let state = payload
                .reattach(&main_ctx)
                .expect("reattach leaf in main ctx");
            let rax = state.get_register("rax").expect("rax on leaf");
            assert_eq!(
                state.solver().borrow().eval(&rax),
                Some(WITNESS as u128),
                "every leaf must re-prove the root's path constraint",
            );
        }
    }

    // angr-1ilq.3: task-boundary cancellation. With many states queued and
    // every task requesting cancel, the first processed task trips the shared
    // CancelToken; all workers must stop at their next task boundary, so the
    // pool drains far fewer than N states. Proves the AtomicBool propagates the
    // stop signal across threads (not just self-cancels one worker).
    #[test]
    fn test_scheduler_cancellation_stops_workers() {
        const N: u64 = 512;
        // The persistent pool stores the `process` closure in an `Arc<WaveJob>`
        // that outlives this frame, so the closure must be `'static` — it can no
        // longer borrow a stack local. Share the counter via `Arc` and read it
        // back after the wave.
        let processed = Arc::new(AtomicUsize::new(0));

        let mut payloads = Vec::with_capacity(N as usize);
        for i in 0..N {
            payloads.push(pinned_state(&format!("cancel_{i}"), 0x1000 + i).detach_for_migration());
        }

        let sched = ParallelScheduler::new(4);
        let collected = sched.run(payloads, {
            let processed = Arc::clone(&processed);
            move |state, _cancel, _cache| {
                processed.fetch_add(1, Ordering::SeqCst);
                // Every task asks to cancel; the first to run trips the token.
                TaskOutcome {
                    continue_states: Vec::new(),
                    terminal_states: vec![state],
                    terminal_summaries: Vec::new(),
                    request_cancel: true,
                }
            }
        });

        let total = processed.load(Ordering::SeqCst);
        assert!(total >= 1, "at least one task must run before cancellation");
        assert!(
            total < N as usize,
            "cancellation must stop the pool early: processed {total} of {N}",
        );
        // Collected == processed here (every processed task is terminal); the
        // bound proves work remained undone when the pool shut down.
        assert_eq!(
            collected.len(),
            total,
            "each processed task is collected once"
        );
    }

    // angr-729vn: the home-context fast path pays zero continue-serde. A single
    // worker explores a binary tree that never exceeds the high-water mark and
    // has no idle sibling to offload to, so NOT ONE continue-state is
    // serialized. `surplus_offloaded == 0` is the proof, since it is the only
    // continue-path detach site.
    #[test]
    fn test_fast_path_never_serializes() {
        const DEPTH: u64 = 5; // 32 leaves; DFS queue depth ~= DEPTH << HWM
        const WITNESS: u64 = 0x5151;

        let mut root = pinned_state("fast_acc", WITNESS);
        root.set_register("rbx", RustBV::concrete(0, 64));

        let sched = ParallelScheduler::new(1); // no sibling => Trigger A never fires
        // Leaves are SUMMARIZED (no serde) so a single-worker DFS that stays
        // below the high-water mark serializes nothing at all — continue-states
        // stay live-local and dead leaves are summarized.
        let (collected, summaries, stats) = sched.run_instrumented(
            vec![root.detach_for_migration()],
            |state, _cancel, _cache| {
                let depth = state
                    .get_register("rbx")
                    .and_then(|d| d.as_u64())
                    .expect("depth marker");
                if depth >= DEPTH {
                    return TaskOutcome::summarized(vec![TerminalSummary::of(
                        &state,
                        TerminalDisposition::Deadended,
                    )]);
                }
                let next = RustBV::concrete((depth + 1) as u128, 64);
                let mut left = state.fork();
                let mut right = state.fork();
                left.set_register("rbx", next.clone());
                right.set_register("rbx", next);
                TaskOutcome::continuing(vec![left, right])
            },
        );

        assert!(
            collected.is_empty(),
            "nothing materialized => nothing serialized"
        );
        assert_eq!(summaries.len(), 1usize << DEPTH, "all leaves summarized");
        assert_eq!(
            stats.surplus_offloaded, 0,
            "fast path must serialize zero continue-states",
        );
        assert_eq!(stats.materialized_terminals, 0, "no terminals serialized");
        // Only the seed entered via the injector; all forks stayed live-local.
        assert_eq!(stats.injector_dispatches, stats.seeds);
        assert_eq!(
            stats.honest_steal_fraction(),
            0.0,
            "zero serde events => zero honest steal fraction",
        );
    }

    // angr-729vn: the high-water cap (Trigger B) sheds surplus deterministically
    // even with a single worker (no idle sibling). A root that forks WIDE past
    // the cap offloads down to HWM/2; exactly `width - HWM/2` states are
    // serialized.
    #[test]
    fn test_surplus_offload_triggers_at_hwm() {
        const WIDTH: usize = 200; // > LOCAL_HWM
        const WITNESS: u64 = 0x7777;
        const { assert!(WIDTH > LOCAL_HWM) };

        let mut root = pinned_state("wide_root", WITNESS);
        root.set_register("rbx", RustBV::concrete(0, 64));

        let sched = ParallelScheduler::new(1);
        let (collected, _summaries, stats) = sched.run_instrumented(
            vec![root.detach_for_migration()],
            |state, _cancel, _cache| {
                let depth = state
                    .get_register("rbx")
                    .and_then(|d| d.as_u64())
                    .expect("depth marker");
                if depth >= 1 {
                    return TaskOutcome::terminal(vec![state]);
                }
                // depth 0: fan out WIDTH leaves at once.
                let one = RustBV::concrete(1, 64);
                let children: Vec<RustSimState> = (0..WIDTH)
                    .map(|_| {
                        let mut c = state.fork();
                        c.set_register("rbx", one.clone());
                        c
                    })
                    .collect();
                TaskOutcome::continuing(children)
            },
        );

        assert_eq!(collected.len(), WIDTH, "all leaves collected");
        assert_eq!(
            stats.surplus_offloaded,
            WIDTH - LOCAL_HWM / 2,
            "Trigger B sheds the wide root's backlog down to HWM/2",
        );
    }

    // angr-729vn: the injector steal path is exercised across workers, and every
    // stolen leaf still re-proves its witness after detach -> steal ->
    // reattach. A wide root with >=2 workers forces real injector traffic.
    #[test]
    fn test_steal_from_injector_path() {
        const WIDTH: usize = 200;
        const WITNESS: u64 = 0xBEEF;
        let main_ctx = Context::thread_local();

        let mut root = pinned_state("steal_root", WITNESS);
        root.set_register("rbx", RustBV::concrete(0, 64));

        let sched = ParallelScheduler::new(4);
        let (collected, _summaries, stats) = sched.run_instrumented(
            vec![root.detach_for_migration()],
            |state, _cancel, _cache| {
                let depth = state
                    .get_register("rbx")
                    .and_then(|d| d.as_u64())
                    .expect("depth marker");
                if depth >= 1 {
                    return TaskOutcome::terminal(vec![state]);
                }
                let one = RustBV::concrete(1, 64);
                let children: Vec<RustSimState> = (0..WIDTH)
                    .map(|_| {
                        let mut c = state.fork();
                        c.set_register("rbx", one.clone());
                        c
                    })
                    .collect();
                TaskOutcome::continuing(children)
            },
        );

        assert_eq!(collected.len(), WIDTH, "all leaves collected");
        assert!(
            stats.injector_dispatches > stats.seeds,
            "injector steal path must be exercised: {} dispatches vs {} seeds",
            stats.injector_dispatches,
            stats.seeds,
        );
        for payload in collected {
            let state = payload.reattach(&main_ctx).expect("reattach leaf");
            let rax = state.get_register("rax").expect("rax on leaf");
            assert_eq!(
                state.solver().borrow().eval(&rax),
                Some(WITNESS as u128),
                "stolen leaf must re-prove the root constraint",
            );
        }
    }

    // angr-729vn: every dispatched task is accounted for exactly once across the
    // two-tier (local + injector) model — quiescence under imbalance loses and
    // duplicates nothing. A skewed tree on 4 workers; total dispatches must
    // equal the exact task count.
    #[test]
    fn test_quiescence_accounting_under_imbalance() {
        const DEPTH: u64 = 7; // 255 total tasks
        const WITNESS: u64 = 0xABCD;

        let mut root = pinned_state("quiesce_root", WITNESS);
        root.set_register("rbx", RustBV::concrete(0, 64));

        let sched = ParallelScheduler::new(4);
        let (collected, _summaries, stats) = sched.run_instrumented(
            vec![root.detach_for_migration()],
            |state, _cancel, _cache| {
                let depth = state
                    .get_register("rbx")
                    .and_then(|d| d.as_u64())
                    .expect("depth marker");
                if depth >= DEPTH {
                    return TaskOutcome::terminal(vec![state]);
                }
                let next = RustBV::concrete((depth + 1) as u128, 64);
                let mut left = state.fork();
                let mut right = state.fork();
                left.set_register("rbx", next.clone());
                right.set_register("rbx", next);
                TaskOutcome::continuing(vec![left, right])
            },
        );

        let total_tasks = (1usize << (DEPTH + 1)) - 1; // full binary tree
        assert_eq!(collected.len(), 1usize << DEPTH, "all leaves collected");
        assert_eq!(
            stats.local_dispatches + stats.injector_dispatches,
            total_tasks,
            "every task dispatched exactly once across local + injector tiers",
        );
    }

    // angr-729vn: the result SET is deterministic despite nondeterministic steal
    // ordering. Run the same independent-states workload K times and assert the
    // multiset of recovered witnesses is identical every time (no lost,
    // duplicated, or corrupted states). Ordering is NOT asserted (it is the
    // downstream fingerprint gate's concern).
    #[test]
    fn test_determinism_result_set() {
        const N: u64 = 64;
        const K: usize = 5;
        let main_ctx = Context::thread_local();

        let mut runs: Vec<BTreeSet<u128>> = Vec::with_capacity(K);
        for _ in 0..K {
            let payloads: Vec<_> = (0..N)
                .map(|i| pinned_state(&format!("det_{i}"), 0xC000 + i).detach_for_migration())
                .collect();
            let sched = ParallelScheduler::new(4);
            let collected = sched.run(payloads, |state, _cancel, _cache| {
                TaskOutcome::terminal(vec![state])
            });
            let witnesses: BTreeSet<u128> = collected
                .into_iter()
                .map(|p| {
                    let s = p.reattach(&main_ctx).expect("reattach");
                    let rax = s.get_register("rax").expect("rax");
                    s.solver().borrow().eval(&rax).expect("concretizable")
                })
                .collect();
            runs.push(witnesses);
        }

        let expected: BTreeSet<u128> = (0..N as u128).map(|i| 0xC000 + i).collect();
        for (k, run) in runs.iter().enumerate() {
            assert_eq!(*run, expected, "run {k} recovered a different witness set");
        }
    }

    // angr-729vn: summaries pay no serde. A workload that splits terminals
    // between materialized (found) and summarized (dead) paths. Summaries land
    // in the summaries vec, never the results vec, and are excluded from the
    // honest steal fraction; the only serde sites are materialized terminals +
    // surplus offloads.
    #[test]
    fn test_summaries_pay_no_serde_and_fraction() {
        const N: u64 = 100;
        const FOUND_EVERY: u64 = 10; // 10 materialized, 90 summarized

        let payloads: Vec<_> = (0..N)
            .map(|i| pinned_state(&format!("term_{i}"), 0xD000 + i).detach_for_migration())
            .collect();

        let sched = ParallelScheduler::new(4);
        let (collected, summaries, stats) =
            sched.run_instrumented(payloads, |state, _cancel, _cache| {
                if state.state_id().is_multiple_of(FOUND_EVERY) {
                    TaskOutcome::terminal(vec![state]) // materialized (serialized)
                } else {
                    TaskOutcome::summarized(vec![TerminalSummary::of(
                        &state,
                        TerminalDisposition::Deadended,
                    )]) // no serde
                }
            });

        // Partition is exact and complete.
        assert_eq!(
            stats.materialized_terminals + stats.summarized_terminals,
            N as usize,
        );
        assert_eq!(collected.len(), stats.materialized_terminals);
        assert_eq!(summaries.len(), stats.summarized_terminals);
        assert!(
            stats.summarized_terminals > 0,
            "test must exercise the summary path",
        );

        // No continue-states, so the only serde is materialized terminals.
        assert_eq!(stats.surplus_offloaded, 0);
        let expected_f = stats.materialized_terminals as f64 / stats.dispatches() as f64;
        assert!(
            (stats.honest_steal_fraction() - expected_f).abs() < 1e-9,
            "honest steal fraction must exclude summaries",
        );
        // Summaries carry the right disposition and cheap fields only.
        for s in &summaries {
            assert_eq!(s.disposition, TerminalDisposition::Deadended);
        }
    }

    // ------------------------------------------------------------------
    // Steady-state session tests (angr-nkoct Phase B). These drive
    // PersistentPool + RunSession directly, acting as a mini-coordinator:
    // inject seeds, receive streamed WorkerUp messages, wake parked workers.
    // ------------------------------------------------------------------

    use super::{PersistentPool, RunSession, WorkerUp};
    use crate::state::StateMigrationPayload;
    use std::time::Duration;

    /// Drive a session until every worker parks (Quiesced or Paused),
    /// collecting streamed terminal payloads. Once all workers are parked no
    /// further message can be produced (each worker's sends precede its own
    /// park ack in the channel's FIFO), so a final `try_recv` drain is
    /// complete. Panics on a 60s stall — the liveness guard for park/wake
    /// bugs. Returns `(terminals, paused_acks, quiesced_acks)` with acks
    /// DEDUPED by worker id (stale wake pings re-ack; see worker_session_loop).
    fn collect_until_parked(
        up_rx: &std::sync::mpsc::Receiver<WorkerUp>,
        workers: usize,
    ) -> (Vec<StateMigrationPayload>, usize, usize) {
        let mut terminals = Vec::new();
        let mut parked: BTreeSet<usize> = BTreeSet::new();
        let (mut paused, mut quiesced) = (0usize, 0usize);
        while parked.len() < workers {
            match up_rx.recv_timeout(Duration::from_secs(60)) {
                Ok(WorkerUp::Terminal { payload }) => terminals.push(payload),
                Ok(WorkerUp::Paused { worker_id }) => {
                    if parked.insert(worker_id) {
                        paused += 1;
                    }
                }
                Ok(WorkerUp::Quiesced { worker_id }) => {
                    if parked.insert(worker_id) {
                        quiesced += 1;
                    }
                }
                Err(e) => panic!("session stalled waiting for workers to park: {e:?}"),
            }
        }
        while let Ok(msg) = up_rx.try_recv() {
            if let WorkerUp::Terminal { payload } = msg {
                terminals.push(payload);
            }
        }
        (terminals, paused, quiesced)
    }

    /// The fork-tree process closure shared by the session tests: `rbx` is a
    /// depth marker; non-leaves fork two children, leaves are materialized.
    fn fork_tree_process(
        depth_max: u64,
    ) -> impl Fn(
        RustSimState,
        &super::CancelToken,
        &mut lru::LruCache<u64, Arc<crate::vex::IRSB>>,
    ) -> TaskOutcome
    + Send
    + Sync
    + 'static {
        move |state, _cancel, _cache| {
            let depth = state
                .get_register("rbx")
                .and_then(|d| d.as_u64())
                .expect("depth marker present");
            if depth >= depth_max {
                return TaskOutcome::terminal(vec![state]);
            }
            let next = crate::symbolic::RustBV::concrete((depth + 1) as u128, 64);
            let mut left = state.fork();
            let mut right = state.fork();
            left.set_register("rbx", next.clone());
            right.set_register("rbx", next);
            TaskOutcome::continuing(vec![left, right])
        }
    }

    // angr-nkoct Phase B: a session streams a dynamically forked tree's leaves
    // up the mpsc channel (no barrier) and every worker parks with Quiesced
    // when the tree is drained. Every leaf re-proves the root constraint after
    // detach -> steal -> fork-in-worker -> detach -> reattach(main).
    #[test]
    fn test_session_fork_tree_streams_and_quiesces() {
        const DEPTH: u64 = 7; // 128 leaves
        const WITNESS: u64 = 0x5E55_0001;
        const WORKERS: usize = 4;
        let main_ctx = Context::thread_local();

        let mut root = pinned_state("sess_tree", WITNESS);
        root.set_register("rbx", RustBV::concrete(0, 64));

        let pool = PersistentPool::new(WORKERS);
        let (session, up_rx) = RunSession::new(Box::new(fork_tree_process(DEPTH)));
        session.inject_seeds(vec![root.detach_for_migration()]);
        pool.start_session(&session);

        let (terminals, paused, quiesced) = collect_until_parked(&up_rx, WORKERS);
        assert_eq!(paused, 0, "no cancel => no Paused acks");
        assert_eq!(quiesced, WORKERS, "every worker parks with Quiesced");
        assert_eq!(
            terminals.len(),
            1usize << DEPTH,
            "no lost/duplicated leaves"
        );
        assert_eq!(session.pending(), 0, "quiescence accounting balanced");

        for payload in terminals {
            let state = payload.reattach(&main_ctx).expect("reattach leaf");
            let rax = state.get_register("rax").expect("rax on leaf");
            assert_eq!(
                state.solver().borrow().eval(&rax),
                Some(WITNESS as u128),
                "every streamed leaf must re-prove the root constraint",
            );
        }

        let stats = session.stats();
        assert_eq!(stats.seeds, 1);
        assert_eq!(stats.materialized_terminals, 1usize << DEPTH);
        assert_eq!(stats.resume_reinjects, 0);
        assert_eq!(stats.residual_drains, 0);
    }

    // angr-nkoct Phase B: park/wake liveness across a simulated Python-callback
    // gap. The session quiesces (all workers parked), the coordinator injects
    // more work via inject_resumed + wake pings, and the SAME session drains
    // the second tree too. resume_reinjects counts exactly the re-injected
    // payloads; stale wake pings (broadcast to all workers when one payload
    // exists) are absorbed without harm.
    #[test]
    fn test_session_reinject_across_callback_gap() {
        const DEPTH: u64 = 5; // 32 leaves per tree
        const WITNESS_A: u64 = 0xAAA0;
        const WITNESS_B: u64 = 0xBBB0;
        const WORKERS: usize = 4;
        let main_ctx = Context::thread_local();

        let pool = PersistentPool::new(WORKERS);
        let (session, up_rx) = RunSession::new(Box::new(fork_tree_process(DEPTH)));

        let mut root_a = pinned_state("gap_a", WITNESS_A);
        root_a.set_register("rbx", RustBV::concrete(0, 64));
        session.inject_seeds(vec![root_a.detach_for_migration()]);
        pool.start_session(&session);

        let (first, _, quiesced) = collect_until_parked(&up_rx, WORKERS);
        assert_eq!(quiesced, WORKERS);
        assert_eq!(first.len(), 1usize << DEPTH);

        // "Python callback gap": all workers are parked; the session (and every
        // worker's Z3 context + warm cache) stays alive. Re-inject and wake.
        let mut root_b = pinned_state("gap_b", WITNESS_B);
        root_b.set_register("rbx", RustBV::concrete(0, 64));
        session.inject_resumed(vec![root_b.detach_for_migration()]);
        for worker_id in 0..WORKERS {
            pool.wake_worker(worker_id, &session);
        }

        let (second, _, quiesced2) = collect_until_parked(&up_rx, WORKERS);
        assert_eq!(quiesced2, WORKERS, "workers re-park after the second tree");
        assert_eq!(second.len(), 1usize << DEPTH, "second tree fully drained");
        assert_eq!(session.pending(), 0);

        let witnesses: BTreeSet<u128> = second
            .into_iter()
            .map(|p| {
                let s = p.reattach(&main_ctx).expect("reattach");
                let rax = s.get_register("rax").expect("rax");
                s.solver().borrow().eval(&rax).expect("concretizable")
            })
            .collect();
        assert_eq!(
            witnesses,
            BTreeSet::from([WITNESS_B as u128]),
            "second-tree leaves prove the re-injected root's constraint",
        );

        let stats = session.stats();
        assert_eq!(stats.seeds, 1, "only the first root is a seed");
        assert_eq!(stats.resume_reinjects, 1, "exactly one resume re-inject");
    }

    // angr-nkoct Phase B: cancel/finalize drains the residual frontier instead
    // of dropping it (the steady-state fix for wave-mode Bug M1). A root fans
    // out WIDTH children; processing any child requests cancel, so whichever
    // worker holds the sibling backlog drains it upstream. Conservation is
    // EXACT: every child is either a processed terminal, a worker-local
    // residual, or an injector residual — nothing lost, pending balanced.
    #[test]
    fn test_session_cancel_drains_residual_frontier() {
        const WIDTH: usize = 40;
        const WITNESS: u64 = 0xF1F1;
        const WORKERS: usize = 2;
        let main_ctx = Context::thread_local();

        let mut root = pinned_state("drain_root", WITNESS);
        root.set_register("rbx", RustBV::concrete(0, 64));

        let pool = PersistentPool::new(WORKERS);
        let (session, up_rx) = RunSession::new(Box::new(
            move |state: RustSimState,
                  _cancel: &super::CancelToken,
                  _cache: &mut lru::LruCache<u64, Arc<crate::vex::IRSB>>| {
                let depth = state
                    .get_register("rbx")
                    .and_then(|d| d.as_u64())
                    .expect("depth marker");
                if depth == 0 {
                    let one = RustBV::concrete(1, 64);
                    let children: Vec<RustSimState> = (0..WIDTH)
                        .map(|_| {
                            let mut c = state.fork();
                            c.set_register("rbx", one.clone());
                            c
                        })
                        .collect();
                    return TaskOutcome::continuing(children);
                }
                // Every processed child is terminal AND requests cancel — the
                // first one to run trips the token while its worker still holds
                // the sibling backlog.
                TaskOutcome {
                    continue_states: Vec::new(),
                    terminal_states: vec![state],
                    terminal_summaries: Vec::new(),
                    request_cancel: true,
                }
            },
        ));
        session.inject_seeds(vec![root.detach_for_migration()]);
        pool.start_session(&session);

        let (terminals, paused, _quiesced) = collect_until_parked(&up_rx, WORKERS);
        assert_eq!(paused, WORKERS, "cancel => every worker acks Paused");
        assert!(session.is_cancelled());

        // Injector half of the residual (offloaded but never stolen).
        let injector_residuals = session.drain_residual_payloads();
        assert_eq!(session.pending(), 0, "accounting balanced after full drain");

        let stats = session.stats();
        assert!(
            stats.residual_drains > 0,
            "cancel must drain a residual frontier (got {} terminals, {} residual)",
            stats.materialized_terminals,
            stats.residual_drains,
        );
        // Exact conservation: every child either materialized on processing or
        // came back as a residual (worker-local drain or injector drain).
        assert_eq!(
            stats.materialized_terminals + stats.residual_drains,
            WIDTH,
            "no child lost or duplicated across cancel",
        );
        assert_eq!(terminals.len() + injector_residuals.len(), WIDTH);

        for payload in terminals.into_iter().chain(injector_residuals) {
            let state = payload.reattach(&main_ctx).expect("reattach residual");
            let rax = state.get_register("rax").expect("rax");
            assert_eq!(
                state.solver().borrow().eval(&rax),
                Some(WITNESS as u128),
                "residual states keep their constraints through the drain",
            );
        }
    }

    // angr-nkoct Phase B: the streamed result SET is deterministic despite
    // nondeterministic steal/stream ordering — the session analogue of
    // test_determinism_result_set.
    #[test]
    fn test_session_determinism_result_set() {
        const N: u64 = 64;
        const K: usize = 3;
        const WORKERS: usize = 4;
        let main_ctx = Context::thread_local();

        let mut runs: Vec<BTreeSet<u128>> = Vec::with_capacity(K);
        for _ in 0..K {
            let pool = PersistentPool::new(WORKERS);
            let (session, up_rx) = RunSession::new(Box::new(
                |state: RustSimState,
                 _cancel: &super::CancelToken,
                 _cache: &mut lru::LruCache<u64, Arc<crate::vex::IRSB>>| {
                    TaskOutcome::terminal(vec![state])
                },
            ));
            let payloads: Vec<_> = (0..N)
                .map(|i| pinned_state(&format!("sdet_{i}"), 0xE000 + i).detach_for_migration())
                .collect();
            session.inject_seeds(payloads);
            pool.start_session(&session);

            let (terminals, _, quiesced) = collect_until_parked(&up_rx, WORKERS);
            assert_eq!(quiesced, WORKERS);
            let witnesses: BTreeSet<u128> = terminals
                .into_iter()
                .map(|p| {
                    let s = p.reattach(&main_ctx).expect("reattach");
                    let rax = s.get_register("rax").expect("rax");
                    s.solver().borrow().eval(&rax).expect("concretizable")
                })
                .collect();
            runs.push(witnesses);
        }

        let expected: BTreeSet<u128> = (0..N as u128).map(|i| 0xE000 + i).collect();
        for (k, run) in runs.iter().enumerate() {
            assert_eq!(*run, expected, "session run {k} lost/changed a witness");
        }
    }
}
