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
#[derive(Default)]
struct SchedulerCounters {
    local_dispatches: AtomicUsize,
    injector_dispatches: AtomicUsize,
    surplus_offloaded: AtomicUsize,
    materialized_terminals: AtomicUsize,
    summarized_terminals: AtomicUsize,
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

/// All per-wave state a worker touches, fed to the persistent pool as an
/// `Arc<WaveJob>`. It bundles the coordination locals the former
/// `run_instrumented` held on its stack (`injector`, `pending`, a fresh
/// per-wave `cancel`, `results` / `summaries`, `idle_workers`, `counters`) with
/// the boxed [`ProcessFn`] that owns/`Arc`-shares every input the per-state work
/// needs. The coordinator recovers sole ownership after the wave barrier via
/// `Arc::into_inner` (sound because each worker drops its clone before signaling
/// `WaveDone`).
pub(crate) struct WaveJob {
    injector: Injector<StateMigrationPayload>,
    pending: AtomicUsize,
    cancel: CancelToken,
    results: Mutex<Vec<StateMigrationPayload>>,
    summaries: Mutex<Vec<TerminalSummary>>,
    idle_workers: AtomicUsize,
    counters: SchedulerCounters,
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
        let injector = Injector::<StateMigrationPayload>::new();
        for payload in initial {
            injector.push(payload);
        }
        Self {
            injector,
            pending: AtomicUsize::new(seeds),
            cancel: CancelToken::new(),
            results: Mutex::new(Vec::new()),
            summaries: Mutex::new(Vec::new()),
            idle_workers: AtomicUsize::new(0),
            counters: SchedulerCounters::default(),
            seeds,
            process,
        }
    }

    /// Snapshot the accumulated counters into a [`SchedulerStats`]. Call after
    /// the wave barrier, when this thread is the sole owner.
    pub(crate) fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            seeds: self.seeds,
            local_dispatches: self.counters.local_dispatches.load(Ordering::SeqCst),
            injector_dispatches: self.counters.injector_dispatches.load(Ordering::SeqCst),
            surplus_offloaded: self.counters.surplus_offloaded.load(Ordering::SeqCst),
            materialized_terminals: self.counters.materialized_terminals.load(Ordering::SeqCst),
            summarized_terminals: self.counters.summarized_terminals.load(Ordering::SeqCst),
        }
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

/// A message to a persistent worker: run one wave, or shut down.
enum WaveMsg {
    Run(Arc<WaveJob>),
    Shutdown,
}

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
    job_txs: Vec<Sender<WaveMsg>>,
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
            let (job_tx, job_rx) = mpsc::channel::<WaveMsg>();
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
            tx.send(WaveMsg::Run(Arc::clone(&job)))
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
}

impl Drop for PersistentPool {
    /// Broadcast `Shutdown` to every worker, then join. Each worker's Z3 context
    /// drops on its own thread — never freed cross-thread.
    fn drop(&mut self) {
        for tx in &self.job_txs {
            let _ = tx.send(WaveMsg::Shutdown);
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

/// A persistent worker: install a private Z3 context ONCE, then loop running one
/// wave per `WaveMsg::Run`. It drops its `Arc<WaveJob>` clone BEFORE sending
/// `WaveDone` so the coordinator's `Arc::into_inner` recovers sole ownership. On
/// `Shutdown` (or a closed channel at pool teardown) it falls out of the loop,
/// dropping its Z3 context on this same thread (no leak, no cross-thread free).
fn worker_thread(worker_id: usize, job_rx: Receiver<WaveMsg>, done_tx: Sender<WaveDone>) {
    // Owned for the worker's whole life, installed as the thread-local. Every
    // reattach mints ASTs into a context this thread alone touches; created and
    // destroyed on the same thread. The per-wave `local` live queue is created
    // inside `worker_loop`, so the irreducibly `!Send` `RustSimState`s in it can
    // never escape this thread.
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
    while let Ok(WaveMsg::Run(job)) = job_rx.recv() {
        worker_loop(&job, &z3ctx, &mut block_cache);
        // Release this worker's Arc clone BEFORE signaling done — the invariant
        // that makes the coordinator's `Arc::into_inner` safe.
        drop(job);
        if done_tx.send(WaveDone { worker_id }).is_err() {
            // Coordinator is gone; nothing to synchronize with. Bail so the
            // context drops on this thread.
            break;
        }
    }
    // `WaveMsg::Shutdown` / channel closed / coordinator gone: `z3ctx` drops here.
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
/// persistent Z3 context and `block_cache` its warm per-worker IRSB cache (both
/// installed once at spawn by [`worker_thread`] and reused across waves).
fn worker_loop(job: &WaveJob, ctx: &Context, block_cache: &mut LruCache<u64, Arc<IRSB>>) {
    // Worker-local live states: home Z3 context, never serialized. Created
    // inside the worker so the `!Send` states it holds cannot escape the thread.
    let mut local: VecDeque<RustSimState> = VecDeque::new();

    loop {
        if job.cancel.is_cancelled() {
            // Bug M1 (known limitation): on cancel (e.g. the run loop hit
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
            // the `run_loop_parallel` doc comment.
            return;
        }

        // Dispatch: drain the live local queue first (LIFO — the freshest child
        // is hottest in cache and the Z3 context). Only when it is empty do we
        // touch the cross-thread injector and pay a reattach.
        let state = match local.pop_back() {
            Some(state) => {
                job.counters.local_dispatches.fetch_add(1, Ordering::SeqCst);
                state
            }
            None => match steal_from_injector(
                &job.injector,
                &job.pending,
                &job.cancel,
                &job.idle_workers,
            ) {
                Some(payload) => {
                    job.counters
                        .injector_dispatches
                        .fetch_add(1, Ordering::SeqCst);
                    // Rebuild the state in THIS worker's context. The ptr-eq
                    // guard inside `reattach` holds because `ctx` is exactly this
                    // thread's thread-local.
                    match payload.reattach(ctx) {
                        Ok(state) => state,
                        Err(err) => {
                            // Unreachable in practice (we set our own ctx as
                            // thread-local above), but never silently keep a
                            // phantom task outstanding.
                            log::error!("scheduler reattach failed, dropping task: {err:?}");
                            job.pending.fetch_sub(1, Ordering::SeqCst);
                            continue;
                        }
                    }
                }
                // No task available and nothing outstanding anywhere (or
                // cancelled): no future task can ever appear.
                None => return,
            },
        };

        let outcome = (job.process)(state, &job.cancel, block_cache);

        // Continue-states stay LIVE and LOCAL — no serde on the fast path.
        let spawned = outcome.continue_states.len();
        for child in outcome.continue_states {
            local.push_back(child);
        }

        // Materialized terminals must cross the join, so they are detached here
        // (correct context). This is part of the honest steal fraction.
        if !outcome.terminal_states.is_empty() {
            let mut guard = job.results.lock().expect("results mutex poisoned");
            for terminal in outcome.terminal_states {
                guard.push(terminal.detach_for_migration());
                job.counters
                    .materialized_terminals
                    .fetch_add(1, Ordering::SeqCst);
            }
        }

        // Summaries pay no serde — record and drop the full states in-context.
        if !outcome.terminal_summaries.is_empty() {
            let n = outcome.terminal_summaries.len();
            let mut guard = job.summaries.lock().expect("summaries mutex poisoned");
            guard.extend(outcome.terminal_summaries);
            job.counters
                .summarized_terminals
                .fetch_add(n, Ordering::SeqCst);
        }

        // Count children IN before counting this task OUT, so `pending` never
        // dips to zero with live descendants queued. Offload (below) only
        // relocates already-counted states, so it leaves `pending` untouched.
        if spawned > 0 {
            job.pending.fetch_add(spawned, Ordering::SeqCst);
        }

        // Shed surplus to the injector if a sibling is starving or we are over
        // the high-water mark. This is the only continue-path serde site.
        offload_surplus(&mut local, &job.injector, &job.idle_workers, &job.counters);

        job.pending.fetch_sub(1, Ordering::SeqCst);

        if outcome.request_cancel {
            job.cancel.cancel();
            return;
        }
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
}
