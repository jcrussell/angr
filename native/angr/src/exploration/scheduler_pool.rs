//! Steady-state session protocol and the persistent OS-thread pool that runs it
//! (extracted from `scheduler.rs`, angr-9ke6b.59).
//!
//! Two layers live here:
//!
//! * **Session protocol** — the duplex coordinator<->worker messages
//!   ([`WorkerCtl`] downstream, [`WorkerUp`] upstream) and [`RunSession`], the
//!   steady-state analogue of [`WaveJob`]: both wrap a
//!   [`WorkTransport`], and differ only in how
//!   terminals leave the worker and what a worker does when it runs dry.
//! * **Pool lifecycle** — [`PersistentPool`], the `num_workers` long-lived
//!   threads each owning a private Z3 context for the pool's whole life, plus the
//!   [`worker_thread`] entry point they run and the [`WaveDone`] barrier token
//!   the coordinator counts.
//!
//! This is a `#[path]` child of [`scheduler`](super), so the parent's
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]` and its `use` block both
//! reach here (the latter via the glob below). Every `.expect` here is one of the
//! invariant guards the parent's Panic policy section covers — read that first.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::stats::snapshot_stats;
use super::*;

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
    pub(super) transport: WorkTransport,
    /// Total payloads injected as fresh work (initial frontier drains), the
    /// `SchedulerStats::seeds` analogue. Resume re-injections are counted in
    /// `counters.resume_reinjects` instead.
    pub(super) seeds: AtomicUsize,
    /// Upstream channel every worker reports on (`std::sync::mpsc::Sender` is
    /// `Send + Sync` since the channel rewrite, so sharing one through the
    /// session `Arc` is sound — proven by the compile-time assertion below).
    pub(super) up_tx: Sender<WorkerUp>,
    /// The per-state processor, invoked once per dispatched state.
    pub(super) process: Box<ProcessFn>,
}

impl RunSession {
    /// Report upstream to the coordinator, ignoring a closed channel.
    ///
    /// SILENT(cat-a): a `SendError` here means the coordinator already dropped
    /// the receiver after finalizing the wave — the shutdown race the `up_tx`
    /// doc comment describes. There is nothing left to report to and nothing
    /// to recover, so the message is dropped by design. This is the single
    /// place workers discard that `Result`; open-coding `let _ = ... .send(..)`
    /// at each of the five call sites is what
    /// `clippy::let_underscore_must_use` now rejects.
    pub(super) fn notify_up(&self, msg: WorkerUp) {
        drop(self.up_tx.send(msg));
    }

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
pub(super) struct WaveDone;

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
            // SILENT(cat-a): a dead worker's channel just means there is
            // nobody left to wake — the ping is advisory (see the idempotence
            // note above), so a `SendError` is expected control flow.
            drop(tx.send(WorkerCtl::Run(Arc::clone(session))));
        }
    }
}

impl Drop for PersistentPool {
    /// Broadcast `Shutdown` to every worker, then join. Each worker's Z3 context
    /// drops on its own thread — never freed cross-thread.
    ///
    /// # Preconditions (angr-nkoct)
    ///
    /// The join below blocks until every worker reaches its control channel, so
    /// a caller dropping a pool with a **live** [`RunSession`] must first:
    ///
    /// 1. cancel that session and wake its parked workers (see
    ///    [`SteadySession::cancel_and_wake`](crate::exploration::run_loop_steady::SteadySession::cancel_and_wake)),
    ///    so a worker mid-dispatch actually reaches a cancel check instead of
    ///    running to natural quiescence; and
    /// 2. **release the GIL** across the drop. A session worker can be blocked
    ///    in `Python::attach` for a cold VEX lift; holding the GIL across the
    ///    join then deadlocks — and pyclass dealloc, the path that reaches this
    ///    impl in production, runs GIL-held.
    ///
    /// [`RustExplorationManager`](crate::exploration::RustExplorationManager)'s
    /// `Drop` is the only production call site that drops a live pool and is
    /// engineered for exactly this (`cancel_and_wake`, then
    /// `Python::attach(|py| py.detach(..))`); see the Steady-state Drop safety
    /// note on that impl. Wave-mode workers are always parked between waves, so
    /// a pool that never started a session needs neither step.
    fn drop(&mut self) {
        for tx in &self.job_txs {
            // SILENT(cat-a): a worker that already exited has dropped its
            // receiver; it needs no `Shutdown`.
            drop(tx.send(WorkerCtl::Shutdown));
        }
        for handle in self.handles.drain(..) {
            // SILENT(cat-a): `join` returns `Err` only when the worker
            // panicked, which it has already reported through the panic hook.
            // A `Drop` impl cannot propagate, and unwinding out of one during
            // an in-flight panic would abort the process.
            drop(handle.join());
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
pub(super) fn worker_thread(
    worker_id: usize,
    job_rx: Receiver<WorkerCtl>,
    done_tx: Sender<WaveDone>,
) {
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
    // Drop this worker's bridge thread-local AST caches eagerly, while `z3ctx`
    // is still the live thread-local Z3 context. The interpreter paths workers
    // run (`claripy_to_rustbv` in interpreter/{mod,expressions}.rs) populate
    // `AST_CACHE` with `RustBV` values whose z3 ASTs are bound to `z3ctx`, and
    // Rust runs `thread_local!` destructors at thread teardown — AFTER this
    // function returns and `z3ctx` (a local) has dropped. This call releases
    // that memory at a known point instead, and touches only this thread's
    // caches, never the cross-thread global registry (angr-1ilq.2).
    //
    // This is proactive resource hygiene, NOT a use-after-free fix, contrary to
    // how angr-bjk8 / angr-1yge9.9 originally described it (angr-9ke6b.39). Two
    // independent reasons the UAF it guarded against cannot occur here:
    //   1. `z3-patched`'s `Context` (native/z3-patched/src/context.rs) wraps an
    //      `Rc<ContextInternal>`, and every cached AST owns its own clone of it
    //      (`BV { ctx: Context, .. }` in z3-patched/src/ast/bv.rs).
    //      `Z3_del_context` runs from `ContextInternal::drop` only once the LAST
    //      handle drops, so a cached `RustBV` outliving the `z3ctx` local keeps
    //      the underlying context alive rather than dangling. Drop order is
    //      irrelevant. (The pre-Rc z3-rs lifetime model that motivated the
    //      original wording was superseded by commit d604c1f04.)
    //   2. The "a panic unwinds past this call and destructors run later against
    //      a freed context" sequence needs unwinding, and the shipped build has
    //      none: root Cargo.toml sets `panic = "abort"` (angr-1cue) — the same
    //      property the "Panic policy" header at the top of this module relies
    //      on. A panic aborts the process at the panic site.
    crate::claripy_bridge::clear_worker_local_caches();
    // Shutdown / channel closed / coordinator gone: `z3ctx` drops here.
}
