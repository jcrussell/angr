//! Wave-mode work distribution: the shared injector/pending/cancel core and the
//! per-wave job that wraps it (extracted from `scheduler.rs`, angr-9ke6b.59).
//!
//! [`WorkTransport`] is the dispatch/steal/quiescence core shared by BOTH
//! scheduling modes; [`WaveJob`] is the level-synchronous wave's envelope around
//! one (`Arc`-shared with every worker for the duration of a wave), and
//! [`ProcessFn`] the boxed per-state processor both modes dispatch through. The
//! steady-state counterpart — [`RunSession`](super::pool::RunSession) and the
//! duplex `WorkerCtl`/`WorkerUp` protocol — lives in
//! [`pool`](super::pool) alongside the thread pool that drives it.
//!
//! This is a `#[path]` child of [`scheduler`](super), so the parent's
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]` and its `use` block both
//! reach here (the latter via the glob below). The `.expect` sites are the
//! results/summaries poison guards covered verbatim by the parent's Panic
//! policy.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::stats::snapshot_stats;
use super::*;

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
/// (angr-vh834 Work Item 3): owned by [`worker_thread`](super::pool::worker_thread), created ONCE alongside
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
    pub(super) injector: Injector<StateMigrationPayload>,
    /// Count of queued + in-flight states (the quiescence detector). A worker
    /// counts spawned children IN before counting the parent task OUT.
    pub(super) pending: AtomicUsize,
    /// Cooperative cancellation, checked at task boundaries. In session mode
    /// this doubles as the finalize/drain signal: a cancelled session worker
    /// detaches its residual frontier upstream instead of dropping it.
    pub(super) cancel: CancelToken,
    /// Number of workers currently spinning for work (starvation signal for
    /// [`worker::offload_surplus`] Trigger A).
    pub(super) idle_workers: AtomicUsize,
    pub(super) counters: SchedulerCounters,
    /// Active-state selection / fork-insertion policy for the worker-local
    /// frontier (angr-1ilq.9). [`worker::dispatch_next`] routes its local pop through
    /// [`SelectionPolicy::select`] and [`worker::absorb_continues`] through
    /// [`SelectionPolicy::on_fork`], so a find-aware policy can reorder the
    /// per-worker frontier without touching the dispatch skeleton. Defaults to
    /// [`Lifo`] — the pre-seam behavior was an open-coded `pop_back` /
    /// `push_back`, exactly what `Lifo` reproduces, so the default is
    /// byte-for-byte zero-regression. Shared across worker threads as an `Arc`
    /// (the trait is `Send + Sync`).
    pub(super) policy: Arc<dyn SelectionPolicy>,
    /// The manager's `max_active_states` cap, mirrored onto the parallel
    /// frontier (angr-9ke6b.48). The serial loop bounds `STASH_ACTIVE` in
    /// `RustExplorationManager::push_to_active_or_drop`, but an in-wave /
    /// in-session frontier never round-trips through that stash, so the same
    /// cap is applied by [`worker::absorb_continues`] against `pending`
    /// (queued + in-flight = the resident frontier). `None` = unbounded, which
    /// is what every Rust-side / test construction gets by default; the two
    /// production sites in `run_loop_wave.rs` / `run_loop_steady.rs` thread the manager's value in.
    pub(super) max_active_states: Option<usize>,
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
    pub(super) max_dispatches: Option<u64>,
}

impl WorkTransport {
    /// Build a transport with an explicit selection policy. Callers that want a
    /// `Lifo` default (verbatim pre-seam dispatch) pass `Arc::new(Lifo)`;
    /// find-aware dispatch (angr-1ilq.9) supplies its own policy here.
    pub(super) fn with_policy(policy: Arc<dyn SelectionPolicy>) -> Self {
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
    pub(super) fn dispatch_budget_exhausted(&self) -> bool {
        self.max_dispatches
            .is_some_and(|limit| self.counters.total_dispatches() >= limit)
    }

    /// Record one dispatch on `worker_id`, sampling the schedulable-frontier
    /// width from `pending` (angr-op0dn.13.9). `pending` counts queued +
    /// in-flight states and is only decremented AFTER the step, so the sample
    /// includes the state being dispatched — the same convention as the serial
    /// model's `active + stepped` width in `record_migration_sample`.
    pub(super) fn record_dispatch(&self, worker_id: usize) {
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
    pub(super) fn drain_residual_payloads(&self) -> Vec<StateMigrationPayload> {
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
    pub(super) transport: WorkTransport,
    pub(super) results: Mutex<Vec<StateMigrationPayload>>,
    /// Test-only accumulation of dead-path summaries. Production reads terminals
    /// via [`take_results`](Self::take_results) and counts summaries through
    /// `CoreCounters::record_summaries`; it never reads this vec, so gating it
    /// (and the `worker_loop` write that fills it) behind `cfg(test)` spares the
    /// production wave path a per-batch mutex lock. Only the `#[cfg(test)]`
    /// [`into_results`](Self::into_results) consumes it (angr-1yge9.8 item 7).
    #[cfg(test)]
    pub(super) summaries: Mutex<Vec<TerminalSummary>>,
    /// Initial seed count (the `SchedulerStats::seeds` field).
    pub(super) seeds: usize,
    /// The per-state processor, invoked once per dispatched state.
    pub(super) process: Box<ProcessFn>,
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
