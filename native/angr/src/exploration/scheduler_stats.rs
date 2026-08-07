//! Scheduler metrics: the shared atomic counters and their read-back snapshot
//! (extracted from `scheduler.rs`, angr-9ke6b.59).
//!
//! [`SchedulerCounters`] is the write side — plain atomics bumped from every
//! worker thread during a wave/session — and [`SchedulerStats`] the read side,
//! a plain-old-data snapshot taken by [`snapshot_stats`] once the workers are
//! quiesced. Keeping the two together isolates the "add a counter" edit (which
//! touches the struct, the recorder, the snapshot struct and the snapshot fn in
//! lockstep) to one file. See the parent module for the transport invariant the
//! steal-fraction counters measure.
//!
//! This is a `#[path]` child of [`scheduler`](super), so the parent's
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]` and its `use` block both
//! reach here (the latter via the glob below).
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::*;

/// Counters accumulated across all workers during a run. Shared as atomics, read
/// back into a [`SchedulerStats`] after the pool joins.
///
/// `pub(crate)` so the coordinator's forthcoming duplex-protocol `RunSession`
/// (run_loop_steady.rs) can own one directly (angr-vh834 steady-state redesign, Phase
/// 1). The scheduler still constructs and reads it here.
#[derive(Default)]
pub(crate) struct SchedulerCounters {
    pub(super) local_dispatches: AtomicUsize,
    pub(super) injector_dispatches: AtomicUsize,
    pub(super) surplus_offloaded: AtomicUsize,
    pub(super) materialized_terminals: AtomicUsize,
    pub(super) summarized_terminals: AtomicUsize,
    /// Per-disposition split of `summarized_terminals` (angr-op0dn.13.15). The
    /// summarized states themselves are dropped in-worker, but their *counts*
    /// must still reach the manager's `deadended_count` / `pruned_count` /
    /// `errored_count` / `avoided_count`, or a parallel run reports zero dead
    /// paths where the serial loop reports N. `Avoided` DOES occur here
    /// (angr-pwu71): avoid-routing is usually a coordinator decision, but
    /// `parallel_process_state` also matches a successor's pc against
    /// `avoid_addrs` in-worker, so the fourth slot is required.
    pub(super) summarized_deadended: AtomicUsize,
    pub(super) summarized_errored: AtomicUsize,
    pub(super) summarized_pruned: AtomicUsize,
    pub(super) summarized_avoided: AtomicUsize,
    /// Payloads pulled from the injector and `reattach`ed into a worker's own Z3
    /// context (the injector-steal path). Observability only — equals
    /// `injector_dispatches` today; the two diverge once the steady-state
    /// coordinator reattaches on paths other than an injector steal (Phase 2+).
    pub(super) reattaches: AtomicUsize,
    /// Bounce states that made a full worker->coordinator->worker round trip.
    /// Wired by the steady-state coordinator (run_loop_steady.rs); 0 in wave mode.
    pub(crate) bounce_roundtrips: AtomicUsize,
    /// States re-injected after a Python resume callback
    /// ([`RunSession::inject_resumed`]); 0 in wave mode.
    pub(super) resume_reinjects: AtomicUsize,
    /// Still-live frontier states a worker detached and handed back to the
    /// coordinator on cancel/finalize instead of dropping them (the Bug M1
    /// cancel-drain: [`worker::drain_local_upstream`] in session mode, the
    /// wave loop's `results` push in wave mode), PLUS the never-stolen injector
    /// surplus the coordinator pulls with [`WorkTransport::drain_residual_payloads`].
    /// Distinct from `materialized_terminals` so residual drains never pollute
    /// the honest steal fraction. 0 unless the run was cancelled.
    pub(super) residual_drains: AtomicUsize,
    /// Steps a worker executed AFTER cancellation was already requested by a
    /// *different* origin (a peer worker's find or the coordinator's num_find
    /// route) — the post-find speculative waste (angr-1ilq.8). A step counts
    /// here when `cancel` is set on `process` return but this step did not
    /// itself raise the cancel (`!request_cancel`). Because both worker loops
    /// re-check `cancel` at the top of every iteration, this captures exactly
    /// the in-flight steps that were committed before the cancel became
    /// visible: the wasted work the find-aware dispatch bead (angr-1ilq.9)
    /// aims to eliminate. 0 until the first find raises a cancel.
    pub(super) post_cancel_steps: AtomicUsize,
    /// Dispatches per worker id (angr-op0dn.13.9): the load-balance column the
    /// S7 find-all gate needs (`max/min dispatched per worker`). Indexed by
    /// `worker_id`; ids >= [`MAX_TRACKED_WORKERS`] fold into the last slot
    /// (documented lossiness — real runs use <= num_cpus workers). When that
    /// fold does happen it is reported by [`Self::folded_worker_dispatches`]
    /// rather than left silent (angr-9ke6b.67).
    pub(super) worker_dispatches: [AtomicUsize; MAX_TRACKED_WORKERS],
    /// Dispatches that landed on a worker id >= [`MAX_TRACKED_WORKERS`] and were
    /// therefore folded into the last [`Self::worker_dispatches`] slot
    /// (angr-9ke6b.67). Non-zero means the per-worker load-balance column is
    /// degraded — the tail workers are merged into one bucket, so its max/min
    /// ratio is not trustworthy. Also drives a one-time `log::warn!` on the
    /// first fold so a run producing degraded data says so on stderr.
    pub(super) folded_worker_dispatches: AtomicUsize,
    /// Step-weighted schedulable-frontier width, bucketed as
    /// `[==1, ==2, 3-4, 5-8, >=9]` — the parallel-path analogue of the serial
    /// model's `parallel_width_hist` (`record_migration_sample` in helpers.rs).
    /// Sampled at dispatch from `pending` (queued + in-flight states), so both
    /// worker loops record it and the frontier-residency mode is no longer
    /// invisible to the width audit.
    pub(super) width_hist: [AtomicUsize; 5],
    /// Peak `pending` observed at dispatch — the parallel `max_active_width`.
    pub(super) max_width: AtomicUsize,
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
    pub(super) fn total_dispatches(&self) -> u64 {
        (self.local_dispatches.load(Ordering::Relaxed)
            + self.injector_dispatches.load(Ordering::Relaxed)) as u64
    }

    /// Record one dispatch on `worker_id` observing `width` schedulable states
    /// (queued + in-flight, including the state being dispatched). Two relaxed
    /// bumps plus a max-CAS: cheap enough for the dispatch hot path.
    ///
    /// Buckets come from the shared `exploration::width_bucket`, the same fn the
    /// serial model's `record_migration_sample` (helpers.rs) uses, so the two
    /// width histograms cannot drift out of comparability.
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
        self.width_hist[crate::exploration::width_bucket(width as u64)]
            .fetch_add(1, Ordering::Relaxed);
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
pub(super) fn snapshot_stats(seeds: usize, counters: &SchedulerCounters) -> SchedulerStats {
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
