//! Scheduler metrics: the shared atomic counters and their read-back snapshot
//! (extracted from `scheduler.rs`, angr-9ke6b.59).
//!
//! [`SchedulerCounters`] is the write side — plain atomics bumped from every
//! worker thread during a wave/session — and [`SchedulerStats`] the read side,
//! a plain-old-data snapshot taken by [`snapshot_stats`] once the workers are
//! quiesced. Both structs and the snapshot fn are generated from ONE field list
//! by the `scheduler_counters!` macro below, so adding a counter is a
//! single-line edit that cannot leave a stat silently stuck at zero. See the
//! parent module for the transport invariant the steal-fraction counters
//! measure.
//!
//! This is a `#[path]` child of [`scheduler`](super), so the parent's
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]` and its `use` block both
//! reach here (the latter via the glob below).
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::*;

/// Per-worker dispatch slots tracked by [`SchedulerCounters::worker_dispatches`].
/// 32 covers every realistic `RUST_PARALLEL_WORKERS`; higher ids fold into the
/// last slot rather than allocating (the counters live on a hot path and
/// `Default`-derived arrays cap at 32).
pub(crate) const MAX_TRACKED_WORKERS: usize = 32;

/// Declare the scheduler's counter set ONCE and generate all three views of it:
/// the atomic write side ([`SchedulerCounters`]), the plain-old-data read side
/// ([`SchedulerStats`]) and the [`snapshot_stats`] fn that loads one into the
/// other.
///
/// Before angr-12jjk.1 these were three hand-maintained parallel field lists;
/// adding a counter to the first two but forgetting the third produced a stat
/// permanently stuck at zero with no compile error. Now a new counter is one
/// line here, and omitting it is not expressible.
///
/// Each field is `<vis> <name>: <kind>`, where `<vis>` is the visibility on the
/// *atomic* struct (the POD mirror is always `pub`) and `<kind>` is one of:
///
/// - `count` — `AtomicUsize` / `usize`
/// - `nanos` — `AtomicU64` / `u64`
/// - `per_worker` — `[AtomicUsize; MAX_TRACKED_WORKERS]` / `Vec<usize>`
/// - `hist` — `[AtomicUsize; 5]` / `[usize; 5]` (the shared width buckets)
///
/// Doc comments on a field are emitted on both sides, so each counter is
/// documented in exactly one place. `seeds` is the one POD-only field (it is a
/// constructor argument, never an atomic) and is hardcoded in the expansion.
macro_rules! scheduler_counters {
    // Per-kind type/loader mappings. Kept first so the field-list rule below is
    // never offered an `@`-prefixed invocation.
    (@atomic count) => { AtomicUsize };
    (@pod count) => { usize };
    (@snap count, $field:expr) => { $field.load(Ordering::SeqCst) };

    (@atomic nanos) => { AtomicU64 };
    (@pod nanos) => { u64 };
    (@snap nanos, $field:expr) => { $field.load(Ordering::SeqCst) };

    (@atomic per_worker) => { [AtomicUsize; MAX_TRACKED_WORKERS] };
    (@pod per_worker) => { Vec<usize> };
    (@snap per_worker, $field:expr) => {
        $field.iter().map(|c| c.load(Ordering::SeqCst)).collect()
    };

    (@atomic hist) => { [AtomicUsize; 5] };
    (@pod hist) => { [usize; 5] };
    (@snap hist, $field:expr) => {
        std::array::from_fn(|i| $field[i].load(Ordering::SeqCst))
    };

    (
        $(
            $(#[$fmeta:meta])*
            $vis:vis $name:ident : $kind:ident
        ),* $(,)?
    ) => {
        /// Counters accumulated across all workers during a run. Shared as
        /// atomics, read back into a [`SchedulerStats`] after the pool joins.
        ///
        /// `pub(crate)` so the coordinator's duplex-protocol `RunSession`
        /// (scheduler_pool.rs) can own one directly (angr-vh834 steady-state
        /// redesign, Phase 1); run_loop_steady.rs's `SteadySession` reaches it
        /// through an `Arc<RunSession>`. The scheduler still constructs and
        /// reads it here, so every *field* stays `pub(super)`.
        #[derive(Default)]
        pub(crate) struct SchedulerCounters {
            $(
                $(#[$fmeta])*
                $vis $name: scheduler_counters!(@atomic $kind),
            )*
        }

        /// Per-run accounting, the basis for the overhead-gate steal-fraction
        /// check.
        ///
        /// The only states that pay the serde / Z3-AST-rebuild tax are
        /// `surplus_offloaded` (cross-worker steals) plus
        /// `materialized_terminals` (found states recovered across the join).
        /// `SchedulerStats::honest_steal_fraction` reports that as a fraction of
        /// non-seed dispatches — the quantity the gate's break-even `f*` bounds.
        #[derive(Clone, Debug, Default, PartialEq, Eq)]
        pub(crate) struct SchedulerStats {
            /// Initial payloads handed to the `#[cfg(test)]`
            /// `ParallelScheduler::run_instrumented` shim.
            pub seeds: usize,
            $(
                $(#[$fmeta])*
                pub $name: scheduler_counters!(@pod $kind),
            )*
        }

        /// Snapshot a [`SchedulerCounters`] into a [`SchedulerStats`] — shared
        /// by [`WaveJob::stats`] (post-barrier) and [`RunSession::stats`] (any
        /// time; the atomics make a mid-session snapshot merely slightly stale,
        /// never torn).
        pub(super) fn snapshot_stats(seeds: usize, counters: &SchedulerCounters) -> SchedulerStats {
            SchedulerStats {
                seeds,
                $( $name: scheduler_counters!(@snap $kind, counters.$name), )*
            }
        }
    };
}

scheduler_counters! {
    /// Tasks dispatched from a worker's local live queue (zero serde).
    pub(super) local_dispatches: count,
    /// Tasks pulled from the injector (each paid one `reattach`). Includes the
    /// initial seeds.
    pub(super) injector_dispatches: count,
    /// Continue-states detached to the injector on imbalance. The migration
    /// count and the *only* continue-path serde site.
    pub(super) surplus_offloaded: count,
    /// Found/matched terminals serialized across the join.
    pub(super) materialized_terminals: count,
    /// Dead-path terminals recorded as summaries (no serde).
    pub(super) summarized_terminals: count,
    /// Per-disposition split of `summarized_terminals` (angr-op0dn.13.15). The
    /// summarized states themselves are dropped in-worker, but their *counts*
    /// must still reach the manager's `deadended_count` / `pruned_count` /
    /// `errored_count` / `avoided_count`, or a parallel run reports zero dead
    /// paths where the serial loop reports N. The coordinator folds these four
    /// in so terminal accounting is worker-count invariant. `Avoided` DOES
    /// occur here (angr-pwu71): avoid-routing is usually a coordinator
    /// decision, but `parallel_process_state` also matches a successor's pc
    /// against `avoid_addrs` in-worker, so the fourth slot is required.
    pub(super) summarized_deadended: count,
    pub(super) summarized_errored: count,
    pub(super) summarized_pruned: count,
    pub(super) summarized_avoided: count,
    /// Payloads pulled from the injector and `reattach`ed into a worker's own Z3
    /// context (the injector-steal path). Observability only — equals
    /// `injector_dispatches` today; the two diverge once the steady-state
    /// coordinator reattaches on paths other than an injector steal (Phase 2+).
    pub(super) reattaches: count,
    /// Bounce states that made a full worker->coordinator->worker round trip.
    /// Wired by the steady-state coordinator (run_loop_steady.rs); 0 in wave mode.
    pub(super) bounce_roundtrips: count,
    /// States re-injected after a Python resume callback
    /// ([`RunSession::inject_resumed`]); 0 in wave mode.
    pub(super) resume_reinjects: count,
    /// Still-live frontier states a worker detached and handed back to the
    /// coordinator on cancel/finalize instead of dropping them (the Bug M1
    /// cancel-drain: [`worker::drain_local_upstream`] in session mode, the
    /// wave loop's `results` push in wave mode), PLUS the never-stolen injector
    /// surplus the coordinator pulls with [`WorkTransport::drain_residual_payloads`].
    /// Distinct from `materialized_terminals` so residual drains never pollute
    /// the honest steal fraction. 0 unless the run was cancelled.
    pub(super) residual_drains: count,
    /// Steps a worker executed AFTER cancellation was already requested by a
    /// *different* origin (a peer worker's find or the coordinator's num_find
    /// route) — the post-find speculative waste (angr-1ilq.8). A step counts
    /// here when `cancel` is set on `process` return but this step did not
    /// itself raise the cancel (`!request_cancel`). Because both worker loops
    /// re-check `cancel` at the top of every iteration, this captures exactly
    /// the in-flight steps that were committed before the cancel became
    /// visible: the wasted work the find-aware dispatch bead (angr-1ilq.9)
    /// aims to eliminate. 0 until the first find raises a cancel.
    pub(super) post_cancel_steps: count,
    /// Dispatches per worker id (angr-op0dn.13.9): the load-balance column the
    /// S7 find-all gate needs (`max/min dispatched per worker`). Indexed by
    /// `worker_id`; ids >= [`MAX_TRACKED_WORKERS`] fold into the last slot
    /// (documented lossiness — real runs use <= num_cpus workers). When that
    /// fold does happen it is reported by `folded_worker_dispatches` rather
    /// than left silent (angr-9ke6b.67). The snapshot is always
    /// [`MAX_TRACKED_WORKERS`] long; the run loop trims it to the configured
    /// worker count before exporting.
    pub(super) worker_dispatches: per_worker,
    /// Dispatches that landed on a worker id >= [`MAX_TRACKED_WORKERS`] and were
    /// therefore folded into the last `worker_dispatches` slot (angr-9ke6b.67).
    /// Non-zero means the per-worker load-balance column is degraded — the tail
    /// workers are merged into one bucket, so its max/min ratio is not
    /// trustworthy. Also drives a one-time `log::warn!` on the first fold so a
    /// run producing degraded data says so on stderr.
    pub(super) folded_worker_dispatches: count,
    /// Step-weighted schedulable-frontier width, bucketed as
    /// `[==1, ==2, 3-4, 5-8, >=9]` — the parallel-path analogue of the serial
    /// model's `parallel_width_hist` (`record_migration_sample` in helpers.rs).
    /// Sampled at dispatch from `pending` (queued + in-flight states), so both
    /// worker loops record it and the frontier-residency mode is no longer
    /// invisible to the width audit.
    pub(super) width_hist: hist,
    /// Peak `pending` observed at dispatch — the parallel `max_active_width`.
    pub(super) max_width: count,
    /// Nanoseconds all workers spent inside Z3 migration serde — every
    /// `detach_for_migration` and every `reattach` on the steal path. Paired with
    /// `step_ns` to form the serde budget that gates Trigger A offloads
    /// (`worker::offload_is_affordable`, angr-8shhe). Surfaced on the snapshot so
    /// a migration-dominated frontier is diagnosable from the wave log rather
    /// than from a flamegraph (angr-faorh/8shhe).
    pub(super) serde_ns: nanos,
    /// Nanoseconds all workers spent inside the step function itself — the useful
    /// work the serde is a tax on.
    pub(super) step_ns: nanos,
}

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
