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
//! **Scope: everything in this section is about the shipped `.so`**, which is
//! built with `[profile.release] panic = "abort"` (workspace `Cargo.toml`,
//! angr-1cue). `cargo test` is the documented exception — see *Under `cargo
//! test`* below. For the shipped build that single fact settles the
//! panic-hardening audit (CQ .8) for this module:
//!
//! * **Mutex poisoning cannot happen.** A `Mutex` is only poisoned when a
//!   thread *unwinds* out of a live `MutexGuard`. Under `panic = "abort"` there
//!   is no unwind: a panic while a guard is held aborts the process at the panic
//!   site, before the guard's `Drop` could ever flag the lock. Every
//!   `.lock().expect("… poisoned")` here (results/summaries/done_rx) is
//!   therefore *provably unreachable in that build* — the message names a state
//!   the shipped profile can never produce.
//! * **A worker cannot "die" mid-wave into a live-pool disconnect.** A worker
//!   thread leaves [`worker_thread`](pool::worker_thread) only on `Shutdown`/closed-channel (clean
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
//! reader wants. A forced-poison test is deliberately **not** added — under the
//! shipped profile it cannot observe a Python exception, only an abort, and
//! under `cargo test` it would assert nothing but the cascade described next.
//!
//! ## Under `cargo test` the argument does *not* hold (and that is expected)
//!
//! Cargo forces `panic = "unwind"` on test/bench harness binaries regardless of
//! the profile's `panic` setting — libtest needs to unwind to report a failing
//! test — and no `[profile.test]` override can opt back into abort. So
//! `cargo test --release --lib`, which is what runs this module's
//! `scheduler_tests.rs` / `scheduler_worker_tests.rs`, executes the code under
//! unwind semantics, where poisoning *is* reachable:
//!
//! * A worker that panics while holding the `job.results` / `job.summaries`
//!   guard ([`worker::worker_loop`]'s terminal and summary drains,
//!   `worker::drain_local_into_results`) unwinds out of the live guard and
//!   poisons that lock.
//! * The coordinator then trips the poison on its next lock —
//!   [`WaveJob::take_results`], `WaveJob::into_results`, or the `done_rx`
//!   guard in [`run_wave`](PersistentPool::run_wave) — so the `.expect("…
//!   poisoned")` fires and the test fails with a *second*, cascaded panic.
//!
//! That cascade is acceptable rather than a violated invariant: the poison guard
//! is faithfully reporting a real prior panic, and the shipped build would have
//! aborted at the first one anyway. When debugging such a failure, ignore the
//! `… mutex poisoned` panic and look for the *earliest* panic in the test
//! output — that one is the actual bug.
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
//!
//! # Module layout (angr-9ke6b.59)
//!
//! This file is the module root and keeps only the pieces both scheduling modes
//! and every sibling need: [`CancelToken`], the task-result types
//! ([`TaskOutcome`] / [`TerminalSummary`] / [`TerminalDisposition`]),
//! [`LOCAL_HWM`], and the `#[cfg(test)]` `ParallelScheduler` compat shim. The
//! separable concerns live in `#[path]` children (all `use super::*`, so the
//! imports and the deny above reach them, and every `scheduler::X` path outside
//! this module is unchanged by the re-exports below):
//!
//! * [`stats`] (`scheduler_stats.rs`) — [`SchedulerCounters`] write side,
//!   [`SchedulerStats`] read side, `snapshot_stats` between them.
//! * [`transport`] (`scheduler_transport.rs`) — [`WorkTransport`], the
//!   dispatch/steal/quiescence core shared by both modes, plus the wave-mode
//!   [`WaveJob`] and the boxed [`ProcessFn`].
//! * [`pool`] (`scheduler_pool.rs`) — the steady-state session protocol
//!   ([`WorkerCtl`](pool::WorkerCtl) / [`WorkerUp`] / [`RunSession`]) and the
//!   [`PersistentPool`] thread lifecycle with its
//!   [`worker_thread`](pool::worker_thread) entry point.
//! * [`worker`] (`scheduler_worker.rs`) — the per-worker dispatch/steal/offload
//!   loops the pool's threads run (predates this split, angr-nbim4.2).
//!
//! Because the four are siblings rather than one file, the struct internals they
//! share (`WorkTransport` / `WaveJob` / `RunSession` fields, the counter fields)
//! are `pub(super)` — i.e. still scheduler-module-private, just spelled out.
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
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
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
///
/// Backed by a single atomic `state` rather than two independent
/// `AtomicBool`s (`flag` + `budget`, angr-9ke6b.52 bug fix follow-up): two
/// atomics can be observed in an inconsistent intermediate combination by a
/// concurrent reader — e.g. a find cancel's `budget.store(false)` then
/// `flag.store(true)` are two separate writes, and a `cancel_for_budget()`
/// racing in the gap between them could see `flag == false`, "win" a
/// set-budget-true, and leave the pair at `(flag=true, budget=true)` even
/// though the cancel in flight was a genuine find/finalize. A single atomic
/// makes every observable state one of exactly the three [`CancelState`]
/// values below, with no in-between.
#[derive(Clone, Default)]
pub(crate) struct CancelToken {
    state: Arc<AtomicU8>,
}

/// `state` values for [`CancelToken`]. See the struct doc for why this is a
/// single atomic instead of two independent booleans.
type CancelState = u8;
const NOT_CANCELLED: CancelState = 0;
/// A `run(n)` dispatch-budget stop ([`WorkTransport::max_dispatches`]):
/// already-dispatched states must finish their step (dropping one unstepped
/// livelocks a small `run(n)` — see [`CancelToken::preempts_in_flight`]).
const BUDGET_CANCELLED: CancelState = 1;
/// A find / finalize cancel: peers must drop the state they just picked up
/// (speculative waste past `num_find`) rather than finish stepping it.
const FIND_CANCELLED: CancelState = 2;

impl CancelToken {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Request that all workers stop at their next task boundary, preempting any
    /// in-flight step (find / finalize semantics). Always wins over a
    /// concurrent [`Self::cancel_for_budget`] — an unconditional store, not a
    /// CAS, so a find cancel can never be "lost" to a racing budget stop.
    pub(crate) fn cancel(&self) {
        self.state.store(FIND_CANCELLED, Ordering::SeqCst);
    }

    /// Request that all workers stop at their next task boundary, but let
    /// already-dispatched states finish their step. See [`BUDGET_CANCELLED`].
    ///
    /// Every worker calls this at the top of its loop whenever the wave's
    /// dispatch budget is exhausted, including on the very next iteration a
    /// worker takes right after a sibling's genuine find/finalize `cancel()`
    /// already fired. The CAS only takes effect from `NOT_CANCELLED`, so it
    /// can never downgrade an in-flight `FIND_CANCELLED` back to
    /// `BUDGET_CANCELLED` — closing the race described on [`CancelState`]
    /// that would otherwise silently flip [`Self::preempts_in_flight`] from
    /// true to false mid-wave, breaking the Bug M1 contract
    /// (angr-op0dn.13.8) `cancel` exists to guarantee.
    pub(crate) fn cancel_for_budget(&self) {
        // `drop(..)` — the usual explicit-discard spelling — is itself a
        // no-op-lint error here because `Result<u8, u8>` is `Copy`, so this
        // discard has to opt out by name.
        #[expect(
            clippy::let_underscore_must_use,
            reason = "SILENT(cat-a): a failed CAS means some other reason already cancelled \
                      this wave, which is precisely the outcome the doc comment above \
                      requires — the current value is deliberately left alone"
        )]
        let _ = self.state.compare_exchange(
            NOT_CANCELLED,
            BUDGET_CANCELLED,
            Ordering::SeqCst,
            Ordering::SeqCst,
        );
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.state.load(Ordering::SeqCst) != NOT_CANCELLED
    }

    /// Whether cancellation should preempt a state that is already dispatched.
    /// True for find/finalize cancels, false for a pure budget stop.
    pub(crate) fn preempts_in_flight(&self) -> bool {
        self.state.load(Ordering::SeqCst) == FIND_CANCELLED
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

#[path = "scheduler_stats.rs"]
mod stats;
pub(crate) use stats::{MAX_TRACKED_WORKERS, SchedulerCounters, SchedulerStats};

#[path = "scheduler_transport.rs"]
mod transport;
pub(crate) use transport::{ProcessFn, WaveJob, WorkTransport};

#[path = "scheduler_pool.rs"]
mod pool;
pub(crate) use pool::{PersistentPool, RunSession, WorkerUp};

#[path = "scheduler_worker.rs"]
mod worker;
use worker::{worker_loop, worker_session_loop};

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

test_submod!("scheduler_tests.rs" => tests);
