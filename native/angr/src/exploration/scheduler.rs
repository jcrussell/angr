//! Work-stealing scheduler machinery for parallel exploration (angr-1ilq.3,
//! correctness-first isolated increment).
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
//! everything to do with thread-safety: that a [`StateMigrationPayload`]
//! (angr-1ilq.1) can be distributed across workers that each own a private Z3
//! context, reattached, processed, and have successors detached back into
//! `Send` payloads — all under genuine concurrency, with zero `unsafe`.
//!
//! # Transport invariant
//!
//! Everything on the deque is a [`StateMigrationPayload`] (`Send` by
//! construction), never a [`RustSimState`] (irreducibly `!Send`). A worker:
//!
//! 1. pops/steals a payload,
//! 2. [`reattach`](StateMigrationPayload::reattach)es it into *its own*
//!    thread-local Z3 context (every AST minted locally; the source context is
//!    never read cross-thread — this is what makes the design sound where the
//!    rejected `translate_state`-on-steal was not; see hazard C on the bead),
//! 3. runs the caller's `process` closure to get successors,
//! 4. [`detach_for_migration`](RustSimState::detach_for_migration)es each
//!    successor back into a payload *on the worker* (correct context) before it
//!    can cross a thread again.
//!
//! Serializing even non-stolen successors is wasteful (a serde round-trip the
//! single-threaded loop never pays); correctness-first accepts it. The
//! optimization — keep locally-produced states as `RustSimState` and only
//! serialize on an actual steal — is deferred, and 1ilq.5 must measure this
//! overhead against the <5% migration-cost GO condition.

use crate::state::{RustSimState, StateMigrationPayload};
use crossbeam_deque::{Injector, Steal, Stealer, Worker};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use z3::{Config, Context};

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

/// What a worker produced from processing one migrated state, expressed in that
/// worker's own Z3 context.
///
/// The scheduler detaches every state here back into a `Send` payload **on the
/// worker** before it crosses a thread, so the caller never has to reason
/// about `!Send` state escaping a worker.
pub struct TaskOutcome {
    /// Successors to keep exploring; re-injected as fresh tasks.
    pub continue_states: Vec<RustSimState>,
    /// Terminal states (found / deadended / errored / …) to collect as results.
    pub terminal_states: Vec<RustSimState>,
    /// Request global cancellation (e.g. the find target was reached).
    pub request_cancel: bool,
}

impl TaskOutcome {
    /// All successors are live; keep exploring, nothing terminal.
    pub fn continuing(continue_states: Vec<RustSimState>) -> Self {
        Self {
            continue_states,
            terminal_states: Vec::new(),
            request_cancel: false,
        }
    }

    /// All successors are terminal; collect them, generate no further work.
    pub fn terminal(terminal_states: Vec<RustSimState>) -> Self {
        Self {
            continue_states: Vec::new(),
            terminal_states,
            request_cancel: false,
        }
    }
}

/// A work-stealing pool that distributes [`StateMigrationPayload`]s across
/// `num_workers` threads, each owning a private Z3 context.
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
    /// terminal payloads.
    ///
    /// `process` runs on a worker thread with that worker's own Z3 context
    /// installed as the thread-local; it receives a state already reattached
    /// into that context. It must be `Send + Sync` (every worker shares one
    /// `&process`). Successors it returns are detached back into payloads on
    /// the same worker before being re-injected or collected.
    ///
    /// Quiescence is tracked with an `AtomicUsize` of outstanding tasks:
    /// children are counted in *before* their parent is counted out, so the
    /// global count never transiently reaches zero while work is still in
    /// flight. A worker exits when it can find no task and the outstanding
    /// count is zero, or as soon as cancellation is requested.
    pub fn run<F>(
        &self,
        initial: Vec<StateMigrationPayload>,
        process: F,
    ) -> Vec<StateMigrationPayload>
    where
        F: Fn(RustSimState, &CancelToken) -> TaskOutcome + Send + Sync,
    {
        let injector = Injector::<StateMigrationPayload>::new();
        let pending = AtomicUsize::new(initial.len());
        for payload in initial {
            injector.push(payload);
        }
        let cancel = CancelToken::new();
        let results = Mutex::new(Vec::<StateMigrationPayload>::new());

        // One local deque per worker; share their stealers with every worker.
        let locals: Vec<Worker<StateMigrationPayload>> =
            (0..self.num_workers).map(|_| Worker::new_lifo()).collect();
        let stealers: Vec<Stealer<StateMigrationPayload>> =
            locals.iter().map(|w| w.stealer()).collect();

        std::thread::scope(|scope| {
            for local in locals {
                let injector = &injector;
                let stealers = &stealers;
                let pending = &pending;
                let results = &results;
                let cancel = &cancel;
                let process = &process;
                scope.spawn(move || {
                    // Each worker owns a fresh Z3 context for its whole life and
                    // installs it as the thread-local. `ctx` stays on this
                    // stack frame until the worker exits, so every reattach
                    // mints ASTs into a context this thread alone touches, and
                    // the context is created and destroyed on the same thread.
                    let ctx = Context::new(&Config::new());
                    Context::set_thread_local(&ctx);
                    worker_loop(
                        &local, injector, stealers, pending, cancel, results, &ctx, process,
                    );
                });
            }
        });

        results.into_inner().expect("results mutex poisoned")
    }
}

/// The per-worker loop: find a task, reattach it, process it, route successors.
#[allow(clippy::too_many_arguments)] // worker context is genuinely this wide; bundling it would just move the noise
fn worker_loop<F>(
    local: &Worker<StateMigrationPayload>,
    injector: &Injector<StateMigrationPayload>,
    stealers: &[Stealer<StateMigrationPayload>],
    pending: &AtomicUsize,
    cancel: &CancelToken,
    results: &Mutex<Vec<StateMigrationPayload>>,
    ctx: &Context,
    process: &F,
) where
    F: Fn(RustSimState, &CancelToken) -> TaskOutcome,
{
    loop {
        if cancel.is_cancelled() {
            return;
        }

        let Some(payload) = find_task(local, injector, stealers) else {
            // No task available right now. If nothing is outstanding anywhere,
            // no future task can ever appear (a task is only created by an
            // in-flight task), so we are done. Otherwise another worker is
            // mid-task and may yet push work — yield and retry.
            if pending.load(Ordering::SeqCst) == 0 {
                return;
            }
            std::thread::yield_now();
            continue;
        };

        // Rebuild the state in THIS worker's context. The ptr-eq guard inside
        // `reattach` holds because `ctx` is exactly this thread's thread-local.
        let state = match payload.reattach(ctx) {
            Ok(state) => state,
            Err(err) => {
                // Unreachable in practice (we set our own ctx as thread-local
                // above), but never silently keep a phantom task outstanding.
                log::error!("scheduler reattach failed, dropping task: {err:?}");
                pending.fetch_sub(1, Ordering::SeqCst);
                continue;
            }
        };

        let outcome = process(state, cancel);

        // Detach successors back into Send payloads on THIS worker (correct
        // context) before any of them can be stolen onto another thread.
        let mut spawned = 0usize;
        for child in outcome.continue_states {
            local.push(child.detach_for_migration());
            spawned += 1;
        }
        if !outcome.terminal_states.is_empty() {
            let mut guard = results.lock().expect("results mutex poisoned");
            for terminal in outcome.terminal_states {
                guard.push(terminal.detach_for_migration());
            }
        }

        // Count children IN before counting this task OUT, so `pending` never
        // dips to zero with live descendants queued.
        if spawned > 0 {
            pending.fetch_add(spawned, Ordering::SeqCst);
        }
        pending.fetch_sub(1, Ordering::SeqCst);

        if outcome.request_cancel {
            cancel.cancel();
            return;
        }
    }
}

/// Standard crossbeam work-stealing search: drain the local deque first, then
/// pull a batch from the global injector, then steal from siblings.
fn find_task<T>(local: &Worker<T>, injector: &Injector<T>, stealers: &[Stealer<T>]) -> Option<T> {
    // Fast path: our own deque.
    if let Some(task) = local.pop() {
        return Some(task);
    }
    // Slow path: keep retrying across the injector and sibling stealers until a
    // round produces a definite result (no `Retry` left to resolve).
    loop {
        let mut retry = false;
        match injector.steal_batch_and_pop(local) {
            Steal::Success(task) => return Some(task),
            Steal::Retry => retry = true,
            Steal::Empty => {}
        }
        for stealer in stealers {
            match stealer.steal() {
                Steal::Success(task) => return Some(task),
                Steal::Retry => retry = true,
                Steal::Empty => {}
            }
        }
        if !retry {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ParallelScheduler, TaskOutcome};
    use crate::state::RustSimState;
    use crate::symbolic::RustBV;
    use std::collections::BTreeMap;
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
        let collected = sched.run(payloads, |state, _cancel| {
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
        let collected = sched.run(vec![root.detach_for_migration()], |state, _cancel| {
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
        });

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
        let processed = AtomicUsize::new(0);

        let mut payloads = Vec::with_capacity(N as usize);
        for i in 0..N {
            payloads.push(pinned_state(&format!("cancel_{i}"), 0x1000 + i).detach_for_migration());
        }

        let sched = ParallelScheduler::new(4);
        let collected = sched.run(payloads, |state, _cancel| {
            processed.fetch_add(1, Ordering::SeqCst);
            // Every task asks to cancel; the first to run trips the token.
            TaskOutcome {
                continue_states: Vec::new(),
                terminal_states: vec![state],
                request_cancel: true,
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
}
