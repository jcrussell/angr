//! Scheduler unit tests (extracted from `scheduler.rs`, angr-nbim4.2).
//!
//! Included as the `#[cfg(test)] mod tests` body of the parent scheduler module
//! via `#[path]`, so `super::*` resolves to the scheduler module exactly as when
//! this block lived inline. The former inline `use super::{… dispatch_next}` is
//! split out because `dispatch_next` now lives in the `worker` submodule.

use super::worker::dispatch_next;
use super::{
    LOCAL_HWM, MAX_TRACKED_WORKERS, ParallelScheduler, TaskOutcome, TerminalDisposition,
    TerminalSummary, WorkTransport,
};
use crate::exploration::selection_policy::{Fifo, Lifo};
use crate::state::RustSimState;
use crate::symbolic::RustBV;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
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

/// Block until `cond` holds, or give up after ~10s (angr-vplge).
///
/// The parallel post-cancel tests need one task to observe another's progress.
/// A fixed sleep is a race under full-suite load — the sleeper can wake before
/// the event it was waiting for — so wait on the condition itself. The deadline
/// only exists so a genuine scheduler regression fails the assertion instead of
/// hanging the test binary.
fn spin_until(cond: impl Fn() -> bool) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !cond() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

// angr-1ilq.9: the worker-local frontier pop honors the injected
// `SelectionPolicy`. Push three states front-to-back onto one worker's
// `local` queue and drain it via `dispatch_next` (empty injector, so no
// steal path is reached), returning `(insertion_order, dispatch_order)` of
// `state_id`s. Both policies see three FRESH states in the SAME insertion
// order; only the pop policy differs.
fn drain_local_with(policy: &'static str) -> (Vec<u64>, Vec<u64>) {
    let ctx = Context::thread_local();
    let transport = match policy {
        "fifo" => WorkTransport::with_policy(Arc::new(Fifo)),
        _ => WorkTransport::with_policy(Arc::new(Lifo)),
    };
    let mut local: VecDeque<RustSimState> = VecDeque::new();
    let mut inserted = Vec::new();
    for i in 0..3u64 {
        let st = pinned_state(&format!("order_{policy}_{i}"), 0xB000 + i);
        inserted.push(st.state_id());
        local.push_back(st);
    }
    let mut dispatched = Vec::new();
    while let Some(st) = dispatch_next(0, &transport, &mut local, &ctx) {
        dispatched.push(st.state_id());
    }
    (inserted, dispatched)
}

#[test]
fn test_dispatch_next_honors_selection_policy() {
    let (fifo_in, fifo_out) = drain_local_with("fifo");
    assert_eq!(
        fifo_out, fifo_in,
        "Fifo must dispatch the worker-local frontier oldest-first (insertion order)",
    );

    let (lifo_in, lifo_out) = drain_local_with("lifo");
    let mut lifo_expected = lifo_in.clone();
    lifo_expected.reverse();
    assert_eq!(
        lifo_out, lifo_expected,
        "Lifo (the default) must dispatch the worker-local frontier newest-first (reverse)",
    );
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

// angr-1ilq.8 (post-find speculative waste): the FIRST task to start becomes
// the "finder" and trips the shared cancel immediately; every other worker
// that was already in-flight (sleeping) observes that cancel on its next
// return WITHOUT having raised it, so it lands in `post_cancel_steps`. The
// finder itself (request_cancel=true) is EXCLUDED — the counter measures
// wasted peer work, not the find. Proves the counter increments exactly on
// cross-worker speculative steps and never on the finder's own step.
#[test]
fn test_post_cancel_steps_counts_peer_speculation() {
    const N: u64 = 8;
    let run_order = Arc::new(AtomicUsize::new(0));

    let mut payloads = Vec::with_capacity(N as usize);
    for i in 0..N {
        payloads.push(pinned_state(&format!("spec_{i}"), 0x2000 + i).detach_for_migration());
    }

    let sched = ParallelScheduler::new(4);
    let (_collected, _summaries, stats) = sched.run_instrumented(payloads, {
        let run_order = Arc::clone(&run_order);
        move |state, cancel, _cache| {
            let request_cancel = if run_order.fetch_add(1, Ordering::SeqCst) == 0 {
                // Finder. Hold until a peer is past the worker's task-boundary
                // cancel check (i.e. actually in-flight) — a worker never
                // dispatches a new task once the token is tripped, so a cancel
                // raised before any peer starts leaves nothing to speculate.
                spin_until(|| run_order.load(Ordering::SeqCst) >= 2);
                true
            } else {
                // Peer: hold until the finder's cancel is visible, then return a
                // normal terminal (does NOT self-cancel) — that step is
                // speculative by construction.
                spin_until(|| cancel.is_cancelled());
                false
            };
            TaskOutcome {
                continue_states: Vec::new(),
                terminal_states: vec![state],
                terminal_summaries: Vec::new(),
                request_cancel,
            }
        }
    });

    assert!(
        stats.post_cancel_steps >= 1,
        "at least one peer must commit a step after the finder cancelled \
             (post_cancel_steps={})",
        stats.post_cancel_steps,
    );
    // The finder's own terminating step is never counted, so at most (N-1)
    // peer steps can be speculative.
    assert!(
        stats.post_cancel_steps < N as usize,
        "the finder's own step must be excluded (post_cancel_steps={})",
        stats.post_cancel_steps,
    );
}

// angr-1ilq.8 invariant: when EVERY step self-cancels, none is speculative —
// each processed step raised its own cancel, so the `!request_cancel`
// exclusion keeps `post_cancel_steps` at zero.
#[test]
fn test_post_cancel_steps_excludes_self_cancel() {
    const N: u64 = 256;
    let mut payloads = Vec::with_capacity(N as usize);
    for i in 0..N {
        payloads.push(pinned_state(&format!("self_{i}"), 0x3000 + i).detach_for_migration());
    }
    let sched = ParallelScheduler::new(4);
    let (_collected, _summaries, stats) =
        sched.run_instrumented(payloads, |state, _c, _cache| TaskOutcome {
            continue_states: Vec::new(),
            terminal_states: vec![state],
            terminal_summaries: Vec::new(),
            request_cancel: true,
        });
    assert_eq!(
        stats.post_cancel_steps, 0,
        "self-cancelling steps are the finder, not speculative waste",
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

use super::{PersistentPool, RunSession, WaveJob, WorkerUp};
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

// angr-op0dn.13.8 (Bug M1): the WAVE loop drains its residual frontier on
// cancel too — the twin of test_session_cancel_drains_residual_frontier. A root
// fans out WIDTH children; processing any child requests cancel, so whichever
// worker holds the sibling backlog drains it into `results` and the coordinator
// pulls the never-stolen injector surplus after the barrier. Conservation is
// EXACT: every child comes back as a processed terminal or a residual.
#[test]
fn test_wave_cancel_drains_residual_frontier() {
    const WIDTH: usize = 40;
    const WITNESS: u64 = 0xF2F2;
    const WORKERS: usize = 2;
    let main_ctx = Context::thread_local();

    let mut root = pinned_state("wave_drain_root", WITNESS);
    root.set_register("rbx", RustBV::concrete(0, 64));

    let pool = PersistentPool::new(WORKERS);
    let job = WaveJob::new(
        vec![root.detach_for_migration()],
        Box::new(
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
                TaskOutcome {
                    continue_states: Vec::new(),
                    terminal_states: vec![state],
                    terminal_summaries: Vec::new(),
                    request_cancel: true,
                }
            },
        ),
    );

    let (job, _barrier_stats) = pool.run_wave(job);
    let mut recovered = job.take_results();
    recovered.extend(job.drain_residual_payloads());

    let stats = job.stats();
    assert!(
        stats.residual_drains > 0,
        "a cancelled wave must drain a residual frontier (got {} terminals, {} residual)",
        stats.materialized_terminals,
        stats.residual_drains,
    );
    assert_eq!(
        stats.materialized_terminals + stats.residual_drains,
        WIDTH,
        "no child lost or duplicated across the wave cancel",
    );
    assert_eq!(recovered.len(), WIDTH, "every child crosses the join");

    for payload in recovered {
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

// angr-op0dn.13.9: the real dispatch path must record BOTH the per-worker
// dispatch vector and the frontier-width histogram, so the steady/wave loops
// stop reporting width 0 (previously only the serial migration model sampled
// width, leaving frontier-residency mode blind to the width audit and the S7
// find-all gate's states/worker balance column).
#[test]
fn test_dispatch_records_width_and_per_worker_counts() {
    const DEPTH: u64 = 6; // 64 leaves, 127 tasks — plenty of frontier to widen
    let mut root = pinned_state("width_acc", 0xF00D);
    root.set_register("rbx", RustBV::concrete(0, 64));

    let sched = ParallelScheduler::new(4);
    let (collected, _summaries, stats) =
        sched.run_instrumented(vec![root.detach_for_migration()], |state, _cancel, _c| {
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
    assert_eq!(collected.len(), 1usize << DEPTH, "no lost/duplicated work");

    let dispatches = stats.dispatches();
    assert_eq!(
        stats.worker_dispatches.len(),
        MAX_TRACKED_WORKERS,
        "the per-worker vector is fixed-size; the run loop trims it",
    );
    assert_eq!(
        stats.worker_dispatches.iter().sum::<usize>(),
        dispatches,
        "every dispatch must be attributed to exactly one worker",
    );
    assert_eq!(
        stats.width_hist.iter().sum::<usize>(),
        dispatches,
        "every dispatch must land in exactly one width bucket",
    );
    assert!(
        stats.max_width >= 2,
        "a 2^{DEPTH} fork tree must widen the schedulable frontier past 1, got {}",
        stats.max_width,
    );
    assert!(
        stats.width_hist[1..].iter().sum::<usize>() > 0,
        "sustained width must be recorded, not just the peak: {:?}",
        stats.width_hist,
    );
}

/// angr-pwu71: `record_summaries` used to `continue` on `Avoided`, bumping the
/// total but no split slot, so a worker-avoided terminal vanished from
/// `stats()["avoided_count"]`. Every disposition must land in exactly one slot,
/// and the four slots must reconstruct the total.
#[test]
fn record_summaries_splits_every_disposition_including_avoided() {
    let counters = super::SchedulerCounters::default();
    let summary = |id: u64, disposition| TerminalSummary {
        state_id: id,
        pc: 0x400000 + id,
        disposition,
    };
    counters.record_summaries(&[
        summary(1, TerminalDisposition::Deadended),
        summary(2, TerminalDisposition::Errored),
        summary(3, TerminalDisposition::Pruned),
        summary(4, TerminalDisposition::Avoided),
        summary(5, TerminalDisposition::Avoided),
    ]);

    let stats = super::snapshot_stats(0, &counters);
    assert_eq!(stats.summarized_terminals, 5);
    assert_eq!(stats.summarized_deadended, 1);
    assert_eq!(stats.summarized_errored, 1);
    assert_eq!(stats.summarized_pruned, 1);
    assert_eq!(
        stats.summarized_avoided, 2,
        "worker-summarized Avoided terminals must reach the manager's avoided_count",
    );
    assert_eq!(
        stats.summarized_deadended
            + stats.summarized_errored
            + stats.summarized_pruned
            + stats.summarized_avoided,
        stats.summarized_terminals,
        "the per-disposition split must reconstruct the total — no silently dropped arm",
    );
}
