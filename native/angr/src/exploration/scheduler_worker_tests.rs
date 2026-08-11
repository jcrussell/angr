//! Unit tests for the worker-local offload policy (angr-ph300.15).
//!
//! Included as the `#[cfg(test)] mod tests` body of `scheduler_worker.rs` via
//! `#[path]`, so `super::*` reaches both the worker helpers and — as a
//! descendant of the scheduler module — the private `SchedulerCounters` fields.
//!
//! `scheduler_tests.rs` exercises the pool end to end, where the serde budget is
//! whatever the workload happens to produce. These drive `offload_surplus`
//! directly with the budget counters preset, so both sides of the Trigger A
//! affordability transition — and Trigger B's independence from it — are pinned.

use super::*;
use crate::exploration::selection_policy::{Fifo, default_policy};

/// A minimal migratable state. `offload_surplus` only detaches and counts, so
/// no registers or constraints are needed.
fn plain_state() -> RustSimState {
    RustSimState::new("amd64").expect("amd64 state")
}

/// Run `offload_surplus` over `n` local states with the budget preset to
/// (`serde_ns`, `step_ns`) and `idle` starving siblings. Returns
/// `(local_len_after, surplus_offloaded)`.
fn run_offload(n: usize, idle: usize, serde_ns: u64, step_ns: u64) -> (usize, usize) {
    let policy: Arc<dyn SelectionPolicy> = Arc::new(Lifo);
    let mut local: VecDeque<RustSimState> = (0..n).map(|_| plain_state()).collect();
    let injector: Injector<StateMigrationPayload> = Injector::new();
    let idle_workers = AtomicUsize::new(idle);
    let counters = SchedulerCounters::default();
    counters.serde_ns.store(serde_ns, Ordering::Relaxed);
    counters.step_ns.store(step_ns, Ordering::Relaxed);

    offload_surplus(&mut local, &injector, &idle_workers, &counters, &policy);

    (
        local.len(),
        counters.surplus_offloaded.load(Ordering::SeqCst),
    )
}

// (a) Under budget: Trigger A sheds one state per starving sibling.
#[test]
fn test_trigger_a_fires_under_serde_budget() {
    // serde * 4 = 400 <= 10_000 step ns → affordable.
    let (len, offloaded) = run_offload(4, 2, 100, 10_000);
    assert_eq!(offloaded, 2, "one state per starving sibling");
    assert_eq!(len, 2, "the offloaded states left the local queue");
}

// Both counters at 0 is the session's opening state: nothing measured yet, so
// the first migrations must be allowed rather than gated off.
#[test]
fn test_trigger_a_allowed_with_no_measurements_yet() {
    let (len, offloaded) = run_offload(3, 1, 0, 0);
    assert_eq!(offloaded, 1);
    assert_eq!(len, 2);
}

// (b) Past budget: Trigger A shuts off entirely, even with idle siblings and a
// backlog to share.
#[test]
fn test_trigger_a_suppressed_past_serde_budget() {
    // serde * 4 = 4_000 > 3_000 step ns → not affordable.
    let (len, offloaded) = run_offload(4, 2, 1_000, 3_000);
    assert_eq!(offloaded, 0, "serde has stopped paying for itself");
    assert_eq!(len, 4, "the whole backlog stays worker-local");
}

// The gate is `<=`, so exactly-at-budget still offloads. Pins the boundary that
// separates the two cases above.
#[test]
fn test_serde_budget_boundary_is_inclusive() {
    let counters = SchedulerCounters::default();
    counters.step_ns.store(4_000, Ordering::Relaxed);

    counters.serde_ns.store(1_000, Ordering::Relaxed);
    assert!(
        offload_is_affordable(&counters),
        "serde * DIVISOR == step is still affordable",
    );

    counters.serde_ns.store(1_001, Ordering::Relaxed);
    assert!(!offload_is_affordable(&counters), "one ns past is not");
}

// The budget is monotone and never reset, so an early serde-heavy phase is
// sticky: a later step-heavy phase re-opens Trigger A only once accumulated step
// time has caught up. Documents the known workload assumption in the bead — if
// this is ever changed to a windowed/decaying ratio, this test is the one to
// rewrite.
#[test]
fn test_serde_budget_reopens_only_when_step_time_catches_up() {
    // Same serde debt (1_000 ns); only the accumulated step time differs.
    let (_, blocked) = run_offload(4, 2, 1_000, 3_999);
    assert_eq!(blocked, 0);
    let (_, reopened) = run_offload(4, 2, 1_000, 4_000);
    assert_eq!(reopened, 2, "Trigger A reopens once step time catches up");
}

// Trigger A never drains the queue below one state — the worker must keep work
// for itself no matter how many siblings are starving.
#[test]
fn test_trigger_a_keeps_one_local_state() {
    let (len, offloaded) = run_offload(2, 8, 0, 10_000);
    assert_eq!(offloaded, 1);
    assert_eq!(len, 1, "never shed the last local state");
}

// A single local state is not a backlog: nothing to share.
#[test]
fn test_trigger_a_needs_a_backlog() {
    let (len, offloaded) = run_offload(1, 4, 0, 10_000);
    assert_eq!(offloaded, 0);
    assert_eq!(len, 1);
}

// No starving sibling → no Trigger A offload regardless of budget headroom.
#[test]
fn test_trigger_a_needs_an_idle_sibling() {
    let (len, offloaded) = run_offload(8, 0, 0, 10_000);
    assert_eq!(offloaded, 0);
    assert_eq!(len, 8);
}

// (c) Trigger B is a memory cap, not a load-balancing choice: it must still shed
// down to HWM/2 with the serde budget blown AND no idle siblings.
#[test]
fn test_trigger_b_fires_with_budget_exhausted() {
    let n = LOCAL_HWM + 1;
    let (len, offloaded) = run_offload(n, 0, u64::MAX / 8, 0);
    assert_eq!(len, LOCAL_HWM / 2, "shed down to the low-water mark");
    assert_eq!(offloaded, n - LOCAL_HWM / 2);
}

// At exactly the high-water mark Trigger B holds off (`>` not `>=`), so a
// budget-blocked worker at HWM sheds nothing at all.
#[test]
fn test_trigger_b_holds_at_the_high_water_mark() {
    let (len, offloaded) = run_offload(LOCAL_HWM, 0, u64::MAX / 8, 0);
    assert_eq!(len, LOCAL_HWM);
    assert_eq!(offloaded, 0);
}

// ---------------------------------------------------------------------------
// offload_surplus -> policy.on_state_removed wiring (angr-ua7fd)
//
// offload_surplus detaches states from `local` via `pop_front`, entirely
// bypassing `policy.select` — the only other removal site in the codebase. A
// memoizing policy (LoopHeadRoundRobin::key_cache) needs to hear about every
// such removal or its per-state_id memo leaks for states that get migrated
// instead of dispatched locally (a stolen-back state re-enters via
// `dispatch_next`'s steal branch directly, never through `on_fork`/`select`
// again). This spy policy pins that `offload_surplus` calls the hook for
// EVERY popped state, on both Trigger A (idle-gated) and Trigger B (HWM cap).
// ---------------------------------------------------------------------------

/// A policy that forwards `select`/`on_fork`/`select_for_offload` to `Lifo` but
/// records every
/// `state_id` passed to `on_state_removed`, so tests can assert exactly which
/// states `offload_surplus` evicted without needing access to any real
/// policy's private memo table.
#[derive(Default)]
struct SpyPolicy {
    removed: Mutex<Vec<u64>>,
}

impl SelectionPolicy for SpyPolicy {
    fn select(&self, active: &mut VecDeque<RustSimState>) -> Option<RustSimState> {
        Lifo.select(active)
    }

    fn on_fork(&self, active: &mut VecDeque<RustSimState>, state: RustSimState) {
        Lifo.on_fork(active, state);
    }

    /// Forwarded too, so the two `on_state_removed` tests below keep asserting
    /// against `Lifo`'s cold end (the front) rather than the trait default's.
    fn select_for_offload(&self, local: &mut VecDeque<RustSimState>) -> Option<RustSimState> {
        Lifo.select_for_offload(local)
    }

    fn name(&self) -> &'static str {
        "spy"
    }

    fn on_state_removed(&self, state_id: u64) {
        self.removed.lock().expect("spy poisoned").push(state_id);
    }
}

// Trigger A (idle-gated): every state `offload_surplus` sheds to the injector
// must fire `on_state_removed` exactly once, with the offloaded state's own
// id — not the survivor's.
#[test]
fn test_offload_surplus_notifies_policy_on_trigger_a_offload() {
    // Keep a concrete `Arc<SpyPolicy>` handle alongside the trait-object `Arc`
    // `offload_surplus` takes, so the assertions below can read `removed`
    // through the same underlying allocation.
    let spy = Arc::new(SpyPolicy::default());
    let policy: Arc<dyn SelectionPolicy> = spy.clone();
    let states: Vec<RustSimState> = (0..4).map(|_| plain_state()).collect();
    let ids: Vec<u64> = states.iter().map(|s| s.state_id()).collect();
    let mut local: VecDeque<RustSimState> = states.into_iter().collect();
    let injector: Injector<StateMigrationPayload> = Injector::new();
    let idle_workers = AtomicUsize::new(2);
    let counters = SchedulerCounters::default();

    offload_surplus(&mut local, &injector, &idle_workers, &counters, &policy);

    assert_eq!(
        local.len(),
        2,
        "two states offloaded to two starving siblings"
    );
    let removed = spy.removed.lock().expect("spy poisoned").clone();
    assert_eq!(
        removed,
        ids[..2],
        "on_state_removed fires for exactly the two coldest (front) states, in pop order",
    );
}

// Trigger B (HWM cap): same contract, but for the memory-cap path, which is
// independent of the serde budget / idle-sibling gates Trigger A uses.
#[test]
fn test_offload_surplus_notifies_policy_on_trigger_b_offload() {
    let spy = Arc::new(SpyPolicy::default());
    let policy: Arc<dyn SelectionPolicy> = spy.clone();
    let n = LOCAL_HWM + 3;
    let states: Vec<RustSimState> = (0..n).map(|_| plain_state()).collect();
    let ids: Vec<u64> = states.iter().map(|s| s.state_id()).collect();
    let mut local: VecDeque<RustSimState> = states.into_iter().collect();
    let injector: Injector<StateMigrationPayload> = Injector::new();
    let idle_workers = AtomicUsize::new(0);
    let counters = SchedulerCounters::default();
    // Blow the serde budget so Trigger A cannot also fire and muddy the count.
    counters.serde_ns.store(u64::MAX / 8, Ordering::Relaxed);

    offload_surplus(&mut local, &injector, &idle_workers, &counters, &policy);

    let expected_offloaded = n - LOCAL_HWM / 2;
    assert_eq!(local.len(), LOCAL_HWM / 2);
    let removed = spy.removed.lock().expect("spy poisoned").clone();
    assert_eq!(
        removed,
        ids[..expected_offloaded],
        "Trigger B also notifies the policy for every state it sheds",
    );
}

// ---------------------------------------------------------------------------
// offload picks the end the POLICY calls cold (angr-03vl4.19)
//
// `offload_one` used to hardcode `local.pop_front()`, documented as "the
// coldest (front) state". That is only true for `Lifo` — and the production
// default is `Fifo`, which dispatches from the front, so every default-policy
// offload shipped away exactly the state `dispatch_next` was about to hand back
// for free, paying a full Z3 detach/reattach round-trip for it. The pick now
// goes through `policy.select_for_offload`; these pin both ends of the mirror,
// and the `default_policy()` test drives the real production wiring rather than
// the `Lifo` helper the rest of this file uses.
// ---------------------------------------------------------------------------

/// Trigger-A-offload `idle` states out of a fresh `n`-state local deque under
/// `policy`. Returns `(all ids in deque order, ids still local afterward)`.
fn offload_under(policy: &Arc<dyn SelectionPolicy>, n: usize, idle: usize) -> (Vec<u64>, Vec<u64>) {
    let states: Vec<RustSimState> = (0..n).map(|_| plain_state()).collect();
    let ids: Vec<u64> = states.iter().map(|s| s.state_id()).collect();
    let mut local: VecDeque<RustSimState> = states.into_iter().collect();
    let injector: Injector<StateMigrationPayload> = Injector::new();
    let idle_workers = AtomicUsize::new(idle);
    let counters = SchedulerCounters::default();

    offload_surplus(&mut local, &injector, &idle_workers, &counters, policy);

    (ids, local.iter().map(|s| s.state_id()).collect())
}

// `Fifo` dispatches the front, so its cold end is the tail.
#[test]
fn test_fifo_offloads_the_tail_and_keeps_its_next_dispatch() {
    let policy: Arc<dyn SelectionPolicy> = Arc::new(Fifo);
    let (ids, left) = offload_under(&policy, 4, 2);
    assert_eq!(
        left,
        ids[..2],
        "the two tail states migrated; the front survivors stay local",
    );
    // The state `Fifo::select` would hand back next must still be here — that
    // is the whole point of offloading the far end.
    assert_eq!(left[0], ids[0], "the next local dispatch was not shipped away");
}

// `Lifo` dispatches the tail, so its cold end is the front — the mirror image.
#[test]
fn test_lifo_offloads_the_front_and_keeps_its_next_dispatch() {
    let policy: Arc<dyn SelectionPolicy> = Arc::new(Lifo);
    let (ids, left) = offload_under(&policy, 4, 2);
    assert_eq!(left, ids[2..], "the two front states migrated");
    assert_eq!(
        *left.last().expect("two survivors"),
        *ids.last().expect("four states"),
        "the next local dispatch was not shipped away",
    );
}

// The production default, exercised through the exact constructor
// `RustExplorationManager` uses (`selection_policy::default_policy()`) rather
// than a policy this file names for itself — the drift that hid the bug.
#[test]
fn test_default_policy_offload_spares_the_next_dispatch() {
    let policy = default_policy();
    assert_eq!(policy.name(), "fifo", "the manager default is BFS");

    // Two states, one starving sibling: one migrates. Then ask the SAME policy
    // which of the survivors it dispatches next and assert that state never
    // left — the property the old hardcoded `pop_front` broke.
    let states: Vec<RustSimState> = (0..2).map(|_| plain_state()).collect();
    let ids: Vec<u64> = states.iter().map(|s| s.state_id()).collect();
    let mut local: VecDeque<RustSimState> = states.into_iter().collect();
    let injector: Injector<StateMigrationPayload> = Injector::new();
    let idle_workers = AtomicUsize::new(1);
    let counters = SchedulerCounters::default();

    offload_surplus(&mut local, &injector, &idle_workers, &counters, &policy);

    assert_eq!(local.len(), 1, "one state went to the starving sibling");
    let next = policy.select(&mut local).expect("one survivor");
    assert_eq!(
        next.state_id(),
        ids[0],
        "the default policy's next dispatch stayed local; the far end migrated",
    );
}

// ---------------------------------------------------------------------------
// dispatch / steal / absorb / drain bookkeeping (angr-ph300.3.2)
//
// `scheduler_tests.rs` drives the pool end to end, where these helpers only ever
// run interleaved across threads. The tests below call them single-threaded with
// the transport counters observable, so the accounting contracts each one owns —
// `pending` conservation, the `idle_workers` balance across a steal, and the
// residual-drain path that Bug M1 added — are pinned independently of any
// workload.
// ---------------------------------------------------------------------------

/// A transport pinned to `Lifo` — NOT what the pool builds by default. The
/// production default is `selection_policy::default_policy()` (`Fifo`, threaded
/// down as `Arc::clone(&self.policy)` by `run_loop_wave.rs` /
/// `run_loop_steady.rs`); this helper exists only because the dispatch-order
/// assertions below are written against `Lifo`'s freshest-first order.
fn lifo_transport() -> WorkTransport {
    WorkTransport::with_policy(Arc::new(Lifo))
}

// `absorb_continues` counts children IN; `dispatch_next` hands them back
// freshest-first under the default `Lifo` and never touches `pending` (the
// caller counts the parent OUT). The two together must conserve the task count.
#[test]
fn test_absorb_then_dispatch_conserves_pending() {
    let ctx = Context::thread_local();
    let t = lifo_transport();
    let mut local: VecDeque<RustSimState> = VecDeque::new();

    let children: Vec<RustSimState> = (0..3).map(|_| plain_state()).collect();
    let ids: Vec<u64> = children.iter().map(|s| s.state_id()).collect();
    absorb_continues(&t, &mut local, children);

    assert_eq!(t.pending.load(Ordering::SeqCst), 3, "children counted IN");
    assert_eq!(local.len(), 3);

    // Exactly three dispatches: a fourth would find the local queue empty and
    // block in `steal_from_injector`, because a real worker retires each task
    // (`pending -= 1`) after processing it and this test never does.
    let mut dispatched = Vec::new();
    for _ in 0..3 {
        let state = dispatch_next(0, &t, &mut local, &ctx).expect("a queued child");
        dispatched.push(state.state_id());
    }
    assert!(local.is_empty(), "the local frontier is drained");
    assert_eq!(
        dispatched,
        ids.iter().rev().copied().collect::<Vec<_>>(),
        "Lifo hands back the freshest child first",
    );
    assert_eq!(
        t.pending.load(Ordering::SeqCst),
        3,
        "dispatch does not decrement pending — the caller counts the task OUT",
    );
    assert_eq!(t.counters.local_dispatches.load(Ordering::SeqCst), 3);
    assert_eq!(
        t.counters.injector_dispatches.load(Ordering::SeqCst),
        0,
        "nothing was stolen: the local frontier covered every dispatch",
    );
}

// angr-9ke6b.48: with a `max_active_states` cap on the transport, forks beyond
// the cap are pruned instead of queued. `pending` starts at 1 (the parent task
// in flight, which `absorb_continues` discounts), so a cap of 3 admits exactly
// 3 of 5 children and summarizes the other 2 as Pruned.
#[test]
fn test_absorb_continues_enforces_max_active_states() {
    let mut t = lifo_transport();
    t.max_active_states = Some(3);
    t.pending.store(1, Ordering::SeqCst); // the parent being processed
    let mut local: VecDeque<RustSimState> = VecDeque::new();

    let children: Vec<RustSimState> = (0..5).map(|_| plain_state()).collect();
    absorb_continues(&t, &mut local, children);

    assert_eq!(local.len(), 3, "only the budgeted forks are queued");
    assert_eq!(
        t.pending.load(Ordering::SeqCst),
        4,
        "pending counts the parent plus the 3 admitted children",
    );
    assert_eq!(
        t.counters.summarized_pruned.load(Ordering::SeqCst),
        2,
        "the 2 over-cap forks are recorded as Pruned summaries, not dropped silently",
    );
}

// The cap must still let a saturated frontier make progress: with the frontier
// already at the limit, the parent's discount leaves room for exactly one
// replacement child — the same steady-state the serial `push_to_active_or_drop`
// reaches (the stepping state is already out of `STASH_ACTIVE`). A naive
// `pending >= limit` check would starve the wave to a halt instead.
#[test]
fn test_absorb_continues_saturated_frontier_still_advances() {
    let mut t = lifo_transport();
    t.max_active_states = Some(2);
    t.pending.store(2, Ordering::SeqCst); // frontier full, parent in flight
    let mut local: VecDeque<RustSimState> = VecDeque::new();

    absorb_continues(&t, &mut local, (0..4).map(|_| plain_state()).collect());

    assert_eq!(local.len(), 1, "one child replaces the retiring parent");
    assert_eq!(t.counters.summarized_pruned.load(Ordering::SeqCst), 3);
}

// No cap configured (the default for every Rust-side / test construction) must
// leave the pre-angr-9ke6b.48 behavior byte-identical: every fork is queued.
#[test]
fn test_absorb_continues_unbounded_without_cap() {
    let t = lifo_transport();
    assert!(t.max_active_states.is_none(), "unbounded by default");
    let mut local: VecDeque<RustSimState> = VecDeque::new();

    absorb_continues(&t, &mut local, (0..8).map(|_| plain_state()).collect());

    assert_eq!(local.len(), 8);
    assert_eq!(t.counters.summarized_pruned.load(Ordering::SeqCst), 0);
}

// Absorbing an empty successor vec must not perturb `pending` (the fetch_add is
// guarded) — otherwise every dead-ended task would inflate the quiescence count.
#[test]
fn test_absorb_no_continues_leaves_pending_untouched() {
    let t = lifo_transport();
    let mut local: VecDeque<RustSimState> = VecDeque::new();
    absorb_continues(&t, &mut local, Vec::new());
    assert_eq!(t.pending.load(Ordering::SeqCst), 0);
    assert!(local.is_empty());
}

// The steal path: with the local queue empty, `dispatch_next` pulls from the
// injector, reattaches into this worker's context, and charges the reattach to
// the serde budget. `pending` is untouched (offload relocates already-counted
// tasks) and the state survives the round trip.
#[test]
fn test_dispatch_next_steals_and_reattaches_from_injector() {
    let ctx = Context::thread_local();
    let t = lifo_transport();
    let mut local: VecDeque<RustSimState> = VecDeque::new();

    let state = plain_state();
    let id = state.state_id();
    t.pending.fetch_add(1, Ordering::SeqCst);
    t.injector.push(detach_timed(state, &t.counters));

    let got = dispatch_next(0, &t, &mut local, &ctx).expect("the injector had a task");
    assert_eq!(
        got.state_id(),
        id,
        "the stolen state survived the migration"
    );
    assert_eq!(
        t.pending.load(Ordering::SeqCst),
        1,
        "a steal relocates a task, it does not create or retire one",
    );
    assert_eq!(t.counters.injector_dispatches.load(Ordering::SeqCst), 1);
    assert_eq!(t.counters.reattaches.load(Ordering::SeqCst), 1);
    assert_eq!(t.counters.local_dispatches.load(Ordering::SeqCst), 0);
    assert!(
        t.counters.serde_ns.load(Ordering::Relaxed) > 0,
        "both halves of the migration are charged to the serde budget",
    );
}

// ---------------------------------------------------------------------------
// angr-9ke6b.58: the worker-side half of the reattach-failure contract. The
// coordinator-side twin lives in `run_loop_wave_tests.rs`
// (`corrupt_payload_is_dropped_and_later_payloads_still_route`, angr-ph300.11);
// `dispatch_next`'s `Err` arm had no failure-injection coverage. `reattach` can
// fail for more than the `ContextMismatch` its inline comment focuses on —
// `RustSimState::from_serialized` propagates too — so the log+retire+continue
// behavior is pinned rather than assumed unreachable.
//
// Both tests rely on `Injector::steal` being FIFO, so the corrupt payload
// pushed first is the one stolen first.
// ---------------------------------------------------------------------------

// A corrupt payload must be retired out of `pending` (never left outstanding as
// a phantom task) and must not stop the worker from stealing the payload queued
// behind it.
#[test]
fn test_dispatch_next_drops_a_corrupt_payload_and_keeps_stealing() {
    let ctx = Context::thread_local();
    let t = lifo_transport();
    let mut local: VecDeque<RustSimState> = VecDeque::new();

    let good = plain_state();
    let good_id = good.state_id();
    t.pending.fetch_add(2, Ordering::SeqCst);
    t.injector.push(StateMigrationPayload::corrupt_for_test());
    t.injector.push(detach_timed(good, &t.counters));

    let got = dispatch_next(0, &t, &mut local, &ctx).expect("the good payload behind the corrupt");
    assert_eq!(
        got.state_id(),
        good_id,
        "the corrupt payload was skipped, not returned"
    );
    assert_eq!(
        t.pending.load(Ordering::SeqCst),
        1,
        "the dropped task was counted OUT; only the dispatched one stays outstanding",
    );
    assert_eq!(
        t.counters.injector_dispatches.load(Ordering::SeqCst),
        2,
        "both steals are counted; the counter is a steal count, not a success count",
    );
    assert_eq!(
        t.counters.reattaches.load(Ordering::SeqCst),
        1,
        "only the payload that actually rebuilt a state counts as a reattach",
    );
}

// A corrupt payload as the ONLY outstanding task: the retire inside the `Err`
// arm is what lets the next `steal_from_injector` see `pending == 0` and report
// quiescence. Without it the worker would spin/sleep forever waiting on a task
// that can never arrive.
#[test]
fn test_dispatch_next_corrupt_only_payload_reaches_quiescence() {
    let ctx = Context::thread_local();
    let t = lifo_transport();
    let mut local: VecDeque<RustSimState> = VecDeque::new();

    t.pending.fetch_add(1, Ordering::SeqCst);
    t.injector.push(StateMigrationPayload::corrupt_for_test());

    assert!(
        dispatch_next(0, &t, &mut local, &ctx).is_none(),
        "a dropped-only injector reaches quiescence instead of hanging",
    );
    assert_eq!(
        t.pending.load(Ordering::SeqCst),
        0,
        "the phantom task is retired, so quiescence is reachable",
    );
    assert_eq!(t.counters.reattaches.load(Ordering::SeqCst), 0);
    assert_eq!(t.counters.injector_dispatches.load(Ordering::SeqCst), 1);
    assert_eq!(
        t.counters.local_dispatches.load(Ordering::SeqCst),
        0,
        "nothing ever reached the local frontier",
    );
}

// Quiescence: an empty injector with nothing outstanding is the only way
// `dispatch_next` returns `None` without cancellation. It must not spin.
#[test]
fn test_dispatch_next_none_at_quiescence() {
    let ctx = Context::thread_local();
    let t = lifo_transport();
    let mut local: VecDeque<RustSimState> = VecDeque::new();
    assert!(dispatch_next(0, &t, &mut local, &ctx).is_none());
}

// Cancellation is observed inside the steal wait even when `pending` says work
// is still outstanding — otherwise a cancelled worker would block until the
// other workers finished. This is the task-boundary cancel contract both loops
// rely on to reach their drain path.
#[test]
fn test_dispatch_next_none_when_cancelled_with_work_outstanding() {
    let ctx = Context::thread_local();
    let t = lifo_transport();
    let mut local: VecDeque<RustSimState> = VecDeque::new();
    // A sibling is mid-task, so quiescence alone would keep us waiting.
    t.pending.fetch_add(1, Ordering::SeqCst);
    t.cancel.cancel();

    assert!(
        dispatch_next(0, &t, &mut local, &ctx).is_none(),
        "cancel wins over an outstanding task",
    );
    assert_eq!(
        t.pending.load(Ordering::SeqCst),
        1,
        "the sibling's task is not stolen or retired by our cancel",
    );
}

// A cancelled worker still dispatches whatever is already local: cancel is only
// checked in the steal wait, so the in-flight frontier is drained by the loop's
// own cancel check (and `drain_local_*`), never dropped here.
#[test]
fn test_dispatch_next_prefers_local_even_when_cancelled() {
    let ctx = Context::thread_local();
    let t = lifo_transport();
    let mut local: VecDeque<RustSimState> = VecDeque::from([plain_state()]);
    t.cancel.cancel();
    assert!(
        dispatch_next(0, &t, &mut local, &ctx).is_some(),
        "the local pop precedes the cancel-aware steal wait",
    );
}

// `steal_from_injector` raises the starvation signal for the duration of the
// wait and lowers it again on every exit path — Trigger A reads that counter, so
// a leaked increment would make productive workers offload forever.
#[test]
fn test_steal_balances_the_idle_signal_on_every_exit() {
    let injector: Injector<StateMigrationPayload> = Injector::new();
    let pending = AtomicUsize::new(0);
    let cancel = CancelToken::new();
    let idle = AtomicUsize::new(0);

    // Quiescent exit.
    assert!(steal_from_injector(&injector, &pending, &cancel, &idle).is_none());
    assert_eq!(idle.load(Ordering::SeqCst), 0, "idle signal cleared");

    // Success exit.
    let counters = SchedulerCounters::default();
    injector.push(detach_timed(plain_state(), &counters));
    pending.store(1, Ordering::SeqCst);
    assert!(steal_from_injector(&injector, &pending, &cancel, &idle).is_some());
    assert_eq!(idle.load(Ordering::SeqCst), 0);

    // Cancelled exit, with work still outstanding.
    cancel.cancel();
    assert!(steal_from_injector(&injector, &pending, &cancel, &idle).is_none());
    assert_eq!(idle.load(Ordering::SeqCst), 0);
}

// The Bug M1 residual drain: a cancelled worker's un-dispatched frontier leaves
// as untagged migration payloads, `pending` is retired for exactly those states,
// and each one is counted as a residual drain. Nothing is lost or double-counted.
#[test]
fn test_drain_local_hands_off_the_residual_frontier() {
    let t = lifo_transport();
    let mut local: VecDeque<RustSimState> = (0..3).map(|_| plain_state()).collect();
    let ids: Vec<u64> = local.iter().map(|s| s.state_id()).collect();
    t.pending.store(3, Ordering::SeqCst);

    let mut sunk: Vec<StateMigrationPayload> = Vec::new();
    drain_local_with(&t, &mut local, |payload| sunk.push(payload));

    assert!(local.is_empty(), "the frontier left the worker");
    assert_eq!(sunk.len(), 3, "every residual state reached the sink");
    assert_eq!(
        t.pending.load(Ordering::SeqCst),
        0,
        "the drained tasks are retired exactly once",
    );
    assert_eq!(t.counters.residual_drains.load(Ordering::SeqCst), 3);

    let ctx = Context::thread_local();
    let mut got: Vec<u64> = sunk
        .into_iter()
        .map(|p| p.reattach(&ctx).expect("reattach").state_id())
        .collect();
    got.sort_unstable();
    let mut want = ids;
    want.sort_unstable();
    assert_eq!(got, want, "the payloads carry the same states, no dupes");
}

// drain_local_with has the identical select()-bypass pattern offload_surplus
// has (angr-ua7fd follow-up): `local.drain(..)` detaches every residual state
// without going through `policy.select`. This path is more routine than
// offload_surplus's — it fires on essentially every steady-session
// pause/finalize (`finalize_steady_session` in run_loop_steady.rs) — and drained
// states get reseeded through the shared injector the same way, so a
// memoizing policy's per-state side table (LoopHeadRoundRobin::key_cache)
// needs the same eviction notification here.
#[test]
fn test_drain_local_with_notifies_policy_for_every_drained_state() {
    let spy = Arc::new(SpyPolicy::default());
    let policy: Arc<dyn SelectionPolicy> = spy.clone();
    let t = WorkTransport::with_policy(policy);
    let states: Vec<RustSimState> = (0..3).map(|_| plain_state()).collect();
    let ids: Vec<u64> = states.iter().map(|s| s.state_id()).collect();
    let mut local: VecDeque<RustSimState> = states.into_iter().collect();
    t.pending.store(3, Ordering::SeqCst);

    let mut sunk: Vec<StateMigrationPayload> = Vec::new();
    drain_local_with(&t, &mut local, |payload| sunk.push(payload));

    assert!(local.is_empty(), "the frontier left the worker");
    let mut removed = spy.removed.lock().expect("spy poisoned").clone();
    removed.sort_unstable();
    let mut want = ids;
    want.sort_unstable();
    assert_eq!(
        removed, want,
        "on_state_removed fires for every drained state, no drops or dupes",
    );
}

// Draining an already-empty frontier is a no-op on both `pending` and the
// counters — the early return matters because both loops call it unconditionally
// on the cancel path, including when the frontier was fully dispatched.
#[test]
fn test_drain_local_empty_is_a_noop() {
    let t = lifo_transport();
    t.pending.store(2, Ordering::SeqCst);
    let mut local: VecDeque<RustSimState> = VecDeque::new();
    let mut calls = 0usize;
    drain_local_with(&t, &mut local, |_| calls += 1);
    assert_eq!(calls, 0);
    assert_eq!(
        t.pending.load(Ordering::SeqCst),
        2,
        "an empty drain must not retire a sibling's outstanding task",
    );
    assert_eq!(t.counters.residual_drains.load(Ordering::SeqCst), 0);
}
