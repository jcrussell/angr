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

/// A minimal migratable state. `offload_surplus` only detaches and counts, so
/// no registers or constraints are needed.
fn plain_state() -> RustSimState {
    RustSimState::new("amd64").expect("amd64 state")
}

/// Run `offload_surplus` over `n` local states with the budget preset to
/// (`serde_ns`, `step_ns`) and `idle` starving siblings. Returns
/// `(local_len_after, surplus_offloaded)`.
fn run_offload(n: usize, idle: usize, serde_ns: u64, step_ns: u64) -> (usize, usize) {
    let mut local: VecDeque<RustSimState> = (0..n).map(|_| plain_state()).collect();
    let injector: Injector<StateMigrationPayload> = Injector::new();
    let idle_workers = AtomicUsize::new(idle);
    let counters = SchedulerCounters::default();
    counters.serde_ns.store(serde_ns, Ordering::Relaxed);
    counters.step_ns.store(step_ns, Ordering::Relaxed);

    offload_surplus(&mut local, &injector, &idle_workers, &counters);

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
