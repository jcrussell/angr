// angr-muje: lineage / ScopePath unit tests, extracted out of the former
// in-file `mod tests` (~630 lines) into a sibling file to shrink
// symbolic/lineage.rs below the god-object threshold. Declared as a direct
// child of `lineage` so `use super::*` reaches the module's private items.

use super::*;

fn make_solver() -> z3::Solver {
    // Use the default tactic — these tests don't depend on the
    // QF_BV-vs-SMT routing in context.rs's `build_solver`. The
    // thread-local Z3 context is implicitly initialized.
    z3::Solver::new()
}

fn bvconst(name: &str, width: u32) -> z3::ast::BV {
    z3::ast::BV::new_const(name, width)
}

fn eq_bv_const(bv: &z3::ast::BV, value: u64) -> Bool {
    bv.eq(z3::ast::BV::from_u64(value, bv.get_size()))
}

/// Look up one counter by name in a `lineage_stats()` / `dismantle_stats()`
/// snapshot. The two return fixed-size arrays of *different* lengths, so this
/// takes a slice and every call site passes `&snapshot`.
///
/// A missing name yields 0, which is what every assertion in this file wants:
/// these counters are global and other tests in the module bump them in
/// parallel, so the checks are all deltas or lower bounds rather than
/// equalities.
fn find_counter(name: &str, snapshot: &[(&'static str, u64)]) -> u64 {
    snapshot
        .iter()
        .find(|(n, _)| *n == name)
        .map_or(0, |(_, v)| *v)
}

/// Frame ids are unique within a single run.
#[test]
fn test_frame_ids_unique() {
    let a = mint_frame_id();
    let b = mint_frame_id();
    let c = mint_frame_id();
    assert_ne!(a, b);
    assert_ne!(b, c);
    assert_ne!(a, c);
}

/// Switching to an empty path on a fresh solver is a no-op.
#[test]
fn test_switch_empty_is_noop() {
    let mut lin = SharedLineageSolver::new(make_solver());
    let empty = ScopePath::new();
    let (pops, pushes) = lin.switch_to(&empty);
    assert_eq!((pops, pushes), (0, 0));
    assert_eq!(lin.loaded_depth(), 0);
}

/// Switching to a path of length k pushes k frames; switching back
/// to empty pops them.
#[test]
fn test_switch_push_then_pop() {
    let mut lin = SharedLineageSolver::new(make_solver());
    let x = bvconst("test_switch_push_then_pop_x", 8);

    let path = vec![
        ScopeFrame::new(true, eq_bv_const(&x, 5)),
        ScopeFrame::new(true, x.bvugt(z3::ast::BV::from_u64(0, 8))),
    ];

    let (pops, pushes) = lin.switch_to(&path);
    assert_eq!((pops, pushes), (0, 2));
    assert_eq!(lin.loaded_depth(), 2);

    let (pops, pushes) = lin.switch_to(&ScopePath::new());
    assert_eq!((pops, pushes), (2, 0));
    assert_eq!(lin.loaded_depth(), 0);
}

/// Switching to the same path twice is a no-op the second time —
/// the hot-path optimization the BFS-thrashing heuristic in the
/// parent bead's description depends on.
#[test]
fn test_switch_same_path_is_hot_noop() {
    let mut lin = SharedLineageSolver::new(make_solver());
    let x = bvconst("test_switch_same_path_is_hot_noop_x", 8);

    let path = vec![ScopeFrame::new(true, eq_bv_const(&x, 7))];

    lin.switch_to(&path);
    let depth_after_first = lin.loaded_depth();
    let (pops, pushes) = lin.switch_to(&path);
    assert_eq!((pops, pushes), (0, 0), "second switch should be a no-op");
    assert_eq!(lin.loaded_depth(), depth_after_first);
}

/// Counters increment by the expected amount on switch/push/pop.
/// Uses delta comparisons (post - pre) because the `LINEAGE_*`
/// atomics are global and other tests in this module run in parallel
/// and may bump them concurrently — we just need to verify our own
/// operations contributed the right deltas.
#[test]
fn test_counters_increment() {
    let pre = lineage_stats();
    let mut lin = SharedLineageSolver::new(make_solver());
    let x = bvconst("test_counters_increment_x", 8);
    let path = vec![ScopeFrame::new(true, eq_bv_const(&x, 7))];

    // Switch into path (1 switch, 1 push), then switch to same path
    // (1 switch, 1 hot no-op via O(1) fast path), then back to empty
    // (1 switch, 1 pop).
    lin.switch_to(&path);
    lin.switch_to(&path);
    lin.switch_to(&ScopePath::new());

    let post = lineage_stats();
    let delta =
        |name: &str| find_counter(name, &post).saturating_sub(find_counter(name, &pre));
    // Lower bounds: other parallel tests can only ADD to these
    // counters, never subtract — but we must contribute at least
    // this many ourselves.
    assert!(
        delta("lineage_switch_count") >= 3,
        "≥3 switches (got {})",
        delta("lineage_switch_count")
    );
    assert!(
        delta("lineage_switch_hot_count") >= 1,
        "≥1 hot no-op (got {})",
        delta("lineage_switch_hot_count")
    );
    assert!(
        delta("lineage_switch_fast_path_count") >= 1,
        "≥1 fast-path hit (got {})",
        delta("lineage_switch_fast_path_count")
    );
    assert!(
        delta("lineage_push_count") >= 1,
        "≥1 push (got {})",
        delta("lineage_push_count")
    );
    assert!(
        delta("lineage_pop_count") >= 1,
        "≥1 pop (got {})",
        delta("lineage_pop_count")
    );
}

/// Sibling paths sharing a 2-frame prefix only pop+push the diverging
/// 1-frame tail (the cost optimization the design targets).
#[test]
fn test_switch_common_prefix_only_diffs() {
    let mut lin = SharedLineageSolver::new(make_solver());
    let x = bvconst("test_switch_common_prefix_only_diffs_x", 8);

    // Shared prefix: two frames the siblings would have inherited
    // from a common ancestor.
    let prefix = [
        ScopeFrame::new(true, x.bvugt(z3::ast::BV::from_u64(0, 8))),
        ScopeFrame::new(true, x.bvult(z3::ast::BV::from_u64(100, 8))),
    ];

    let mut sibling_a = prefix.to_vec();
    sibling_a.push(ScopeFrame::new(true, eq_bv_const(&x, 5)));

    let mut sibling_b = prefix.to_vec();
    sibling_b.push(ScopeFrame::new(true, eq_bv_const(&x, 42)));

    let (pops, pushes) = lin.switch_to(&sibling_a);
    assert_eq!((pops, pushes), (0, 3));

    // Switching to sibling_b should reuse the 2-frame prefix —
    // exactly one pop, exactly one push.
    let (pops, pushes) = lin.switch_to(&sibling_b);
    assert_eq!(
        (pops, pushes),
        (1, 1),
        "only the diverging tail frame should move"
    );
    assert_eq!(lin.loaded_depth(), 3);
}

/// Loaded constraints actually constrain the solver: an SMT query
/// against `x == 5` finds x = 5; switching to a sibling with
/// `x == 42` finds 42.
#[test]
fn test_loaded_constraint_actually_holds() {
    let mut lin = SharedLineageSolver::new(make_solver());
    let x = bvconst("test_loaded_constraint_actually_holds_x", 32);

    let path_5 = vec![ScopeFrame::new(true, eq_bv_const(&x, 5))];
    let val = lin.with_solver(&path_5, |solver| {
        assert_eq!(solver.check(), z3::SatResult::Sat);
        let m = solver.get_model().expect("model after Sat");
        m.eval(&x, true).and_then(|bv| bv.as_u64())
    });
    assert_eq!(val, Some(5));

    let path_42 = vec![ScopeFrame::new(true, eq_bv_const(&x, 42))];
    let val = lin.with_solver(&path_42, |solver| {
        assert_eq!(solver.check(), z3::SatResult::Sat);
        let m = solver.get_model().expect("model after Sat");
        m.eval(&x, true).and_then(|bv| bv.as_u64())
    });
    assert_eq!(val, Some(42));
}

/// Base assertions persist across switch_to/pop cycles.
#[test]
fn test_base_assertions_persist() {
    let mut lin = SharedLineageSolver::new(make_solver());
    let x = bvconst("test_base_assertions_persist_x", 8);

    // x > 10 at scope 0
    lin.assert_base(&x.bvugt(z3::ast::BV::from_u64(10, 8)));

    // Push a frame constraining x < 20, query, then pop back.
    let scoped = vec![ScopeFrame::new(true, x.bvult(z3::ast::BV::from_u64(20, 8)))];
    lin.switch_to(&scoped);
    lin.switch_to(&ScopePath::new());

    // The base constraint must still be in force after pop.
    let val = lin.with_solver(&ScopePath::new(), |solver| {
        // Add a temporary constraint forcing x == 5 (violates base),
        // verify the solver reports UNSAT.
        solver.push();
        solver.assert(eq_bv_const(&x, 5));
        let r = solver.check();
        solver.pop(1);
        r
    });
    assert_eq!(
        val,
        z3::SatResult::Unsat,
        "base x > 10 must still hold after pop"
    );
}

/// Three-deep sibling: A=[a,b,c], B=[a,b,d], C=[a,e]. Switching
/// A → B reuses 2 frames; A → C reuses 1.
#[test]
fn test_switch_three_deep_siblings() {
    let mut lin = SharedLineageSolver::new(make_solver());
    let x = bvconst("test_switch_three_deep_siblings_x", 16);

    let a = ScopeFrame::new(true, x.bvugt(z3::ast::BV::from_u64(0, 16)));
    let b = ScopeFrame::new(true, x.bvult(z3::ast::BV::from_u64(1000, 16)));
    let c = ScopeFrame::new(true, eq_bv_const(&x, 5));
    let d = ScopeFrame::new(true, eq_bv_const(&x, 7));
    let e = ScopeFrame::new(true, x.bvugt(z3::ast::BV::from_u64(500, 16)));

    let path_abc = vec![a.clone(), b.clone(), c];
    let path_abd = vec![a.clone(), b, d];
    let path_ae = vec![a, e];

    let (pops, pushes) = lin.switch_to(&path_abc);
    assert_eq!((pops, pushes), (0, 3));

    // A→B: 2-frame prefix (a, b) reused; pop 1, push 1.
    let (pops, pushes) = lin.switch_to(&path_abd);
    assert_eq!((pops, pushes), (1, 1));

    // B→C: 1-frame prefix (a) reused; pop 2 (b, d), push 1 (e).
    let (pops, pushes) = lin.switch_to(&path_ae);
    assert_eq!((pops, pushes), (2, 1));
    assert_eq!(lin.loaded_depth(), 2);
}

/// O(1) hot-cache fast path: a same-path re-switch must return
/// (0, 0) AND increment `lineage_switch_fast_path_count`. Divergent
/// switches return non-zero pop/push counts. Counter is a lower-bound
/// check because other parallel tests in the module also bump it.
#[test]
fn test_switch_fast_path_counter_only_fires_on_same_path() {
    let pre = lineage_stats();
    let fp_pre = find_counter("lineage_switch_fast_path_count", &pre);

    let mut lin = SharedLineageSolver::new(make_solver());
    let x = bvconst("test_switch_fast_path_counter_x", 8);
    let path = vec![ScopeFrame::new(true, eq_bv_const(&x, 7))];

    // First switch: divergent (empty → 1-frame). NOT a fast-path hit;
    // the return value reflects that 1 frame was pushed.
    let (pops, pushes) = lin.switch_to(&path);
    assert_eq!((pops, pushes), (0, 1), "first switch pushes 1 frame");

    // Second switch: same path → O(1) fast path fires, returns (0, 0).
    let (pops, pushes) = lin.switch_to(&path);
    assert_eq!(
        (pops, pushes),
        (0, 0),
        "second same-path switch must be a no-op"
    );

    // Third switch: back to empty → divergent, returns (1, 0).
    let (pops, pushes) = lin.switch_to(&ScopePath::new());
    assert_eq!((pops, pushes), (1, 0), "third switch pops 1 frame");

    // Lower-bound check: at least one fast-path hit was contributed
    // by us (the second switch). Other parallel tests can only ADD
    // to the global counter, never subtract.
    let post = lineage_stats();
    let fp_post = find_counter("lineage_switch_fast_path_count", &post);
    assert!(
        fp_post.saturating_sub(fp_pre) >= 1,
        "at least 1 fast-path hit expected (got {})",
        fp_post.saturating_sub(fp_pre)
    );
}

/// Repeated re-entry on the SAME deep path (the BFS intra-state
/// query-burst pattern the hot-cache targets) hits the fast path on
/// every call after the first.
#[test]
fn test_switch_fast_path_deep_path_repeated_reentry() {
    let mut lin = SharedLineageSolver::new(make_solver());
    let x = bvconst("test_switch_fast_path_deep_path_x", 32);

    // 10-frame deep path representing many accumulated constraints
    // (e.g., from a long-running state).
    let path: ScopePath = (0..10)
        .map(|i| ScopeFrame::new(true, x.bvugt(z3::ast::BV::from_u64(i as u64, 32))))
        .collect();

    // Initial switch into the deep path.
    lin.switch_to(&path);
    assert_eq!(lin.loaded_depth(), 10);

    // Capture fast-path counter, then re-enter 100 times. Every
    // re-entry must hit the O(1) cache (same path, same tail id,
    // same depth) — no Z3 push/pop should fire.
    let before = lineage_stats();
    let fp_before = find_counter("lineage_switch_fast_path_count", &before);
    let push_before = find_counter("lineage_push_count", &before);
    let pop_before = find_counter("lineage_pop_count", &before);

    let mut local_push = 0u64;
    let mut local_pop = 0u64;
    for _ in 0..100 {
        let (pops, pushes) = lin.switch_to(&path);
        assert_eq!(
            (pops, pushes),
            (0, 0),
            "every same-path re-entry must be a no-op"
        );
        local_push += pushes as u64;
        local_pop += pops as u64;
    }
    // Behavioural proof, race-free: this caller's own switch_to
    // calls returned 0 pops and 0 pushes for every re-entry.
    assert_eq!(local_push, 0);
    assert_eq!(local_pop, 0);

    let after = lineage_stats();
    let fp_after = find_counter("lineage_switch_fast_path_count", &after);
    // Counter check: at least 100 fast-path hits contributed by us.
    // Lower-bound rather than equality because other parallel tests
    // in this module also share the global counter.
    assert!(
        fp_after.saturating_sub(fp_before) >= 100,
        "expected ≥100 fast-path hits, got {}",
        fp_after.saturating_sub(fp_before)
    );
    // push_before/pop_before remain unused snapshots in the race-safe
    // version — kept as documentation of the invariant being
    // demonstrated. The behavioural assertions above (local_push,
    // local_pop) are the authoritative check.
    let _ = (push_before, pop_before);
}

/// Test-only mutex serializing the sampler tests. Sampler tests
/// touch the global LAST_SAMPLE_* / LINEAGE_DISMANTLED state and the
/// shared LINEAGE_SWITCH_*/HOT_* counters in ways that would race
/// with each other or with the rest of this module's parallel
/// tests. Take this lock for the duration of any test that calls
/// `sample_for_thrash` or asserts on the dismantle flag.
static SAMPLER_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// `sample_for_thrash` is a no-op on steps that are not a multiple
/// of the sample interval. Pure logic — no counter reads.
#[test]
fn test_sample_for_thrash_off_step_is_noop() {
    let _g = SAMPLER_TEST_LOCK.lock().unwrap();
    reset_dismantle_state_for_test();
    assert!(!is_lineage_dismantled());
    for step in 1..10 {
        let fired = sample_for_thrash(step, 10, 1, 35);
        assert!(!fired, "step {step} should not sample");
    }
    assert!(!is_lineage_dismantled());
}

/// Insufficient switch volume keeps the dismantle flag off even
/// when the hot-ratio over the (tiny) window is zero. Uses a very
/// high `min_switches` value to dwarf any pollution from parallel
/// non-sampler tests in the module.
#[test]
fn test_sample_for_thrash_min_switches_gate() {
    let _g = SAMPLER_TEST_LOCK.lock().unwrap();
    reset_dismantle_state_for_test();
    let mut lin = SharedLineageSolver::new(make_solver());
    let x = bvconst("test_sample_min_switches_x", 8);
    for i in 0..5u64 {
        let path = vec![ScopeFrame::new(true, eq_bv_const(&x, i))];
        lin.switch_to(&path);
    }
    // 1,000,000 min_switches: parallel tests cannot plausibly bump
    // the global to that height during one sample window — keeps
    // this test asserting "below min" deterministically.
    let fired = sample_for_thrash(10, 10, 1_000_000, 35);
    assert!(!fired, "fewer than min_switches must not trigger");
    assert!(!is_lineage_dismantled());
}

/// Cold workload (no hot-cache hits) triggers dismantle once
/// min_switches is met. Generates 200 cold switches; even if
/// parallel tests dump a handful of hot-cache hits into the global
/// during our window, the ratio stays well below 35%.
#[test]
fn test_sample_for_thrash_cold_workload_triggers() {
    let _g = SAMPLER_TEST_LOCK.lock().unwrap();
    reset_dismantle_state_for_test();
    let mut lin = SharedLineageSolver::new(make_solver());
    let x = bvconst("test_sample_cold_x", 8);
    for i in 0..200u64 {
        let path = vec![ScopeFrame::new(true, eq_bv_const(&x, i))];
        lin.switch_to(&path);
    }
    let fired = sample_for_thrash(10, 10, 100, 35);
    assert!(fired, "cold workload must trigger dismantle");
    assert!(is_lineage_dismantled());

    let stats = dismantle_stats();
    let dismantled = find_counter("lineage_dismantled", &stats);
    let count = find_counter("lineage_dismantle_count", &stats);
    assert_eq!(dismantled, 1);
    // dismantle_count is RESET to 0 by reset_dismantle_state_for_test()
    // and the SAMPLER_TEST_LOCK serializes against other tests that
    // would call sample_for_thrash, so this is exact.
    assert_eq!(count, 1);
}

/// Hot workload (every switch hits the fast path after the first)
/// stays above threshold and does NOT trigger dismantle. 1000 hot
/// hits dwarfs any cold-switch pollution from parallel tests.
#[test]
fn test_sample_for_thrash_hot_workload_stays_on() {
    let _g = SAMPLER_TEST_LOCK.lock().unwrap();
    reset_dismantle_state_for_test();
    let mut lin = SharedLineageSolver::new(make_solver());
    let x = bvconst("test_sample_hot_x", 8);
    let path = vec![ScopeFrame::new(true, eq_bv_const(&x, 7))];
    lin.switch_to(&path);
    for _ in 0..1000 {
        lin.switch_to(&path);
    }
    let fired = sample_for_thrash(10, 10, 100, 35);
    assert!(!fired, "hot workload must not trigger dismantle");
    assert!(!is_lineage_dismantled());
}

/// After dismantle has fired, subsequent sampler calls are cheap
/// no-ops (idempotent and don't double-count).
#[test]
fn test_sample_for_thrash_is_idempotent_once_dismantled() {
    let _g = SAMPLER_TEST_LOCK.lock().unwrap();
    reset_dismantle_state_for_test();
    set_lineage_dismantled(true);
    for step in (10..=100).step_by(10) {
        let fired = sample_for_thrash(step, 10, 1, 35);
        assert!(!fired, "already dismantled — must not re-fire");
    }
    let stats = dismantle_stats();
    let count = find_counter("lineage_dismantle_count", &stats);
    assert_eq!(
        count, 0,
        "dismantle_count must not increment on no-op calls"
    );
}

/// `reset_dismantle_state_for_test()` clears the dismantle flag so
/// the sampler can fire again, without disturbing the shared
/// LINEAGE_SWITCH_*/HOT_* counters that other parallel tests
/// snapshot.
#[test]
fn test_reset_dismantle_state_for_test_is_narrow() {
    let _g = SAMPLER_TEST_LOCK.lock().unwrap();
    // Capture counters before reset.
    let pre = lineage_stats();
    let switch_pre = find_counter("lineage_switch_count", &pre);

    set_lineage_dismantled(true);
    assert!(is_lineage_dismantled());
    reset_dismantle_state_for_test();
    assert!(!is_lineage_dismantled());

    // LINEAGE_SWITCH_COUNT must NOT have been reset (other tests
    // depend on it being monotonically nondecreasing). It can only
    // have grown via parallel test contributions.
    let post = lineage_stats();
    let switch_post = find_counter("lineage_switch_count", &post);
    assert!(
        switch_post >= switch_pre,
        "narrow reset must not touch LINEAGE_SWITCH_COUNT"
    );

    let stats = dismantle_stats();
    let count = find_counter("lineage_dismantle_count", &stats);
    assert_eq!(count, 0);
}

/// Per-call overhead of [`tick_and_sample_for_thrash`] (the always-on
/// run-loop hook) must stay well under the angr-v5ht 0.1ms budget,
/// even when the sampler does real work on every sample step. The
/// hook is invoked once per `run_loop` iteration; a bench at the
/// 0.1ms ceiling would alone account for a 10% regression at 1000
/// iterations/s.
///
/// Measures 10k invocations of the production-default hook
/// (`sample_interval=10`, `min_switches=20`, `hot_threshold_pct=35`)
/// and asserts mean per-call cost <50µs. The implementation does at
/// most ~6 atomic loads + 1 fetch_add + 1 multiply + 2 stores on the
/// sample-step path and 1 load + 1 branch on the off-step path —
/// expected actual cost is well under 1µs/call. The 50µs assertion
/// gives a 2x safety margin to the 0.1ms gate while still flagging
/// any future change that orders-of-magnitude regresses the hook
/// (e.g. accidentally taking a lock per call).
#[test]
fn test_sampler_hook_overhead_under_budget() {
    let _g = SAMPLER_TEST_LOCK.lock().unwrap();
    reset_dismantle_state_for_test();
    // Bump LINEAGE_SWITCH_COUNT well past min_switches so the
    // sample-step path lands in the decision branch (not the
    // "accumulate window" early-return).
    let mut lin = SharedLineageSolver::new(make_solver());
    let x = bvconst("test_sampler_hook_overhead_x", 8);
    for i in 0..500u64 {
        let path = vec![ScopeFrame::new(true, eq_bv_const(&x, i))];
        lin.switch_to(&path);
    }

    const N: u64 = 10_000;
    // Re-snapshot LAST_SAMPLE_* so the first sample inside the loop
    // sees a window with switch_delta >= min_switches. Without this,
    // the 500 switches above are folded into the first window only.
    reset_dismantle_state_for_test();

    let start = std::time::Instant::now();
    for _ in 0..N {
        // Bump switch count between calls so each sample window
        // continues to find delta >= min_switches and continues
        // taking the decision-path branch. Without this, after the
        // first decision the window is empty and subsequent calls
        // short-circuit through the "accumulate" branch.
        LINEAGE_SWITCH_COUNT.fetch_add(50, Ordering::Relaxed);
        LINEAGE_SWITCH_HOT_COUNT.fetch_add(30, Ordering::Relaxed);
        // 30/50 = 60% hot, above 35% threshold → no dismantle, so
        // subsequent iterations continue to take the same path.
        let _ = tick_and_sample_for_thrash(10, 20, 35);
    }
    let elapsed_ns = start.elapsed().as_nanos();
    let per_call_ns = elapsed_ns / N as u128;

    assert!(
        !is_lineage_dismantled(),
        "60% hot ratio must not trigger dismantle"
    );
    assert!(
        per_call_ns < 50_000,
        "sampler hook too slow: {per_call_ns} ns/call exceeds 50_000 ns budget \
         (acceptance gate: <100_000 ns per angr-0hdq.2)"
    );
}

/// Two distinct paths that happen to share the same length but
/// diverge in the tail must NOT collide on the fast path (different
/// tail FrameIds).
#[test]
fn test_switch_fast_path_does_not_collide_on_different_tails() {
    let mut lin = SharedLineageSolver::new(make_solver());
    let x = bvconst("test_switch_fast_path_no_collide_x", 16);

    // Two paths of identical length 2, sharing the first frame `a`
    // but diverging in the tail (`c` vs `d`).
    let a = ScopeFrame::new(true, x.bvugt(z3::ast::BV::from_u64(0, 16)));
    let c = ScopeFrame::new(true, eq_bv_const(&x, 5));
    let d = ScopeFrame::new(true, eq_bv_const(&x, 42));

    let path_ac = vec![a.clone(), c];
    let path_ad = vec![a, d];

    lin.switch_to(&path_ac);

    // Switch to a different path that has the same length but a
    // different tail id. Must NOT fire the fast path; the (pops,
    // pushes) return value of (1, 1) is the race-free behavioural
    // proof that the prefix walk did execute and identified the
    // tail-only divergence. (Global counter delta cannot prove
    // negative "did not fire" because other parallel tests in this
    // module also bump the same atomic.)
    let (pops, pushes) = lin.switch_to(&path_ad);
    assert_eq!(
        (pops, pushes),
        (1, 1),
        "different-tail switch must pop+push the diverging frame"
    );
    assert_eq!(lin.loaded_depth(), 2);
}

/// Gate (c) of `SymContext::fork`'s three-gate materialization
/// contract (`invariant-v5ht-dismantle-child-none`): once the runtime
/// thrash detector has dismantled lineage minting, the child gets
/// `None` — NOT an `Arc::clone` of the parent's lineage. The Arc-clone
/// would hand the child a base frozen at some ancestor's mint time,
/// missing every constraint the parent added since, which leaks
/// unconstrained SAT solutions (the baby-re `chr()` repro in angr-v5ht).
///
/// This is the arm gates (a) and (b) have no coverage for: their guards
/// are pinned by `context_tests::smtlib2_snapshot::test_fork_skips_mint_when_flag_off`
/// and `::test_fork_skips_mint_when_bare_push_outstanding`, both of which
/// start from a parent whose own lineage is already `None`, so they
/// cannot tell "refused to mint" apart from "cloned a None".
///
/// Lives here rather than beside its two siblings because it mutates the
/// process-global `LINEAGE_DISMANTLED` flag and so must hold
/// `SAMPLER_TEST_LOCK`.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_fork_drops_lineage_when_dismantled() {
    use crate::symbolic::SymContext;
    use parking_lot::Mutex;
    use std::sync::Arc;

    let _g = SAMPLER_TEST_LOCK.lock().unwrap();
    reset_dismantle_state_for_test();

    // A parent that is opted in AND already holds a lineage: the only
    // configuration where the dismantled and non-dismantled arms differ
    // observably (None vs a non-None Arc).
    let parent = SymContext::new();
    parent.set_use_shared_lineage_solver(true);
    let parent_lin = crate::arc_shared(Mutex::new(SharedLineageSolver::new(make_solver())));
    parent.set_lineage_for_testing(Arc::clone(&parent_lin));
    assert_eq!(parent.bare_z3_push_depth(), 0);

    // Gates (a) and (b) pass and (c) has not fired → fresh mint, which
    // is neither None nor the parent's Arc.
    let child_live = parent.fork();
    let live_arc = child_live
        .lineage_arc()
        .expect("undismantled fork with (a)+(b) satisfied must mint");
    assert!(
        !Arc::ptr_eq(&live_arc, &parent_lin),
        "mint must allocate a fresh SharedLineageSolver, not reuse the parent's"
    );

    // Fire the detector; the same fork must now hand the child None.
    set_lineage_dismantled(true);
    let child_dead = parent.fork();
    assert!(
        child_dead.lineage_arc().is_none(),
        "dismantled fork must drop the lineage, not Arc::clone the parent's stale base"
    );
    // The parent keeps its own lineage either way — the gate only
    // decides what the child gets.
    assert!(
        parent
            .lineage_arc()
            .is_some_and(|a| Arc::ptr_eq(&a, &parent_lin)),
        "fork must not disturb the parent's lineage"
    );

    reset_dismantle_state_for_test();
}
