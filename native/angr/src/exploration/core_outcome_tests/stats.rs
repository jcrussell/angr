//! Step-stats accumulation on [`ParallelProfiling`]: `accumulate_step` must
//! fold every sum-typed interpreter counter, and `drain_into` must reset the
//! accumulator so a long-lived (steady-state) one folds deltas.

use super::*;

/// angr-qhkye: `accumulate_step` must fold EVERY sum-typed interpreter counter
/// (not just cache hits/misses) into the shared accumulator, so the parallel
/// path reports `lift_time_ns` and siblings like the single-threaded
/// `accumulated_stats.merge(&step.step_stats)` does. Guards against the
/// regression where worker-thread `lift_time_ns` silently dropped to 0.
#[test]
fn accumulate_step_folds_full_stats_no_double_count() {
    let prof = ParallelProfiling::default();

    // Two worker dispatches with representative interpreter counters set.
    prof.accumulate_step(&ExecutionStats {
        lift_time_ns: 500,
        cache_hit_count: 3,
        cache_miss_count: 1,
        blocks_executed: 2,
        ..Default::default()
    });
    prof.accumulate_step(&ExecutionStats {
        lift_time_ns: 250,
        cache_hit_count: 4,
        blocks_executed: 1,
        ..Default::default()
    });

    // Independently, the post-step arms fold solver timing into the atomics.
    ParallelProfiling::add(&prof.solver_sat_count, 7);

    let mut stats = ExecutionStats::default();
    prof.fold_into(&mut stats);

    // Sum-typed interpreter counters accumulate across both steps.
    assert_eq!(stats.lift_time_ns, 750);
    assert_eq!(stats.cache_hit_count, 7);
    assert_eq!(stats.cache_miss_count, 1);
    assert_eq!(stats.blocks_executed, 3);
    // Atomic-tracked solver counter still folds (interpreter never sets it, so
    // the full-stats merge cannot double-count it).
    assert_eq!(stats.solver_sat_count, 7);
}

/// `drain_into` must reset the step-stats accumulator so a long-lived
/// (steady-state) accumulator folds deltas, not cumulative totals.
#[test]
fn drain_into_resets_step_stats_accumulator() {
    let prof = ParallelProfiling::default();

    prof.accumulate_step(&ExecutionStats {
        lift_time_ns: 100,
        cache_hit_count: 2,
        ..Default::default()
    });

    let mut first = ExecutionStats::default();
    prof.drain_into(&mut first);
    assert_eq!(first.lift_time_ns, 100);
    assert_eq!(first.cache_hit_count, 2);

    // Second drain with no further accumulation must add zero (the accumulator
    // was reset), not re-add the first step's totals.
    let mut second = ExecutionStats::default();
    prof.drain_into(&mut second);
    assert_eq!(second.lift_time_ns, 0);
    assert_eq!(second.cache_hit_count, 0);
}
