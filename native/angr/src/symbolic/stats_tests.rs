//! File-local tests for `stats.rs`'s own aggregation logic (angr-9ke6b.146).
//!
//! Every other exercise of `get_solver_stats()` lives in a sibling module and
//! drives the counters *indirectly*, through whichever Rust op happens to bump
//! the underlying atomics. Nothing pinned the derivation this file owns: the
//! per-`CheckSite` key-presence filter, the always-emit query-class loop, and
//! the `z3_saved_check_total` sum.
//!
//! The two loops are tested against synthetic inputs via the
//! `insert_check_site_stats` / `insert_query_class_stats` helpers, so they are
//! fully deterministic. `z3_saved_check_total` has no such seam — it reads the
//! process-global atomics directly — so it is tested by bumping each summand by
//! a marker far larger than anything a sibling test can produce, observing the
//! total, and restoring the counter. See `total_while_bumped` for how that stays
//! race-tolerant under the parallel test runner.

use super::*;

/// A bump large enough that no concurrent test can manufacture it, so
/// "the total moved by at least MARK" implies *our* counter is a summand.
/// Restored after each observation, so the marker never leaks into another
/// test's reading.
const MARK: u64 = 1 << 40;

/// Serializes the two marker-bump tests against each other. They both read
/// `z3_saved_check_total` while a `MARK` is live, so running concurrently the
/// exclusion test would observe the *sum* test's marker and conclude
/// `z3_ast_memo_hit` had been folded in. Poison-tolerant: a panic in one must
/// not cascade into the other.
static MARKER_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Take `MARKER_TEST_LOCK` for the duration of a marker-bump window.
fn marker_guard() -> std::sync::MutexGuard<'static, ()> {
    MARKER_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Subtract a previously-applied `MARK`, saturating at 0.
///
/// A sibling test may call `reset_solver_stats()` between our bump and this
/// restore; a plain `fetch_sub` would then wrap the counter to ~u64::MAX and
/// poison every later reader.
fn undo_bump(counter: &AtomicU64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
        Some(v.saturating_sub(MARK))
    });
}

/// Bump `counter` by `MARK`, read `z3_saved_check_total` while the bump is
/// live, then restore. Returns the largest total observed.
///
/// Retries a few times because a concurrent `reset_solver_stats()` landing
/// inside the window would zero the total we are about to read. Retrying can
/// only ever *raise* the observed value, so a counter that genuinely is not a
/// summand still never reaches `MARK`.
fn total_while_bumped(counter: &AtomicU64) -> u64 {
    let mut best = 0u64;
    for _ in 0..4 {
        counter.fetch_add(MARK, Ordering::Relaxed);
        let observed = get_solver_stats()
            .get("z3_saved_check_total")
            .copied()
            .expect("z3_saved_check_total must always be emitted");
        undo_bump(counter);
        best = best.max(observed);
        if best >= MARK {
            break;
        }
    }
    best
}

/// A site with count 0 contributes no keys at all — not even a zeroed
/// `_time_ns` — while a site that fired contributes exactly its two.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_check_site_stats_omit_untouched_sites() {
    let mut stats: HashMap<String, u64> = HashMap::new();
    insert_check_site_stats(
        &mut stats,
        &[0, 5, 0],
        &[7, 11, 13],
        &["alpha", "beta", "gamma"],
    );

    assert_eq!(
        stats.len(),
        2,
        "only the one site with count>0 may contribute keys, got {stats:?}"
    );
    assert_eq!(stats.get("z3_site_beta_count"), Some(&5));
    assert_eq!(stats.get("z3_site_beta_time_ns"), Some(&11));
    for absent in [
        "z3_site_alpha_count",
        "z3_site_alpha_time_ns",
        "z3_site_gamma_count",
        "z3_site_gamma_time_ns",
    ] {
        assert!(
            !stats.contains_key(absent),
            "untouched site key {absent} must be omitted, not emitted as 0"
        );
    }
}

/// A site that fired but accumulated 0 ns still emits both keys — the filter
/// is on the count, not the time.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_check_site_stats_emit_zero_time_when_count_nonzero() {
    let mut stats: HashMap<String, u64> = HashMap::new();
    insert_check_site_stats(&mut stats, &[3], &[0], &["fast"]);

    assert_eq!(stats.get("z3_site_fast_count"), Some(&3));
    assert_eq!(
        stats.get("z3_site_fast_time_ns"),
        Some(&0),
        "a sub-nanosecond site must still report its time key"
    );
}

/// The query-class loop is unconditional: every class key is present even
/// when the classifier never fired, so a counters-json capture can tell
/// "classifier off" from "key dropped".
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_query_class_stats_always_emitted() {
    let mut stats: HashMap<String, u64> = HashMap::new();
    insert_query_class_stats(&mut stats, &[0, 0, 4], &["unclassified", "quiet", "busy"]);

    assert_eq!(stats.len(), 3, "every class must emit, got {stats:?}");
    assert_eq!(stats.get("z3_check_class_unclassified"), Some(&0));
    assert_eq!(stats.get("z3_check_class_quiet"), Some(&0));
    assert_eq!(stats.get("z3_check_class_busy"), Some(&4));
}

/// The real name tables must line up with their counter arrays and carry no
/// duplicates — a repeated name would silently collapse two buckets into one
/// key, and the survivor would be whichever the loop visited last.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_bucket_name_tables_are_consistent() {
    assert_eq!(SITE_NAMES.len(), NUM_CHECK_SITES);
    assert_eq!(Z3_CHECK_SITE_COUNT.len(), NUM_CHECK_SITES);
    assert_eq!(Z3_CHECK_SITE_TIME_NS.len(), NUM_CHECK_SITES);
    let unique: std::collections::HashSet<&str> = SITE_NAMES.iter().copied().collect();
    assert_eq!(
        unique.len(),
        NUM_CHECK_SITES,
        "duplicate name in SITE_NAMES"
    );

    use crate::symbolic::query_class::{CLASS_NAMES, NUM_QUERY_CLASSES, Z3_CHECK_CLASS_COUNT};
    assert_eq!(CLASS_NAMES.len(), NUM_QUERY_CLASSES);
    assert_eq!(Z3_CHECK_CLASS_COUNT.len(), NUM_QUERY_CLASSES);
    let unique: std::collections::HashSet<&str> = CLASS_NAMES.iter().copied().collect();
    assert_eq!(
        unique.len(),
        NUM_QUERY_CLASSES,
        "duplicate name in CLASS_NAMES"
    );
}

/// Each of the six check-avoidance families is wired into
/// `z3_saved_check_total`, and each is also surfaced under its own raw key.
///
/// Catches a dropped summand (someone edits the sum and loses a family) and a
/// typo'd raw key, neither of which any other test would notice.
#[test]
fn test_saved_check_total_sums_every_family() {
    let _serial = marker_guard();
    for (counter, raw_key) in [
        (&Z3_BRANCH_CONCRETE_COUNT, "z3_branch_concrete"),
        (&Z3_ASSUME_CONCRETE_COUNT, "z3_assume_concrete"),
        (
            &ZEXT_CMP_TRIVIAL_DECIDE_COUNT,
            "zext_cmp_trivial_decide_count",
        ),
        (&Z3_BRANCH_MODEL_HIT_COUNT, "z3_branch_model_hit"),
        (&Z3_EXTREMA_MODEL_HIT_COUNT, "z3_extrema_model_hit"),
        (&Z3_EVAL_UPTO_MODEL_HIT_COUNT, "z3_eval_upto_model_hit"),
    ] {
        assert!(
            total_while_bumped(counter) >= MARK,
            "{raw_key} must be a summand of z3_saved_check_total"
        );

        counter.fetch_add(MARK, Ordering::Relaxed);
        let raw = get_solver_stats().get(raw_key).copied();
        undo_bump(counter);
        assert!(
            raw.is_some_and(|v| v >= MARK),
            "raw counter key {raw_key} missing or not wired to its atomic (got {raw:?})"
        );
    }
}

/// The sum deliberately excludes `z3_ast_memo_hit`: it saves Z3 *node
/// construction*, not a `check()`, and is an order of magnitude larger, so
/// folding it in would drown the check-avoidance signal the key exists to
/// carry. Pins that exclusion so it cannot be "fixed" by accident.
#[test]
fn test_saved_check_total_excludes_ast_memo_hit() {
    let _serial = marker_guard();
    Z3_AST_MEMO_HIT_COUNT.fetch_add(MARK, Ordering::Relaxed);
    let total = get_solver_stats()
        .get("z3_saved_check_total")
        .copied()
        .expect("z3_saved_check_total must always be emitted");
    undo_bump(&Z3_AST_MEMO_HIT_COUNT);
    assert!(
        total < MARK,
        "z3_ast_memo_hit must NOT be folded into z3_saved_check_total (total={total})"
    );
}

// NOTE: this file deliberately does NOT call `reset_solver_stats()`. Sibling
// tests (`context_tests/constraints.rs`) delta-assert process-global counters
// with a plain `after - before`, which underflows if a reset lands inside
// their window. Reset coverage belongs with the counter-wiring work
// (angr-9ke6b.150), where it can be serialized against those tests.
