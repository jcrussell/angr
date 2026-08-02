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

/// Serializes *every* `MARK` bump in this file against every other. A test
/// that reads a process-global stat while some other test's marker is live
/// misattributes that marker to its own bump: the exclusion test would see the
/// sum test's marker folded into `z3_saved_check_total`, and the ticker test —
/// which scans the *whole* stats map for any `>= MARK` value — would report a
/// leak for whichever counter `test_measurement_counters_reach_their_stats_key`
/// happened to be bumping. So the rule is: hold this lock for the entire window
/// in which a `MARK` is applied, not just in the two tests that read the total.
/// Poison-tolerant: a panic in one must not cascade into the others.
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

/// Every measurement counter is emitted under exactly the key the table
/// names, and the key strings are the ones a `--counters-json` capture (and
/// `bench_diff.py`) expects.
///
/// The table is the single source of truth for both emit and reset, so this
/// is where a typo'd key string would surface — nothing else in the crate
/// looks these up. `ZEXT_CMP_TRIVIAL_DECIDE_COUNT` already carries a
/// `_count` suffix its neighbours lack, which is exactly the confusion this
/// pins.
#[test]
fn test_measurement_counter_keys_are_stable() {
    let keys: Vec<&str> = MEASUREMENT_COUNTERS.iter().map(|(k, _)| *k).collect();
    assert_eq!(
        keys,
        [
            "zext_cmp_collapse_count",
            "zext_cmp_trivial_decide_count",
            "rust_export_sound_clz",
            "rust_export_unconstrained_clz",
            "rust_export_unconstrained_fp",
            "rustbv_commutative_canonicalize_count",
            "rustbv_commutative_swap_count",
            "claripy_ast_cache_hit_count",
            "claripy_ast_cache_miss_count",
            "add_constraint_raw_dedup_scanned",
            "add_constraint_raw_dedup_hit",
            "z3_assume_dedup_scanned",
            "z3_assume_dedup_hit",
        ],
        "measurement counter key names are a public capture-format contract"
    );
    let unique: std::collections::HashSet<&str> = keys.iter().copied().collect();
    assert_eq!(
        unique.len(),
        keys.len(),
        "a duplicate key would collapse two counters into one entry"
    );
}

/// Each table entry's atomic really reaches `get_solver_stats()` under its
/// own key — i.e. the emit loop is wired to `MEASUREMENT_COUNTERS` and the
/// pairing inside the table is correct.
///
/// Marker-bump rather than a delta: these counters are process-global and a
/// concurrently-running sibling test may bump them, which can only raise the
/// observed value. See `MARK` / `undo_bump`.
///
/// Takes `marker_guard()` for the whole loop: these bumps land on real emitted
/// counters, so a marker live here is visible to any sibling reading the stats
/// map — see `MARKER_TEST_LOCK`.
#[test]
fn test_measurement_counters_reach_their_stats_key() {
    let _serial = marker_guard();
    for (key, counter) in MEASUREMENT_COUNTERS {
        counter.fetch_add(MARK, Ordering::Relaxed);
        let observed = get_solver_stats().get(key).copied();
        undo_bump(counter);
        assert!(
            observed.is_some_and(|v| v >= MARK),
            "{key} missing from get_solver_stats() or not wired to its atomic (got {observed:?})"
        );
    }
}

/// The emit half of the table loop: always-emit semantics, zeros included,
/// so a capture can distinguish "path never fired" from "key dropped".
///
/// Driven through local atomics, so it is deterministic under the parallel
/// runner — the same pure-seam technique as `insert_query_class_stats`.
#[test]
fn test_insert_measurement_stats_emits_every_pair() {
    let quiet = AtomicU64::new(0);
    let busy = AtomicU64::new(9);
    let mut stats: HashMap<String, u64> = HashMap::new();
    insert_measurement_stats(&mut stats, &[("quiet", &quiet), ("busy", &busy)]);

    assert_eq!(stats.len(), 2, "every pair must emit, got {stats:?}");
    assert_eq!(stats.get("quiet"), Some(&0));
    assert_eq!(stats.get("busy"), Some(&9));
}

/// The reset half: `reset_measurement_counters` zeroes every entry it is
/// given, leaving nothing behind for the next capture.
///
/// Combined with `reset_solver_stats()` passing the very same
/// `MEASUREMENT_COUNTERS` that `get_solver_stats()` emits from, this is what
/// rules out the "counter emitted but its `.store(0, ..)` was dropped" bug —
/// checking that by actually calling `reset_solver_stats()` here is not an
/// option, see the note at the bottom of this file.
#[test]
fn test_reset_measurement_counters_zeroes_every_entry() {
    let a = AtomicU64::new(7);
    let b = AtomicU64::new(u64::MAX);
    let c = AtomicU64::new(0);
    reset_measurement_counters(&[("a", &a), ("b", &b), ("c", &c)]);

    for (name, counter) in [("a", &a), ("b", &b), ("c", &c)] {
        assert_eq!(
            counter.load(Ordering::Relaxed),
            0,
            "{name} must be cleared by reset_measurement_counters"
        );
    }
}

/// `SIMPLIFY_SAMPLE_TICKER` is the one declared-and-reset counter in `stats.rs`
/// with no `stats.insert(...)`, because it is sampling *phase* rather than a
/// measurement — see its doc comment. Pins that so the asymmetry cannot be
/// "fixed" by someone auditing declared-vs-emitted counters, and pins the two
/// keys the doc points at instead as the real denominator/numerator.
///
/// Marker-bump: bump the ticker far beyond anything a sibling test produces and
/// assert no emitted value carries it, so a new key under any spelling fails.
#[test]
fn test_simplify_sample_ticker_is_not_emitted() {
    let _serial = marker_guard();
    SIMPLIFY_SAMPLE_TICKER.fetch_add(MARK, Ordering::Relaxed);
    let stats = get_solver_stats();
    undo_bump(&SIMPLIFY_SAMPLE_TICKER);

    // Nanosecond keys are excluded: they are wall-clock sums, and a long enough
    // suite run could legitimately exceed MARK (~18 min) without any leak.
    let leaked: Vec<&String> = stats
        .iter()
        .filter(|(k, v)| **v >= MARK && !k.contains("time") && !k.ends_with("_ns"))
        .map(|(k, _)| k)
        .collect();
    assert!(
        leaked.is_empty(),
        "SIMPLIFY_SAMPLE_TICKER must stay unemitted, but it reached {leaked:?}"
    );
    for key in ["add_constraint_raw_total", "branch_cond_simplify_sampled"] {
        assert!(
            stats.contains_key(key),
            "{key} is the emitted stand-in the ticker's doc comment defers to"
        );
    }
}

// NOTE: this file deliberately does NOT call `reset_solver_stats()`. Sibling
// tests (`context_tests/constraints.rs`, `syscalls/fd_io_tests.rs`,
// `memory/tests/symbolic.rs`, `context_tests/solver.rs`, ...) delta-assert
// process-global counters with a plain `after - before` or `after > before`,
// any of which a reset landing inside their window would break — and they are
// spread across six files, so no lock short of a crate-wide one would make it
// safe. Reset is instead covered structurally (angr-9ke6b.150): both halves
// run off `MEASUREMENT_COUNTERS`, and the loop itself is tested above against
// local atomics.
