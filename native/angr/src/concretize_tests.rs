// Unit tests for concretize.rs (AddressConcretizer).
// Split out of the parent module; see CLAUDE.md test-split recipe.
use super::*;

#[test]
fn test_concrete_address() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();

    let addr = RustBV::concrete(0x1000, 64);
    let result = concretizer.concretize(&addr, &ctx);

    match result {
        ConcretizationResult::Single(a) => assert_eq!(a, 0x1000),
        _ => panic!("expected Single result"),
    }
}

#[test]
fn test_result_helpers() {
    let single = ConcretizationResult::Single(0x1000);
    assert_eq!(single.addresses(), Some(vec![0x1000]));

    let multi = ConcretizationResult::Multiple(vec![0x1000, 0x1004, 0x1008]);
    assert_eq!(multi.addresses(), Some(vec![0x1000, 0x1004, 0x1008]));

    let strided = ConcretizationResult::Strided {
        base: 0x1000,
        stride: 4,
        count: 10,
    };
    assert_eq!(
        strided.addresses(),
        Some(vec![
            0x1000, 0x1004, 0x1008, 0x100c, 0x1010, 0x1014, 0x1018, 0x101c, 0x1020, 0x1024
        ])
    );

    let failed = ConcretizationResult::Failed("test".to_string());
    assert_eq!(failed.addresses(), None);

    let too_large = ConcretizationResult::TooLarge {
        min: 0x1000,
        max: 0x9000,
        limit: 1024,
    };
    assert_eq!(too_large.addresses(), None);
}

#[test]
fn test_gcd() {
    assert_eq!(AddressConcretizer::gcd(12, 8), 4);
    assert_eq!(AddressConcretizer::gcd(17, 13), 1);
    assert_eq!(AddressConcretizer::gcd(100, 25), 25);
    assert_eq!(AddressConcretizer::gcd(0, 5), 5);
    assert_eq!(AddressConcretizer::gcd(5, 0), 5);
}

#[test]
fn test_stride_detection_from_solutions() {
    let concretizer = AddressConcretizer::new();

    // Perfect stride pattern
    let addrs = vec![0x1000, 0x1004, 0x1008, 0x100c, 0x1010];
    let result = concretizer.detect_stride_from_solutions(&addrs);
    assert!(result.is_some());
    if let Some(ConcretizationResult::Strided {
        base,
        stride,
        count,
    }) = result
    {
        assert_eq!(base, 0x1000);
        assert_eq!(stride, 4);
        assert_eq!(count, 5);
    }

    // Irregular pattern - no stride
    let irregular = vec![0x1000, 0x1004, 0x1010, 0x1020];
    let result = concretizer.detect_stride_from_solutions(&irregular);
    assert!(result.is_none());
}

#[test]
fn test_default_config() {
    let concretizer = AddressConcretizer::default();
    assert_eq!(concretizer.read_range_limit, 1024); // Match Python default
    assert_eq!(concretizer.write_range_limit, 128); // Match Python default
    assert_eq!(concretizer.max_solutions, 256);
    assert_eq!(concretizer.max_stride_count, 16384);
    // This is the *only* stride-detection switch (angr-c7xno.5 removed the
    // disconnected `ExecutionConfig::enable_stride_detection` twin): a caller
    // that wants stride detection off must clear it here.
    assert!(concretizer.enable_stride_detection);
    assert!(!concretizer.use_approximate);
    assert!(!concretizer.symbolic_write_addresses);
    assert!(concretizer.read_fallback_any);
    assert!(concretizer.write_fallback_max);
}

#[test]
fn test_configure() {
    let mut concretizer = AddressConcretizer::default();

    // Configure without approximate
    concretizer.configure(false, Some(2048));
    assert_eq!(concretizer.read_range_limit, 2048);
    assert!(!concretizer.use_approximate);

    // Configure with approximate - should increase range to at least 4096
    concretizer.configure(true, None);
    assert!(concretizer.use_approximate);
    assert!(concretizer.read_range_limit >= 4096);
}

// angr-sqfj8.130: `configure` and `configure_strategies` now share the
// approximate read-range floor. Pin the property that share must not change:
// the legacy `configure` still leaves `write_range_limit` alone (it has no
// write-range parameter), even under `use_approximate`.
#[test]
fn test_configure_applies_read_floor_and_leaves_write_range_untouched() {
    let mut concretizer = AddressConcretizer::default();
    let default_write = concretizer.write_range_limit;

    concretizer.configure(false, Some(2048));
    assert_eq!(concretizer.read_range_limit, 2048);
    assert_eq!(concretizer.write_range_limit, default_write);

    // The approximate floor bumps the read limit but not the write limit.
    concretizer.configure(true, Some(512));
    assert_eq!(concretizer.read_range_limit, 4096);
    assert_eq!(concretizer.write_range_limit, default_write);

    // A limit already above the floor is left as given.
    concretizer.configure(true, Some(8192));
    assert_eq!(concretizer.read_range_limit, 8192);
}

#[test]
fn test_configure_strategies() {
    let mut concretizer = AddressConcretizer::default();

    concretizer.configure_strategies(false, Some(2048), Some(256), true, false, false);
    assert_eq!(concretizer.read_range_limit, 2048);
    assert_eq!(concretizer.write_range_limit, 256);
    assert!(concretizer.symbolic_write_addresses);
    assert!(!concretizer.use_approximate);
    assert!(!concretizer.avoid_multivalued_reads);
    assert!(!concretizer.avoid_multivalued_writes);

    // With approximate, limits should increase to at least 4096
    concretizer.configure_strategies(true, Some(512), Some(128), false, false, false);
    assert!(concretizer.read_range_limit >= 4096);
    assert!(concretizer.write_range_limit >= 4096);

    // Avoid-multivalued flags pass through.
    concretizer.configure_strategies(false, Some(1024), Some(128), false, true, true);
    assert!(concretizer.avoid_multivalued_reads);
    assert!(concretizer.avoid_multivalued_writes);
}

// angr-lf108: try_detect_stride's sampled-GCD grid can exclude feasible
// addresses. `stride_grid_excludes_feasible` is the soundness gate — it must
// return true exactly when some feasible address is off the `{ base+i*stride }`
// grid, so try_detect_stride can fall back to TooLarge instead of emitting an
// unsound Strided result.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_stride_grid_excludes_feasible_off_grid() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let addr = ctx.new_bv("addr", 64);

    // Feasible set {0, 4, 8, 7}: the on-stride members {0,4,8} would tempt a
    // stride=4 grid, but 7 is off-grid (7 % 4 != 0). base = true min = 0.
    let disj = addr
        .eq(&RustBV::concrete(0, 64), &ctx)
        .or(&addr.eq(&RustBV::concrete(4, 64), &ctx), &ctx)
        .or(&addr.eq(&RustBV::concrete(8, 64), &ctx), &ctx)
        .or(&addr.eq(&RustBV::concrete(7, 64), &ctx), &ctx);
    ctx.assume_true(&disj);

    assert!(
        concretizer.stride_grid_excludes_feasible(&addr, &ctx, 0, 4),
        "off-grid feasible address 0x7 must be detected"
    );
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_stride_grid_excludes_feasible_on_grid() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let addr = ctx.new_bv("addr", 64);

    // Feasible set {0, 4, 8} is exactly the stride=4 grid over [0, 8]; no
    // feasible address lies off it, so the gate must not fire.
    let disj = addr
        .eq(&RustBV::concrete(0, 64), &ctx)
        .or(&addr.eq(&RustBV::concrete(4, 64), &ctx), &ctx)
        .or(&addr.eq(&RustBV::concrete(8, 64), &ctx), &ctx);
    ctx.assume_true(&disj);

    assert!(
        !concretizer.stride_grid_excludes_feasible(&addr, &ctx, 0, 4),
        "grid {{0,4,8}} covers the whole feasible set; gate must not fire"
    );
}

// angr-mv08h: the Any/Max TooLarge fallback must pin `addr == chosen` on the
// path, matching Python's AddressConcretizationMixin. Before the fix the
// fallback picked one cell but left the address var ranging over [min,max], so
// a later eval() of inputs on a "found" state could yield path-infeasible
// solutions. After the fix the address has exactly one solution.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_read_fallback_pins_address_to_single_solution() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let addr = ctx.new_bv("a", 64);

    // Constrain to a ~256MB range — far above read_range_limit (1024), so the
    // Range strategy overflows and the Any fallback fires.
    ctx.assume_true(&addr.uge(&RustBV::concrete(0x100000, 64), &ctx));
    ctx.assume_true(&addr.ult(&RustBV::concrete(0x10100000, 64), &ctx));

    // Sanity: pre-fallback the address genuinely has many solutions.
    assert!(
        ctx.eval_upto(&addr, 5).len() > 1,
        "precondition: wide range must admit multiple addresses"
    );

    let result = concretizer.concretize_read(&addr, &ctx);
    let chosen = match result {
        ConcretizationResult::Single(a) => a,
        other => panic!("expected Single fallback, got {other:?}"),
    };

    // The pin collapses the address to exactly one solution == chosen.
    let sols = ctx.eval_upto(&addr, 5);
    assert_eq!(
        sols,
        vec![chosen as u128],
        "fallback must pin addr == chosen"
    );
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_write_fallback_pins_address_to_single_solution() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let addr = ctx.new_bv("w", 64);

    ctx.assume_true(&addr.uge(&RustBV::concrete(0x100000, 64), &ctx));
    ctx.assume_true(&addr.ult(&RustBV::concrete(0x10100000, 64), &ctx));

    let result = concretizer.concretize_write(&addr, &ctx);
    let chosen = match result {
        ConcretizationResult::Single(a) => a,
        other => panic!("expected Single fallback, got {other:?}"),
    };
    // Max fallback picks the range maximum; the pin makes it the unique solution.
    let sols = ctx.eval_upto(&addr, 5);
    assert_eq!(
        sols,
        vec![chosen as u128],
        "write fallback must pin addr == chosen"
    );
}

/// angr-9ke6b.194: with `SYMBOLIC_WRITE_ADDRESSES` off, Python's write chain
/// is `Range(128, filter=_multiwrite_filter)` → `Max()`. An address that
/// carries no `MultiwriteAnnotation` fails the filter, so the *only* strategy
/// that runs is `Max()` — the store lands on a single address (the maximum
/// satisfying one), never on a candidate set. Rust used to run the Range
/// strategy unconditionally and fan the store out to every candidate.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn unannotated_write_without_symbolic_write_addresses_is_max_only() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    assert!(!concretizer.symbolic_write_addresses, "precondition");

    // Three satisfying addresses, well inside write_range_limit (128), so the
    // Range strategy would happily return Multiple if it were in the chain.
    let addr = RustBV::symbolic(&ctx, "unannotated_w", 64);
    let a0 = addr.eq(&RustBV::concrete(0x1000, 64), &ctx);
    let a1 = addr.eq(&RustBV::concrete(0x1004, 64), &ctx);
    let a2 = addr.eq(&RustBV::concrete(0x1008, 64), &ctx);
    ctx.assume_true(&a0.or(&a1, &ctx).or(&a2, &ctx));

    match concretizer.concretize_write(&addr, &ctx) {
        ConcretizationResult::Single(a) => {
            assert_eq!(a, 0x1008, "Max() must pick the maximum satisfying address")
        }
        other => panic!("expected Single(0x1008) from the Max-only chain, got {other:?}"),
    }
}

/// The complement: a `MultiwriteAnnotation`-tagged address (routed in through
/// `store_symbolic_unified_multi`) passes `_multiwrite_filter`, so Range stays
/// in the chain even with `SYMBOLIC_WRITE_ADDRESSES` off and the store keeps
/// all its candidates.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn multiwrite_annotated_write_keeps_the_range_strategy() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();

    let addr = RustBV::symbolic(&ctx, "annotated_w", 64);
    let a0 = addr.eq(&RustBV::concrete(0x1000, 64), &ctx);
    let a1 = addr.eq(&RustBV::concrete(0x1001, 64), &ctx);
    let a2 = addr.eq(&RustBV::concrete(0x1003, 64), &ctx);
    ctx.assume_true(&a0.or(&a1, &ctx).or(&a2, &ctx));

    // {0, 1, 3} deltas -> GCD 1, so stride detection declines and the result
    // is a genuine Multiple rather than Strided.
    match concretizer.concretize_write_multiwrite(&addr, &ctx) {
        ConcretizationResult::Multiple(addrs) => {
            assert_eq!(addrs, vec![0x1000, 0x1001, 0x1003]);
        }
        other => panic!("expected Multiple for an annotated write, got {other:?}"),
    }
}

/// `SYMBOLIC_WRITE_ADDRESSES` on restores the Range strategy for *every*
/// write, annotated or not — the option is the global equivalent of the
/// annotation.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn symbolic_write_addresses_on_restores_range_for_unannotated_writes() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer {
        symbolic_write_addresses: true,
        ..AddressConcretizer::new()
    };

    let addr = RustBV::symbolic(&ctx, "swa_on_w", 64);
    let a0 = addr.eq(&RustBV::concrete(0x2000, 64), &ctx);
    let a1 = addr.eq(&RustBV::concrete(0x2001, 64), &ctx);
    let a2 = addr.eq(&RustBV::concrete(0x2003, 64), &ctx);
    ctx.assume_true(&a0.or(&a1, &ctx).or(&a2, &ctx));

    match concretizer.concretize_write(&addr, &ctx) {
        ConcretizationResult::Multiple(addrs) => {
            assert_eq!(addrs, vec![0x2000, 0x2001, 0x2003]);
        }
        other => panic!("expected Multiple with SYMBOLIC_WRITE_ADDRESSES on, got {other:?}"),
    }
}

/// The Max-only chain still pins `addr == chosen` so downstream loads and the
/// exported solver agree on where the store landed.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn max_only_write_pins_the_chosen_address() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();

    let addr = RustBV::symbolic(&ctx, "max_only_pin", 64);
    let a0 = addr.eq(&RustBV::concrete(0x4000, 64), &ctx);
    let a1 = addr.eq(&RustBV::concrete(0x4008, 64), &ctx);
    ctx.assume_true(&a0.or(&a1, &ctx));

    let chosen = match concretizer.concretize_write(&addr, &ctx) {
        ConcretizationResult::Single(a) => a,
        other => panic!("expected Single, got {other:?}"),
    };
    assert_eq!(chosen, 0x4008);
    assert_eq!(
        ctx.solutions(&addr, 4),
        vec![chosen as u128],
        "Max() must pin addr == chosen, like the TooLarge fallback does"
    );
}

// The sibling hazard — a set truncated by a Z3 *timeout* rather than by the cap
// (angr-03vl4.85) — is covered by
// `test_concretize_refuses_timeout_truncated_solution_set` in
// `symbolic/context_tests/solver.rs`, which lives there to reuse that file's
// budget-stalling rig.
//
// angr-03vl4.77: `concretize_internal`'s "range is manageable" branch used to
// ask for exactly `max_solutions` and hand the result back as `Multiple`.
// `solutions()` returns `min(n, #feasible)` with no truncation signal, so a
// feasible set larger than the cap came back as a silently-truncated address
// list that every consumer (ITE load default arm, store disjunction hoist,
// `invalidate_code_for_concretization`) treats as exhaustive. It must now
// report `TooLarge` instead.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_over_max_solutions_is_too_large_not_truncated_multiple() {
    let ctx = SymContext::new_mock();
    let mut concretizer = AddressConcretizer::new();
    concretizer.max_solutions = 20;
    // Contiguous set → gcd of differences is 1, so no Strided result; disabled
    // explicitly so the assertion does not depend on which samples Z3 picks.
    concretizer.enable_stride_detection = false;

    // 33 feasible addresses over a 32-byte range: past FAST_ENUM_LIMIT (16) so
    // the range path runs, but well inside read_range_limit (1024) so the
    // "manageable" branch — not the over-range one — is what fires.
    let addr = ctx.new_bv("trunc", 64);
    ctx.assume_true(&addr.uge(&RustBV::concrete(0x1000, 64), &ctx));
    ctx.assume_true(&addr.ule(&RustBV::concrete(0x1020, 64), &ctx));

    match concretizer.concretize(&addr, &ctx) {
        ConcretizationResult::TooLarge { min, max, .. } => {
            assert_eq!((min, max), (0x1000, 0x1020));
        }
        other => panic!("expected TooLarge for a set over max_solutions, got {other:?}"),
    }
}

// Companion: with the cap above the feasible count the same address still
// enumerates, and the returned set is complete (not clipped at the cap).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_under_max_solutions_still_enumerates_completely() {
    let ctx = SymContext::new_mock();
    let mut concretizer = AddressConcretizer::new();
    concretizer.max_solutions = 64;
    concretizer.enable_stride_detection = false;

    let addr = ctx.new_bv("full", 64);
    ctx.assume_true(&addr.uge(&RustBV::concrete(0x1000, 64), &ctx));
    ctx.assume_true(&addr.ule(&RustBV::concrete(0x1020, 64), &ctx));

    match concretizer.concretize(&addr, &ctx) {
        ConcretizationResult::Multiple(addrs) => {
            assert_eq!(addrs.len(), 33, "whole feasible set must come back");
            assert_eq!(addrs[0], 0x1000);
            assert_eq!(addrs[32], 0x1020);
        }
        other => panic!("expected Multiple, got {other:?}"),
    }
}

#[test]
fn strided_addrs_wraps_at_the_top_of_the_address_space() {
    assert_eq!(strided_addrs(0x1000, 4, 3), vec![0x1000, 0x1004, 0x1008]);
    assert_eq!(strided_addrs(0x1000, 4, 0), Vec::<u64>::new());
    // Base near the top: the third address wraps rather than panicking under
    // `--profile release-checked` (angr-xloth.3).
    assert_eq!(
        strided_addrs(u64::MAX - 3, 2, 4),
        vec![u64::MAX - 3, u64::MAX - 1, 0, 2]
    );
    // A huge stride wraps in the multiply too.
    assert_eq!(strided_addrs(0, u64::MAX, 3), vec![0, u64::MAX, u64::MAX - 1]);
}
