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
    assert!(single.is_success());
    assert_eq!(single.single(), Some(0x1000));
    assert_eq!(single.addresses(), Some(vec![0x1000]));
    assert!(!single.is_strided());

    let multi = ConcretizationResult::Multiple(vec![0x1000, 0x1004, 0x1008]);
    assert!(multi.is_success());
    assert_eq!(multi.single(), None);
    assert_eq!(multi.addresses(), Some(vec![0x1000, 0x1004, 0x1008]));
    assert!(!multi.is_strided());

    let strided = ConcretizationResult::Strided {
        base: 0x1000,
        stride: 4,
        count: 10,
    };
    assert!(strided.is_success());
    assert!(strided.is_strided());
    assert_eq!(strided.strided_params(), Some((0x1000, 4, 10)));
    assert_eq!(
        strided.addresses(),
        Some(vec![
            0x1000, 0x1004, 0x1008, 0x100c, 0x1010, 0x1014, 0x1018, 0x101c, 0x1020, 0x1024
        ])
    );

    let failed = ConcretizationResult::Failed("test".to_string());
    assert!(!failed.is_success());
    assert_eq!(failed.addresses(), None);
}

#[test]
fn test_offset_adjustment() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();

    let base = RustBV::concrete(0x8000, 64);
    let result = concretizer.concretize_with_offset(&base, -0x100, &ctx);

    match result {
        ConcretizationResult::Single(a) => assert_eq!(a, 0x7F00),
        _ => panic!("expected Single result"),
    }
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
    assert_eq!(concretizer.max_range, 1024); // Legacy compatibility
    assert_eq!(concretizer.max_solutions, 256);
    assert_eq!(concretizer.max_stride_count, 16384);
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
