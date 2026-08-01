use super::super::*;

/// Returns true if `bv` is an `Expression` whose top-level op is `Ite`.
fn is_ite(bv: &RustBV) -> bool {
    use crate::symbolic::BVOp;
    matches!(bv, RustBV::Expression { op: BVOp::Ite, .. })
}

// angr-269l: de-duplicate ITE arms in concretization fan-out. When every
// candidate address maps to the same loaded content (zero pages, repeated
// initializers), the balanced ITE tree should collapse to a single leaf
// instead of emitting log-depth ITE nodes that Z3's max-bv-sharing tactic
// cannot dedup (it matches by AST node identity, not structural equality).

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_ite_dedup_zero_page_load() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();

    // Zero page: never written, so every aligned load returns Concrete(0).
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // Symbolic 64-bit addr constrained to {0x1000, 0x1010, 0x1020, 0x1030}
    // — four candidates, all backed by the same zero content.
    let addr = RustBV::symbolic(&ctx, "zp_addr", 64);
    let mut clause = addr.eq(&RustBV::concrete(0x1000, 64), &ctx);
    for off in [0x10u64, 0x20, 0x30] {
        let eq = addr.eq(&RustBV::concrete((0x1000 + off) as u128, 64), &ctx);
        clause = clause.or(&eq, &ctx);
    }
    ctx.assume_true(&clause);
    assert!(ctx.is_sat(), "four-solution constraint must be SAT");

    let loaded = mem
        .load_symbolic_unified(addr, 8, &ctx, &concretizer)
        .expect("zero-page load must succeed");

    // Dedup contract: identical zero loads collapse to a single Concrete leaf.
    assert!(
        !is_ite(&loaded),
        "expected dedup → single leaf, got ITE: {loaded:?}"
    );
    assert_eq!(
        loaded.as_u64(),
        Some(0),
        "all candidates load zero → result must be Concrete(0)"
    );
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_ite_dedup_repeated_initializer_collapses() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();

    // Repeated initializer: four 8-byte slots, all storing the same word.
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);
    for off in [0u64, 0x10, 0x20, 0x30] {
        mem.store_concrete(0x1000 + off, RustBV::concrete(0xDEAD_BEEF, 64))
            .expect("store");
    }

    let addr = RustBV::symbolic(&ctx, "ri_addr", 64);
    let mut clause = addr.eq(&RustBV::concrete(0x1000, 64), &ctx);
    for off in [0x10u64, 0x20, 0x30] {
        let eq = addr.eq(&RustBV::concrete((0x1000 + off) as u128, 64), &ctx);
        clause = clause.or(&eq, &ctx);
    }
    ctx.assume_true(&clause);
    assert!(ctx.is_sat(), "four-solution constraint must be SAT");

    let loaded = mem
        .load_symbolic_unified(addr, 8, &ctx, &concretizer)
        .expect("repeated-initializer load must succeed");

    assert!(
        !is_ite(&loaded),
        "expected dedup → single leaf, got ITE: {loaded:?}"
    );
    assert_eq!(loaded.as_u64(), Some(0xDEAD_BEEF));
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_ite_dedup_distinct_values_keeps_ite() {
    // Control test: when candidate addresses load distinct values, the
    // dedup must NOT collapse — we must still see an ITE at the root.
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();

    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);
    mem.store_concrete(0x1000, RustBV::concrete(0x11, 64))
        .expect("store a");
    mem.store_concrete(0x1010, RustBV::concrete(0x22, 64))
        .expect("store b");

    let addr = RustBV::symbolic(&ctx, "dv_addr", 64);
    let eq_a = addr.eq(&RustBV::concrete(0x1000, 64), &ctx);
    let eq_b = addr.eq(&RustBV::concrete(0x1010, 64), &ctx);
    ctx.assume_true(&eq_a.or(&eq_b, &ctx));
    assert!(ctx.is_sat());

    let loaded = mem
        .load_symbolic_unified(addr, 8, &ctx, &concretizer)
        .expect("distinct-values load must succeed");

    // Distinct contents → ITE is required. Verifies the dedup is conservative.
    assert!(
        is_ite(&loaded),
        "distinct candidate values must keep the ITE, got: {loaded:?}"
    );
}

#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_ite_dedup_strided_zero_collapses() {
    // Strided variant: a stride-aligned region of zeros routes through
    // `load_strided_balanced` / `build_strided_ite_tree`. The recursive
    // sibling-collapse must propagate up to a single leaf.
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();

    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);
    // Page is never written, so every offset is zero.

    // Constrain addr to a regular stride of 8 across four candidates; the
    // concretizer should classify this as Strided.
    let addr = RustBV::symbolic(&ctx, "st_addr", 64);
    let mut clause = addr.eq(&RustBV::concrete(0x1000, 64), &ctx);
    for off in [0x8u64, 0x10, 0x18] {
        let eq = addr.eq(&RustBV::concrete((0x1000 + off) as u128, 64), &ctx);
        clause = clause.or(&eq, &ctx);
    }
    ctx.assume_true(&clause);
    assert!(ctx.is_sat());

    let loaded = mem
        .load_symbolic_unified(addr, 8, &ctx, &concretizer)
        .expect("strided zero load must succeed");

    assert!(
        !is_ite(&loaded),
        "strided zero load expected to collapse to a leaf, got: {loaded:?}"
    );
    assert_eq!(loaded.as_u64(), Some(0));
}

// ============================================================================
// angr-62li: address-concretization disjunction hoisting. The `Multiple`
// arm of `store_symbolic_unified`, `store_with_concretization`,
// `store_symbolic_unified_multi`, and `load_symbolic_unified` now asserts
// `Or(addr == a0, ..., addr == aK)` to the Rust solver so Z3's
// propagate-values tactic sees the addr domain restriction. Strided is
// deliberately *not* hoisted in this pass.
// ============================================================================

/// A `Multiple` write must hoist the disjunction: the counter bumps and the
/// solver gains a constraint that restricts the addr's domain to the
/// candidate set.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_assert_address_disjunction_multiple_write_hoists() {
    use crate::concretize::AddressConcretizer;
    use crate::symbolic::get_solver_stats;

    let pre = get_solver_stats();
    let pre_count = pre
        .get("concretize_disjunction_count")
        .copied()
        .unwrap_or(0);
    let pre_terms = pre
        .get("concretize_disjunction_terms_total")
        .copied()
        .unwrap_or(0);

    let ctx = SymContext::new_mock();
    // SYMBOLIC_WRITE_ADDRESSES on so the Range strategy applies; the default
    // Max-only chain resolves an unannotated symbolic address to one value and
    // hoists no disjunction (angr-9ke6b.194).
    let concretizer = AddressConcretizer {
        symbolic_write_addresses: true,
        ..AddressConcretizer::new()
    };
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // Two non-strided candidates so the concretizer returns Multiple, not
    // Strided. 0x1000 and 0x1080 with a single anchor in between excludes
    // strided detection.
    let addr = RustBV::symbolic(&ctx, "dj_addr", 64);
    let a0 = addr.eq(&RustBV::concrete(0x1000, 64), &ctx);
    let a1 = addr.eq(&RustBV::concrete(0x1080, 64), &ctx);
    let a2 = addr.eq(&RustBV::concrete(0x1300, 64), &ctx);
    ctx.assume_true(&a0.or(&a1, &ctx).or(&a2, &ctx));
    assert!(ctx.is_sat());

    let value = RustBV::concrete(0xCAFE_BABE, 32);
    mem.store_symbolic_unified(addr.clone(), value, &ctx, &concretizer)
        .expect("multi store must succeed");

    let post = get_solver_stats();
    let post_count = post.get("concretize_disjunction_count").copied().unwrap();
    let post_terms = post
        .get("concretize_disjunction_terms_total")
        .copied()
        .unwrap();
    assert!(
        post_count > pre_count,
        "expected concretize_disjunction_count to increment (pre={pre_count}, post={post_count})"
    );
    assert!(
        post_terms >= pre_terms + 3,
        "expected at least 3 terms added (pre={pre_terms}, post={post_terms})"
    );

    // After the hoist, addr is provably NOT 0x9999 (outside the candidate
    // set). Without the hoist Z3 would have permitted that branch.
    let outside = addr.eq(&RustBV::concrete(0x9999, 64), &ctx);
    assert!(
        !ctx.can_be_true(&outside),
        "addr must not be able to equal an address outside the candidate set"
    );
}

/// The `assert_address_disjunction` helper must early-return when given
/// fewer than two candidates (the disjunction would be empty or trivial).
/// Test by calling the helper directly with a fresh symbolic addr — if the
/// hoist over-fires we observe the addr pinned to the single candidate.
#[test]
fn test_assert_address_disjunction_empty_and_single_addr_lists_are_noops() {
    let ctx = SymContext::new_mock();
    let addr = RustBV::symbolic(&ctx, "dj_helper_addr", 64);

    // Empty addrs: early return, no constraint added.
    SymbolicMemory::assert_address_disjunction(&addr, &[], &ctx);

    // Single addr: early return, no constraint added.
    SymbolicMemory::assert_address_disjunction(&addr, &[0x4000], &ctx);

    // addr is still completely unconstrained — both 0x1000 and 0xDEAD are
    // satisfiable values.
    let probe_a = addr.eq(&RustBV::concrete(0x1000, 64), &ctx);
    let probe_b = addr.eq(&RustBV::concrete(0xDEAD, 64), &ctx);
    assert!(
        ctx.can_be_true(&probe_a),
        "addr must still be free to equal 0x1000 after a [single]-addr helper call"
    );
    assert!(
        ctx.can_be_true(&probe_b),
        "addr must still be free to equal 0xDEAD after a [single]-addr helper call"
    );
}

/// A `Multiple` *load* must hoist the disjunction too — covers
/// `load_symbolic_unified` Multiple arm.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_assert_address_disjunction_multiple_load_hoists() {
    use crate::concretize::AddressConcretizer;
    use crate::symbolic::get_solver_stats;

    let pre = get_solver_stats();
    let pre_count = pre
        .get("concretize_disjunction_count")
        .copied()
        .unwrap_or(0);

    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);
    // Pre-populate both candidates so they are not unmapped.
    mem.store_concrete(0x1000, RustBV::concrete(0x11_22_33_44, 32))
        .unwrap();
    mem.store_concrete(0x1080, RustBV::concrete(0xAA_BB_CC_DD, 32))
        .unwrap();
    mem.store_concrete(0x1300, RustBV::concrete(0xDE_AD_BE_EF, 32))
        .unwrap();

    let addr = RustBV::symbolic(&ctx, "dj_load", 64);
    let a0 = addr.eq(&RustBV::concrete(0x1000, 64), &ctx);
    let a1 = addr.eq(&RustBV::concrete(0x1080, 64), &ctx);
    let a2 = addr.eq(&RustBV::concrete(0x1300, 64), &ctx);
    ctx.assume_true(&a0.or(&a1, &ctx).or(&a2, &ctx));

    let _value = mem
        .load_symbolic_unified(addr, 4, &ctx, &concretizer)
        .expect("multi load must succeed");

    let post = get_solver_stats();
    let post_count = post.get("concretize_disjunction_count").copied().unwrap();
    assert!(
        post_count > pre_count,
        "expected concretize_disjunction_count to increment (pre={pre_count}, post={post_count})"
    );
}
