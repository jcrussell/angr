//! Counters and metrics fired from the memory subsystem: memory ITE depth,
//! load/store volume, concretization counts, lazy page faults and the
//! symbolic-address counters, each asserted to tick from the production
//! entry points rather than from a test-only shim.
//!
//! Carved out of the former monolithic `symbolic.rs` (angr-c7xno.56).

use super::super::*;

/// angr-0nme Phase 0: an eager symbolic store with N candidate addresses
/// produces an ITE chain of depth N. `record_mem_ite_depth` must bump the
/// `mem_ite_depth_max` watermark and add to the cumulative total. This is
/// the baseline metric that Phase 1 (Multi cells, angr-czph) will be
/// compared against.
///
/// The two `mem_ite_depth_*` counters are process-global atomics shared
/// with `cargo test` parallel runners, so this asserts on **deltas**
/// from a captured baseline rather than absolute values. The pre/post
/// difference for `mem_ite_depth_total` must be at least 3 (one
/// 3-candidate eager store from this test); for `mem_ite_depth_max` the
/// post-store watermark must be at least 3 (it can only climb).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_mem_ite_depth_counter_records_eager_multi_store() {
    use crate::concretize::AddressConcretizer;
    use crate::symbolic::get_solver_stats;

    let pre = get_solver_stats();
    let pre_total = pre.get("mem_ite_depth_total").copied().unwrap_or(0);
    assert!(
        pre.contains_key("mem_ite_depth_max"),
        "mem_ite_depth_max key must be reported by get_solver_stats"
    );
    assert!(
        pre.contains_key("mem_ite_depth_total"),
        "mem_ite_depth_total key must be reported by get_solver_stats"
    );

    let ctx = SymContext::new_mock();
    // SYMBOLIC_WRITE_ADDRESSES on so the Range strategy applies and the store
    // fans out to 3 eager ITE candidates; the default Max-only chain would
    // resolve to one address and record no ITE depth (angr-9ke6b.194).
    let concretizer = AddressConcretizer {
        symbolic_write_addresses: true,
        ..AddressConcretizer::new()
    };
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // Build a symbolic address constrained to {0x1000, 0x1004, 0x1008}.
    let addr = RustBV::symbolic(&ctx, "addr", 64);
    let a0 = addr.eq(&RustBV::concrete(0x1000, 64), &ctx);
    let a1 = addr.eq(&RustBV::concrete(0x1004, 64), &ctx);
    let a2 = addr.eq(&RustBV::concrete(0x1008, 64), &ctx);
    let or_01 = a0.or(&a1, &ctx);
    let or_all = or_01.or(&a2, &ctx);
    ctx.assume_true(&or_all);
    assert!(ctx.is_sat(), "candidate-address context must be SAT");

    let value = RustBV::concrete(0xDEAD, 32);
    mem.store_symbolic(addr, value, &ctx, &concretizer)
        .expect("symbolic store with multi solutions must succeed");

    let post = get_solver_stats();
    let post_max = post.get("mem_ite_depth_max").copied().unwrap();
    let post_total = post.get("mem_ite_depth_total").copied().unwrap();
    assert!(
        post_max >= 3,
        "expected mem_ite_depth_max >= 3 (got {post_max}) after a 3-candidate eager store"
    );
    assert!(
        post_total >= pre_total + 3,
        "expected mem_ite_depth_total delta >= 3 (pre={pre_total}, post={post_total})"
    );
}

/// angr-2j5v end-to-end: a real `SymbolicMemory::{store,load}` round-trip
/// must increment `mem_load_count` / `mem_store_count` / `mem_load_bytes` /
/// `mem_store_bytes`. Delta-based assertions (counters are process-global).
#[test]
fn test_memory_volume_counters_fire_on_load_store() {
    use crate::symbolic::get_solver_stats;

    let pre = get_solver_stats();
    let pre_load = pre.get("mem_load_count").copied().unwrap_or(0);
    let pre_store = pre.get("mem_store_count").copied().unwrap_or(0);
    let pre_load_bytes = pre.get("mem_load_bytes").copied().unwrap_or(0);
    let pre_store_bytes = pre.get("mem_store_bytes").copied().unwrap_or(0);

    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x2000, 0x1000, Permission::RWX);

    // Concrete address, 4 bytes stored, 4 bytes loaded.
    let addr = RustBV::concrete(0x2000, 64);
    let value = RustBV::concrete(0xCAFEBABE, 32);
    mem.store(&addr, value, &ctx).expect("store ok");
    let _ = mem.load(addr, 4, &ctx).expect("load ok");

    let post = get_solver_stats();
    assert!(
        post.get("mem_load_count").copied().unwrap() > pre_load,
        "mem_load_count must climb"
    );
    assert!(
        post.get("mem_store_count").copied().unwrap() > pre_store,
        "mem_store_count must climb"
    );
    assert!(
        post.get("mem_load_bytes").copied().unwrap() >= pre_load_bytes + 4,
        "mem_load_bytes must climb by load size"
    );
    assert!(
        post.get("mem_store_bytes").copied().unwrap() >= pre_store_bytes + 4,
        "mem_store_bytes must climb by store size"
    );
}

/// angr-2j5v: a symbolic-address store routed through `concretize_write`
/// must bump `concretize_write_count` and `concretize_total_candidates`.
/// Uses the same setup as the ITE-depth test (3 candidate addresses).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_concretize_counters_fire_on_symbolic_store() {
    use crate::concretize::AddressConcretizer;
    use crate::symbolic::get_solver_stats;

    let pre = get_solver_stats();
    let pre_write = pre.get("concretize_write_count").copied().unwrap_or(0);
    let pre_total = pre.get("concretize_total_candidates").copied().unwrap_or(0);
    let pre_max = pre.get("concretize_max_candidates").copied().unwrap_or(0);

    let ctx = SymContext::new_mock();
    // SYMBOLIC_WRITE_ADDRESSES on so the Range strategy applies and the store
    // sees K=3 candidates; the default Max-only chain would report K=1
    // (angr-9ke6b.194).
    let concretizer = AddressConcretizer {
        symbolic_write_addresses: true,
        ..AddressConcretizer::new()
    };
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x3000, 0x1000, Permission::RWX);

    let addr = RustBV::symbolic(&ctx, "addr_2j5v", 64);
    let a0 = addr.eq(&RustBV::concrete(0x3000, 64), &ctx);
    let a1 = addr.eq(&RustBV::concrete(0x3004, 64), &ctx);
    let a2 = addr.eq(&RustBV::concrete(0x3008, 64), &ctx);
    let or_all = a0.or(&a1, &ctx).or(&a2, &ctx);
    ctx.assume_true(&or_all);
    assert!(ctx.is_sat());

    let value = RustBV::concrete(0xC0DE, 32);
    mem.store_symbolic(addr, value, &ctx, &concretizer)
        .expect("symbolic store must succeed");

    let post = get_solver_stats();
    assert!(
        post.get("concretize_write_count").copied().unwrap() > pre_write,
        "concretize_write_count must climb"
    );
    assert!(
        post.get("concretize_total_candidates").copied().unwrap() >= pre_total + 3,
        "concretize_total_candidates must climb by K=3"
    );
    assert!(
        post.get("concretize_max_candidates").copied().unwrap() >= pre_max.max(3),
        "concretize_max_candidates must reach >= 3"
    );
}

/// angr-0nme Phase 0: direct calls to `record_mem_ite_depth` increment the
/// cumulative total and lift the max watermark monotonically. A 0-depth
/// call is a no-op. Verified via deltas because the underlying atomics
/// are process-global and may be touched by other parallel tests.
#[test]
fn test_record_mem_ite_depth_helper() {
    use crate::symbolic::{get_solver_stats, record_mem_ite_depth};

    let baseline = get_solver_stats();
    let base_total = baseline.get("mem_ite_depth_total").copied().unwrap_or(0);
    let base_max = baseline.get("mem_ite_depth_max").copied().unwrap_or(0);

    record_mem_ite_depth(0); // no-op
    let after_zero = get_solver_stats();
    assert_eq!(
        after_zero.get("mem_ite_depth_total").copied().unwrap_or(0),
        base_total,
        "record_mem_ite_depth(0) must not change the total"
    );

    record_mem_ite_depth(5);
    record_mem_ite_depth(8);
    record_mem_ite_depth(3);
    let after = get_solver_stats();
    let after_total = after.get("mem_ite_depth_total").copied().unwrap();
    let after_max = after.get("mem_ite_depth_max").copied().unwrap();
    assert!(
        after_total >= base_total + 16,
        "expected total delta of 16 (5+8+3); base={base_total} after={after_total}"
    );
    assert!(
        after_max >= base_max.max(8),
        "expected max to reach at least 8; base={base_max} after={after_max}"
    );
}

/// angr-9ke6b.228: `mem_lazy_page_fault_count` must tick on **both** the load
/// and the store side. It used to be bumped only in the public
/// `SymbolicMemory::{load,store}` wrappers — which no production path calls —
/// and the store-side branch there was outright dead, because `store_concrete`
/// has no lazy classification and never yields `UnmappedPageInRegion`. The
/// bumps now live on the producers, so the lazy entry points the interpreter
/// actually uses are covered.
///
/// Counters are process-global and tests run in parallel, so this asserts
/// strict growth against a baseline read just before the faulting op rather
/// than an exact delta.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_lazy_page_fault_counter_ticks_on_both_sides() {
    use crate::symbolic::get_solver_stats;

    let fault_count = || {
        get_solver_stats()
            .get("mem_lazy_page_fault_count")
            .copied()
            .unwrap_or(0)
    };

    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    // Page 0x2000 is unmapped but declared lazy: Python still holds its backer.
    mem.add_lazy_region(0x2000, 0x1000);

    let base_store = fault_count();
    let err = mem
        .store_concrete_lazy(0x2000, RustBV::concrete(0xAA, 8))
        .expect_err("lazy-region store must report the fetch-me signal");
    assert!(
        matches!(err, MemoryError::UnmappedPageInRegion { page_addr } if page_addr == 0x2000),
        "expected UnmappedPageInRegion at 0x2000, got {err:?}"
    );
    assert!(
        fault_count() > base_store,
        "store-side lazy fault must bump mem_lazy_page_fault_count"
    );

    let base_load = fault_count();
    let err = mem
        .load_concrete_lazy(0x2000, 1, &ctx)
        .expect_err("lazy-region load must report the fetch-me signal");
    assert!(
        matches!(err, MemoryError::UnmappedPageInRegion { page_addr } if page_addr == 0x2000),
        "expected UnmappedPageInRegion at 0x2000, got {err:?}"
    );
    assert!(
        fault_count() > base_load,
        "load-side lazy fault must bump mem_lazy_page_fault_count"
    );

    // A hard (non-lazy) miss must classify as plain `Unmapped`, which is the
    // branch that leaves the counter alone. (Asserted on the error shape, not
    // on the counter staying equal: it is process-global and a concurrently
    // running test may bump it.)
    assert!(matches!(
        mem.load_concrete_lazy(0x9000, 1, &ctx),
        Err(MemoryError::Unmapped { .. })
    ));
    assert!(matches!(
        mem.store_concrete_lazy(0x9000, RustBV::concrete(1, 8)),
        Err(MemoryError::Unmapped { .. })
    ));
}

/// angr-9ke6b.229: the `mem_{load,store}_symbolic_addr` counters must tick from
/// the *production* symbolic-address entry points, not from the test-only
/// `SymbolicMemory::{load,store}` wrappers (which `.228` showed nothing calls).
/// Lower-bound assertions only — the counters are process-global and other
/// tests run concurrently in the same process.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_symbolic_addr_counters_tick_from_production_entry_points() {
    let load_sym = || {
        crate::symbolic::get_solver_stats()
            .get("mem_load_symbolic_addr")
            .copied()
            .unwrap_or(0)
    };
    let store_sym = || {
        crate::symbolic::get_solver_stats()
            .get("mem_store_symbolic_addr")
            .copied()
            .unwrap_or(0)
    };

    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // A symbolic address pinned to a single concrete solution: still symbolic
    // to `as_u64()`, so it goes past every fast path into the concretizer.
    let addr = RustBV::symbolic(&ctx, "counter_addr", 64);
    ctx.assume_true(&addr.eq(&RustBV::concrete(0x1000, 64), &ctx));
    let value = RustBV::concrete(0x41, 8);

    let base_store = store_sym();
    mem.store_symbolic(addr.clone(), value.clone(), &ctx, &concretizer)
        .expect("store_symbolic must succeed");
    mem.store_symbolic_unified(addr.clone(), value.clone(), &ctx, &concretizer)
        .expect("store_symbolic_unified must succeed");
    mem.store_symbolic_unified_multi(addr.clone(), value.clone(), &ctx, &concretizer)
        .expect("store_symbolic_unified_multi must succeed");
    mem.store_with_concretization(&addr, value, &ConcretizationResult::Single(0x1000), &ctx)
        .expect("store_with_concretization must succeed");
    assert!(
        store_sym() >= base_store + 4,
        "each of the four symbolic store entry points must bump \
         mem_store_symbolic_addr (base {base_store}, now {})",
        store_sym()
    );

    let base_load = load_sym();
    mem.load_symbolic(addr.clone(), 1, &ctx, &concretizer)
        .expect("load_symbolic must succeed");
    mem.load_symbolic_unified(addr, 1, &ctx, &concretizer)
        .expect("load_symbolic_unified must succeed");
    assert!(
        load_sym() >= base_load + 2,
        "both symbolic load entry points must bump mem_load_symbolic_addr \
         (base {base_load}, now {})",
        load_sym()
    );
}
