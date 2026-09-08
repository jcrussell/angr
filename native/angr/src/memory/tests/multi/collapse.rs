//! Phase 1.2 (angr-n082): Multi-cell collapse on the load path, plus the
//! angr-9ke6b.96 follow-up that made the non-lazy `load_concrete` path
//! dispatch to the same assembler.

use super::*;

// Phase 1.2 (angr-n082): Multi-cell collapse in load_concrete_lazy_inner.
// ============================================================================

/// Single-byte Multi load: install two alternatives at one byte, constrain
/// the address variable to either candidate, then probe-fork to pin the
/// address and verify the load eval'd to the matching alternative's value.
/// Also asserts that `mem_ite_depth_max` reflects the 2-alt collapse.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_multi_cell_load_single_byte() {
    use crate::symbolic::get_solver_stats;

    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "load1_addr", 64);
    let payload = MultiPayload::from_alternatives(vec![
        make_alt(&ctx, &addr_var, 0x1000, 0xAA),
        make_alt(&ctx, &addr_var, 0x2000, 0xBB),
    ]);
    mem.set_multi_alternatives(0x1000, payload);

    // Constrain addr to {0x1000, 0x2000} so both alternatives are reachable.
    let eq_a = addr_var.eq(&RustBV::concrete(0x1000, 64), &ctx);
    let eq_b = addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx);
    ctx.assume_true(&eq_a.or(&eq_b, &ctx));
    assert!(ctx.is_sat());

    let pre_max = get_solver_stats()
        .get("mem_ite_depth_max")
        .copied()
        .unwrap_or(0);

    let loaded = mem
        .load_concrete_lazy(0x1000, 1, &ctx)
        .expect("Multi-cell load must succeed");

    // The collapse must record the alternative count.
    let post_max = get_solver_stats()
        .get("mem_ite_depth_max")
        .copied()
        .unwrap_or(0);
    assert!(
        post_max >= pre_max.max(2),
        "expected mem_ite_depth_max >= 2 after a 2-alt collapse \
         (pre={pre_max}, post={post_max})"
    );

    // Probe-fork: under addr == 0x1000, loaded == 0xAA.
    let probe_a = ctx.fork();
    probe_a.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe_a));
    assert!(probe_a.is_sat());
    assert_eq!(probe_a.eval(&loaded), Some(0xAA));

    // Probe-fork: under addr == 0x2000, loaded == 0xBB.
    let probe_b = ctx.fork();
    probe_b.assume_true(&addr_var.eq(&RustBV::concrete(0x2000, 64), &probe_b));
    assert!(probe_b.is_sat());
    assert_eq!(probe_b.eval(&loaded), Some(0xBB));
}

/// Multi-byte load that mixes a Multi cell with surrounding concrete bytes.
/// 4-byte little-endian load at 0x1000 where:
///   byte 0 (0x1000): Multi {addr==A -> 0xAA, addr==B -> 0xBB}
///   byte 1 (0x1001): concrete 0x11
///   byte 2 (0x1002): concrete 0x22
///   byte 3 (0x1003): concrete 0x33
/// Under addr==A the load must read 0x332211AA, under addr==B 0x332211BB.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_multi_cell_load_mixed_concrete() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // Surrounding concrete bytes.
    mem.store_concrete(0x1001, RustBV::concrete(0x11, 8))
        .unwrap();
    mem.store_concrete(0x1002, RustBV::concrete(0x22, 8))
        .unwrap();
    mem.store_concrete(0x1003, RustBV::concrete(0x33, 8))
        .unwrap();

    let addr_var = RustBV::symbolic(&ctx, "load_mix_addr", 64);
    let payload = MultiPayload::from_alternatives(vec![
        make_alt(&ctx, &addr_var, 0x4000, 0xAA),
        make_alt(&ctx, &addr_var, 0x5000, 0xBB),
    ]);
    mem.set_multi_alternatives(0x1000, payload);

    let eq_a = addr_var.eq(&RustBV::concrete(0x4000, 64), &ctx);
    let eq_b = addr_var.eq(&RustBV::concrete(0x5000, 64), &ctx);
    ctx.assume_true(&eq_a.or(&eq_b, &ctx));

    let loaded = mem
        .load_concrete_lazy(0x1000, 4, &ctx)
        .expect("4-byte Multi+concrete load must succeed");
    assert_eq!(loaded.width(), 32);

    let probe_a = ctx.fork();
    probe_a.assume_true(&addr_var.eq(&RustBV::concrete(0x4000, 64), &probe_a));
    assert_eq!(probe_a.eval(&loaded), Some(0x33_22_11_AA));

    let probe_b = ctx.fork();
    probe_b.assume_true(&addr_var.eq(&RustBV::concrete(0x5000, 64), &probe_b));
    assert_eq!(probe_b.eval(&loaded), Some(0x33_22_11_BB));
}

/// Big-endian variant of the mixed concrete + Multi load. byte 0 is the
/// MSB so addr==A should yield 0xAA_11_22_33 and addr==B 0xBB_11_22_33.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_multi_cell_load_big_endian() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Big);
    mem.map(0x1000, 0x1000, Permission::RWX);

    mem.store_concrete(0x1001, RustBV::concrete(0x11, 8))
        .unwrap();
    mem.store_concrete(0x1002, RustBV::concrete(0x22, 8))
        .unwrap();
    mem.store_concrete(0x1003, RustBV::concrete(0x33, 8))
        .unwrap();

    let addr_var = RustBV::symbolic(&ctx, "load_be_addr", 64);
    let payload = MultiPayload::from_alternatives(vec![
        make_alt(&ctx, &addr_var, 0x4000, 0xAA),
        make_alt(&ctx, &addr_var, 0x5000, 0xBB),
    ]);
    mem.set_multi_alternatives(0x1000, payload);

    let eq_a = addr_var.eq(&RustBV::concrete(0x4000, 64), &ctx);
    let eq_b = addr_var.eq(&RustBV::concrete(0x5000, 64), &ctx);
    ctx.assume_true(&eq_a.or(&eq_b, &ctx));

    let loaded = mem
        .load_concrete_lazy(0x1000, 4, &ctx)
        .expect("BE Multi+concrete load must succeed");

    let probe_a = ctx.fork();
    probe_a.assume_true(&addr_var.eq(&RustBV::concrete(0x4000, 64), &probe_a));
    assert_eq!(probe_a.eval(&loaded), Some(0xAA_11_22_33));

    let probe_b = ctx.fork();
    probe_b.assume_true(&addr_var.eq(&RustBV::concrete(0x5000, 64), &probe_b));
    assert_eq!(probe_b.eval(&loaded), Some(0xBB_11_22_33));
}

/// Multi cells at multiple bytes within the load range, plus a concrete
/// byte in between, exercises the per-byte loop's ability to handle
/// several independent ITE chains in one load.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_multi_cell_load_multiple_multi_bytes() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    mem.store_concrete(0x1001, RustBV::concrete(0x11, 8))
        .unwrap();
    // Last byte (offset 3) is also concrete via no-op (default 0).

    let addr_var = RustBV::symbolic(&ctx, "load_multi_addr", 64);
    mem.set_multi_alternatives(
        0x1000,
        MultiPayload::from_alternatives(vec![
            make_alt(&ctx, &addr_var, 0x4000, 0xAA),
            make_alt(&ctx, &addr_var, 0x5000, 0xBB),
        ]),
    );
    mem.set_multi_alternatives(
        0x1002,
        MultiPayload::from_alternatives(vec![
            make_alt(&ctx, &addr_var, 0x4000, 0xCC),
            make_alt(&ctx, &addr_var, 0x5000, 0xDD),
        ]),
    );

    let eq_a = addr_var.eq(&RustBV::concrete(0x4000, 64), &ctx);
    let eq_b = addr_var.eq(&RustBV::concrete(0x5000, 64), &ctx);
    ctx.assume_true(&eq_a.or(&eq_b, &ctx));

    let loaded = mem
        .load_concrete_lazy(0x1000, 4, &ctx)
        .expect("two-multi-byte load must succeed");

    let probe_a = ctx.fork();
    probe_a.assume_true(&addr_var.eq(&RustBV::concrete(0x4000, 64), &probe_a));
    assert_eq!(probe_a.eval(&loaded), Some(0x00_CC_11_AA));

    let probe_b = ctx.fork();
    probe_b.assume_true(&addr_var.eq(&RustBV::concrete(0x5000, 64), &probe_b));
    assert_eq!(probe_b.eval(&loaded), Some(0x00_DD_11_BB));
}

/// Concrete overwrite of a Multi byte must clear the multi_bitmap bit on
/// the page — the half of the cleanup `page.store_concrete` performs, so
/// later loads do not see a stale Multi marker.
///
/// This pins only that page-level half. The sidecar half — dropping the
/// owning `multi_objects` entry and bumping its `multi_versions` counter —
/// is `SymbolicMemory::store_concrete`'s own responsibility since angr-1tes,
/// and is pinned by the sibling
/// `multi/cache.rs::test_concrete_overwrite_clears_multi_cell`, whose doc
/// describes the pre-angr-1tes bug where only this page bit was cleared.
#[test]
fn test_concrete_overwrite_clears_multi_bit() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x3000, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "addr", 64);
    mem.set_multi_alternatives(
        0x3000,
        MultiPayload::from_alternatives(vec![make_alt(&ctx, &addr_var, 0x3000, 0x42)]),
    );
    {
        let page = mem.pages.get(&(0x3000 >> 12)).expect("page must exist");
        assert!(page.is_multi(0));
    }

    // Concrete overwrite at the same byte.
    mem.store_concrete(0x3000, RustBV::concrete(0xFE, 8))
        .unwrap();

    let page = mem.pages.get(&(0x3000 >> 12)).expect("page must exist");
    assert!(
        !page.is_multi(0),
        "concrete overwrite must clear the multi_bitmap bit"
    );
}

/// angr-9ke6b.95: the mirror direction of the test above — an ordinary
/// `store_concrete` of a *symbolic* value over a pre-existing Multi cell must
/// also drop the cell. `load_concrete_lazy_inner` dispatches to Multi before
/// consulting `symbolic_objects`, so a surviving cell would shadow the value
/// just stored.
#[test]
fn test_symbolic_overwrite_clears_multi_cell() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x3000, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "sym_over_multi_addr", 64);
    mem.set_multi_alternatives(
        0x3000,
        MultiPayload::from_alternatives(vec![make_alt(&ctx, &addr_var, 0x3000, 0x42)]),
    );
    assert_eq!(mem.multi_cell_count(), 1);

    // Ordinary store of a symbolic byte at the same address.
    let fresh = RustBV::symbolic(&ctx, "sym_over_multi_val", 8);
    mem.store_concrete(0x3000, fresh).unwrap();

    assert!(
        mem.get_multi_alternatives(0x3000).is_none(),
        "symbolic overwrite must drop the Multi sidecar entry"
    );
    let page = mem.pages.get(&(0x3000 >> 12)).expect("page must exist");
    assert!(
        !page.is_multi(0),
        "symbolic overwrite must clear the multi_bitmap bit"
    );
    assert!(
        page.is_symbolic(0),
        "the freshly stored symbolic byte must be marked Symbolic"
    );
}

/// Value-level companion to the test above: after the symbolic overwrite the
/// lazy load must resolve to the newly stored symbol, not the stale Multi
/// alternative (0x42).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_symbolic_overwrite_of_multi_loads_new_value() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x3000, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "sovm_addr", 64);
    mem.set_multi_alternatives(
        0x3000,
        MultiPayload::from_alternatives(vec![make_alt(&ctx, &addr_var, 0x3000, 0x42)]),
    );

    let fresh = RustBV::symbolic(&ctx, "sovm_val", 8);
    mem.store_concrete(0x3000, fresh.clone()).unwrap();

    let loaded = mem
        .load_concrete_lazy(0x3000, 1, &ctx)
        .expect("lazy load must succeed");

    // Pin the address to the Multi candidate so a surviving cell would fold
    // to 0x42, then pin the fresh symbol to a distinguishable value.
    let probe = ctx.fork();
    probe.assume_true(&addr_var.eq(&RustBV::concrete(0x3000, 64), &probe));
    probe.assume_true(&fresh.eq(&RustBV::concrete(0x77, 8), &probe));
    assert_eq!(
        probe.eval(&loaded),
        Some(0x77),
        "load must return the freshly stored symbol, not the stale Multi alternative"
    );
}

/// angr-6cp06.62: `import_symbolic_value` is the third Multi->Symbolic
/// transition site (after `store_concrete`'s symbolic branch and `merge`'s
/// per-byte collapse) and owes the same cleanup. It is on the live
/// `flush_stores` / `flush_stores_to_rust_memory` path, so a surviving Multi
/// cell would shadow every flushed symbolic store at a previously-Multi byte.
#[test]
fn test_import_symbolic_value_clears_multi_cell() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x3000, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "imp_over_multi_addr", 64);
    mem.set_multi_alternatives(
        0x3000,
        MultiPayload::from_alternatives(vec![make_alt(&ctx, &addr_var, 0x3000, 0x42)]),
    );
    mem.set_multi_alternatives(
        0x3001,
        MultiPayload::from_alternatives(vec![make_alt(&ctx, &addr_var, 0x3001, 0x43)]),
    );
    assert_eq!(mem.multi_cell_count(), 2);

    // A 2-byte symbolic import covering both Multi bytes.
    let fresh = RustBV::symbolic(&ctx, "imp_over_multi_val", 16);
    mem.import_symbolic_value(0x3000, fresh, None).unwrap();

    let page = mem.pages.get(&(0x3000 >> 12)).expect("page must exist");
    for offset in 0..2u16 {
        assert!(
            mem.get_multi_alternatives(0x3000 + u64::from(offset)).is_none(),
            "import must drop the Multi sidecar entry at +{offset}"
        );
        assert!(
            !page.is_multi(offset),
            "import must clear the multi_bitmap bit at +{offset}"
        );
        assert!(
            page.is_symbolic(offset),
            "the imported byte at +{offset} must be marked Symbolic"
        );
    }
    assert_eq!(mem.multi_cell_count(), 0);
}

/// Value-level companion to the test above: after the import the load must
/// resolve to the imported symbol, not the stale Multi alternative (0x42).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_import_symbolic_value_over_multi_loads_new_value() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x3000, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "iovm_addr", 64);
    mem.set_multi_alternatives(
        0x3000,
        MultiPayload::from_alternatives(vec![make_alt(&ctx, &addr_var, 0x3000, 0x42)]),
    );

    let fresh = RustBV::symbolic(&ctx, "iovm_val", 8);
    mem.import_symbolic_value(0x3000, fresh.clone(), None)
        .unwrap();

    let loaded = mem
        .load_concrete_lazy(0x3000, 1, &ctx)
        .expect("lazy load must succeed");

    let probe = ctx.fork();
    probe.assume_true(&addr_var.eq(&RustBV::concrete(0x3000, 64), &probe));
    probe.assume_true(&fresh.eq(&RustBV::concrete(0x77, 8), &probe));
    assert_eq!(
        probe.eval(&loaded),
        Some(0x77),
        "load must return the imported symbol, not the stale Multi alternative"
    );
}

// angr-9ke6b.96: `load_concrete` (the non-lazy load path behind
// `RustSimState::memory_load` / `_pending_memory_load`) must dispatch to
// `assemble_load_with_multi` too. `install_multi_for_candidates` never sets
// the page's symbolic bit, so before the fix every symbolic fast path in
// `load_concrete` missed and the load silently returned the stale concrete
// placeholder byte.
// ============================================================================

/// `load_concrete` on an un-flushed Multi cell must reconstruct the
/// alternative, not return the placeholder byte underneath it.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_load_concrete_sees_unflushed_multi_cell() {
    let ctx = SymContext::new_mock();
    let concretizer = multi_write_concretizer();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "lc_multi_addr", 64);
    ctx.assume_true(
        &addr_var
            .eq(&RustBV::concrete(0x1000, 64), &ctx)
            .or(&addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
    );

    mem.store_symbolic_unified(
        addr_var.clone(),
        RustBV::concrete(0xDEADBEEF, 32),
        &ctx,
        &concretizer,
    )
    .expect("symbolic-address store must succeed");
    assert_eq!(
        mem.multi_cell_count(),
        8,
        "both candidates get 4 Multi bytes"
    );

    // No flush: the value lives only in the Multi sidecar.
    let loaded = mem
        .load_concrete(0x1000, 4, &ctx)
        .expect("load_concrete over a Multi range must succeed");
    assert!(
        loaded.as_u64().is_none(),
        "a Multi-covered load must be symbolic, not the placeholder concrete byte"
    );
    let probe = ctx.fork();
    probe.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe));
    assert_eq!(
        probe.eval(&loaded),
        Some(0xDEADBEEF),
        "load_concrete must reconstruct the Multi alternative"
    );

    // The lazy path must agree — the two must not diverge.
    let lazy = mem.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    assert_eq!(probe.eval(&lazy), probe.eval(&loaded));
}

/// A Multi cell that only partially covers the loaded range must still
/// route through the per-byte assembler; the untouched bytes keep their
/// concrete page values.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_load_concrete_partial_multi_overlap() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);
    mem.store_concrete(0x1000, RustBV::concrete(0x1122_3344, 32))
        .expect("concrete seed store");

    // Install a single Multi byte over the lowest byte only (LE byte 0).
    let addr_var = RustBV::symbolic(&ctx, "lc_partial_addr", 64);
    let payload = MultiPayload::from_alternatives(vec![make_alt(&ctx, &addr_var, 0x1000, 0xEE)]);
    mem.set_multi_alternatives(0x1000, payload);

    let loaded = mem.load_concrete(0x1000, 4, &ctx).unwrap();
    let probe = ctx.fork();
    probe.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe));
    assert_eq!(
        probe.eval(&loaded),
        Some(0x1122_33EE),
        "Multi byte 0 must override; bytes 1..3 keep their concrete values"
    );
}

/// angr-4r8mh: `read_concrete_bytes_for_lift` must stop its concrete run at a
/// Multi byte, not just a symbolic one. A cell installed on a never-symbolic
/// byte carries no symbolic bit (see `test_multi_payload_round_trip`), and its
/// page byte is the don't-care default `MultiPayload::collapse` was handed —
/// handing it to the native lifter would decode a placeholder as an
/// instruction byte.
#[test]
fn test_lift_read_stops_at_multi_byte() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);
    mem.store_concrete(0x1000, RustBV::concrete(0x1122_3344, 32))
        .expect("concrete seed store");

    // Sanity: the whole range lifts before any Multi cell exists.
    assert_eq!(
        mem.read_concrete_bytes_for_lift(0x1000, 4),
        Some(vec![0x44, 0x33, 0x22, 0x11]),
        "all-concrete range must lift in full"
    );

    // Install a Multi cell on byte 2 — never symbolic, so only the multi
    // bitmap bit is set.
    let addr_var = RustBV::symbolic(&ctx, "lift_multi_addr", 64);
    let payload = MultiPayload::from_alternatives(vec![
        make_alt(&ctx, &addr_var, 0x1000, 0xEE),
        make_alt(&ctx, &addr_var, 0x2000, 0xFF),
    ]);
    mem.set_multi_alternatives(0x1002, payload);
    let page = mem.pages.get(&(0x1000 >> 12)).expect("page must exist");
    assert!(page.is_multi(2), "byte 2 must be marked Multi");
    assert!(
        !page.is_symbolic(2),
        "never-symbolic byte must not gain a symbolic bit — this is what \
         hid the bug: the is_symbolic check alone lets it through"
    );

    assert_eq!(
        mem.read_concrete_bytes_for_lift(0x1000, 4),
        Some(vec![0x44, 0x33]),
        "the run must stop at the Multi byte, keeping only the concrete prefix"
    );

    // A lift starting *on* the Multi byte has no concrete prefix at all.
    assert_eq!(
        mem.read_concrete_bytes_for_lift(0x1002, 2),
        None,
        "a leading Multi byte must be reported as no readable code"
    );
}
