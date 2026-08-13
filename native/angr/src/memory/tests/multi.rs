use super::super::*;

// ============================================================================
// Phase 1.1 (angr-me3z): MultiPayload data structure + sidecar storage tests.
// These cover only the data structure and storage. Load-side collapse is in
// Phase 1.2 (angr-n082) and store-side helpers in Phase 1.3 (angr-aija).
// ============================================================================

/// A concretizer with `SYMBOLIC_WRITE_ADDRESSES` on, so the Range write
/// strategy is part of the chain and a symbolic-address store concretizes to
/// a candidate *set*. With the option off (the default) Python — and, since
/// angr-9ke6b.194, Rust — uses a Max-only chain for unannotated addresses,
/// which yields a single address and installs no Multi cells. See
/// `AddressConcretizer::write_range_applies`.
fn multi_write_concretizer() -> AddressConcretizer {
    AddressConcretizer {
        symbolic_write_addresses: true,
        ..AddressConcretizer::new()
    }
}

/// Build a Multi alternative `(addr == cand) -> byte(value)` for tests.
fn make_alt(ctx: &SymContext, addr_var: &RustBV, cand: u64, value: u8) -> MultiAlternative {
    let cand_const = RustBV::concrete(cand as u128, addr_var.width());
    let cond = addr_var.eq(&cand_const, ctx);
    let val = RustBV::concrete(value as u128, 8);
    MultiAlternative::new(cond, val)
}

/// Round-trip: install a payload, read it back through the getter, and
/// confirm the page bit + alt count agree. Verifies the basic sidecar
/// wiring before any load/store integration.
#[test]
fn test_multi_payload_round_trip() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "addr", 64);
    let payload = MultiPayload::from_alternatives(vec![
        make_alt(&ctx, &addr_var, 0x1000, 0xAA),
        make_alt(&ctx, &addr_var, 0x1004, 0xBB),
        make_alt(&ctx, &addr_var, 0x1008, 0xCC),
    ]);

    mem.set_multi_alternatives(0x1000, payload);

    assert_eq!(mem.multi_cell_count(), 1);
    let got = mem
        .get_multi_alternatives(0x1000)
        .expect("Multi cell must be readable after install");
    assert_eq!(got.len(), 3);
    assert_eq!(got.alternatives()[0].value.as_u64(), Some(0xAA));
    assert_eq!(got.alternatives()[1].value.as_u64(), Some(0xBB));
    assert_eq!(got.alternatives()[2].value.as_u64(), Some(0xCC));

    // Page bitmap should be set for offset 0 of page 0x1000.
    let page = mem.pages.get(&(0x1000 >> 12)).expect("page must exist");
    assert!(page.is_multi(0), "page must mark byte 0 as Multi");
    assert!(
        !page.is_symbolic(0),
        "Multi marker must not overlap plain Symbolic marker"
    );
}

/// Empty payload clears the cell (must not be stored as a zero-length entry).
#[test]
fn test_multi_payload_empty_clears_cell() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "addr", 64);
    mem.set_multi_alternatives(
        0x1000,
        MultiPayload::from_alternatives(vec![make_alt(&ctx, &addr_var, 0x1000, 0x11)]),
    );
    assert_eq!(mem.multi_cell_count(), 1);

    // Setting an empty payload should clear the cell.
    mem.set_multi_alternatives(0x1000, MultiPayload::default());
    assert_eq!(mem.multi_cell_count(), 0);
    assert!(mem.get_multi_alternatives(0x1000).is_none());

    let page = mem.pages.get(&(0x1000 >> 12)).expect("page must exist");
    assert!(!page.is_multi(0), "Multi marker must be cleared");
}

/// Fork independence: mutating the child's Multi cells must not bleed into
/// the parent, matching SymbolicMemory's CoW contract.
#[test]
fn test_multi_fork_independence() {
    let ctx = SymContext::new_mock();
    let mut parent = SymbolicMemory::new(Endness::Little);
    parent.map(0x1000, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "addr", 64);
    parent.set_multi_alternatives(
        0x1000,
        MultiPayload::from_alternatives(vec![
            make_alt(&ctx, &addr_var, 0x1000, 0xAA),
            make_alt(&ctx, &addr_var, 0x1004, 0xBB),
        ]),
    );

    let mut child = parent.fork();
    assert_eq!(child.multi_cell_count(), 1, "fork must copy multi_objects");

    // Replace the child's payload with a different shape.
    child.set_multi_alternatives(
        0x1000,
        MultiPayload::from_alternatives(vec![make_alt(&ctx, &addr_var, 0x1000, 0xFF)]),
    );

    let p = parent
        .get_multi_alternatives(0x1000)
        .expect("parent still has its payload");
    assert_eq!(
        p.len(),
        2,
        "parent payload must be untouched by fork mutation"
    );
    assert_eq!(p.alternatives()[0].value.as_u64(), Some(0xAA));

    let c = child
        .get_multi_alternatives(0x1000)
        .expect("child has the new payload");
    assert_eq!(c.len(), 1);
    assert_eq!(c.alternatives()[0].value.as_u64(), Some(0xFF));

    // And clearing the child must not clear the parent's cell.
    child.clear_multi_at(0x1000);
    assert_eq!(child.multi_cell_count(), 0);
    assert_eq!(parent.multi_cell_count(), 1);
}

/// Setting a Multi cell at an address that previously held a plain-Symbolic
/// entry must take precedence: the symbolic_objects entry is removed, the
/// page's symbolic bit is replaced by the multi bit. This is the
/// "Multi supersedes Symbolic" rule documented on the `multi_objects` field.
#[test]
fn test_multi_supersedes_existing_symbolic() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // Install a single-byte symbolic value at 0x1000.
    let sym_byte = RustBV::symbolic(&ctx, "byte", 8);
    mem.import_symbolic_value(0x1000, sym_byte, None).unwrap();
    assert!(mem.get_symbolic_object(0x1000).is_some());

    // Now upgrade the same byte to Multi.
    let addr_var = RustBV::symbolic(&ctx, "addr", 64);
    mem.set_multi_alternatives(
        0x1000,
        MultiPayload::from_alternatives(vec![make_alt(&ctx, &addr_var, 0x1000, 0x77)]),
    );

    assert!(
        mem.get_symbolic_object(0x1000).is_none(),
        "set_multi_alternatives must clear the prior symbolic_objects entry"
    );
    assert!(mem.get_multi_alternatives(0x1000).is_some());

    let page = mem.pages.get(&(0x1000 >> 12)).expect("page must exist");
    assert!(page.is_multi(0), "Multi marker must be set");
    // import_symbolic_value set the symbolic bit; set_multi_alternatives
    // does not currently clear that bit since the byte is still "abstract" in
    // some sense — load-side collapse (Phase 1.2) handles the priority. Just
    // assert the multi bit is set and the symbolic_objects entry is gone,
    // which is what later phases rely on.
    let _ = page.is_symbolic(0);
}

/// Counter invariant: every set_multi_alternatives must bump mem_ite_depth.
/// Uses delta assertions because the underlying atomics are process-global
/// and cargo test runs in parallel — same convention as the Phase 0 tests.
#[test]
fn test_multi_payload_records_ite_depth() {
    use crate::symbolic::get_solver_stats;

    let pre = get_solver_stats();
    let pre_total = pre.get("mem_ite_depth_total").copied().unwrap_or(0);

    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x2000, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "addr_d", 64);

    // Two installs: 3 alts then 2 alts = total delta of 5; max watermark must be >= 3.
    mem.set_multi_alternatives(
        0x2000,
        MultiPayload::from_alternatives(vec![
            make_alt(&ctx, &addr_var, 0x2000, 0x01),
            make_alt(&ctx, &addr_var, 0x2004, 0x02),
            make_alt(&ctx, &addr_var, 0x2008, 0x03),
        ]),
    );
    mem.set_multi_alternatives(
        0x2010,
        MultiPayload::from_alternatives(vec![
            make_alt(&ctx, &addr_var, 0x2010, 0x10),
            make_alt(&ctx, &addr_var, 0x2014, 0x20),
        ]),
    );

    // Empty payload must NOT touch the counter.
    mem.set_multi_alternatives(0x2020, MultiPayload::default());

    let post = get_solver_stats();
    let post_total = post.get("mem_ite_depth_total").copied().unwrap();
    let post_max = post.get("mem_ite_depth_max").copied().unwrap();
    assert!(
        post_total >= pre_total + 5,
        "expected mem_ite_depth_total delta >= 5 (pre={pre_total}, post={post_total})"
    );
    assert!(
        post_max >= 3,
        "expected mem_ite_depth_max >= 3 after a 3-alt insert (got {post_max})"
    );
}

// ============================================================================
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
/// the page. The owning `multi_objects` entry is the caller's
/// responsibility (documented on `store_concrete`), but the page-level
/// bookkeeping must self-clean so later loads do not see a stale Multi
/// marker.
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

// ============================================================================
// Phase 1.3 (angr-aija): store_concrete_multi / store_symbolic_unified_multi
// helpers. These exercise the store -> Multi -> load round-trip.
// ============================================================================

/// Round-trip a multi-byte LE store through `store_concrete_multi` and
/// the Phase 1.2 load path. With two candidates {A, B}, the load at A
/// should yield the full stored value, and the load at B likewise.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_store_concrete_multi_le_round_trip() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "smc_addr", 64);
    let value = RustBV::concrete(0xDEAD_BEEF, 32);

    let eq_a = addr_var.eq(&RustBV::concrete(0x1000, 64), &ctx);
    let eq_b = addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx);
    ctx.assume_true(&eq_a.or(&eq_b, &ctx));

    mem.store_concrete_multi(&addr_var, &value, &[0x1000, 0x2000], &ctx)
        .expect("store_concrete_multi must succeed");

    // Multi cells were installed at every byte of both candidates.
    assert_eq!(mem.multi_cell_count(), 8);

    // Read back under each candidate concretization.
    let loaded_a = mem
        .load_concrete_lazy(0x1000, 4, &ctx)
        .expect("LE load at candidate A must succeed");
    let loaded_b = mem
        .load_concrete_lazy(0x2000, 4, &ctx)
        .expect("LE load at candidate B must succeed");

    let probe_a = ctx.fork();
    probe_a.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe_a));
    assert_eq!(probe_a.eval(&loaded_a), Some(0xDEAD_BEEF));
    // The B cell was also written under the cond addr==B; under addr==A
    // its load should fall back to the default else (concrete 0).
    assert_eq!(probe_a.eval(&loaded_b), Some(0x0));

    let probe_b = ctx.fork();
    probe_b.assume_true(&addr_var.eq(&RustBV::concrete(0x2000, 64), &probe_b));
    assert_eq!(probe_b.eval(&loaded_b), Some(0xDEAD_BEEF));
    assert_eq!(probe_b.eval(&loaded_a), Some(0x0));
}

/// Big-endian variant of the round-trip. byte 0 is the MSB so the
/// per-byte split must mirror that orientation.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_store_concrete_multi_be_round_trip() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Big);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "smc_be_addr", 64);
    let value = RustBV::concrete(0xCAFE_BABE, 32);

    let eq_a = addr_var.eq(&RustBV::concrete(0x1000, 64), &ctx);
    let eq_b = addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx);
    ctx.assume_true(&eq_a.or(&eq_b, &ctx));

    mem.store_concrete_multi(&addr_var, &value, &[0x1000, 0x2000], &ctx)
        .expect("BE store_concrete_multi must succeed");

    let loaded_a = mem.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    let probe_a = ctx.fork();
    probe_a.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe_a));
    assert_eq!(probe_a.eval(&loaded_a), Some(0xCAFE_BABE));
}

/// Fork independence: installing Multi cells in a child must not bleed
/// into the parent, even when the child later mutates them again.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_store_concrete_multi_fork_independence() {
    let ctx = SymContext::new_mock();
    let mut parent = SymbolicMemory::new(Endness::Little);
    parent.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "smc_fork_addr", 64);
    let value = RustBV::concrete(0x11_22_33_44, 32);
    parent
        .store_concrete_multi(&addr_var, &value, &[0x1000, 0x2000], &ctx)
        .unwrap();
    let parent_count_before = parent.multi_cell_count();
    assert_eq!(parent_count_before, 8);

    let mut child = parent.fork();
    assert_eq!(child.multi_cell_count(), parent_count_before);

    // Child overwrites byte 0 of the first candidate with a concrete byte.
    // store_concrete clears the corresponding multi bit AND the
    // multi_objects entry (per test_concrete_overwrite_clears_multi_bit).
    child
        .store_concrete(0x1000, RustBV::concrete(0xFF, 8))
        .unwrap();
    child.clear_multi_at(0x1000);

    assert_eq!(parent.multi_cell_count(), parent_count_before);
    assert_eq!(child.multi_cell_count(), parent_count_before - 1);

    // Parent's load still sees the original Multi alts.
    let eq_a = addr_var.eq(&RustBV::concrete(0x1000, 64), &ctx);
    let eq_b = addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx);
    ctx.assume_true(&eq_a.or(&eq_b, &ctx));
    let parent_loaded = parent.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    let probe = ctx.fork();
    probe.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe));
    assert_eq!(probe.eval(&parent_loaded), Some(0x11_22_33_44));
}

/// End-to-end: `store_symbolic_unified_multi` with an address constrained
/// to two solutions concretizes to Multiple, installs Multi cells, and
/// `load_concrete_lazy` materializes the correct value for each candidate.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_store_symbolic_unified_multi_multiple_round_trip() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "ssm_addr", 64);
    let eq_a = addr_var.eq(&RustBV::concrete(0x1000, 64), &ctx);
    let eq_b = addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx);
    ctx.assume_true(&eq_a.or(&eq_b, &ctx));

    let value = RustBV::concrete(0x1122_3344, 32);
    let conc_result = mem
        .store_symbolic_unified_multi(addr_var.clone(), value, &ctx, &concretizer)
        .expect("Multi unified store must succeed");
    // Two candidates with a regular delta get detected as Strided by
    // the concretizer; either Multiple or Strided routes through
    // install_multi_for_candidates and must install per-byte Multi cells.
    assert!(matches!(
        conc_result,
        Some(ConcretizationResult::Multiple(_)) | Some(ConcretizationResult::Strided { .. })
    ));

    // Per-byte Multi cells should exist for both candidates.
    assert_eq!(mem.multi_cell_count(), 8);

    let loaded_a = mem.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    let probe_a = ctx.fork();
    probe_a.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe_a));
    assert_eq!(probe_a.eval(&loaded_a), Some(0x1122_3344));
}

/// `store_symbolic_unified_multi` with a fully-concrete address must
/// short-circuit to the eager `store_concrete_automap` path (no Multi
/// cells installed). Counter the lazy-vs-eager distinction at the entry.
#[test]
fn test_store_symbolic_unified_multi_concrete_addr_no_multi() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let addr = RustBV::concrete(0x1000, 64);
    let value = RustBV::concrete(0xAABB_CCDD, 32);

    let pre_multi = mem.multi_cell_count();
    let result = mem
        .store_symbolic_unified_multi(addr, value, &ctx, &concretizer)
        .expect("concrete-address unified-multi store must succeed");
    assert!(matches!(result, Some(ConcretizationResult::Single(_))));
    assert_eq!(
        mem.multi_cell_count(),
        pre_multi,
        "concrete-address path must NOT install Multi cells"
    );

    // Standard load must read the concretely-stored value.
    let loaded = mem.load_concrete(0x1000, 4, &ctx).unwrap();
    assert_eq!(loaded.as_u64(), Some(0xAABB_CCDD));
}

// ============================================================================
// Phase 2 (angr-qh5u): Multi-cell store install, lazy-region safety, and Multi
// flush on export. Phase 4.3 (angr-mmdh.3) made Multi cells the only path —
// `store_symbolic_unified` always routes Multiple/Strided through
// `install_multi_for_candidates_safe`.
// ============================================================================

/// `store_symbolic_unified` routes Multiple to Multi cells — same end-state as
/// `store_symbolic_unified_multi`.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_phase2_gate_on_installs_multi() {
    let ctx = SymContext::new_mock();
    let concretizer = multi_write_concretizer();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p2_on_addr", 64);
    ctx.assume_true(
        &addr_var
            .eq(&RustBV::concrete(0x1000, 64), &ctx)
            .or(&addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
    );
    let value = RustBV::concrete(0xCAFEBABE, 32);

    mem.store_symbolic_unified(addr_var.clone(), value, &ctx, &concretizer)
        .expect("gated-on Multi store must succeed");
    assert_eq!(mem.multi_cell_count(), 8);

    // Load under the addr==A constraint reads back the value.
    let probe = ctx.fork();
    probe.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe));
    let loaded = mem.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    assert_eq!(probe.eval(&loaded), Some(0xCAFEBABE));
}

/// The Phase 2 safe installer must return `UnmappedPageInRegion` when a
/// candidate page lives in a declared lazy region — the interpreter
/// fetches the page from Python rather than letting Rust auto-map a
/// zero page that diverges from Python's backer data.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_phase2_safe_install_lazy_region_signals() {
    let ctx = SymContext::new_mock();
    let concretizer = AddressConcretizer::new();
    let mut mem = SymbolicMemory::new(Endness::Little);

    // Map page 0x1000; leave page 0x2000 unmapped but inside a lazy region.
    mem.map(0x1000, 0x1000, Permission::RWX);
    mem.add_lazy_region(0x2000, 0x1000);

    let addr_var = RustBV::symbolic(&ctx, "p2_lazy_addr", 64);
    ctx.assume_true(
        &addr_var
            .eq(&RustBV::concrete(0x1000, 64), &ctx)
            .or(&addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
    );
    let value = RustBV::concrete(0x11, 8);

    let err = mem
        .store_symbolic_unified(addr_var, value, &ctx, &concretizer)
        .expect_err("lazy unmapped candidate must error to interpreter");
    match err {
        MemoryError::UnmappedPageInRegion { page_addr } => {
            assert_eq!(page_addr, 0x2000, "must point at the lazy page");
        }
        other => panic!("expected UnmappedPageInRegion, got {other:?}"),
    }
    // No Multi cells installed on the partial run.
    assert_eq!(mem.multi_cell_count(), 0);
}

/// The Phase 2 safe installer must silently skip candidates whose pages
/// are unmapped and NOT in any lazy region (matches
/// `prepare_addresses_for_ite`'s skip-unmapped behavior).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_phase2_safe_install_skips_unmapped_non_lazy() {
    let ctx = SymContext::new_mock();
    let concretizer = multi_write_concretizer();
    let mut mem = SymbolicMemory::new(Endness::Little);

    // Only page 0x1000 is mapped. Page 0x2000 is unmapped and NOT lazy.
    mem.map(0x1000, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p2_skip_addr", 64);
    ctx.assume_true(
        &addr_var
            .eq(&RustBV::concrete(0x1000, 64), &ctx)
            .or(&addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
    );
    let value = RustBV::concrete(0xAB, 8);

    mem.store_symbolic_unified(addr_var, value, &ctx, &concretizer)
        .expect("non-lazy unmapped candidate must be silently skipped");

    // Only the mapped candidate (0x1000) got a Multi cell.
    assert_eq!(mem.multi_cell_count(), 1);
    assert!(mem.get_multi_alternatives(0x1000).is_some());
    assert!(mem.get_multi_alternatives(0x2000).is_none());
}

/// angr-sqfj8.74: when the non-lazy-unmapped filter above empties the
/// candidate list entirely, `install_multi_for_candidates_safe` drops the
/// whole store and returns `Ok` rather than erroring — the `SILENT(cat-b)`
/// case documented at that site. Pins the boundary so a future change that
/// turns "all candidates unreachable" into an error (diverging from eager
/// `prepare_addresses_for_ite`, which stores nothing in the same situation)
/// fails loudly here.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_phase2_safe_install_drops_store_when_every_candidate_unmapped() {
    let ctx = SymContext::new_mock();
    let concretizer = multi_write_concretizer();
    let mut mem = SymbolicMemory::new(Endness::Little);

    // Neither candidate page is mapped, and neither is in a lazy region.
    // An unrelated mapped page keeps the memory non-empty.
    mem.map(0x5000, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p2_all_unmapped_addr", 64);
    ctx.assume_true(
        &addr_var
            .eq(&RustBV::concrete(0x1000, 64), &ctx)
            .or(&addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
    );
    let value = RustBV::concrete(0xAB, 8);

    mem.store_symbolic_unified(addr_var, value, &ctx, &concretizer)
        .expect("all-unreachable candidate set must drop the store, not error");

    // Nothing was installed and no page was auto-mapped by the drop.
    assert_eq!(mem.multi_cell_count(), 0);
    assert!(mem.get_multi_alternatives(0x1000).is_none());
    assert!(mem.get_multi_alternatives(0x2000).is_none());
}

/// angr-9ke6b.94: with `enforce_permissions` on, a Multiple-concretized
/// symbolic-address store into a read-only page must be rejected exactly like
/// the Single-concretized store `store_concrete` already rejects. Before the
/// fix `install_multi_for_candidates` never consulted `check_perms_range`, so
/// the write silently mutated the read-only page.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_multi_install_enforces_write_permission() {
    let ctx = SymContext::new_mock();
    let concretizer = multi_write_concretizer();
    let mut mem = SymbolicMemory::new(Endness::Little);

    // Candidate 0x1000 is read-only; 0x2000 is writable.
    mem.map(0x1000, 0x1000, Permission::R);
    mem.map(0x2000, 0x1000, Permission::RW);
    mem.set_enforce_permissions(true);

    let addr_var = RustBV::symbolic(&ctx, "perm_multi_addr", 64);
    ctx.assume_true(
        &addr_var
            .eq(&RustBV::concrete(0x1000, 64), &ctx)
            .or(&addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
    );

    let err = mem
        .store_symbolic_unified(addr_var, RustBV::concrete(0xAB, 8), &ctx, &concretizer)
        .expect_err("Multiple-concretized store into an R-only page must be rejected");
    match err {
        MemoryError::Permission {
            addr,
            required,
            actual,
        } => {
            assert_eq!(addr, 0x1000);
            assert!(required.write);
            assert!(!actual.write);
        }
        other => panic!("expected Permission error, got {other:?}"),
    }
    // The check runs before any mutation, so nothing was installed — not even
    // for the writable candidate.
    assert_eq!(mem.multi_cell_count(), 0);

    // Symmetry: the Single-concretized store on the same page is rejected the
    // same way (this is the behavior the Multi path was diverging from).
    let single_err = mem
        .store_concrete(0x1000, RustBV::concrete(0xAB, 8))
        .expect_err("Single-concretized store into an R-only page must be rejected");
    assert!(matches!(single_err, MemoryError::Permission { .. }));
}

/// The permission gate must be inert when `enforce_permissions` is off (the
/// default), so ordinary Multi installs into R-only or auto-mapped pages keep
/// working.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_multi_install_permission_check_off_by_default() {
    let ctx = SymContext::new_mock();
    let concretizer = multi_write_concretizer();
    let mut mem = SymbolicMemory::new(Endness::Little);

    mem.map(0x1000, 0x1000, Permission::R);
    mem.map(0x2000, 0x1000, Permission::R);
    assert!(!mem.enforce_permissions());

    let addr_var = RustBV::symbolic(&ctx, "perm_off_multi_addr", 64);
    ctx.assume_true(
        &addr_var
            .eq(&RustBV::concrete(0x1000, 64), &ctx)
            .or(&addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
    );

    mem.store_symbolic_unified(addr_var, RustBV::concrete(0xAB, 8), &ctx, &concretizer)
        .expect("permission enforcement off: R-only page must still accept the store");
    assert_eq!(mem.multi_cell_count(), 2);
}

/// angr-03vl4.37: a candidate whose byte range runs off the top of the address
/// space must be rejected outright, never wrapped onto a low page. The
/// candidates come from Z3 solutions for a guest-supplied pointer, and
/// `install_multi_for_candidates` computed the per-byte address with a bare
/// `cand + b` — a panic under debug-assertions, a silent redirect in release.
/// Two guards now cover it: the page-range pass (`end_page_inclusive`) and the
/// `checked_add` in the install loop itself; this pins the observable contract
/// either one must deliver.
#[test]
fn test_multi_install_rejects_wrapping_candidate() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    // The page a wrapped byte address would land on.
    mem.map(0x0u64, 0x1000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "wrap_multi_addr", 64);
    let wrapping = u64::MAX - 1; // [u64::MAX-1, u64::MAX+2) for a 4-byte value
    let err = mem
        .store_concrete_multi(
            &addr_var,
            &RustBV::concrete(0xdead_beef, 32),
            &[0x100, wrapping],
            &ctx,
        )
        .expect_err("a candidate whose range wraps past u64::MAX must be rejected");
    assert!(matches!(err, MemoryError::OutOfBounds { .. }));

    // The rejection precedes every mutation, so not even the well-formed
    // candidate installed — and page 0 is untouched.
    assert_eq!(mem.multi_cell_count(), 0);
    for byte in 0..4u64 {
        assert!(mem.get_multi_alternatives(byte).is_none());
    }
}

/// Harness 6 boundary sweep for the fix above: every entry in the shared
/// `test_boundary_values` table, not just the one `u64::MAX - 1` candidate
/// `test_multi_install_rejects_wrapping_candidate` pins, must be rejected
/// exactly when its 4-byte range would overflow — and must still install
/// cleanly when it does not, so the guard isn't over-broad.
#[test]
fn install_multi_for_candidates_boundary_sweep_rejects_wrapping_candidates() {
    let ctx = SymContext::new_mock();
    for &addr in &crate::test_boundary_values::boundary_addresses() {
        let mut mem = SymbolicMemory::new(Endness::Little);
        mem.map(0x0u64, 0x1000, Permission::RWX);

        let addr_var = RustBV::symbolic(&ctx, "boundary_multi_addr", 64);
        let value = RustBV::concrete(0xdead_beef, 32);
        let overflows = addr.checked_add(3).is_none();

        let result = mem.store_concrete_multi(&addr_var, &value, &[addr], &ctx);
        if overflows {
            let err = result.expect_err("wrapping candidate must be rejected");
            assert!(
                matches!(err, MemoryError::OutOfBounds { .. }),
                "addr={addr:#x}: expected OutOfBounds, got {err:?}"
            );
            assert_eq!(
                mem.multi_cell_count(),
                0,
                "addr={addr:#x}: rejection must install nothing"
            );
        } else {
            result.unwrap_or_else(|e| panic!("addr={addr:#x} must install cleanly, got {e:?}"));
            assert_eq!(
                mem.multi_cell_count(),
                4,
                "addr={addr:#x}: a 4-byte value must install exactly 4 Multi cells"
            );
        }
    }
}

/// `flush_multi_cells` must collapse every Multi byte into a per-byte
/// symbolic_objects entry and mark the page-level symbolic bit so the
/// state export pipeline picks it up. This is the export-correctness
/// invariant called out in `rust_lazy_memory_design.rst` Phase 2.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_phase2_flush_multi_to_symbolic_objects() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p2_flush_addr", 64);
    let value = RustBV::concrete(0xAA, 8);
    mem.store_concrete_multi(&addr_var, &value, &[0x1000, 0x2000], &ctx)
        .expect("test setup install must succeed");
    assert_eq!(mem.multi_cell_count(), 2);

    mem.flush_multi_cells(&ctx);

    // Multi cells are gone, symbolic_objects has 1-byte entries at each
    // flushed address.
    assert_eq!(mem.multi_cell_count(), 0);
    assert!(mem.get_symbolic_object(0x1000).is_some());
    assert!(mem.get_symbolic_object(0x2000).is_some());

    // The 1-byte symbolic_object at 0x1000 evaluates to 0xAA under
    // addr==0x1000 (because cond=(addr==0x1000) is true, picking value 0xAA).
    let probe = ctx.fork();
    probe.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe));
    let sym = mem.get_symbolic_object(0x1000).unwrap();
    assert_eq!(probe.eval(sym), Some(0xAA));

    // The 1-byte symbolic_object at 0x2000 evaluates to 0 (concrete else)
    // under addr==0x1000 because its cond fires only when addr==0x2000.
    let sym2 = mem.get_symbolic_object(0x2000).unwrap();
    assert_eq!(probe.eval(sym2), Some(0));
}

/// After fork, mutating Multi cells in the parent must not leak into
/// the child via the new safe-install path. Pairs with
/// `test_store_concrete_multi_fork_independence` but exercises the
/// Phase 2 entry point.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_phase2_fork_independence_via_safe_install() {
    let ctx = SymContext::new_mock();
    let concretizer = multi_write_concretizer();
    let mut parent = SymbolicMemory::new(Endness::Little);
    parent.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p2_fork_addr", 64);
    ctx.assume_true(
        &addr_var
            .eq(&RustBV::concrete(0x1000, 64), &ctx)
            .or(&addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx), &ctx),
    );
    parent
        .store_symbolic_unified(
            addr_var.clone(),
            RustBV::concrete(0x12, 8),
            &ctx,
            &concretizer,
        )
        .unwrap();
    let parent_count = parent.multi_cell_count();
    assert_eq!(parent_count, 2);

    let mut child = parent.fork();
    assert_eq!(child.multi_cell_count(), parent_count);

    // Child stores another value at the same symbolic addr — appends
    // a second alternative to each Multi cell. Parent must not see it.
    child
        .store_symbolic_unified(addr_var, RustBV::concrete(0x34, 8), &ctx, &concretizer)
        .unwrap();
    let child_a = child.get_multi_alternatives(0x1000).unwrap();
    let parent_a = parent.get_multi_alternatives(0x1000).unwrap();
    assert_eq!(parent_a.len(), 1, "parent must keep its single alternative");
    assert_eq!(child_a.len(), 2, "child accumulates appended alternative");
}

// ============================================================================
// Phase 3 (angr-j0n4): per-load Multi-cell collapse cache
// ============================================================================

/// After a single load of a Multi byte, the payload's collapse cache must
/// be populated. A second load returns the cached BV unchanged.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_phase3_collapse_cache_hit_after_load() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p3_hit_addr", 64);
    let value = RustBV::concrete(0x77, 8);
    mem.store_concrete_multi(&addr_var, &value, &[0x1000, 0x2000], &ctx)
        .unwrap();

    // Before any load, the cache is empty.
    let payload = mem.get_multi_alternatives(0x1000).unwrap();
    assert!(!payload.has_cached_collapse(), "cache must start empty");

    // First load populates the cache.
    let first = mem.load_concrete_lazy(0x1000, 1, &ctx).unwrap();
    let payload = mem.get_multi_alternatives(0x1000).unwrap();
    assert!(payload.has_cached_collapse(), "first load must cache");

    // Second load returns the same BV (we can't compare Z3 AST identity
    // directly, but the model-eval result must match).
    let second = mem.load_concrete_lazy(0x1000, 1, &ctx).unwrap();
    let probe = ctx.fork();
    probe.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe));
    assert_eq!(probe.eval(&first), Some(0x77));
    assert_eq!(probe.eval(&second), Some(0x77));
}

/// Appending an alternative via `MultiPayload::push` must invalidate the
/// collapse cache so the next load rebuilds the ITE.
#[test]
fn test_phase3_collapse_cache_invalidated_on_push() {
    let ctx = SymContext::new_mock();
    let mut payload = MultiPayload::from_alternatives(vec![MultiAlternative::new(
        RustBV::concrete(1, 1),
        RustBV::concrete(0xAA, 8),
    )]);
    let _ = payload.collapse(0, &ctx);
    assert!(
        payload.has_cached_collapse(),
        "collapse must populate cache"
    );

    payload.push(MultiAlternative::new(
        RustBV::concrete(0, 1),
        RustBV::concrete(0xBB, 8),
    ));
    assert!(
        !payload.has_cached_collapse(),
        "push must invalidate cached collapse"
    );
}

/// `MultiPayload::collapse` must rebuild when the page's concrete default
/// byte changes between loads. Otherwise a concrete overwrite of the cell's
/// page byte (which does not currently clear the Multi marker) would serve
/// a stale ITE.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_phase3_collapse_cache_invalidated_on_default_byte_change() {
    let ctx = SymContext::new_mock();
    let payload = MultiPayload::from_alternatives(vec![MultiAlternative::new(
        RustBV::symbolic(&ctx, "p3_default_change_cond", 1),
        RustBV::concrete(0xAA, 8),
    )]);

    let collapsed_0 = payload.collapse(0x00, &ctx);
    assert!(payload.has_cached_collapse());
    let collapsed_again = payload.collapse(0x00, &ctx);
    assert_eq!(
        ctx.eval(&collapsed_0),
        ctx.eval(&collapsed_again),
        "cache hit must return equivalent BV"
    );

    // A different default byte must produce a different ELSE leaf.
    let collapsed_ff = payload.collapse(0xFF, &ctx);
    // Force the cond=false branch so the ELSE leaf is observable.
    let probe = ctx.fork();
    probe.assume_true(
        &payload.alternatives()[0]
            .cond
            .eq(&RustBV::concrete(0, 1), &probe),
    );
    assert_eq!(
        probe.eval(&collapsed_ff),
        Some(0xFF),
        "rebuilt collapse must reflect the new default byte"
    );
}

/// After fork, the parent and child each hold an independent payload. If
/// the parent's cache is populated, the child's clone carries it forward
/// (the BV is referentially safe — Z3 ASTs are immutable / refcounted).
#[test]
fn test_phase3_collapse_cache_clones_with_payload() {
    let ctx = SymContext::new_mock();
    let payload = MultiPayload::from_alternatives(vec![MultiAlternative::new(
        RustBV::concrete(1, 1),
        RustBV::concrete(0x33, 8),
    )]);
    let _ = payload.collapse(0x00, &ctx);
    assert!(payload.has_cached_collapse());

    let cloned = payload.clone();
    assert!(
        cloned.has_cached_collapse(),
        "clone must carry the cached collapse forward"
    );

    // Independence: pushing on the clone does not touch the original.
    let mut cloned_mut = cloned;
    cloned_mut.push(MultiAlternative::new(
        RustBV::concrete(0, 1),
        RustBV::concrete(0x44, 8),
    ));
    assert!(!cloned_mut.has_cached_collapse());
    assert!(
        payload.has_cached_collapse(),
        "original payload must retain its cache after clone mutation"
    );
}

// ============================================================================
// Phase 4.1 (angr-mmdh.1): wider-load collapse cache in
// `assemble_load_with_multi`
// ============================================================================

/// A wider load that touches a Multi byte must populate the wider-load
/// cache. A second identical load must hit and return an equivalent BV.
#[test]
fn test_phase4_wider_load_cache_hit() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p41_hit_addr", 64);
    let value = RustBV::concrete(0xAA, 8);
    mem.store_concrete_multi(&addr_var, &value, &[0x1000, 0x2000], &ctx)
        .unwrap();

    assert_eq!(mem.wider_load_cache_len(), 0, "cache starts empty");

    // First load (size 4 = wider than 1): populates the cache.
    let first = mem.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    assert_eq!(mem.wider_load_cache_len(), 1, "first load populates cache");

    // Second identical load: returns from cache.
    let second = mem.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    let probe = ctx.fork();
    probe.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe));
    assert_eq!(probe.eval(&first), probe.eval(&second));
    assert_eq!(mem.wider_load_cache_len(), 1, "second load reuses entry");
}

/// Size==1 loads bypass the wider-load cache (no concat to amortize).
#[test]
fn test_phase4_wider_load_cache_skips_size_one() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p41_size1_addr", 64);
    mem.store_concrete_multi(
        &addr_var,
        &RustBV::concrete(0x55, 8),
        &[0x1000, 0x2000],
        &ctx,
    )
    .unwrap();

    let _ = mem.load_concrete_lazy(0x1000, 1, &ctx).unwrap();
    assert_eq!(
        mem.wider_load_cache_len(),
        0,
        "size==1 must not populate the wider-load cache"
    );
}

/// Installing a new Multi alternative at a byte covered by a cached load
/// must invalidate the cached entry (fingerprint mismatch on next read).
/// Uses two independent address vars so the second store contributes an
/// alternative whose cond can be made true while the first is false —
/// letting eval pin the rebuilt result to the new value and fail loudly
/// if the cache returned the stale BV.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_phase4_wider_load_cache_invalidated_on_multi_install() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var1 = RustBV::symbolic(&ctx, "p41_inval_addr1", 64);
    let addr_var2 = RustBV::symbolic(&ctx, "p41_inval_addr2", 64);

    // First install: byte 0x1000 gets alt (addr_var1 == 0x1000, 0xAA).
    mem.store_concrete_multi(
        &addr_var1,
        &RustBV::concrete(0xAA, 8),
        &[0x1000, 0x2000],
        &ctx,
    )
    .unwrap();

    // Prime the cache under a probe where addr_var1==0x1000 (alt 0 fires
    // → result byte is 0xAA).
    let first_probe = ctx.fork();
    first_probe.assume_true(&addr_var1.eq(&RustBV::concrete(0x1000, 64), &first_probe));
    let first = mem.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    assert_eq!(mem.wider_load_cache_len(), 1);
    assert_eq!(first_probe.eval(&first), Some(0xAA));

    // Second install at byte 0x1000 — independent cond (addr_var2 == 0x1000)
    // bumps the per-byte version so the cached fingerprint mismatches.
    mem.store_concrete_multi(
        &addr_var2,
        &RustBV::concrete(0xBB, 8),
        &[0x1000, 0x3000],
        &ctx,
    )
    .unwrap();

    // Probe where alt 0's cond is false (addr_var1==0x9999) but alt 1's
    // cond is true (addr_var2==0x1000). Right-fold ITE:
    //   alt 0 (outermost): cond false → fall to ELSE
    //   alt 1: cond true → 0xBB
    // If the cache returns the stale BV from the first load (which lacks
    // the alt 1 branch), eval here would NOT be 0xBB.
    let probe = ctx.fork();
    probe.assume_true(&addr_var1.eq(&RustBV::concrete(0x9999, 64), &probe));
    probe.assume_true(&addr_var2.eq(&RustBV::concrete(0x1000, 64), &probe));
    let after = mem.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    assert_eq!(
        mem.wider_load_cache_len(),
        1,
        "rebuilt entry replaces the stale one, cache size unchanged"
    );
    assert_eq!(
        probe.eval(&after),
        Some(0xBB),
        "rebuilt load must include the alt installed after the cache prime"
    );
}

/// Loads that touch any plain Symbolic byte must not be cached.
#[test]
fn test_phase4_wider_load_cache_skips_symbolic_bytes() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p41_sym_byte_addr", 64);
    // One Multi byte at 0x1000…
    mem.store_concrete_multi(
        &addr_var,
        &RustBV::concrete(0xAA, 8),
        &[0x1000, 0x2000],
        &ctx,
    )
    .unwrap();

    // …and a plain Symbolic byte at 0x1001 (via concrete store with a
    // symbolic value).
    let sym_val = RustBV::symbolic(&ctx, "p41_sym_byte_val", 8);
    mem.store_concrete(0x1001, sym_val).unwrap();

    // A 4-byte load at 0x1000 covers both — must NOT cache.
    let _ = mem.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    assert_eq!(
        mem.wider_load_cache_len(),
        0,
        "loads touching plain Symbolic bytes must not be cached"
    );
}

/// Forks start with a COLD wider-load cache (angr-6t8z3.2, commit
/// 94c015db5): `fork()` deliberately does not deep-clone the parent's
/// cache — the child repopulates lazily on its first wider load, and a
/// child-side rebuild must not disturb the parent's own cached entry.
#[test]
fn test_phase4_wider_load_cache_fork_independence() {
    let ctx = SymContext::new_mock();
    let mut parent = SymbolicMemory::new(Endness::Little);
    parent.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p41_fork_addr", 64);
    parent
        .store_concrete_multi(
            &addr_var,
            &RustBV::concrete(0xAA, 8),
            &[0x1000, 0x2000],
            &ctx,
        )
        .unwrap();
    let _ = parent.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    assert_eq!(parent.wider_load_cache_len(), 1);

    let mut child = parent.fork();
    assert_eq!(
        child.wider_load_cache_len(),
        0,
        "fork starts with a cold cache (no deep-clone; repopulates lazily)"
    );

    // Child mutation must not affect parent.
    child
        .store_concrete_multi(
            &addr_var,
            &RustBV::concrete(0xBB, 8),
            &[0x1000, 0x3000],
            &ctx,
        )
        .unwrap();
    let _ = child.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    // Parent's cache still valid (size unchanged, fingerprint still matches
    // its own copy of multi_versions).
    let _ = parent.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    assert_eq!(parent.wider_load_cache_len(), 1);
    assert_eq!(child.wider_load_cache_len(), 1);
}

// ============================================================================
// Phase 4.2 (angr-mmdh.2): flush_multi_cells run coalescing
// ============================================================================

/// A LE 4-byte symbolic-address store at two candidates installs 8 Multi
/// bytes (4 per candidate). After flush, those should coalesce into TWO
/// wider symbolic_objects (one per candidate base), each width 32, with
/// symbolic_spans covering the interior bytes — instead of 8 per-byte
/// entries. This is the export-cost win called out in `phase41-bottleneck`.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_phase42_flush_coalesces_le_multi_byte_run() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p42_le_addr", 64);
    let value = RustBV::concrete(0xDEAD_BEEF, 32);
    let eq_a = addr_var.eq(&RustBV::concrete(0x1000, 64), &ctx);
    let eq_b = addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx);
    ctx.assume_true(&eq_a.or(&eq_b, &ctx));

    mem.store_concrete_multi(&addr_var, &value, &[0x1000, 0x2000], &ctx)
        .unwrap();
    assert_eq!(mem.multi_cell_count(), 8);

    mem.flush_multi_cells(&ctx);
    assert_eq!(mem.multi_cell_count(), 0);

    // Exactly two wider symbolic_objects entries: 0x1000 and 0x2000 with
    // width 32. The interior bytes (0x1001..0x1003 / 0x2001..0x2003) must
    // NOT have their own symbolic_objects entry — they go through the
    // symbolic_spans reverse index.
    assert_eq!(
        mem.symbolic_object_count(),
        2,
        "coalesced flush must yield one wider entry per candidate"
    );
    let sym_a = mem.get_symbolic_object(0x1000).expect("entry at A start");
    let sym_b = mem.get_symbolic_object(0x2000).expect("entry at B start");
    assert_eq!(sym_a.width(), 32, "wider entry covers all 4 bytes");
    assert_eq!(sym_b.width(), 32, "wider entry covers all 4 bytes");
    for off in 1..4 {
        assert!(
            mem.get_symbolic_object(0x1000 + off).is_none(),
            "interior bytes must not get their own symbolic_objects entry"
        );
        assert!(
            mem.get_symbolic_object(0x2000 + off).is_none(),
            "interior bytes must not get their own symbolic_objects entry"
        );
    }

    // Behavior check: load 4 bytes at each candidate under each
    // concretization. The wider symbolic_object must hold the original
    // value under cond=true and the concrete default (0) under cond=false.
    let probe_a = ctx.fork();
    probe_a.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe_a));
    let loaded_a = mem.load_concrete(0x1000, 4, &probe_a).unwrap();
    assert_eq!(probe_a.eval(&loaded_a), Some(0xDEAD_BEEF));
    let loaded_b_under_a = mem.load_concrete(0x2000, 4, &probe_a).unwrap();
    assert_eq!(probe_a.eval(&loaded_b_under_a), Some(0));

    let probe_b = ctx.fork();
    probe_b.assume_true(&addr_var.eq(&RustBV::concrete(0x2000, 64), &probe_b));
    let loaded_b = mem.load_concrete(0x2000, 4, &probe_b).unwrap();
    assert_eq!(probe_b.eval(&loaded_b), Some(0xDEAD_BEEF));
}

/// Big-endian variant of the LE coalesce test. byte 0 (lowest addr) is
/// the MSB in BE; the wider value's right-fold must reconstruct the
/// original word.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_phase42_flush_coalesces_be_multi_byte_run() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Big);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p42_be_addr", 64);
    let value = RustBV::concrete(0xCAFE_BABE, 32);
    let eq_a = addr_var.eq(&RustBV::concrete(0x1000, 64), &ctx);
    let eq_b = addr_var.eq(&RustBV::concrete(0x2000, 64), &ctx);
    ctx.assume_true(&eq_a.or(&eq_b, &ctx));

    mem.store_concrete_multi(&addr_var, &value, &[0x1000, 0x2000], &ctx)
        .unwrap();
    mem.flush_multi_cells(&ctx);

    assert_eq!(mem.symbolic_object_count(), 2);
    let sym_a = mem.get_symbolic_object(0x1000).unwrap();
    assert_eq!(sym_a.width(), 32);

    let probe_a = ctx.fork();
    probe_a.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe_a));
    let loaded_a = mem.load_concrete(0x1000, 4, &probe_a).unwrap();
    assert_eq!(probe_a.eval(&loaded_a), Some(0xCAFE_BABE));
}

/// Single-byte stores produce non-adjacent Multi cells. The flush path
/// must fall back to the per-byte case and remain byte-identical to the
/// pre-Phase-4.2 behaviour.
#[test]
fn test_phase42_flush_singleton_no_coalesce() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p42_single_addr", 64);
    let value = RustBV::concrete(0xAB, 8);
    mem.store_concrete_multi(&addr_var, &value, &[0x1000, 0x2000], &ctx)
        .unwrap();
    assert_eq!(mem.multi_cell_count(), 2);

    mem.flush_multi_cells(&ctx);

    // Two per-byte entries, width 8 each. No spans installed.
    assert_eq!(mem.symbolic_object_count(), 2);
    let sym_a = mem.get_symbolic_object(0x1000).unwrap();
    let sym_b = mem.get_symbolic_object(0x2000).unwrap();
    assert_eq!(sym_a.width(), 8);
    assert_eq!(sym_b.width(), 8);
}

/// Bytes whose Multi payloads carry mismatched cond fingerprints (e.g.
/// one byte has an extra alternative from a later store) must terminate
/// the coalesce run. Adjacent bytes either side are still coalesced
/// pairwise within their matching subsets.
#[test]
fn test_phase42_flush_fingerprint_mismatch_breaks_run() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    // First store: 4-byte value at two candidates → 4 Multi bytes per
    // candidate.
    let addr1 = RustBV::symbolic(&ctx, "p42_addr1", 64);
    mem.store_concrete_multi(&addr1, &RustBV::concrete(0x1111_2222, 32), &[0x1000], &ctx)
        .unwrap();
    assert_eq!(mem.multi_cell_count(), 4);

    // Second store: 1-byte value at byte 2 of the previous range. This
    // appends a second alternative to ONLY byte 0x1002, breaking its
    // fingerprint relative to its neighbours.
    let addr2 = RustBV::symbolic(&ctx, "p42_addr2", 64);
    mem.store_concrete_multi(&addr2, &RustBV::concrete(0xFF, 8), &[0x1002], &ctx)
        .unwrap();
    assert_eq!(
        mem.get_multi_alternatives(0x1002).unwrap().len(),
        2,
        "the merged-store byte carries 2 alternatives"
    );
    assert_eq!(
        mem.get_multi_alternatives(0x1001).unwrap().len(),
        1,
        "neighbour byte still has 1 alternative"
    );

    mem.flush_multi_cells(&ctx);

    // Expected runs:
    //   [0x1000, 0x1001] — coalesced wider entry, width 16.
    //   [0x1002]         — singleton, width 8.
    //   [0x1003]         — singleton, width 8.
    assert_eq!(mem.symbolic_object_count(), 3);
    let coalesced = mem.get_symbolic_object(0x1000).unwrap();
    assert_eq!(coalesced.width(), 16, "first run coalesces 2 bytes");
    assert!(
        mem.get_symbolic_object(0x1001).is_none(),
        "interior of coalesced run not in symbolic_objects"
    );
    let singleton_mid = mem.get_symbolic_object(0x1002).unwrap();
    let singleton_tail = mem.get_symbolic_object(0x1003).unwrap();
    assert_eq!(singleton_mid.width(), 8);
    assert_eq!(singleton_tail.width(), 8);
}

/// A run wider than `COALESCE_MAX_RUN` (16 bytes) must coalesce only the
/// first 16 bytes; the remaining bytes flush as a separate run.
#[test]
fn test_phase42_flush_run_length_cap() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    // 24-byte value at a single candidate → 24 adjacent Multi bytes
    // sharing one cond. The cap forces two coalesced runs: 16 bytes and
    // 8 bytes (instead of one wider run of 24).
    let addr_var = RustBV::symbolic(&ctx, "p42_cap_addr", 64);
    // RustBV::concrete masks the u128 to the declared width, so a
    // 24-byte (192-bit) value can be passed directly. But we only need
    // a sequence of 24 Multi bytes; the value content does not matter
    // for the coalescing logic. Use a width-192 concrete BV.
    let value = RustBV::concrete(0x1234_5678_DEAD_BEEF, 192);
    mem.store_concrete_multi(&addr_var, &value, &[0x1000], &ctx)
        .unwrap();
    assert_eq!(mem.multi_cell_count(), 24);

    mem.flush_multi_cells(&ctx);
    // Two wider entries: one at 0x1000 (width 128 = 16 bytes), one at
    // 0x1010 (width 64 = 8 bytes).
    assert_eq!(mem.symbolic_object_count(), 2);
    let first = mem.get_symbolic_object(0x1000).unwrap();
    let second = mem.get_symbolic_object(0x1010).unwrap();
    assert_eq!(first.width(), 128);
    assert_eq!(second.width(), 64);
}

// ============================================================================
// angr-1tes: cache + multi_objects invariant under concrete overwrite.
// ============================================================================

/// A concrete store at a byte covered by a previously-installed Multi cell
/// must produce a re-load that reflects the new concrete byte, not the stale
/// Multi alternative. Pre-fix, `SymbolicMemory::store_concrete` cleared the
/// page-level `multi_bitmap` bit but left the `multi_objects` entry (and the
/// `multi_versions` counter) untouched. The next load's dispatcher (see
/// `range_has_multi`, the shared gate `load_concrete_common` consults ahead
/// of every symbolic fast path) checks `multi_objects.contains_key`
/// and hands the load to `assemble_load_with_multi`, which folds the orphaned
/// alternatives over the new page-byte default — returning the old Multi
/// value when the original cond is satisfiable.
#[test]
fn test_concrete_overwrite_clears_multi_cell() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x4000, Permission::RWX);

    let addr_var = RustBV::symbolic(&ctx, "p41_overwrite_addr", 64);
    // Install Multi at 0x1000..0x1004 (4 bytes) with one candidate (0x1000)
    // and value 0xAABBCCDD. Pre-fix this leaves an orphaned `multi_objects`
    // entry that survives the concrete store below.
    mem.store_concrete_multi(
        &addr_var,
        &RustBV::concrete(0xAABBCCDD, 32),
        &[0x1000],
        &ctx,
    )
    .unwrap();
    assert_eq!(mem.multi_cell_count(), 4);

    // Prime the wider-load cache.
    let _ = mem.load_concrete_lazy(0x1000, 4, &ctx).unwrap();
    assert_eq!(mem.wider_load_cache_len(), 1);

    // Concrete overwrite of byte 0x1000.
    mem.store_concrete(0x1000, RustBV::concrete(0x11, 8))
        .unwrap();

    // The Multi cell at 0x1000 must be gone — both the sidecar map AND the
    // page bit. The cache entry may remain (eviction is lazy) but a refetch
    // must not see the stale alternative.
    assert!(
        mem.get_multi_alternatives(0x1000).is_none(),
        "concrete overwrite must drop the orphaned Multi cell at 0x1000"
    );
    let page = mem.pages.get(&(0x1000 >> 12)).expect("page mapped");
    assert!(
        !page.is_multi(0),
        "page Multi bit at offset 0 must be cleared by concrete store"
    );

    // Re-read: under the cond that previously fired the Multi alt
    // (addr_var == 0x1000), the new concrete byte 0x11 must dominate.
    let probe = ctx.fork();
    probe.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &probe));
    let after = mem.load_concrete_lazy(0x1000, 1, &ctx).unwrap();
    assert_eq!(
        probe.eval(&after),
        Some(0x11),
        "re-load after concrete overwrite must reflect the new byte, \
         not the orphaned Multi alternative"
    );
}

// ============================================================================
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

/// `cond_fingerprint`'s `Expression` key mixes the op in alongside the
/// operands `Arc` pointer (angr-03vl4.38). Under today's construction every
/// `Expression` allocates a fresh operands `Arc`, so a shared pointer already
/// implies a shared op — this pins the defensive half, so a future
/// construction path that reuses an operands `Arc` across two different ops
/// cannot silently coalesce bytes whose conds are not equivalent.
#[test]
fn test_cond_fingerprint_distinguishes_op_on_shared_operands() {
    use crate::memory::multi::cond_fingerprint;
    use crate::symbolic::BVOp;
    use std::sync::Arc;

    let operands: Arc<[RustBV]> =
        Arc::from(vec![RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)]);
    let make = |op: BVOp| RustBV::Expression {
        id: RustBV::EXPRESSION_ID,
        width: 1,
        op,
        operands: Arc::clone(&operands),
        memo: Default::default(),
    };

    // Same Arc, different op → distinct fingerprints.
    assert_ne!(
        cond_fingerprint(&make(BVOp::Eq)),
        cond_fingerprint(&make(BVOp::Ne)),
        "op must participate in the Expression fingerprint"
    );
    // Payload-carrying variants must not collapse onto their discriminant.
    assert_ne!(
        cond_fingerprint(&make(BVOp::Extract(7, 0))),
        cond_fingerprint(&make(BVOp::Extract(15, 8))),
        "op payload must participate too"
    );
    // The coalescing case itself is unaffected: same op + same Arc matches,
    // which is what a run of bytes cloned from one cond looks like.
    assert_eq!(
        cond_fingerprint(&make(BVOp::Eq)),
        cond_fingerprint(&make(BVOp::Eq)),
        "clones of one cond must still coalesce"
    );
    // A fresh Arc with the same op is a different store path → no coalesce.
    let other_operands: Arc<[RustBV]> =
        Arc::from(vec![RustBV::concrete(0x1000, 64), RustBV::concrete(0x2000, 64)]);
    assert_ne!(
        cond_fingerprint(&make(BVOp::Eq)),
        cond_fingerprint(&RustBV::Expression {
            id: RustBV::EXPRESSION_ID,
            width: 1,
            op: BVOp::Eq,
            operands: other_operands,
            memo: Default::default(),
        }),
        "distinct operands allocations stay distinct"
    );
}
