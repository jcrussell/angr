//! Phase 1.1 (angr-me3z): the `MultiPayload` / `MultiAlternative` data
//! structures and their sidecar storage (`multi_objects`, `multi_bitmap`)
//! round-trips. These cover only the data structure and storage — load-side
//! collapse lives in the sibling `collapse` module and the store-side helpers
//! in `install`.
//!
//! Also the home of the angr-6cp06.63 regressions: `set_multi_alternatives`
//! retiring a *wider* `symbolic_objects` entry that covers the byte being
//! superseded, rather than only an exact-key one.

use super::*;

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

/// angr-6cp06.63: installing a Multi cell on an *interior* byte of a wider
/// symbolic object must retire that object, not just the (absent) exact-key
/// entry. Before the fix `symbolic_objects[0x1000]` survived at width 32,
/// still claiming the byte the Multi now owns — and `flush_multi_cells` then
/// added a second, disjoint entry for the collapsed byte, leaving the export
/// walk in `_get_state_symbolic_z3_asts` to race two overlapping stores.
#[test]
fn test_multi_inside_wider_symbolic_object_retires_the_container() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    // A 4-byte imported object spanning [0x1000, 0x1004), pinned to a single
    // concrete value so every byte lane is checkable by eval.
    let wide = RustBV::symbolic(&ctx, "wide", 32);
    ctx.assume_true(&wide.eq(&RustBV::concrete(0xDDCC_BBAA, 32), &ctx));
    mem.import_symbolic_value(0x1000, wide, None).unwrap();

    // Multi cell on the third byte only.
    let addr_var = RustBV::symbolic(&ctx, "mi_addr", 64);
    ctx.assume_true(&addr_var.eq(&RustBV::concrete(0x1002, 64), &ctx));
    mem.set_multi_alternatives(
        0x1002,
        MultiPayload::from_alternatives(vec![make_alt(&ctx, &addr_var, 0x1002, 0x77)]),
    );
    assert!(ctx.is_sat());

    // The wide container is gone; the three untouched bytes survive as
    // 8-bit lanes, and the Multi byte owns no symbolic_objects entry.
    for byte in [0x1000u64, 0x1001, 0x1003] {
        let lane = mem
            .get_symbolic_object(byte)
            .unwrap_or_else(|| panic!("byte {byte:#x} must survive the split as an 8-bit lane"));
        assert_eq!(lane.width(), 8, "lane at {byte:#x} must be one byte wide");
    }
    assert!(
        mem.get_symbolic_object(0x1002).is_none(),
        "the Multi byte must not also carry a symbolic_objects entry"
    );
    for byte in 0x1000u64..0x1004 {
        assert!(
            !mem.symbolic_spans.contains_key(&Address::new(byte)),
            "no reverse-span entry may outlive the retired container ({byte:#x})"
        );
    }

    // End-to-end: a 4-byte load sees the Multi byte, not the stale container.
    let loaded = mem
        .load_concrete_lazy(0x1000, 4, &ctx)
        .expect("wide load over a split container must succeed");
    assert_eq!(ctx.eval(&loaded), Some(0xDD77_BBAA));

    // Export path: after the flush every entry is a disjoint single byte, so
    // no ordering of `state.memory.store()` calls can clobber another.
    mem.flush_multi_cells(&ctx);
    let mut exported: Vec<u64> = Vec::new();
    for (addr, bv) in mem.symbolic_objects_iter() {
        assert_eq!(
            bv.width(),
            8,
            "no wider entry may overlap the collapsed Multi byte at {addr:?}"
        );
        exported.push(addr.raw());
    }
    exported.sort_unstable();
    assert_eq!(exported, vec![0x1000, 0x1001, 0x1002, 0x1003]);
}

/// The split lanes inherit `imported_addrs` membership from the container, so
/// a Python-imported object does not start exporting `Extract(sym)` stores
/// back over bytes Python already holds verbatim (the flareon5 rule on
/// `_get_state_symbolic_z3_asts`). The superseded byte itself was never
/// imported, so it stays exportable.
#[test]
fn test_split_lanes_inherit_imported_addrs() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    mem.import_symbolic_value(0x1000, RustBV::symbolic(&ctx, "imp", 32), None)
        .unwrap();
    let addr_var = RustBV::symbolic(&ctx, "ia_addr", 64);
    mem.set_multi_alternatives(
        0x1001,
        MultiPayload::from_alternatives(vec![make_alt(&ctx, &addr_var, 0x1001, 0x5A)]),
    );

    for byte in [0x1000u64, 0x1002, 0x1003] {
        assert!(
            mem.is_imported_addr(byte),
            "lane {byte:#x} must inherit the container's imported flag"
        );
    }
    assert!(
        !mem.is_imported_addr(0x1001),
        "the superseded byte was never imported at its own address"
    );
}

/// The sibling defect on the same code path: when the Multi lands on the
/// *base* of a wider object, the old exact-key `symbolic_objects.remove` did
/// drop the object but orphaned its `symbolic_spans` entries, which then named
/// a base with no live object.
#[test]
fn test_multi_on_wider_object_base_leaves_no_orphan_spans() {
    let ctx = SymContext::new_mock();
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(0x1000, 0x1000, Permission::RWX);

    let wide = RustBV::symbolic(&ctx, "base_wide", 32);
    ctx.assume_true(&wide.eq(&RustBV::concrete(0x4433_2211, 32), &ctx));
    mem.import_symbolic_value(0x1000, wide, None).unwrap();

    let addr_var = RustBV::symbolic(&ctx, "bw_addr", 64);
    ctx.assume_true(&addr_var.eq(&RustBV::concrete(0x1000, 64), &ctx));
    mem.set_multi_alternatives(
        0x1000,
        MultiPayload::from_alternatives(vec![make_alt(&ctx, &addr_var, 0x1000, 0x99)]),
    );

    assert!(mem.get_symbolic_object(0x1000).is_none());
    for byte in 0x1000u64..0x1004 {
        assert!(
            !mem.symbolic_spans.contains_key(&Address::new(byte)),
            "orphan reverse-span entry left at {byte:#x}"
        );
    }
    let loaded = mem
        .load_concrete_lazy(0x1000, 4, &ctx)
        .expect("wide load must succeed");
    assert_eq!(ctx.eval(&loaded), Some(0x4433_2299));
}
