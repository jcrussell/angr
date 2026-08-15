//! Multi-cell union + pending-write guard merge semantics
//! (angr-op0dn.11.2.2, M3-2b).
//!
//! `SymbolicMemory::merge` must, per the `rust_lazy_memory_design.rst` merge
//! rule, merge a byte that is Multi on *both* arms into a single lazy Multi
//! cell whose alternatives union the arms under merge-condition guards (staying
//! lazy — no eager collapse to a `symbolic_objects` ITE). It must also guard
//! deferred symbolic stores (`pending_writes`) by the merge condition instead
//! of the old blind `extend`, so an arm-specific write only materializes on the
//! path that issued it.
//!
//! These tests assert:
//!   * union alternative count == n_self + n_other (the union);
//!   * the merged byte stays Multi (not collapsed into `symbolic_objects`);
//!   * the union collapses to the correct arm value under a concrete merge
//!     condition (other under m==1, self under m==0);
//!   * pending writes are guarded — self under `!m`, other under `m`, composing
//!     with any pre-existing conditional-store condition via `And`.

use super::super::*;

const ADDR: u64 = 0x2000; // page-aligned byte 0

fn multi_cell(cond: RustBV, value: u8) -> MultiAlternative {
    MultiAlternative::new(cond, RustBV::concrete(value as u128, 8))
}

/// Both arms Multi at the same byte → union of guarded alternatives, count ==
/// n_self + n_other, and the byte stays a Multi cell (not eagerly collapsed).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_both_multi_unions_alternatives() {
    let ctx = SymContext::new();
    let addr = Address(ADDR);

    let mut a = SymbolicMemory::new(Endness::Little);
    a.set_multi_alternatives(
        addr,
        MultiPayload::from_alternatives(vec![
            multi_cell(RustBV::symbolic(&ctx, "ca1", 1), 0x01),
            multi_cell(RustBV::symbolic(&ctx, "ca2", 1), 0x02),
        ]),
    );

    let mut b = SymbolicMemory::new(Endness::Little);
    b.set_multi_alternatives(
        addr,
        MultiPayload::from_alternatives(vec![
            multi_cell(RustBV::symbolic(&ctx, "cb1", 1), 0x11),
            multi_cell(RustBV::symbolic(&ctx, "cb2", 1), 0x12),
            multi_cell(RustBV::symbolic(&ctx, "cb3", 1), 0x13),
        ]),
    );

    let cond = RustBV::symbolic(&ctx, "merge_cond", 1);
    assert!(a.merge(&b, &cond, &ctx), "merge reports a change");

    let merged = a
        .get_multi_alternatives(addr)
        .expect("byte stays a Multi cell after merge");
    assert_eq!(
        merged.len(),
        2 + 3,
        "union alternative count == n_self + n_other"
    );
    // Multi and plain-symbolic are mutually exclusive: the byte must NOT have
    // been collapsed into a symbolic_objects ITE.
    assert!(
        !a.symbolic_objects.contains_key(&addr),
        "merged Multi byte is not collapsed into symbolic_objects"
    );
    let page = a.pages.get(&addr.page_num()).expect("page mapped");
    assert!(page.is_multi(addr.page_offset()), "byte remains Multi");
}

/// Under a concrete merge condition the union collapses to the correct arm:
/// other under `m == 1`, self under `m == 0`. Deterministic — every cond is
/// concrete so `as_u128()` resolves.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_both_multi_union_collapses_to_correct_arm() {
    let ctx = SymContext::new();
    let addr = Address(ADDR);
    let one = || RustBV::concrete(1, 1);

    let build = || {
        let mut a = SymbolicMemory::new(Endness::Little);
        a.set_multi_alternatives(
            addr,
            MultiPayload::from_alternatives(vec![multi_cell(one(), 0x11)]),
        );
        let mut b = SymbolicMemory::new(Endness::Little);
        b.set_multi_alternatives(
            addr,
            MultiPayload::from_alternatives(vec![multi_cell(one(), 0x22)]),
        );
        (a, b)
    };

    // m == 1 selects other's value (0x22).
    let (mut a, b) = build();
    assert!(a.merge(&b, &RustBV::concrete(1, 1), &ctx));
    let v = a
        .get_multi_alternatives(addr)
        .unwrap()
        .collapse(0x00, &ctx)
        .as_u128();
    assert_eq!(v, Some(0x22), "m==1 collapses to other's alternative");

    // m == 0 selects self's value (0x11).
    let (mut a, b) = build();
    assert!(a.merge(&b, &RustBV::concrete(0, 1), &ctx));
    let v = a
        .get_multi_alternatives(addr)
        .unwrap()
        .collapse(0x00, &ctx)
        .as_u128();
    assert_eq!(v, Some(0x11), "m==0 collapses to self's alternative");
}

/// A byte Multi on one arm and plain-concrete on the other falls back to the
/// collapse-and-ITE path (a single `symbolic_objects` cell), not a Multi union.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_mixed_multi_and_concrete_collapses() {
    let ctx = SymContext::new();
    let addr = Address(ADDR);

    let mut a = SymbolicMemory::new(Endness::Little);
    a.set_multi_alternatives(
        addr,
        MultiPayload::from_alternatives(vec![multi_cell(RustBV::concrete(1, 1), 0x11)]),
    );

    // b maps the page but leaves the byte concrete (0x00).
    let mut b = SymbolicMemory::new(Endness::Little);
    b.map(ADDR & !0xfff, PAGE_SIZE, Permission::RW);

    assert!(a.merge(&b, &RustBV::symbolic(&ctx, "m", 1), &ctx));
    assert!(
        a.symbolic_objects.contains_key(&addr),
        "mixed Multi/concrete byte collapses to a symbolic_objects ITE"
    );
}

/// Pending writes are guarded by the merge condition, not blindly extended.
/// Deterministic: `m == 1`, so self's write (guarded `!m`) resolves to a `0`
/// condition and other's write (guarded `m`, composed with its prior `1`
/// condition via `And`) resolves to `1`.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_pending_writes_are_merge_guarded() {
    let ctx = SymContext::new();

    let mut a = SymbolicMemory::new(Endness::Little);
    a.add_pending_write(PendingWrite {
        addr: RustBV::symbolic(&ctx, "wa_addr", 64),
        value: RustBV::symbolic(&ctx, "wa_val", 8),
        size: 1,
        condition: None,
    });

    let mut b = SymbolicMemory::new(Endness::Little);
    b.add_pending_write(PendingWrite {
        addr: RustBV::symbolic(&ctx, "wb_addr", 64),
        value: RustBV::symbolic(&ctx, "wb_val", 8),
        size: 1,
        condition: Some(RustBV::concrete(1, 1)),
    });

    assert!(a.merge(&b, &RustBV::concrete(1, 1), &ctx));

    let writes = a.pending_writes();
    assert_eq!(writes.len(), 2, "both arms' pending writes are retained");

    // self's write: guarded with !m == !1 == 0, no prior condition.
    let self_cond = writes[0]
        .condition
        .as_ref()
        .expect("self write is now guarded (was None)");
    assert_eq!(
        self_cond.as_u128(),
        Some(0),
        "self write guarded off on the other arm (m==1)"
    );

    // other's write: guarded with m == 1, composed with prior 1 via And → 1.
    let other_cond = writes[1]
        .condition
        .as_ref()
        .expect("other write stays guarded");
    assert_eq!(
        other_cond.as_u128(),
        Some(1),
        "other write live on the other arm (m==1), And-composed with its prior condition"
    );
}

/// A page that exists only in `other` and carries Multi cells must bring its
/// `multi_objects` payloads along (angr-c7xno.50). The page clone copies the
/// `multi_bitmap`, but the payload table is flat on `SymbolicMemory`, so
/// without an explicit copy the adopted bit points at nothing: `range_has_multi`
/// reports false and the load silently returns the page's stale concrete byte.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_merge_adopts_multi_cells_from_other_only_page() {
    let ctx = SymContext::new();
    let addr = Address(ADDR);

    // `a` maps a *different* page, so ADDR's page exists only in `b`.
    let mut a = SymbolicMemory::new(Endness::Little);
    a.map(0x1000, PAGE_SIZE, Permission::RW);

    let mut b = SymbolicMemory::new(Endness::Little);
    b.set_multi_alternatives(
        addr,
        MultiPayload::from_alternatives(vec![multi_cell(RustBV::concrete(1, 1), 0x5a)]),
    );

    assert!(
        a.merge(&b, &RustBV::symbolic(&ctx, "m", 1), &ctx),
        "adopting a page counts as a merge"
    );

    let page = a.pages.get(&addr.page_num()).expect("page adopted");
    assert!(
        page.is_multi(addr.page_offset()),
        "adopted page keeps its Multi bit"
    );
    let payload = a
        .get_multi_alternatives(addr)
        .expect("payload copied alongside the adopted page");
    assert_eq!(payload.len(), 1, "the arm's single alternative survives");

    let v = a.load_concrete(ADDR, 1, &ctx).unwrap();
    assert_eq!(
        v.as_u64(),
        Some(0x5a),
        "load dispatches through the Multi cell, not the page's concrete default"
    );
}

/// The second half of angr-c7xno.50: after adopting an other-only page with a
/// Multi cell, a *second* merge whose third state is Multi at the same byte
/// hits the `s_multi implies a payload` expect. With the payload orphaned that
/// expect panics; with it copied the union merges normally.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_second_level_merge_over_adopted_multi_page_does_not_panic() {
    let ctx = SymContext::new();
    let addr = Address(ADDR);

    let mut a = SymbolicMemory::new(Endness::Little);
    a.map(0x1000, PAGE_SIZE, Permission::RW);

    let mut b = SymbolicMemory::new(Endness::Little);
    b.set_multi_alternatives(
        addr,
        MultiPayload::from_alternatives(vec![multi_cell(RustBV::concrete(1, 1), 0x5a)]),
    );
    assert!(a.merge(&b, &RustBV::concrete(1, 1), &ctx));

    let mut c = SymbolicMemory::new(Endness::Little);
    c.set_multi_alternatives(
        addr,
        MultiPayload::from_alternatives(vec![multi_cell(RustBV::concrete(1, 1), 0x77)]),
    );
    // Pre-fix this panics inside merge on the orphaned `self.multi_objects`.
    assert!(a.merge(&c, &RustBV::concrete(0, 1), &ctx));

    let merged = a
        .get_multi_alternatives(addr)
        .expect("byte stays a Multi cell after the second merge");
    assert_eq!(merged.len(), 2, "both arms' alternatives are unioned");
    assert_eq!(
        merged.collapse(0x00, &ctx).as_u128(),
        Some(0x5a),
        "m==0 keeps the value adopted by the first merge"
    );
}

/// Collapsing a self-side Multi cell into a plain merge ITE must also *retire*
/// the cell: drop the `multi_objects` payload and clear the page's
/// `multi_bitmap` bit (angr-0jh0j.31).
///
/// Leaving them behind made the freshly-merged value unreachable —
/// `load_concrete_common`'s `range_has_multi` check dispatches a Multi-marked
/// byte to the Multi path before `symbolic_objects` is consulted, so every later
/// load returned the stale single-arm payload, and `flush_multi_cells` wrote that
/// stale value back over the merged entry on export.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_collapsed_multi_side_is_retired_from_the_multi_sidecar() {
    let ctx = SymContext::new();
    let addr = Address(ADDR);

    let mut a = SymbolicMemory::new(Endness::Little);
    a.set_multi_alternatives(
        addr,
        MultiPayload::from_alternatives(vec![multi_cell(RustBV::concrete(1, 1), 0x11)]),
    );

    // b maps the page and writes a different concrete byte, so the byte diverges.
    let mut b = SymbolicMemory::new(Endness::Little);
    b.map(ADDR & !0xfff, PAGE_SIZE, Permission::RW);
    b.store_concrete(addr, RustBV::concrete(0x22, 8))
        .expect("other-arm concrete write");

    // `m == 1` selects other's 0x22 at this byte.
    assert!(a.merge(&b, &RustBV::concrete(1, 1), &ctx));

    assert!(
        a.get_multi_alternatives(addr).is_none(),
        "the superseded Multi payload must be dropped"
    );
    assert!(
        !a.pages
            .get(&addr.page_num())
            .expect("page present")
            .is_multi(addr.page_offset()),
        "the superseded multi_bitmap bit must be cleared"
    );
    // With the bit cleared, the load reaches the merged `symbolic_objects` cell
    // instead of being intercepted by the stale Multi path.
    let loaded = a
        .load(RustBV::concrete(ADDR as u128, 64), 1, &ctx)
        .expect("load of the merged byte");
    assert_eq!(
        loaded.as_u128(),
        Some(0x22),
        "the merged value is reachable, not shadowed by the stale Multi cell"
    );
}
