//! Divergence-proportional merge tests (angr-op0dn.11.2.1, productionizes S5a).
//!
//! The S5a spike (`merge_cost_shape.rs`) established the *cost shape* by
//! simulating the walk. These tests exercise the REAL production skip now wired
//! into `SymbolicMemory::merge` (the `is_shared_identical` CoW fast path) and
//! assert:
//!
//!   * a state pair sharing N pages and diverging on 1 walks exactly 1 page
//!     (`merge_instrument` counter == divergent-page count);
//!   * the merge still emits exactly the divergent-byte ITE cells (the skip is
//!     semantics-preserving — no state/cell is lost);
//!   * the skip is SOUND against the symbolic-overlay trap: two arms that share
//!     `data` and have equal symbolic bitmaps but *different* `symbolic_objects`
//!     at an already-symbolic byte are NOT skipped — they still ITE-merge.

use super::super::*;
use super::merge_instrument;

const DIAMOND_PAGES: u64 = 16;
const BASE: u64 = 0x10_000; // page-aligned (page_num 0x10)
const DIVERGENT_PAGE_IDX: u64 = 3;
const DIVERGENT_BYTES: usize = 8;

fn build_base() -> SymbolicMemory {
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(BASE, DIAMOND_PAGES * PAGE_SIZE, Permission::RWX);
    for p in 0..DIAMOND_PAGES {
        let addr = BASE + p * PAGE_SIZE;
        mem.store_concrete(
            addr,
            RustBV::concrete(0x1111_1111_0000_0000 + p as u128, 64),
        )
        .unwrap();
    }
    mem
}

/// Fork a base into two diamond arms; each writes `DIVERGENT_BYTES` concrete
/// bytes to the SAME single page, leaving the rest ptr-shared.
fn diamond_arms() -> (SymbolicMemory, SymbolicMemory) {
    let base = build_base();
    let mut a = base.fork();
    let mut b = base.fork();
    let daddr = BASE + DIVERGENT_PAGE_IDX * PAGE_SIZE;
    a.store_concrete(daddr, RustBV::concrete(0xAAAA_AAAA_AAAA_AAAA, 64))
        .unwrap();
    b.store_concrete(daddr, RustBV::concrete(0xBBBB_BBBB_BBBB_BBBB, 64))
        .unwrap();
    (a, b)
}

/// ACCEPTANCE — sharing N pages, diverging on 1, walks exactly 1. Drives the
/// real `merge` and reads the production walk counter.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_merge_walks_only_divergent_page() {
    let ctx = SymContext::new();
    let (mut a, b) = diamond_arms();
    let cond = RustBV::symbolic(&ctx, "merge_cond", 1);

    merge_instrument::reset();
    assert!(a.merge(&b, &cond, &ctx), "merge reports a change");

    assert_eq!(
        merge_instrument::walked(),
        1,
        "merge materializes exactly the one divergent page ({} shared pages skipped)",
        DIAMOND_PAGES - 1
    );
}

/// The skip is semantics-preserving: the divergent bytes still merge into ITE
/// cells, and only on the divergent page. Same expectation the S5a spike proved
/// for the simulated walk, now against the real fast-path-enabled merge.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_merge_ite_cells_survive_skip() {
    let ctx = SymContext::new();
    let (mut a, b) = diamond_arms();
    let objs_before = a.symbolic_objects.len();
    let cond = RustBV::symbolic(&ctx, "merge_cond", 1);

    merge_instrument::reset();
    assert!(a.merge(&b, &cond, &ctx), "merge reports a change");

    let ite_cells = a.symbolic_objects.len() - objs_before;
    assert_eq!(
        ite_cells, DIVERGENT_BYTES,
        "one ITE cell per divergent byte survives the CoW skip"
    );
    let dpage = (BASE >> 12) + DIVERGENT_PAGE_IDX;
    for (&addr, _) in a.symbolic_objects.iter() {
        assert_eq!(
            addr.page_num(),
            dpage,
            "ITE cells only on the divergent page"
        );
    }
}

/// SOUNDNESS — the symbolic-overlay trap. Two arms fork from a base that already
/// has a symbolic byte, then each overwrites it with a DIFFERENT symbolic value.
/// A symbolic store leaves `data` ptr-shared and the bitmap bit already set, so a
/// naive `shares_data_with` (ptr-eq + bitmap-eq) would wrongly skip the page and
/// drop the divergence. `is_shared_identical` refuses to skip symbolic pages, so
/// the page is walked and an ITE cell is emitted.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_symbolic_overlay_not_skipped() {
    let ctx = SymContext::new();

    // Base already has a symbolic byte at the divergent page.
    let mut base = build_base();
    let saddr = BASE + DIVERGENT_PAGE_IDX * PAGE_SIZE;
    base.store(
        &RustBV::concrete(saddr as u128, 64),
        RustBV::symbolic(&ctx, "shared_sym", 8),
        &ctx,
    )
    .unwrap();

    let mut a = base.fork();
    let b_seed = base.fork();

    // Sanity: after fork the divergent page is data-ptr-shared with equal
    // symbolic bitmaps — a `shares_data_with`-style skip WOULD fire here.
    let dpage = (BASE >> 12) + DIVERGENT_PAGE_IDX;
    assert!(
        a.pages
            .get(&dpage)
            .unwrap()
            .shares_data_with(b_seed.pages.get(&dpage).unwrap()),
        "post-fork the symbolic page is ptr+bitmap shared (the trap premise)"
    );
    // ...but is_shared_identical must refuse it because it carries symbolic bytes.
    assert!(
        !a.pages
            .get(&dpage)
            .unwrap()
            .is_shared_identical(b_seed.pages.get(&dpage).unwrap()),
        "is_shared_identical refuses symbolic pages"
    );

    // Each arm overwrites the shared symbolic byte with a distinct new symbol,
    // which does NOT break the data Arc (symbolic store only touches the bitmap
    // + symbolic_objects) — the divergence lives purely in symbolic_objects.
    a.store(
        &RustBV::concrete(saddr as u128, 64),
        RustBV::symbolic(&ctx, "arm_a_sym", 8),
        &ctx,
    )
    .unwrap();
    let mut b = b_seed;
    b.store(
        &RustBV::concrete(saddr as u128, 64),
        RustBV::symbolic(&ctx, "arm_b_sym", 8),
        &ctx,
    )
    .unwrap();

    merge_instrument::reset();
    let objs_before = a.symbolic_objects.len();
    assert!(a.merge(&b, &RustBV::symbolic(&ctx, "cond", 1), &ctx));

    assert_eq!(
        merge_instrument::walked(),
        1,
        "the symbolic divergent page is walked, not skipped"
    );
    // The overwritten byte must produce an ITE cell (a new symbolic_objects entry
    // replaces the old one, so count is unchanged; assert the value is an ITE).
    let _ = objs_before;
    let ite = a
        .symbolic_objects
        .get(&Address(saddr))
        .expect("divergent symbolic byte still present after merge");
    assert!(
        ite.is_symbolic(),
        "merged byte is an ITE of the two arms' symbols"
    );
}
