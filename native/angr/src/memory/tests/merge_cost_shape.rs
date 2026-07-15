//! S5a — memory-merge cost-shape spike (angr-op0dn.11.1.1).
//!
//! Contingent moonshot spike (parent angr-op0dn.11.1, human-GO 2026-07-15,
//! decoupled from S6). Prototype/measurement code — may be throwaway; it exists
//! to answer the M3 merge GO/NO-GO's *cost shape* question, not to ship an impl
//! (that is angr-op0dn.11.2.x).
//!
//! Question: on a diamond CFG, does `SymbolicMemory::merge` cost scale with the
//! *divergence* between the two arms, or with their *total* size?
//!
//! Answer, established here:
//!   * The production merge (`memory/mod.rs::merge`) walks EVERY shared page:
//!     it `load_concrete(0, PAGE_SIZE)`s both sides (materializing 4 KB each)
//!     and byte-compares all `PAGE_SIZE` bytes, even for pages neither arm
//!     touched since the fork. Cost ∝ total shared pages.
//!   * The primitives for a divergence-proportional merge ALREADY SHIP: pages
//!     live in an `im::OrdMap<u64, MemoryPage>` (O(1) structural-sharing clone
//!     on `fork`) and each page's `data: Arc<Vec<u8>>` is CoW via
//!     `Arc::make_mut` in `store_concrete`. So an untouched page's `data` Arc
//!     stays ptr-equal across both arms all the way to the merge point; only a
//!     page a arm actually wrote breaks sharing.
//!   * A prototype skip keyed on `MemoryPage::shares_data_with` (Arc::ptr_eq +
//!     bitmap match, added for this spike) makes the merge walk exactly the
//!     divergent pages and emit exactly the divergent-byte ITE cells.
//!
//! The tests below MEASURE both cost shapes on a synthetic diamond and record
//! the OrdMap/Arc-sharing-survives-to-reconvergence finding (which retires the
//! risk premise behind angr-op0dn.11.2.1).

use super::super::*;

const DIAMOND_PAGES: u64 = 16;
const BASE: u64 = 0x10_000; // page-aligned (page_num 0x10)
const DIVERGENT_PAGE_IDX: u64 = 3; // one arm writes here; the rest stay shared
const DIVERGENT_BYTES: usize = 8; // an 8-byte store on the divergent page

/// Build the diamond pre-header: `DIAMOND_PAGES` mapped pages, each seeded with
/// a distinct 8-byte concrete value so every page is materialized and non-zero.
fn build_base() -> SymbolicMemory {
    let mut mem = SymbolicMemory::new(Endness::Little);
    mem.map(BASE, DIAMOND_PAGES * PAGE_SIZE, Permission::RWX);
    for p in 0..DIAMOND_PAGES {
        let addr = BASE + p * PAGE_SIZE;
        // distinct per-page payload so pages are not accidentally byte-identical
        mem.store_concrete(
            addr,
            RustBV::concrete(0x1111_1111_0000_0000 + p as u128, 64),
        )
        .unwrap();
    }
    mem
}

/// Cost of the *production* merge shape: every page present in both arms is
/// walked (load_concrete ×2 + full PAGE_SIZE byte compare).
fn current_merge_cost(a: &SymbolicMemory, b: &SymbolicMemory) -> (u64, u64) {
    let mut pages_walked = 0u64;
    for (&page_num, ap) in a.pages.iter() {
        if let Some(bp) = b.pages.get(&page_num) {
            let _ = (
                ap.load_concrete(0, PAGE_SIZE as u16),
                bp.load_concrete(0, PAGE_SIZE as u16),
            );
            pages_walked += 1;
        }
    }
    (pages_walked, pages_walked * PAGE_SIZE)
}

/// Cost of the *prototype* CoW-skip merge shape: pages whose `data` Arc is still
/// ptr-shared (and whose overlay bitmaps match) are skipped with zero byte walk.
fn cow_merge_cost(a: &SymbolicMemory, b: &SymbolicMemory) -> (u64, u64) {
    let mut pages_walked = 0u64;
    for (&page_num, ap) in a.pages.iter() {
        if let Some(bp) = b.pages.get(&page_num) {
            if ap.shares_data_with(bp) {
                continue; // divergence-proportional skip
            }
            pages_walked += 1;
        }
    }
    (pages_walked, pages_walked * PAGE_SIZE)
}

/// Fork a base into two diamond arms; each writes `DIVERGENT_BYTES` to the SAME
/// single page (`DIVERGENT_PAGE_IDX`), leaving the other pages untouched/shared.
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

/// FINDING 1 — OrdMap + Arc structural sharing SURVIVES to the reconvergence
/// (merge) point. Untouched pages remain ptr-equal across both arms after a
/// fork + a divergent write; only the written page breaks sharing. This retires
/// the risk premise behind angr-op0dn.11.2.1 ("sharing already destroyed by
/// reconvergence").
#[test]
fn test_ordmap_sharing_survives_to_merge_point() {
    let (a, b) = diamond_arms();

    let mut shared = 0u64;
    let mut diverged = 0u64;
    for (&page_num, ap) in a.pages.iter() {
        let bp = b.pages.get(&page_num).expect("both arms map every page");
        if ap.shares_data_with(bp) {
            shared += 1;
        } else {
            diverged += 1;
            assert_eq!(
                page_num,
                (BASE >> 12) + DIVERGENT_PAGE_IDX,
                "only the written page may break sharing"
            );
        }
    }
    assert_eq!(diverged, 1, "exactly one page diverged");
    assert_eq!(
        shared,
        DIAMOND_PAGES - 1,
        "every untouched page stays ptr-shared across arms at merge time"
    );
}

/// FINDING 2 — cost shape. Prints a before/after table and asserts the prototype
/// merge is divergence-proportional (pages_walked == divergent-page count) while
/// the production merge is total-size-proportional (walks all shared pages).
#[test]
fn test_merge_cost_shape_divergence_proportional() {
    let (a, b) = diamond_arms();

    let (cur_pages, cur_bytes) = current_merge_cost(&a, &b);
    let (cow_pages, cow_bytes) = cow_merge_cost(&a, &b);

    eprintln!(
        "\n=== S5a memory-merge cost shape (diamond: {DIAMOND_PAGES} pages, 1 divergent) ==="
    );
    eprintln!("                    pages_walked   bytes_walked");
    eprintln!("  current (prod)  : {cur_pages:>12}   {cur_bytes:>12}");
    eprintln!("  cow-skip (proto): {cow_pages:>12}   {cow_bytes:>12}");
    eprintln!(
        "  reduction       : {:>11.1}x   {:>11.1}x\n",
        cur_pages as f64 / cow_pages.max(1) as f64,
        cur_bytes as f64 / cow_bytes.max(1) as f64
    );

    // Production shape: walks every shared page.
    assert_eq!(
        cur_pages, DIAMOND_PAGES,
        "production merge walks all shared pages"
    );
    // Prototype shape: walks only the divergent page → divergence-proportional.
    assert_eq!(cow_pages, 1, "cow-skip merge walks only the divergent page");
    assert!(cow_bytes < cur_bytes, "cow-skip walks strictly fewer bytes");
}

/// FINDING 3 — the real `merge` emits ITE cells for the divergent bytes only,
/// and the prototype skip would leave exactly those cells (nothing on the shared
/// pages changes). Confirms ITE-cell count == divergent-byte count.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn test_merge_ite_cells_equal_divergent_bytes() {
    let ctx = SymContext::new();
    let (mut a, b) = diamond_arms();

    let objs_before = a.symbolic_objects.len();
    let cond = RustBV::symbolic(&ctx, "merge_cond", 1);
    assert!(a.merge(&b, &cond, &ctx), "merge reports a change");

    let ite_cells = a.symbolic_objects.len() - objs_before;
    eprintln!(
        "\n=== S5a ITE-cell count: {ite_cells} (expected {DIVERGENT_BYTES} divergent bytes) ===\n"
    );
    assert_eq!(
        ite_cells, DIVERGENT_BYTES,
        "merge emits exactly one ITE cell per divergent byte"
    );

    // Every emitted ITE cell lands on the divergent page — the shared pages
    // contribute nothing, which is precisely what the cow-skip elides.
    let dpage = (BASE >> 12) + DIVERGENT_PAGE_IDX;
    for (&addr, _) in a.symbolic_objects.iter() {
        assert_eq!(
            addr.page_num(),
            dpage,
            "ITE cells only on the divergent page"
        );
    }
}
