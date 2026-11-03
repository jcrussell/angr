//! Sidecar-map merge coverage for `SymbolicMemory::merge` (angr-91vj9.3).
//!
//! `RustSimState`'s `#[derive(angr_macros::MergePolicy)]` proves every
//! top-level field declares a merge treatment, but `memory` is labelled
//! `delegate` — so the guarantee stops at [`SymbolicMemory`]'s boundary, and
//! the same bug family recurred one level deeper (an other-only
//! page was adopted with its `multi_bitmap` but without the matching
//! `multi_objects` payloads; angr-c7xno.50). [`SymbolicMemory`] now carries the derive too,
//! which makes an *unlabelled* new field a compile error; this module is the
//! behavioural half, and asserts that every address-keyed sidecar actually
//! survives a merge.
//!
//! The census test [`every_address_keyed_sidecar_survives_other_only_merge`]
//! populates all of them at once on a page only `other` has. That page is
//! walked against `self`'s zero stand-in (angr-vmi2l.187), so every populated
//! byte must resolve to `other`'s value on `other`'s arm and to zero on
//! `self`'s. The single-map tests below pin down each one's failure mode.

use super::super::*;

/// A page number neither memory touches by default.
const OTHER_ONLY_PAGE: u64 = 0x9000;

/// Evaluate `v` with the merge condition `m` pinned to `arm` (1 = `other`),
/// plus any `extra` assumptions.
fn eval_on_arm(m: &RustBV, arm: u128, v: &RustBV, extra: &[RustBV]) -> Option<u128> {
    let ctx = SymContext::new();
    ctx.assume_true(&m.eq(&RustBV::concrete(arm, 1)));
    for e in extra {
        ctx.assume_true(e);
    }
    ctx.eval(v)
}

/// A wide symbolic object on a page only `other` holds: every byte resolves
/// (the reverse-span shape of angr-91vj9.3), and the whole object reads as
/// `other`'s value on `other`'s arm and as zero on `self`'s (angr-vmi2l.187).
#[test]
fn other_only_page_wide_object_is_guarded() {
    let addr = Address(OTHER_ONLY_PAGE);
    let wide = RustBV::symbol("wide", 32);

    let mut b = SymbolicMemory::new(Endness::Little);
    b.map(addr, 0x1000, Permission::RWX);
    b.store_concrete(addr, wide.clone())
        .expect("wide symbolic store");

    let mut a = SymbolicMemory::new(Endness::Little);
    let m = RustBV::symbol("m", 1);
    a.merge(&b, &m);

    let loaded = a.load_concrete(addr, 4).expect("load of the merged object");
    let pin = wide.eq(&RustBV::concrete(0x1234_5678, 32));
    assert_eq!(eval_on_arm(&m, 1, &loaded, std::slice::from_ref(&pin)), Some(0x1234_5678));
    assert_eq!(eval_on_arm(&m, 0, &loaded, &[pin]), Some(0));
}

/// The accumulating sidecars: sets that grow monotonically over a branch's life
/// and are keyed off nothing the page walk can rebuild.
#[test]
fn merge_unions_the_accumulating_sidecars() {
    let cond = RustBV::symbol("m", 1);

    let mut a = SymbolicMemory::new(Endness::Little);
    a.map(0x1000u64, 0x1000, Permission::RWX);
    a.store_concrete(Address(0x1000), RustBV::concrete(0xaa, 8))
        .expect("self write");
    a.add_lazy_region(0x20000u64, 0x1000u64);

    let mut b = SymbolicMemory::new(Endness::Little);
    b.map(0x5000u64, 0x1000, Permission::RWX);
    b.store_concrete(Address(0x5000), RustBV::concrete(0xbb, 8))
        .expect("other write");
    b.add_lazy_region(0x30000u64, 0x1000u64);
    b.import_symbolic_value(0x5008u64, RustBV::symbol("imp", 8), None)
        .expect("import");

    a.merge(&b, &cond);

    let dirty = a.get_dirty_pages();
    assert!(dirty.contains(&0x1), "self's own dirty page is kept");
    assert!(
        dirty.contains(&0x5),
        "a page only `other` dirtied must replay to the Python mirror too"
    );
    assert!(
        a.is_in_lazy_region(0x20u64) && a.is_in_lazy_region(0x30u64),
        "lazy regions union across arms"
    );
    // The record unions; the byte itself is now a merge ITE, so it no
    // longer counts as an unmodified import and is exported (angr-ccp2w).
    assert!(
        a.imported_addrs.contains_key(&Address(0x5008)),
        "imported-address provenance unions across arms"
    );
}

/// Census: one merge, every address-keyed sidecar populated only on `other`,
/// on a page `self` does not have. Every populated byte must read back as
/// `other`'s value on `other`'s arm and as zero on `self`'s.
#[test]
fn every_address_keyed_sidecar_survives_other_only_merge() {
    let base = Address(OTHER_ONLY_PAGE);
    let multi_addr = base; // multi_objects + multi_versions
    let sym_addr = base + 0x40; // symbolic_objects + symbolic_spans
    let imported_addr = base + 0x80; // imported_addrs

    let multi_cond = RustBV::symbol("census_mc", 1);
    let wide = RustBV::symbol("census_wide", 32);
    let imp = RustBV::symbol("census_imp", 16);

    let mut b = SymbolicMemory::new(Endness::Little);
    b.map(base, 0x1000, Permission::RWX);
    b.set_multi_alternatives(
        multi_addr,
        MultiPayload::from_alternatives(vec![MultiAlternative::new(
            multi_cond.clone(),
            RustBV::concrete(0x7f, 8),
        )]),
    );
    b.store_concrete(sym_addr, wide.clone())
        .expect("wide symbolic store");
    b.import_symbolic_value(imported_addr, imp.clone(), None)
        .expect("import");

    let mut a = SymbolicMemory::new(Endness::Little);
    let m = RustBV::symbol("m", 1);
    a.merge(&b, &m);

    let pins = [
        multi_cond.eq(&RustBV::concrete(1, 1)),
        wide.eq(&RustBV::concrete(0xa1b2_c3d4, 32)),
        imp.eq(&RustBV::concrete(0x5eed, 16)),
    ];
    for (addr, size, want) in [
        (multi_addr, 1, 0x7f),
        (sym_addr, 4, 0xa1b2_c3d4),
        (imported_addr, 2, 0x5eed),
    ] {
        let v = a.load_concrete(addr, size).expect("merged byte resolves");
        assert_eq!(eval_on_arm(&m, 1, &v, &pins), Some(want), "{addr:?} on other's arm");
        assert_eq!(eval_on_arm(&m, 0, &v, &pins), Some(0), "{addr:?} on self's arm");
    }
    assert!(a.imported_addrs.contains_key(&imported_addr), "imported_addrs");
    assert!(
        a.get_dirty_pages().contains(&(OTHER_ONLY_PAGE >> 12)),
        "dirty_pages"
    );
}

/// A wide (>1-byte) symbolic object that diverges inside the *both-pages-present*
/// per-byte walk must contribute its per-byte **lanes**, resolved through the
/// `symbolic_spans` reverse index, not `symbolic_objects[byte]` alone
/// (angr-0jh0j.30).
///
/// Before the fix `merge_byte_value` consulted only `symbolic_objects`, which is
/// keyed at an object's base address, so:
///   * every interior byte fell through to the page's concrete placeholder,
///     discarding the object's real content;
///   * the base byte got the whole 32-bit object, feeding a width-32 `then` and a
///     width-8 `else` into `RustBV::ite_into` — a `debug_assert` trip here, a
///     malformed-width `Ite` node in the shipped release build.
#[test]
fn wide_symbolic_divergence_merges_per_byte_lanes() {
    let addr = Address(0xa000);

    let arm = |name: &str| {
        let mut m = SymbolicMemory::new(Endness::Little);
        m.map(addr, 0x1000, Permission::RWX);
        m.store_concrete(addr, RustBV::symbol(name, 32))
            .expect("wide symbolic store");
        m
    };

    let (mut a, b) = (arm("wide_a"), arm("wide_b"));
    assert!(a.merge(&b, &RustBV::symbol("m", 1)));

    for i in 0..4u64 {
        let cell = a
            .get_symbolic_object(addr + i)
            .unwrap_or_else(|| panic!("byte {i} of the diverged wide object must merge"));
        assert_eq!(
            cell.width(),
            8,
            "byte {i} merges to a byte-wide ITE, not the whole wide object"
        );
        assert!(
            cell.is_symbolic(),
            "byte {i} keeps the arms' symbolic content instead of the page placeholder"
        );
    }
}

/// The semantic half of [`wide_symbolic_divergence_merges_per_byte_lanes`]: with
/// a *concrete* merge condition selecting `other`, every byte's ITE folds to
/// `other`'s value, so the whole wide read resolves. Before the fix bytes 1..4
/// of `self` merged against a concrete-0 placeholder and byte 0's ITE was
/// width-mismatched, so this load could not resolve to `other`'s value.
#[test]
fn wide_symbolic_divergence_resolves_under_a_concrete_merge_cond() {
    let ctx = SymContext::new();
    let addr = Address(0xa000);

    let mut a = SymbolicMemory::new(Endness::Little);
    a.map(addr, 0x1000, Permission::RWX);
    a.store_concrete(addr, RustBV::symbol("wide_a", 32))
        .expect("wide symbolic store");

    let mut b = SymbolicMemory::new(Endness::Little);
    b.map(addr, 0x1000, Permission::RWX);
    b.store_concrete(addr, RustBV::concrete(0xdead_beef, 32))
        .expect("wide concrete store");

    // `m == 1` selects `other` at every byte.
    assert!(a.merge(&b, &RustBV::concrete(1, 1)));

    let loaded = a
        .load(RustBV::concrete(addr.raw() as u128, 64), 4, &ctx)
        .expect("load of the merged wide object");
    assert_eq!(
        loaded.as_u128(),
        Some(0xdead_beef),
        "m==1 folds every lane to other's byte"
    );
}

/// angr-fs8kb.68 / angr-vmi2l.187: a wide symbolic object straddling a page
/// boundary where the **base** page is both-present and the **tail** page is
/// other-only. Both pages are walked, so every byte of the object is a
/// guarded lane: on `other`'s arm the whole object reads back, on `self`'s
/// arm the base bytes keep `self`'s write and the tail reads as zero (it is
/// on a page `self` never mapped), not as `other`'s raw lanes.
#[test]
fn split_page_object_backfills_tail_bytes_when_base_key_is_occupied() {
    // 0x8ffe..0x9002: two bytes on page 0x8, two on page 0x9.
    let base = Address(0x8ffe);
    let straddle = RustBV::symbol("straddle", 32);

    let mut b = SymbolicMemory::new(Endness::Little);
    b.map(0x8000u64, 0x2000, Permission::RWX);
    b.store_concrete(base, straddle.clone())
        .expect("wide symbolic store across the page boundary");

    // `self` has only the base page.
    let mut a = SymbolicMemory::new(Endness::Little);
    a.map(0x8000u64, 0x1000, Permission::RWX);
    a.store_concrete(base, RustBV::concrete(0xef, 8))
        .expect("self's concrete write on the base page");

    let m = RustBV::symbol("m", 1);
    a.merge(&b, &m);

    let loaded = a.load_concrete(base, 4).expect("load of the straddling object");
    let pin = [straddle.eq(&RustBV::concrete(0x1122_3344, 32))];
    assert_eq!(eval_on_arm(&m, 1, &loaded, &pin), Some(0x1122_3344));
    assert_eq!(eval_on_arm(&m, 0, &loaded, &pin), Some(0xef), "self's base write, zero tail");
    for i in 0..4u64 {
        assert_eq!(
            a.symbolic_spans.get(&(base + i)).copied(),
            Some((base + i, 8)),
            "byte {i} is an 8-bit merge lane; no stale wide span survives"
        );
    }
}
