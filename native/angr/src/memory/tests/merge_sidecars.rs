//! Sidecar-map merge coverage for `SymbolicMemory::merge` (angr-91vj9.3).
//!
//! `RustSimState`'s `#[derive(angr_macros::MergePolicy)]` proves every
//! top-level field declares a merge treatment, but `memory` is labelled
//! `delegate` — so the guarantee stops at [`SymbolicMemory`]'s boundary, and
//! the same bug family recurred one level deeper (angr-c7xno.50: an other-only
//! page was adopted with its `multi_bitmap` but without the matching
//! `multi_objects` payloads). [`SymbolicMemory`] now carries the derive too,
//! which makes an *unlabelled* new field a compile error; this module is the
//! behavioural half, and asserts that every address-keyed sidecar actually
//! survives a merge.
//!
//! The census test [`every_address_keyed_sidecar_survives_other_only_merge`]
//! populates all of them at once on a page only `other` has — the shape that
//! bypasses the both-present per-byte walk entirely — so a future sidecar
//! added to `merge` incorrectly fails here even if it compiles. The
//! single-map tests below pin down each one's specific failure mode.

use super::super::*;

/// A page number neither memory touches by default.
const OTHER_ONLY_PAGE: u64 = 0x9000;

fn multi_payload(ctx: &SymContext, name: &str, value: u8) -> MultiPayload {
    MultiPayload::from_alternatives(vec![MultiAlternative::new(
        RustBV::symbolic(ctx, name, 1),
        RustBV::concrete(value as u128, 8),
    )])
}

/// angr-c7xno.50 regression: a Multi cell living on a page only `other` has.
/// The page clone carries the `multi_bitmap` bit; the payload lives in the flat
/// `multi_objects` side table and must be copied explicitly.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn other_only_page_adopts_multi_objects_payload() {
    let ctx = SymContext::new();
    let addr = Address(OTHER_ONLY_PAGE);

    let mut a = SymbolicMemory::new(Endness::Little);
    let mut b = SymbolicMemory::new(Endness::Little);
    b.set_multi_alternatives(addr, multi_payload(&ctx, "mo", 0x41));

    let cond = RustBV::symbolic(&ctx, "m", 1);
    a.merge(&b, &cond, &ctx);

    assert!(
        a.get_multi_alternatives(addr).is_some(),
        "adopted Multi bit must arrive with its multi_objects payload"
    );
    // A second merge is where the missing payload used to surface as a panic
    // on the `s_multi implies a payload` expect.
    let mut c = SymbolicMemory::new(Endness::Little);
    c.set_multi_alternatives(addr, multi_payload(&ctx, "mo2", 0x42));
    a.merge(&c, &cond, &ctx);
    assert_eq!(
        a.get_multi_alternatives(addr)
            .expect("still Multi after the second merge")
            .len(),
        2,
        "second merge unions both arms' alternatives"
    );
}

/// `multi_versions` is the fingerprint sidecar the Phase 4.1 wider-load cache
/// keys on. An adopted Multi cell must bump it, or a cached pre-merge load of
/// that range would still look fresh.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn other_only_page_adopts_multi_versions_entry() {
    let ctx = SymContext::new();
    let addr = Address(OTHER_ONLY_PAGE);

    let mut a = SymbolicMemory::new(Endness::Little);
    assert!(!a.multi_versions.contains_key(&addr));

    let mut b = SymbolicMemory::new(Endness::Little);
    b.set_multi_alternatives(addr, multi_payload(&ctx, "mv", 0x43));

    a.merge(&b, &RustBV::symbolic(&ctx, "m", 1), &ctx);
    assert!(
        a.multi_versions.contains_key(&addr),
        "adopting a Multi cell must bump the byte's version"
    );
}

/// `symbolic_spans` is the reverse index: a wider-than-one-byte symbolic object
/// owns an entry per covered byte. Adopting the object while indexing only its
/// base left the interior bytes symbolic in the page bitmap but unresolvable
/// through either sidecar (angr-91vj9.3).
#[cfg(feature = "vex-engine-z3")]
#[test]
fn other_only_page_adopts_full_symbolic_span_range() {
    let ctx = SymContext::new();
    let addr = Address(OTHER_ONLY_PAGE);

    let mut b = SymbolicMemory::new(Endness::Little);
    b.map(addr, 0x1000, Permission::RWX);
    b.store_concrete(addr, RustBV::symbolic(&ctx, "wide", 32))
        .expect("wide symbolic store");

    let mut a = SymbolicMemory::new(Endness::Little);
    a.merge(&b, &RustBV::symbolic(&ctx, "m", 1), &ctx);

    assert!(
        a.get_symbolic_object(addr).is_some(),
        "base object is adopted"
    );
    for i in 1..4u64 {
        let byte = addr + i;
        assert_eq!(
            a.symbolic_spans.get(&byte).copied(),
            Some((addr, 32)),
            "interior byte {i} of the adopted object must be span-indexed"
        );
    }
    // The whole object must read back symbolically rather than falling through
    // to the page's concrete placeholder bytes.
    let loaded = a
        .load(RustBV::concrete(addr.raw() as u128, 64), 4, &ctx)
        .expect("load of the adopted object");
    assert!(loaded.is_symbolic(), "adopted object stays symbolic");
}

/// The accumulating sidecars: sets that grow monotonically over a branch's life
/// and are keyed off nothing the page walk can rebuild.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn merge_unions_the_accumulating_sidecars() {
    let ctx = SymContext::new();
    let cond = RustBV::symbolic(&ctx, "m", 1);

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
    b.import_symbolic_value(0x5008u64, RustBV::symbolic(&ctx, "imp", 8), None)
        .expect("import");

    a.merge(&b, &cond, &ctx);

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
    assert!(
        a.is_imported_addr(0x5008u64),
        "imported-address provenance unions across arms"
    );
}

/// Census: one merge, every address-keyed sidecar populated only on `other`,
/// on a page `self` does not have. Any sidecar the merge forgets shows up here.
#[cfg(feature = "vex-engine-z3")]
#[test]
fn every_address_keyed_sidecar_survives_other_only_merge() {
    let ctx = SymContext::new();
    let base = Address(OTHER_ONLY_PAGE);
    let multi_addr = base; // multi_objects + multi_versions
    let sym_addr = base + 0x40; // symbolic_objects + symbolic_spans
    let imported_addr = base + 0x80; // imported_addrs

    let mut b = SymbolicMemory::new(Endness::Little);
    b.map(base, 0x1000, Permission::RWX);
    b.set_multi_alternatives(multi_addr, multi_payload(&ctx, "census", 0x7f));
    b.store_concrete(sym_addr, RustBV::symbolic(&ctx, "census_wide", 32))
        .expect("wide symbolic store");
    b.import_symbolic_value(imported_addr, RustBV::symbolic(&ctx, "census_imp", 16), None)
        .expect("import");

    let mut a = SymbolicMemory::new(Endness::Little);
    a.merge(&b, &RustBV::symbolic(&ctx, "m", 1), &ctx);

    assert!(
        a.get_multi_alternatives(multi_addr).is_some(),
        "multi_objects"
    );
    assert!(a.multi_versions.contains_key(&multi_addr), "multi_versions");
    assert!(a.get_symbolic_object(sym_addr).is_some(), "symbolic_objects");
    assert_eq!(
        a.symbolic_spans.get(&(sym_addr + 3)).copied(),
        Some((sym_addr, 32)),
        "symbolic_spans covers every byte of the adopted object"
    );
    assert!(
        a.get_symbolic_object(imported_addr).is_some(),
        "imported object value"
    );
    assert!(a.is_imported_addr(imported_addr), "imported_addrs");
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
#[cfg(feature = "vex-engine-z3")]
#[test]
fn wide_symbolic_divergence_merges_per_byte_lanes() {
    let ctx = SymContext::new();
    let addr = Address(0xa000);

    let arm = |name: &str| {
        let mut m = SymbolicMemory::new(Endness::Little);
        m.map(addr, 0x1000, Permission::RWX);
        m.store_concrete(addr, RustBV::symbolic(&ctx, name, 32))
            .expect("wide symbolic store");
        m
    };

    let (mut a, b) = (arm("wide_a"), arm("wide_b"));
    assert!(a.merge(&b, &RustBV::symbolic(&ctx, "m", 1), &ctx));

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
#[cfg(feature = "vex-engine-z3")]
#[test]
fn wide_symbolic_divergence_resolves_under_a_concrete_merge_cond() {
    let ctx = SymContext::new();
    let addr = Address(0xa000);

    let mut a = SymbolicMemory::new(Endness::Little);
    a.map(addr, 0x1000, Permission::RWX);
    a.store_concrete(addr, RustBV::symbolic(&ctx, "wide_a", 32))
        .expect("wide symbolic store");

    let mut b = SymbolicMemory::new(Endness::Little);
    b.map(addr, 0x1000, Permission::RWX);
    b.store_concrete(addr, RustBV::concrete(0xdead_beef, 32))
        .expect("wide concrete store");

    // `m == 1` selects `other` at every byte.
    assert!(a.merge(&b, &RustBV::concrete(1, 1), &ctx));

    let loaded = a
        .load(RustBV::concrete(addr.raw() as u128, 64), 4, &ctx)
        .expect("load of the merged wide object");
    assert_eq!(
        loaded.as_u128(),
        Some(0xdead_beef),
        "m==1 folds every lane to other's byte"
    );
}
