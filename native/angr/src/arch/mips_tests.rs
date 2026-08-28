//! Unit tests for [`super`] (arch/mips.rs).
//! Split out of mips.rs per `rust-mod-tests-sibling-extraction` (cfg(test)-only reorg).

use super::*;

/// Every name a MIPS variant advertises must round-trip offset -> name, so a
/// future edit to a CANONICAL_* table that drops an entry (leaving the name
/// forward-resolvable via ALIASES_* but invisible to `register_name`) fails
/// loudly. Shared by the MIPS32 and MIPS64 tests below — the two tables are
/// hand-curated separately but owe the same guarantee.
fn assert_reverse_name_lookup_is_complete(arch: &dyn Arch, names: &[&'static str]) {
    let arch_name = arch.name();
    for &name in names {
        let off = arch
            .register_offset(name)
            .unwrap_or_else(|| panic!("{arch_name}: no offset for {name}"));
        assert_eq!(
            arch.register_name(off),
            Some(name),
            "{arch_name}: offset {off} ({name}) did not reverse-resolve",
        );
    }
}

#[test]
fn test_mips64_reverse_name_lookup_is_complete() {
    // Regression (angr-1yge9.13): CANONICAL_MIPS64 formerly reverse-mapped
    // only 7 of 35 GPRs, unlike MIPS32 which maps all of them. Every canonical
    // name must round-trip offset -> name for consistency across the family.
    let arch = MIPS64;
    assert_reverse_name_lookup_is_complete(&arch, REGISTER_NAMES_MIPS64);

    // N64 ABI: $8-$11 canonicalize to a4-a7; the O32 t0-t3 remain aliases that
    // forward-resolve to the same offsets but never appear in reverse lookup.
    assert_eq!(arch.register_name(offsets64::R8), Some("a4"));
    assert_eq!(arch.register_offset("t0"), Some(offsets64::R8));
    // $30 canonicalizes to fp; s8 is an alias.
    assert_eq!(arch.register_name(offsets64::R30), Some("fp"));
    assert_eq!(arch.register_offset("s8"), Some(offsets64::R30));
}

#[test]
fn test_mips32_reverse_name_lookup_is_complete() {
    // MIPS32's offset table is the largest hand-curated one in this directory,
    // so it is the likeliest place for the angr-1yge9.13 class of bug (a
    // canonical entry deleted, leaving the name only in ALIASES_MIPS32 and
    // therefore reverse-unresolvable) to reappear. MIPS64 has pinned this
    // since that fix; MIPS32 had nothing (angr-9ke6b.14).
    let arch = MIPS32;
    assert_reverse_name_lookup_is_complete(&arch, REGISTER_NAMES_MIPS32);

    // O32 ABI: unlike N64, $8-$11 canonicalize to the temporaries t0-t3.
    assert_eq!(arch.register_name(offsets32::R8), Some("t0"));
    assert_eq!(arch.register_name(offsets32::R11), Some("t3"));
    // $30 canonicalizes to fp; s8 is an alias, matching MIPS64.
    assert_eq!(arch.register_name(offsets32::R30), Some("fp"));
    assert_eq!(arch.register_offset("s8"), Some(offsets32::R30));
}
