use super::*;

#[test]
fn test_mips64_reverse_name_lookup_is_complete() {
    // Regression (angr-1yge9.13): CANONICAL_MIPS64 formerly reverse-mapped
    // only 7 of 35 GPRs, unlike MIPS32 which maps all of them. Every canonical
    // name must round-trip offset -> name for consistency across the family.
    let arch = MIPS64;
    for &name in REGISTER_NAMES_MIPS64 {
        let off = arch
            .register_offset(name)
            .unwrap_or_else(|| panic!("no offset for {name}"));
        assert_eq!(
            arch.register_name(off),
            Some(name),
            "offset {off} ({name}) did not reverse-resolve",
        );
    }

    // N64 ABI: $8-$11 canonicalize to a4-a7; the O32 t0-t3 remain aliases that
    // forward-resolve to the same offsets but never appear in reverse lookup.
    assert_eq!(arch.register_name(offsets64::R8), Some("a4"));
    assert_eq!(arch.register_offset("t0"), Some(offsets64::R8));
    // $30 canonicalizes to fp; s8 is an alias.
    assert_eq!(arch.register_name(offsets64::R30), Some("fp"));
    assert_eq!(arch.register_offset("s8"), Some(offsets64::R30));
}
