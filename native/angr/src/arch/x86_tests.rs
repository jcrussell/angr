use super::*;

#[test]
fn test_x86_basics() {
    let arch = X86;

    assert_eq!(arch.bits(), 32);
    assert_eq!(arch.name(), "X86");
    assert!(arch.is_little_endian());
}

#[test]
fn test_register_lookup() {
    let arch = X86;

    assert_eq!(arch.register_offset("eax"), Some(8));
    assert_eq!(arch.register_size("eax"), Some(4));

    assert_eq!(arch.register_offset("ax"), Some(8));
    assert_eq!(arch.register_size("ax"), Some(2));

    assert_eq!(arch.register_offset("al"), Some(8));
    assert_eq!(arch.register_size("al"), Some(1));
}

#[test]
fn test_special_registers() {
    let arch = X86;

    assert_eq!(arch.ip_offset(), 68);
    assert_eq!(arch.sp_offset(), 24);
    assert_eq!(arch.bp_offset(), Some(28));
}

#[test]
fn test_segment_base_aliases() {
    // angr-5spy.2: segment-base entries (ldt/gdt) must resolve through the
    // ALIASES fallback. angr-rfxc7: x86 has NO fs_const/gs_const — that pair
    // is amd64-only (arch_prctl is an amd64 syscall). 320/324 are VEX's
    // guest_EMNOTE/guest_CMSTART and are now named as such, so a name-based
    // write can no longer clobber them under a segment-base alias.
    let arch = X86;

    assert_eq!(arch.register_offset("fs_const"), None);
    assert_eq!(arch.register_offset("gs_const"), None);
    assert_eq!(arch.register_offset("emnote"), Some(320));
    assert_eq!(arch.register_size("emnote"), Some(4));
    assert_eq!(arch.register_offset("cmstart"), Some(324));
    assert_eq!(arch.register_offset("ip_at_syscall"), Some(340));

    // ldt/gdt match archinfo.ArchX86 exactly (304/312, 8B).
    assert_eq!(arch.register_offset("ldt"), Some(304));
    assert_eq!(arch.register_size("ldt"), Some(8));
    assert_eq!(arch.register_offset("gdt"), Some(312));
    assert_eq!(arch.register_size("gdt"), Some(8));
}

#[test]
fn test_sp_bp_aliases_are_full_width() {
    // angr-6qzik: see the amd64 sibling. X86 'sp' == (24, 4) per archinfo.
    let arch = X86;

    assert_eq!(arch.register_offset("sp"), Some(24));
    assert_eq!(arch.register_size("sp"), Some(4));
    assert_eq!(arch.register_offset("bp"), Some(28));
    assert_eq!(arch.register_size("bp"), Some(4));

    assert_eq!(arch.register_size("ax"), Some(2));
    assert_eq!(arch.register_size("di"), Some(2));
}
