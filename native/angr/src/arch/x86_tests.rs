use super::*;

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
