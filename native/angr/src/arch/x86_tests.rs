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
    // angr-5spy.2: segment-base entries (fs_const/gs_const + the
    // archinfo-real ldt/gdt) must resolve through the ALIASES
    // fallback. fs_const/gs_const are placeholders past the
    // archinfo-named slots (320/324 collide with archinfo's
    // emnote/cmstart, but no current callers read those by name).
    let arch = X86;

    assert_eq!(arch.register_offset("fs_const"), Some(320));
    assert_eq!(arch.register_size("fs_const"), Some(4));
    assert_eq!(arch.register_offset("gs_const"), Some(324));
    assert_eq!(arch.register_size("gs_const"), Some(4));

    // ldt/gdt match archinfo.ArchX86 exactly (304/312, 8B).
    assert_eq!(arch.register_offset("ldt"), Some(304));
    assert_eq!(arch.register_size("ldt"), Some(8));
    assert_eq!(arch.register_offset("gdt"), Some(312));
    assert_eq!(arch.register_size("gdt"), Some(8));
}
