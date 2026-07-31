// Tests for arch/amd64.rs — extracted from the inline `mod tests` block.
// Included via `#[path = "amd64_tests.rs"] mod tests;` in amd64.rs.

use super::*;

#[test]
fn test_segment_base_offsets_match_archinfo() {
    // angr-a68t: GS_CONST belongs at offset 1032 per archinfo, not the
    // SSEROUND slot at 216. Pin both offsets and confirm the alias map
    // routes gs/gs_const to the real slot, not the sseround collision
    // that arch_prctl ARCH_SET_GS used to silently corrupt.
    let arch = AMD64;

    assert_eq!(arch.register_offset("fs_const"), Some(208));
    assert_eq!(arch.register_size("fs_const"), Some(8));
    assert_eq!(arch.register_offset("fs"), Some(208));

    assert_eq!(arch.register_offset("gs_const"), Some(1032));
    assert_eq!(arch.register_size("gs_const"), Some(8));
    assert_eq!(arch.register_offset("gs"), Some(1032));

    assert_eq!(arch.register_offset("sseround"), Some(216));
    assert_eq!(arch.register_size("sseround"), Some(8));

    // Reverse lookup: 216 owns sseround now, 1032 owns gs_const.
    assert_eq!(arch.register_name(216), Some("sseround"));
    assert_eq!(arch.register_name(1032), Some("gs_const"));
    assert_eq!(arch.register_name(208), Some("fs_const"));

    // State must be wide enough to hold the gs_const slot (offset 1032
    // + 8 bytes). Allocations narrower than this drop ARCH_SET_GS writes.
    assert!(arch.state_size() >= 1040);
}
