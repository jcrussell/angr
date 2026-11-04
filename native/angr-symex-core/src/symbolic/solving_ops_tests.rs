//! File-local tests for `solving_ops.rs` helpers that are private to the
//! module (a sibling test file could not reach them, hence the `#[path]`
//! child-module wiring at the bottom of `solving_ops.rs`).

use super::max_val_for_width;

/// Pins the width-boundary contract of `max_val_for_width` — the shape that
/// used to be copy-pasted at seven call sites (angr-9ke6b.141). The 127/128
/// pair is the interesting one: `1u128 << 128` overflows, so 128 and above
/// must saturate rather than shift.
#[test]
fn test_max_val_for_width_boundaries() {
    assert_eq!(max_val_for_width(1), 1);
    assert_eq!(max_val_for_width(8), 0xff);
    assert_eq!(max_val_for_width(32), 0xffff_ffff);
    assert_eq!(max_val_for_width(64), u64::MAX as u128);
    assert_eq!(max_val_for_width(127), (1u128 << 127) - 1);
    assert_eq!(max_val_for_width(128), u128::MAX);
    // Widths above 128 saturate too; `min`/`max` reject them up front
    // (angr-cxw7) rather than binary-search a truncated range.
    assert_eq!(max_val_for_width(256), u128::MAX);
}

/// Every representable width is a full mask: `max_val + 1` is a power of two
/// (or wraps to 0 at the u128 ceiling), so no off-by-one can hide in a single
/// hand-checked constant above.
#[test]
fn test_max_val_for_width_is_full_mask() {
    for width in 1u32..=128 {
        let m = max_val_for_width(width);
        assert_eq!(
            m.wrapping_add(1),
            if width == 128 { 0 } else { 1u128 << width },
            "width {width}"
        );
        assert_eq!(m.count_ones(), width, "width {width}");
    }
}
