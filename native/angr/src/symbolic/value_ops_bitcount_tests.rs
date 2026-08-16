//! Unit tests for the shared `Clz`/`Ctz` concrete-fold helpers.
//!
//! The regression these exist for is angr-0jh0j.9: the copy of the `Clz`
//! adjustment in `claripy_bridge::export::rustbv_to_claripy_memo` computed
//! `leading_zeros() - (128 - width)`, which underflows for `width > 128`.
//! Widths above 128 are ordinary here — a page-sized memory chunk or the
//! 1024-bit BVs the solver round-trip tests build both exceed it — and
//! `RustBV::as_u128` still answers `Some` for such a value as long as it fits
//! in a `u128`, so the fold is entered with a `width` the old spelling could
//! not represent.

use super::{concrete_clz, concrete_ctz};

#[test]
fn test_concrete_clz_narrow_widths() {
    assert_eq!(concrete_clz(0, 8), 8);
    assert_eq!(concrete_clz(1, 8), 7);
    assert_eq!(concrete_clz(0x80, 8), 0);
    assert_eq!(concrete_clz(0xff, 8), 0);
    assert_eq!(concrete_clz(1, 64), 63);
    assert_eq!(concrete_clz(u128::MAX, 128), 0);
    assert_eq!(concrete_clz(0, 128), 128);
}

#[test]
fn test_concrete_clz_width_above_128_does_not_underflow() {
    // The angr-0jh0j.9 case. Old spelling: `leading_zeros() - (128 - width)`
    // with `width = 256` underflows `128 - width` to 4294967168.
    assert_eq!(concrete_clz(1, 256), 255);
    assert_eq!(concrete_clz(0, 256), 256);
    assert_eq!(concrete_clz(1 << 127, 256), 128);
    assert_eq!(concrete_clz(u128::MAX, 1024), 896);
}

#[test]
fn test_concrete_clz_value_wider_than_width_saturates() {
    // Impossible by construction (`as_u128` only answers for a value that fits
    // the declared width), but the helper must not underflow if it happens.
    assert_eq!(concrete_clz(0xffff, 4), 0);
}

#[test]
fn test_concrete_ctz_matches_width_clamp() {
    assert_eq!(concrete_ctz(0, 8), 8);
    assert_eq!(concrete_ctz(1, 8), 0);
    assert_eq!(concrete_ctz(0x80, 8), 7);
    assert_eq!(concrete_ctz(0, 256), 256);
    assert_eq!(concrete_ctz(1 << 127, 256), 127);
}
