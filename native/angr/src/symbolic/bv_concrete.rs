//! Z3-independent concrete bitvector folds.
//!
//! These helpers operate purely on `u128` and carry no Z3 dependency, so —
//! unlike the sibling [`bv_codec`](super::bv_codec) module — they compile under
//! every feature combo (including `--no-default-features`). Keep Z3-free folds
//! here so the concrete `RustBV` code paths that call them stay buildable
//! without the `vex-engine-z3` feature (angr-1yge9.14: the no-default-features
//! combo referenced `bv_codec::concrete_extract_u128` while that module was
//! gated behind `vex-engine-z3`).

/// Fold a concrete `Extract(high, low)` at the u128 level, returning the
/// extracted bits right-aligned.
///
/// A `Concrete` stores its value in a u128, so any bit at position `>= 128`
/// (possible when the logical width exceeds 128) is logically zero. Shifting a
/// u128 by `>= 128` is not a plain zero in Rust — it panics in debug and wraps
/// the shift amount mod 128 in release — so both the shift (`low`) and the mask
/// (`result_width`) must be guarded. This is the single source of truth for the
/// concrete-extract fold shared by `RustBV::extract_into`, `extract_no_ctx`, and
/// the Z3 emitter `emit_extract_z3_cached`; keeping one copy prevents the three
/// sites from silently diverging (they have before — see angr-ph300.36).
#[inline]
pub(super) fn concrete_extract_u128(value: u128, low: u32, result_width: u32) -> u128 {
    let shifted = if low >= 128 { 0 } else { value >> low };
    let mask = if result_width >= 128 {
        u128::MAX
    } else {
        (1u128 << result_width) - 1
    };
    shifted & mask
}

#[cfg(test)]
mod tests {
    use super::concrete_extract_u128;

    #[test]
    fn low_bits() {
        // Extract [7:0] of 0xDEAD → 0xAD
        assert_eq!(concrete_extract_u128(0xDEAD, 0, 8), 0xAD);
    }

    #[test]
    fn high_bits() {
        // Extract [15:8] of 0xDEAD → 0xDE
        assert_eq!(concrete_extract_u128(0xDEAD, 8, 8), 0xDE);
    }

    #[test]
    fn full_width_identity() {
        // result_width == 128 must mask with u128::MAX, not (1<<128)-1 (which
        // would overflow the shift).
        let v = 0x1234_5678_9abc_def0_1122_3344_5566_7788u128;
        assert_eq!(concrete_extract_u128(v, 0, 128), v);
    }

    #[test]
    fn shift_at_or_past_128_is_zero() {
        // low >= 128: every logical bit is zero (Concrete holds <=128 bits).
        assert_eq!(concrete_extract_u128(u128::MAX, 128, 8), 0);
        assert_eq!(concrete_extract_u128(u128::MAX, 200, 32), 0);
    }

    #[test]
    fn high_slice_of_full_u128() {
        // Extract [127:120] of a value whose top byte is 0x12.
        let v = 0x12FF_0000_0000_0000_0000_0000_0000_0000u128;
        assert_eq!(concrete_extract_u128(v, 120, 8), 0x12);
    }

    #[test]
    fn result_width_127_masks_correctly() {
        // result_width == 127: mask is (1<<127)-1, clears exactly the top bit.
        let v = u128::MAX;
        assert_eq!(concrete_extract_u128(v, 0, 127), (1u128 << 127) - 1);
    }
}
