//! Codec between concrete values and Z3 BV constants.
//!
//! Extracted from `context.rs` (angr-a2br.2 slice 5) — leaf associated
//! functions on `SymContext` that took no `&self` and touched no context
//! state. Two directions:
//!
//! - **decode** Z3 model results into concrete values:
//!   [`extract_bv_value`] (low 128 bits), [`extract_bv_value_from_string`]
//!   (the slow path that parses Z3's `#x` / `#b` / decimal numeral string via
//!   the [`super::parse`] helpers), and [`extract_bv_value_wide`] (full-width
//!   big-endian bytes for values > 128 bits).
//! - **encode** concrete values into Z3 BV constants: [`make_bv_const`]
//!   (from a `u128`) and [`make_bv_from_bytes`] (from big-endian bytes,
//!   chunked + concatenated for widths > 64).
//!
//! All callers live in `context.rs` (the eval / eval_wide / eval_upto /
//! min / max solving paths). Crate-visible surface is `pub(super)`.

use super::parse::{
    Z3Numeral, parse_binary_to_bytes, parse_decimal_to_bytes, parse_hex_to_bytes,
    parse_wide_binary_low128, parse_wide_hex_low128,
};

/// Extract a u128 value from a Z3 BV result.
/// For values > 128 bits, returns the low 128 bits (caller should use
/// [`extract_bv_value_wide`] for arbitrarily large values).
pub(super) fn extract_bv_value(bv: &z3::ast::BV) -> Option<u128> {
    // Try as u64 first (fast path for <= 64-bit)
    if let Some(v) = bv.as_u64() {
        return Some(v as u128);
    }
    // For larger values, parse the string representation
    extract_bv_value_from_string(bv)
}

/// Extract a BV value by parsing its string representation.
/// Handles arbitrarily large values, returns low 128 bits.
pub(super) fn extract_bv_value_from_string(bv: &z3::ast::BV) -> Option<u128> {
    let s = format!("{bv}");
    // Z3 uses formats: #xHEXDIGITS, #bBINARY, or decimal (see Z3Numeral).
    match Z3Numeral::classify(&s) {
        Z3Numeral::Hex(hex_str) => parse_wide_hex_low128(hex_str),
        Z3Numeral::Bin(bin_str) => parse_wide_binary_low128(bin_str),
        Z3Numeral::Dec(dec_str) => dec_str.parse::<u128>().ok(),
    }
}

/// Extract an arbitrarily large BV value as a Vec<u8> (big-endian).
/// Used for values > 128 bits where we need the full value.
pub(super) fn extract_bv_value_wide(bv: &z3::ast::BV, width: u32) -> Option<Vec<u8>> {
    let s = format!("{bv}");
    match Z3Numeral::classify(&s) {
        Z3Numeral::Hex(hex_str) => parse_hex_to_bytes(hex_str, width),
        Z3Numeral::Bin(bin_str) => parse_binary_to_bytes(bin_str, width),
        Z3Numeral::Dec(dec_str) => parse_decimal_to_bytes(dec_str, width),
    }
}

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

/// Create a Z3 BV constant from a u128 value.
pub(super) fn make_bv_const(value: u128, width: u32) -> z3::ast::BV {
    if width <= 64 {
        z3::ast::BV::from_u64(value as u64, width)
    } else {
        let lo = z3::ast::BV::from_u64(value as u64, 64);
        let hi = z3::ast::BV::from_u64((value >> 64) as u64, width - 64);
        hi.concat(&lo)
    }
}

/// Create a Z3 BV constant from big-endian bytes.
/// Handles arbitrary widths by building 64-bit chunks and concatenating.
pub(super) fn make_bv_from_bytes(bytes: &[u8], width: u32) -> z3::ast::BV {
    if width <= 64 {
        let mut val: u64 = 0;
        for &b in bytes {
            val = (val << 8) | (b as u64);
        }
        return z3::ast::BV::from_u64(val, width);
    }

    // Build from 64-bit chunks (big-endian)
    let byte_len = bytes.len();
    let mut result: Option<z3::ast::BV> = None;
    let mut bits_remaining = width;
    let mut pos = 0;

    while bits_remaining > 0 {
        let chunk_bits = std::cmp::min(bits_remaining, 64);
        let chunk_bytes = chunk_bits.div_ceil(8) as usize;
        let mut val: u64 = 0;
        for i in 0..chunk_bytes {
            if pos + i < byte_len {
                val = (val << 8) | (bytes[pos + i] as u64);
            } else {
                val <<= 8;
            }
        }
        let chunk = z3::ast::BV::from_u64(val, chunk_bits);
        result = Some(match result {
            Some(prev) => prev.concat(&chunk),
            None => chunk,
        });
        pos += chunk_bytes;
        bits_remaining -= chunk_bits;
    }

    result.unwrap_or_else(|| z3::ast::BV::from_u64(0, width))
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
