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
    parse_binary_to_bytes, parse_decimal_to_bytes, parse_hex_to_bytes, parse_wide_binary_low128,
    parse_wide_hex_low128,
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
    // Z3 uses formats: #xHEXDIGITS, #bBINARY, or decimal
    if let Some(hex_str) = s.strip_prefix("#x") {
        // Parse as hex, taking low 128 bits
        parse_wide_hex_low128(hex_str)
    } else if let Some(bin_str) = s.strip_prefix("#b") {
        // Parse as binary, taking low 128 bits
        parse_wide_binary_low128(bin_str)
    } else {
        // Try decimal
        s.parse::<u128>().ok()
    }
}

/// Extract an arbitrarily large BV value as a Vec<u8> (big-endian).
/// Used for values > 128 bits where we need the full value.
pub(super) fn extract_bv_value_wide(bv: &z3::ast::BV, width: u32) -> Option<Vec<u8>> {
    let s = format!("{bv}");
    if let Some(hex_str) = s.strip_prefix("#x") {
        // Parse full hex value to bytes
        parse_hex_to_bytes(hex_str, width)
    } else if let Some(bin_str) = s.strip_prefix("#b") {
        // Parse full binary value to bytes
        parse_binary_to_bytes(bin_str, width)
    } else {
        // Decimal - parse and convert
        parse_decimal_to_bytes(&s, width)
    }
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
