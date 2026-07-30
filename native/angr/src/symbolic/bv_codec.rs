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

    // The byte array is a right-aligned big-endian value of `width` bits, so
    // when width % 8 != 0 the top byte is ragged (its high 8-(width%8) bits are
    // zero padding). Chunk from the LSB in aligned 64-bit groups; the leading
    // bytes form a single ragged top chunk of `width % 64` bits. Splitting from
    // the FRONT instead (the previous implementation) misaligned every chunk by
    // 8-(width%8) bits for non-byte-multiple widths, silently corrupting the
    // constant — the wide eval_upto exclusion then encoded the wrong value
    // (angr-ph300.34).
    let num_full = (width / 64) as usize; // trailing full 64-bit chunks
    let top_bits = width % 64; // ragged top chunk width (0 when 64 | width)
    let top_bytes = top_bits.div_ceil(8) as usize;

    // Read `count` big-endian bytes starting at `start`, zero-filling any
    // out-of-range index (defensive: the sole caller sizes `bytes` exactly).
    let read = |start: usize, count: usize| -> u64 {
        let mut val: u64 = 0;
        for i in 0..count {
            val = (val << 8) | (bytes.get(start + i).copied().unwrap_or(0) as u64);
        }
        val
    };

    let mut result: Option<z3::ast::BV> = None;
    let mut pos = 0usize;

    if top_bits > 0 {
        // from_u64 keeps the low `top_bits` bits, dropping the top byte's padding.
        result = Some(z3::ast::BV::from_u64(read(pos, top_bytes), top_bits));
        pos += top_bytes;
    }
    for _ in 0..num_full {
        let chunk = z3::ast::BV::from_u64(read(pos, 8), 64);
        result = Some(match result {
            Some(prev) => prev.concat(&chunk),
            None => chunk,
        });
        pos += 8;
    }

    // width > 64 guarantees num_full >= 1, so `result` is always Some here.
    result.unwrap_or_else(|| z3::ast::BV::from_u64(0, width))
}

#[cfg(test)]
mod tests {
    use super::{extract_bv_value_wide, make_bv_const, make_bv_from_bytes};

    /// Canonical numeral string of a concrete BV (z3 prints #x.../#b.../decimal).
    fn numeral(bv: &z3::ast::BV) -> String {
        use z3::ast::Ast;
        format!("{}", bv.simplify())
    }

    /// Round-trip a `<=128`-bit value at a ragged width: build the reference
    /// constant via make_bv_const, extract its bytes, rebuild via
    /// make_bv_from_bytes, and require identical Z3 numerals. Before the
    /// angr-ph300.34 fix the rebuilt constant was corrupted for width % 8 != 0.
    fn assert_roundtrip_u128(value: u128, width: u32) {
        let masked = if width >= 128 {
            value
        } else {
            value & ((1u128 << width) - 1)
        };
        use z3::ast::Ast;
        // extract_bv_value_wide parses a concrete numeral literal (as produced
        // by model eval); simplify the concat expression down to one first.
        let reference = make_bv_const(masked, width).simplify();
        let bytes = extract_bv_value_wide(&reference, width).expect("extract bytes");
        let rebuilt = make_bv_from_bytes(&bytes, width);
        assert_eq!(
            numeral(&rebuilt),
            numeral(&reference),
            "round-trip mismatch at width {width} (value {masked:#x})"
        );
    }

    #[test]
    fn make_bv_from_bytes_ragged_widths_roundtrip() {
        // width % 8 != 0 — the buggy path. Use values whose low/high bytes and
        // sub-byte top bits are all set so any misalignment shifts the numeral.
        for &width in &[65u32, 72, 96, 125] {
            assert_roundtrip_u128(0x1234_5678_9abc_def0_1122_3344_5566_7788, width);
            assert_roundtrip_u128(u128::MAX, width);
            assert_roundtrip_u128(1, width);
        }
    }

    #[test]
    fn make_bv_from_bytes_aligned_widths_roundtrip() {
        // Multiples of 8 and/or 64 — already-correct paths, guard no regression.
        for &width in &[72u32, 96, 128] {
            assert_roundtrip_u128(0x0fed_cba9_8765_4321_1020_3040_5060_7080, width);
        }
    }

    #[test]
    fn make_bv_from_bytes_129_bit_roundtrip() {
        // 129-bit value: Concat(1-bit top, 128-bit body) — the exact shape the
        // wide eval_upto exclusion builds. Round-trip through the byte codec.
        let body = make_bv_const(0xdead_beef_cafe_babe_0102_0304_0506_0708, 128);
        let top = z3::ast::BV::from_u64(1, 1);
        use z3::ast::Ast;
        let wide = top.concat(&body).simplify(); // 129 bits, top bit set
        let bytes = extract_bv_value_wide(&wide, 129).expect("extract 129-bit bytes");
        let rebuilt = make_bv_from_bytes(&bytes, 129);
        assert_eq!(
            numeral(&rebuilt),
            numeral(&wide),
            "129-bit round-trip mismatch"
        );
    }
}
