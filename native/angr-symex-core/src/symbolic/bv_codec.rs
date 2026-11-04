//! Codec between concrete values and Z3 BV constants.
//!
//! Extracted from `context.rs` (angr-a2br.2 slice 5) — leaf associated
//! functions on `SymContext` that took no `&self` and touched no context
//! state. Two directions:
//!
//! - **decode** Z3 model results into concrete values:
//!   [`extract_bv_value`] (low 128 bits) and [`extract_bv_value_wide`]
//!   (full-width big-endian bytes). Both read the numeral string Z3's
//!   `Display` prints (`#x<hex>`, `#b<binary>` or a bare decimal) through
//!   [`parse_numeral`], which hands the digits to `num_bigint::BigUint`.
//! - **encode** concrete values into Z3 BV constants: [`make_bv_const`]
//!   (from a `u128`) and [`make_bv_from_bytes`] (from big-endian bytes).
//!
//! All callers live in `solving_ops.rs` (the eval / eval_wide / eval_upto /
//! min / max solving paths). Crate-visible surface is `pub(super)`.

use num_bigint::BigUint;

/// Split a Z3 BV numeral string into its digit payload and radix: a `#x` /
/// `#b` prefix selects hex / binary and is stripped; anything else is read as
/// a decimal numeral verbatim.
fn numeral_digits(s: &str) -> (&str, u32) {
    if let Some(hex) = s.strip_prefix("#x") {
        (hex, 16)
    } else if let Some(bin) = s.strip_prefix("#b") {
        (bin, 2)
    } else {
        (s, 10)
    }
}

/// Parse a Z3 BV numeral string of any width. `None` when the string is not a
/// numeral (a symbolic expression's `Display`, or malformed digits).
fn parse_numeral(s: &str) -> Option<BigUint> {
    let (digits, radix) = numeral_digits(s);
    BigUint::parse_bytes(digits.as_bytes(), radix)
}

/// Low 128 bits of a Z3 BV numeral string. A payload that fits `u128` skips
/// the `BigUint` allocation; a wider one is reduced modulo 2^128.
fn parse_numeral_low128(s: &str) -> Option<u128> {
    let (digits, radix) = numeral_digits(s);
    if let Ok(v) = u128::from_str_radix(digits, radix) {
        return Some(v);
    }
    let wide = BigUint::parse_bytes(digits.as_bytes(), radix)?;
    let mut limbs = wide.iter_u64_digits();
    let lo = limbs.next().unwrap_or(0);
    let hi = limbs.next().unwrap_or(0);
    Some((u128::from(hi) << 64) | u128::from(lo))
}

/// `value` as exactly `width.div_ceil(8)` big-endian bytes: right-aligned and
/// zero-padded when narrower, keeping the low bytes when wider.
fn biguint_to_be_bytes(value: &BigUint, width: u32) -> Vec<u8> {
    let byte_len = width.div_ceil(8) as usize;
    let bytes = value.to_bytes_be();
    if bytes.len() >= byte_len {
        // overflow-ok: guarded by this branch's `bytes.len() >= byte_len`.
        bytes[bytes.len() - byte_len..].to_vec()
    } else {
        let mut result = vec![0u8; byte_len];
        // overflow-ok: guarded by the `else`, i.e. `bytes.len() < byte_len`.
        result[byte_len - bytes.len()..].copy_from_slice(&bytes);
        result
    }
}

/// Extract a u128 value from a Z3 BV result.
/// For values > 128 bits, returns the low 128 bits (caller should use
/// [`extract_bv_value_wide`] for arbitrarily large values).
pub(super) fn extract_bv_value(bv: &z3::ast::BV) -> Option<u128> {
    // Try as u64 first (fast path for <= 64-bit)
    if let Some(v) = bv.as_u64() {
        return Some(v as u128);
    }
    parse_numeral_low128(&format!("{bv}"))
}

/// Extract an arbitrarily large BV value as `width.div_ceil(8)` big-endian
/// bytes. Used for values > 128 bits where we need the full value.
pub(super) fn extract_bv_value_wide(bv: &z3::ast::BV, width: u32) -> Option<Vec<u8>> {
    let value = parse_numeral(&format!("{bv}"))?;
    Some(biguint_to_be_bytes(&value, width))
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

/// Create a Z3 BV constant from a right-aligned big-endian byte array of
/// `width` bits (the shape [`extract_bv_value_wide`] returns). Widths up to
/// 128 bits fold into a `u128` and go through [`make_bv_const`]; wider ones
/// pass the value to Z3 as a decimal numeral.
pub(super) fn make_bv_from_bytes(bytes: &[u8], width: u32) -> z3::ast::BV {
    if width <= 128 {
        let value = bytes
            .iter()
            .fold(0u128, |acc, &b| (acc << 8) | u128::from(b));
        return make_bv_const(value, width);
    }
    let decimal = BigUint::from_bytes_be(bytes).to_str_radix(10);
    // SILENT(cat-a): Z3_mk_numeral only rejects a string that is not a
    // numeral; a decimal rendering of a `BigUint` always is, so the zero
    // fallback is unreachable.
    z3::ast::BV::from_str(width, &decimal).unwrap_or_else(|| z3::ast::BV::from_u64(0, width))
}

#[cfg(test)]
mod tests {
    use super::{
        BigUint, biguint_to_be_bytes, extract_bv_value, extract_bv_value_wide, make_bv_const,
        make_bv_from_bytes, parse_numeral, parse_numeral_low128,
    };

    /// 2^128 + 1 in decimal: one past the widest value `u128` can parse.
    const ABOVE_2_128_DEC: &str = "340282366920938463463374607431768211457";

    #[test]
    fn parse_numeral_dispatches_on_prefix() {
        assert_eq!(
            parse_numeral("#xdeadbeef"),
            Some(BigUint::from(0xdead_beefu32))
        );
        assert_eq!(parse_numeral("#b1010"), Some(BigUint::from(0b1010u32)));
        assert_eq!(parse_numeral("12345"), Some(BigUint::from(12345u32)));
    }

    #[test]
    fn parse_numeral_rejects_non_numerals() {
        // A symbolic expression's Display, an empty payload, bad digits.
        assert_eq!(parse_numeral("(bvadd x y)"), None);
        assert_eq!(parse_numeral(""), None);
        assert_eq!(parse_numeral("#x"), None);
        assert_eq!(parse_numeral("#xxyz"), None);
        assert_eq!(parse_numeral("#b102"), None);
        assert_eq!(parse_numeral("#x1€2"), None);
    }

    #[test]
    fn parse_numeral_low128_takes_low_bits_of_wide_values() {
        assert_eq!(parse_numeral_low128("#xff"), Some(0xff));
        assert_eq!(
            parse_numeral_low128(&format!("#x{}", "f".repeat(32))),
            Some(u128::MAX)
        );
        // 33 hex digits / 129 binary digits: the top digit is dropped.
        assert_eq!(
            parse_numeral_low128(&format!("#x1{}", "f".repeat(32))),
            Some(u128::MAX)
        );
        assert_eq!(
            parse_numeral_low128(&format!("#b1{}", "1".repeat(128))),
            Some(u128::MAX)
        );
        // 2^128 + 1 decimal reduces to 1 (angr-mx00p.40).
        assert_eq!(parse_numeral_low128(ABOVE_2_128_DEC), Some(1));
        assert_eq!(parse_numeral_low128("#xxyz"), None);
    }

    #[test]
    fn biguint_to_be_bytes_right_aligns_and_truncates() {
        assert_eq!(
            biguint_to_be_bytes(&BigUint::from(0x1234u32), 16),
            vec![0x12, 0x34]
        );
        assert_eq!(
            biguint_to_be_bytes(&BigUint::from(0xffu32), 16),
            vec![0x00, 0xff]
        );
        assert_eq!(biguint_to_be_bytes(&BigUint::from(0u32), 24), vec![0, 0, 0]);
        // Ragged width rounds up to whole bytes.
        assert_eq!(
            biguint_to_be_bytes(&BigUint::from(0x1abu32), 12),
            vec![0x01, 0xab]
        );
        // Wider than the field keeps the low bytes (truncation to width).
        assert_eq!(biguint_to_be_bytes(&BigUint::from(300u32), 8), vec![44]);
        let out = biguint_to_be_bytes(&BigUint::from(300u32), 200);
        assert_eq!(out.len(), 25);
        assert_eq!(&out[23..], &[1, 44]);
        assert!(out[..23].iter().all(|&b| b == 0));
    }

    /// angr-mx00p.40: a decimal numeral above 2^128 decodes to its full bytes
    /// instead of `None`.
    #[test]
    fn decimal_numeral_above_2_128_decodes_wide() {
        let value = parse_numeral(ABOVE_2_128_DEC).expect("decimal parses");
        let bytes = biguint_to_be_bytes(&value, 256);
        let mut want = vec![0u8; 32];
        want[15] = 1; // 2^128
        want[31] = 1; // + 1
        assert_eq!(bytes, want);
    }

    /// End-to-end through Z3: a 256-bit constant above 2^128 built from bytes
    /// extracts back to the same bytes and to its low 128 bits.
    #[test]
    fn wide_constant_above_2_128_roundtrips_through_z3() {
        use z3::ast::Ast;
        let mut bytes = vec![0u8; 32];
        bytes[0] = 0x80;
        bytes[15] = 1;
        bytes[31] = 7;
        let bv = make_bv_from_bytes(&bytes, 256).simplify();
        assert_eq!(extract_bv_value_wide(&bv, 256), Some(bytes));
        assert_eq!(extract_bv_value(&bv), Some(7));
    }

    /// Canonical numeral string of a concrete BV (z3 prints #x.../#b.../decimal).
    fn numeral(bv: &z3::ast::BV) -> String {
        use z3::ast::Ast;
        format!("{}", bv.simplify())
    }

    /// Round-trip a `<=128`-bit value at a ragged width: build the reference
    /// constant via make_bv_const, extract its bytes, rebuild via
    /// make_bv_from_bytes, and require identical Z3 numerals, including for
    /// width % 8 != 0 (angr-ph300.34).
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
