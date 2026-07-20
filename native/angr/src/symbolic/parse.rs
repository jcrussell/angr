//! Helpers for parsing Z3 BV string representations into concrete values.
//!
//! Extracted from `context.rs` (angr-a2br.2 slice 4) — leaf free functions
//! on the concretization path with no `SymContext` dependency. They turn the
//! hex / binary / decimal numeral strings Z3 hands back (via
//! `get_numeral_string`) into either a low-128-bit `u128` or a big-endian
//! byte vector sized to the BV width. All callers live in
//! `context.rs` (the `concrete_value_*` path).
//!
//! Crate-visible surface: [`parse_wide_hex_low128`],
//! [`parse_wide_binary_low128`], [`parse_hex_to_bytes`],
//! [`parse_binary_to_bytes`], [`parse_decimal_to_bytes`].

/// Classification of a Z3 BV numeral string by its format prefix.
///
/// `z3::ast::BV`'s `Display` emits a concrete constant in one of three
/// forms: `#x<hex>`, `#b<binary>`, or a bare decimal. This enum captures
/// that three-way `strip_prefix` dispatch in a single place so both the
/// low-128 decoder ([`super::bv_codec::extract_bv_value_from_string`]) and
/// the full-width decoder ([`super::bv_codec::extract_bv_value_wide`]) share
/// one copy of the format knowledge instead of open-coding it twice. Each
/// variant carries the payload slice with the prefix already stripped.
pub(super) enum Z3Numeral<'a> {
    Hex(&'a str),
    Bin(&'a str),
    Dec(&'a str),
}

impl<'a> Z3Numeral<'a> {
    /// Classify a Z3 numeral string by its prefix. A `#x` / `#b` prefix
    /// selects hex / binary and is stripped; anything else is treated as a
    /// decimal numeral and returned verbatim.
    pub(super) fn classify(s: &'a str) -> Self {
        if let Some(hex) = s.strip_prefix("#x") {
            Z3Numeral::Hex(hex)
        } else if let Some(bin) = s.strip_prefix("#b") {
            Z3Numeral::Bin(bin)
        } else {
            Z3Numeral::Dec(s)
        }
    }
}

/// Parse a hex string to u128, taking low 128 bits if larger.
pub(super) fn parse_wide_hex_low128(s: &str) -> Option<u128> {
    // For values > 128 bits (> 32 hex chars), take low 32 chars
    let low_hex = if s.len() > 32 { &s[s.len() - 32..] } else { s };
    u128::from_str_radix(low_hex, 16).ok()
}

/// Parse a binary string to u128, taking low 128 bits if larger.
pub(super) fn parse_wide_binary_low128(s: &str) -> Option<u128> {
    // For values > 128 bits (> 128 bin chars), take low 128 chars
    let low_bin = if s.len() > 128 {
        &s[s.len() - 128..]
    } else {
        s
    };
    u128::from_str_radix(low_bin, 2).ok()
}

/// Parse a hex string to full bytes (big-endian).
pub(super) fn parse_hex_to_bytes(s: &str, width: u32) -> Option<Vec<u8>> {
    let byte_len = width.div_ceil(8) as usize;
    let mut result = vec![0u8; byte_len];

    // Pad hex string to even length
    let padded = if s.len() % 2 == 1 {
        format!("0{s}")
    } else {
        s.to_string()
    };

    // Parse hex pairs from right to left (big-endian output)
    let hex_bytes: Vec<u8> = (0..padded.len())
        .step_by(2)
        .filter_map(|i| u8::from_str_radix(&padded[i..i + 2], 16).ok())
        .collect();

    // Copy to result (right-aligned, big-endian)
    let offset = byte_len.saturating_sub(hex_bytes.len());
    for (i, &b) in hex_bytes.iter().enumerate() {
        if offset + i < byte_len {
            result[offset + i] = b;
        }
    }

    Some(result)
}

/// Parse a binary string to full bytes (big-endian).
pub(super) fn parse_binary_to_bytes(s: &str, width: u32) -> Option<Vec<u8>> {
    let byte_len = width.div_ceil(8) as usize;
    let mut result = vec![0u8; byte_len];

    // Parse bits from right to left
    let bits: Vec<u8> = s
        .chars()
        .filter_map(|c| match c {
            '0' => Some(0),
            '1' => Some(1),
            _ => None,
        })
        .collect();

    // Build bytes from bits (big-endian). The sole caller
    // (extract_bv_value_wide in bv_codec.rs) derives `bits` and `width` from
    // the same Z3 BV, so `bits.len() <= byte_len * 8` always holds. Guard the
    // pub(super) contract: saturating_sub avoids an underflow-panic if a future
    // caller passes a too-long bit string, and the debug_assert flags it in dev.
    debug_assert!(
        bits.len() <= byte_len * 8,
        "parse_binary_to_bytes: {} bits exceed {} byte capacity",
        bits.len(),
        byte_len
    );
    let bit_offset = (byte_len * 8).saturating_sub(bits.len());
    for (i, &bit) in bits.iter().enumerate() {
        let bit_pos = bit_offset + i;
        let byte_idx = bit_pos / 8;
        let bit_idx = 7 - (bit_pos % 8);
        if byte_idx < byte_len {
            result[byte_idx] |= bit << bit_idx;
        }
    }

    Some(result)
}

/// Parse a decimal string to bytes (big-endian).
pub(super) fn parse_decimal_to_bytes(s: &str, width: u32) -> Option<Vec<u8>> {
    // For small values (<= 128 bits), parse and convert.
    if let Ok(v) = s.parse::<u128>() {
        let byte_len = width.div_ceil(8) as usize;
        let mut result = vec![0u8; byte_len];
        let src = v.to_be_bytes(); // 16-byte big-endian repr of `v`
        // Right-align: the LSB of `v` must land in the last byte of `result`.
        // Copy the low `min(byte_len, 16)` bytes of `src` into the tail of
        // `result`. When byte_len > 16 the higher bytes stay zero (v is only
        // 128 bits); when byte_len < 16 the high `src` bytes are dropped
        // (truncation to width, matching the hex/binary decoders). The old
        // code copied the *leading* (high, zero) bytes of `src` for
        // byte_len < 16, silently zeroing every real value (angr-ph300.35).
        let copy = byte_len.min(16);
        result[byte_len - copy..].copy_from_slice(&src[16 - copy..]);
        return Some(result);
    }

    // For very large decimals, we'd need big integer parsing
    // This is rare in practice as Z3 typically uses hex format
    None
}

#[cfg(test)]
#[path = "parse_tests.rs"]
mod tests;
