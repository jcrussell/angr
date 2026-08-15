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

    // Keep only ASCII hex digits, then decode whole-byte pairs. Z3's Display
    // emits ASCII hex, but stay char-safe like the sibling decoders: the old
    // `&padded[i..i + 2]` byte-slice panicked ('not a char boundary') on any
    // multibyte UTF-8 char (angr-qwyti.23). Filtering to `is_ascii_hexdigit`
    // first — same style as parse_binary_to_bytes — is panic-free on any input.
    let mut digits: Vec<u8> = s.bytes().filter(u8::is_ascii_hexdigit).collect();
    // Prepend a zero nibble on odd length so pairs align to whole bytes.
    if digits.len() % 2 == 1 {
        digits.insert(0, b'0');
    }

    // Decode each hex pair to a byte (big-endian output). Every byte is a
    // validated hex digit, so both nibble conversions succeed.
    let hex_bytes: Vec<u8> = digits
        .chunks_exact(2)
        .filter_map(|pair| {
            let hi = (pair[0] as char).to_digit(16)?;
            let lo = (pair[1] as char).to_digit(16)?;
            Some(((hi << 4) | lo) as u8)
        })
        .collect();

    // Copy to result (right-aligned, big-endian)
    let offset = byte_len.saturating_sub(hex_bytes.len());
    for (i, &b) in hex_bytes.iter().enumerate() {
        // overflow-ok: `offset <= byte_len` (the `saturating_sub` above) and
        // `i < hex_bytes.len() <= s.len()`, both far below `usize::MAX`.
        let idx = offset + i;
        if idx < byte_len {
            result[idx] = b;
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
    // pub(super) contract. Always-on, NOT `debug_assert!` (angr-9ke6b.220):
    // without it a too-long bit string would be *silently truncated* — the
    // `saturating_sub` clamps `bit_offset` to 0 and the `byte_idx < byte_len`
    // guard below drops the overflow bits, yielding a wrong value with no
    // diagnostic. Wide-BV eval only, so the length compare is off the hot path.
    assert!(
        bits.len() <= byte_len * 8,
        "parse_binary_to_bytes: {} bits exceed {} byte capacity",
        bits.len(),
        byte_len
    );
    let bit_offset = (byte_len * 8).saturating_sub(bits.len());
    for (i, &bit) in bits.iter().enumerate() {
        // overflow-ok: the `assert!` above bounds `bits.len()` (and so `i`) by
        // `byte_len * 8`, and `bit_offset <= byte_len * 8` by construction.
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
        // overflow-ok: `copy` is `<=` both operands it is subtracted from.
        let copy = byte_len.min(16);
        result[byte_len - copy..].copy_from_slice(&src[16 - copy..]);
        return Some(result);
    }

    // For very large decimals, we'd need big integer parsing
    // This is rare in practice as Z3 typically uses hex format
    None
}

test_submod!("parse_tests.rs" => tests);
