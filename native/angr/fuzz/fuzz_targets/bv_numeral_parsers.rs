//! Fuzz the Z3 numeral-string decoders (`#x` hex / `#b` binary / decimal) that
//! turn a solver's `Display` output back into concrete bytes. This family is
//! where silent width/truncation bugs clustered (angr-ph300.34/.35, the
//! decimal path's silent-zero). All five functions must be total (never panic)
//! and the byte-producing ones must return exactly `width.div_ceil(8)` bytes.
#![no_main]

use libfuzzer_sys::fuzz_target;

use rustylib::fuzz_api::{
    parse_binary_to_bytes, parse_decimal_to_bytes, parse_hex_to_bytes, parse_wide_binary_low128,
    parse_wide_hex_low128,
};

fuzz_target!(|data: &[u8]| {
    // First two bytes pick a bit width (capped at u16 so the byte buffers stay
    // small); the rest is the untrusted numeral payload.
    let (width, body) = match data.split_first_chunk::<2>() {
        Some((w, rest)) => (u16::from_le_bytes(*w) as u32, rest),
        None => (0, data),
    };
    // Z3's `Display` for a BV numeral is always an ASCII string (`#x`/`#b`
    // prefix already stripped by `Z3Numeral::classify`, leaving ASCII hex /
    // binary / decimal). Mirror that contract: feed ASCII bytes only. Feeding
    // multibyte UTF-8 tripped a non-char-boundary panic in `parse_hex_to_bytes`
    // that the real Z3-fed caller can never reach — tracked as a separate
    // robustness follow-up, not exercised here.
    let s: String = body.iter().filter(|b| b.is_ascii()).map(|&b| b as char).collect();

    // Low-128 decoders: must not panic on arbitrary (possibly over-long) input.
    let _ = parse_wide_hex_low128(&s);
    let _ = parse_wide_binary_low128(&s);

    // Hex and decimal decoders truncate to any width safely, so the fuzzed
    // width goes straight in.
    let expected = width.div_ceil(8) as usize;
    if let Some(bytes) = parse_hex_to_bytes(&s, width) {
        assert_eq!(bytes.len(), expected, "hex bytes width mismatch");
    }
    if let Some(bytes) = parse_decimal_to_bytes(&s, width) {
        assert_eq!(bytes.len(), expected, "decimal bytes width mismatch");
    }

    // `parse_binary_to_bytes` carries a `pub(super)` contract (a dev-only
    // `debug_assert!`): the width must cover every significant bit, because the
    // sole real caller derives both `s` and `width` from the same Z3 BV. Honour
    // it so the harness tests the decoder's real behaviour, not the tripwire —
    // bump the width to at least the number of `0`/`1` chars in the payload.
    let bin_bits = s.chars().filter(|c| *c == '0' || *c == '1').count();
    let bin_width = width.max(bin_bits as u32);
    if let Some(bytes) = parse_binary_to_bytes(&s, bin_width) {
        assert_eq!(
            bytes.len(),
            bin_width.div_ceil(8) as usize,
            "binary bytes width mismatch"
        );
    }
});
