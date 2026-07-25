//! Pure unit tests for the Z3 numeral classifier and the string->value
//! parse helpers (angr-ph300.40). None of these touch a live Z3 model — the
//! functions operate on the raw numeral strings Z3's `Display` produces, so
//! they are exercised directly with hand-written inputs.

use super::*;

// --- Z3Numeral::classify ---------------------------------------------------

#[test]
fn test_classify_hex_strips_prefix() {
    match Z3Numeral::classify("#xdeadbeef") {
        Z3Numeral::Hex(s) => assert_eq!(s, "deadbeef"),
        _ => panic!("expected Hex"),
    }
}

#[test]
fn test_classify_binary_strips_prefix() {
    match Z3Numeral::classify("#b1010") {
        Z3Numeral::Bin(s) => assert_eq!(s, "1010"),
        _ => panic!("expected Bin"),
    }
}

#[test]
fn test_classify_decimal_verbatim() {
    match Z3Numeral::classify("12345") {
        Z3Numeral::Dec(s) => assert_eq!(s, "12345"),
        _ => panic!("expected Dec"),
    }
}

#[test]
fn test_classify_empty_is_decimal() {
    // The `#x` / `#b` markers require content after them, but classify only
    // dispatches on the prefix — an empty string carries no prefix, so it
    // falls through to the decimal arm (which parse helpers reject downstream).
    match Z3Numeral::classify("") {
        Z3Numeral::Dec(s) => assert_eq!(s, ""),
        _ => panic!("expected Dec"),
    }
}

// --- parse_wide_hex_low128 -------------------------------------------------

#[test]
fn test_parse_wide_hex_low128_small() {
    assert_eq!(parse_wide_hex_low128("ff"), Some(0xff));
    assert_eq!(parse_wide_hex_low128("deadbeef"), Some(0xdead_beef));
}

#[test]
fn test_parse_wide_hex_low128_exactly_32_chars() {
    let s = "f".repeat(32);
    assert_eq!(parse_wide_hex_low128(&s), Some(u128::MAX));
}

#[test]
fn test_parse_wide_hex_low128_over_128_bits_takes_low_bits() {
    // 40 hex chars = 160 bits; only the low 32 chars (128 bits) survive.
    // High byte "aa" must be discarded, low 32 chars are all 'f'.
    let s = format!("aabbbbbbbb{}", "f".repeat(32));
    assert_eq!(parse_wide_hex_low128(&s), Some(u128::MAX));
}

#[test]
fn test_parse_wide_hex_low128_invalid_is_none() {
    assert_eq!(parse_wide_hex_low128("xyz"), None);
}

// --- parse_wide_binary_low128 ----------------------------------------------

#[test]
fn test_parse_wide_binary_low128_small() {
    assert_eq!(parse_wide_binary_low128("1010"), Some(0b1010));
}

#[test]
fn test_parse_wide_binary_low128_over_128_bits_takes_low_bits() {
    // 130 bits: leading "11" is dropped, low 128 bits are all 1.
    let s = format!("11{}", "1".repeat(128));
    assert_eq!(parse_wide_binary_low128(&s), Some(u128::MAX));
}

#[test]
fn test_parse_wide_binary_low128_invalid_is_none() {
    assert_eq!(parse_wide_binary_low128("102"), None);
}

// --- parse_hex_to_bytes (big-endian, right-aligned) ------------------------

#[test]
fn test_parse_hex_to_bytes_exact_width() {
    assert_eq!(parse_hex_to_bytes("1234", 16), Some(vec![0x12, 0x34]));
}

#[test]
fn test_parse_hex_to_bytes_odd_length_padded() {
    // "1" pads to "01" -> single byte.
    assert_eq!(parse_hex_to_bytes("1", 8), Some(vec![0x01]));
    assert_eq!(parse_hex_to_bytes("abc", 16), Some(vec![0x0a, 0xbc]));
}

#[test]
fn test_parse_hex_to_bytes_right_aligned_in_wider_field() {
    // 8 bits of value in a 16-bit field: high byte zero, low byte 0xff.
    assert_eq!(parse_hex_to_bytes("ff", 16), Some(vec![0x00, 0xff]));
}

#[test]
fn test_parse_hex_to_bytes_non_ascii_no_panic() {
    // Regression (angr-qwyti.23): a multibyte UTF-8 char used to panic on the
    // `&padded[i..i + 2]` byte-slice ('not a char boundary'). Now the parser
    // filters to ASCII hex digits and never panics: the '€' is dropped, leaving
    // "12" -> 0x12.
    assert_eq!(parse_hex_to_bytes("1€2", 8), Some(vec![0x12]));
    // Purely non-hex input decodes to an all-zero field, not a panic.
    assert_eq!(parse_hex_to_bytes("€", 8), Some(vec![0x00]));
}

// --- parse_binary_to_bytes (big-endian, right-aligned) ---------------------

#[test]
fn test_parse_binary_to_bytes_exact_byte() {
    assert_eq!(parse_binary_to_bytes("11111111", 8), Some(vec![0xff]));
    assert_eq!(parse_binary_to_bytes("00001010", 8), Some(vec![0x0a]));
}

#[test]
fn test_parse_binary_to_bytes_right_aligned() {
    // 4 significant bits (1010) right-aligned in one byte -> 0x0a.
    assert_eq!(parse_binary_to_bytes("1010", 8), Some(vec![0x0a]));
}

#[test]
fn test_parse_binary_to_bytes_two_bytes() {
    assert_eq!(
        parse_binary_to_bytes("0000000100000010", 16),
        Some(vec![0x01, 0x02])
    );
}

// --- parse_decimal_to_bytes ------------------------------------------------

#[test]
fn test_parse_decimal_to_bytes_wide_right_aligned() {
    // The wide decoder only reaches this helper for widths > 128 bits, so
    // test the real contract: value right-aligned in a >16-byte field.
    let out = parse_decimal_to_bytes("255", 136).unwrap();
    assert_eq!(out.len(), 17);
    let mut expected = vec![0u8; 17];
    expected[16] = 0xff;
    assert_eq!(out, expected);
}

#[test]
fn test_parse_decimal_to_bytes_narrow_right_aligned() {
    // Regression for angr-ph300.35: width < 128 (byte_len < 16) must copy the
    // LOW bytes of the 16-byte BE repr, not the leading zero bytes. "300" in a
    // 32-bit field is 0x0000_012c -> [0, 0, 1, 44].
    assert_eq!(parse_decimal_to_bytes("300", 32), Some(vec![0, 0, 1, 44]));
    // Single byte: low byte survives.
    assert_eq!(parse_decimal_to_bytes("255", 8), Some(vec![0xff]));
    // Truncation to width matches the hex/binary decoders: 300 & 0xff = 44.
    assert_eq!(parse_decimal_to_bytes("300", 8), Some(vec![44]));
}

#[test]
fn test_parse_decimal_to_bytes_width_200_right_aligned() {
    // width 200 -> 25 bytes; value right-aligned, high bytes zero.
    let out = parse_decimal_to_bytes("300", 200).unwrap();
    assert_eq!(out.len(), 25);
    let mut expected = vec![0u8; 25];
    expected[23] = 1;
    expected[24] = 44;
    assert_eq!(out, expected);
}

#[test]
fn test_parse_decimal_to_bytes_exactly_128_bits() {
    let out = parse_decimal_to_bytes("255", 128).unwrap();
    assert_eq!(out.len(), 16);
    let mut expected = vec![0u8; 16];
    expected[15] = 0xff;
    assert_eq!(out, expected);
}

#[test]
fn test_parse_decimal_to_bytes_overflow_u128_is_none() {
    // A value beyond u128 cannot be parsed without bignum support.
    let too_big = "9".repeat(40);
    assert_eq!(parse_decimal_to_bytes(&too_big, 256), None);
}
