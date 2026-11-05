//! Tests for the scanf numeric spelling constraint (angr-vmi2l.108).

use super::*;

#[test]
fn max_digits_matches_python_table() {
    // math.ceil(math.log(2**bits, base)) for bits 8, 16, 32, 64.
    for (base, expect) in [(8, [3, 6, 11, 22]), (10, [3, 5, 10, 20]), (16, [2, 4, 8, 16])] {
        let got: Vec<u64> = [8, 16, 32, 64].iter().map(|&b| max_digits(base, b)).collect();
        assert_eq!(got, expect, "base {base}");
    }
}

/// The spelling of concrete `text` against a concrete value folds to a
/// concrete verdict.
fn spells(text: &[u8], conv: u8, base: u32, value: i64, bits: u32) -> bool {
    let chars: Vec<RustBV> = text.iter().map(|&c| byte(c)).collect();
    let target = RustBV::concrete(value as u128, bits);
    let verdict = numeric_spelling(&chars, conv, base, &target, bits);
    match verdict.as_u64() {
        Some(v) => v != 0,
        None => panic!("{} did not fold", String::from_utf8_lossy(text)),
    }
}

#[test]
fn decimal_spellings() {
    assert!(spells(b"0000133337", b'd', 10, 133_337, 32));
    // Ten digits cover all 32 bits, so a negative value is its unsigned spelling.
    assert!(spells(b"4294965959", b'd', 10, -1337, 32));
    assert!(!spells(b"-000001337", b'd', 10, -1337, 32));
    assert!(!spells(b"0000133338", b'd', 10, 133_337, 32));
}

#[test]
fn narrow_width_allows_a_minus_sign() {
    // `%2d`: two digits cannot cover 32 bits, so a leading '-' is allowed.
    assert!(spells(b"12", b'd', 10, 12, 32));
    assert!(spells(b"-7", b'd', 10, -7, 32));
    assert!(!spells(b"-7", b'd', 10, 7, 32));
    // The range bound: 100 has no two-digit spelling.
    assert!(!spells(b"00", b'd', 10, 100, 32));
}

#[test]
fn hex_spellings_take_either_case_and_a_0x_prefix() {
    assert!(spells(b"0000aaAA", b'x', 16, 0xaaaa, 32));
    assert!(spells(b"0x00aaaa", b'x', 16, 0xaaaa, 32));
    assert!(spells(b"-00CdcD", b'x', 16, -0xcdcd, 32));
    assert!(!spells(b"0000aaab", b'x', 16, 0xaaaa, 32));
}

#[test]
fn percent_i_detects_the_base() {
    // A leading 0 is octal, so a small value has no zero-padded decimal
    // spelling (Python's quirk, mirrored).
    assert!(!spells(b"0000000042", b'i', 10, 42, 32));
    assert!(spells(b"1000000000", b'i', 10, 1_000_000_000, 32));
    assert!(spells(b"0x0000002a", b'i', 10, 42, 32));
    assert!(spells(b"0000000052", b'i', 10, 42, 32)); // leading 0: octal
    assert!(spells(b"-000000052", b'i', 10, -42, 32));
}
