// Tests for format_common.rs (printf-family format-spec parsing helpers).
// Extracted from the inline `mod tests` per the sibling-extraction campaign.

use super::*;

#[test]
fn width_no_digits() {
    assert_eq!(parse_width_digits(b"abc", 0), (0, 0));
    assert_eq!(parse_width_digits(b"d", 0), (0, 0));
}

#[test]
fn width_single_digit() {
    assert_eq!(parse_width_digits(b"5d", 0), (5, 1));
    assert_eq!(parse_width_digits(b"0s", 0), (0, 1));
}

#[test]
fn width_multi_digit() {
    assert_eq!(parse_width_digits(b"42x", 0), (42, 2));
    assert_eq!(parse_width_digits(b"100", 0), (100, 3));
    assert_eq!(parse_width_digits(b"007o", 0), (7, 3));
}

#[test]
fn width_offset_start() {
    assert_eq!(parse_width_digits(b"%10s", 1), (10, 2));
    assert_eq!(parse_width_digits(b"%-5d", 2), (5, 1));
}

#[test]
fn width_end_of_buffer() {
    assert_eq!(parse_width_digits(b"42", 0), (42, 2));
    assert_eq!(parse_width_digits(b"", 0), (0, 0));
}

#[test]
fn length_none_on_other_chars() {
    assert_eq!(parse_length_modifier(b"d", 0), (LengthModifier::None, 0));
    assert_eq!(parse_length_modifier(b"s", 0), (LengthModifier::None, 0));
    assert_eq!(parse_length_modifier(b"x", 0), (LengthModifier::None, 0));
    assert_eq!(parse_length_modifier(b"", 0), (LengthModifier::None, 0));
}

#[test]
fn length_l_variants() {
    assert_eq!(parse_length_modifier(b"ld", 0), (LengthModifier::Long, 1));
    assert_eq!(
        parse_length_modifier(b"lld", 0),
        (LengthModifier::LongLong, 2)
    );
    // `l` at end of buffer is just Long (1 byte consumed)
    assert_eq!(parse_length_modifier(b"l", 0), (LengthModifier::Long, 1));
}

#[test]
fn length_h_variants() {
    assert_eq!(parse_length_modifier(b"hd", 0), (LengthModifier::Short, 1));
    assert_eq!(parse_length_modifier(b"hhd", 0), (LengthModifier::Char, 2));
    assert_eq!(parse_length_modifier(b"h", 0), (LengthModifier::Short, 1));
}

#[test]
fn length_zjt() {
    assert_eq!(parse_length_modifier(b"zd", 0), (LengthModifier::SizeT, 1));
    assert_eq!(parse_length_modifier(b"jd", 0), (LengthModifier::IntMax, 1));
    assert_eq!(
        parse_length_modifier(b"td", 0),
        (LengthModifier::PtrDiff, 1)
    );
}

#[test]
fn int_conv_bits_lp64() {
    assert_eq!(LengthModifier::Char.int_conv_bits(64), 8);
    assert_eq!(LengthModifier::Short.int_conv_bits(64), 16);
    assert_eq!(LengthModifier::None.int_conv_bits(64), 32);
    assert_eq!(LengthModifier::Long.int_conv_bits(64), 64);
    assert_eq!(LengthModifier::LongLong.int_conv_bits(64), 64);
    assert_eq!(LengthModifier::SizeT.int_conv_bits(64), 64);
    assert_eq!(LengthModifier::IntMax.int_conv_bits(64), 64);
    assert_eq!(LengthModifier::PtrDiff.int_conv_bits(64), 64);
}

/// `l`/`z`/`t` are `long`-width and must narrow to 32 on ILP32 targets
/// (Python's `SimTypeLong` / `SimTypeLength`), while `ll`/`j`
/// (`SimTypeLongLong`) stay 64-bit everywhere. angr-9ke6b.111.
#[test]
fn int_conv_bits_ilp32_narrows_long_family_only() {
    assert_eq!(LengthModifier::Long.int_conv_bits(32), 32);
    assert_eq!(LengthModifier::SizeT.int_conv_bits(32), 32);
    assert_eq!(LengthModifier::PtrDiff.int_conv_bits(32), 32);
    assert_eq!(LengthModifier::LongLong.int_conv_bits(32), 64);
    assert_eq!(LengthModifier::IntMax.int_conv_bits(32), 64);
    // The narrow modifiers are arch-independent.
    assert_eq!(LengthModifier::Char.int_conv_bits(32), 8);
    assert_eq!(LengthModifier::Short.int_conv_bits(32), 16);
    assert_eq!(LengthModifier::None.int_conv_bits(32), 32);
}
