//! The ASCII spelling constraint behind a native numeric scanf conversion on
//! stdin (angr-vmi2l.108).
//!
//! A port of `angr/procedures/stubs/format_parser.py::FormatString`'s
//! SimPackets branch of `interpret`: a numeric conversion reads a fixed run of
//! digit bytes off the stream and constrains them to spell the stored value,
//! so `posix.dumps(0)` holds text real scanf(3) parses back to that value. The
//! arithmetic follows claripy's operators (`//` is `UDiv`, `%` is `URem`, an
//! `int` operand is a value of the target's width), so the two engines build
//! the same constraint.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use crate::symbolic::RustBV;

/// Python's `max_digits = math.ceil(math.log(2**bits, base))`: the digit
/// bytes a numeric conversion without a field width reads. Computed in the
/// same floating point as Python, `log(x) / log(base)`, so it rounds the same.
pub(super) fn max_digits(base: u32, bits: u32) -> u64 {
    let digits = 2f64.powi(bits as i32).ln() / f64::from(base).ln();
    digits.ceil() as u64
}

/// `base**exp` reduced to `bits` bits, the value claripy builds for the
/// Python int `base**exp` against a `bits`-wide operand.
fn pow_mod(base: u32, exp: usize, bits: u32) -> u128 {
    let mask = if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    };
    (0..exp).fold(1u128, |acc, _| acc.wrapping_mul(u128::from(base)) & mask)
}

fn byte(c: u8) -> RustBV {
    RustBV::concrete(u128::from(c), 8)
}

fn all(constraints: impl IntoIterator<Item = RustBV>) -> RustBV {
    constraints
        .into_iter()
        .reduce(|acc, c| acc.and(&c))
        .unwrap_or_else(|| RustBV::concrete(1, 1))
}

fn any(constraints: impl IntoIterator<Item = RustBV>) -> RustBV {
    constraints
        .into_iter()
        .reduce(|acc, c| acc.or(&c))
        .unwrap_or_else(|| RustBV::concrete(0, 1))
}

/// Python's `_digit_chars`: the lowercase and uppercase ASCII byte for a
/// digit value of `bits` bits.
fn digit_chars(value: &RustBV, base: u32, bits: u32) -> (RustBV, RustBV) {
    let at = |c: u8| RustBV::concrete(u128::from(c), bits);
    let mut lower = value.add(&at(b'0'));
    let mut upper = lower.clone();
    if base > 10 {
        let letter = value.uge(&at(10));
        lower = letter.ite(&value.add(&at(b'a' - 10)), &lower);
        upper = letter.ite(&value.add(&at(b'A' - 10)), &upper);
    }
    (lower.extract(7, 0), upper.extract(7, 0))
}

/// Python's `_spell`: `chars` (most significant first) spell `target` in
/// `base`, with a leading `-` for a negative value when the digits cannot
/// cover the whole range of the variable.
fn spell(chars: &[RustBV], base: u32, target: &RustBV, bits: u32, allow_negative: bool) -> RustBV {
    let digits = chars.len();
    let at = |v: u128| RustBV::concrete(v, bits);
    let negative = target.slt(&at(0));
    let not_enough_bits = (digits as f64) * f64::from(base).log2() < f64::from(bits);

    let mut constraints = Vec::new();
    if not_enough_bits {
        let span = at(pow_mod(base, digits, bits));
        let one = at(1);
        if allow_negative {
            let below = at(pow_mod(base, digits.saturating_sub(1), bits));
            constraints.push(target.sle(&span.sub(&one)));
            constraints.push(target.sge(&below.sub(&one).neg()));
        } else {
            constraints.push(target.ule(&span.sub(&one)));
        }
    }

    let neg_target = target.neg();
    let radix = at(u128::from(base));
    for (i, digit) in chars.iter().rev().enumerate() {
        let place = at(pow_mod(base, i, bits));
        let (mut lower, mut upper) = digit_chars(&target.udiv(&place).urem(&radix), base, bits);
        if not_enough_bits && allow_negative {
            let (neg_lower, neg_upper) = if i == digits - 1 {
                (byte(b'-'), byte(b'-'))
            } else {
                digit_chars(&neg_target.udiv(&place).urem(&radix), base, bits)
            };
            lower = negative.ite(&neg_lower, &lower);
            upper = negative.ite(&neg_upper, &upper);
        }
        constraints.push(digit.eq(&lower).or(&digit.eq(&upper)));
    }
    all(constraints)
}

/// The constraint `interpret` adds for one numeric conversion: `chars`, the
/// digit bytes read off the stream, spell `target` in one of the spellings
/// the conversion `conv` (`d i u x X o`) accepts. `chars` is never empty.
pub(super) fn numeric_spelling(
    chars: &[RustBV],
    conv: u8,
    base: u32,
    target: &RustBV,
    bits: u32,
) -> RustBV {
    let digits = chars.len();
    let is = |j: usize, c: u8| chars[j].eq(&byte(c));
    let isnt = |j: usize, c: u8| chars[j].ne(&byte(c));
    let mut spellings = Vec::new();
    if conv == b'i' && digits > 1 {
        // %i detects the base: 0x/0X means hex, a leading 0 means octal,
        // after an optional '-'.
        let neg_target = target.neg();
        spellings.push(all([
            isnt(0, b'0'),
            isnt(0, b'-').or(&isnt(1, b'0')),
            spell(chars, 10, target, bits, true),
        ]));
        spellings.push(is(0, b'0').and(&spell(&chars[1..], 8, target, bits, false)));
        if digits > 2 {
            spellings.push(all([
                is(0, b'-'),
                is(1, b'0'),
                spell(&chars[2..], 8, &neg_target, bits, false),
            ]));
        }
        if digits > 3 {
            spellings.push(all([
                is(0, b'-'),
                is(1, b'0'),
                is(2, b'x').or(&is(2, b'X')),
                spell(&chars[3..], 16, &neg_target, bits, false),
            ]));
        }
    } else {
        spellings.push(spell(chars, base, target, bits, true));
    }
    if (base == 16 || conv == b'i') && digits > 2 {
        spellings.push(all([
            is(0, b'0'),
            is(1, b'x').or(&is(1, b'X')),
            spell(&chars[2..], 16, target, bits, false),
        ]));
    }
    any(spellings)
}

test_submod!("scanf_spell_tests.rs" => scanf_spell_tests);
