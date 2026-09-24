//! scanf format-specifier *parsing*.
//!
//! [`super::scanf`]'s `do_scanf` has two halves: work out what the format
//! string asks for, then mint and store a symbolic value per conversion.
//! This file is the first half. Every conversion the native engine declines
//! to model — the `%[...]` scanset, `%n`, the float conversions — is decided
//! here, in [`parse_scanf_format`]'s arms, each of which records why
//! deferring is the faithful behavior.
//!
//! [`LengthModifier::int_conv_bits`]: super::format_common::LengthModifier::int_conv_bits
//!
//! **Panic policy (angr-qwyti.11, angr-9ke6b.212):** the format string is
//! guest data, so this module carries the same
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]` as [`super::scanf`];
//! unparseable input returns a `ProcedureError` that falls back to Python.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use super::ProcedureError;
use super::format_common::{parse_length_modifier, parse_width_digits};

/// Max bytes a `%s` conversion mints when the format string gives no explicit
/// field width.
pub(super) const MAX_SCANF_STR_LEN: u64 = 256;

/// Parsed scanf format specifier.
pub(super) struct ScanfSpec {
    /// Width of the value to create in bits.
    pub(super) bits: u32,
    /// Whether this is a string specifier (%s).
    pub(super) is_string: bool,
    /// Whether this is the single-character specifier (%c). Unlike the numeric
    /// conversions it maps one input byte to one stored byte, so it can consume
    /// a harness-seeded stdin byte directly (angr-ggb66).
    pub(super) is_char: bool,
    /// Max width for %s (from field width, e.g. %10s), or MAX_SCANF_STR_LEN.
    pub(super) max_str_len: u64,
    /// Range bound implied by an explicit field width on a numeric conversion
    /// (`%2d`), as `(hi, lo_magnitude)` — see `digit_range_bound`. `None` for
    /// non-numeric conversions and for numeric ones whose width can already
    /// spell every value of the destination type.
    pub(super) digit_bound: Option<(u64, u64)>,
    /// Whether to suppress assignment (*).
    pub(super) suppress: bool,
}

/// Range bound for a numeric conversion carrying an explicit field width,
/// mirroring `angr/procedures/stubs/format_parser.py::FormatString.interpret`'s
/// `not_enough_bits` branch: a `%<w>d` can only spell `base**w` distinct
/// values, so when that is narrower than the destination type Python constrains
/// the parsed variable to `[-(base**(w-1) - 1), base**w - 1]`. Without it the
/// natively minted BVS is free across all 32/64 bits and the engine explores
/// states (`x == 12345` for a `%2d`) neither real scanf(3) nor angr's own
/// Python engine can reach.
///
/// Returns `(hi, lo_magnitude)`; the caller builds
/// `sym <=s hi && sym >=s -lo_magnitude`. `None` means "no constraint", for
/// three reasons:
///
/// * the width can already spell the whole type (Python's
///   `available_bits >= bits`),
/// * `bits` is outside the `1..=64` range a `u64` bound can describe, or
/// * the width is the degenerate `%0d`, where Python computes `base ** -1` —
///   a float its own `SGE` would choke on — so glibc's UB is left unmodelled
///   rather than mis-modelled.
pub(super) fn digit_range_bound(base: u32, digits: u64, bits: u32) -> Option<(u64, u64)> {
    if digits == 0 || bits == 0 || bits > 64 {
        return None;
    }
    // `base**digits < 2**bits` is the exact integer form of Python's
    // `digits * log2(base) < bits`. An overflow means the width dwarfs any
    // 64-bit type, i.e. it constrains nothing.
    let span = u128::from(base).checked_pow(u32::try_from(digits).ok()?)?;
    if span >= 1u128 << bits {
        return None;
    }
    // `span < 2**bits <= 2**64` and `digits >= 1`, so both fit a `u64` and the
    // division is exact: `span / base == base**(digits - 1)`.
    let hi = u64::try_from(span.checked_sub(1)?).ok()?;
    let lo_magnitude = u64::try_from((span / u128::from(base)).checked_sub(1)?).ok()?;
    Some((hi, lo_magnitude))
}

/// Parse scanf format specifiers from a format string.
/// Returns a list of specifiers (one per conversion that stores a value).
///
/// `arch_bits` sizes the `l`/`z`/`t` modifiers, which are `long`-width and so
/// 32-bit on ILP32 targets — see `LengthModifier::int_conv_bits`. Each spec's
/// `bits` becomes the width of the value stored at the caller's pointer, so
/// over-sizing it writes past the guest object.
pub(super) fn parse_scanf_format(
    fmt: &[u8],
    arch_bits: u32,
) -> Result<Vec<ScanfSpec>, ProcedureError> {
    let mut specs = Vec::new();
    let mut i = 0;

    while i < fmt.len() {
        if fmt[i] != b'%' {
            // Non-format characters are literal matchers; skip them
            i += 1;
            continue;
        }
        i += 1; // skip '%'
        if i >= fmt.len() {
            break;
        }

        // Handle %%
        if fmt[i] == b'%' {
            i += 1;
            continue;
        }

        // Check for suppression (*)
        let suppress = if i < fmt.len() && fmt[i] == b'*' {
            i += 1;
            true
        } else {
            false
        };

        // Parse field width. Clamp against MAX_SCANF_STR_LEN here, before the
        // value is used as a Vec/collect bound: parse_width_digits only
        // saturates to usize::MAX on overflow, which does not stop a short
        // digit run like "%9999999999999999999s" from encoding a near-MAX
        // width. Left unclamped, `(0..field_width).map(...).collect()` in
        // do_scanf hits an allocator capacity-overflow panic before any
        // allocation is attempted (angr-mi56k).
        let (width_val, w_adv) = parse_width_digits(fmt, i);
        let has_width = w_adv > 0;
        let field_width = (width_val as u64).min(MAX_SCANF_STR_LEN);
        i += w_adv;

        // Parse length modifier. `h`/`hh` narrow the destination to 16/8 bits
        // and `l`/`z`/`t` widen it to `long` width (matching Python's
        // format_parser.py int_len_mod); `z`/`j`/`t` are honoured at all
        // because glibc accepts them in scanf.
        let (modifier, m_adv) = parse_length_modifier(fmt, i);
        i += m_adv;

        if i >= fmt.len() {
            break;
        }

        let spec = fmt[i];
        i += 1;

        match spec {
            b'd' | b'i' | b'u' | b'x' | b'X' | b'o' => {
                let bits = modifier.int_conv_bits(arch_bits);
                // Matches Python's `interpret`: %x/%X read hex, %o octal,
                // everything else decimal.
                let base = match spec {
                    b'x' | b'X' => 16,
                    b'o' => 8,
                    _ => 10,
                };
                // Note `field_width` is the MAX_SCANF_STR_LEN-clamped value;
                // any width that large is far too wide to bound a <=64-bit
                // type, so `digit_range_bound` returns `None` either way.
                let digit_bound = if has_width {
                    digit_range_bound(base, field_width, bits)
                } else {
                    None
                };
                specs.push(ScanfSpec {
                    bits,
                    is_string: false,
                    is_char: false,
                    max_str_len: 0,
                    digit_bound,
                    suppress,
                });
            }
            b'c' => {
                specs.push(ScanfSpec {
                    bits: 8,
                    is_string: false,
                    is_char: true,
                    max_str_len: 0,
                    digit_bound: None,
                    suppress,
                });
            }
            b's' => {
                let max_len = if has_width {
                    field_width
                } else {
                    MAX_SCANF_STR_LEN
                };
                specs.push(ScanfSpec {
                    bits: 8,
                    is_string: true,
                    is_char: false,
                    max_str_len: max_len,
                    digit_bound: None,
                    suppress,
                });
            }
            b'[' => {
                // Scanset %[...] / %[^...]. Deliberately NOT implemented
                // natively: angr's Python
                // `format_parser.py::ScanfFormatParser.basic_spec` has no `[`
                // entry, and `_match_spec` only matches a nugget by prefix
                // against those, so `%[a-z]` never becomes a FormatSpecifier
                // there. Python's `extract_components` instead keeps the `%`
                // as a literal one-char component and matches the bracket
                // expression against the input stream as literal text,
                // consuming no pointer argument. A native implementation that
                // minted a symbolic string and consumed a pointer would
                // silently disagree with the reference engine on the stored
                // buffer, the return value, and the argument bookkeeping of
                // every conversion after it, so the faithful behavior is to
                // defer the whole call (angr-6cp06.7).
                return Err(ProcedureError::Other(
                    "scanf %[...] scanset not supported natively".to_string(),
                ));
            }
            b'n' => {
                // %n stores the count of chars consumed so far. Deliberately
                // NOT implemented natively: deferring to Python is the faithful
                // behavior. Python's format_parser.py::FormatString.interpret
                // raises SimProcedureError on %n in the addr-based (sscanf-from-
                // memory) path, and treats it as a numeric read in the SimPackets
                // (stdin/file) path. A native write of the count would diverge
                // from both. The fallback reproduces Python exactly for free.
                return Err(ProcedureError::Other(
                    "scanf %n not supported natively".to_string(),
                ));
            }
            _ => {
                // Unknown specifier — fall back to Python. Includes the float
                // specifiers %f/%e/%g: Python's format_parser.py::FormatString
                // .interpret raises SimProcedureError on them, so a native
                // symbolic-float read would diverge. Faithful behavior is to defer.
                return Err(ProcedureError::Other(format!(
                    "scanf: unsupported specifier '%{}'",
                    spec as char
                )));
            }
        }
    }

    Ok(specs)
}
