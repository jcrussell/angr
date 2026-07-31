//! Shared leaf-parsers for printf/scanf-family format strings.
//!
//! The sprintf state machine and the scanf state machine differ at the
//! top level (flags/precision vs. suppression/width-only), but they share
//! two pieces of syntax verbatim: a run of ASCII decimal digits (field
//! width) and the length-modifier characters (`l`, `ll`, `h`, `hh`, plus
//! printf's `z`/`j`/`t`).
//!
//! Keeping those two sub-parsers in one place avoids drift between the
//! two consumers (they were already on slightly different schedules for
//! adopting new modifiers) and gives a single home for the unit tests
//! that exercise edge cases like "ll at end of buffer".
//!
//! The top-level state machine in each consumer still drives its own
//! logic — these helpers only cover the leaves.
//!
//! See `procedures/sprintf.rs::format_string` and
//! `procedures/scanf.rs::parse_scanf_format` for the callers.

/// Shared upper bound on a concrete format-string scan for the printf/sprintf/
/// scanf family. One home so a future reduction is applied everywhere at once
/// rather than silently missing a consumer that kept its own local `4096`
/// (angr-myzjx.7). The str-family shares [`super::strings::MAX_STRING_SCAN`].
pub(crate) const MAX_FORMAT_LEN: usize = 4096;

/// Parsed length modifier. Both printf-family and scanf-family parsers
/// share these kinds; consumers map them onto their internal flags.
///
/// `SizeT`/`IntMax`/`PtrDiff` are C99/printf modifiers (`z`/`j`/`t`).
/// `scanf` historically didn't recognise them in this codebase, but
/// glibc accepts them, so honouring them in both consumers is the
/// correct (and forward-compatible) behaviour.
// Crate-`pub` for the same `fuzz_api` re-export reason as `parse_width_digits`:
// it is the return type of `parse_length_modifier`, so narrowing it makes that
// signature reference a private type (E0446 in the `fuzz/` cargo-fuzz crate).
#[cfg_attr(not(feature = "fuzzing"), allow(unreachable_pub))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LengthModifier {
    None,
    Char,     // hh
    Short,    // h
    Long,     // l
    LongLong, // ll
    SizeT,    // z
    IntMax,   // j
    PtrDiff,  // t
}

impl LengthModifier {
    /// True if this modifier promotes integer operands to 64-bit width.
    /// On 64-bit targets, `long`, `long long`, `size_t`, `intmax_t`, and
    /// `ptrdiff_t` are all 64-bit.
    pub(crate) fn is_64bit(self) -> bool {
        matches!(
            self,
            LengthModifier::Long
                | LengthModifier::LongLong
                | LengthModifier::SizeT
                | LengthModifier::IntMax
                | LengthModifier::PtrDiff
        )
    }
}

/// Parse a run of ASCII decimal digits at `fmt[start..]` into a width value.
/// Returns `(width, bytes_consumed)`. If no digits are present, returns
/// `(0, 0)` so callers can distinguish "no width" from "width zero" by
/// looking at the consumed count.
// Stays crate-`pub` (not `pub(crate)`) because `lib.rs::fuzz_api` re-exports it
// for the cargo-fuzz targets under `fuzz/`. That re-export only exists under the
// `fuzzing` feature, so a stock build sees an unreachable `pub` — suppressed
// here rather than widened, since narrowing it breaks `--features fuzzing`
// with E0364 (angr-9ke6b.214).
#[cfg_attr(not(feature = "fuzzing"), allow(unreachable_pub))]
pub fn parse_width_digits(fmt: &[u8], start: usize) -> (usize, usize) {
    let mut width: usize = 0;
    let mut i = start;
    while i < fmt.len() && fmt[i].is_ascii_digit() {
        width = width
            .saturating_mul(10)
            .saturating_add((fmt[i] - b'0') as usize);
        i += 1;
    }
    (width, i - start)
}

/// Parse a printf/scanf length modifier at `fmt[start..]`. Returns the
/// modifier kind and the number of bytes consumed (0 if no modifier).
// Crate-`pub` for the same `fuzz_api` re-export reason as `parse_width_digits`.
#[cfg_attr(not(feature = "fuzzing"), allow(unreachable_pub))]
pub fn parse_length_modifier(fmt: &[u8], start: usize) -> (LengthModifier, usize) {
    if start >= fmt.len() {
        return (LengthModifier::None, 0);
    }
    match fmt[start] {
        b'l' => {
            if start + 1 < fmt.len() && fmt[start + 1] == b'l' {
                (LengthModifier::LongLong, 2)
            } else {
                (LengthModifier::Long, 1)
            }
        }
        b'h' => {
            if start + 1 < fmt.len() && fmt[start + 1] == b'h' {
                (LengthModifier::Char, 2)
            } else {
                (LengthModifier::Short, 1)
            }
        }
        b'z' => (LengthModifier::SizeT, 1),
        b'j' => (LengthModifier::IntMax, 1),
        b't' => (LengthModifier::PtrDiff, 1),
        _ => (LengthModifier::None, 0),
    }
}

#[cfg(test)]
#[path = "format_common_tests.rs"]
mod tests;
