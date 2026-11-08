//! Conversion *rendering* for the native sprintf family.
//!
//! The second half of [`super::sprintf`]'s `format_string` loop: given a
//! [`ConversionSpec`] already parsed by [`super::sprintf_spec`], pull the
//! variadic argument it names and append its bytes to the output buffer.
//! Every Python-parity defer that depends on the argument *value* rather than
//! the specifier shape — an unterminated `%s` string, a `%.*s` precision
//! past the NUL — is decided here.

use super::format_common::LengthModifier;
use super::sprintf_spec::ConversionSpec;
use super::strings::{MAX_STRING_SCAN, scan_concrete_bounded};
use super::{ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Mask `val` to `bits` then sign-extend to i64, mirroring Python's mask
/// (`c_val &= (1<<size*8)-1`) followed by the signed fold (`if signed and
/// high bit set: c_val -= 1<<size*8`) for `d`/`i` specs.
fn signed_at_width(val: u64, bits: u32) -> i64 {
    match bits {
        8 => val as i8 as i64,
        16 => val as i16 as i64,
        32 => val as i32 as i64,
        _ => val as i64,
    }
}

/// Mask `val` to `bits` (zero-extended), mirroring Python's `c_val &=
/// (1<<size*8)-1` for the unsigned `u`/`x`/`X`/`o` specs.
fn unsigned_at_width(val: u64, bits: u32) -> u64 {
    if bits >= 64 {
        val
    } else {
        val & ((1u64 << bits) - 1)
    }
}
/// Pad a formatted value and push to output.
fn pad_and_push(
    output: &mut Vec<u8>,
    value: &[u8],
    width: usize,
    left_align: bool,
    zero_pad: bool,
    is_negative: bool,
) {
    if width <= value.len() {
        output.extend_from_slice(value);
        return;
    }
    // overflow-ok: width > value.len() is guaranteed by the early return above.
    let pad_count = width - value.len();
    let pad_char = if zero_pad { b'0' } else { b' ' };

    if left_align {
        output.extend_from_slice(value);
        for _ in 0..pad_count {
            output.push(b' ');
        }
    } else if zero_pad && is_negative {
        // For negative numbers with zero-pad: "-007" not "00-7"
        output.push(b'-');
        for _ in 0..pad_count {
            output.push(b'0');
        }
        output.extend_from_slice(&value[1..]); // skip the '-' already in value
    } else {
        for _ in 0..pad_count {
            output.push(pad_char);
        }
        output.extend_from_slice(value);
    }
}

/// Consume the next variadic argument as a concrete `u64`.
///
/// `what` names the argument in the "ran off the end of the declared varargs"
/// error; a symbolic argument is reported by `extract_concrete_arg` under its
/// positional name instead. Shared by every value-consuming arm of
/// [`render_conversion`] — the arms used to open with this same five-line
/// bounds-check/extract/advance preamble apiece.
fn next_arg(args: &[RustBV], arg_idx: &mut usize, what: &str) -> Result<u64, ProcedureError> {
    if *arg_idx >= args.len() {
        return Err(ProcedureError::SymbolicArgument(what.to_string()));
    }
    let val = extract_concrete_arg(&args[*arg_idx], &format!("arg{}", *arg_idx))?;
    // overflow-ok: `*arg_idx < args.len()` was checked above.
    *arg_idx += 1;
    Ok(val)
}

/// [`next_arg`] narrowed to the spec's conversion width for the unsigned
/// `%u`/`%x`/`%X`/`%o` family.
///
/// `FormatSpecifier.signed` reads the spec type's own signedness, so
/// `FormatString.replace` masks an unsigned operand without sign-folding it:
/// a high-bit value renders as its plain unsigned digits, as here.
fn unsigned_operand(
    args: &[RustBV],
    arg_idx: &mut usize,
    cspec: &ConversionSpec,
    arch_bits: u32,
    what: &str,
) -> Result<u64, ProcedureError> {
    let val = next_arg(args, arg_idx, what)?;
    Ok(unsigned_at_width(
        val,
        cspec.modifier.int_conv_bits(arch_bits),
    ))
}

/// Render one parsed specifier into `output`, consuming its variadic argument.
///
/// `arch_bits` is the guest's `long`/pointer width, threaded through to
/// [`LengthModifier::int_conv_bits`](super::format_common::LengthModifier::int_conv_bits).
pub(super) fn render_conversion(
    state: &mut RustSimState,
    cspec: &ConversionSpec,
    args: &[RustBV],
    arg_idx: &mut usize,
    arch_bits: u32,
    output: &mut Vec<u8>,
) -> Result<(), ProcedureError> {
    // `%ls`/`%lc` take `wchar_t*`/`wint_t`: glibc renders the wide string
    // and Python's `_match_spec` emits the literal, so a narrow native render
    // matches neither (angr-vmi2l.114).
    if matches!(cspec.conv, b's' | b'c') && cspec.modifier != LengthModifier::None {
        return Err(ProcedureError::Other(
            "%s/%c with a length modifier defers to Python".to_string(),
        ));
    }
    match cspec.conv {
        b'd' | b'i' => {
            let val = next_arg(args, arg_idx, "int arg")?;
            // Interpret as signed, narrowed to the modifier's width.
            let signed_val = signed_at_width(val, cspec.modifier.int_conv_bits(arch_bits));
            let formatted = if cspec.plus_sign && signed_val >= 0 {
                format!("+{signed_val}")
            } else if cspec.space_sign && signed_val >= 0 {
                format!(" {signed_val}")
            } else {
                format!("{signed_val}")
            };
            pad_and_push(
                output,
                formatted.as_bytes(),
                cspec.width,
                cspec.left_align,
                cspec.zero_pad,
                signed_val < 0,
            );
        }
        b'u' => {
            let unsigned_val = unsigned_operand(args, arg_idx, cspec, arch_bits, "uint arg")?;
            let formatted = format!("{unsigned_val}");
            pad_and_push(
                output,
                formatted.as_bytes(),
                cspec.width,
                cspec.left_align,
                cspec.zero_pad,
                false,
            );
        }
        b'x' | b'X' => {
            let unsigned_val = unsigned_operand(args, arg_idx, cspec, arch_bits, "hex arg")?;
            let mut formatted = if cspec.conv == b'x' {
                format!("{unsigned_val:x}")
            } else {
                format!("{unsigned_val:X}")
            };
            if cspec.hash_flag && unsigned_val != 0 {
                let prefix = if cspec.conv == b'x' { "0x" } else { "0X" };
                formatted = format!("{prefix}{formatted}");
            }
            pad_and_push(
                output,
                formatted.as_bytes(),
                cspec.width,
                cspec.left_align,
                cspec.zero_pad,
                false,
            );
        }
        b'o' => {
            let unsigned_val = unsigned_operand(args, arg_idx, cspec, arch_bits, "octal arg")?;
            let mut formatted = format!("{unsigned_val:o}");
            if cspec.hash_flag && unsigned_val != 0 {
                formatted = format!("0{formatted}");
            }
            pad_and_push(
                output,
                formatted.as_bytes(),
                cspec.width,
                cspec.left_align,
                cspec.zero_pad,
                false,
            );
        }
        b'c' => {
            let val = next_arg(args, arg_idx, "char arg")?;
            let ch = [val as u8];
            // `FormatString.replace` pads `%c` with the spec's `pad_chr`, so
            // `%03c` renders "00A" where glibc ignores the `0` flag
            // (angr-vmi2l.115).
            pad_and_push(
                output,
                &ch,
                cspec.width,
                cspec.left_align,
                cspec.zero_pad,
                false,
            );
        }
        b's' => {
            let str_addr = next_arg(args, arg_idx, "string arg")?;
            // Deliberately `MAX_STRING_SCAN`, not the enclosing format
            // string's `format_common::MAX_FORMAT_LEN`: a `%s` conversion
            // *argument* is a plain C string with no length bound, so it
            // belongs to the str-family cap per the deciding test in bd
            // memory `invariant-syscall-byte-cap-shared`. The two constants
            // are independently defined and only happen to be equal today —
            // a future security-driven reduction of one is expected to move
            // this scan without moving the format-string scan, and vice
            // versa. Pinned behaviourally by
            // `sprintf_tests::test_sprintf_percent_s_scan_bounded_by_max_string_scan`.
            let (s, null_found) =
                scan_concrete_bounded(state, str_addr, MAX_STRING_SCAN, "string")?;
            // Cap-without-null is an error here, unlike in
            // `format_common::read_format_string` where it is explicitly
            // tolerated for the format string itself. Python's
            // `format_parser.py::FormatString._get_str_at` measures a `%s`
            // argument with the `strlen` SimProcedure, which keeps doubling
            // its search window past `libc.max_str_len` and raises
            // `SimMemoryLimitError` at 0x10000 rather than truncating — so a
            // 4096-byte prefix would be a silently wrong render both for an
            // unterminated argument (Python errors) and for a merely long
            // one whose terminator sits between 4096 and 0x10000 (Python
            // renders it in full). Defer instead, as the sibling
            // strcpy/strcat/getopt/perror scans do (angr-6fg46).
            if !null_found {
                return Err(ProcedureError::MaxIterations(MAX_STRING_SCAN));
            }
            // `FormatString._get_str_at` loads exactly `prec` bytes for
            // `%.*s`, NUL included, where glibc stops at the NUL. The two
            // agree only when the string is at least `prec` long; a zero
            // precision is a zero-width load in Python (angr-vmi2l.115).
            let s = match cspec.precision {
                Some(prec) if prec == 0 || prec > s.len() => {
                    return Err(ProcedureError::Other(
                        "'%.*s' precision past the NUL defers to Python".to_string(),
                    ));
                }
                Some(prec) => &s[..prec],
                None => &s,
            };
            // `FormatString.replace` never pads `%s`: its `s` arm appends the
            // loaded bytes and ignores the width (angr-vmi2l.115).
            output.extend_from_slice(s);
        }
        b'p' => {
            // `_mod_spec` pairs length modifiers with integer and float
            // conversions only, so Python never matches `%lp` and friends.
            if cspec.modifier != LengthModifier::None {
                return Err(ProcedureError::Other(
                    "%p with a length modifier defers to Python".to_string(),
                ));
            }
            // `FormatString.replace` masks to the pointer width and renders
            // `f"{c_val:#x}"`, or glibc's "(nil)" for a null pointer; a
            // pointer type is unsigned, so there is no sign fold.
            let val = unsigned_at_width(next_arg(args, arg_idx, "pointer arg")?, arch_bits.min(64));
            let formatted = if val == 0 {
                "(nil)".to_string()
            } else {
                format!("{val:#x}")
            };
            pad_and_push(
                output,
                formatted.as_bytes(),
                cspec.width,
                cspec.left_align,
                cspec.zero_pad,
                false,
            );
        }
        b'n' => {
            // %n writes the number of chars written so far to an int pointer.
            // Deliberately NOT implemented natively: deferring to Python is
            // the faithful behavior. Python's format_parser.py::FormatString
            // .replace has no %n arm and falls through to
            // `raise SimProcedureError("Unimplemented format specifier 'n'")`.
            // A native write of the count would succeed where Python errors,
            // diverging from the engine we mirror. The fallback reproduces
            // Python exactly for free.
            return Err(ProcedureError::Other("%n not supported".to_string()));
        }
        _ => {
            // Unknown specifier — fall back to Python. This includes the
            // float specifiers %f/%e/%g: Python's format_parser.py::
            // FormatString.replace raises SimProcedureError on them, so a
            // native float formatter would diverge (succeed where Python
            // errors). Faithful behavior is to defer.
            return Err(ProcedureError::Other(format!(
                "unsupported format specifier '%{}'",
                cspec.conv as char
            )));
        }
    }

    Ok(())
}
