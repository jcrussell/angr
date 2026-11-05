//! Conversion *rendering* for the native sprintf family.
//!
//! The second half of [`super::sprintf`]'s `format_string` loop: given a
//! [`ConversionSpec`] already parsed by [`super::sprintf_spec`], pull the
//! variadic argument it names and append its bytes to the output buffer.
//! Every Python-parity defer that depends on the argument *value* rather than
//! the specifier shape — an unsigned operand with its high bit set, an
//! unterminated `%s` string — is decided here.

use super::sprintf_spec::ConversionSpec;
use super::strings::{MAX_STRING_SCAN, scan_concrete_bounded};
use super::{ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Python's `FormatSpecifier.signed` is buggy — `getattr(ty, "size", False)`
/// returns the truthy size int even for unsigned specs — so
/// format_parser.py::FormatString.replace sign-folds EVERY int conversion
/// (`if c_val >= 2^(bits-1): c_val -= 2^bits`). For unsigned specs
/// (%u/%x/%X/%o) with the high bit clear the fold is a no-op and native output
/// matches; with the high bit set Python renders a negative value that native
/// can't cheaply reproduce (Rust `{:x}` prints two's-complement bits, Python
/// prints `-<abs hex>`), so those must defer to Python. `masked` is already
/// masked to `bits` (angr-3i88a).
fn unsigned_high_bit_set(masked: u64, bits: u32) -> bool {
    masked >> (bits - 1) != 0
}

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
    *arg_idx += 1;
    Ok(val)
}

/// [`next_arg`] narrowed to the spec's conversion width for the unsigned
/// `%u`/`%x`/`%X`/`%o` family, with the shared high-bit parity defer applied.
///
/// All four arms need exactly this; see [`unsigned_high_bit_set`] for why a
/// high bit set means deferring to Python rather than rendering natively.
fn unsigned_operand(
    args: &[RustBV],
    arg_idx: &mut usize,
    cspec: &ConversionSpec,
    arch_bits: u32,
    what: &str,
) -> Result<u64, ProcedureError> {
    let val = next_arg(args, arg_idx, what)?;
    let bits = cspec.modifier.int_conv_bits(arch_bits);
    let unsigned_val = unsigned_at_width(val, bits);
    if unsigned_high_bit_set(unsigned_val, bits) {
        return Err(ProcedureError::Other(
            "unsigned high-bit value defers to Python".to_string(),
        ));
    }
    Ok(unsigned_val)
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
            pad_and_push(output, &ch, cspec.width, cspec.left_align, false, false);
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
            let s = if let Some(prec) = cspec.precision {
                if prec < s.len() { &s[..prec] } else { &s }
            } else {
                &s
            };
            pad_and_push(output, s, cspec.width, cspec.left_align, false, false);
        }
        b'p' => {
            // %p always diverges from Python: native emitted a "0x" prefix
            // (format!("0x{val:x}")) while format_parser.py emits bare hex
            // (f"{c_val:x}"), and Python additionally sign-folds bit-63-set
            // pointers. Defer for parity (angr-3i88a).
            return Err(ProcedureError::Other("%p defers to Python".to_string()));
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
