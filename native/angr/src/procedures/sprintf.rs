//! Native sprintf/snprintf implementations.
//!
//! Handles %s, %d, %i, %u, %x, %X, %o, %c, %p, %% format specifiers
//! with width, zero-padding, and left-alignment flags. Falls back to
//! Python for symbolic format strings or arguments.

use super::format_common::{MAX_FORMAT_LEN, parse_length_modifier, parse_width_digits};
use super::strings::{scan_concrete_bounded, write_cstr};
use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_OUTPUT_LEN: usize = 4096;

/// Read a null-terminated string from memory at `addr` (up to `MAX_FORMAT_LEN`
/// bytes; exhausting the cap without a null is not an error). A symbolic byte
/// or out-of-bounds read propagates as an `Err` and falls back to Python.
fn read_string(state: &mut RustSimState, addr: u64) -> Result<Vec<u8>, ProcedureError> {
    let (buf, _null_found) = scan_concrete_bounded(state, addr, MAX_FORMAT_LEN, "string")?;
    Ok(buf)
}

/// Format arguments according to a printf-style format string.
///
/// `args` is the slice of variadic arguments (after dest/format/size).
/// Returns the formatted output bytes and the number of variadic args consumed.
/// Python's `FormatSpecifier.signed` is buggy — `getattr(ty, "size", False)`
/// returns the truthy size int even for unsigned specs — so
/// format_parser.py::FormatString.replace sign-folds EVERY int conversion
/// (`if c_val >= 2^(bits-1): c_val -= 2^bits`). For unsigned specs
/// (%u/%x/%X/%o) with the high bit clear the fold is a no-op and native output
/// matches; with the high bit set Python renders a negative value that native
/// can't cheaply reproduce (Rust `{:x}` prints two's-complement bits, Python
/// prints `-<abs hex>`), so those must defer to Python. `masked` is already
/// masked to `bits`. See `format-unsigned-highbit-no-native-parity`
/// (angr-3i88a).
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

fn format_string(
    state: &mut RustSimState,
    fmt: &[u8],
    args: &[RustBV],
) -> Result<Vec<u8>, ProcedureError> {
    let mut output = Vec::new();
    let mut arg_idx: usize = 0;
    let mut i = 0;
    // `l`/`z`/`t` are `long`-width, i.e. 32-bit on ILP32 targets — see
    // `LengthModifier::int_conv_bits`.
    let arch_bits = state.arch().bits();

    while i < fmt.len() {
        if fmt[i] != b'%' {
            output.push(fmt[i]);
            i += 1;
            continue;
        }
        i += 1; // skip '%'
        if i >= fmt.len() {
            break;
        }

        // Handle %%
        if fmt[i] == b'%' {
            output.push(b'%');
            i += 1;
            continue;
        }

        // Parse flags
        let mut left_align = false;
        let mut zero_pad = false;
        let mut plus_sign = false;
        let mut space_sign = false;
        let mut hash_flag = false;
        loop {
            if i >= fmt.len() {
                break;
            }
            match fmt[i] {
                b'-' => left_align = true,
                b'0' => zero_pad = true,
                b'+' => plus_sign = true,
                b' ' => space_sign = true,
                b'#' => hash_flag = true,
                _ => break,
            }
            i += 1;
        }
        if left_align {
            zero_pad = false; // '-' overrides '0'
        }

        // Parity defer: Python's format_parser.py only recognizes the bare '0'
        // (zero-pad) flag. Its _match_spec has no arm for '-', '+', ' ', or '#',
        // so it fails to match the specifier entirely, emits a literal '%', and
        // does NOT consume the corresponding variadic arg. Native honoring these
        // flags would therefore produce different bytes AND a different
        // arg-consumption count than vanilla angr — a real cross-engine parity
        // gap (angr-1yge9.2). Defer to Python so both engines agree. Bare
        // '0'+width ("%05d") is matched by both parsers and stays native. The
        // downstream flag handling below is retained (unreachable while this
        // guard stands) so a future format_parser.py fix can drop just this
        // block. See `format-string-parity-defers`; do NOT "fix" by changing
        // Python's parser — that alters vanilla angr semantics for all users.
        if left_align || plus_sign || space_sign || hash_flag {
            return Err(ProcedureError::Other(
                "'-'/'+'/' '/'#' conversion flags defer to Python".to_string(),
            ));
        }

        // Parse width. '*' dynamic width diverges from Python: format_parser.py's
        // _match_spec has no '*' arm, so extract_components swallows the '%*'
        // without consuming a width arg, shifting every later variadic arg by
        // one. Native can't cheaply reproduce that arg-shift; defer to Python for
        // faithful parity. See `format-star-width-no-native-parity` (angr-3i88a).
        if i < fmt.len() && fmt[i] == b'*' {
            return Err(ProcedureError::Other(
                "'*' dynamic width defers to Python".to_string(),
            ));
        }
        // Clamp against MAX_OUTPUT_LEN here, before `width` is used as
        // pad_and_push's padding-loop bound: parse_width_digits only
        // saturates to usize::MAX on overflow, which does not stop a short
        // digit run like "%9999999999d" from encoding a near-MAX width. The
        // `output.len() > MAX_OUTPUT_LEN` check below only fires AFTER
        // pad_and_push's loop returns, so it can't bound the loop itself —
        // unclamped this is an unbounded-allocation / OOM loop (angr-mi56k).
        let (width, advanced) = parse_width_digits(fmt, i);
        let width = width.min(MAX_OUTPUT_LEN);
        i += advanced;

        // Parse precision.
        let mut precision: Option<usize> = None;
        if i < fmt.len() && fmt[i] == b'.' {
            i += 1;
            if i < fmt.len() && fmt[i] == b'*' {
                // '.*' precision (arg-supplied) is matched correctly by Python.
                if arg_idx >= args.len() {
                    return Err(ProcedureError::SymbolicArgument(
                        "precision arg".to_string(),
                    ));
                }
                let p = extract_concrete_arg(&args[arg_idx], "precision")?;
                precision = Some(p as usize);
                arg_idx += 1;
                i += 1;
            } else {
                // '.N' digit precision diverges: format_parser.py's _match_spec
                // mis-slices the consumed '.', dropping the actual conversion
                // letter, so FormatString.replace raises SimProcedureError and
                // the state errors. Native previously truncated/ignored the
                // precision and continued — succeeding where Python errors.
                // Defer for parity. See
                // `format-digit-precision-no-native-parity` (angr-3i88a).
                return Err(ProcedureError::Other(
                    "'.N' digit precision defers to Python".to_string(),
                ));
            }
        }

        // Parse length modifier
        let (modifier, m_adv) = parse_length_modifier(fmt, i);
        i += m_adv;

        if i >= fmt.len() {
            break;
        }

        // Parse conversion specifier
        let spec = fmt[i];
        i += 1;

        match spec {
            b'd' | b'i' => {
                if arg_idx >= args.len() {
                    return Err(ProcedureError::SymbolicArgument("int arg".to_string()));
                }
                let val = extract_concrete_arg(&args[arg_idx], &format!("arg{arg_idx}"))?;
                arg_idx += 1;
                // Interpret as signed, narrowed to the modifier's width.
                let signed_val = signed_at_width(val, modifier.int_conv_bits(arch_bits));
                let formatted = if plus_sign && signed_val >= 0 {
                    format!("+{signed_val}")
                } else if space_sign && signed_val >= 0 {
                    format!(" {signed_val}")
                } else {
                    format!("{signed_val}")
                };
                pad_and_push(
                    &mut output,
                    formatted.as_bytes(),
                    width,
                    left_align,
                    zero_pad,
                    signed_val < 0,
                );
            }
            b'u' => {
                if arg_idx >= args.len() {
                    return Err(ProcedureError::SymbolicArgument("uint arg".to_string()));
                }
                let val = extract_concrete_arg(&args[arg_idx], &format!("arg{arg_idx}"))?;
                arg_idx += 1;
                let bits = modifier.int_conv_bits(arch_bits);
                let unsigned_val = unsigned_at_width(val, bits);
                if unsigned_high_bit_set(unsigned_val, bits) {
                    return Err(ProcedureError::Other(
                        "unsigned high-bit value defers to Python".to_string(),
                    ));
                }
                let formatted = format!("{unsigned_val}");
                pad_and_push(
                    &mut output,
                    formatted.as_bytes(),
                    width,
                    left_align,
                    zero_pad,
                    false,
                );
            }
            b'x' | b'X' => {
                if arg_idx >= args.len() {
                    return Err(ProcedureError::SymbolicArgument("hex arg".to_string()));
                }
                let val = extract_concrete_arg(&args[arg_idx], &format!("arg{arg_idx}"))?;
                arg_idx += 1;
                let bits = modifier.int_conv_bits(arch_bits);
                let unsigned_val = unsigned_at_width(val, bits);
                if unsigned_high_bit_set(unsigned_val, bits) {
                    return Err(ProcedureError::Other(
                        "unsigned high-bit value defers to Python".to_string(),
                    ));
                }
                let mut formatted = if spec == b'x' {
                    format!("{unsigned_val:x}")
                } else {
                    format!("{unsigned_val:X}")
                };
                if hash_flag && unsigned_val != 0 {
                    let prefix = if spec == b'x' { "0x" } else { "0X" };
                    formatted = format!("{prefix}{formatted}");
                }
                pad_and_push(
                    &mut output,
                    formatted.as_bytes(),
                    width,
                    left_align,
                    zero_pad,
                    false,
                );
            }
            b'o' => {
                if arg_idx >= args.len() {
                    return Err(ProcedureError::SymbolicArgument("octal arg".to_string()));
                }
                let val = extract_concrete_arg(&args[arg_idx], &format!("arg{arg_idx}"))?;
                arg_idx += 1;
                let bits = modifier.int_conv_bits(arch_bits);
                let unsigned_val = unsigned_at_width(val, bits);
                if unsigned_high_bit_set(unsigned_val, bits) {
                    return Err(ProcedureError::Other(
                        "unsigned high-bit value defers to Python".to_string(),
                    ));
                }
                let mut formatted = format!("{unsigned_val:o}");
                if hash_flag && unsigned_val != 0 {
                    formatted = format!("0{formatted}");
                }
                pad_and_push(
                    &mut output,
                    formatted.as_bytes(),
                    width,
                    left_align,
                    zero_pad,
                    false,
                );
            }
            b'c' => {
                if arg_idx >= args.len() {
                    return Err(ProcedureError::SymbolicArgument("char arg".to_string()));
                }
                let val = extract_concrete_arg(&args[arg_idx], &format!("arg{arg_idx}"))?;
                arg_idx += 1;
                let ch = [val as u8];
                pad_and_push(&mut output, &ch, width, left_align, false, false);
            }
            b's' => {
                if arg_idx >= args.len() {
                    return Err(ProcedureError::SymbolicArgument("string arg".to_string()));
                }
                let str_addr = extract_concrete_arg(&args[arg_idx], &format!("arg{arg_idx}"))?;
                arg_idx += 1;
                let s = read_string(state, str_addr)?;
                let s = if let Some(prec) = precision {
                    if prec < s.len() { &s[..prec] } else { &s }
                } else {
                    &s
                };
                pad_and_push(&mut output, s, width, left_align, false, false);
            }
            b'p' => {
                // %p always diverges from Python: native emitted a "0x" prefix
                // (format!("0x{val:x}")) while format_parser.py emits bare hex
                // (f"{c_val:x}"), and Python additionally sign-folds bit-63-set
                // pointers. Defer for parity. See `format-p-no-native-parity`
                // (angr-3i88a).
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
                // Python exactly for free. See bd memory `format-n-no-native-parity`.
                return Err(ProcedureError::Other("%n not supported".to_string()));
            }
            _ => {
                // Unknown specifier — fall back to Python. This includes the
                // float specifiers %f/%e/%g: Python's format_parser.py::
                // FormatString.replace raises SimProcedureError on them, so a
                // native float formatter would diverge (succeed where Python
                // errors). Faithful behavior is to defer. See bd memory
                // `format-float-no-native-parity`.
                return Err(ProcedureError::Other(format!(
                    "unsupported format specifier '%{}'",
                    spec as char
                )));
            }
        }

        if output.len() > MAX_OUTPUT_LEN {
            return Err(ProcedureError::MaxIterations(MAX_OUTPUT_LEN));
        }
    }

    Ok(output)
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

/// Native sprintf implementation.
///
/// ```c
/// int sprintf(char *str, const char *format, ...);
/// ```
pub(crate) struct NativeSprintf;

impl NativeSimProcedure for NativeSprintf {
    fn name(&self) -> &'static str {
        "sprintf"
    }

    fn num_args(&self) -> usize {
        8 // dest + format + up to 6 variadic args
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let dest = extract_concrete_arg(&args[0], "dest")?;
        let fmt_addr = extract_concrete_arg(&args[1], "format")?;

        let fmt = read_string(state, fmt_addr)?;
        let varargs = &args[2..];
        let output = format_string(state, &fmt, varargs)?;

        // Write output to destination, then the null terminator
        write_cstr(state, dest, &output)?;

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(output.len() as u128, bits)))
    }
}

/// Native asprintf implementation.
///
/// ```c
/// int asprintf(char **strp, const char *format, ...);
/// ```
///
/// Like sprintf, but allocates the destination buffer (`heap_alloc`, matching
/// Python `asprintf` inline-calling `malloc`), writes the buffer pointer to
/// `*strp`, and returns the formatted length (excluding the NUL). Reuses the
/// shared [`format_string`] core (DRY with sprintf/snprintf) and falls back to
/// Python on symbolic format strings/args via the same `ProcedureError` paths.
pub(crate) struct NativeAsprintf;

impl NativeSimProcedure for NativeAsprintf {
    fn name(&self) -> &'static str {
        "asprintf"
    }

    fn num_args(&self) -> usize {
        8 // strp + format + up to 6 variadic args
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let strp = extract_concrete_arg(&args[0], "strp")?;
        let fmt_addr = extract_concrete_arg(&args[1], "format")?;

        let fmt = read_string(state, fmt_addr)?;
        let varargs = &args[2..];
        let output = format_string(state, &fmt, varargs)?;

        // Allocate output.len() + 1 bytes (data + NUL) and write the string.
        let dst = state.heap_alloc(output.len() as u64 + 1);
        write_cstr(state, dst, &output)?;

        // Write the allocated buffer pointer back to *strp (honors mem endness).
        let bits = state.arch().bits();
        state.memory_store(strp, RustBV::concrete(dst as u128, bits))?;

        Ok(Some(RustBV::concrete(output.len() as u128, bits)))
    }
}

/// Native snprintf implementation.
///
/// ```c
/// int snprintf(char *str, size_t size, const char *format, ...);
/// ```
pub(crate) struct NativeSnprintf;

impl NativeSimProcedure for NativeSnprintf {
    fn name(&self) -> &'static str {
        "snprintf"
    }

    fn num_args(&self) -> usize {
        9 // dest + size + format + up to 6 variadic args
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let dest = extract_concrete_arg(&args[0], "dest")?;
        let size = extract_concrete_arg(&args[1], "size")? as usize;
        let fmt_addr = extract_concrete_arg(&args[2], "format")?;

        let fmt = read_string(state, fmt_addr)?;
        let varargs = &args[3..];
        let output = format_string(state, &fmt, varargs)?;

        // Write output to destination, respecting size limit
        if size > 0 {
            let write_len = output.len().min(size - 1);
            // Write truncated output + always null-terminate at write_len.
            write_cstr(state, dest, &output[..write_len])?;
        }

        // Return would-have-been length (not truncated)
        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(output.len() as u128, bits)))
    }
}

/// Native vsnprintf implementation.
///
/// ```c
/// int vsnprintf(char *str, size_t size, const char *format, va_list ap);
/// ```
///
/// This deliberately does **not** format. It matches Python angr's
/// `procedures/libc/vsnprintf.py`, which is a no-op stub: `size == 0`
/// returns 0; otherwise it stores a single NUL byte at `str` and returns 1.
/// Reading the `va_list` properly is arch-specific (x86-64 SysV
/// `reg_save_area`/`gp_offset`) and unmodeled by angr, so a "real" formatter
/// here would diverge from the Python engine and explore different symbolic
/// states — violating the faithful-reimplementation invariant. See bd memory
/// `avoid-vsnprintf-real-formatting`.
pub(crate) struct NativeVsnprintf;

impl NativeSimProcedure for NativeVsnprintf {
    fn name(&self) -> &'static str {
        "vsnprintf"
    }

    fn num_args(&self) -> usize {
        4 // str + size + format + va_list (format/va_list unused by the stub)
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let dest = extract_concrete_arg(&args[0], "str")?;
        let size = extract_concrete_arg(&args[1], "size")?;

        let bits = state.arch().bits();
        if size == 0 {
            return Ok(Some(RustBV::concrete(0u128, bits)));
        }

        // Match the Python stub: a single NUL terminator at `str`, return 1.
        write_cstr(state, dest, &[])?;
        Ok(Some(RustBV::concrete(1u128, bits)))
    }
}

/// Native vsprintf implementation.
///
/// ```c
/// int vsprintf(char *str, const char *format, va_list ap);
/// ```
///
/// Like the `vprintf = printf` / `vfprintf = fprintf` aliases (printf.rs), this
/// deliberately does **not** %-substitute: reading the `va_list` is arch-specific
/// (x86-64 SysV `reg_save_area`) and unmodeled by angr, so a "real" formatter via
/// the [`format_string`] core would explore divergent symbolic states (see bd
/// memory `avoid-vsnprintf-real-formatting`). Instead it copies the RAW format
/// string into `str` (NUL-terminated) and returns its length — matching Python
/// `vsprintf` (strcpy + strlen). Falls back to Python on a symbolic dest/format
/// *address* or a symbolic format *byte* via `read_string`/`extract_concrete_arg`.
pub(crate) struct NativeVsprintf;

impl NativeSimProcedure for NativeVsprintf {
    fn name(&self) -> &'static str {
        "vsprintf"
    }

    fn num_args(&self) -> usize {
        3 // str + format + va_list (va_list unused by the raw-write impl)
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let dest = extract_concrete_arg(&args[0], "str")?;
        let fmt_addr = extract_concrete_arg(&args[1], "format")?;

        // Raw format string, no %-substitution (va_list is unmodeled).
        let fmt = read_string(state, fmt_addr)?;
        write_cstr(state, dest, &fmt)?;

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(fmt.len() as u128, bits)))
    }
}

#[cfg(test)]
#[path = "sprintf_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable. The module `deny` overrides lib.rs's crate-wide `cfg_attr(test, allow(..))`, hence the explicit opt-out"
)]
mod sprintf_tests;
