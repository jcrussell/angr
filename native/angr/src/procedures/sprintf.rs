//! Native sprintf/snprintf implementations.
//!
//! Formats %s, %d, %i, %u, %x, %X, %o, %c and %% natively, with an
//! explicit width and the bare '0' (zero-pad) flag.
//!
//! Everything else defers to Python — not because it is unimplemented,
//! but because native output would diverge from `format_parser.py`,
//! which is the engine this mirrors (see `format-string-parity-defers`):
//!
//! * `%p` — Python emits bare hex and sign-folds bit-63-set pointers
//!   (angr-3i88a).
//! * the `-`, `+`, ' ' and `#` conversion flags — Python's `_match_spec`
//!   has no arm for them, so it emits a literal '%' and consumes no
//!   variadic arg (angr-1yge9.2). The downstream `left_align` /
//!   `plus_sign` / `space_sign` / `hash_flag` handling below is retained
//!   but unreachable while that guard stands.
//! * `*` dynamic width; `.N` digit precision; `.*` precision that is
//!   not the first thing after the '%' (angr-6cp06.8).
//! * `%n`, the float specifiers, and any unknown specifier.
//! * a symbolic format string or argument.

use super::arch_word;
use super::format_common::read_format_string;
use super::sprintf_conv::render_conversion;
use super::sprintf_spec::parse_conversion_spec;
use super::strings::write_cstr;
use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Cap on the bytes a single `format_string` call may produce.
///
/// Also clamps a specifier's parsed width in
/// [`parse_conversion_spec`],
/// which is why it is visible to the sibling modules.
pub(super) const MAX_OUTPUT_LEN: usize = 4096;

/// Max variadic args the sprintf/snprintf family requests from the caller.
///
/// A native proc declares a fixed `num_args()`, so this is the cut-off past
/// which a `%`-specifier finds no argument to consume. `fortify_printf.rs`'s
/// `__sprintf_chk`/`__snprintf_chk` wrappers forward their variadic tail to
/// these base procs, so they must request the same count — they import this
/// constant rather than redeclaring it, since a wrapper asking for fewer would
/// silently truncate the forwarded list with no compiler or test signal
/// (angr-6cp06.10).
pub(crate) const MAX_VARARGS: usize = 6;

/// Format arguments according to a printf-style format string.
///
/// `args` is the slice of variadic arguments (after dest/format/size).
/// Returns the formatted output bytes.
///
/// The two halves of the per-specifier work live in sibling modules:
/// [`parse_conversion_spec`]
/// decides what a `%...` means and
/// [`render_conversion`] emits it.
/// This loop owns only the literal bytes, `%%`, and the output-length cap.
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

        // A format string that ends inside a specifier yields no spec; stop
        // formatting and keep what was built, as the `i >= fmt.len()` break
        // above does.
        let Some(cspec) = parse_conversion_spec(fmt, &mut i, args, &mut arg_idx)? else {
            break;
        };
        render_conversion(state, &cspec, args, &mut arg_idx, arch_bits, &mut output)?;

        if output.len() > MAX_OUTPUT_LEN {
            return Err(ProcedureError::MaxIterations(MAX_OUTPUT_LEN));
        }
    }

    Ok(output)
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
        2 + MAX_VARARGS // dest + format + varargs
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let dest = extract_concrete_arg(&args[0], "dest")?;
        let fmt_addr = extract_concrete_arg(&args[1], "format")?;

        let fmt = read_format_string(state, fmt_addr)?;
        let varargs = &args[2..];
        let output = format_string(state, &fmt, varargs)?;

        // Write output to destination, then the null terminator
        write_cstr(state, dest, &output)?;

        Ok(Some(arch_word(state, output.len() as u64)))
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
        2 + MAX_VARARGS // strp + format + varargs
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let strp = extract_concrete_arg(&args[0], "strp")?;
        let fmt_addr = extract_concrete_arg(&args[1], "format")?;

        let fmt = read_format_string(state, fmt_addr)?;
        let varargs = &args[2..];
        let output = format_string(state, &fmt, varargs)?;

        // Allocate output.len() + 1 bytes (data + NUL) and write the string.
        let dst = state.heap_alloc(output.len() as u64 + 1);
        write_cstr(state, dst, &output)?;

        // Write the allocated buffer pointer back to *strp (honors mem endness).
        state.memory_store(strp, arch_word(state, dst))?;

        Ok(Some(arch_word(state, output.len() as u64)))
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
        3 + MAX_VARARGS // dest + size + format + varargs
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let dest = extract_concrete_arg(&args[0], "dest")?;
        let size = extract_concrete_arg(&args[1], "size")? as usize;
        let fmt_addr = extract_concrete_arg(&args[2], "format")?;

        let fmt = read_format_string(state, fmt_addr)?;
        let varargs = &args[3..];
        let output = format_string(state, &fmt, varargs)?;

        // Write output to destination, respecting size limit
        if size > 0 {
            // overflow-ok: size >= 1 is guaranteed by the enclosing `if size > 0`.
            let write_len = output.len().min(size - 1);
            // Write truncated output + always null-terminate at write_len.
            write_cstr(state, dest, &output[..write_len])?;
        }

        // Return would-have-been length (not truncated)
        Ok(Some(arch_word(state, output.len() as u64)))
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

        if size == 0 {
            return Ok(Some(arch_word(state, 0u64)));
        }

        // Match the Python stub: a single NUL terminator at `str`, return 1.
        write_cstr(state, dest, &[])?;
        Ok(Some(arch_word(state, 1u64)))
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
/// *address* or a symbolic format *byte* via
/// [`read_format_string`]/[`extract_concrete_arg`].
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
        let fmt = read_format_string(state, fmt_addr)?;
        write_cstr(state, dest, &fmt)?;

        Ok(Some(arch_word(state, fmt.len() as u64)))
    }
}

test_submod!("sprintf_tests.rs" => sprintf_tests);
