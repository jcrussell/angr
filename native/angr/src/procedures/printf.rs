//! Native printf / fprintf implementations, and the raw vprintf / vfprintf.
//!
//! printf and fprintf substitute their conversions through the shared
//! [`format_string`] core, the same one sprintf/snprintf use, so their output
//! and return value match Python's `FormatString.replace`. Every case that
//! core refuses (a symbolic format byte or argument, and the specifiers and
//! flags listed in the `sprintf.rs` module doc) defers to Python
//! (angr-vmi2l.109).
//!
//! vprintf and vfprintf write the RAW format string: the `va_list` is
//! unmodeled, and their Python counterparts write the format unsubstituted
//! too (see [`NativeVsprintf`](super::sprintf::NativeVsprintf)). A symbolic
//! format byte defers to Python, which writes it as is.

use super::format_common::read_format_string;
use super::sprintf::{MAX_VARARGS, format_string};
use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Native printf implementation.
///
/// ```c
/// int printf(const char *format, ...);
/// ```
///
/// Returns the number of bytes written, like Python `printf`.
pub(crate) struct NativePrintf;

impl NativeSimProcedure for NativePrintf {
    fn name(&self) -> &'static str {
        "printf"
    }

    fn num_args(&self) -> usize {
        1 + MAX_VARARGS // format + varargs
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let fmt_addr = extract_concrete_arg(&args[0], "format")?;
        let fmt = read_format_string(state, fmt_addr)?;
        let output = format_string(state, &fmt, &args[1..])?;

        // A refusal (fd 1 dup2'd onto a bounded-symbolic-content fd, now
        // demoted) bounces to Python — see FileSystem::write.
        if !state.write_stdout(&output) {
            return Err(ProcedureError::Other(
                "printf to stdout with symbolic content falls back to Python (demoted)".to_string(),
            ));
        }
        Ok(Some(RustBV::concrete(output.len() as u128, 32)))
    }
}

/// Native fprintf implementation.
///
/// ```c
/// int fprintf(FILE *stream, const char *format, ...);
/// ```
///
/// The stream variant of [`NativePrintf`]: resolves `stream->_fileno` and
/// writes the formatted output to that fd. Returns the number of bytes
/// written, or -1 on a closed/negative fd (matching Python `fprintf`, which
/// returns -1 when `simfd is None`). A symbolic FILE* defers to Python.
pub(crate) struct NativeFprintf;

impl NativeSimProcedure for NativeFprintf {
    fn name(&self) -> &'static str {
        "fprintf"
    }

    fn num_args(&self) -> usize {
        2 + MAX_VARARGS // stream + format + varargs
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let stream = extract_concrete_arg(&args[0], "stream")?;
        let fmt_addr = extract_concrete_arg(&args[1], "format")?;
        let fd = super::stream_ops::read_fileno_or_demote_all(state, stream)?;
        if fd < 0 {
            return Ok(Some(RustBV::concrete((-1i64 as u64) as u128, 32)));
        }
        let fmt = read_format_string(state, fmt_addr)?;
        let output = format_string(state, &fmt, &args[2..])?;
        // A refusal means the fd carried bounded symbolic content (now
        // demoted) — bounce to Python at the choke point (angr-0xyq2 Phase 2).
        if !state.write_fd(fd as u32, &output) {
            return Err(ProcedureError::Other(format!(
                "fprintf to fd={fd} with symbolic content falls back to Python (demoted)"
            )));
        }
        Ok(Some(RustBV::concrete(output.len() as u128, 32)))
    }
}

crate::declare_proc! {
    /// Native vprintf implementation: writes the raw format string.
    ///
    /// ```c
    /// int vprintf(const char *format, va_list ap);
    /// ```
    ///
    /// A symbolic format byte defers to Python. Returns the number of bytes
    /// written.
    name = "vprintf",
    struct = NativeVprintf,
    args = [fmt_addr: concrete],
    call |state| {
        let buf = read_format_string(state, fmt_addr)?;
        // Refusal contract — see NativePrintf.
        if !state.write_stdout(&buf) {
            return Err(ProcedureError::Other(
                "vprintf to stdout with symbolic content falls back to Python (demoted)"
                    .to_string(),
            ));
        }
        Ok(Some(RustBV::concrete(buf.len() as u128, 32)))
    }
}

crate::declare_proc! {
    /// Native vfprintf implementation: writes the raw format string to the
    /// stream's fd.
    ///
    /// ```c
    /// int vfprintf(FILE *stream, const char *format, va_list ap);
    /// ```
    ///
    /// The raw-write counterpart of [`NativeFprintf`]. Returns -1 on a
    /// closed/negative fd.
    name = "vfprintf",
    struct = NativeVfprintf,
    args = [stream: concrete, fmt_addr: concrete],
    call |state| {
        let fd = crate::procedures::stream_ops::read_fileno_or_demote_all(state, stream)?;
        if fd < 0 {
            return Ok(Some(RustBV::concrete((-1i64 as u64) as u128, 32)));
        }
        let buf = read_format_string(state, fmt_addr)?;
        // Refusal contract — see NativeFprintf.
        if !state.write_fd(fd as u32, &buf) {
            return Err(ProcedureError::Other(format!(
                "vfprintf to fd={fd} with symbolic content falls back to Python (demoted)"
            )));
        }
        Ok(Some(RustBV::concrete(buf.len() as u128, 32)))
    }
}

test_submod!("printf_tests.rs" => tests);
