//! Native fortify-source `_chk` wrappers for the printf formatting family.
//!
//! glibc's `_FORTIFY_SOURCE` build mode redirects `printf`/`sprintf`/`snprintf`
//! to `__*_chk` variants that take extra leading/middle arguments the compiler
//! injects (a `flag` word, and for the buffer-writing variants a `slen`
//! destination-size word) and abort via `__chk_fail` when the write provably
//! overflows the destination object.
//!
//! Matching Python angr (`procedures/libc/{printf,snprintf,sprintf}.py`, where
//! each `__*_chk` subclasses the base and drops the injected args), the native
//! variants **ignore** the `flag`/`slen` words and forward to the base routine.
//! Emulating the runtime bound check would diverge from the Python engine and
//! prune paths angr otherwise explores, so we deliberately do not. These
//! wrappers add zero formatting logic — they delegate to the existing native
//! `printf`/`sprintf`/`snprintf` impls (DRY) and only drop the extra args.
//!
//! glibc `_chk` ABI (the args the binary actually passes), vs the base call we
//! forward to:
//!
//! * `int __printf_chk(int flag, const char *fmt, ...)`
//!   → `printf(fmt, ...)`            (drop `flag`)
//! * `int __sprintf_chk(char *s, int flag, size_t slen, const char *fmt, ...)`
//!   → `sprintf(s, fmt, ...)`        (drop `flag`, `slen`)
//! * `int __snprintf_chk(char *s, size_t maxlen, int flag, size_t slen,
//!                       const char *fmt, ...)`
//!   → `snprintf(s, maxlen, fmt, ...)` (drop `flag`, `slen`)
//! * `int __fprintf_chk(FILE *fp, int flag, const char *fmt, ...)`
//!   → `fprintf(fp, fmt, ...)`        (drop `flag`)
//! * `int __vsnprintf_chk(char *s, size_t maxlen, int flag, size_t slen,
//!                        const char *fmt, va_list ap)`
//!   → `vsnprintf(s, maxlen, fmt, ap)` (drop `flag`, `slen`)

use super::printf::{NativeFprintf, NativePrintf};
use super::sprintf::{NativeSnprintf, NativeSprintf, NativeVsnprintf};
use super::{NativeSimProcedure, ProcedureError};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Max variadic args the base sprintf/snprintf procs consume.
const MAX_VARARGS: usize = 6;

/// `int __printf_chk(int flag, const char *format, ...)`.
///
/// Forwards to native `printf`, dropping the leading `flag` (matches Python
/// angr's `__printf_chk`). Native `printf` only reads the (concrete) format
/// string and ignores the variadic args, so we request just `flag` + `fmt`.
pub(crate) struct NativePrintfChk;

impl NativeSimProcedure for NativePrintfChk {
    fn name(&self) -> &'static str {
        "__printf_chk"
    }

    fn num_args(&self) -> usize {
        2 // flag + format
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        // Drop `flag`; native printf reads args[0] as the format string.
        NativePrintf.call(state, &args[1..])
    }
}

/// `int __sprintf_chk(char *s, int flag, size_t slen, const char *format, ...)`.
///
/// Forwards to native `sprintf`, dropping the injected `flag` and `slen`
/// (matches Python angr's `__sprintf_chk`).
pub(crate) struct NativeSprintfChk;

impl NativeSimProcedure for NativeSprintfChk {
    fn name(&self) -> &'static str {
        "__sprintf_chk"
    }

    fn num_args(&self) -> usize {
        4 + MAX_VARARGS // dest + flag + slen + format + varargs
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        // sprintf expects [dest, format, varargs...]; drop flag (args[1]) and
        // slen (args[2]).
        let mut forwarded = Vec::with_capacity(args.len() - 2);
        forwarded.push(args[0].clone());
        forwarded.extend_from_slice(&args[3..]);
        NativeSprintf.call(state, &forwarded)
    }
}

/// `int __snprintf_chk(char *s, size_t maxlen, int flag, size_t slen,
///                     const char *format, ...)`.
///
/// Forwards to native `snprintf`, dropping the injected `flag` and `slen`
/// (matches Python angr's `__snprintf_chk`).
pub(crate) struct NativeSnprintfChk;

impl NativeSimProcedure for NativeSnprintfChk {
    fn name(&self) -> &'static str {
        "__snprintf_chk"
    }

    fn num_args(&self) -> usize {
        5 + MAX_VARARGS // dest + maxlen + flag + slen + format + varargs
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        // snprintf expects [dest, size, format, varargs...]; drop flag (args[2])
        // and slen (args[3]).
        let mut forwarded = Vec::with_capacity(args.len() - 2);
        forwarded.push(args[0].clone());
        forwarded.push(args[1].clone());
        forwarded.extend_from_slice(&args[4..]);
        NativeSnprintf.call(state, &forwarded)
    }
}

/// `int __fprintf_chk(FILE *fp, int flag, const char *format, ...)`.
///
/// Forwards to native `fprintf`, dropping the injected `flag` (matches Python
/// angr's `__fprintf_chk`). Native `fprintf` resolves the stream's fd and
/// writes the (concrete) raw format string, ignoring the variadic args, so we
/// request just `fp` + `flag` + `fmt`.
pub(crate) struct NativeFprintfChk;

impl NativeSimProcedure for NativeFprintfChk {
    fn name(&self) -> &'static str {
        "__fprintf_chk"
    }

    fn num_args(&self) -> usize {
        3 // stream + flag + format
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        // fprintf expects [stream, format]; drop flag (args[1]).
        let mut forwarded = Vec::with_capacity(args.len() - 1);
        forwarded.push(args[0].clone());
        forwarded.extend_from_slice(&args[2..]);
        NativeFprintf.call(state, &forwarded)
    }
}

/// `int __vsnprintf_chk(char *s, size_t maxlen, int flag, size_t slen,
///                      const char *format, va_list ap)`.
///
/// Forwards to native `vsnprintf`, dropping the injected `flag` and `slen`
/// (matches Python angr's `__vsnprintf_chk`). The base `vsnprintf` is the
/// degenerate stub (size==0 → 0, else writes a single NUL and returns 1), so
/// only `s` and `maxlen` are consumed — the `format`/`va_list` words are
/// passed through but unused.
pub(crate) struct NativeVsnprintfChk;

impl NativeSimProcedure for NativeVsnprintfChk {
    fn name(&self) -> &'static str {
        "__vsnprintf_chk"
    }

    fn num_args(&self) -> usize {
        6 // dest + maxlen + flag + slen + format + va_list
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        // vsnprintf expects [dest, size, format, va_list]; drop flag (args[2])
        // and slen (args[3]).
        let mut forwarded = Vec::with_capacity(args.len() - 2);
        forwarded.push(args[0].clone());
        forwarded.push(args[1].clone());
        forwarded.extend_from_slice(&args[4..]);
        NativeVsnprintf.call(state, &forwarded)
    }
}

#[cfg(test)]
#[path = "fortify_printf_tests.rs"]
mod tests;
