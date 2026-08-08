//! Native printf implementation (simplified).
//!
//! Reads the format string from memory and appends it to the state's
//! stdout buffer. Does NOT perform format string substitution — just
//! writes the raw format string. This is sufficient for predicates that
//! check for fixed output strings (the common CTF pattern).
//!
//! Falls back to Python when the format *address* is symbolic
//! (`extract_concrete_arg`). A symbolic format *byte* is handled
//! differently from the rest of the family: rather than falling back, the
//! byte loop stops at the first symbolic byte, writes the concrete prefix,
//! and returns success. This is intentional (printf only needs the raw
//! string for fixed-output predicates), but it is the one asymmetry in the
//! family — scanf/sprintf raise `SymbolicArgument` on the first symbolic
//! byte instead. See the format-string worked example in
//! `docs/extending-angr/simprocedures.rst`.

use super::format_common::MAX_FORMAT_LEN;
use crate::symbolic::RustBV;

crate::declare_proc! {
    /// Native printf implementation.
    ///
    /// ```c
    /// int printf(const char *format, ...);
    /// ```
    ///
    /// printf is variadic, but we only read the (concrete) format string.
    /// Returns a non-negative value (number of characters printed).
    ///
    /// Aliased to `vprintf(const char *format, va_list ap)`: since the native
    /// impl never substitutes (it writes the raw format string), the trailing
    /// `va_list` is irrelevant and `format` is arg 0 either way, so vprintf is
    /// byte-for-byte identical to printf — no separate impl needed (DRY).
    name = "printf",
    struct = NativePrintf,
    args = [fmt_addr: concrete],
    aliases = ["vprintf"],
    call |state| {
        // Read the format string byte-by-byte from memory. A symbolic byte,
        // null terminator, or failed load quietly stops the scan (see the
        // module docstring's note on printf's intentional symbolic-byte
        // asymmetry) — exactly `scan_concrete_lossy`'s contract, so reuse it
        // rather than re-rolling the loop (matches NativeFprintf below).
        let buf = crate::procedures::strings::scan_concrete_lossy(state, fmt_addr, MAX_FORMAT_LEN);

        // Append to stdout buffer. A refusal (fd 1 dup2'd onto a bounded-
        // symbolic-content fd, now demoted) bounces to Python — see
        // FileSystem::write.
        if !state.write_stdout(&buf) {
            return Err(crate::procedures::ProcedureError::Other(
                "printf to stdout with symbolic content falls back to Python (demoted)"
                    .to_string(),
            ));
        }

        // Number of characters written. C/glibc and Python both return 0 for
        // an empty format string (printf("") == 0), so return the raw length.
        let len = buf.len() as u128;
        Ok(Some(RustBV::concrete(len, 32)))
    }
}

crate::declare_proc! {
    /// Native fprintf implementation (simplified).
    ///
    /// ```c
    /// int fprintf(FILE *stream, const char *format, ...);
    /// ```
    ///
    /// The stream variant of [`NativePrintf`]: resolves `stream->_fileno` and
    /// writes the RAW format string (no substitution) to that fd via
    /// `write_fd`, mirroring NativePrintf's stdout write and NativeFputc's fd
    /// resolution. Like printf, a symbolic format *byte* stops the scan and the
    /// concrete prefix is written (via `scan_concrete_lossy`); a symbolic
    /// FILE* or format *address* falls back to Python. Returns the number of
    /// bytes written, or -1 on a closed/negative fd (matching Python `fprintf`,
    /// which returns -1 when `simfd is None`).
    ///
    /// Aliased to `vfprintf(FILE *stream, const char *format, va_list ap)`:
    /// like the printf/vprintf pair, the native impl writes the raw format
    /// string without substitution, so the trailing `va_list` is irrelevant
    /// and `stream`/`format` are args 0/1 either way — vfprintf is identical
    /// to fprintf (DRY).
    name = "fprintf",
    struct = NativeFprintf,
    args = [stream: concrete, fmt_addr: concrete],
    aliases = ["vfprintf"],
    call |state| {
        let fd = match crate::procedures::stdio::read_fileno_for_stream(state, stream) {
            Ok(fd) => fd,
            Err(e) => {
                // Unresolvable fd on a write path: any bounded symbolic
                // file could be the target (angr-0xyq2 A4; O(1) when none
                // attached).
                state.file_system().demote_all_symbolic_content();
                return Err(e);
            }
        };
        if fd < 0 {
            return Ok(Some(RustBV::concrete((-1i64 as u64) as u128, 32)));
        }
        let buf = crate::procedures::strings::scan_concrete_lossy(state, fmt_addr, MAX_FORMAT_LEN);
        // A refusal means the fd carried bounded symbolic content (now
        // demoted) — bounce to Python (angr-0xyq2 Phase 2 choke point).
        if !state.write_fd(fd as u32, &buf) {
            return Err(crate::procedures::ProcedureError::Other(format!(
                "fprintf to fd={fd} with symbolic content falls back to Python (demoted)"
            )));
        }
        Ok(Some(RustBV::concrete(buf.len() as u128, 32)))
    }
}

test_submod!("printf_tests.rs" => tests);
