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

use crate::symbolic::RustBV;

const MAX_PRINTF_LEN: usize = 4096;

crate::declare_proc! {
    /// Native printf implementation.
    ///
    /// ```c
    /// int printf(const char *format, ...);
    /// ```
    ///
    /// printf is variadic, but we only read the (concrete) format string.
    /// Returns a non-negative value (number of characters printed).
    name = "printf",
    struct = NativePrintf,
    args = [fmt_addr: concrete],
    call |state| {
        // Read the format string byte-by-byte from memory
        let mut buf = Vec::new();
        for i in 0..MAX_PRINTF_LEN {
            match state.memory_load(fmt_addr + i as u64, 1) {
                Ok(bv) => {
                    if let Some(val) = bv.as_u64() {
                        let byte = val as u8;
                        if byte == 0 {
                            break;
                        }
                        buf.push(byte);
                    } else {
                        // Symbolic byte — stop reading
                        break;
                    }
                }
                Err(_) => break,
            }
        }

        // Append to stdout buffer
        state.write_stdout(&buf);

        let len = buf.len() as u128;
        Ok(Some(RustBV::concrete(if len == 0 { 1 } else { len }, 32)))
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
    name = "fprintf",
    struct = NativeFprintf,
    args = [stream: concrete, fmt_addr: concrete],
    call |state| {
        let fd = crate::procedures::stdio::read_fileno_for_stream(state, stream)?;
        if fd < 0 {
            return Ok(Some(RustBV::concrete((-1i64 as u64) as u128, 32)));
        }
        let buf = crate::procedures::strings::scan_concrete_lossy(state, fmt_addr, MAX_PRINTF_LEN);
        state.write_fd(fd as u32, &buf);
        Ok(Some(RustBV::concrete(buf.len() as u128, 32)))
    }
}

#[cfg(test)]
#[path = "printf_tests.rs"]
mod tests;
