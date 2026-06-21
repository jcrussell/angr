//! Native puts implementation.
//!
//! puts writes a string to stdout followed by a newline.
//! Reads the string from memory and appends it (plus newline) to the
//! state's stdout buffer for predicate evaluation.
//!
//! # Behavior
//!
//! - Reads the string at arg\[0\] byte-by-byte until NUL or MAX_PUTS_LEN
//! - Appends string + newline to state.stdout_buffer
//! - Returns length + 1 (for the appended newline)
//! - Falls back to Python if address is symbolic

use crate::symbolic::RustBV;

const MAX_PUTS_LEN: usize = 4096;

crate::declare_proc! {
    /// Native puts implementation.
    ///
    /// ```c
    /// int puts(const char *s);
    /// ```
    ///
    /// Returns a non-negative value on success (we return strlen(s) + 1).
    name = "puts",
    struct = NativePuts,
    args = [s: concrete],
    call |state| {
        // Read the concrete prefix byte-by-byte; a symbolic byte, failed load,
        // or the cap quietly stops the scan (puts prints what it has).
        let buf = crate::procedures::strings::scan_concrete_lossy(state, s, MAX_PUTS_LEN);

        // Append string + newline to stdout buffer
        state.write_stdout(&buf);
        state.write_stdout(b"\n");

        let len = buf.len() as u128 + 1; // +1 for newline
        Ok(Some(RustBV::concrete(len, 32)))
    }
}

crate::declare_proc! {
    /// Native putchar implementation.
    ///
    /// ```c
    /// int putchar(int c);
    /// ```
    ///
    /// Writes one byte to stdout. Returns the character written (as int).
    name = "putchar",
    struct = NativePutchar,
    args = [c: concrete],
    call |state| {
        let byte = (c & 0xFF) as u8;
        state.write_stdout(&[byte]);
        Ok(Some(RustBV::concrete(byte as u128, 32)))
    }
}

crate::declare_proc! {
    /// Native fputc implementation.
    ///
    /// ```c
    /// int fputc(int c, FILE *stream);
    /// ```
    ///
    /// Writes one byte to the stream. Stream argument is ignored (treated as
    /// stdout); declared `bv` so a symbolic FILE* does not trigger a Python
    /// fallback, matching the prior hand-rolled behavior.
    name = "fputc",
    struct = NativeFputc,
    args = [c: concrete, _stream: bv],
    call |state| {
        let byte = (c & 0xFF) as u8;
        state.write_stdout(&[byte]);
        Ok(Some(RustBV::concrete(byte as u128, 32)))
    }
}

crate::declare_proc! {
    /// Native putc implementation (alias for fputc).
    name = "putc",
    struct = NativePutc,
    args = [c: concrete, _stream: bv],
    call |state| {
        let byte = (c & 0xFF) as u8;
        state.write_stdout(&[byte]);
        Ok(Some(RustBV::concrete(byte as u128, 32)))
    }
}

#[cfg(test)]
#[path = "puts_tests.rs"]
mod tests;
