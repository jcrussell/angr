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

use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_PUTS_LEN: usize = 4096;

/// Native puts implementation.
///
/// ```c
/// int puts(const char *s);
/// ```
///
/// Returns a non-negative value on success (we return strlen(s) + 1).
pub struct NativePuts;

impl NativeSimProcedure for NativePuts {
    fn name(&self) -> &'static str {
        "puts"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let s_addr = extract_concrete_arg(&args[0], "s")?;

        // Read the string byte-by-byte from memory
        let mut buf = Vec::new();
        for i in 0..MAX_PUTS_LEN {
            match state.memory_load(s_addr + i as u64, 1) {
                Ok(bv) => {
                    if let Some(val) = bv.as_u64() {
                        let byte = val as u8;
                        if byte == 0 {
                            break;
                        }
                        buf.push(byte);
                    } else {
                        // Symbolic byte — stop reading, append what we have
                        break;
                    }
                }
                Err(_) => break,
            }
        }

        // Append string + newline to stdout buffer
        state.write_stdout(&buf);
        state.write_stdout(b"\n");

        let len = buf.len() as u128 + 1; // +1 for newline
        Ok(Some(RustBV::concrete(len, 32)))
    }
}

/// Native putchar implementation.
///
/// ```c
/// int putchar(int c);
/// ```
///
/// Writes one byte to stdout. Returns the character written (as int).
pub struct NativePutchar;

impl NativeSimProcedure for NativePutchar {
    fn name(&self) -> &'static str {
        "putchar"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let c = extract_concrete_arg(&args[0], "c")?;
        let byte = (c & 0xFF) as u8;
        state.write_stdout(&[byte]);
        Ok(Some(RustBV::concrete(byte as u128, 32)))
    }
}

/// Native fputc implementation.
///
/// ```c
/// int fputc(int c, FILE *stream);
/// ```
///
/// Writes one byte to the stream. Stream argument is ignored (treated as stdout).
pub struct NativeFputc;

impl NativeSimProcedure for NativeFputc {
    fn name(&self) -> &'static str {
        "fputc"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        // Delegate to putchar (ignore FILE* stream arg)
        NativePutchar.call(state, &args[..1])
    }
}

/// Native putc implementation (alias for fputc).
pub struct NativePutc;

impl NativeSimProcedure for NativePutc {
    fn name(&self) -> &'static str {
        "putc"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        NativeFputc.call(state, args)
    }
}

#[cfg(test)]
#[path = "puts_tests.rs"]
mod tests;
