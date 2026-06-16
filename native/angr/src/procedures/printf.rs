//! Native printf implementation (simplified).
//!
//! Reads the format string from memory and appends it to the state's
//! stdout buffer. Does NOT perform format string substitution — just
//! writes the raw format string. This is sufficient for predicates that
//! check for fixed output strings (the common CTF pattern).
//!
//! Falls back to Python when the format address is symbolic.

use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_PRINTF_LEN: usize = 4096;

/// Native printf implementation.
///
/// ```c
/// int printf(const char *format, ...);
/// ```
///
/// Returns a non-negative value (number of characters printed).
pub struct NativePrintf;

impl NativeSimProcedure for NativePrintf {
    fn name(&self) -> &'static str {
        "printf"
    }

    fn num_args(&self) -> usize {
        1 // Variadic, but we only read the format string
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let fmt_addr = extract_concrete_arg(&args[0], "format")?;

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

#[cfg(test)]
#[path = "printf_tests.rs"]
mod tests;
