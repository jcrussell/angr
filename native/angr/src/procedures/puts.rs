//! Native puts implementation.
//!
//! puts writes a string to stdout followed by a newline.
//! Reads the string from memory and appends it (plus newline) to the
//! state's stdout buffer for predicate evaluation.
//!
//! # Behavior
//!
//! - Reads the string at arg[0] byte-by-byte until NUL or MAX_PUTS_LEN
//! - Appends string + newline to state.stdout_buffer
//! - Returns length + 1 (for the appended newline)
//! - Falls back to Python if address is symbolic

use crate::state::RustSimState;
use crate::symbolic::RustBV;
use super::{NativeSimProcedure, ProcedureError};

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
        let s_addr = args[0].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("s".to_string())
        })?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;

    #[test]
    fn test_puts_basic() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);

        let result = NativePuts.call(&mut state, &[RustBV::concrete(0x1000, 64)]).unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(6)); // 5 + newline
        assert_eq!(state.stdout_buffer(), b"hello\n");
    }

    #[test]
    fn test_puts_empty_string() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"\x00", Permission::RWX);

        let result = NativePuts.call(&mut state, &[RustBV::concrete(0x1000, 64)]).unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(1)); // just newline
        assert_eq!(state.stdout_buffer(), b"\n");
    }

    #[test]
    fn test_puts_multiple_calls() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"abc\x00", Permission::RWX);
        state.map_memory_data(0x2000, b"def\x00", Permission::RWX);

        NativePuts.call(&mut state, &[RustBV::concrete(0x1000, 64)]).unwrap();
        NativePuts.call(&mut state, &[RustBV::concrete(0x2000, 64)]).unwrap();
        assert_eq!(state.stdout_buffer(), b"abc\ndef\n");
    }

    #[test]
    fn test_puts_symbolic_addr() {
        let mut state = RustSimState::new("amd64").unwrap();
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "addr", 64);
        drop(ctx);
        let result = NativePuts.call(&mut state, &[sym]);
        assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
    }
}
