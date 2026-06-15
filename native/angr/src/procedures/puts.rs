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
mod tests {
    use super::*;
    use crate::memory::Permission;

    #[test]
    fn test_puts_basic() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"hello\x00", Permission::RWX);

        let result = NativePuts
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(6)); // 5 + newline
        assert_eq!(state.stdout_buffer(), b"hello\n");
    }

    #[test]
    fn test_puts_empty_string() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"\x00", Permission::RWX);

        let result = NativePuts
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(1)); // just newline
        assert_eq!(state.stdout_buffer(), b"\n");
    }

    #[test]
    fn test_puts_multiple_calls() {
        let mut state = RustSimState::new("amd64").unwrap();
        state.map_memory_data(0x1000, b"abc\x00", Permission::RWX);
        state.map_memory_data(0x2000, b"def\x00", Permission::RWX);

        NativePuts
            .call(&mut state, &[RustBV::concrete(0x1000, 64)])
            .unwrap();
        NativePuts
            .call(&mut state, &[RustBV::concrete(0x2000, 64)])
            .unwrap();
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

    #[test]
    fn test_putchar_basic() {
        let mut state = RustSimState::new("amd64").unwrap();
        let result = NativePutchar
            .call(&mut state, &[RustBV::concrete(b'A' as u128, 32)])
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(b'A' as u64));
        assert_eq!(state.stdout_buffer(), b"A");
    }

    #[test]
    fn test_putchar_multiple() {
        let mut state = RustSimState::new("amd64").unwrap();
        NativePutchar
            .call(&mut state, &[RustBV::concrete(b'H' as u128, 32)])
            .unwrap();
        NativePutchar
            .call(&mut state, &[RustBV::concrete(b'i' as u128, 32)])
            .unwrap();
        assert_eq!(state.stdout_buffer(), b"Hi");
    }

    #[test]
    fn test_putchar_symbolic() {
        let mut state = RustSimState::new("amd64").unwrap();
        let ctx = state.solver().borrow();
        let sym = RustBV::symbolic(&ctx, "c", 32);
        drop(ctx);
        let result = NativePutchar.call(&mut state, &[sym]);
        assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
    }

    #[test]
    fn test_fputc_basic() {
        let mut state = RustSimState::new("amd64").unwrap();
        let result = NativeFputc
            .call(
                &mut state,
                &[RustBV::concrete(b'X' as u128, 32), RustBV::concrete(0, 64)],
            )
            .unwrap();
        assert_eq!(result.unwrap().as_u64(), Some(b'X' as u64));
        assert_eq!(state.stdout_buffer(), b"X");
    }
}
