//! Native strlen implementation.
//!
//! strlen returns the length of a null-terminated string, not including
//! the null terminator.
//!
//! # Behavior
//!
//! - If the address is symbolic, falls back to Python
//! - If any byte in the string is symbolic, falls back to Python
//! - Maximum string length is 4096 bytes (configurable)

use crate::state::RustSimState;
use crate::symbolic::RustBV;
use super::{NativeSimProcedure, ProcedureError};

/// Maximum string length before falling back to Python.
const MAX_STRLEN: usize = 4096;

/// Native strlen implementation.
///
/// ```c
/// size_t strlen(const char *s);
/// ```
///
/// Returns the number of bytes before the first null byte.
pub struct NativeStrlen;

impl NativeSimProcedure for NativeStrlen {
    fn name(&self) -> &'static str {
        "strlen"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        // Get the string address
        let addr = args[0].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("addr".to_string())
        })?;

        // Scan for null terminator
        let mut length: u64 = 0;

        for i in 0..MAX_STRLEN as u64 {
            let byte_addr = addr.wrapping_add(i);

            // Load a single byte
            let byte_val = state.memory_load(byte_addr, 1)
                .map_err(|e| ProcedureError::MemoryError(e.to_string()))?;

            // Check if byte is concrete
            let byte = byte_val.as_u64().ok_or_else(|| {
                ProcedureError::SymbolicArgument(format!("memory byte at 0x{:x}", byte_addr))
            })?;

            // Check for null terminator
            if byte == 0 {
                length = i;
                break;
            }

            // Update length if we haven't found null yet
            if i == MAX_STRLEN as u64 - 1 {
                return Err(ProcedureError::MaxIterations(MAX_STRLEN));
            }
        }

        // Return length as pointer-sized value
        let ptr_bits = state.arch().bits();
        Ok(Some(RustBV::concrete(length as u128, ptr_bits)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Permission;

    #[test]
    fn test_strlen_basic() {
        let mut state = RustSimState::new("amd64").unwrap();

        // Map memory with "test\0"
        let data = b"test\x00";
        state.map_memory_data(0x1000, data, Permission::RWX);

        let proc = NativeStrlen;
        let result = proc.call(&mut state, &[RustBV::concrete(0x1000, 64)]).unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(4));
    }

    #[test]
    fn test_strlen_empty() {
        let mut state = RustSimState::new("amd64").unwrap();

        // Map memory with just null byte
        let data = b"\x00";
        state.map_memory_data(0x1000, data, Permission::RWX);

        let proc = NativeStrlen;
        let result = proc.call(&mut state, &[RustBV::concrete(0x1000, 64)]).unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(0));
    }

    #[test]
    fn test_strlen_longer_string() {
        let mut state = RustSimState::new("amd64").unwrap();

        // Map memory with "hello world\0"
        let data = b"hello world\x00";
        state.map_memory_data(0x1000, data, Permission::RWX);

        let proc = NativeStrlen;
        let result = proc.call(&mut state, &[RustBV::concrete(0x1000, 64)]).unwrap();

        assert_eq!(result.unwrap().as_u64(), Some(11));
    }

    #[test]
    fn test_strlen_symbolic_addr() {
        let mut state = RustSimState::new("amd64").unwrap();

        // Create a symbolic address
        let ctx = state.solver().borrow();
        let sym_addr = RustBV::symbolic(&ctx, "addr", 64);
        drop(ctx);

        let proc = NativeStrlen;
        let result = proc.call(&mut state, &[sym_addr]);

        // Should fail with symbolic argument error
        assert!(matches!(result, Err(ProcedureError::SymbolicArgument(_))));
    }
}
