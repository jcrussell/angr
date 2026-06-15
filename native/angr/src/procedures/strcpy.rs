//! Native strcpy/strncpy implementation.
//!
//! # Behavior
//!
//! - If any address is symbolic, falls back to Python
//! - If any source byte is symbolic, falls back to Python
//! - Maximum string length is 4096 bytes

use super::strings::{scan_concrete_bounded, scan_concrete_until_null};
use super::{NativeSimProcedure, ProcedureError, extract_concrete_arg};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

const MAX_STRLEN: usize = 4096;

/// Native strcpy implementation.
///
/// ```c
/// char *strcpy(char *dest, const char *src);
/// ```
pub struct NativeStrcpy;

impl NativeSimProcedure for NativeStrcpy {
    fn name(&self) -> &'static str {
        "strcpy"
    }

    fn num_args(&self) -> usize {
        2
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let dest = extract_concrete_arg(&args[0], "dest")?;
        let src = extract_concrete_arg(&args[1], "src")?;

        // Read source string up to (but not including) the null terminator.
        let buf = scan_concrete_until_null(state, src, MAX_STRLEN, "src")?;

        // Write to destination byte-by-byte, then the null terminator.
        for (i, &byte) in buf.iter().enumerate() {
            state.memory_store(
                dest.wrapping_add(i as u64),
                RustBV::concrete(byte as u128, 8),
            )?;
        }
        state.memory_store(
            dest.wrapping_add(buf.len() as u64),
            RustBV::concrete(0u128, 8),
        )?;

        Ok(Some(args[0].clone()))
    }
}

/// Native strncpy implementation.
///
/// ```c
/// char *strncpy(char *dest, const char *src, size_t n);
/// ```
pub struct NativeStrncpy;

impl NativeSimProcedure for NativeStrncpy {
    fn name(&self) -> &'static str {
        "strncpy"
    }

    fn num_args(&self) -> usize {
        3
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let dest = extract_concrete_arg(&args[0], "dest")?;
        let src = extract_concrete_arg(&args[1], "src")?;
        let n = extract_concrete_arg(&args[2], "n")?;

        if n > MAX_STRLEN as u64 {
            return Err(ProcedureError::MaxIterations(n as usize));
        }

        // Read up to n bytes from source, stopping early at null. Pre-null
        // bytes go in `buf` (null itself excluded); when null is found we
        // pad the rest of the n-byte window with zeros.
        let (mut buf, null_found) = scan_concrete_bounded(state, src, n as usize, "src")?;
        if null_found {
            buf.resize(n as usize, 0);
        }

        for (i, &byte) in buf.iter().enumerate() {
            state.memory_store(
                dest.wrapping_add(i as u64),
                RustBV::concrete(byte as u128, 8),
            )?;
        }

        Ok(Some(args[0].clone()))
    }
}

/// Native strdup implementation.
///
/// ```c
/// char *strdup(const char *s);
/// ```
///
/// Allocates a new string via heap_alloc, copies the source string
/// (including null terminator), and returns pointer to the new string.
pub struct NativeStrdup;

impl NativeSimProcedure for NativeStrdup {
    fn name(&self) -> &'static str {
        "strdup"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let src = extract_concrete_arg(&args[0], "s")?;

        // Read source string up to (but not including) the null terminator.
        let buf = scan_concrete_until_null(state, src, MAX_STRLEN, "src")?;

        // Allocate new buffer (strlen + 1 for null terminator).
        let new_addr = state.heap_alloc(buf.len() as u64 + 1);

        for (i, &byte) in buf.iter().enumerate() {
            state.memory_store(
                new_addr.wrapping_add(i as u64),
                RustBV::concrete(byte as u128, 8),
            )?;
        }
        state.memory_store(
            new_addr.wrapping_add(buf.len() as u64),
            RustBV::concrete(0u128, 8),
        )?;

        let bits = state.arch().bits();
        Ok(Some(RustBV::concrete(new_addr as u128, bits)))
    }
}

#[cfg(test)]
#[path = "strcpy_tests.rs"]
mod tests;
