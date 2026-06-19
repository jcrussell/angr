//! Native strcpy/strncpy implementation.
//!
//! # Behavior
//!
//! - If any address is symbolic, falls back to Python
//! - If any source byte is symbolic, falls back to Python
//! - Maximum string length is 4096 bytes

use super::strings::{scan_concrete_bounded, scan_concrete_until_null};
use super::{ProcedureError, extract_concrete_arg};
use crate::symbolic::RustBV;

const MAX_STRLEN: usize = 4096;

crate::declare_proc! {
    /// Native strcpy implementation.
    ///
    /// ```c
    /// char *strcpy(char *dest, const char *src);
    /// ```
    ///
    /// `dest` is declared `bv` so the original pointer BV is returned
    /// verbatim; it is extracted to a concrete u64 in the body.
    name = "strcpy",
    struct = NativeStrcpy,
    args = [dest_bv: bv, src: concrete],
    call |state| {
        let dest = extract_concrete_arg(&dest_bv, "dest")?;

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

        Ok(Some(dest_bv))
    }
}

crate::declare_proc! {
    /// Native strncpy implementation.
    ///
    /// ```c
    /// char *strncpy(char *dest, const char *src, size_t n);
    /// ```
    ///
    /// `dest` is declared `bv` so the original pointer BV is returned
    /// verbatim; it is extracted to a concrete u64 in the body.
    name = "strncpy",
    struct = NativeStrncpy,
    args = [dest_bv: bv, src: concrete, n: concrete],
    call |state| {
        let dest = extract_concrete_arg(&dest_bv, "dest")?;

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

        Ok(Some(dest_bv))
    }
}

crate::declare_proc! {
    /// Native strdup implementation.
    ///
    /// ```c
    /// char *strdup(const char *s);
    /// ```
    ///
    /// Allocates a new string via heap_alloc, copies the source string
    /// (including null terminator), and returns pointer to the new string.
    name = "strdup",
    struct = NativeStrdup,
    args = [src: concrete],
    call |state| {
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
