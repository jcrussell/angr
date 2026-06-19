//! Native memset implementation.
//!
//! memset fills a memory region with a constant byte value.
//!
//! # Behavior
//!
//! - If the address is symbolic, falls back to Python
//! - If the size is symbolic, falls back to Python
//! - A symbolic byte `value` is handled natively: the low 8 bits are stored
//!   (symbolically) into every byte of the region — no Python fallback.
//! - Maximum size is 1MB (configurable)

use super::{ProcedureError, extract_concrete_arg};
use crate::symbolic::RustBV;

/// Maximum memset size before falling back to Python.
const MAX_MEMSET_SIZE: u64 = 1024 * 1024;

crate::declare_proc! {
    /// Native memset implementation.
    ///
    /// ```c
    /// void *memset(void *s, int c, size_t n);
    /// ```
    ///
    /// Fills n bytes at s with byte value c. Returns s. `dest` is declared
    /// `bv` (not `concrete`) so the original pointer BV can be returned
    /// verbatim; it is extracted to a concrete u64 in the body. `value` is
    /// also `bv`: a concrete byte takes the fast 8-byte-chunk path, while a
    /// symbolic byte is stored (low 8 bits) into every byte natively.
    name = "memset",
    struct = NativeMemset,
    args = [dest_bv: bv, value_bv: bv, size: concrete],
    call |state| {
        let dest = extract_concrete_arg(&dest_bv, "dest")?;

        if size > MAX_MEMSET_SIZE {
            return Err(ProcedureError::Other(format!(
                "memset size {} exceeds maximum {}",
                size, MAX_MEMSET_SIZE
            )));
        }

        if size == 0 {
            return Ok(Some(dest_bv));
        }

        // Build the 8-bit fill byte and a 64-bit chunk (the byte repeated 8
        // times). For a concrete value both are concrete; for a symbolic
        // value the byte is the low 8 bits and the chunk concatenates 8
        // copies of it (endianness-agnostic since every byte is identical).
        let (byte_bv, chunk_bv) = match value_bv.as_u64() {
            Some(value) => {
                let byte_val = (value & 0xFF) as u8;
                let fill_8 = {
                    let mut val: u64 = 0;
                    for i in 0..8 {
                        val |= (byte_val as u64) << (i * 8);
                    }
                    val as u128
                };
                (
                    RustBV::concrete(byte_val as u128, 8),
                    RustBV::concrete(fill_8, 64),
                )
            }
            None => {
                let ctx = state.solver().borrow();
                let byte = value_bv.extract(7, 0, &ctx);
                let parts: [RustBV; 8] = core::array::from_fn(|_| byte.clone());
                let chunk = RustBV::concat_balanced(&parts, &ctx);
                (byte, chunk)
            }
        };

        // Fill memory using 8-byte chunks where possible.
        let mut offset: u64 = 0;
        while offset + 8 <= size {
            state.memory_store(dest.wrapping_add(offset), chunk_bv.clone())?;
            offset += 8;
        }
        // Handle remaining bytes
        while offset < size {
            state.memory_store(dest.wrapping_add(offset), byte_bv.clone())?;
            offset += 1;
        }

        // Return dest pointer
        Ok(Some(dest_bv))
    }
}

#[cfg(test)]
#[path = "memset_tests.rs"]
mod tests;
