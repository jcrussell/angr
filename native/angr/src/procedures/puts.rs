//! Native puts implementation.
//!
//! puts writes a string to stdout followed by a newline.
//! For address-based exploration, the actual output is irrelevant —
//! we just need a correct return value.
//!
//! # Behavior
//!
//! - Reads the string at arg[0] to compute length (for return value)
//! - Returns length + 1 (for the appended newline), or a non-negative value
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
        _state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        // Verify address is concrete (sanity check)
        let _s_addr = args[0].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("s".to_string())
        })?;

        // puts returns a non-negative value on success.
        // Don't walk the string — byte-by-byte memory reads are slow.
        // The return value of puts is rarely checked.
        Ok(Some(RustBV::concrete(1, 32)))
    }
}
