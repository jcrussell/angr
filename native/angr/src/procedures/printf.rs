//! Native printf implementation (simplified).
//!
//! For address-based find/avoid exploration, printf's exact output is
//! irrelevant. We return a plausible non-negative value without parsing
//! the format string or writing to stdout.
//!
//! Falls back to Python when callable predicates check stdout content.

use crate::state::RustSimState;
use crate::symbolic::RustBV;
use super::{NativeSimProcedure, ProcedureError};

/// Native printf implementation.
///
/// ```c
/// int printf(const char *format, ...);
/// ```
///
/// Returns a non-negative value (number of characters printed).
/// We return 1 as a minimal plausible value.
pub struct NativePrintf;

impl NativeSimProcedure for NativePrintf {
    fn name(&self) -> &'static str {
        "printf"
    }

    fn num_args(&self) -> usize {
        1  // Variadic, but we only check the format string address
    }

    fn call(
        &self,
        _state: &mut RustSimState,
        args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        // Verify format string address is concrete (sanity check)
        let _fmt_addr = args[0].as_u64().ok_or_else(|| {
            ProcedureError::SymbolicArgument("format".to_string())
        })?;

        // Return 1 (plausible printf return value)
        let width = 32; // printf returns int (32-bit)
        Ok(Some(RustBV::concrete(1, width)))
    }
}
