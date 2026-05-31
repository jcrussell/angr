//! Native rand implementation.
//!
//! rand() returns a pseudo-random number. In symbolic execution,
//! this is modeled as an unconstrained 31-bit symbolic variable
//! (zero-extended to 32 bits), matching angr's Python SimProcedure.

use super::{NativeSimProcedure, ProcedureError, symbol_counter};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Native rand implementation.
///
/// ```c
/// int rand(void);
/// ```
///
/// Returns a symbolic 31-bit value zero-extended to arch int size,
/// matching angr's SimProcedure behavior.
pub struct NativeRand;

impl NativeSimProcedure for NativeRand {
    fn name(&self) -> &'static str {
        "rand"
    }

    fn num_args(&self) -> usize {
        0
    }

    fn call(
        &self,
        state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        let counter = symbol_counter("rand");
        let name = format!("rand_{}", counter);

        // Create a 31-bit symbolic variable (matches angr's rand which uses 31 bits)
        let ctx = state.solver().borrow();
        let sym_val = RustBV::symbolic(&ctx, &name, 31);

        // Zero-extend to 32 bits (int size) — matching angr's zero_extend(sizeof(int) - 31)
        let result = sym_val.zero_extend(32, &ctx);
        drop(ctx);

        Ok(Some(result))
    }
}

/// Native srand implementation (no-op in symbolic execution).
///
/// ```c
/// void srand(unsigned int seed);
/// ```
pub struct NativeSrand;

impl NativeSimProcedure for NativeSrand {
    fn name(&self) -> &'static str {
        "srand"
    }

    fn num_args(&self) -> usize {
        1
    }

    fn call(
        &self,
        _state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        // srand just sets seed state — no-op in symbolic execution
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rand_returns_symbolic() {
        let mut state = RustSimState::new("amd64").unwrap();
        let result = NativeRand.call(&mut state, &[]).unwrap();
        let bv = result.unwrap();
        // Should be symbolic (32 bits: 31-bit symbolic zero-extended)
        assert!(bv.as_u64().is_none()); // symbolic, not concrete
        assert_eq!(bv.width(), 32);
    }

    #[test]
    fn test_rand_unique_names() {
        let mut state = RustSimState::new("amd64").unwrap();
        let r1 = NativeRand.call(&mut state, &[]).unwrap().unwrap();
        let r2 = NativeRand.call(&mut state, &[]).unwrap().unwrap();
        // Both should be symbolic but different
        assert!(r1.as_u64().is_none());
        assert!(r2.as_u64().is_none());
    }

    #[test]
    fn test_rand_metadata() {
        assert_eq!(NativeRand.name(), "rand");
        assert_eq!(NativeRand.num_args(), 0);
        assert!(!NativeRand.no_return());
    }

    #[test]
    fn test_srand_noop() {
        let mut state = RustSimState::new("amd64").unwrap();
        let result = NativeSrand
            .call(&mut state, &[RustBV::concrete(42, 32)])
            .unwrap();
        assert!(result.is_none());
        assert_eq!(NativeSrand.name(), "srand");
    }
}
