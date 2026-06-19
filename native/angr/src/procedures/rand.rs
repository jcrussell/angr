//! Native rand implementation.
//!
//! rand() returns a pseudo-random number. In symbolic execution,
//! this is modeled as an unconstrained 31-bit symbolic variable
//! (zero-extended to 32 bits), matching angr's Python SimProcedure.

use super::symbol_counter;
use crate::symbolic::RustBV;

crate::declare_proc! {
    /// rand: return a symbolic 31-bit value zero-extended to arch int size,
    /// matching angr's SimProcedure behavior.
    ///
    /// ```c
    /// int rand(void);
    /// ```
    name = "rand",
    struct = NativeRand,
    args = [],
    call |state| {
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

crate::declare_proc! {
    /// srand: no-op in symbolic execution. The seed is ignored (declared `bv`
    /// so a symbolic seed does NOT trigger a Python fallback — it is dropped
    /// either way).
    ///
    /// ```c
    /// void srand(unsigned int seed);
    /// ```
    name = "srand",
    struct = NativeSrand,
    args = [_seed: bv],
    call |_state| {
        // srand just sets seed state — no-op in symbolic execution
        Ok(None)
    }
}

#[cfg(test)]
#[path = "rand_tests.rs"]
mod tests;
