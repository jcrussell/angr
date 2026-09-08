//! Native rand implementation.
//!
//! rand() returns a pseudo-random number. In symbolic execution,
//! this is modeled as an unconstrained 31-bit symbolic variable
//! (zero-extended to 32 bits), matching angr's Python SimProcedure.

use super::fresh_zero_extended_symbol;

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
        // 31-bit symbolic value (matches angr's rand, which uses 31 bits)
        // zero-extended to the 32-bit C int width — angr spells the same thing
        // as `zero_extend(sizeof(int) - 31)`.
        Ok(Some(fresh_zero_extended_symbol(state, "rand", 31, 32)))
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

test_submod!("rand_tests.rs" => tests);
