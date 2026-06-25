//! Native access() implementation.
//!
//! `access(const char *path, int mode)` checks file accessibility and returns
//! 0 on success or -1 on error. Symbolic execution has no real filesystem to
//! consult, so angr's libc SimProcedure (`angr/procedures/libc/access.py`)
//! models the result as a fresh symbolic `int` constrained to be either 0 or
//! -1:
//!
//! ```python
//! ret = claripy.BVS("access", self.arch.sizeof["int"])
//! self.state.add_constraints(claripy.Or(ret == 0, ret == -1))
//! return ret
//! ```
//!
//! (The `linux_kernel/access.py` variant actually walks `state.fs`; we mirror
//! the simpler libc proc here — the filesystem path stays a Python fallback.)
//!
//! The native side previously had no `access`, so a PLT libc call round-tripped
//! to Python. This mirrors the libc proc directly: both `path` and `mode` are
//! taken as raw `bv` and discarded (Python never inspects them, so there is no
//! reason to force concretization and fall back), and the return is a fresh
//! symbolic 32-bit C int constrained to {0, -1}.

use super::symbol_counter;
use crate::symbolic::RustBV;

crate::declare_proc! {
    /// ```c
    /// int access(const char *path, int mode);
    /// ```
    /// Returns a symbolic 32-bit int constrained to be either 0 or -1.
    name = "access",
    struct = NativeAccess,
    args = [_path: bv, _mode: bv],
    call |state| {
        let counter = symbol_counter("access");
        let name = format!("access_{}", counter);

        // sizeof(int) == 32 on all supported arches (matches rand.rs/system.rs).
        let constraint = {
            let ctx = state.solver().borrow();
            let ret = RustBV::symbolic(&ctx, &name, 32);
            let is_ok = ret.eq(&RustBV::concrete(0, 32), &ctx);
            // -1 as a 32-bit two's-complement int.
            let is_err = ret.eq(&RustBV::concrete(0xFFFF_FFFF, 32), &ctx);
            let or = is_ok.or(&is_err, &ctx);
            (ret, or)
        };
        let (ret, or) = constraint;
        // add_constraint borrows the solver, so build the constraint inside the
        // borrow scope above, drop it, then apply (mirrors read.rs/fgets.rs).
        state.add_constraint(or);

        Ok(Some(ret))
    }
}

#[cfg(test)]
#[path = "access_tests.rs"]
mod tests;
