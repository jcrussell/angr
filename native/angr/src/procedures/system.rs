//! Native system() implementation.
//!
//! `system(const char *command)` runs a shell command and returns its exit
//! status. Symbolic execution cannot run a real shell, so angr's Python
//! SimProcedure (`angr/procedures/libc/system.py`) models the result as an
//! unconstrained 8-bit return code zero-extended to `sizeof(int)`:
//!
//! ```python
//! retcode = self.state.solver.Unconstrained("system_returncode", 8, ...)
//! return retcode.zero_extend(self.arch.sizeof["int"] - 8)
//! ```
//!
//! The native side previously had no `system`, so a PLT libc call round-tripped
//! to Python. This mirrors the Python proc directly: the `command` pointer is
//! taken as a raw `bv` and discarded (parity holds for concrete and symbolic
//! pointers alike — Python never inspects it, so there is no reason to force
//! concretization and fall back), and the return is a fresh 8-bit symbolic BV
//! zero-extended to the 32-bit C int width (matching rand.rs, which likewise
//! zero-extends an int-returning symbolic result to 32 bits).

use super::symbol_counter;
use crate::symbolic::RustBV;

crate::declare_proc! {
    /// ```c
    /// int system(const char *command);
    /// ```
    /// Returns an unconstrained 8-bit exit status zero-extended to 32-bit int.
    name = "system",
    struct = NativeSystem,
    args = [_command: bv],
    call |state| {
        let counter = symbol_counter("system");
        let name = format!("system_returncode_{counter}");

        let ctx = state.solver().borrow();
        // 8-bit unconstrained return code, zero-extended to the 32-bit C int
        // width (sizeof(int) == 32 on all supported arches; matches rand.rs).
        let sym_val = RustBV::symbolic(&ctx, &name, 8);
        let result = sym_val.zero_extend(32, &ctx);
        drop(ctx);

        Ok(Some(result))
    }
}

#[cfg(test)]
#[path = "system_tests.rs"]
mod tests;
