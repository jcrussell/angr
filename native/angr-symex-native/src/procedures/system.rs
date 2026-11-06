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
//! zero-extended to the 32-bit C int width.
//!
//! `Unconstrained` is not `BVS`: it returns `BVV(0)` unless
//! `SYMBOLIC_INITIAL_VALUES` is set, so without the option the native return
//! is a concrete 0, as in stub.rs (angr-vmi2l.127). rand.py calls `BVS`, so
//! rand.rs stays symbolic whatever the options.

use super::fresh_zero_extended_symbol;
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
        // sizeof(int) == 32 on all supported arches.
        if !state.has_option(crate::state::SYMBOLIC_INITIAL_VALUES) {
            return Ok(Some(RustBV::concrete(0, 32)));
        }
        Ok(Some(fresh_zero_extended_symbol(
            state,
            "system_returncode",
            8,
            32,
            &["api", "system"],
        )))
    }
}

test_submod!("system_tests.rs" => tests);
