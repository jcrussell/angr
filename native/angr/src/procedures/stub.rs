//! Native counterpart of angr's `ReturnUnconstrained` stub SimProcedure.
//!
//! `angr/procedures/stubs/ReturnUnconstrained.py` is what a `SimLibrary` hands
//! out for a symbol it has no model for (an unresolved import, or a declared-
//! but-unimplemented libc entry): it takes any arguments, ignores them, and
//! returns a fresh unconstrained symbol sized to the prototype's return type.
//! Nothing else happens — no memory is touched, no constraint is added.
//!
//! Those stubs are the *only* SimProcedure crossing on some benches, so every
//! call round-trips into Python purely to mint one symbol. This module serves
//! them natively.
//!
//! Unlike every other native procedure, a stub is not keyed on a fixed libc
//! symbol: the dispatch name is whatever the *binary* called (`get_flag`,
//! `some_unresolved_import`, …), and the return width comes from that hook's
//! prototype. So instances are built at setup time from the Python side —
//! `RustExplorationManager::register_unconstrained_stubs` walks
//! `project._sim_procedures`, picks out the plain `ReturnUnconstrained`
//! instances, and registers one `NativeReturnUnconstrained` per display name.
//!
//! Parity notes:
//! - "Unconstrained" is gated on the `SYMBOLIC_INITIAL_VALUES` SimOption, the
//!   same way `SimSolver::Unconstrained` gates it: symbol when set, concrete
//!   `BVV(0, ret_bits)` when not. Every stock angr mode bundle ships the
//!   option, so the symbol is the common case; a user who passes
//!   `remove_options={SYMBOLIC_INITIAL_VALUES}` gets zeros here too.
//! - The symbol width is the prototype's `returnty.size`, matching Python's
//!   `state.solver.Unconstrained(..., size)`. Python then stores it through
//!   `SimCC.return_val(ty)`, which *refines* the return register to that many
//!   bytes (a partial register write on a narrow return, e.g. `eax` within
//!   `rax`); `set_register_by_offset` with a narrow BV does the same thing.
//! - A stub carrying an explicit `return_val=` kwarg, or one whose prototype
//!   has no return size, is never registered here — Python keeps those.

use super::{NativeSimProcedure, ProcedureError, symbol_counter};
use crate::state::RustSimState;
use crate::symbolic::RustBV;

/// Native `ReturnUnconstrained`: return a fresh unconstrained symbol.
pub(crate) struct NativeReturnUnconstrained {
    /// Dispatch name (the hooked symbol, e.g. `get_flag`). Leaked once at
    /// construction so the trait's `name(&self) -> &'static str` is free;
    /// one leak per distinct stub symbol per process.
    name: &'static str,
    /// Width of the returned symbol, in bits (the prototype's `returnty.size`).
    ret_bits: u32,
}

impl NativeReturnUnconstrained {
    pub(crate) fn new(name: &str, ret_bits: u32) -> Self {
        Self {
            name: Box::leak(name.to_owned().into_boxed_str()),
            ret_bits,
        }
    }
}

impl NativeSimProcedure for NativeReturnUnconstrained {
    fn name(&self) -> &'static str {
        self.name
    }

    /// Arguments are ignored (Python's stub is `run(*args, **kwargs)`), so the
    /// dispatcher need not extract any on our behalf.
    fn num_args(&self) -> usize {
        0
    }

    fn call(
        &self,
        state: &mut RustSimState,
        _args: &[RustBV],
    ) -> Result<Option<RustBV>, ProcedureError> {
        // Parity gate (angr-c7xno.61): Python's `ReturnUnconstrained` goes
        // through `SimSolver::Unconstrained`, which returns `BVV(0, bits)`
        // unless `SYMBOLIC_INITIAL_VALUES` is in `state.options`. Minting a
        // symbol unconditionally is the angr-8mjd regression shape — a
        // symbolic pointer out of a C++-ABI stub feeds null-checks and
        // store/load addresses, forking or paying concretization at each.
        if !state.has_option(crate::state::SYMBOLIC_INITIAL_VALUES) {
            return Ok(Some(RustBV::concrete(0, self.ret_bits)));
        }
        let id = symbol_counter("unconstrained_ret");
        let sym = format!("unconstrained_ret_{}_{id}", self.name);
        let ctx = state.solver().borrow();
        Ok(Some(RustBV::symbolic(&ctx, &sym, self.ret_bits)))
    }
}

test_submod!("stub_tests.rs" => tests);
