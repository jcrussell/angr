//! Helpers shared by the `statements*` test modules.
//!
//! The statement tests are split per prod sibling — `statements_tests.rs`
//! (registered from `statements.rs`), `statements_dirty_tests.rs` (from
//! `statements_dirty.rs`) and `statements_cas_tests.rs` (from
//! `statements_cas.rs`) — so they are no longer children of a single module
//! and `super::statements_tests_support::` cannot reach across them. This
//! module is therefore registered one level up, from `interpreter/mod.rs`,
//! and its helpers are `pub(super)`: visible to the whole `interpreter`
//! subtree, which is exactly the set of modules that may use them.
//!
//! A helper used by only ONE of those modules stays private there — see
//! `statements_tests.rs`'s `new_interp_multiwrite` and
//! `statements_cas_tests.rs`'s `cas_stmt`.

use super::*;
use crate::vex::ir::IRType;

pub(super) fn new_interp(ctx: &SymContext) -> VEXInterpreter<'_> {
    VEXInterpreter::new(VexArch::AMD64, ctx)
}

pub(super) fn make_irsb_with_temps(addr: u64, temp_types: &[IRType]) -> IRSB {
    let mut irsb = IRSB::new(addr, VexArch::AMD64);
    irsb.statements.push(IRStmt::IMark {
        addr,
        len: 4,
        delta: 0,
    });
    for ty in temp_types {
        irsb.tyenv.new_temp(*ty);
    }
    irsb
}

/// Initialize Python once for tests that need to call `execute_stmt_with_callbacks`.
pub(super) fn with_python<F, R>(f: F) -> R
where
    F: FnOnce(&PythonCallbacks) -> R,
{
    Python::initialize();
    let callbacks = PythonCallbacks::new();
    Python::attach(|_py| f(&callbacks))
}
