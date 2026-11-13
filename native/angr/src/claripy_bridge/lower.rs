//! The PyO3 face of the Z3 lowering in [`crate::symbolic::z3_lower`]: take
//! the clarirs AST out of a Python claripy object and lower it.

use super::BridgeError;
use crate::claripy::ast::base::Base;
use crate::symbolic::LoweredConstraint;
use crate::symbolic::z3_lower::LowerError;
use pyo3::prelude::*;

impl From<LowerError> for BridgeError {
    fn from(err: LowerError) -> Self {
        match err {
            LowerError::UnsupportedOp(m) => BridgeError::UnsupportedOp(m),
            LowerError::InvalidArgs(m) => BridgeError::InvalidArgs(m),
        }
    }
}

/// Lower the clarirs AST behind a Python claripy object to a Z3 Bool, the
/// form a constraint is asserted in, keeping the clarirs term beside it for
/// the assertion log.
///
/// A lowering of any other sort is `Err` rather than a mis-sorted `Bool`:
/// asserting a non-Bool trips Z3's `CHECK_FORMULA` sort guard, which — under
/// the engine's no-op error handler — drops the constraint silently while the
/// caller sees success (angr-d01qu, angr-mx00p.41). A caller wanting a BV
/// treated as a truthiness test lowers it through `claripy_to_rustbv` instead.
///
/// `Err(TypeMismatch)` when `obj` is not a claripy AST of *this* extension's
/// type — notably under `cargo test`, where the embedded interpreter imports
/// the installed `angr.rustylib`, whose `Base` is a different type object.
pub(crate) fn lower_claripy_constraint(
    obj: &Bound<'_, PyAny>,
) -> Result<LoweredConstraint, BridgeError> {
    let base = obj
        .cast::<Base>()
        .map_err(|e| BridgeError::TypeMismatch(e.to_string()))?;
    Ok(LoweredConstraint::lower(base.get().ast())?)
}
