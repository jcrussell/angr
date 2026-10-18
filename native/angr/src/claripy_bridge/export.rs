//! RustBV -> claripy AST export: a `Concrete` becomes `claripy.BVV`, a clarirs
//! term the claripy object over that very node (annotations and all).
//!
//! What lives where (angr-5mnx3.11 split):
//!
//! - here: [`rustbv_to_claripy`] and [`assumed_guard_to_claripy`];
//! - [`ast_helpers`]: `concrete_value_to_bvv`, which needs a live interpreter;
//! - [`width_decisions`]: the pure `BVV` encoding decision
//!   ([`width_decisions::ConcreteBvvEncoding`]), unit-testable without claripy.

mod ast_helpers;
mod width_decisions;

use pyo3::prelude::*;

use crate::symbolic::RustBV;

use ast_helpers::concrete_value_to_bvv;

/// Materialize one `(guard, is_assumed_true)` entry from a `SymContext`'s
/// assumed log as a claripy **boolean** constraint (angr-op0dn.14.4.1).
///
/// The guard is a 1-bit `RustBV`. `rustbv_to_claripy` lowers it to whatever
/// claripy type its op tree implies — a `Bool` for a comparison, but a 1-bit
/// `BV` for e.g. an `Extract`/`Ite`-derived guard. `claripy.Not()` on a BV
/// returns `NotImplemented` rather than raising, so a BV-typed guard used to
/// export as the `NotImplemented` singleton. Compare a BV guard against
/// `BVV(1|0, 1)` instead — the same `guard != 0` semantics the Python engine
/// gives a VEX exit guard — and reserve `Not` for genuinely boolean guards.
///
/// Shared by `_export_state_constraints` and the native `constraints` inspect
/// dispatch so both render an assumed guard identically.
pub(crate) fn assumed_guard_to_claripy(
    py: Python<'_>,
    bv: &RustBV,
    claripy_mod: &Bound<'_, PyAny>,
    is_true: bool,
) -> PyResult<Py<PyAny>> {
    let ast = rustbv_to_claripy(py, bv, claripy_mod)?;
    let bound = ast.bind(py);
    // BV ASTs carry their bit width as `.length`; Bool ASTs have none (clarirs
    // `Bool` has no `length` attribute at all). Any *other* failed read leaves
    // the two indistinguishable, and both guesses are wrong-answer outcomes:
    // the Bool branch on a BV re-creates the `NotImplemented` constraint this
    // function exists to prevent, and the BV branch on a Bool builds a
    // `Bool == BVV(_, 1)` comparison. Propagate instead — both callers
    // (`push_assumed_constraint_or_log` in `exploration/state_api.rs` and
    // `call_inspect_constraints` in `callbacks/inspect.rs`) already log and drop
    // a conversion `Err`, which beats handing the solver a bogus constraint
    // (angr-sqfj8.22).
    let is_bv = match bound.getattr("length") {
        Ok(length) => !length.is_none(),
        Err(e) if e.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) => false,
        Err(e) => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "assumed_guard_to_claripy: could not read `.length` on the exported guard \
                 (cannot tell a Bool guard from a 1-bit BV guard): {e}"
            )));
        }
    };
    if is_bv {
        let bit = u64::from(is_true);
        let bvv = claripy_mod.call_method1("BVV", (bit, 1u32))?;
        return Ok(bound.call_method1("__eq__", (bvv,))?.unbind());
    }
    if is_true {
        Ok(ast)
    } else {
        Ok(claripy_mod.call_method1("Not", (ast,))?.unbind())
    }
}

/// Convert a RustBV back to a claripy AST, for returning a value to Python.
pub(crate) fn rustbv_to_claripy(
    py: Python<'_>,
    bv: &RustBV,
    claripy_mod: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    // GIL-work timing (angr-1ilq.7): the whole export, claripy calls
    // included, as one region; the depth guard prevents double-counting when
    // a dispatch method called us already.
    let _gil =
        crate::gil_profile::GilWorkGuard::enter_as(crate::gil_profile::GilClass::ClaripyExport);

    match bv {
        RustBV::Concrete { value, width } => concrete_value_to_bvv(py, claripy_mod, *value, *width),
        // Already a claripy AST: wrap the node, annotations and all. A Bool
        // held in the 1-bit encoding exports as the Bool.
        RustBV::Ast { ast, .. } => {
            let node = crate::symbolic::bool_inside(ast).unwrap_or_else(|| ast.clone());
            crate::claripy::ast::base::Base::from_ast(py, node)
                .map(|b| b.into_any().unbind())
                .map_err(PyErr::from)
        }
    }
}
