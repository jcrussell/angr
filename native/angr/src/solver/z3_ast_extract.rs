//! Raw Z3-pointer plumbing for [`RustSolverContext`].
//!
//! Everything in this module deals in `Z3_ast` pointers lowered from claripy
//! ASTs rather than in the ASTs themselves: producing one, wrapping
//! it as an evaluable [`RustBV`], and evaluating it. It is split out of the
//! parent `solver` module because it is the only part of the solver surface
//! that carries `unsafe` and its attendant safety obligations (angr-9ke6b.205).
//!
//! This is the *consumer* side: [`Z3AstPtr`] itself — the refcounted wrapper
//! with the `Drop`/`Z3_inc_ref` obligations — is defined in
//! `symbolic/z3_ast_ptr.rs`. This module only obtains one from
//! `claripy_bridge`'s lowering and hands it to the solver (angr-03vl4.75).
//!
//! None of it is a `#[pymethods]` item — [`Z3AstPtr`] is not PyO3-bridgeable
//! (it owns a Z3 refcount and cannot be reconstructed from a Python value), so
//! these are internal helpers the claripy-AST API in the parent module calls.

// The sibling `solver` module forbids `unsafe_code` for itself so the "all
// the `unsafe` is in one file" invariant is the compiler's to keep rather
// than a doc comment's (angr-5mnx3.10). This module is declared independently
// in `lib.rs`, outside `solver`'s module subtree, specifically so `forbid`
// (which — unlike `deny` — cannot be overridden by any descendant) never
// reaches it (angr-8ucb3): this `#![allow]` is therefore an ordinary, opt-in
// lint level on an unrelated module, not an override. Every unsafe block
// below carries its own `SAFETY:` rationale.
#![allow(unsafe_code)]

// Both are consumed only by the `Z3AstPtr` paths below, which are themselves
// `vex-engine-z3`-gated — importing them unconditionally warns in the no-z3
// combos `make check-no-z3` gates (angr-sqfj8.139).
#[cfg(feature = "vex-engine-z3")]
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[cfg(feature = "vex-engine-z3")]
use clarirs_core::prelude::AstRef;

#[cfg(feature = "vex-engine-z3")]
use crate::claripy::ast::base::Base;

use crate::claripy_bridge::claripy_to_rustbv;
#[cfg(feature = "vex-engine-z3")]
use crate::symbolic::Z3AstPtr;
use crate::symbolic::{RustBV, SymContext};

use crate::solver::RustSolverContext;

/// Lower a claripy AST to a typed [`Z3AstPtr`] in the thread-local Z3 context.
///
/// claripy is clarirs, so this reads the [`AstRef`](clarirs_core::prelude::AstRef)
/// behind the Python object and lowers it with
/// [`claripy_bridge::lower_claripy_ast`](crate::claripy_bridge::lower_claripy_ast)
/// — losslessly, for every Bool / bitvector / float op, and into the context
/// the `SymContext`s on this thread share. Returns `Err` for an object that is
/// not one of this extension's claripy ASTs or holds an op with no Z3 lowering
/// (strings, VSA); callers fall back to `claripy_to_rustbv`.
#[cfg(feature = "vex-engine-z3")]
pub(crate) fn extract_z3_ast_ptr(_py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<Z3AstPtr> {
    crate::claripy_bridge::lower_claripy_ast(ast)
        .map_err(|e| PyRuntimeError::new_err(format!("claripy AST has no Z3 lowering: {e}")))
}

/// The clarirs node behind a Python claripy AST, or `None` when `ast` is not
/// one of this extension's claripy ASTs.
#[cfg(feature = "vex-engine-z3")]
pub(crate) fn claripy_node(ast: &Bound<'_, PyAny>) -> Option<AstRef<'static>> {
    Some(ast.cast::<Base>().ok()?.get().ast())
}

/// A claripy node as a [`RustBV`] to evaluate: a bitvector as itself, a Bool
/// as the bit `ite(b, 1, 0)` (angr-58ks: never as a mis-sorted BV). Any other
/// sort is `Err` with the sort, for the caller's error.
///
/// The value is the node itself, so it lowers to the same Z3 term as a
/// constraint built from that node: evaluation binds the variables the
/// solver already constrains.
#[cfg(feature = "vex-engine-z3")]
pub(crate) fn node_to_eval_bv(node: &AstRef<'static>) -> Result<RustBV, String> {
    let sort = node.ast_type();
    if !(sort.is_bool() || sort.is_bitvec()) {
        return Err(format!("{sort:?}"));
    }
    RustBV::from_clarirs(node.clone()).map_err(|e| format!("{sort:?}: {e}"))
}

/// Wrap a Bool-sorted live Z3 AST as a `z3::ast::Bool` for assertion.
///
/// Returns `None` for any other sort, so the sort check and the `wrap` it
/// licenses cannot drift apart at a call site. `add_constraint_tracked_ast`
/// used to do the check and the `unsafe` wrap itself, which was the one
/// `unsafe` block left outside this module after the angr-9ke6b.205 split —
/// contradicting both this module's doc and `solver.rs`'s own header
/// (angr-5mnx3.10).
///
/// Wrapping a non-Bool node as `Bool` and asserting it trips Z3's
/// `CHECK_FORMULA` sort-mismatch guard, which — because our context installs
/// a no-op error handler — fails *silently*: the constraint is never asserted
/// while the caller still sees success (angr-d01qu). Callers that need the
/// non-Bool case handled must fall back to a `claripy_to_rustbv` lowering
/// rather than wrap anyway.
#[cfg(feature = "vex-engine-z3")]
pub(crate) fn z3_ast_to_bool(z3_ast: &Z3AstPtr) -> Option<z3::ast::Bool> {
    use z3::ast::Ast;
    if !z3_ast.is_bool() {
        return None;
    }
    // SAFETY: `z3_ast` holds an active ref to a live `Z3_ast`, checked
    // Bool-sorted just above, and `z3_ast.context()` is the context that ref
    // was taken under; `Bool::wrap` takes its own ref.
    Some(unsafe { z3::ast::Bool::wrap(z3_ast.context(), z3_ast.as_z3_ast()) })
}

impl RustSolverContext {
    /// Lower a claripy AST to a `RustBV` usable for evaluation, mirroring the
    /// conversion order in [`RustSolverContext::eval`]: the standard
    /// claripy → RustBV import first, then the raw Z3 AST pointer (which
    /// preserves identity with constraints already asserted in the solver).
    /// Returns `None` when neither path applies. Used by `eval_batch`.
    pub(crate) fn ast_to_bv_for_eval(
        &self,
        py: Python<'_>,
        ast: &Bound<'_, PyAny>,
        ctx: &SymContext,
    ) -> Option<RustBV> {
        if let Ok(bv) = claripy_to_rustbv(py, ast, ctx) {
            return Some(bv);
        }
        #[cfg(feature = "vex-engine-z3")]
        {
            let node = claripy_node(ast)?;
            // SILENT(cat-a): a non-BV-sorted AST is an expected miss on this
            // path — the caller (`eval_batch`) falls back to Python.
            //
            // Bool is declined here rather than lowered to a 1-bit BV the way
            // the `eval` paths do: `eval_batch`'s Python fallback yields a
            // Python bool for a Bool-sorted AST, and quietly swapping that for
            // 0/1 would change the value `eval_batch` returns.
            if !node.ast_type().is_bitvec() {
                return None;
            }
            node_to_eval_bv(&node).ok()
        }
        #[cfg(not(feature = "vex-engine-z3"))]
        {
            None
        }
    }

    /// Evaluate a claripy node directly in the solver context: the path for
    /// an AST `claripy_to_rustbv` cannot import operand-wise. Bool evaluates
    /// as 0/1 (see [`node_to_eval_bv`]); any other non-BV sort is an error.
    #[cfg(feature = "vex-engine-z3")]
    pub(crate) fn eval_claripy_node(
        &self,
        py: Python<'_>,
        node: &AstRef<'static>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let ctx = self.i().ctx();
        let bv = node_to_eval_bv(node).map_err(|sort| {
            PyRuntimeError::new_err(format!(
                "eval: claripy AST has unsupported sort {sort} (expected BV or Bool)"
            ))
        })?;

        // The wide/narrow split follows the width of the BV actually built
        // (Bool→1-bit lowering or the Z3 BV sort), not a separately read
        // claripy `.length` that could disagree with it (angr-9ke6b.202).
        if bv.width() <= 128 {
            match ctx.eval(&bv) {
                Some(v) => Ok(Some(v.into_pyobject(py)?.into())),
                None => Ok(None),
            }
        } else {
            match ctx.eval_wide(&bv) {
                Some(bytes) => {
                    let py_bytes = pyo3::types::PyBytes::new(py, &bytes);
                    let int_class = py.get_type::<pyo3::types::PyInt>();
                    let py_int = int_class.call_method1("from_bytes", (py_bytes, "big"))?;
                    Ok(Some(py_int.into()))
                }
                None => Ok(None),
            }
        }
    }
}

test_submod!("z3_ast_extract_tests.rs" => tests);
