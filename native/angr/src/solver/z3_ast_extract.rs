//! Claripy-node evaluation for [`RustSolverContext`]: reading the clarirs
//! node behind a Python claripy AST, turning it into an evaluable [`RustBV`],
//! and evaluating it. Split out of the parent `solver` module
//! (angr-9ke6b.205).
//!
//! None of it is a `#[pymethods]` item; these are internal helpers the
//! claripy-AST API in the parent module calls. The claripy → Z3 lowering the
//! constraint fast paths use is `claripy_bridge::lower_claripy_bool`.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use clarirs_core::prelude::AstRef;

use crate::claripy_bridge::{BridgeError, claripy_to_rustbv, clarirs_node};
use crate::symbolic::{RustBV, SymContext};

use super::RustSolverContext;

/// A claripy node as a [`RustBV`] to evaluate: a bitvector as itself, a Bool
/// as the bit `ite(b, 1, 0)`, never as a mis-sorted BV (angr-58ks). Any other
/// sort is `Err` with the sort, for the caller's error.
///
/// The value is the node itself, so it lowers to the same Z3 term as a
/// constraint built from that node: evaluation binds the variables the
/// solver already constrains.
pub(crate) fn node_to_eval_bv(node: &AstRef<'static>) -> Result<RustBV, String> {
    let sort = node.ast_type();
    if !(sort.is_bool() || sort.is_bitvec()) {
        return Err(format!("{sort:?}"));
    }
    crate::symbolic::admit_clarirs(node.clone()).map_err(|e| format!("{sort:?}: {e}"))
}

impl RustSolverContext {
    /// Lower a claripy AST to a `RustBV` usable for evaluation, mirroring the
    /// conversion order in [`RustSolverContext::eval`]: the standard
    /// claripy → RustBV import first, then the clarirs node taken whole
    /// (which preserves identity with constraints already asserted in the
    /// solver).
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
        {
            let node = clarirs_node(ast)?;
            // SILENT(cat-a): a non-BV-sorted AST is an expected miss on this
            // path — the caller (`eval_batch`) falls back to Python.
            //
            // This declines only the node path. A Bool that
            // `claripy_to_rustbv` imports above already evaluates as 0/1,
            // like `eval`; the Python wrappers turn that back into a bool
            // (`_cast_eval_result`, angr-vmi2l.218).
            if !node.ast_type().is_bitvec() {
                return None;
            }
            node_to_eval_bv(&node).ok()
        }
    }

    /// Lower a claripy AST for `min` / `max` / `solution`. Unlike
    /// [`ast_to_bv_for_eval`](Self::ast_to_bv_for_eval), a conversion error
    /// propagates (a refused width must not reach Z3 through the node:
    /// `TestClaripyBridgeWidthBounds`). The one exception is the import's
    /// depth limit, which says nothing about the AST's validity: a too-deep BV
    /// is taken whole, as `eval` would (angr-vmi2l.254).
    pub(crate) fn ast_to_bv_strict(
        &self,
        py: Python<'_>,
        ast: &Bound<'_, PyAny>,
        ctx: &SymContext,
    ) -> PyResult<RustBV> {
        match claripy_to_rustbv(py, ast, ctx) {
            Ok(bv) => Ok(bv),
            Err(err @ BridgeError::RecursionLimit(_)) => clarirs_node(ast)
                .filter(|node| node.ast_type().is_bitvec())
                .and_then(|node| node_to_eval_bv(&node).ok())
                .ok_or_else(|| err.into()),
            Err(err) => Err(err.into()),
        }
    }

    /// Evaluate a claripy node directly in the solver context: the path for
    /// an AST `claripy_to_rustbv` cannot import operand-wise. Bool evaluates
    /// as 0/1 (see [`node_to_eval_bv`]); any other non-BV sort is an error.
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
