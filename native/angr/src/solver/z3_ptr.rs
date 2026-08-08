//! Raw Z3-pointer plumbing for [`RustSolverContext`].
//!
//! Everything in this module deals in `Z3_ast` pointers borrowed out of
//! claripy's z3 backend rather than in claripy ASTs: extracting one, wrapping
//! it as an evaluable [`RustBV`], and evaluating it. It is split out of the
//! parent `solver` module because it is the only part of the solver surface
//! that carries `unsafe` and its attendant safety obligations (angr-9ke6b.205).
//!
//! None of it is a `#[pymethods]` item — [`Z3AstPtr`] is not PyO3-bridgeable
//! (it owns a Z3 refcount and cannot be reconstructed from a Python value), so
//! these are internal helpers the claripy-AST API in the parent module calls.

// Both are consumed only by the `Z3AstPtr` paths below, which are themselves
// `vex-engine-z3`-gated — importing them unconditionally warns in the no-z3
// combos `make check-no-z3` gates (angr-sqfj8.139).
#[cfg(feature = "vex-engine-z3")]
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[cfg(feature = "vex-engine-z3")]
use std::sync::Arc;

use crate::claripy_bridge::claripy_to_rustbv;
#[cfg(feature = "vex-engine-z3")]
use crate::symbolic::Z3AstPtr;
use crate::symbolic::{RustBV, SymContext};

use super::RustSolverContext;

/// Extract a typed [`Z3AstPtr`] from a claripy AST's z3 backend.
///
/// Returns `Err` if the claripy → z3 backend conversion fails or yields a
/// null pointer. The returned handle carries its own refcount (taken via
/// `Z3_inc_ref` at construction); claripy's original AST remains alive
/// independently in claripy's cache.
///
/// # Safety
///
/// This function is itself safe — the unsafety is encapsulated inside
/// [`Z3AstPtr::from_borrowed_raw`]. The precondition (pointer denotes a
/// live `Z3_ast` in the active thread-local Z3 context) is met by
/// construction: `claripy.backends.z3.convert(...)` always returns a
/// live Z3 AST in the process-global Z3 context (which is also our
/// thread-local context because z3-rs 0.19+ shares it).
#[cfg(feature = "vex-engine-z3")]
pub(super) fn extract_z3_ast_ptr(py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<Z3AstPtr> {
    let claripy = py.import("claripy")?;
    let z3_backend = claripy.getattr("backends")?.getattr("z3")?;
    let z3_obj = z3_backend.call_method1("convert", (ast,))?;
    let ast_ref = z3_obj.call_method0("as_ast")?;
    let ptr: usize = ast_ref.getattr("value")?.extract()?;
    let ctx = z3::Context::thread_local();
    // SAFETY: claripy's z3 backend returned this pointer for a live AST
    // it holds in its own cache; the AST is in the process-global Z3
    // context, which matches our thread-local context (z3-rs 0.19+).
    //
    // INVARIANT (claripy-AST-alive): `from_borrowed_raw` takes a fresh
    // `Z3_inc_ref`, so the borrowed `Z3_ast` must have refcount >= 1 at
    // this point. That holds *only* because `z3_obj` / `ast_ref` (the
    // claripy backend result) are still in scope, keeping the AST retained
    // in claripy's cache for the duration of `convert`. A future caller
    // that sources `ptr` from a holder already dropped would violate this
    // silently and inc_ref a dangling node — do not reorder the extraction
    // below the point where the claripy result goes out of scope.
    let handle = unsafe { Z3AstPtr::from_borrowed_raw(&ctx, ptr) }.ok_or_else(|| {
        PyRuntimeError::new_err("claripy z3 backend returned null Z3_ast pointer")
    })?;
    // Defensive liveness probe: a live, well-formed AST always resolves to
    // a concrete sort; `Unknown` signals a dangling/garbage pointer, i.e. a
    // violated claripy-AST-alive precondition. Debug-only — compiles out in
    // release, so this is hardening with zero runtime cost on the hot path.
    debug_assert!(
        handle.sort_kind() != z3_sys::SortKind::Unknown,
        "extract_z3_ast_ptr: borrowed Z3_ast resolved to no sort -- the \
         claripy-AST-alive invariant was likely violated (pointer not \
         retained by a live claripy AST at borrow time)"
    );
    Ok(handle)
}

/// Wrap a live Z3 AST as an anonymous [`RustBV::Symbolic`] for evaluation.
///
/// This is the one place that builds the `id: 0` / `name: ""` sentinel form of
/// `RustBV::Symbolic`: a throwaway BV handed straight to `SymContext::eval*`
/// that never enters the symbol table, so it needs neither a handle id nor a
/// name. Keeping the construction — and the `wrap` `unsafe` behind it — in one
/// function means a future change to `RustBV::Symbolic`'s invariants has a
/// single site to update (angr-9ke6b.203).
///
/// A Bool-sorted AST is *lowered* to a 1-bit BV (1 when true, 0 when false)
/// rather than BV-wrapped: operating on a mis-sorted node trips Z3's error
/// handler, which aborts the process instead of returning an error
/// (angr-58ks). Returns `None` for any other sort; callers that want a hard
/// error report [`Z3AstPtr::sort_kind`] themselves.
#[cfg(feature = "vex-engine-z3")]
pub(super) fn z3_ast_to_eval_bv(z3_ast: &Z3AstPtr) -> Option<RustBV> {
    use z3::ast::Ast;
    // SAFETY: `z3_ast` holds an active ref to a live `Z3_ast`, and each branch
    // checks the sort before wrapping the node as that sort. `Bool::wrap` /
    // `BV::wrap` take their own refs.
    unsafe {
        let z3_ctx = z3::Context::thread_local();
        let (ast, width) = if z3_ast.is_bool() {
            let z3_bool = z3::ast::Bool::wrap(&z3_ctx, z3_ast.as_z3_ast());
            let as_bv = z3_bool.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1));
            (as_bv, 1)
        } else {
            // Width comes from the Z3 sort, not claripy's `.length` — the sort
            // is what `BV::wrap` is constrained by, and `bv_width()` returning
            // `Some` is exactly the precondition that makes the wrap sound
            // (angr-9ke6b.202).
            let width = z3_ast.bv_width()?;
            (z3::ast::BV::wrap(&z3_ctx, z3_ast.as_z3_ast()), width)
        };
        Some(RustBV::Symbolic {
            id: 0,
            ast,
            width,
            name: Arc::from(""),
        })
    }
}
impl RustSolverContext {
    /// Lower a claripy AST to a `RustBV` usable for evaluation, mirroring the
    /// conversion order in [`RustSolverContext::eval`]: the standard
    /// claripy → RustBV import first, then the raw Z3 AST pointer (which
    /// preserves identity with constraints already asserted in the solver).
    /// Returns `None` when neither path applies. Used by `eval_batch`.
    pub(super) fn ast_to_bv_for_eval(
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
            let z3_ast = extract_z3_ast_ptr(py, ast).ok()?;
            // SILENT(cat-a): a non-BV-sorted AST is an expected miss on this
            // path — the caller (`eval_batch`) falls back to Python.
            //
            // Bool is declined here rather than lowered to a 1-bit BV the way
            // the `eval` paths do: `eval_batch`'s Python fallback yields a
            // Python bool for a Bool-sorted AST, and quietly swapping that for
            // 0/1 would change the value `eval_batch` returns. Every other
            // non-BV sort is declined by `z3_ast_to_eval_bv` itself.
            if z3_ast.is_bool() {
                return None;
            }
            z3_ast_to_eval_bv(&z3_ast)
        }
        #[cfg(not(feature = "vex-engine-z3"))]
        {
            None
        }
    }

    /// Evaluate a typed Z3 AST handle directly in the solver context.
    ///
    /// Lives outside `#[pymethods]` because [`Z3AstPtr`] is not a PyO3-
    /// bridgeable type (it owns a Z3 refcount and cannot be reconstructed
    /// from a Python value).
    #[cfg(feature = "vex-engine-z3")]
    pub(super) fn eval_z3_ast_ptr(
        &self,
        py: Python<'_>,
        z3_ast: Z3AstPtr,
    ) -> PyResult<Option<Py<PyAny>>> {
        let ctx = self.i().ctx();

        // angr-58ks: the sort is verified before wrapping. A Bool-sorted AST
        // (e.g. an fpEQ comparison that `claripy_to_rustbv` cannot lower and
        // so reaches this raw path with claripy `length == None`) must NOT be
        // BV-wrapped — operating on the mis-sorted node trips Z3's error
        // handler, which aborts the process instead of returning a PyErr.
        // `z3_ast_to_eval_bv` lowers a Bool to a 1-bit BV instead, and
        // declines any sort that is neither.
        let Some(bv) = z3_ast_to_eval_bv(&z3_ast) else {
            return Err(PyRuntimeError::new_err(format!(
                "eval: Z3 AST has unsupported sort kind {:?} (expected BV or Bool)",
                z3_ast.sort_kind()
            )));
        };

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

test_submod!("z3_ptr_tests.rs" => tests);
