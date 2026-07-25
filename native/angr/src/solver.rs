//! PyO3-exposed Rust solver context.
//!
//! This module provides a Python-accessible constraint solver that uses
//! Z3 under the hood. It bridges claripy ASTs to Rust's SymContext.
//!
//! With z3-rs 0.19+, the Z3 context is thread-local, so we don't need
//! to manage explicit context lifetimes.
//!
//! This is a Python-boundary module; `unwrap`/`expect` are denied here so a
//! future panic-on-input landmine cannot be reintroduced without a reviewed,
//! reasoned `#[allow]` (angr-qwyti.11 enforcement layer).
#![deny(clippy::unwrap_used, clippy::expect_used)]

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

use std::cell::{Ref, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use crate::claripy_bridge::{BridgeError, claripy_to_rustbv, try_extract_bvv};
#[cfg(feature = "vex-engine-z3")]
use crate::symbolic::Z3AstPtr;
use crate::symbolic::{BinaryOpError, RustBV, RustBVHandle, RustSymbolTable, SymContext};

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
fn extract_z3_ast_ptr(py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<Z3AstPtr> {
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

/// Convert a BridgeError to a PyErr, preserving the variant's structure at
/// the Python boundary (angr-ghwsd.2). Mirrors the per-variant mapping in
/// `errors.rs::From<RustExecError>`: a type mismatch surfaces as `TypeError`
/// and bad arguments / unsupported ops as `ValueError`, matching how the
/// codebase classifies these elsewhere (e.g. `fuzzer.rs` uses `PyTypeError`).
/// `#[non_exhaustive]` forces a wildcard arm.
impl From<BridgeError> for PyErr {
    fn from(err: BridgeError) -> Self {
        let msg = err.to_string();
        match err {
            BridgeError::TypeMismatch(_) => PyTypeError::new_err(msg),
            BridgeError::InvalidArgs(_) | BridgeError::UnsupportedOp(_) => {
                PyValueError::new_err(msg)
            }
            // PythonError + any future variant fall back to RuntimeError.
            _ => PyRuntimeError::new_err(msg),
        }
    }
}

/// Build the canonical "invalid handle id" error for a symbol-table lookup
/// miss (`symbol_table().get(id)` / `op_*` returning `None`). Standardized on
/// `PyValueError` (a bad argument value, mirroring `From<BridgeError>`'s
/// `InvalidArgs` mapping) so Python callers can reliably `except ValueError`
/// across all sites, and always names the offending id(s) — previously the 34
/// `op_*` sites raised a bare `PyRuntimeError` with no id (angr-ghwsd.1).
///
/// Pass every handle id the operation dereferenced; the message reports them
/// all since the table cannot say which one missed.
pub(crate) fn invalid_handle_id(ids: &[u64]) -> PyErr {
    match ids {
        [id] => PyValueError::new_err(format!("invalid handle id: {id}")),
        _ => PyValueError::new_err(format!("invalid handle id (one of {ids:?})")),
    }
}

/// Map a two-operand table failure to a `PyValueError`. Owns the pyo3
/// conversion so `symbolic::table` stays Python-agnostic (angr-ph300.32).
/// Reuses [`invalid_handle_id`] for the missing-handle case so its message
/// stays identical to the single-operand ops.
impl From<BinaryOpError> for PyErr {
    fn from(err: BinaryOpError) -> Self {
        match err {
            BinaryOpError::MissingHandle { a_id, b_id } => invalid_handle_id(&[a_id, b_id]),
            BinaryOpError::WidthMismatch { lhs, rhs } => PyValueError::new_err(format!(
                "binary op width mismatch: {lhs}-bit vs {rhs}-bit operands"
            )),
        }
    }
}

/// Storage for SymContext: either owned or shared via Rc.
/// Shared mode allows Python callbacks to use the pending state's solver
/// directly (O(1)) instead of forking it (~3ms Z3 clone per callback).
///
/// `Owned` holds `SymContext` inline (the common path); `Shared` is a
/// thin `Rc<RefCell<…>>` pointer. Boxing `Owned` to equalize variants
/// would add a heap allocation per solver fork — the whole point of
/// `Owned` is to avoid that. Variant size disparity is intentional.
#[allow(clippy::large_enum_variant)]
enum SolverCtxStorage {
    Owned(SymContext),
    Shared(Rc<RefCell<SymContext>>),
}

/// Guard for borrowing SymContext from either storage variant.
/// Implements Deref<Target=SymContext> for transparent access.
enum SolverCtxGuard<'a> {
    Ref(&'a SymContext),
    Borrowed(Ref<'a, SymContext>),
}

impl<'a> std::ops::Deref for SolverCtxGuard<'a> {
    type Target = SymContext;
    fn deref(&self) -> &SymContext {
        match self {
            SolverCtxGuard::Ref(r) => r,
            SolverCtxGuard::Borrowed(b) => b,
        }
    }
}

/// Python-accessible Rust solver context.
///
/// This wraps a Z3-backed SymContext and provides solver operations
/// that can be called from Python with claripy ASTs.
///
/// Supports two modes:
/// - Owned: holds its own SymContext (normal case, forked contexts)
/// - Shared: borrows a state's solver via Rc (zero-cost callback solver)
///
/// With z3-rs 0.19+, we don't need to manage a separate Z3 context -
/// it's automatically handled via thread-local storage.
#[pyclass(unsendable)]
pub struct RustSolverContext {
    // Inner context (same structure whether Z3 is enabled or not).
    //
    // `Option` so `close()` can extract and drop the heavy payload
    // (owned SymContext + cloned Z3 solver + RustSymbolTable) *on the
    // owning thread*. Under RUST_PARALLEL_WORKERS>1 a dead context can
    // otherwise take its final dealloc on a scheduler worker thread,
    // where pyo3 refuses to drop an `unsendable` pyclass and permanently
    // leaks the box (angr-87e56). Explicit main-thread `close()` empties
    // this to `None` first, so the later off-thread tp_dealloc drops
    // nothing.
    inner: Option<Box<SolverInner>>,

    // Thread that constructed this context (and therefore the only thread
    // on which its non-Send payload may be dropped). `close()` compares
    // against this and refuses to drop off-owner, so a `close()` routed
    // from a scheduler worker (via the Python graveyard drain) is a safe
    // no-op rather than a cross-thread drop of the SymContext/Rc internals.
    owner: std::thread::ThreadId,
}

struct SolverInner {
    sym_ctx: SolverCtxStorage,
    /// Symbol table for handle-based API (claripy bypass).
    symbol_table: RustSymbolTable,
}

impl SolverInner {
    /// Get a guard for the SymContext, works for both owned and shared.
    fn ctx(&self) -> SolverCtxGuard<'_> {
        match &self.sym_ctx {
            SolverCtxStorage::Owned(ctx) => SolverCtxGuard::Ref(ctx),
            SolverCtxStorage::Shared(rc) => SolverCtxGuard::Borrowed(rc.borrow()),
        }
    }
}

#[pymethods]
impl RustSolverContext {
    /// Create a new Rust solver context.
    #[new]
    pub fn new() -> Self {
        // With z3-rs 0.19+, the Z3 context is thread-local
        // SymContext::new() handles the setup
        RustSolverContext {
            owner: std::thread::current().id(),
            inner: Some(Box::new(SolverInner {
                sym_ctx: SolverCtxStorage::Owned(SymContext::new()),
                symbol_table: RustSymbolTable::new(),
            })),
        }
    }

    /// Add a constraint from a claripy AST.
    ///
    /// Uses fast path when Z3 context is shared: extracts the raw Z3_ast
    /// from claripy's z3 backend and asserts it directly, preserving the
    /// original Z3 AST structure. Falls back to RustBV conversion otherwise.
    pub fn add_constraint_ast(&self, py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<()> {
        let ctx = self.i().ctx();

        // Fast path: try to extract raw Z3 AST from claripy's z3 backend.
        // This preserves the original AST structure (Python's claripy creates
        // different Z3 trees than our build_z3_ast_cached), avoiding the structural
        // divergence that causes 3-7x slower Z3 solving.
        #[cfg(feature = "vex-engine-z3")]
        {
            // angr-58ks: `add_constraint_raw` Bool-wraps its input by
            // contract; a non-Bool AST would trip Z3's process-aborting
            // error handler. Only take the raw fast path for Bool-sorted
            // ASTs. A non-Bool (e.g. a BV used as a truthiness constraint)
            // falls through to the slow path below, which lowers it to a
            // proper `!= 0` Bool — preserving semantics without the panic.
            if let Ok(z3_ast) = extract_z3_ast_ptr(py, ast)
                && z3_ast.is_bool()
            {
                ctx.add_constraint_raw(z3_ast);
                // Also track in RustBV for export (best-effort, non-critical)
                if let Ok(bv) = claripy_to_rustbv(py, ast, &ctx) {
                    ctx.assumed_constraints_push(bv, true);
                }
                return Ok(());
            }
        }

        // Slow path: convert claripy AST to RustBV, then build Z3 AST
        let bv = claripy_to_rustbv(py, ast, &ctx)?;

        #[cfg(feature = "vex-engine-z3")]
        {
            if bv.width() == 1 {
                ctx.assume_true(&bv);
            } else {
                let zero = RustBV::concrete(0, bv.width());
                let neq = bv.ne(&zero, &ctx);
                ctx.assume_true(&neq);
            }
        }

        Ok(())
    }

    /// Add multiple constraints from claripy ASTs.
    ///
    /// Fast path: when every AST resolves to a raw Z3 AST pointer via the
    /// claripy z3 backend (the same fast path used by `add_constraint_ast`),
    /// dispatch through `SymContext::add_constraints_raw_batch` so the whole
    /// batch shares one `local_constraints` lock, one `solver()` guard, and
    /// one model invalidation pass. Any AST that fails the raw extraction
    /// flips the call back to the per-constraint slow path so behavior stays
    /// identical to the unbatched loop.
    pub fn add_constraints(&self, py: Python<'_>, asts: &Bound<'_, PyList>) -> PyResult<()> {
        #[cfg(feature = "vex-engine-z3")]
        {
            let ctx = self.i().ctx();
            let n = asts.len();
            let mut entries: Vec<(Z3AstPtr, RustBV, bool)> = Vec::with_capacity(n);
            let mut all_raw = true;
            for ast in asts.iter() {
                let ptr = match extract_z3_ast_ptr(py, &ast) {
                    Ok(p) if p.is_bool() => p,
                    // angr-58ks: a non-Bool AST cannot go through the raw
                    // batch (add_constraints_raw_batch Bool-wraps by
                    // contract); drop to the per-constraint slow path which
                    // lowers it correctly.
                    Ok(_) | Err(_) => {
                        all_raw = false;
                        break;
                    }
                };
                let bv = match claripy_to_rustbv(py, &ast, &ctx) {
                    Ok(b) => b,
                    Err(_) => {
                        all_raw = false;
                        break;
                    }
                };
                entries.push((ptr, bv, true));
            }
            if all_raw {
                ctx.add_constraints_raw_batch(entries);
                return Ok(());
            }
        }
        // Slow path: any AST that resisted the raw extraction sends the
        // whole batch through the per-constraint route to preserve semantics.
        for ast in asts.iter() {
            self.add_constraint_ast(py, &ast)?;
        }
        Ok(())
    }

    /// Add a constraint with unsat-core tracking enabled.
    ///
    /// Mirrors `add_constraint_ast` but routes through
    /// `SymContext::add_constraint_tracked`, which uses
    /// `solver.assert_and_track` so the constraint participates in
    /// `solver.get_unsat_core()`. The index of the constraint within the
    /// per-context tracker vector is returned and is what
    /// [`Self::unsat_core`] will report.
    ///
    /// Untracked constraints (added via `add_constraint_ast`) coexist with
    /// tracked ones in the solver but never appear in `unsat_core()` output.
    /// Tracking adds overhead (fresh Bool symbol, name formatting, mutex
    /// acquisition) so callers should opt in only when they need core
    /// extraction.
    pub fn add_constraint_tracked_ast(
        &self,
        py: Python<'_>,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        let ctx = self.i().ctx();
        #[cfg(feature = "vex-engine-z3")]
        {
            if let Ok(z3_ast) = extract_z3_ast_ptr(py, ast) {
                let z3_ctx = z3::Context::thread_local();
                // SAFETY: `z3_ast` is a live Bool-sorted `Z3_ast` (holds its
                // own ref via Z3AstPtr); `Ast::wrap` takes its own ref.
                let constraint: z3::ast::Bool =
                    unsafe { z3::ast::Ast::wrap(&z3_ctx, z3_ast.as_z3_ast()) };
                let idx = ctx.add_constraint_tracked_indexed(constraint);
                if let Ok(bv) = claripy_to_rustbv(py, ast, &ctx) {
                    ctx.assumed_constraints_push(bv, true);
                }
                return Ok(idx);
            }

            // Slow path: build the Z3 Bool from a RustBV. Mirrors
            // assume_true's symbolic path (BV width 1 → bool via to_z3_bool;
            // wider BV → ne(0) bool).
            let bv = claripy_to_rustbv(py, ast, &ctx)?;
            let constraint = if bv.width() == 1 {
                bv.to_z3_bool()
            } else {
                let zero = RustBV::concrete(0, bv.width());
                bv.ne(&zero, &ctx).to_z3_bool()
            };
            let idx = ctx.add_constraint_tracked_indexed(constraint);
            ctx.assumed_constraints_push(bv, true);
            Ok(idx)
        }
        #[cfg(not(feature = "vex-engine-z3"))]
        {
            let _ = (py, ast);
            Err(PyRuntimeError::new_err(
                "unsat_core tracking requires Z3 support (vex-engine-z3 feature)",
            ))
        }
    }

    /// Check if the current constraints are satisfiable.
    pub fn satisfiable(&self) -> bool {
        self.i().ctx().is_sat()
    }

    /// Set the Z3 solver timeout in milliseconds.
    pub fn set_timeout(&self, timeout_ms: u32) {
        self.i().ctx().set_timeout(timeout_ms);
    }

    /// Get the Z3 solver timeout in milliseconds.
    pub fn timeout_ms(&self) -> u32 {
        self.i().ctx().timeout_ms()
    }

    /// Evaluate a claripy AST to a single concrete value.
    ///
    /// Returns None if unsatisfiable or the expression cannot be evaluated.
    /// For values > 128 bits, use eval_wide which returns a Python int.
    pub fn eval(&self, py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
        // Fast path: concrete BVV doesn't need solver
        if let Some((value, _width)) = try_extract_bvv(ast) {
            return Ok(Some(value.into_pyobject(py)?.into()));
        }

        let ctx = self.i().ctx();

        // Try standard claripy → RustBV conversion first
        if let Ok(bv) = claripy_to_rustbv(py, ast, &ctx) {
            let width = bv.width();
            if width <= 128 {
                if let Some(v) = ctx.eval(&bv) {
                    return Ok(Some(v.into_pyobject(py)?.into()));
                }
                // fall through to Z3 AST pointer path
            } else if let Some(bytes) = ctx.eval_wide(&bv) {
                let py_bytes = pyo3::types::PyBytes::new(py, &bytes);
                let int_class = py.get_type::<pyo3::types::PyInt>();
                let py_int = int_class.call_method1("from_bytes", (py_bytes, "big"))?;
                return Ok(Some(py_int.into()));
                // None → fall through to Z3 AST pointer path
            }
        }

        // Z3 AST pointer fast path via shared context.
        // This handles: (1) expressions where claripy_to_rustbv creates a new
        // Z3 variable that has no constraints (e.g., stdin BVS from Python),
        // (2) complex expressions containing imported Z3 ASTs.
        // Using the original Z3 AST pointer preserves identity with
        // constraints already in the solver.
        #[cfg(feature = "vex-engine-z3")]
        {
            if let Ok(z3_ast) = extract_z3_ast_ptr(py, ast) {
                return self.eval_z3_ast_ptr(py, z3_ast, ast);
            }
        }
        Ok(None)
    }

    /// Evaluate several claripy ASTs against ONE model (angr-ue4ro).
    ///
    /// `eval` on each AST in turn is NOT equivalent: every call may land on a
    /// different satisfying assignment, so the results need not be jointly
    /// consistent. Callers that reassemble the parts into one value — the
    /// byte-`Extract` decomposition in `RustSolverFallback._rust_eval`, used
    /// when a >64-bit expression cannot be converted whole — must use this.
    ///
    /// Returns `None` if the constraints are unsat or any AST cannot be
    /// evaluated; the caller then falls back to the Python solver.
    pub fn eval_batch(
        &self,
        py: Python<'_>,
        asts: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Option<Vec<u128>>> {
        let ctx = self.i().ctx();
        let mut bvs = Vec::with_capacity(asts.len());
        for ast in &asts {
            if let Some((value, width)) = try_extract_bvv(ast) {
                bvs.push(RustBV::concrete(value, width));
                continue;
            }
            match self.ast_to_bv_for_eval(py, ast, &ctx) {
                Some(bv) => bvs.push(bv),
                None => return Ok(None),
            }
        }
        Ok(ctx.eval_many(&bvs))
    }

    /// Evaluate a claripy AST and return up to n solutions.
    pub fn eval_upto(
        &self,
        py: Python<'_>,
        ast: &Bound<'_, PyAny>,
        n: usize,
    ) -> PyResult<Py<PyAny>> {
        let result_list = pyo3::types::PyList::empty(py);

        // Fast path: concrete BVV has exactly one solution
        if n > 0
            && let Some((value, _width)) = try_extract_bvv(ast)
        {
            result_list.append(value.into_pyobject(py)?)?;
            return Ok(result_list.into());
        }

        let width: u32 = ast
            .getattr("length")
            .and_then(|l| l.extract())
            .unwrap_or(64);
        let is_wide = width > 128;

        let ctx = self.i().ctx();
        let bv = match claripy_to_rustbv(py, ast, &ctx) {
            Ok(bv) => bv,
            Err(_) => {
                // Z3 fast path for complex expressions
                #[cfg(feature = "vex-engine-z3")]
                {
                    use z3::ast::Ast;
                    if let Ok(z3_ast) = extract_z3_ast_ptr(py, ast) {
                        // angr-58ks: verify the Z3 sort before wrapping (see
                        // eval_z3_ast_ptr). A Bool-sorted AST must be lowered
                        // to a 1-bit BV, never BV-wrapped — the latter trips
                        // Z3's process-aborting error handler.
                        if z3_ast.is_bool() {
                            // SAFETY: `z3_ast` is a live Bool-sorted Z3_ast
                            // (verified via `is_bool()`). `Bool::wrap` takes
                            // its own ref; `ite` lowers it to a 1-bit BV.
                            unsafe {
                                let z3_ctx = z3::Context::thread_local();
                                let z3_bool = z3::ast::Bool::wrap(&z3_ctx, z3_ast.as_z3_ast());
                                let as_bv = z3_bool.ite(
                                    &z3::ast::BV::from_u64(1, 1),
                                    &z3::ast::BV::from_u64(0, 1),
                                );
                                RustBV::Symbolic {
                                    id: 0,
                                    ast: as_bv,
                                    width: 1,
                                    name: Arc::from(""),
                                }
                            }
                        } else if z3_ast.is_bv() {
                            // SAFETY: `z3_ast` is a live BV-sorted Z3_ast
                            // (verified via `is_bv()`); width comes from
                            // claripy and matches the BV sort. `BV::wrap`
                            // takes its own ref.
                            let z3_bv = unsafe {
                                let z3_ctx = z3::Context::thread_local();
                                z3::ast::BV::wrap(&z3_ctx, z3_ast.as_z3_ast())
                            };
                            RustBV::Symbolic {
                                id: 0,
                                ast: z3_bv,
                                width,
                                name: Arc::from(""),
                            }
                        } else {
                            return Err(PyRuntimeError::new_err(format!(
                                "eval_upto: Z3 AST has unsupported sort kind {:?} (expected BV or Bool)",
                                z3_ast.sort_kind()
                            )));
                        }
                    } else {
                        return Ok(result_list.into());
                    }
                }
                #[cfg(not(feature = "vex-engine-z3"))]
                {
                    return Ok(result_list.into());
                }
            }
        };

        if is_wide {
            // Wide values: use eval_upto_wide to get full-precision bytes
            let results = ctx.eval_upto_wide(&bv, n);
            let int_class = py.get_type::<pyo3::types::PyInt>();
            for bytes in results {
                let py_bytes = pyo3::types::PyBytes::new(py, &bytes);
                let py_int = int_class.call_method1("from_bytes", (py_bytes, "big"))?;
                result_list.append(py_int)?;
            }
        } else {
            // Standard path: u128 values
            let results = ctx.eval_upto(&bv, n);
            for v in results {
                result_list.append(v.into_pyobject(py)?)?;
            }
        }
        Ok(result_list.into())
    }

    /// Get the minimum value of a claripy AST.
    #[pyo3(signature = (ast, signed=false))]
    pub fn min(
        &self,
        py: Python<'_>,
        ast: &Bound<'_, PyAny>,
        signed: bool,
    ) -> PyResult<Option<u128>> {
        // Fast path: concrete BVV
        if let Some((value, _width)) = try_extract_bvv(ast) {
            return Ok(Some(value));
        }
        let ctx = self.i().ctx();
        let bv = claripy_to_rustbv(py, ast, &ctx)?;
        Ok(ctx.min(&bv, signed))
    }

    /// Get the maximum value of a claripy AST.
    #[pyo3(signature = (ast, signed=false))]
    pub fn max(
        &self,
        py: Python<'_>,
        ast: &Bound<'_, PyAny>,
        signed: bool,
    ) -> PyResult<Option<u128>> {
        // Fast path: concrete BVV
        if let Some((value, _width)) = try_extract_bvv(ast) {
            return Ok(Some(value));
        }
        let ctx = self.i().ctx();
        let bv = claripy_to_rustbv(py, ast, &ctx)?;
        Ok(ctx.max(&bv, signed))
    }

    /// Check if the given AST is definitely true.
    pub fn is_true(&self, py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<bool> {
        let ctx = self.i().ctx();
        let bv = claripy_to_rustbv(py, ast, &ctx)?;
        if bv.width() != 1 {
            return Ok(false);
        }

        // Definitely true if it can only be true
        let can_be_false = ctx.can_be_false(&bv);
        Ok(!can_be_false)
    }

    /// Check if the given AST is definitely false.
    pub fn is_false(&self, py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<bool> {
        let ctx = self.i().ctx();
        let bv = claripy_to_rustbv(py, ast, &ctx)?;
        if bv.width() != 1 {
            return Ok(false);
        }

        // Definitely false if it can only be false
        let can_be_true = ctx.can_be_true(&bv);
        Ok(!can_be_true)
    }

    /// Check if a specific value is a valid solution for an AST.
    pub fn solution(&self, py: Python<'_>, ast: &Bound<'_, PyAny>, value: u128) -> PyResult<bool> {
        let ctx = self.i().ctx();
        let bv = claripy_to_rustbv(py, ast, &ctx)?;
        Ok(ctx.solution(&bv, value))
    }

    /// Save solver state for temporary constraints.
    pub fn push(&self) {
        self.i().ctx().push();
    }

    /// Restore solver state.
    ///
    /// Refuses an unbalanced pop (no matching `push()`) with a `ValueError`
    /// instead of letting it reach z3-rs's under-pop panic (angr-ph300.48).
    pub fn pop(&self) -> PyResult<()> {
        if !self.i().ctx().try_pop() {
            return Err(PyValueError::new_err(
                "pop() with no matching push() — solver scope stack is empty",
            ));
        }
        Ok(())
    }

    /// Fork the solver context.
    ///
    /// Returns a new RustSolverContext with the same constraints and symbol table.
    /// With z3-rs 0.19+, all solvers on the same thread share the
    /// thread-local Z3 context.
    pub fn fork(&self) -> Self {
        let ctx = self.i().ctx();
        let forked_sym_ctx = ctx.fork();
        let forked_symbol_table = self.i().symbol_table.fork();
        RustSolverContext {
            owner: std::thread::current().id(),
            inner: Some(Box::new(SolverInner {
                sym_ctx: SolverCtxStorage::Owned(forked_sym_ctx),
                symbol_table: forked_symbol_table,
            })),
        }
    }

    /// Deterministically release the inner payload on the owning thread.
    ///
    /// Takes the `Box<SolverInner>` out and drops it here (the thread that
    /// created it), then leaves `self` an empty shell. Idempotent — a second
    /// call is a no-op.
    ///
    /// Callers (RustStateProxy / state export) invoke this at proxy
    /// invalidation instead of relying on Python GC. Under
    /// RUST_PARALLEL_WORKERS>1 a dead context can otherwise be collected on a
    /// scheduler worker thread, where pyo3 refuses to drop this `unsendable`
    /// pyclass and permanently leaks the owned SymContext + cloned Z3 solver
    /// (angr-87e56). After `close()` the later off-thread tp_dealloc finds
    /// `None` and drops nothing. Any solver method invoked after `close()`
    /// panics (surfaced as `PyRuntimeError`) — see [`Self::i`].
    pub fn close(&mut self) {
        // Only drop on the owning thread. A `close()` routed here from a
        // scheduler worker (Python graveyard drain) must NOT move the
        // non-Send SymContext/Rc drop off-owner — that is the exact
        // unsoundness `unsendable` guards. Off-owner we leave `inner`
        // intact; the later tp_dealloc (also off-owner) is refused by
        // pyo3, preserving today's leak-safe behavior. On-owner we drop
        // here, so a subsequent off-owner tp_dealloc finds `None`.
        if std::thread::current().id() == self.owner {
            drop(self.inner.take());
        }
    }

    /// Whether this context still holds its payload (`false` after `close()`).
    pub fn is_closed(&self) -> bool {
        self.inner.is_none()
    }

    /// Pop solver state multiple times.
    ///
    /// This is used for deferred fork processing to restore solver state
    /// to before specific branch constraints were added.
    pub fn pop_to_level(&self, target_level: u32, current_level: u32) -> PyResult<()> {
        let pops = current_level.saturating_sub(target_level);
        let ctx = self.i().ctx();
        for _ in 0..pops {
            if !ctx.try_pop() {
                return Err(PyValueError::new_err(format!(
                    "pop_to_level({target_level}, {current_level}) exceeds the \
                     solver scope depth — no matching push()",
                )));
            }
        }
        Ok(())
    }

    /// Add a constraint that a 1-bit value is true.
    ///
    /// Used for applying branch constraints during fork processing.
    pub fn assume_true_ast(&self, py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<()> {
        let ctx = self.i().ctx();
        let bv = claripy_to_rustbv(py, ast, &ctx)?;
        if bv.width() != 1 {
            return Err(PyRuntimeError::new_err(format!(
                "constraint must be a 1-bit value, got {} bits",
                bv.width()
            )));
        }
        #[cfg(feature = "vex-engine-z3")]
        {
            ctx.assume_true(&bv);
        }
        Ok(())
    }

    /// Add a constraint that a 1-bit value is false.
    ///
    /// Used for applying negated branch constraints during fork processing.
    pub fn assume_false_ast(&self, py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<()> {
        let ctx = self.i().ctx();
        let bv = claripy_to_rustbv(py, ast, &ctx)?;
        if bv.width() != 1 {
            return Err(PyRuntimeError::new_err(format!(
                "constraint must be a 1-bit value, got {} bits",
                bv.width()
            )));
        }
        #[cfg(feature = "vex-engine-z3")]
        {
            ctx.assume_false(&bv);
        }
        Ok(())
    }

    /// Get the number of constraints.
    pub fn num_constraints(&self) -> usize {
        self.i().ctx().num_constraints()
    }

    /// Create a new symbolic bitvector name.
    ///
    /// Returns a unique name for a symbol.
    pub fn unique_name(&self, base: &str) -> String {
        self.i().ctx().unique_name(base)
    }

    /// Check if Z3 is available.
    #[staticmethod]
    pub fn z3_available() -> bool {
        cfg!(feature = "vex-engine-z3")
    }

    /// Check if this solver context is shared (borrowed from a pending state).
    /// Shared solvers write constraints directly to the state, so constraint
    /// sync after callbacks can be skipped.
    pub fn is_shared(&self) -> bool {
        matches!(self.i().sym_ctx, SolverCtxStorage::Shared(_))
    }

    /// Get the unsat core as indices of constraints added.
    ///
    /// Returns the indices of constraints that form the unsatisfiable core.
    /// Call this after checking satisfiability and finding UNSAT.
    pub fn unsat_core(&self) -> PyResult<Vec<usize>> {
        Ok(self.i().ctx().unsat_core())
    }

    /// Get all Z3 solver assertions as strings.
    ///
    /// Returns string representations of all active constraints in the Z3 solver.
    /// Useful for debugging and for verifying constraint sync between Rust and Python.
    pub fn get_all_constraints_str(&self) -> Vec<String> {
        self.i().ctx().get_all_constraints_str()
    }

    /// Get the number of assertions in the Z3 solver.
    ///
    /// Returns the total count of active constraints.
    pub fn z3_assertion_count(&self) -> usize {
        self.i().ctx().z3_assertion_count()
    }

    /// Export constraints as serialized data for Python sync.
    ///
    /// This returns a list of (description, is_trackable) tuples for each
    /// constraint in the solver. The descriptions can be used for debugging
    /// and the is_trackable flag indicates if the constraint could be
    /// reconstructed from tracked handles.
    ///
    /// Note: Full Z3->claripy AST conversion is complex. This method provides
    /// constraint info for debugging. The primary sync mechanism is through
    /// the bidirectional constraint flow via add_constraints_to_pending.
    pub fn export_constraint_info(&self) -> Vec<(String, bool)> {
        self.i()
            .ctx()
            .get_all_constraints_str()
            .into_iter()
            .map(|s| (s, true)) // All Z3 constraints are trackable
            .collect()
    }

    /// Get the number of new constraints added since last sync.
    ///
    /// This helps track constraint growth during callbacks.
    pub fn constraint_delta(&self, baseline: usize) -> usize {
        let current = self.i().ctx().num_constraints();
        current.saturating_sub(baseline)
    }

    // =========================================================================
    // Handle-based API (Claripy Bypass)
    // These methods allow Python to perform symbolic operations without
    // converting to/from claripy ASTs, providing significant speedups.
    // =========================================================================

    /// Create a new symbolic bitvector and return a handle.
    ///
    /// This bypasses claripy.BVS() for native Rust symbolic value creation.
    pub fn create_symbolic(&self, name: &str, width: u32) -> RustBVHandle {
        let ctx = self.i().ctx();
        self.i().symbol_table.create_symbolic(&ctx, name, width)
    }

    /// Create a new concrete bitvector and return a handle.
    ///
    /// This bypasses claripy.BVV() for native Rust concrete value creation.
    pub fn create_concrete(&self, value: u128, width: u32) -> RustBVHandle {
        self.i().symbol_table.create_concrete(value, width)
    }

    /// Evaluate a handle to get a concrete value.
    ///
    /// Returns None if unsatisfiable or the value cannot be evaluated.
    pub fn eval_handle(&self, handle_id: u64) -> Option<u128> {
        let bv = self.i().symbol_table.get(handle_id)?;
        self.i().ctx().eval(&bv)
    }

    /// Get the minimum value for a handle.
    #[pyo3(signature = (handle_id, signed=false))]
    pub fn min_handle(&self, handle_id: u64, signed: bool) -> Option<u128> {
        let bv = self.i().symbol_table.get(handle_id)?;
        self.i().ctx().min(&bv, signed)
    }

    /// Get the maximum value for a handle.
    #[pyo3(signature = (handle_id, signed=false))]
    pub fn max_handle(&self, handle_id: u64, signed: bool) -> Option<u128> {
        let bv = self.i().symbol_table.get(handle_id)?;
        self.i().ctx().max(&bv, signed)
    }

    /// Evaluate a handle and return up to n solutions.
    pub fn eval_upto_handle(&self, handle_id: u64, n: usize) -> Vec<u128> {
        if let Some(bv) = self.i().symbol_table.get(handle_id) {
            self.i().ctx().eval_upto(&bv, n)
        } else {
            Vec::new()
        }
    }

    /// Add a constraint from a handle (must be 1-bit).
    pub fn add_constraint_handle(&self, handle_id: u64) -> PyResult<()> {
        let bv = self
            .i()
            .symbol_table
            .get(handle_id)
            .ok_or_else(|| invalid_handle_id(&[handle_id]))?;

        #[cfg(feature = "vex-engine-z3")]
        {
            let ctx = self.i().ctx();
            if bv.width() == 1 {
                ctx.assume_true(&bv);
            } else {
                let zero = RustBV::concrete(0, bv.width());
                let neq = bv.ne(&zero, &ctx);
                ctx.assume_true(&neq);
            }
        }

        Ok(())
    }

    /// Check if a specific value is a valid solution for a handle.
    pub fn solution_handle(&self, handle_id: u64, value: u128) -> bool {
        if let Some(bv) = self.i().symbol_table.get(handle_id) {
            self.i().ctx().solution(&bv, value)
        } else {
            false
        }
    }

    /// Get the number of handles in the symbol table.
    pub fn handle_count(&self) -> usize {
        self.i().symbol_table.len()
    }

    /// Convert a claripy AST to a handle.
    ///
    /// This performs a one-time conversion of a claripy AST to a RustBV,
    /// storing it in the symbol table and returning a handle. Subsequent
    /// operations can use the handle directly, avoiding repeated AST traversal.
    ///
    /// This is the key optimization for the callback return path: instead of
    /// returning a claripy AST that must be traversed on every use, we convert
    /// once and return a handle for O(1) lookups.
    pub fn claripy_ast_to_handle(
        &self,
        py: Python<'_>,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        let bv = claripy_to_rustbv(py, ast, &ctx)?;
        Ok(self.i().symbol_table.insert(bv))
    }

    // =========================================================================
    // Handle-based Arithmetic Operations
    // =========================================================================

    /// Add two handles and return a new handle.
    pub fn op_add(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_add(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Subtract two handles and return a new handle.
    pub fn op_sub(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_sub(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Multiply two handles and return a new handle.
    pub fn op_mul(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_mul(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Unsigned division of two handles.
    pub fn op_udiv(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_udiv(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Signed division of two handles.
    pub fn op_sdiv(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_sdiv(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Unsigned remainder of two handles.
    pub fn op_urem(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_urem(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Signed remainder of two handles.
    pub fn op_srem(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_srem(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Negation of a handle.
    pub fn op_neg(&self, a_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_neg(a_id, &ctx)
            .ok_or_else(|| invalid_handle_id(&[a_id]))
    }

    // =========================================================================
    // Handle-based Bitwise Operations
    // =========================================================================

    /// Bitwise AND of two handles.
    pub fn op_and(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_and(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Bitwise OR of two handles.
    pub fn op_or(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_or(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Bitwise XOR of two handles.
    pub fn op_xor(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_xor(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Bitwise NOT of a handle.
    pub fn op_not(&self, a_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_not(a_id, &ctx)
            .ok_or_else(|| invalid_handle_id(&[a_id]))
    }

    // =========================================================================
    // Handle-based Shift Operations
    // =========================================================================

    /// Left shift.
    pub fn op_shl(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_shl(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Logical right shift.
    pub fn op_lshr(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_lshr(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Arithmetic right shift.
    pub fn op_ashr(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_ashr(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Rotate left.
    pub fn op_rotl(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_rotl(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Rotate right.
    pub fn op_rotr(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_rotr(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    // =========================================================================
    // Handle-based Comparison Operations
    // =========================================================================

    /// Equality comparison (returns 1-bit handle).
    pub fn op_eq(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_eq(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Inequality comparison (returns 1-bit handle).
    pub fn op_ne(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_ne(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Unsigned less than.
    pub fn op_ult(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_ult(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Unsigned less than or equal.
    pub fn op_ule(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_ule(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Unsigned greater than.
    pub fn op_ugt(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_ugt(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Unsigned greater than or equal.
    pub fn op_uge(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_uge(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Signed less than.
    pub fn op_slt(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_slt(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Signed less than or equal.
    pub fn op_sle(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_sle(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Signed greater than.
    pub fn op_sgt(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_sgt(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    /// Signed greater than or equal.
    pub fn op_sge(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_sge(a_id, b_id, &ctx)
            .map_err(PyErr::from)
    }

    // =========================================================================
    // Handle-based Conversion Operations
    // =========================================================================

    /// Zero-extend to a wider width.
    pub fn op_zero_extend(&self, a_id: u64, to_width: u32) -> PyResult<RustBVHandle> {
        self.reject_extend_narrowing("op_zero_extend", a_id, to_width)?;
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_zero_extend(a_id, to_width, &ctx)
            .ok_or_else(|| invalid_handle_id(&[a_id]))
    }

    /// Sign-extend to a wider width.
    pub fn op_sign_extend(&self, a_id: u64, to_width: u32) -> PyResult<RustBVHandle> {
        self.reject_extend_narrowing("op_sign_extend", a_id, to_width)?;
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_sign_extend(a_id, to_width, &ctx)
            .ok_or_else(|| invalid_handle_id(&[a_id]))
    }

    /// Truncate to a narrower width.
    pub fn op_truncate(&self, a_id: u64, to_width: u32) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_truncate(a_id, to_width, &ctx)
            .ok_or_else(|| invalid_handle_id(&[a_id]))
    }

    /// Extract bits \[high:low\] (inclusive).
    pub fn op_extract(&self, a_id: u64, high: u32, low: u32) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_extract(a_id, high, low, &ctx)
            .ok_or_else(|| invalid_handle_id(&[a_id]))
    }

    /// Concatenate two values (a becomes high bits).
    pub fn op_concat(&self, a_id: u64, b_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_concat(a_id, b_id, &ctx)
            .ok_or_else(|| invalid_handle_id(&[a_id, b_id]))
    }

    /// If-then-else: if cond then then_val else else_val.
    pub fn op_ite(&self, cond_id: u64, then_id: u64, else_id: u64) -> PyResult<RustBVHandle> {
        let ctx = self.i().ctx();
        self.i()
            .symbol_table
            .op_ite(cond_id, then_id, else_id, &ctx)
            .ok_or_else(|| invalid_handle_id(&[cond_id, then_id, else_id]))
    }

    // =========================================================================
    // Solver Profiling Stats
    // =========================================================================

    /// Get global Z3 solver profiling stats as a dict.
    #[staticmethod]
    pub fn get_solver_stats() -> std::collections::HashMap<String, u64> {
        crate::symbolic::get_solver_stats()
    }

    /// Reset global Z3 solver profiling stats to zero.
    #[staticmethod]
    pub fn reset_solver_stats() {
        crate::symbolic::reset_solver_stats()
    }
}

impl Default for RustSolverContext {
    fn default() -> Self {
        Self::new()
    }
}

impl RustSolverContext {
    /// Borrow the inner payload.
    ///
    /// Panics if the context has already been [`close`](Self::close)d. The
    /// Python contract is that `close()` is called only at proxy invalidation
    /// / wave teardown, right before the reference is dropped — so no method
    /// call ever races an emptied context. A panic here (surfaced as a
    /// `PyRuntimeError` through pyo3) therefore signals a genuine
    /// use-after-close bug, not a normal control-flow path.
    #[inline]
    #[allow(
        clippy::expect_used,
        reason = "use-after-close is an internal-invariant violation, not an input path: the Python contract calls close() only at proxy invalidation / wave teardown, right before drop, so no method call races an emptied context"
    )]
    fn i(&self) -> &SolverInner {
        self.inner
            .as_ref()
            .expect("RustSolverContext used after close()")
    }

    /// Reject a narrowing width passed to an extend op at the Python boundary.
    ///
    /// `op_zero_extend`/`op_sign_extend` with `to_width < source_width` would
    /// otherwise return a handle wider than the caller asked for, which only
    /// surfaces much later as a Z3 sort error (abort under `panic=abort`) or a
    /// silent `eq_into`-False — far from the misuse site (angr-ph300.38). A
    /// missing handle is left to the op's own `invalid_handle_id` path.
    fn reject_extend_narrowing(&self, op: &str, a_id: u64, to_width: u32) -> PyResult<()> {
        if let Some(width) = self.i().symbol_table.with_value(a_id, |bv| bv.width())
            && to_width < width
        {
            return Err(PyValueError::new_err(format!(
                "{op}: to_width {to_width} < source width {width}; \
                 use op_truncate/op_extract to narrow"
            )));
        }
        Ok(())
    }

    /// Lower a claripy AST to a `RustBV` usable for evaluation, mirroring the
    /// conversion order in [`RustSolverContext::eval`]: the standard
    /// claripy → RustBV import first, then the raw Z3 AST pointer (which
    /// preserves identity with constraints already asserted in the solver).
    /// Returns `None` when neither path applies. Used by `eval_batch`.
    fn ast_to_bv_for_eval(
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
            use z3::ast::Ast;
            let z3_ast = extract_z3_ast_ptr(py, ast).ok()?;
            if !z3_ast.is_bv() {
                return None;
            }
            let width: u32 = ast.getattr("length").ok()?.extract().ok()?;
            // SAFETY: `z3_ast` is a live BV-sorted `Z3_ast` (checked via
            // `is_bv()`; Z3AstPtr holds an active ref) and `width` is claripy's
            // matching length. `BV::wrap` takes its own ref.
            let z3_bv = unsafe {
                let z3_ctx = z3::Context::thread_local();
                z3::ast::BV::wrap(&z3_ctx, z3_ast.as_z3_ast())
            };
            Some(RustBV::Symbolic {
                id: 0,
                ast: z3_bv,
                width,
                name: Arc::from(""),
            })
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
    fn eval_z3_ast_ptr(
        &self,
        py: Python<'_>,
        z3_ast: Z3AstPtr,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<PyAny>>> {
        use z3::ast::Ast;
        let ctx = self.i().ctx();

        // angr-58ks: verify the Z3 sort before wrapping. A Bool-sorted AST
        // (e.g. an fpEQ comparison that `claripy_to_rustbv` cannot lower and
        // so reaches this raw path with claripy `length == None`) must NOT be
        // BV-wrapped — operating on the mis-sorted node trips Z3's error
        // handler, which aborts the process instead of returning a PyErr.
        // Lower a Bool to a 1-bit BV (1 when true, 0 when false) and eval it.
        if z3_ast.is_bool() {
            // SAFETY: `z3_ast` is a live Bool-sorted Z3_ast (verified via
            // `is_bool()`; the Z3AstPtr holds an active ref). `Bool::wrap`
            // takes its own ref; `ite` lowers it to a 1-bit BV.
            let bv = unsafe {
                let z3_ctx = z3::Context::thread_local();
                let z3_bool = z3::ast::Bool::wrap(&z3_ctx, z3_ast.as_z3_ast());
                let as_bv = z3_bool.ite(&z3::ast::BV::from_u64(1, 1), &z3::ast::BV::from_u64(0, 1));
                RustBV::Symbolic {
                    id: 0,
                    ast: as_bv,
                    width: 1,
                    name: Arc::from(""),
                }
            };
            return match ctx.eval(&bv) {
                Some(v) => Ok(Some(v.into_pyobject(py)?.into())),
                None => Ok(None),
            };
        }
        if !z3_ast.is_bv() {
            return Err(PyRuntimeError::new_err(format!(
                "eval: Z3 AST has unsupported sort kind {:?} (expected BV or Bool)",
                z3_ast.sort_kind()
            )));
        }

        // Get the bit width from claripy
        let width: u32 = match ast.getattr("length") {
            Ok(l) => l.extract().unwrap_or(64),
            Err(_) => 64,
        };

        // SAFETY: `z3_ast` is a live BV-sorted `Z3_ast` (verified via
        // `is_bv()` above; Z3AstPtr holds an active ref). The width came from
        // claripy's `length` attribute, which matches the BV-sortedness of
        // the underlying AST in claripy's z3 backend. `BV::wrap` takes its
        // own ref.
        let z3_bv = unsafe {
            let z3_ctx = z3::Context::thread_local();
            z3::ast::BV::wrap(&z3_ctx, z3_ast.as_z3_ast())
        };
        let bv = RustBV::Symbolic {
            id: 0,
            ast: z3_bv,
            width,
            name: Arc::from(""),
        };

        if width <= 128 {
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

    /// Create a RustSolverContext from an existing SymContext.
    ///
    /// This is used when forking solver contexts during callback handling,
    /// allowing Python callbacks to inherit the full constraint context
    /// from Rust exploration.
    pub fn from_sym_context(sym_ctx: SymContext) -> Self {
        RustSolverContext {
            owner: std::thread::current().id(),
            inner: Some(Box::new(SolverInner {
                sym_ctx: SolverCtxStorage::Owned(sym_ctx),
                symbol_table: RustSymbolTable::new(),
            })),
        }
    }

    /// Create a RustSolverContext from an existing SymContext with a forked symbol table.
    ///
    /// This preserves both constraints and symbolic variable mappings.
    pub fn from_sym_context_with_symbols(
        sym_ctx: SymContext,
        symbol_table: RustSymbolTable,
    ) -> Self {
        RustSolverContext {
            owner: std::thread::current().id(),
            inner: Some(Box::new(SolverInner {
                sym_ctx: SolverCtxStorage::Owned(sym_ctx),
                symbol_table,
            })),
        }
    }

    /// Create a RustSolverContext that shares the solver from a state's `Rc<RefCell<SymContext>>`.
    ///
    /// This is O(1) — just an Rc clone (reference count increment) instead of
    /// a full Z3 solver clone (~3ms). The shared solver writes constraints directly
    /// to the pending state, eliminating the need for post-callback constraint sync.
    ///
    /// Safety: Only use when Rust exploration is suspended (during Python callbacks).
    pub fn from_shared_sym_context(shared: Rc<RefCell<SymContext>>) -> Self {
        RustSolverContext {
            owner: std::thread::current().id(),
            inner: Some(Box::new(SolverInner {
                sym_ctx: SolverCtxStorage::Shared(shared),
                symbol_table: RustSymbolTable::new(),
            })),
        }
    }

    /// Get a reference to the inner SymContext.
    ///
    /// This is used by the Rust VEX engine to share the solver context,
    /// ensuring branch constraints are properly tracked during execution.
    /// Only works for owned contexts; returns None for shared contexts.
    pub fn sym_context(&self) -> Option<&SymContext> {
        match &self.i().sym_ctx {
            SolverCtxStorage::Owned(ctx) => Some(ctx),
            SolverCtxStorage::Shared(_) => None,
        }
    }

    /// Get a reference to the symbol table.
    ///
    /// This is used by the interpreter to look up handles returned from Python.
    pub fn symbol_table(&self) -> &RustSymbolTable {
        &self.i().symbol_table
    }
}

/// Register the solver module with Python.
pub fn solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustSolverContext>()?;
    Ok(())
}

#[cfg(test)]
#[path = "solver_tests.rs"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: unwrap/expect are the idiomatic assertion form and are not input-reachable"
)]
mod tests;
