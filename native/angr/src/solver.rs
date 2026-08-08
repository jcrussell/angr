//! PyO3-exposed Rust solver context.
//!
//! This module provides a Python-accessible constraint solver that uses
//! Z3 under the hood. It bridges claripy ASTs to Rust's SymContext.
//!
//! With z3-rs 0.19+, the Z3 context is thread-local, so we don't need
//! to manage explicit context lifetimes.
//!
//! What lives where (angr-9ke6b.205 split):
//!
//! - here: the Python-boundary error mapping, the [`RustSolverContext`]
//!   pyclass itself, and the "normal" claripy-AST solver API
//!   (`add_constraint*` / `eval*` / `min` / `max` / `push` / `pop` / `fork`);
//! - [`z3_ptr`]: raw `Z3_ast`-pointer extraction and evaluation, i.e. every
//!   `unsafe` in the solver surface;
//! - [`handle_api`]: the handle-based claripy-bypass API (symbol-table
//!   lifecycle plus the ~25 `op_*` arithmetic wrappers).
//!
//! Both submodules add their items to *this* type, so the split is purely
//! about where the source lives — the Python-visible surface is unchanged.
//!
//! This is a Python-boundary module; `unwrap`/`expect` are denied here so a
//! future panic-on-input landmine cannot be reintroduced without a reviewed,
//! reasoned `#[allow]` (angr-qwyti.11 enforcement layer). The `deny` covers
//! the submodules below too, since lint levels propagate into nested modules.
#![deny(clippy::unwrap_used, clippy::expect_used)]

mod handle_api;
mod z3_ptr;

use pyo3::exceptions::{PyRecursionError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

use std::cell::{Ref, RefCell};
use std::rc::Rc;

use crate::claripy_bridge::{BridgeError, claripy_to_rustbv, try_extract_bvv};
use crate::symbolic::{BinaryOpError, RustBV, RustSymbolTable, SymContext};

#[cfg(feature = "vex-engine-z3")]
use self::z3_ptr::{extract_z3_ast_ptr, z3_ast_to_eval_bv};
#[cfg(feature = "vex-engine-z3")]
use crate::symbolic::Z3AstPtr;

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
            // angr-2a3i9: surface as Python's own RecursionError so a caller
            // already prepared to catch that (e.g. from a plain Python
            // recursive helper) also catches a bridge depth-guard trip.
            BridgeError::RecursionLimit(_) => PyRecursionError::new_err(msg),
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
/// Reuses `invalid_handle_id` for the missing-handle case so its message
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
#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
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

#[allow(
    unreachable_pub,
    reason = "pyo3 `#[pymethods]`/`#[pyclass]` surface: these items are reached from Python, not from Rust. See the `unreachable_pub` note in lib.rs (angr-9ke6b.50)."
)]
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
    // Without Z3 the assert half compiles out and the slow path's `bv` is
    // unread — the conversion still runs for its `?` validation (angr-sqfj8.139).
    #[cfg_attr(
        not(feature = "vex-engine-z3"),
        allow(unused_variables, reason = "Z3-only consumer")
    )]
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
                // angr-sqfj8.121: a constraint with a RustBV form must land
                // in EXACTLY ONE of the residual (`add_constraint_raw`) or
                // assumed (`add_constraint_raw_assumed` +
                // `assumed_constraints_push`) logs, never both — see
                // `add_constraint_raw_assumed`'s doc comment for why
                // double-listing corrupts `unsat_core_assumed`. Mirrors the
                // already-correct `import_python_constraints` pattern
                // (`exploration/constraints.rs`): try the RustBV conversion
                // first and branch on it, instead of unconditionally calling
                // both.
                if let Ok(bv) = claripy_to_rustbv(py, ast, &ctx) {
                    ctx.add_constraint_raw_assumed(z3_ast);
                    ctx.assumed_constraints_push(bv, true);
                } else {
                    ctx.add_constraint_raw(z3_ast);
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
    /// one model invalidation pass.
    ///
    /// An AST that fails the raw extraction does *not* discard the work
    /// already done for its predecessors (angr-9ke6b.206): the successfully
    /// converted prefix is flushed as one raw batch, and only the failing
    /// AST and everything after it drops to the per-constraint slow path.
    /// Constraint order is unchanged, so behavior stays identical to the
    /// unbatched loop.
    pub fn add_constraints_ast(&self, py: Python<'_>, asts: &Bound<'_, PyList>) -> PyResult<()> {
        // Index of the first AST not yet asserted. The raw fast path below
        // advances it past every AST it manages to batch; the remainder goes
        // through `add_constraint_ast` one at a time.
        let resume_from: usize;

        #[cfg(not(feature = "vex-engine-z3"))]
        {
            resume_from = 0;
        }

        #[cfg(feature = "vex-engine-z3")]
        {
            let ctx = self.i().ctx();
            let n = asts.len();
            let mut entries: Vec<(Z3AstPtr, RustBV, bool)> = Vec::with_capacity(n);
            for ast in asts.iter() {
                let ptr = match extract_z3_ast_ptr(py, &ast) {
                    Ok(p) if p.is_bool() => p,
                    // angr-58ks: a non-Bool AST cannot go through the raw
                    // batch (add_constraints_raw_batch Bool-wraps by
                    // contract); drop to the per-constraint slow path which
                    // lowers it correctly.
                    Ok(_) | Err(_) => break,
                };
                let bv = match claripy_to_rustbv(py, &ast, &ctx) {
                    Ok(b) => b,
                    Err(_) => break,
                };
                entries.push((ptr, bv, true));
            }
            resume_from = entries.len();
            // No-op when the very first AST failed; otherwise the prefix
            // still gets the single-lock / single-invalidation batch.
            ctx.add_constraints_raw_batch(entries);
            if resume_from == n {
                return Ok(());
            }
        }
        // Slow path: the first AST that resisted the raw extraction, plus
        // everything after it, goes through the per-constraint route.
        for ast in asts.iter().skip(resume_from) {
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
        #[cfg(feature = "vex-engine-z3")]
        {
            // Bound inside the gate: the no-z3 arm below errors out without
            // ever touching the context (angr-sqfj8.139).
            let ctx = self.i().ctx();
            // angr-d01qu: mirror add_constraint_ast's is_bool() gate. A
            // non-Bool AST wrapped as z3::ast::Bool via Ast::wrap would trip
            // Z3's CHECK_FORMULA sort-mismatch guard inside
            // Z3_solver_assert_and_track, which -- since our context installs
            // a no-op error handler -- fails *silently*: no panic, no abort,
            // just a constraint that never actually gets asserted while this
            // function still returns Ok(idx) as if tracking succeeded. Only
            // take the raw fast path for Bool-sorted ASTs; fall through to
            // the slow path below for anything else.
            if let Ok(z3_ast) = extract_z3_ast_ptr(py, ast)
                && z3_ast.is_bool()
            {
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
                return self.eval_z3_ast_ptr(py, z3_ast);
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

        let ctx = self.i().ctx();
        let bv = match claripy_to_rustbv(py, ast, &ctx) {
            Ok(bv) => bv,
            Err(_) => {
                // Z3 fast path for complex expressions
                #[cfg(feature = "vex-engine-z3")]
                {
                    let Ok(z3_ast) = extract_z3_ast_ptr(py, ast) else {
                        return Ok(result_list.into());
                    };
                    match z3_ast_to_eval_bv(&z3_ast) {
                        Some(bv) => bv,
                        None => {
                            return Err(PyRuntimeError::new_err(format!(
                                "eval_upto: Z3 AST has unsupported sort kind {:?} (expected BV or Bool)",
                                z3_ast.sort_kind()
                            )));
                        }
                    }
                }
                #[cfg(not(feature = "vex-engine-z3"))]
                {
                    return Ok(result_list.into());
                }
            }
        };

        // The wide/narrow split follows the width of the BV we actually built
        // (claripy import, Bool→1-bit lowering, or Z3 sort), not a separately
        // read claripy `.length` that could disagree with it (angr-9ke6b.202).
        if bv.width() > 128 {
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

    /// Create a RustSolverContext from an existing SymContext.
    ///
    /// This is used when forking solver contexts during callback handling,
    /// allowing Python callbacks to inherit the full constraint context
    /// from Rust exploration.
    pub(crate) fn from_sym_context(sym_ctx: SymContext) -> Self {
        RustSolverContext {
            owner: std::thread::current().id(),
            inner: Some(Box::new(SolverInner {
                sym_ctx: SolverCtxStorage::Owned(sym_ctx),
                symbol_table: RustSymbolTable::new(),
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
    pub(crate) fn from_shared_sym_context(shared: Rc<RefCell<SymContext>>) -> Self {
        RustSolverContext {
            owner: std::thread::current().id(),
            inner: Some(Box::new(SolverInner {
                sym_ctx: SolverCtxStorage::Shared(shared),
                symbol_table: RustSymbolTable::new(),
            })),
        }
    }

    /// Get a reference to the symbol table.
    ///
    /// This is used by the interpreter to look up handles returned from Python.
    pub(crate) fn symbol_table(&self) -> &RustSymbolTable {
        &self.i().symbol_table
    }
}

test_submod!("solver_tests.rs" => tests);
