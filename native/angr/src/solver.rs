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
//! - [`z3_ast_extract`]: claripy-node evaluation (`node_to_eval_bv` /
//!   `eval_claripy_node`);
//! - [`claripy_bridge::lower_claripy_bool`](crate::claripy_bridge::lower_claripy_bool):
//!   the sort-checked claripy → Z3 `Bool` lowering the raw constraint fast
//!   paths assert.
//!
//! Python-boundary module: `unwrap`/`expect` denied (angr-qwyti.11; panic
//! policy in `lib.rs`).
#![deny(clippy::unwrap_used, clippy::expect_used)]
// A raw-AST wrap needed here goes behind a sort check in z3-rs
// (`Dynamic::as_bool` / `as_bv`), never inline (angr-5mnx3.10, angr-mx00p.32).
// Unlike `deny`, `forbid` cannot be overridden by a nested
// `#[allow(unsafe_code)]` (angr-8ucb3).
#![forbid(unsafe_code)]

use crate::symbolic::ToZ3;
use pyo3::exceptions::{PyRecursionError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

use std::cell::{Ref, RefCell};
use std::rc::Rc;

use crate::claripy_bridge::{BridgeError, claripy_to_rustbv, clarirs_node, try_extract_bvv};
use crate::symbolic::{RustBV, SymContext};

use crate::claripy_bridge::lower_claripy_bool;
use z3_ast_extract::node_to_eval_bv;

pub(crate) mod z3_ast_extract;

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

/// Coerce a bitvector into the boolean constraint it stands for.
///
/// A 1-bit BV *is* the constraint, so it is returned borrowed; anything wider
/// is a truthiness test and lowers to `bv != 0`. This mirrors
/// `SymContext::assume_true`'s own symbolic path, and is shared by the three
/// sites that used to open-code it (angr-c7xno.95): `add_constraint_ast`'s and
/// `add_constraint_tracked_ast`'s slow paths. Callers either hand the result
/// to `assume_true` or take `.to_z3_bool()` of it.
///
/// `Cow` rather than a plain `RustBV` so the (common) 1-bit case stays
/// clone-free.
fn bool_constraint_bv<'a>(bv: &'a RustBV) -> std::borrow::Cow<'a, RustBV> {
    if bv.width() == 1 {
        std::borrow::Cow::Borrowed(bv)
    } else {
        let zero = RustBV::concrete(0, bv.width());
        std::borrow::Cow::Owned(bv.ne(&zero))
    }
}

/// Storage for SymContext: either owned or shared via Rc.
/// Shared mode allows Python callbacks to use the pending state's solver
/// directly (O(1)) instead of forking it (a full Z3 solver clone per callback).
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
    // (owned SymContext + cloned Z3 solver) *on the
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
            })),
        }
    }

    /// Add a constraint from a claripy AST.
    ///
    /// A Bool node is lowered straight to Z3 (`lower_claripy_bool`) and
    /// asserted as is; anything else is imported to a RustBV and asserted
    /// as its `!= 0` test.
    pub fn add_constraint_ast(&self, py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<()> {
        let ctx = self.i().ctx();

        // Fast path: lower the clarirs Bool node to Z3 directly.
        {
            // angr-58ks: `add_constraint_raw` takes a `Bool`, so only a
            // Bool-sorted lowering takes the raw fast path. A non-Bool (e.g.
            // a BV used as a truthiness constraint) falls through to the slow
            // path below, which lowers it to a proper `!= 0` Bool.
            if let Ok(z3_ast) = lower_claripy_bool(ast) {
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

        // Slow path: import to a RustBV and assert its truthiness test.
        let bv = claripy_to_rustbv(py, ast, &ctx)?;

        {
            ctx.assume_true(&bool_constraint_bv(&bv));
        }

        Ok(())
    }

    /// Add multiple constraints from claripy ASTs.
    ///
    /// Fast path: when every AST lowers to a Z3 Bool (the same fast path
    /// `add_constraint_ast` takes),
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

        {
            let ctx = self.i().ctx();
            let n = asts.len();
            let mut entries: Vec<(z3::ast::Bool, RustBV, bool)> = Vec::with_capacity(n);
            for ast in asts.iter() {
                // angr-58ks: a non-Bool AST cannot go through the raw batch
                // (`add_constraints_raw_batch` takes a `Bool`); drop to the
                // per-constraint slow path, which lowers it correctly.
                let Ok(ptr) = lower_claripy_bool(&ast) else {
                    break;
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
    /// `SymContext::add_constraint_tracked_indexed`, which uses
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
        {
            let ctx = self.i().ctx();
            // angr-d01qu: mirror add_constraint_ast's is_bool() gate. A
            // non-Bool AST wrapped as z3::ast::Bool would trip Z3's
            // CHECK_FORMULA sort-mismatch guard inside
            // Z3_solver_assert_and_track, which -- since our context installs
            // a no-op error handler -- fails *silently*: no panic, no abort,
            // just a constraint that never actually gets asserted while this
            // function still returns Ok(idx) as if tracking succeeded.
            // `lower_claripy_bool` refuses a non-Bool lowering, which falls
            // through to the slow path below.
            if let Ok(constraint) = lower_claripy_bool(ast) {
                // Cloned (a `Z3_inc_ref`) so the residual branch below still
                // has the Bool `add_constraint_tracked_indexed` consumes.
                let idx = ctx.add_constraint_tracked_indexed(constraint.clone());
                // angr-03vl4.74: same "exactly one log" invariant
                // `add_constraint_ast` upholds above — a constraint that is
                // live on the solver must be recorded in the assumed log or
                // the residual log, because every reconstruction path
                // (`to_snapshot`'s `dump_non_bv_smtlib2`, `unsat_core_assumed`,
                // `_export_state_constraints`) rebuilds from those two logs and
                // never from `z3_assertions`. This branch used to have no
                // `else`, so an AST `claripy_to_rustbv` cannot lower (an FP
                // comparison, say) was enforced in-process but vanished from
                // `state.solver.constraints`, from `unsat_core()`, and from any
                // snapshot/restore of the state.
                if let Ok(bv) = claripy_to_rustbv(py, ast, &ctx) {
                    ctx.assumed_constraints_push(bv, true);
                } else {
                    ctx.push_residual_assertion(constraint);
                }
                return Ok(idx);
            }

            // Slow path: build the Z3 Bool from a RustBV. Mirrors
            // assume_true's symbolic path (BV width 1 → bool via to_z3_bool;
            // wider BV → ne(0) bool) — see `bool_constraint_bv`.
            let bv = claripy_to_rustbv(py, ast, &ctx)?;
            // A constraint with no Z3 form raises (a RecursionError when too
            // deep), rather than being tracked as something it is not.
            let constraint = bool_constraint_bv(&bv)
                .to_z3_bool()
                .map_err(BridgeError::from)?;
            let idx = ctx.add_constraint_tracked_indexed(constraint);
            ctx.assumed_constraints_push(bv, true);
            Ok(idx)
        }
    }

    /// Check if the current constraints are satisfiable.
    ///
    /// Lenient, claripy-compatible form: an undecided query (Z3 Unknown /
    /// timeout) reports `False`, with a `log::warn!` from `SymContext::is_sat`
    /// so the collapse is at least observable. Callers that must distinguish
    /// "proven contradictory" from "solver gave up" — anything that would drop
    /// a state on `False` — should use `satisfiable_checked` (angr-03vl4.62).
    pub fn satisfiable(&self) -> bool {
        self.i().ctx().is_sat()
    }

    /// Satisfiability with the undecided case preserved: `True` / `False` for
    /// a decided query, `None` when Z3 returned Unknown (timeout).
    pub fn satisfiable_checked(&self) -> Option<bool> {
        self.i().ctx().is_sat_checked()
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
    /// Returns None when there is no model (unsat, or Unknown at the
    /// timeout); an AST the bridge cannot import raises. A value wider than
    /// 128 bits comes back as a Python int.
    pub fn eval(&self, py: Python<'_>, ast: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
        // Fast path: concrete BVV doesn't need solver
        if let Some((value, _width)) = try_extract_bvv(ast) {
            return Ok(Some(value.into_pyobject(py)?.into()));
        }

        let ctx = self.i().ctx();

        // An imported AST answers once: `None` is the context's no-model
        // answer (unsat, or Unknown at the timeout), and re-evaluating the
        // node would only rerun the same check (angr-vmi2l.220). The node
        // path is the fallback for an AST `claripy_to_rustbv` refuses.
        match claripy_to_rustbv(py, ast, &ctx) {
            Ok(bv) if bv.width() <= 128 => Ok(match ctx.eval(&bv) {
                Some(v) => Some(v.into_pyobject(py)?.into()),
                None => None,
            }),
            Ok(bv) => {
                let Some(bytes) = ctx.eval_wide(&bv) else {
                    // SILENT(cat-a): no model is eval's documented None answer.
                    return Ok(None);
                };
                let py_bytes = pyo3::types::PyBytes::new(py, &bytes);
                let int_class = py.get_type::<pyo3::types::PyInt>();
                let py_int = int_class.call_method1("from_bytes", (py_bytes, "big"))?;
                Ok(Some(py_int.into()))
            }
            Err(err) => {
                // `None` means no model, so an AST the bridge cannot import
                // raises rather than reading as unsat (angr-vmi2l.221).
                match clarirs_node(ast) {
                    Some(node) => self.eval_claripy_node(py, &node),
                    None => Err(err.into()),
                }
            }
        }
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
    ///
    /// Lenient, claripy-compatible form: an enumeration cut short by a Z3
    /// Unknown (timeout) is indistinguishable from an exhaustive one — both
    /// come back as a short list. Callers whose correctness depends on the set
    /// being complete — anything that would conclude "only these values are
    /// possible" — should use [`eval_upto_checked`](Self::eval_upto_checked)
    /// (the PyO3 twin of the internal fix) (angr-0qqun, angr-03vl4.85).
    pub fn eval_upto(
        &self,
        py: Python<'_>,
        ast: &Bound<'_, PyAny>,
        n: usize,
    ) -> PyResult<Py<PyAny>> {
        Ok(self.eval_upto_enumerate(py, ast, n)?.0)
    }

    /// [`eval_upto`](Self::eval_upto) with the undecided case preserved: the
    /// list of solutions for a decided enumeration, `None` when Z3 returned
    /// Unknown part-way through and the set is therefore not known to be
    /// complete.
    ///
    /// Mirrors [`satisfiable_checked`](Self::satisfiable_checked): `None` means
    /// "the solver gave up", never "no solutions". Note that a non-`None`
    /// result of exactly `n` values may still have hit the cap — request
    /// `n + 1` and treat `len > n` as an overflow, same as the Rust-side
    /// `Enumeration` contract.
    pub fn eval_upto_checked(
        &self,
        py: Python<'_>,
        ast: &Bound<'_, PyAny>,
        n: usize,
    ) -> PyResult<Option<Py<PyAny>>> {
        let (values, undecided) = self.eval_upto_enumerate(py, ast, n)?;
        Ok((!undecided).then_some(values))
    }

    /// Get the minimum value of a claripy AST.
    ///
    /// `None` when unsat. Converts through
    /// [`ast_to_bv_strict`](Self::ast_to_bv_strict), so a refused width
    /// raises instead of reaching Z3.
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
        let bv = self.ast_to_bv_strict(py, ast, &ctx)?;
        Ok(ctx.min(&bv, signed))
    }

    /// Get the maximum value of a claripy AST; converts as [`min`](Self::min).
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
        let bv = self.ast_to_bv_strict(py, ast, &ctx)?;
        Ok(ctx.max(&bv, signed))
    }

    /// Check if a specific value is a valid solution for an AST; converts as
    /// [`min`](Self::min).
    pub fn solution(&self, py: Python<'_>, ast: &Bound<'_, PyAny>, value: u128) -> PyResult<bool> {
        let ctx = self.i().ctx();
        let bv = self.ast_to_bv_strict(py, ast, &ctx)?;
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
    /// Returns a new RustSolverContext with the same constraints. Tracked
    /// constraints stay tracked: the child's `unsat_core` reports the
    /// parent's indices, and its own tracked adds continue after them.
    /// With z3-rs 0.19+, all solvers on the same thread share the
    /// thread-local Z3 context.
    pub fn fork(&self) -> Self {
        let ctx = self.i().ctx();
        let forked_sym_ctx = ctx.fork();
        RustSolverContext {
            owner: std::thread::current().id(),
            inner: Some(Box::new(SolverInner {
                sym_ctx: SolverCtxStorage::Owned(forked_sym_ctx),
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

    /// Shared body of [`eval_upto`](Self::eval_upto) and
    /// [`eval_upto_checked`](Self::eval_upto_checked): the solution list plus
    /// the `undecided` flag from the underlying [`Enumeration`].
    ///
    /// Every early return here is a *decided* outcome — a concrete fast path,
    /// or an expression this bridge cannot convert at all (the latter yields an
    /// empty list, which is what the lenient API has always returned and is not
    /// a Z3 verdict). Only the two `_checked` enumerations below can set the
    /// flag.
    ///
    /// [`Enumeration`]: crate::symbolic::Enumeration
    fn eval_upto_enumerate(
        &self,
        py: Python<'_>,
        ast: &Bound<'_, PyAny>,
        n: usize,
    ) -> PyResult<(Py<PyAny>, bool)> {
        let result_list = pyo3::types::PyList::empty(py);

        // Fast path: concrete BVV has exactly one solution
        if n > 0
            && let Some((value, _width)) = try_extract_bvv(ast)
        {
            result_list.append(value.into_pyobject(py)?)?;
            return Ok((result_list.into(), false));
        }

        let ctx = self.i().ctx();
        let bv = match claripy_to_rustbv(py, ast, &ctx) {
            Ok(bv) => bv,
            Err(_) => {
                // Z3 fast path for complex expressions
                {
                    let Some(node) = clarirs_node(ast) else {
                        return Ok((result_list.into(), false));
                    };
                    match node_to_eval_bv(&node) {
                        Ok(bv) => bv,
                        Err(sort) => {
                            return Err(PyRuntimeError::new_err(format!(
                                "eval_upto: claripy AST has unsupported sort {sort} (expected BV or Bool)"
                            )));
                        }
                    }
                }
            }
        };

        // The wide/narrow split follows the width of the BV we actually built
        // (claripy import, Bool→1-bit lowering, or Z3 sort), not a separately
        // read claripy `.length` that could disagree with it (angr-9ke6b.202).
        let undecided = if bv.width() > 128 {
            // Wide values: use eval_upto_wide to get full-precision bytes
            let results = ctx.eval_upto_wide_checked(&bv, n);
            let int_class = py.get_type::<pyo3::types::PyInt>();
            for bytes in &results.values {
                let py_bytes = pyo3::types::PyBytes::new(py, bytes);
                let py_int = int_class.call_method1("from_bytes", (py_bytes, "big"))?;
                result_list.append(py_int)?;
            }
            results.undecided
        } else {
            // Standard path: u128 values
            let results = ctx.eval_upto_checked(&bv, n);
            for v in results.values {
                result_list.append(v.into_pyobject(py)?)?;
            }
            results.undecided
        };
        Ok((result_list.into(), undecided))
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
            })),
        }
    }

    /// Create a RustSolverContext that shares the solver from a state's `Rc<RefCell<SymContext>>`.
    ///
    /// This is O(1) — just an Rc clone (reference count increment) instead of
    /// a full Z3 solver clone. The shared solver writes constraints directly
    /// to the pending state, eliminating the need for post-callback constraint sync.
    ///
    /// Safety: Only use when Rust exploration is suspended (during Python callbacks).
    pub(crate) fn from_shared_sym_context(shared: Rc<RefCell<SymContext>>) -> Self {
        RustSolverContext {
            owner: std::thread::current().id(),
            inner: Some(Box::new(SolverInner {
                sym_ctx: SolverCtxStorage::Shared(shared),
            })),
        }
    }
}

test_submod!("solver_tests.rs" => tests);
