//! Constraint-related state for `RustExplorationManager`.
//!
//! Splits the manager's constraint configuration and tracking into two
//! sub-structs:
//!
//! * `ConstraintSolver` — solver configuration knobs (timeout, lazy solves).
//! * `ConstraintTracker` — per-run bookkeeping for the native uniqueness
//!   filter and the find/avoid predicate skip sets that prevent infinite
//!   loops after `resume_*_predicate(false)`.
//!
//! It also holds the two cfg-gated free functions that are the single seam for
//! the strict-deterministic flag — `apply_state_deterministic` and
//! `state_is_deterministic` — both inert on a non-`vex-engine-z3` build, plus
//! the claripy constraint-import seam: `import_one_constraint` (the single
//! per-constraint body, shared with `constraint_sync.rs`) and the
//! `import_python_constraints` list wrapper over it.
//!
//! Same pattern as `ProfilingCollector`: `pub(crate)` direct field access by
//! design — callers read/write the inner fields through a thin delegation.
//!
//! **Panic policy (angr-qwyti.11, angr-sqfj8.58):** `import_python_constraints`
//! takes a `Bound<'_, PyList>`, making this a Python-boundary module, so it
//! carries `#![deny(clippy::unwrap_used, clippy::expect_used)]`.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashSet;

use crate::symbolic::DEFAULT_SOLVER_TIMEOUT_MS;

/// Solver configuration knobs propagated to each per-state solver.
#[derive(Debug, Default)]
pub(crate) struct ConstraintSolver {
    /// When true, skip satisfiability checks on forked states (LAZY_SOLVES).
    pub(crate) lazy_solves: bool,
    /// Z3 solver timeout in milliseconds (default: [`DEFAULT_SOLVER_TIMEOUT_MS`]).
    pub(crate) solver_timeout_ms: u32,
    /// Strict-deterministic witness selection (angr-op0dn.10.3, M2.3). When
    /// true, every state entering a stash gets
    /// [`SymContext::set_deterministic`](crate::symbolic::SymContext::set_deterministic),
    /// so `eval` returns the unsigned-minimum witness and `eval_upto` the
    /// ascending prefix of the feasible set. Forks inherit from their parent,
    /// so seeding the states that enter the manager covers the lineage.
    pub(crate) deterministic: bool,
}

impl ConstraintSolver {
    pub(crate) fn new() -> Self {
        ConstraintSolver {
            lazy_solves: false,
            solver_timeout_ms: DEFAULT_SOLVER_TIMEOUT_MS,
            deterministic: false,
        }
    }
}

/// Apply strict-deterministic witness selection to one state's solver.
///
/// Single seam for the `deterministic` flag so `_create_state`, `_add_state`
/// and `set_deterministic` share one cfg-gated body: on a non-Z3 build there
/// is no `SymContext::set_deterministic` and the flag is inert.
#[cfg(feature = "vex-engine-z3")]
pub(crate) fn apply_state_deterministic(state: &crate::state::RustSimState, v: bool) {
    state.solver().borrow().set_deterministic(v);
}

#[cfg(not(feature = "vex-engine-z3"))]
pub(crate) fn apply_state_deterministic(_state: &crate::state::RustSimState, _v: bool) {}

/// Apply strict-deterministic witness selection to the one solver context a
/// parked callback owns that is NOT a `RustSimState` (angr-sqfj8.32).
///
/// A state in `pending_callbacks` lives in NO stash by design, so
/// `set_deterministic`'s stash loop cannot see it. Three contexts have to be
/// reached, because each one becomes the solver of a state that enters a stash
/// when the callback resumes:
///
/// * `pending.state` — the continuing successor `_resume_*` routes back to a
///   stash. Its `SymContext` is the same `Rc` that `pending.solver_ctx` wraps
///   (`from_shared_sym_context`, see the `PendingCallback::with_context` call
///   sites in `stepping_bounce.rs` and `run_loop_single.rs`), so the
///   Python-facing handle is covered with it.
/// * `pending.pre_callback_snapshot` — `_resume_after_simprocedure`'s
///   `fork_base`, i.e. the parent every materialized deferred fork forks off.
/// * `pending.fork_snapshots` — `fork_materialize::build_unexplored_fork` turns
///   each into a state via `RustSimState::fork_from_snapshot`, so the
///   snapshot's context *is* the fork's context and inheritance from
///   `fork_base` never happens for it.
///
/// The first two are `RustSimState`s, so
/// [`RustExplorationManager::all_live_states`](super::RustExplorationManager::all_live_states)
/// already yields them and `set_deterministic` covers them in its single
/// live-state walk (angr-0jh0j.82). A `BranchSnapshot` carries a raw
/// `solver`/memory sidecar instead, so the third is outside that walk's reach
/// and needs this call.
///
/// Missing any of the three silently leaves that lineage on the old
/// witness-selection mode with no error — the failure shape bd memory
/// `invariant-lineage-flag-propagation` describes.
#[cfg(feature = "vex-engine-z3")]
pub(crate) fn apply_fork_snapshots_deterministic(pending: &super::PendingCallback, v: bool) {
    for snapshot in pending.fork_snapshots.values() {
        snapshot.solver.set_deterministic(v);
    }
}

#[cfg(not(feature = "vex-engine-z3"))]
pub(crate) fn apply_fork_snapshots_deterministic(_pending: &super::PendingCallback, _v: bool) {}

/// Whether a state's solver is in strict-deterministic mode. Always false on
/// a non-Z3 build, where the mode does not exist.
#[cfg(feature = "vex-engine-z3")]
pub(crate) fn state_is_deterministic(state: &crate::state::RustSimState) -> bool {
    state.solver().borrow().is_deterministic()
}

#[cfg(not(feature = "vex-engine-z3"))]
pub(crate) fn state_is_deterministic(_state: &crate::state::RustSimState) -> bool {
    false
}

/// Which conversion tier [`import_one_constraint`] tries first.
///
/// Both orders are correct — the two tiers assert semantically the same
/// constraint, they differ only in *which* Z3 AST reaches the solver
/// (claripy's own, or the one `claripy_to_rustbv` re-lowers) and in which
/// export log records it. The order is a per-call-site choice, so it is a
/// parameter here rather than two hand-synced copies of the body
/// (angr-6cp06.26).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConstraintTier {
    /// Ask claripy's z3 backend for the raw pointer first, so what lands on
    /// the solver is claripy's own AST (lossless even when
    /// `claripy_to_rustbv` would have mis-modelled the op). Used by
    /// [`import_python_constraints`].
    Z3PtrFirst,
    /// Convert to a `RustBV` first and assume that. Used by
    /// [`RustExplorationManager::sync_constraints_from_python`](super::RustExplorationManager::sync_constraints_from_python) —
    /// see its doc comment for why that call site keeps the other order.
    RustBvFirst,
}

/// How [`import_one_constraint`] disposed of one constraint.
#[cfg_attr(
    not(feature = "vex-engine-z3"),
    allow(
        dead_code,
        reason = "RawOnly is only reachable through the Z3-pointer tier"
    )
)]
pub(crate) enum ConstraintImport {
    /// Asserted, with a width-1 `RustBV` form recorded in the assumed-constraint
    /// export log so a Python re-import can reconstruct it.
    Assumed,
    /// Asserted straight from claripy's raw Z3 pointer: `claripy_to_rustbv`
    /// declined the op (FP, etc), so only the residual log has it. Carries
    /// that error for the caller's log line.
    RawOnly(crate::claripy_bridge::BridgeError),
    /// Not asserted — neither tier could convert it.
    Failed(crate::claripy_bridge::BridgeError),
}

/// Resolve `claripy.backends.z3`, the entry point for the raw-Z3-pointer tier.
///
/// Resolved once per constraint *list* by both callers and handed to
/// [`import_one_constraint`], so a list of N constraints pays one attribute
/// walk rather than N.
///
/// SILENT(cat-a): probing for claripy's optional z3 backend. A missing backend
/// is expected control flow — `import_one_constraint` simply takes the RustBV
/// tier instead — so collapsing the error to `None` loses no correctness.
#[cfg(feature = "vex-engine-z3")]
pub(crate) fn resolve_claripy_z3_backend(
    py: pyo3::Python<'_>,
) -> Option<pyo3::Bound<'_, pyo3::types::PyAny>> {
    use pyo3::prelude::*;

    match py
        .import("claripy")
        .and_then(|c| c.getattr("backends"))
        .and_then(|b| b.getattr("z3"))
    {
        Ok(backend) => Some(backend),
        Err(e) => {
            log::debug!("claripy.backends.z3 unavailable for the raw-pointer tier: {e}");
            None
        }
    }
}

/// Non-Z3 mirror: there is no raw-pointer tier at all, so every caller's
/// backend is `None` and the cfg stays out of the call sites.
#[cfg(not(feature = "vex-engine-z3"))]
pub(crate) fn resolve_claripy_z3_backend(
    _py: pyo3::Python<'_>,
) -> Option<pyo3::Bound<'_, pyo3::types::PyAny>> {
    None
}

/// Pull the raw `Z3_ast` behind a claripy AST via
/// `claripy.backends.z3.convert(ast).as_ast().value`.
///
/// SILENT(cat-a): a claripy AST the backend declines to convert (or whose
/// pointer comes back null) is expected control flow — the caller falls
/// through to the RustBV tier, which is an equally complete alternative, not a
/// degraded one.
///
/// The returned handle owns its own `Z3_inc_ref`, so dropping it does not
/// disturb claripy's cached AST.
#[cfg(feature = "vex-engine-z3")]
fn claripy_z3_ptr(
    z3_backend: &pyo3::Bound<'_, pyo3::types::PyAny>,
    item: &pyo3::Bound<'_, pyo3::types::PyAny>,
) -> Option<crate::symbolic::Z3AstPtr> {
    use pyo3::prelude::*;

    if let Ok(z3_obj) = z3_backend.call_method1("convert", (item,))
        && let Ok(ast_ref) = z3_obj.call_method0("as_ast")
        && let Ok(ptr) = ast_ref.getattr("value").and_then(|v| v.extract::<usize>())
    {
        let z3_ctx = z3::Context::thread_local();
        // SAFETY: claripy's z3 backend returned this pointer for a live AST it
        // caches; the AST lives in the process-global Z3 context, which is the
        // thread-local one (`rust_manager._setup_shared_z3_context`).
        return unsafe { crate::symbolic::Z3AstPtr::from_borrowed_raw(&z3_ctx, ptr) };
    }
    None
}

/// Coerce a converted constraint to the width-1 form the solver's assumed log
/// and `assume_true` both require: a wider value means "non-zero".
fn as_bool(
    bv: crate::symbolic::RustBV,
    ctx: &crate::symbolic::SymContext,
) -> crate::symbolic::RustBV {
    if bv.width() == 1 {
        bv
    } else {
        let zero = crate::symbolic::RustBV::concrete(0, bv.width());
        bv.ne(&zero, ctx)
    }
}

/// Convert one claripy constraint and assert it on `ctx`.
///
/// **The single seam for Python→Rust constraint conversion.** Both importers
/// route through here — [`import_python_constraints`] (initial/pending state
/// constraints) and
/// [`RustExplorationManager::sync_constraints_from_python`](super::RustExplorationManager::sync_constraints_from_python)
/// (constraints a bounced Python callback added) — so a correctness fix to
/// either tier lands once. They used to be two independent reimplementations
/// with *inverted* tier precedence and no cross-reference between them
/// (angr-6cp06.26); `tier` is the only thing that still differs.
///
/// `z3_backend` is what [`resolve_claripy_z3_backend`] returned; `None`
/// disables the raw-pointer tier for this call.
#[cfg_attr(
    not(feature = "vex-engine-z3"),
    allow(unused_variables, reason = "no raw-pointer tier without Z3")
)]
pub(crate) fn import_one_constraint(
    py: pyo3::Python<'_>,
    ctx: &crate::symbolic::SymContext,
    item: &pyo3::Bound<'_, pyo3::types::PyAny>,
    z3_backend: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>,
    tier: ConstraintTier,
) -> ConstraintImport {
    use crate::claripy_bridge::claripy_to_rustbv;

    // Tier: raw Z3 pointer. Only reachable when it is preferred *and* claripy
    // hands one over; otherwise both orders fall through to the RustBV tier
    // below, which is also the whole body on a non-Z3 build.
    #[cfg(feature = "vex-engine-z3")]
    if tier == ConstraintTier::Z3PtrFirst
        && let Some(backend) = z3_backend
        && let Some(z3_ast) = claripy_z3_ptr(backend, item)
    {
        return match claripy_to_rustbv(py, item, ctx) {
            // A constraint with a RustBV form is recorded in the assumed IR
            // rather than double-logged as a residual — see
            // `SymContext::add_constraint_raw_assumed` (angr-op0dn.14.2).
            Ok(bv) => {
                ctx.add_constraint_raw_assumed(z3_ast);
                ctx.assumed_constraints_push(as_bool(bv, ctx), true);
                ConstraintImport::Assumed
            }
            Err(e) => {
                ctx.add_constraint_raw(z3_ast);
                ConstraintImport::RawOnly(e)
            }
        };
    }

    // Tier: RustBV. Preserves the assumed-constraint tracking Python re-import
    // depends on, and is the only tier on a non-Z3 build.
    match claripy_to_rustbv(py, item, ctx) {
        Ok(bv) => {
            ctx.assume_true(&as_bool(bv, ctx));
            ConstraintImport::Assumed
        }
        Err(e) => {
            // Rescue an op `claripy_to_rustbv` does not model (FP, etc) by
            // asserting claripy's own AST, keeping the round-trip lossless.
            #[cfg(feature = "vex-engine-z3")]
            if let Some(backend) = z3_backend
                && let Some(z3_ast) = claripy_z3_ptr(backend, item)
            {
                ctx.add_constraint_raw(z3_ast);
                return ConstraintImport::RawOnly(e);
            }
            ConstraintImport::Failed(e)
        }
    }
}

/// Import a Python list of claripy constraints into one solver context.
///
/// Single body shared by `_add_constraints_to_state` (state_api.rs) and
/// `_add_constraints_to_pending` (pending_api.rs), which differ only in how
/// they reach the `SymContext` and in their log wording — a correctness fix to
/// the fast path used to have to land in two places (angr-9ke6b.71).
///
/// The per-constraint work is [`import_one_constraint`] with
/// [`ConstraintTier::Z3PtrFirst`]: what reaches the solver is claripy's own
/// AST, so an op `claripy_to_rustbv` models *incorrectly* cannot corrupt an
/// initial constraint.
///
/// `kind` is the noun used in the per-constraint failure log ("initial" /
/// "pending"); the caller emits its own summary line. Returns the number of
/// constraints successfully asserted — unconvertible ones are skipped, matching
/// the pre-existing behavior of both call sites.
pub(crate) fn import_python_constraints(
    py: pyo3::Python<'_>,
    ctx: &crate::symbolic::SymContext,
    constraints: &pyo3::Bound<'_, pyo3::types::PyList>,
    kind: &str,
) -> u32 {
    use pyo3::types::PyListMethods;

    let z3_backend = resolve_claripy_z3_backend(py);

    let mut added = 0u32;
    for item in constraints.iter() {
        match import_one_constraint(
            py,
            ctx,
            &item,
            z3_backend.as_ref(),
            ConstraintTier::Z3PtrFirst,
        ) {
            ConstraintImport::Assumed | ConstraintImport::RawOnly(_) => added += 1,
            ConstraintImport::Failed(e) => {
                log::debug!("Could not convert {kind} constraint: {e}");
            }
        }
    }
    added
}

/// Per-run tracking sets used by uniqueness filtering and the find/avoid
/// predicate skip-list machinery.
#[derive(Debug, Default)]
pub(crate) struct ConstraintTracker {
    /// Native uniqueness filter: register names to check.
    pub(crate) uniqueness_registers: Vec<String>,
    /// Set of seen register tuple hashes for uniqueness checking.
    pub(crate) uniqueness_set: HashSet<u64>,
    /// State IDs to skip the find predicate check for on next pop.
    /// Set after `resume_find_predicate(false)` to prevent infinite loops.
    pub(crate) skip_find_predicate_states: HashSet<u64>,
    /// State IDs to skip the avoid predicate check for on next pop.
    pub(crate) skip_avoid_predicate_states: HashSet<u64>,
}

test_submod!("constraints_tests.rs" => tests);
