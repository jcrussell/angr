//! Callback-reason and pending-callback data model for the exploration loop.
//!
//! Split out of `exploration::mod` (angr-zel8z.3). `CallbackReason` describes
//! *why* the Rust loop handed control back to Python; `PendingCallback` holds
//! the state (and deferred-fork bookkeeping) parked while that callback runs.

use crate::callbacks::DeferredFork;
use crate::solver::RustSolverContext;
use crate::state::RustSimState;
use crate::symbolic::RustBV;
use crate::vex::ir::JumpKind;
use rustc_hash::FxHashMap;

/// Reason for returning to Python.
///
/// `#[non_exhaustive]` per angr-irwe: new callback reasons land in
/// minor versions as more event-flow paths emerge; match sites must
/// include a wildcard arm.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub(crate) enum CallbackReason {
    /// SimProcedure hook hit.
    SimProcedure {
        addr: u64,
        name: String,
        num_args: usize,
        return_addr: u64,
    },
    /// Syscall instruction. `num` is `None` when the syscall register is
    /// symbolic (angr-gffd) — Python's `engines/successors.py::_resolve_syscall`
    /// handles enumeration / `NO_SYMBOLIC_SYSCALL_RESOLUTION` resolution.
    Syscall { num: Option<u64> },
    /// Find predicate needs Python evaluation.
    FindPredicate { addr: u64 },
    /// Avoid predicate needs Python evaluation.
    AvoidPredicate { addr: u64 },
    /// Error during execution.
    ///
    /// Constructed only by the in-crate test suite today (`resume_tests`,
    /// `helpers_tests`, `pending_api_tests`); production code reaches the
    /// Python error path through `ExplorationEvent::error` directly rather
    /// than by parking a `PendingCallback`. The read side is live —
    /// `callback_event` and `resume_after_error` both match it — so the
    /// variant is retained as the contract, not deleted (angr-9ke6b.214).
    #[cfg_attr(not(test), allow(dead_code))]
    Error { message: String },
    /// Symbolic branch - both paths feasible, need Python to fork states.
    ///
    /// Same situation as `Error`: only the test suite constructs it, because
    /// production branch splitting goes through the deferred-fork path
    /// (`ForkBundle` / `apply_deferred_fork_constraints`) instead of a
    /// symbolic-branch bounce. `callback_event`, `pending_condition_id` and
    /// `resume_after_symbolic_branch` still match it (angr-9ke6b.214).
    #[cfg_attr(not(test), allow(dead_code))]
    SymbolicBranch {
        condition_id: u64,
        true_target: u64,
        false_target: u64,
    },
    /// Need Python VEX engine to handle block with unsupported operations.
    PythonVEXFallback { addr: u64, reason: String },
}

/// The SimProcedure-hook descriptor: the payload of a
/// [`CallbackReason::SimProcedure`], bundled so the post-step handler and the
/// `need_simprocedure` event constructor both stay under the argument
/// threshold instead of passing the four fields positionally.
pub(crate) struct SimProcCall {
    pub(crate) addr: u64,
    pub(crate) name: String,
    pub(crate) num_args: usize,
    pub(crate) return_addr: u64,
}

/// The deferred-fork bookkeeping a bounce carries from the interpreter to the
/// Python callback. The three pieces always travel together, so they move as
/// one bundle rather than three positional params.
pub(crate) struct ForkBundle {
    /// Deferred forks accumulated before the callback.
    pub(crate) deferred_forks: Vec<DeferredFork>,
    /// Stored conditions for deferred fork handling, keyed by condition_id.
    pub(crate) stored_conditions: FxHashMap<u64, RustBV>,
    /// Pre-constraint state snapshots, keyed by condition_id.
    pub(crate) fork_snapshots: FxHashMap<u64, crate::interpreter::BranchSnapshot>,
}

impl ForkBundle {
    /// No deferred forks — the run-loop bounce path, which never defers.
    pub(crate) fn empty() -> Self {
        ForkBundle {
            deferred_forks: Vec::new(),
            stored_conditions: FxHashMap::default(),
            fork_snapshots: FxHashMap::default(),
        }
    }
}

/// State held during a Python callback.
pub(crate) struct PendingCallback {
    pub(crate) state: RustSimState,
    /// Clean snapshot of state BEFORE any callback modifications.
    /// Used for creating deferred forks - they diverged before the callback,
    /// so they should not inherit callback constraints.
    pub(crate) pre_callback_snapshot: Option<RustSimState>,
    pub(crate) reason: CallbackReason,
    /// Jumpkind that led to this callback (e.g., "Ijk_Call", "Ijk_Boring").
    pub(crate) jumpkind: Option<String>,
    /// Forked solver context for Python callbacks.
    pub(crate) solver_ctx: Option<RustSolverContext>,
    /// Deferred forks accumulated before the callback.
    /// These should be processed when the callback returns.
    pub(crate) deferred_forks: Vec<DeferredFork>,
    /// Stored conditions for deferred fork handling.
    pub(crate) stored_conditions: FxHashMap<u64, RustBV>,
    /// Full state snapshots from before branch constraints were added.
    /// Keyed by condition_id, enables correct alternate-path forking.
    pub(crate) fork_snapshots: FxHashMap<u64, crate::interpreter::BranchSnapshot>,
}

impl PendingCallback {
    /// Create a lightweight callback with no solver context or deferred state.
    /// Used for predicate evaluation (find/avoid predicates).
    pub(crate) fn lightweight(state: RustSimState, reason: CallbackReason) -> Self {
        PendingCallback {
            state,
            pre_callback_snapshot: None,
            reason,
            jumpkind: None,
            solver_ctx: None,
            deferred_forks: Vec::new(),
            stored_conditions: FxHashMap::default(),
            fork_snapshots: FxHashMap::default(),
        }
    }

    /// Create a callback with full interpreter context (deferred forks, conditions, snapshots).
    /// Used for SimProcedure, syscall, and symbolic branch callbacks after interpreter execution.
    pub(crate) fn with_context(
        state: RustSimState,
        pre_callback_snapshot: Option<RustSimState>,
        reason: CallbackReason,
        jumpkind: &str,
        solver_ctx: Option<RustSolverContext>,
        forks: ForkBundle,
    ) -> Self {
        let ForkBundle {
            deferred_forks,
            stored_conditions,
            fork_snapshots,
        } = forks;
        PendingCallback {
            state,
            pre_callback_snapshot,
            reason,
            jumpkind: Some(jumpkind.to_string()),
            solver_ctx,
            deferred_forks,
            stored_conditions,
            fork_snapshots,
        }
    }

    /// Jumpkind of the exit that led to this callback, defaulting to
    /// `Ijk_Boring` when there is none.
    ///
    /// `jumpkind` is `None` exactly for the callbacks built by `lightweight`
    /// (predicate evaluation), which carry no exit of their own. Both Python
    /// export paths in `exploration::pending_api`
    /// (`_get_pending_history_and_jumpkind` and `_export_callback_bundle`)
    /// go through this one helper so the default can only ever be changed
    /// in one place — the same reason `import_python_constraints` was
    /// centralized.
    pub(crate) fn jumpkind_or_boring(&self) -> String {
        // SILENT(cat-a): a lightweight callback legitimately has no jumpkind;
        // `Ijk_Boring` is the documented default for that case.
        self.jumpkind
            .clone()
            .unwrap_or_else(|| JumpKind::Boring.ijk_name().to_string())
    }
}

/// Apply each deferred fork's taken-path constraint onto `state`'s solver.
///
/// Without these, the solver doesn't know which branch was taken, causing
/// incorrect results for subsequent symbolic operations. Callers pass the
/// state to constrain (the resumed main state, or `pending.state` before a
/// symbolic-branch fork so both children inherit the constraints). Taken as a
/// free fn over the two field borrows rather than a `&self` method so it works
/// even after `pending.state` has been moved out (resume_after_simprocedure).
pub(crate) fn apply_deferred_fork_constraints(
    state: &RustSimState,
    deferred_forks: &[DeferredFork],
    stored_conditions: &FxHashMap<u64, RustBV>,
) {
    for fork in deferred_forks {
        if let Some(cond) = stored_conditions.get(&fork.condition_id) {
            if fork.path_taken {
                state.solver().borrow().assume_true(cond);
            } else {
                state.solver().borrow().assume_false(cond);
            }
        }
    }
}

test_submod!("callback_types_tests.rs" => tests);
