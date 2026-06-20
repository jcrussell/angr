//! Callback-reason and pending-callback data model for the exploration loop.
//!
//! Split out of `exploration::mod` (angr-zel8z.3). `CallbackReason` describes
//! *why* the Rust loop handed control back to Python; `PendingCallback` holds
//! the state (and deferred-fork bookkeeping) parked while that callback runs.

use crate::callbacks::DeferredFork;
use crate::solver::RustSolverContext;
use crate::state::RustSimState;
use crate::symbolic::RustBV;
use rustc_hash::FxHashMap;

/// Reason for returning to Python.
///
/// `#[non_exhaustive]` per angr-irwe: new callback reasons land in
/// minor versions as more event-flow paths emerge; match sites must
/// include a wildcard arm.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum CallbackReason {
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
    Error { message: String },
    /// Symbolic branch - both paths feasible, need Python to fork states.
    SymbolicBranch {
        condition_id: u64,
        true_target: u64,
        false_target: u64,
    },
    /// Need Python VEX engine to handle block with unsupported operations.
    PythonVEXFallback { addr: u64, reason: String },
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
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_context(
        state: RustSimState,
        pre_callback_snapshot: Option<RustSimState>,
        reason: CallbackReason,
        jumpkind: &str,
        solver_ctx: Option<RustSolverContext>,
        deferred_forks: Vec<DeferredFork>,
        stored_conditions: FxHashMap<u64, RustBV>,
        fork_snapshots: FxHashMap<u64, crate::interpreter::BranchSnapshot>,
    ) -> Self {
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
}
