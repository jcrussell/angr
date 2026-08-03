//! Interpreter control-flow result types.
//!
//! Split out of `interpreter/mod.rs` (angr-9ke6b.91). [`BlockResult`] is what
//! `execution.rs::run` hands back to the exploration manager; [`StmtResult`]
//! and [`ConcretizedJump`] are the interpreter-internal steps that produce it.

use crate::symbolic::RustBV;
use crate::vex::ir::JumpKind;

/// Result of concretizing a symbolic jump target.
pub(super) enum ConcretizedJump {
    /// Single concrete address (common case for deterministic jumps).
    Single(u64),
    /// Multiple concrete addresses (for symbolic ret/call/jmp).
    /// Contains the list of targets and the original symbolic expression.
    Multiple { targets: Vec<u64>, expr: RustBV },
    /// Too many targets - exceeds max_symbolic_ip_targets limit.
    /// State should be marked as unconstrained.
    TooMany { min: u64, max: u64, limit: usize },
}

/// Result of executing a single statement.
pub(super) enum StmtResult {
    /// Continue to next statement.
    Continue,
    /// Exit the block early.
    Exit { target: u64, jumpkind: JumpKind },
    /// Symbolic branch detected - need to fork.
    SymbolicBranch {
        condition: RustBV,
        true_target: u64,
        false_target: u64,
    },
}

/// Result of executing a single block.
#[derive(Debug)]
pub(crate) enum BlockResult {
    /// Syscall encountered. `num` is `None` when the syscall-number register
    /// is symbolic (angr-gffd); callers must route those cases to Python so
    /// `engines/successors.py::_resolve_syscall` can enumerate or honor
    /// `NO_SYMBOLIC_SYSCALL_RESOLUTION` instead of silently dispatching to
    /// `read` (amd64 syscall 0).
    Syscall { num: Option<u64> },
    /// Symbolic branch - need to fork.
    SymbolicBranch {
        condition_id: u64,
        true_target: u64,
        false_target: u64,
    },
    /// Hook address hit.
    Hook { addr: u64 },
    /// Normal block end with jumpkind.
    BlockEnd { next_addr: u64, jumpkind: JumpKind },
    /// Symbolic jump target with multiple concrete targets after concretization.
    /// The exploration manager should fork states for each target.
    SymbolicJumpTarget {
        /// Concrete target addresses after concretization.
        targets: Vec<u64>,
        /// ID for the stored symbolic expression (for constraint addition).
        ///
        /// The expression itself is NOT carried on this variant: producers
        /// (`interpreter::exits::handle_default_exit`) park it in the
        /// interpreter's pending-condition store under this id, and every
        /// consumer re-reads it from there (angr-9ke6b.218 item 8).
        condition_id: u64,
        /// Jump kind (Ijk_Ret, Ijk_Call, etc.).
        jumpkind: JumpKind,
    },
    /// Unconstrained jump - too many targets, exceeds limit.
    /// The state should be moved to the "unconstrained" stash.
    UnconstrainedJump {
        /// Minimum possible target address.
        min_target: u64,
        /// Maximum possible target address.
        max_target: u64,
        /// The configured limit that was exceeded.
        limit: usize,
        /// Jump kind.
        jumpkind: JumpKind,
    },
    /// Unmodeled function call - target is not hooked but is a CALL.
    /// Need Python to check if a SimProcedure can be resolved.
    UnmodeledCall {
        /// Address of the unmodeled function.
        addr: u64,
        /// Return address (from stack).
        return_addr: u64,
        /// Symbol name if available.
        symbol_name: Option<String>,
    },
}
