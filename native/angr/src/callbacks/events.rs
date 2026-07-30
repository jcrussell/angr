//! Run-loop wire types: results, memory-load payloads, and the
//! `LoopExecutionEvent` returned to Python from the run loop.
//!
//! Split out of `callbacks.rs` (god-module decomposition, angr-zel8z.2).

use pyo3::prelude::*;

use crate::symbolic::RustBV;

use super::DeferredFork;

/// How a [`RunResult::Error`] should be routed by the exploration stepping
/// loop. Carried as a typed signal so routing does not depend on matching
/// substrings of the human-readable error message (angr-zzju9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunErrorKind {
    /// Graceful deadend: the block could not be lifted (the Python lift
    /// callback returned the empty-IRSB sentinel, e.g. on
    /// `SimEngineError: No bytes in memory`). Routed to the deadended stash,
    /// matching the vanilla Python engine, which deadends states that reach
    /// invalid/unliftable code addresses.
    Deadend,
    /// Genuine execution error (malformed IRSB, memory/op/temp/callback
    /// failure). Routed to the errored stash.
    Fatal,
}

/// Result of a memory load callback.
#[derive(Debug, Clone)]
pub struct MemoryLoadResult {
    /// The concrete bytes loaded.
    pub data: Vec<u8>,
    /// Whether the value is symbolic (has an associated AST).
    pub is_symbolic: bool,
    /// The symbolic AST (if symbolic). This is a Python object reference.
    pub symbolic_ast: Option<Py<PyAny>>,
    /// The RustBV representation for the engine.
    pub value: RustBV,
}

/// Result of running the execution loop.
#[derive(Debug, Clone)]
pub enum RunResult {
    /// Reached max blocks limit - continue later.
    MaxBlocks { pc: u64 },
    /// Hit a hook address - need Python to handle.
    /// This is the legacy variant without pre-extracted arguments.
    Hook { addr: u64 },
    /// SimProcedure hook hit with pre-extracted arguments.
    /// This allows Python to directly use the arguments without re-extracting.
    SimProcedure {
        /// Address where the SimProcedure is hooked.
        addr: u64,
        /// Name of the SimProcedure (e.g., "strlen", "malloc").
        name: String,
        /// Number of arguments extracted (for Python to know how many to use).
        num_args: usize,
        /// Return address (from stack for calls, or 0 if unknown).
        return_addr: u64,
    },
    /// Syscall encountered - need Python to handle.
    ///
    /// `num` is `None` when the syscall-number register is symbolic
    /// (angr-gffd) — the dispatch loop must force a Python callback in that
    /// case rather than picking a native handler.
    Syscall { num: Option<u64>, pc: u64 },
    /// Symbolic branch - need Python to fork states.
    /// This is returned when use_deferred_forks is false or max_deferred_forks is reached.
    SymbolicBranch {
        condition_id: u64,
        true_target: u64,
        false_target: u64,
    },
    /// Normal block end.
    BlockEnd { next_addr: u64, jumpkind: String },
    /// Error during execution. `kind` tells the stepping loop whether to
    /// deadend the state (unliftable block) or move it to the errored stash
    /// (genuine error), without inspecting the message text.
    Error {
        message: String,
        addr: u64,
        kind: RunErrorKind,
    },
    /// Rust VEX interpreter hit an unsupported operation - need Python VEX engine fallback.
    NeedPythonVEX { addr: u64, reason: String },
    /// Need to lift a block at the given address.
    NeedLift { addr: u64 },
    /// Reached max deferred forks limit - return to Python with accumulated forks.
    MaxDeferredForks { pc: u64 },
    /// Symbolic jump target - multiple concrete targets after concretization.
    /// This is returned when a jump target (e.g., ret instruction) is symbolic
    /// but can be concretized to a bounded set of concrete addresses.
    SymbolicJumpTarget {
        /// Concrete target addresses after concretization.
        targets: Vec<u64>,
        /// ID for the stored symbolic expression (for constraint addition).
        condition_id: u64,
        /// Jump kind (Ijk_Ret, Ijk_Call, etc.).
        jumpkind: String,
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
        /// Jump kind (Ijk_Ret, Ijk_Call, etc.).
        jumpkind: String,
    },
    /// Unmodeled function call - need Python to resolve.
    /// This is returned when execution reaches a CALL to an address that isn't hooked.
    /// Python should check if a SimProcedure exists for this address.
    UnmodeledCall {
        /// Address of the unmodeled function.
        addr: u64,
        /// Return address (where to continue after the call).
        return_addr: u64,
        /// Symbol name if available from binary.
        symbol_name: Option<String>,
    },
}

/// Execution event returned to Python from run_loop.
#[pyclass]
#[derive(Debug, Clone, Default)]
pub struct LoopExecutionEvent {
    /// Type of event: "max_blocks", "hook", "simprocedure", "syscall", "symbolic_branch",
    /// "block_end", "error", "need_lift", "max_deferred_forks", "symbolic_jump_target", "unconstrained_jump"
    #[pyo3(get)]
    pub event_type: String,
    /// Current/next PC address.
    #[pyo3(get)]
    pub pc: Option<u64>,
    /// Hook/target address.
    #[pyo3(get)]
    pub addr: Option<u64>,
    /// Syscall number.
    #[pyo3(get)]
    pub syscall_num: Option<u64>,
    /// True target for symbolic branch.
    #[pyo3(get)]
    pub true_target: Option<u64>,
    /// False target for symbolic branch.
    #[pyo3(get)]
    pub false_target: Option<u64>,
    /// Jump kind string.
    #[pyo3(get)]
    pub jumpkind: Option<String>,
    /// Error message.
    #[pyo3(get)]
    pub error: Option<String>,
    /// Number of blocks executed this loop.
    #[pyo3(get)]
    pub blocks_executed: u32,
    /// Deferred forks collected during execution.
    /// Each fork represents a branch where we took one path and deferred the other.
    #[pyo3(get)]
    pub deferred_forks: Vec<DeferredFork>,
    /// Current solver push level after execution.
    /// Used for proper constraint handling during fork processing.
    #[pyo3(get)]
    pub push_level: u32,
    /// SimProcedure name (for "simprocedure" events).
    #[pyo3(get)]
    pub simprocedure_name: Option<String>,
    /// Number of arguments for SimProcedure.
    #[pyo3(get)]
    pub simprocedure_num_args: Option<usize>,
    /// Return address for SimProcedure (from stack).
    #[pyo3(get)]
    pub simprocedure_return_addr: Option<u64>,
    /// Symbolic jump targets (for "symbolic_jump_target" events).
    #[pyo3(get)]
    pub jump_targets: Option<Vec<u64>>,
    /// Condition ID for symbolic jump (used to add constraints).
    #[pyo3(get)]
    pub jump_condition_id: Option<u64>,
    /// Minimum target for unconstrained jump.
    #[pyo3(get)]
    pub unconstrained_min: Option<u64>,
    /// Maximum target for unconstrained jump.
    #[pyo3(get)]
    pub unconstrained_max: Option<u64>,
    /// Limit exceeded for unconstrained jump.
    #[pyo3(get)]
    pub unconstrained_limit: Option<usize>,
    /// Address of unmodeled function call (for "unmodeled_call" events).
    #[pyo3(get)]
    pub unmodeled_call_addr: Option<u64>,
    /// Return address for unmodeled function call.
    #[pyo3(get)]
    pub unmodeled_call_return_addr: Option<u64>,
    /// Symbol name for unmodeled function call (if available).
    #[pyo3(get)]
    pub unmodeled_call_symbol: Option<String>,
}

impl LoopExecutionEvent {
    /// Create an event from a run result with deferred forks and push level.
    pub fn from_run_result_with_forks(
        result: RunResult,
        blocks_executed: u32,
        deferred_forks: Vec<DeferredFork>,
        push_level: u32,
    ) -> Self {
        let base = LoopExecutionEvent {
            blocks_executed,
            deferred_forks,
            push_level,
            ..Default::default()
        };
        match result {
            RunResult::MaxBlocks { pc } => LoopExecutionEvent {
                event_type: "max_blocks".to_string(),
                pc: Some(pc),
                ..base
            },
            RunResult::Hook { addr } => LoopExecutionEvent {
                event_type: "hook".to_string(),
                pc: Some(addr),
                addr: Some(addr),
                ..base
            },
            RunResult::SimProcedure {
                addr,
                name,
                num_args,
                return_addr,
            } => LoopExecutionEvent {
                event_type: "simprocedure".to_string(),
                pc: Some(addr),
                addr: Some(addr),
                jumpkind: Some("Ijk_Call".to_string()),
                simprocedure_name: Some(name),
                simprocedure_num_args: Some(num_args),
                simprocedure_return_addr: if return_addr != 0 {
                    Some(return_addr)
                } else {
                    None
                },
                ..base
            },
            RunResult::Syscall { num, pc } => LoopExecutionEvent {
                event_type: "syscall".to_string(),
                pc: Some(pc),
                syscall_num: num,
                jumpkind: Some("Ijk_Sys_syscall".to_string()),
                ..base
            },
            RunResult::SymbolicBranch {
                true_target,
                false_target,
                ..
            } => LoopExecutionEvent {
                event_type: "symbolic_branch".to_string(),
                true_target: Some(true_target),
                false_target: Some(false_target),
                ..base
            },
            RunResult::BlockEnd {
                next_addr,
                jumpkind,
            } => LoopExecutionEvent {
                event_type: "block_end".to_string(),
                pc: Some(next_addr),
                addr: Some(next_addr),
                jumpkind: Some(jumpkind),
                ..base
            },
            RunResult::Error {
                message,
                addr,
                kind: _,
            } => LoopExecutionEvent {
                event_type: "error".to_string(),
                pc: Some(addr),
                addr: Some(addr),
                error: Some(message),
                ..base
            },
            RunResult::NeedLift { addr } => LoopExecutionEvent {
                event_type: "need_lift".to_string(),
                pc: Some(addr),
                addr: Some(addr),
                ..base
            },
            RunResult::MaxDeferredForks { pc } => LoopExecutionEvent {
                event_type: "max_deferred_forks".to_string(),
                pc: Some(pc),
                ..base
            },
            RunResult::SymbolicJumpTarget {
                targets,
                condition_id,
                jumpkind,
            } => LoopExecutionEvent {
                event_type: "symbolic_jump_target".to_string(),
                pc: targets.first().copied(),
                jumpkind: Some(jumpkind),
                jump_targets: Some(targets),
                jump_condition_id: Some(condition_id),
                ..base
            },
            RunResult::UnconstrainedJump {
                min_target,
                max_target,
                limit,
                jumpkind,
            } => LoopExecutionEvent {
                event_type: "unconstrained_jump".to_string(),
                jumpkind: Some(jumpkind),
                unconstrained_min: Some(min_target),
                unconstrained_max: Some(max_target),
                unconstrained_limit: Some(limit),
                ..base
            },
            RunResult::UnmodeledCall {
                addr,
                return_addr,
                symbol_name,
            } => LoopExecutionEvent {
                event_type: "unmodeled_call".to_string(),
                pc: Some(addr),
                addr: Some(addr),
                jumpkind: Some("Ijk_Call".to_string()),
                unmodeled_call_addr: Some(addr),
                unmodeled_call_return_addr: Some(return_addr),
                unmodeled_call_symbol: symbol_name,
                ..base
            },
            RunResult::NeedPythonVEX { addr, reason } => LoopExecutionEvent {
                event_type: "python_vex_fallback".to_string(),
                pc: Some(addr),
                addr: Some(addr),
                error: Some(reason),
                ..base
            },
        }
    }

    /// Create an event from a run result (backward compatibility, no deferred forks).
    pub fn from_run_result(result: RunResult, blocks_executed: u32) -> Self {
        Self::from_run_result_with_forks(result, blocks_executed, Vec::new(), 0)
    }
}

#[cfg(test)]
#[path = "events_tests.rs"]
mod events_tests;
