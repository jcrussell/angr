//! Rust-native exploration manager for symbolic execution.
//!
//! `RustExplorationManager` provides a Rust-first exploration loop that:
//! - Manages states entirely in Rust (using `RustSimState`)
//! - Processes symbolic branches with deferred forks
//! - Only returns to Python for SimProcedures and syscalls
//! - Implements find/avoid address checking in Rust
//!
//! This achieves ~3x speedup by keeping state management in Rust and
//! minimizing Python callback overhead.

use crate::stash::{
    STASH_ACTIVE, STASH_AVOID, STASH_DEADENDED, STASH_ERRORED, STASH_FOUND, STASH_PRUNED,
    STASH_UNCONSTRAINED, StashManager,
};
use rustc_hash::FxHashMap;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use pyo3::class::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::arch::{arch_from_name, default_cc_for_arch};
use crate::callbacks::{DeferredFork, ExecutionConfig, PythonCallbacks, RunResult};
use crate::claripy_bridge::{claripy_to_rustbv, rustbv_to_claripy};
use crate::interpreter_cb::{CallbackInterpreter, DCAS_UNSUPPORTED_REASON, ExecutionStats};
use crate::memory::Permission;
use crate::procedures::NativeProcedureRegistry;
use crate::solver::RustSolverContext;
use crate::state::{RustSimState, StateChanges};
use crate::symbolic::{RustBV, SymContext};
use crate::syscalls::{NativeSyscallRegistry, SyscallOutcome};

use std::cell::Cell;

mod constraints;
mod execution_env;
mod helpers;
mod memory_config;
mod pending_api;
mod profiling;
mod resume;
mod run_loop;
mod state_api;
mod state_lifecycle;
mod stats_api;
mod stepping;

use self::constraints::{ConstraintSolver, ConstraintTracker};
use self::execution_env::ExecutionEnvironment;
use self::memory_config::MemoryConfiguration;
use self::profiling::ProfilingCollector;
use self::stepping::StepError;

// Thread-local stepping state ID, accessible from callbacks without borrow conflicts.
thread_local! {
    static STEPPING_STATE_ID: Cell<Option<u64>> = Cell::new(None);
}

/// Get the current stepping state ID (safe to call from callbacks).
#[pyfunction]
pub fn get_stepping_state_id() -> Option<u64> {
    STEPPING_STATE_ID.with(|cell| cell.get())
}

/// Reason for returning to Python.
#[derive(Debug, Clone)]
pub enum CallbackReason {
    /// SimProcedure hook hit.
    SimProcedure {
        addr: u64,
        name: String,
        num_args: usize,
        return_addr: u64,
    },
    /// Syscall instruction.
    Syscall { num: u64 },
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

/// Event returned from exploration to Python.
#[pyclass]
#[derive(Debug, Clone)]
pub struct ExplorationEvent {
    /// Type: "found", "deadended", "need_callback", "step_complete", "errored", "active_empty"
    #[pyo3(get)]
    pub event_type: String,
    /// Number of states in found stash.
    #[pyo3(get)]
    pub found_count: usize,
    /// Number of states in active stash.
    #[pyo3(get)]
    pub active_count: usize,
    /// Number of steps taken.
    #[pyo3(get)]
    pub steps_taken: u64,
    /// State ID for callback (if need_callback).
    #[pyo3(get)]
    pub callback_state_id: Option<u64>,
    /// Callback reason string.
    #[pyo3(get)]
    pub callback_reason: Option<String>,
    /// Callback address.
    #[pyo3(get)]
    pub callback_addr: Option<u64>,
    /// Callback name (e.g., SimProcedure name).
    #[pyo3(get)]
    pub callback_name: Option<String>,
    /// Syscall number (if syscall callback).
    #[pyo3(get)]
    pub callback_syscall_num: Option<u64>,
    /// Return address for SimProcedure.
    #[pyo3(get)]
    pub callback_return_addr: Option<u64>,
    /// Number of arguments for SimProcedure.
    #[pyo3(get)]
    pub callback_num_args: Option<usize>,
    /// Symbolic branch true target (if symbolic_branch callback).
    #[pyo3(get)]
    pub branch_true_target: Option<u64>,
    /// Symbolic branch false target (if symbolic_branch callback).
    #[pyo3(get)]
    pub branch_false_target: Option<u64>,
    /// Symbolic branch condition ID (if symbolic_branch callback).
    #[pyo3(get)]
    pub branch_condition_id: Option<u64>,
}

impl ExplorationEvent {
    /// Base constructor with common fields; all Optional fields default to None.
    pub(crate) fn base(
        event_type: &str,
        found_count: usize,
        active_count: usize,
        steps: u64,
    ) -> Self {
        ExplorationEvent {
            event_type: event_type.to_string(),
            found_count,
            active_count,
            steps_taken: steps,
            callback_state_id: None,
            callback_reason: None,
            callback_addr: None,
            callback_name: None,
            callback_syscall_num: None,
            callback_return_addr: None,
            callback_num_args: None,
            branch_true_target: None,
            branch_false_target: None,
            branch_condition_id: None,
        }
    }

    pub(crate) fn found(found_count: usize, active_count: usize, steps: u64) -> Self {
        Self::base(STASH_FOUND, found_count, active_count, steps)
    }

    #[allow(dead_code)]
    pub(crate) fn deadended(found_count: usize, active_count: usize, steps: u64) -> Self {
        Self::base(STASH_DEADENDED, found_count, active_count, steps)
    }

    pub(crate) fn active_empty(found_count: usize, steps: u64) -> Self {
        Self::base("active_empty", found_count, 0, steps)
    }

    pub(crate) fn step_complete(found_count: usize, active_count: usize, steps: u64) -> Self {
        Self::base("step_complete", found_count, active_count, steps)
    }

    pub(crate) fn need_simprocedure(
        state_id: u64,
        addr: u64,
        name: String,
        num_args: usize,
        return_addr: u64,
        found_count: usize,
        active_count: usize,
        steps: u64,
    ) -> Self {
        ExplorationEvent {
            callback_state_id: Some(state_id),
            callback_reason: Some("simprocedure".to_string()),
            callback_addr: Some(addr),
            callback_name: Some(name),
            callback_return_addr: Some(return_addr),
            callback_num_args: Some(num_args),
            ..Self::base("need_callback", found_count, active_count, steps)
        }
    }

    pub(crate) fn need_syscall(
        state_id: u64,
        syscall_num: u64,
        found_count: usize,
        active_count: usize,
        steps: u64,
    ) -> Self {
        ExplorationEvent {
            callback_state_id: Some(state_id),
            callback_reason: Some("syscall".to_string()),
            callback_syscall_num: Some(syscall_num),
            ..Self::base("need_callback", found_count, active_count, steps)
        }
    }

    pub(crate) fn need_symbolic_branch(
        state_id: u64,
        condition_id: u64,
        true_target: u64,
        false_target: u64,
        found_count: usize,
        active_count: usize,
        steps: u64,
    ) -> Self {
        ExplorationEvent {
            callback_state_id: Some(state_id),
            callback_reason: Some("symbolic_branch".to_string()),
            branch_true_target: Some(true_target),
            branch_false_target: Some(false_target),
            branch_condition_id: Some(condition_id),
            ..Self::base("need_callback", found_count, active_count, steps)
        }
    }

    pub(crate) fn error(
        message: String,
        found_count: usize,
        active_count: usize,
        steps: u64,
    ) -> Self {
        ExplorationEvent {
            callback_reason: Some(message),
            ..Self::base(STASH_ERRORED, found_count, active_count, steps)
        }
    }

    pub(crate) fn need_python_vex(
        state_id: u64,
        addr: u64,
        reason: &str,
        found_count: usize,
        active_count: usize,
        steps: u64,
    ) -> Self {
        ExplorationEvent {
            callback_state_id: Some(state_id),
            callback_reason: Some("python_vex_fallback".to_string()),
            callback_addr: Some(addr),
            callback_name: Some(reason.to_string()),
            ..Self::base("need_callback", found_count, active_count, steps)
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
    pub(crate) fork_snapshots: FxHashMap<u64, crate::interpreter_cb::BranchSnapshot>,
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
        deferred_forks: Vec<DeferredFork>,
        stored_conditions: FxHashMap<u64, RustBV>,
        fork_snapshots: FxHashMap<u64, crate::interpreter_cb::BranchSnapshot>,
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

/// Native exploration technique variants.
///
/// These techniques run entirely in Rust during the exploration loop,
/// avoiding Python callback overhead for common technique patterns.
#[derive(Debug, Clone)]
pub(crate) enum NativeTechnique {
    /// Limits path length by block count. States exceeding `max_length` blocks
    /// are moved to "cut" (or "_DROP" if `drop` is true).
    LengthLimiter { max_length: usize, drop: bool },
    /// Wall-clock timeout. Exploration stops after `timeout_secs` seconds.
    Timeout {
        timeout_secs: f64,
        start_time: Option<std::time::Instant>,
    },
    /// Basic loop bounding: limits how many times a single address can appear
    /// in a state's history. States exceeding the bound are moved to `discard_stash`.
    LoopBound { bound: usize, discard_stash: String },
}

/// Rust-native exploration manager.
///
/// Manages states entirely in Rust with O(1) forking.
/// Returns to Python only for SimProcedures, syscalls, and predicates.
#[pyclass(unsendable)]
pub struct RustExplorationManager {
    /// Per-binary execution environment: arch metadata, calling convention,
    /// binary regions, block cache, endianness, history cap.
    pub(crate) environment: ExecutionEnvironment,
    /// Stash system (mirrors Python's).
    pub(crate) sm: StashManager,
    /// Find addresses.
    pub(crate) find_addrs: HashSet<u64>,
    /// Avoid addresses.
    pub(crate) avoid_addrs: HashSet<u64>,
    /// Whether find condition has callable predicates.
    pub(crate) find_needs_python: bool,
    /// Whether avoid condition has callable predicates.
    pub(crate) avoid_needs_python: bool,
    /// Execution config for deferred forks.
    pub(crate) exec_config: ExecutionConfig,
    /// Python callbacks for memory/lifting.
    pub(crate) callbacks: Option<PythonCallbacks>,
    /// Hook addresses.
    pub(crate) hooks: HashSet<u64>,
    /// SimProcedures: address -> (name, num_args, no_return).
    pub(crate) simprocedures: HashMap<u64, (String, usize, bool)>,
    /// Pending state waiting for Python callback result.
    pub(crate) pending_callback: Option<PendingCallback>,
    /// ID of the state currently being stepped (for Python callbacks to identify)
    pub(crate) current_stepping_state_id: Option<u64>,
    /// Total steps executed.
    pub(crate) steps: u64,
    /// Error log: (addr, message, state_id).
    pub(crate) errors: Vec<(u64, String, u64)>,
    /// Number of finds required before stopping.
    pub(crate) num_find: usize,
    /// Maximum steps per run iteration.
    pub(crate) max_steps_per_run: u32,
    /// Native procedure registry.
    pub(crate) native_procedures: NativeProcedureRegistry,
    /// Native syscall registry (skip Python `_handle_syscall_callback` round-trip).
    pub(crate) native_syscalls: NativeSyscallRegistry,
    /// VEX fallback tracking: count and unique addresses.
    pub(crate) vex_fallback_count: u64,
    pub(crate) vex_fallback_addrs: HashMap<u64, String>,
    /// Visibility counter for `IRStmt::CAS` double-CAS (cmpxchg16b) fallbacks.
    /// Incremented alongside `vex_fallback_count` whenever the reason carries
    /// `DCAS_UNSUPPORTED_REASON`. Surfaced via `stats()` and
    /// `get_fallback_stats()` so DCAS-driven deadends are diagnosable.
    pub(crate) dcas_unsupported_count: u64,
    /// Total count of SimProcedure invocations that were dispatched to the
    /// Python `_handle_simprocedure_callback` (rather than handled natively).
    /// This includes: native handler missing, native handler returned `Err`,
    /// and addresses inside the binary (user-placed Python hooks). Surfaced
    /// via `stats()` so a regression that flips a hot procedure off the
    /// native path is visible without rebuilding.
    pub(crate) simprocedure_python_fallback_count: u64,
    /// Per-procedure breakdown of `simprocedure_python_fallback_count`. Keyed
    /// by the SimProcedure name (as registered via `add_simprocedure`).
    /// Surfaced via `stats()` and `get_fallback_stats()` under
    /// `simprocedure_fallback_by_name` so we can see which native procedures
    /// to implement next without running a separate profiling pass.
    pub(crate) simprocedure_fallback_by_name: HashMap<String, u64>,
    /// Total count of syscalls dispatched to the Python
    /// `_handle_syscall_callback` (rather than handled by `NativeSyscall`).
    /// Includes: no native handler registered for (arch, num) and native
    /// handler returned `Err`. Surfaced via `stats()`.
    pub(crate) syscall_python_fallback_count: u64,
    /// State IDs that have already produced a DCAS warning. We log the first
    /// DCAS hit per state to avoid spamming the log on tight DCAS loops.
    pub(crate) dcas_warned_states: HashSet<u64>,
    /// Stack of (address, expiry_step) for zero-length hook skip tracking.
    /// Each entry represents an address to skip, valid until the specified step.
    /// This prevents infinite loops when a hook with length=0 runs and
    /// returns to the same address. Stack-based to handle nested hooks.
    pub(crate) skip_hook_stack: Vec<(u64, u64)>,
    // state_roots is now in self.sm (StashManager)
    /// P9 fix: Use LIFO (stack) state selection instead of FIFO (queue).
    /// When true, states are popped from the back (DFS). Default is false (BFS).
    pub(crate) use_lifo: bool,
    /// Solver configuration: lazy_solves flag and Z3 timeout.
    pub(crate) constraint_solver: ConstraintSolver,
    /// Memory and VEX configuration: zero-fill, concretizer, vex opt levels.
    pub(crate) memory_config: MemoryConfiguration,
    /// Maximum number of states in the active stash. None = unlimited.
    pub(crate) max_active_states: Option<usize>,
    // drop_terminal_states, avoided_count, pruned_count, deadended_count
    // are now in self.sm (StashManager)
    /// Native exploration techniques that run entirely in Rust.
    pub(crate) native_techniques: Vec<NativeTechnique>,
    /// Constraint-related per-run tracking: uniqueness filter sets and
    /// the find/avoid predicate skip lists.
    pub(crate) constraint_tracker: ConstraintTracker,
    /// Profiling state: enable flag, per-step `ExecutionStats`, and
    /// `NativeProcStats` accumulator.
    pub(crate) profiling: ProfilingCollector,
    // state_index is now in self.sm (StashManager)
}

#[pymethods]
impl RustExplorationManager {
    /// Create a new exploration manager.
    #[new]
    #[pyo3(signature = (arch="amd64", little_endian=None))]
    pub fn new(arch: &str, little_endian: Option<bool>) -> PyResult<Self> {
        let arch_info = arch_from_name(arch)
            .ok_or_else(|| PyValueError::new_err(format!("unsupported architecture: {}", arch)))?;

        let vex_arch = arch_info.vex_arch();

        Ok(RustExplorationManager {
            environment: ExecutionEnvironment::new(
                arch.to_string(),
                vex_arch,
                default_cc_for_arch(arch),
                little_endian,
            ),
            sm: StashManager::new(),
            find_addrs: HashSet::new(),
            avoid_addrs: HashSet::new(),
            find_needs_python: false,
            avoid_needs_python: false,
            exec_config: ExecutionConfig::default(),
            callbacks: None,
            hooks: HashSet::new(),
            simprocedures: HashMap::new(),
            pending_callback: None,
            current_stepping_state_id: None,
            steps: 0,
            errors: Vec::new(),
            num_find: 1,
            max_steps_per_run: 5000,
            native_procedures: NativeProcedureRegistry::new(),
            native_syscalls: NativeSyscallRegistry::new(),
            vex_fallback_count: 0,
            vex_fallback_addrs: HashMap::new(),
            dcas_unsupported_count: 0,
            simprocedure_python_fallback_count: 0,
            simprocedure_fallback_by_name: HashMap::new(),
            syscall_python_fallback_count: 0,
            dcas_warned_states: HashSet::new(),
            skip_hook_stack: Vec::new(),
            use_lifo: false, // P9: Default to BFS (FIFO)
            constraint_solver: ConstraintSolver::new(),
            memory_config: MemoryConfiguration::default(),
            max_active_states: None,
            native_techniques: Vec::new(),
            constraint_tracker: ConstraintTracker::default(),
            profiling: ProfilingCollector::default(),
        })
    }

    // =========================================================================
    // PyAPI methods (from pyapi.rs)
    // =========================================================================

    /// Get the architecture name.
    #[getter]
    pub fn arch(&self) -> &str {
        &self.environment.arch_name
    }

    /// Get the total number of steps executed.
    #[getter]
    pub fn step_count(&self) -> u64 {
        self.steps
    }

    /// Get active state count.
    pub fn active_count(&self) -> usize {
        self.sm.get(STASH_ACTIVE).map(|s| s.len()).unwrap_or(0)
    }

    /// Get found state count.
    pub fn found_count(&self) -> usize {
        self.sm
            .stashes()
            .get(STASH_FOUND)
            .map(|s| s.len())
            .unwrap_or(0)
    }

    /// Get stash counts as a dictionary.
    pub fn stash_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (name, stash) in self.sm.stashes() {
            dict.set_item(name, stash.len())?;
        }
        Ok(dict)
    }

    /// Set find addresses.
    pub fn set_find_addrs(&mut self, addrs: Vec<u64>) {
        self.find_addrs = addrs.into_iter().collect();
        self.find_needs_python = false;
    }

    /// Set avoid addresses.
    pub fn set_avoid_addrs(&mut self, addrs: Vec<u64>) {
        self.avoid_addrs = addrs.into_iter().collect();
        self.avoid_needs_python = false;
    }

    /// Mark that find condition has callable predicates (needs Python).
    pub fn set_find_needs_python(&mut self, needs: bool) {
        self.find_needs_python = needs;
    }

    /// Mark that avoid condition has callable predicates (needs Python).
    pub fn set_avoid_needs_python(&mut self, needs: bool) {
        self.avoid_needs_python = needs;
    }

    /// P9 fix: Set state selection to LIFO (DFS - depth-first search).
    pub fn set_state_selection_lifo(&mut self) {
        self.use_lifo = true;
        log::debug!("State selection set to LIFO (DFS)");
    }

    /// P9 fix: Set state selection to FIFO (BFS - breadth-first search).
    pub fn set_state_selection_fifo(&mut self) {
        self.use_lifo = false;
        log::debug!("State selection set to FIFO (BFS)");
    }

    /// Set the number of solutions to find before stopping.
    pub fn set_num_find(&mut self, n: usize) {
        self.num_find = n;
    }

    /// Set maximum steps per run iteration.
    pub fn set_max_steps_per_run(&mut self, n: u32) {
        self.max_steps_per_run = n;
    }

    /// Enable lazy solves mode (skip satisfiability checks on forks).
    pub fn set_lazy_solves(&mut self, enabled: bool) {
        self.constraint_solver.lazy_solves = enabled;
    }

    /// Enable zero-fill for unconstrained memory reads.
    /// When true, unmapped memory returns zero instead of fresh symbolic values.
    pub fn set_zero_fill_unconstrained(&mut self, enabled: bool) {
        self.memory_config.zero_fill_unconstrained = enabled;
    }

    /// Set the Z3 solver timeout in milliseconds (default: 30000).
    pub fn set_solver_timeout(&mut self, timeout_ms: u32) {
        self.constraint_solver.solver_timeout_ms = timeout_ms;
    }

    /// Set the maximum number of states in the active stash.
    /// When the limit is reached, new forked states are pruned to avoid OOM.
    /// None (default) means no limit.
    #[pyo3(signature = (limit=None))]
    pub fn set_max_active_states(&mut self, limit: Option<usize>) {
        self.max_active_states = limit;
    }

    /// Get the current max_active_states limit.
    pub fn get_max_active_states(&self) -> Option<usize> {
        self.max_active_states
    }

    /// Set the global VEX optimization level (0-3).
    /// None = use pyvex default (typically 1).
    /// Level 0: no optimization. Level 1: standard. Level 2-3: aggressive.
    #[pyo3(signature = (level=None))]
    pub fn set_vex_opt_level(&mut self, level: Option<i32>) {
        self.memory_config.vex_opt_level = level;
        // Invalidate block cache since opt_level affects IR output
        self.environment.block_cache.clear();
    }

    /// Get the current VEX optimization level.
    pub fn get_vex_opt_level(&self) -> Option<i32> {
        self.memory_config.vex_opt_level
    }

    /// Set a per-address VEX optimization level override.
    /// Blocks at this address will be lifted with the specified opt_level.
    pub fn set_vex_opt_level_override(&mut self, addr: u64, level: i32) {
        self.memory_config
            .vex_opt_level_overrides
            .insert(addr, level);
        // Remove this address from block cache since opt_level changed
        self.environment.block_cache.pop(&addr);
    }

    /// Remove a per-address VEX optimization level override.
    pub fn remove_vex_opt_level_override(&mut self, addr: u64) {
        self.memory_config.vex_opt_level_overrides.remove(&addr);
        self.environment.block_cache.pop(&addr);
    }

    /// Clear all per-address VEX optimization level overrides.
    pub fn clear_vex_opt_level_overrides(&mut self) {
        let addrs: Vec<u64> = self
            .memory_config
            .vex_opt_level_overrides
            .keys()
            .copied()
            .collect();
        self.memory_config.vex_opt_level_overrides.clear();
        for addr in addrs {
            self.environment.block_cache.pop(&addr);
        }
    }

    /// Resolve the VEX optimization level for a given address.
    /// Per-address overrides take precedence over the global level.
    pub fn resolve_vex_opt_level(&self, addr: u64) -> Option<i32> {
        self.memory_config
            .vex_opt_level_overrides
            .get(&addr)
            .copied()
            .or(self.memory_config.vex_opt_level)
    }

    /// Set whether to drop terminal states (avoid/pruned/deadended) immediately.
    /// When true (default), terminal states are dropped to save memory.
    /// Set to false when states need to be recovered (e.g., factory.callable()).
    pub fn set_drop_terminal_states(&mut self, enabled: bool) {
        self.sm.set_drop_terminal_states(enabled);
    }

    /// Configure address concretization strategies to match Python's configuration.
    ///
    /// # Arguments
    /// * `use_approximate` - Whether APPROXIMATE_MEMORY_INDICES is enabled
    /// * `read_range_limit` - Range limit for read strategies (default: 1024)
    /// * `write_range_limit` - Range limit for write strategies (default: 128)
    /// * `symbolic_write_addresses` - Whether SYMBOLIC_WRITE_ADDRESSES is enabled
    #[pyo3(signature = (use_approximate, read_range_limit=None, write_range_limit=None, symbolic_write_addresses=false))]
    pub fn configure_concretization_strategies(
        &mut self,
        use_approximate: bool,
        read_range_limit: Option<u64>,
        write_range_limit: Option<u64>,
        symbolic_write_addresses: bool,
    ) {
        self.memory_config.concretizer_config.configure_strategies(
            use_approximate,
            read_range_limit,
            write_range_limit,
            symbolic_write_addresses,
        );
    }

    /// Enable or disable Rust-side profiling.
    /// When enabled, per-step timing and counters are accumulated.
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling.profiling_enabled = enabled;
    }

    /// Set the maximum length of each state's `history` / `detailed_history`
    /// ring buffers. 0 means unlimited (legacy behavior — can OOM on long
    /// explorations). Default is 1000. The new value is applied to every
    /// state already in any stash, plus any future state created via this
    /// manager.
    pub fn set_max_history(&mut self, max: usize) {
        self.environment.max_history = max;
        for stash in self.sm.stashes_mut().values_mut() {
            for state in stash.iter_mut() {
                state.set_max_history(max);
            }
        }
    }

    /// Get the current per-state max_history value. 0 = unlimited.
    pub fn get_max_history(&self) -> usize {
        self.environment.max_history
    }

    /// Get accumulated execution statistics as a dict.
    pub fn get_execution_stats(&self) -> HashMap<String, u64> {
        self.profiling.accumulated_stats.to_hashmap()
    }

    /// Reset accumulated execution statistics.
    pub fn reset_execution_stats(&mut self) {
        self.profiling.accumulated_stats.reset();
    }

    /// Set Python callbacks for memory/lifting.
    pub fn set_callbacks(&mut self, callbacks: PythonCallbacks) {
        self.callbacks = Some(callbacks);
    }

    /// Clear callbacks.
    pub fn clear_callbacks(&mut self) {
        self.callbacks = None;
    }

    /// GC traversal: visit Python callback refs held inside the cloned
    /// PythonCallbacks struct. The Python wrapper `mgr` owns
    /// `mgr._rust_mgr` (this object), and via `set_callbacks` this object
    /// holds a cloned PythonCallbacks whose Py<PyAny> bound methods point
    /// back at `mgr` — a non-trivial cycle that cycle-GC can break only if
    /// __traverse__/__clear__ are exposed.
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(cbs) = &self.callbacks {
            cbs.traverse_fields(&visit)?;
        }
        Ok(())
    }

    /// GC clear: drop the bound-method refs held inside the cloned
    /// PythonCallbacks struct. After this returns, the engine can no
    /// longer call back into Python — but cycle-GC only invokes __clear__
    /// when the object is being collected, so further callbacks would not
    /// be issued.
    fn __clear__(&mut self) {
        if let Some(cbs) = &mut self.callbacks {
            cbs.clear_fields();
        }
    }

    /// Add a hook address.
    pub fn add_hook(&mut self, addr: u64) {
        self.hooks.insert(addr);
    }

    /// Add multiple hook addresses.
    pub fn add_hooks(&mut self, addrs: Vec<u64>) {
        for addr in addrs {
            self.hooks.insert(addr);
        }
    }

    /// Clear all hooks.
    pub fn clear_hooks(&mut self) {
        self.hooks.clear();
    }

    /// Register a SimProcedure.
    #[pyo3(signature = (addr, name, num_args=0, no_return=false))]
    pub fn register_simprocedure(
        &mut self,
        addr: u64,
        name: String,
        num_args: usize,
        no_return: bool,
    ) {
        self.hooks.insert(addr);
        self.simprocedures.insert(addr, (name, num_args, no_return));
    }

    /// Register multiple SimProcedures.
    pub fn register_simprocedures(&mut self, procs: Vec<(u64, String, usize, bool)>) {
        for (addr, name, num_args, no_return) in procs {
            self.hooks.insert(addr);
            self.simprocedures.insert(addr, (name, num_args, no_return));
        }
    }

    /// Load binary code regions.
    pub fn load_binary_regions(&mut self, regions: Vec<(u64, Vec<u8>)>) {
        self.environment.binary_regions = regions
            .into_iter()
            .map(|(base, data)| (base, Arc::new(data)))
            .collect();
    }

    /// Create a new RustSimState and add it to a stash.
    /// See [`state_lifecycle::_create_state`] for the body.
    #[pyo3(signature = (stash="active"))]
    pub fn create_state(&mut self, stash: &str) -> PyResult<u64> {
        self._create_state(stash)
    }

    /// Add an existing RustSimState to a stash.
    /// See [`state_lifecycle::_add_state`] for the body.
    #[pyo3(signature = (stash, state))]
    pub fn add_state(&mut self, stash: &str, state: &crate::state::PyRustSimState) {
        self._add_state(stash, state)
    }

    /// Merge multiple states into one using symbolic merge conditions.
    ///
    /// Each state's constraints are guarded by a fresh 1-bit merge flag.
    /// Registers and memory that differ between states become ITE expressions.
    /// The merged state is placed into `dest_stash`.
    ///
    /// Returns the merged state's ID.
    /// See [`state_lifecycle::_merge_states`] for the body.
    #[pyo3(signature = (state_ids, dest_stash="active"))]
    pub fn merge_states(&mut self, state_ids: Vec<u64>, dest_stash: &str) -> PyResult<u64> {
        self._merge_states(state_ids, dest_stash)
    }

    /// Get the PC of a state in a stash by index.
    #[pyo3(signature = (stash="active", index=0))]
    pub fn get_state_pc(&self, stash: &str, index: usize) -> Option<u64> {
        self.sm
            .get(stash)
            .and_then(|s| s.get(index))
            .map(|s| s.pc())
    }

    /// Get the PC of a state by its ID (O(1) via state index, no full export).
    pub fn get_state_pc_by_id(&self, state_id: u64) -> Option<u64> {
        // find_state already checks pending_callback first.
        self.find_state(state_id).map(|s| s.pc())
    }

    /// Get state IDs in a stash.
    #[pyo3(signature = (stash="active"))]
    pub fn get_state_ids(&self, stash: &str) -> Vec<u64> {
        self.sm
            .get(stash)
            .map(|s| s.iter().map(|state| state.state_id()).collect())
            .unwrap_or_default()
    }

    /// Get (state_id, addr, stdout_len) tuples for states in a stash.
    /// Used by Python predicate caching to skip re-evaluation when
    /// a state's address and stdout haven't changed.
    #[pyo3(signature = (stash="active"))]
    pub fn get_state_predicate_info(&self, stash: &str) -> Vec<(u64, u64, usize)> {
        self.sm
            .get(stash)
            .map(|s| {
                s.iter()
                    .map(|state| (state.state_id(), state.pc(), state.stdout_buffer().len()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check if there are any active states (O(1), no allocation).
    pub fn has_active_states(&self) -> bool {
        self.sm.get(STASH_ACTIVE).map_or(false, |s| !s.is_empty())
    }

    /// Get the number of states in a stash (O(1), no allocation).
    #[pyo3(signature = (stash="active"))]
    pub fn stash_count(&self, stash: &str) -> usize {
        self.sm.get(stash).map_or(0, |s| s.len())
    }

    /// Get the stash name a state currently belongs to (O(1) via state index).
    /// Returns None if the state isn't found in any stash. Used by
    /// RustStateProxy.__repr__ for cheap REPL debugging output.
    pub fn state_stash(&self, state_id: u64) -> Option<String> {
        self.sm.stash_of(state_id).map(|s| s.to_string())
    }

    /// Get the number of solver constraints for a state (O(1), reads
    /// the SymContext's atomic counter — no Z3 traversal). Returns None
    /// if the state isn't found. Used by RustStateProxy.__repr__.
    pub fn state_constraint_count(&self, state_id: u64) -> Option<usize> {
        let state = self.find_state(state_id)?;
        Some(state.solver().borrow().num_constraints())
    }

    /// Rebuild the state index after run() modifies stashes internally.
    /// Call from Python after run() returns to keep index up to date.
    pub fn sync_state_index(&mut self) {
        self.rebuild_state_index();
    }

    /// Get the root state ID for any state.
    /// Returns the original (initial) state from which this state was forked.
    pub fn get_state_root(&self, state_id: u64) -> Option<u64> {
        self.sm.roots().get(&state_id).copied()
    }

    /// Set the PC of the pending callback state (for external initialization).
    /// See [`pending_api::_set_pending_state_pc`] for the body.
    pub fn set_pending_state_pc(&mut self, pc: u64) -> PyResult<()> {
        self._set_pending_state_pc(pc)
    }

    /// Map memory in the pending state.
    /// See [`pending_api::_pending_state_map_memory`] for the body.
    #[pyo3(signature = (addr, data, permissions=7))]
    pub fn pending_state_map_memory(
        &mut self,
        addr: u64,
        data: &[u8],
        permissions: u8,
    ) -> PyResult<()> {
        self._pending_state_map_memory(addr, data, permissions)
    }

    /// Map memory in active states.
    /// See [`pending_api::_active_states_map_memory`] for the body.
    #[pyo3(signature = (addr, data, permissions=7))]
    pub fn active_states_map_memory(&mut self, addr: u64, data: &[u8], permissions: u8) {
        self._active_states_map_memory(addr, data, permissions)
    }

    /// Get the branch condition from the pending symbolic branch callback.
    ///
    /// Returns the condition as a claripy AST that Python can use for forking.
    /// See [`pending_api::_get_pending_branch_condition`] for the body.
    pub fn get_pending_branch_condition(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self._get_pending_branch_condition(py)
    }

    /// Get register value from pending state (concrete only).
    /// See [`pending_api::_get_pending_register`] for the body.
    pub fn get_pending_register(&self, name: &str) -> PyResult<Option<u128>> {
        self._get_pending_register(name)
    }

    /// Get register as claripy AST from pending state (handles symbolic).
    /// See [`pending_api::_get_pending_register_ast`] for the body.
    pub fn get_pending_register_ast(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        self._get_pending_register_ast(py, name)
    }

    /// Get history (BBL addresses) from pending callback state.
    ///
    /// This is used by Python to initialize history on callback states,
    /// preventing IndexError when hooks access `state.history.recent_bbl_addrs[-1]`.
    /// See [`pending_api::_get_pending_history`] for the body.
    pub fn get_pending_history(&self) -> PyResult<Vec<u64>> {
        self._get_pending_history()
    }

    /// Get jumpkind for pending callback.
    ///
    /// Returns the jumpkind that led to this callback (e.g., "Ijk_Call", "Ijk_Boring").
    /// This is used by Python to properly initialize callstack management.
    /// See [`pending_api::_get_pending_jumpkind`] for the body.
    pub fn get_pending_jumpkind(&self) -> PyResult<String> {
        self._get_pending_jumpkind()
    }

    /// Set register value in pending state.
    /// See [`pending_api::_set_pending_register`] for the body.
    pub fn set_pending_register(&mut self, name: &str, value: u128) -> PyResult<()> {
        self._set_pending_register(name, value)
    }

    /// Set register to a symbolic value from a handle ID.
    ///
    /// Used for syncing symbolic return values from SimProcedures.
    /// The handle_id should reference a RustBV in the solver's symbol table.
    /// See [`pending_api::_set_pending_register_symbolic`] for the body.
    pub fn set_pending_register_symbolic(&mut self, name: &str, handle_id: u64) -> PyResult<()> {
        self._set_pending_register_symbolic(name, handle_id)
    }

    /// Set a symbolic register value in the pending state from claripy AST.
    ///
    /// This allows direct sync of symbolic register values from Python callbacks.
    /// The claripy AST is converted to RustBV and stored in the pending state.
    /// See [`pending_api::_set_pending_register_symbolic_ast`] for the body.
    pub fn set_pending_register_symbolic_ast(
        &mut self,
        py: Python<'_>,
        reg_name: &str,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self._set_pending_register_symbolic_ast(py, reg_name, ast)
    }

    /// Import symbolic memory from Python hook into Rust's symbolic_objects.
    ///
    /// Called after a hook writes symbolic memory. Converts the claripy AST
    /// to RustBV and imports it into the pending state's SymbolicMemory.
    /// Import symbolic memory into a state by ID (for init-time symbolic data).
    /// See [`pending_api::_import_symbolic_to_state`] for the body.
    #[pyo3(signature = (state_id, addr, ast))]
    pub fn import_symbolic_to_state(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        addr: u64,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self._import_symbolic_to_state(py, state_id, addr, ast)
    }

    /// See [`pending_api::_import_symbolic_memory`] for the body.
    pub fn import_symbolic_memory(
        &mut self,
        py: Python<'_>,
        addr: u64,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self._import_symbolic_memory(py, addr, ast)
    }

    /// Get memory from pending state.
    /// See [`pending_api::_get_pending_memory`] for the body.
    pub fn get_pending_memory(&self, addr: u64, size: u32) -> PyResult<Vec<u8>> {
        self._get_pending_memory(addr, size)
    }

    /// Store memory in pending state.
    /// See [`pending_api::_set_pending_memory`] for the body.
    pub fn set_pending_memory(&mut self, addr: u64, data: &[u8]) -> PyResult<()> {
        self._set_pending_memory(addr, data)
    }

    /// Get dirty page addresses from pending state.
    ///
    /// This returns the list of page-aligned addresses that have been
    /// modified in the pending callback state.
    /// See [`pending_api::_get_pending_dirty_pages`] for the body.
    pub fn get_pending_dirty_pages(&self) -> PyResult<Vec<u64>> {
        self._get_pending_dirty_pages()
    }

    /// Clear dirty page tracking in pending state.
    /// See [`pending_api::_clear_pending_dirty_tracking`] for the body.
    pub fn clear_pending_dirty_tracking(&mut self) -> PyResult<()> {
        self._clear_pending_dirty_tracking()
    }

    /// Export pending constraints as a list of claripy ASTs.
    ///
    /// Returns constraints that can be added to Python state.solver.
    /// This exports stored branch conditions accumulated during Rust execution.
    /// See [`pending_api::_export_pending_constraints`] for the body.
    pub fn export_pending_constraints(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self._export_pending_constraints(py)
    }

    /// Get handle IDs that are actively referenced in the pending state.
    ///
    /// Returns handle IDs used in stored conditions and deferred forks.
    /// These should not be evicted from the AST handle cache.
    /// See [`pending_api::_get_active_handle_ids`] for the body.
    pub fn get_active_handle_ids(&self) -> Vec<u64> {
        self._get_active_handle_ids()
    }

    /// Export the pending state as a full snapshot.
    ///
    /// This allows Python to get a complete snapshot of the pending state
    /// including all registers, memory pages, and metadata.
    /// See [`pending_api::_export_pending_state`] for the body.
    pub fn export_pending_state(&self) -> PyResult<crate::state::ExplorationStateSnapshot> {
        self._export_pending_state()
    }

    /// Get the root state ID for the pending callback state.
    ///
    /// When Rust forks states internally, Python only has cached data for the
    /// original state that was added via Python. This method returns the root
    /// state ID (the original state) for any forked descendant.
    ///
    /// Returns:
    ///     The root state ID if available, or None if the state has no tracked root.
    /// See [`pending_api::_get_pending_root_state_id`] for the body.
    pub fn get_pending_root_state_id(&self) -> PyResult<Option<u64>> {
        self._get_pending_root_state_id()
    }

    /// Get the full ancestry chain for the pending callback state.
    ///
    /// Returns a list of state IDs starting with the current state and walking
    /// up the parent chain: [state_id, parent_id, grandparent_id, ...].
    ///
    /// This is used by Python to find cached state data when the current state
    /// is a multi-level fork of an original state.
    /// See [`pending_api::_get_pending_ancestry`] for the body.
    pub fn get_pending_ancestry(&self) -> PyResult<Vec<u64>> {
        self._get_pending_ancestry()
    }

    /// Fork the pending state's solver context for Python callbacks.
    ///
    /// This creates a new RustSolverContext that inherits all constraints
    /// accumulated during Rust exploration. The forked context can be
    /// attached to the Python callback state, ensuring SimProcedures
    /// see the full constraint context.
    ///
    /// This is critical for proper constraint propagation: without it,
    /// callbacks would create fresh solver contexts without parent
    /// constraints, leading to incorrect symbolic evaluation.
    /// Export a callback bundle: registers, solver context, history, jumpkind
    /// in a single FFI call. Reduces ~20 individual calls to 1.
    ///
    /// Returns a Python dict with:
    /// - "registers": dict of register_name -> concrete u128 value (None if symbolic)
    /// - "solver": forked RustSolverContext
    /// - "history": list of u64 BBL addresses
    /// - "jumpkind": string
    /// - "constraint_count": u64
    /// - "stdout": bytes (accumulated stdout buffer)
    /// See [`pending_api::_export_callback_bundle`] for the body.
    #[pyo3(signature = (register_names, shared_solver=true))]
    pub fn export_callback_bundle<'py>(
        &self,
        py: Python<'py>,
        register_names: Vec<String>,
        shared_solver: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        self._export_callback_bundle(py, register_names, shared_solver)
    }

    /// See [`pending_api::_fork_pending_solver`] for the body.
    pub fn fork_pending_solver(&self) -> PyResult<RustSolverContext> {
        self._fork_pending_solver()
    }

    /// Borrow the pending state's solver context without forking.
    ///
    /// Returns a RustSolverContext that shares the same Z3 solver as the
    /// pending state via Rc reference counting. This is O(1) instead of
    /// the ~3ms Z3 solver clone in fork_pending_solver().
    ///
    /// Constraints added through this solver go directly to the pending state,
    /// so post-callback constraint sync via add_constraints_to_pending() should
    /// be skipped to avoid double-adding.
    /// See [`pending_api::_borrow_pending_solver`] for the body.
    pub fn borrow_pending_solver(&self) -> PyResult<RustSolverContext> {
        self._borrow_pending_solver()
    }

    /// Add constraints from Python callbacks back to the pending state.
    ///
    /// This is called after a SimProcedure executes to sync any new
    /// constraints added during the callback back to the Rust solver.
    /// This ensures bidirectional constraint flow between Rust and Python.
    ///
    /// Args:
    ///     constraints: List of claripy AST constraints to add
    /// See [`pending_api::_add_constraints_to_pending`] for the body.
    pub fn add_constraints_to_pending(
        &mut self,
        py: Python<'_>,
        constraints: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<()> {
        self._add_constraints_to_pending(py, constraints)
    }

    /// Add constraints from Python to a state in a stash by state ID.
    /// This is used to sync initial constraints from the Python state.
    pub fn add_constraints_to_state(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        constraints: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<bool> {
        self._add_constraints_to_state(py, state_id, constraints)
    }

    /// Export raw Z3 assertion pointers from a state's solver.
    /// Lossless — captures ALL Z3 assertions, not just those tracked
    /// in assumed_constraints (which drops constraints where claripy_to_rustbv fails).
    #[cfg(feature = "vex-engine-z3")]
    pub fn export_z3_constraint_ptrs(&self, state_id: u64) -> PyResult<Vec<usize>> {
        self._export_z3_constraint_ptrs(state_id)
    }

    /// Import raw Z3 assertion pointers to a state's solver.
    #[cfg(feature = "vex-engine-z3")]
    pub fn import_z3_constraint_ptrs(&mut self, state_id: u64, ptrs: Vec<usize>) -> PyResult<bool> {
        self._import_z3_constraint_ptrs(state_id, ptrs)
    }

    /// Debug: dump solver state for a given state.
    #[cfg(feature = "vex-engine-z3")]
    pub fn debug_solver_info(&self, state_id: u64) -> PyResult<String> {
        self._debug_solver_info(state_id)
    }

    /// Export constraints from a state in any stash as claripy ASTs.
    pub fn export_state_constraints(
        &self,
        py: Python<'_>,
        state_id: u64,
    ) -> PyResult<Vec<Py<PyAny>>> {
        self._export_state_constraints(py, state_id)
    }

    /// Set the Z3 solver timeout (ms) on a specific state's solver context.
    ///
    /// Future forks of this state inherit the new timeout.  Used by
    /// RustSolverProxy to honor `state.solver.timeout = N` assignments.
    pub fn set_state_solver_timeout(&self, state_id: u64, timeout_ms: u32) -> PyResult<()> {
        self._set_state_solver_timeout(state_id, timeout_ms)
    }

    /// Get the Z3 solver timeout (ms) on a specific state's solver context.
    pub fn get_state_solver_timeout(&self, state_id: u64) -> PyResult<u32> {
        self._get_state_solver_timeout(state_id)
    }

    /// Get the per-state mmap base pointer (mirrors Python's
    /// `state.heap.mmap_base`). The native mmap syscall handler bumps this
    /// on `addr=0` calls; Python imports it on stash export to keep the two
    /// engines from handing out overlapping mmap regions.
    pub fn get_state_mmap_base(&self, state_id: u64) -> PyResult<u64> {
        self._get_state_mmap_base(state_id)
    }

    /// Set the per-state mmap base pointer. Used by tests and by Python-side
    /// fallbacks that allocate from `state.heap.mmap_base` and need to push
    /// the advance back into Rust so subsequent native mmaps don't collide.
    pub fn set_state_mmap_base(&mut self, state_id: u64, addr: u64) -> PyResult<()> {
        self._set_state_mmap_base(state_id, addr)
    }

    /// Get the per-state posix brk pointer (mirrors Python's
    /// `state.posix.brk`). The native brk syscall handler bumps this on
    /// concrete `brk(addr)` calls; Python imports it on stash export to keep
    /// a Python-side `set_brk` fallback from handing out heap addresses that
    /// overlap a Rust-allocated region.
    pub fn get_state_posix_brk(&self, state_id: u64) -> PyResult<u64> {
        self._get_state_posix_brk(state_id)
    }

    /// Set the per-state posix brk pointer. Used by tests and by Python-side
    /// fallbacks (symbolic `brk` argument, collision retry) that bump
    /// `state.posix.brk` and need to push the advance back into Rust.
    pub fn set_state_posix_brk(&mut self, state_id: u64, addr: u64) -> PyResult<()> {
        self._set_state_posix_brk(state_id, addr)
    }

    // ----- Per-state Python-AST metadata (symbolic_pages /
    //       hook_symbolic_memory / addr_to_ast). Storage now lives in
    //       RustSimState; these methods are the FFI surface that replaces the
    //       old Python `_state_metadata: Dict[int, StateMetadata]` map.

    /// Replace the whole `symbolic_pages` map for a state. Mirrors the
    /// previous Python-side `_state_md(sid).symbolic_pages = pages` write.
    pub fn set_state_symbolic_pages<'py>(
        &mut self,
        py: Python<'py>,
        state_id: u64,
        pages: &Bound<'py, PyDict>,
    ) -> PyResult<()> {
        self._set_state_symbolic_pages(py, state_id, pages)
    }

    /// Snapshot the `symbolic_pages` map for a state as a Python dict.
    /// Returns an empty dict if the state is unknown or has no entries — the
    /// truthiness check at call sites already handles both cases.
    pub fn get_state_symbolic_pages<'py>(
        &self,
        py: Python<'py>,
        state_id: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        self._get_state_symbolic_pages(py, state_id)
    }

    /// Insert/replace an entry in `hook_symbolic_memory` for a state.
    pub fn set_state_hook_symbolic_memory(
        &mut self,
        state_id: u64,
        addr: u64,
        ast: Py<PyAny>,
        size: u32,
    ) -> PyResult<()> {
        self._set_state_hook_symbolic_memory(state_id, addr, ast, size)
    }

    /// Snapshot the `hook_symbolic_memory` map as a `dict[int, (ast, size)]`.
    pub fn get_state_hook_symbolic_memory<'py>(
        &self,
        py: Python<'py>,
        state_id: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        self._get_state_hook_symbolic_memory(py, state_id)
    }

    /// Insert/replace an entry in `addr_to_ast` for a state.
    pub fn set_state_addr_to_ast(
        &mut self,
        state_id: u64,
        addr: u64,
        ast: Py<PyAny>,
        size: u32,
    ) -> PyResult<()> {
        self._set_state_addr_to_ast(state_id, addr, ast, size)
    }

    /// Snapshot the `addr_to_ast` map as a `dict[int, (ast, size)]`.
    pub fn get_state_addr_to_ast<'py>(
        &self,
        py: Python<'py>,
        state_id: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        self._get_state_addr_to_ast(py, state_id)
    }

    /// Drop all per-state metadata (`symbolic_pages`, `hook_symbolic_memory`,
    /// `addr_to_ast`) for a state. No-op if the state is unknown — matches the
    /// `_state_metadata.pop(state_id, None)` semantics it replaces.
    pub fn clear_state_metadata(&mut self, state_id: u64) -> PyResult<()> {
        self._clear_state_metadata(state_id)
    }

    /// Fork the solver context of an arbitrary state (by ID).
    ///
    /// Returns a new RustSolverContext with all of the state's constraints,
    /// allowing Python to evaluate/solve against any state — not just the
    /// pending callback state.  This is used by RustStateProxy.
    pub fn fork_state_solver(&self, state_id: u64) -> PyResult<RustSolverContext> {
        self._fork_state_solver(state_id)
    }

    /// Get the number of constraints in the pending state's solver.
    /// See [`pending_api::_pending_constraint_count`] for the body.
    pub fn pending_constraint_count(&self) -> PyResult<usize> {
        self._pending_constraint_count()
    }

    /// Get the ID of the state currently being stepped.
    pub fn get_current_stepping_state_id(&self) -> Option<u64> {
        self.current_stepping_state_id
    }

    /// Load from pending callback state's Rust memory.
    /// Used by SimProcedure callbacks to read the correct per-state memory.
    /// Get all mapped page addresses from pending callback state's memory.
    /// See [`pending_api::_get_pending_mapped_pages`] for the body.
    pub fn get_pending_mapped_pages(&self) -> PyResult<Vec<u64>> {
        self._get_pending_mapped_pages()
    }

    /// Load an entire page (4096 bytes) from pending callback state's memory.
    /// See [`pending_api::_pending_memory_load_page`] for the body.
    pub fn pending_memory_load_page(&self, page_addr: u64) -> PyResult<Vec<u8>> {
        self._pending_memory_load_page(page_addr)
    }

    /// Symbolic counterpart of `pending_memory_load_page`: returns the
    /// (addr, claripy AST) pairs for every multi-byte symbolic object whose
    /// base address falls on the given page.
    /// See [`pending_api::_pending_memory_load_symbolic_page`] for the body.
    pub fn pending_memory_load_symbolic_page<'py>(
        &self,
        py: Python<'py>,
        page_addr: u64,
    ) -> PyResult<Vec<(u64, Py<PyAny>)>> {
        self._pending_memory_load_symbolic_page(py, page_addr)
    }

    /// See [`pending_api::_pending_memory_load`] for the body.
    pub fn pending_memory_load(&self, addr: u64, size: u32) -> PyResult<Vec<u8>> {
        self._pending_memory_load(addr, size)
    }

    /// Store to pending callback state's Rust memory.
    /// See [`pending_api::_pending_memory_store`] for the body.
    pub fn pending_memory_store(&mut self, addr: u64, data: &[u8]) -> PyResult<()> {
        self._pending_memory_store(addr, data)
    }

    /// Map memory with data in pending callback state.
    /// See [`pending_api::_pending_memory_map_data`] for the body.
    pub fn pending_memory_map_data(&mut self, addr: u64, data: &[u8], perm: u8) -> PyResult<()> {
        self._pending_memory_map_data(addr, data, perm)
    }

    /// Set address to skip hook check for on next step.
    ///
    /// This is used to prevent infinite loops with zero-length hooks.
    /// When a hook with length=0 runs, it returns to the same address.
    /// Without this skip mechanism, the hook would trigger again immediately.
    ///
    /// The skip is automatically cleared after one step or when the address is used.
    /// GAP 6: Stack-based tracking allows for nested zero-length hooks.
    /// See [`pending_api::_set_skip_hook_addr`] for the body.
    pub fn set_skip_hook_addr(&mut self, addr: u64) {
        self._set_skip_hook_addr(addr)
    }

    /// Clear all pending skip_hook entries.
    /// See [`pending_api::_clear_skip_hook_addr`] for the body.
    pub fn clear_skip_hook_addr(&mut self) {
        self._clear_skip_hook_addr()
    }

    /// Clear skip entry for a specific address.
    /// See [`pending_api::_clear_skip_hook_for_addr`] for the body.
    pub fn clear_skip_hook_for_addr(&mut self, addr: u64) {
        self._clear_skip_hook_for_addr(addr)
    }

    /// Get errors encountered during exploration.
    pub fn get_errors(&self) -> Vec<(u64, String, u64)> {
        self.errors.clone()
    }

    /// Clear error log.
    pub fn clear_errors(&mut self) {
        self.errors.clear();
    }

    /// Move states between stashes.
    /// See [`state_lifecycle::_move_states`] for the body.
    pub fn move_states(
        &mut self,
        from_stash: &str,
        to_stash: &str,
        filter_fn: Option<Py<PyAny>>,
    ) -> PyResult<usize> {
        self._move_states(from_stash, to_stash, filter_fn)
    }

    /// P8 fix: Move a single state by ID between stashes.
    /// See [`state_lifecycle::_move_state`] for the body.
    pub fn move_state(
        &mut self,
        state_id: u64,
        from_stash: &str,
        to_stash: &str,
    ) -> PyResult<bool> {
        self._move_state(state_id, from_stash, to_stash)
    }

    /// P8 fix: Clear all states from a stash.
    pub fn clear_stash(&mut self, stash: &str) {
        self.sm.clear(stash);
    }

    /// Prepare for a new exploration stage: move a specific found state
    /// to active and clear all other stashes. Returns the state ID of the
    /// moved state. This avoids constraint transfer between managers.
    /// See [`state_lifecycle::_reset_for_stage`] for the body.
    pub fn reset_for_stage(&mut self, found_state_id: u64) -> PyResult<u64> {
        self._reset_for_stage(found_state_id)
    }

    /// Get statistics.
    /// See [`stats_api::_stats`] for the body.
    pub fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self._stats(py)
    }

    /// Get VEX fallback statistics: total count and per-address reasons.
    ///
    /// Returns a dict with:
    ///   "count": total number of VEX fallbacks
    ///   "addresses": dict mapping hex address string -> reason string
    ///   "dcas_unsupported_count": subset of fallbacks driven by double-CAS
    ///   "simprocedure_python_fallback_count": SimProcedures dispatched to Python
    ///   "syscall_python_fallback_count": syscalls dispatched to Python
    /// See [`stats_api::_get_fallback_stats`] for the body.
    pub fn get_fallback_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self._get_fallback_stats(py)
    }

    // =========================================================================
    // Native Procedure Management
    // =========================================================================

    /// Disable all native procedures (always use Python).
    pub fn disable_native_procedures(&mut self) {
        self.native_procedures.disable_all();
    }

    /// Enable all native procedures.
    pub fn enable_native_procedures(&mut self) {
        self.native_procedures.enable_all();
    }

    /// Check if native procedures are enabled.
    pub fn native_procedures_enabled(&self) -> bool {
        self.native_procedures.is_enabled()
    }

    /// Disable a specific native procedure (fall back to Python).
    pub fn disable_native_procedure(&mut self, name: &str) {
        self.native_procedures.disable(name);
    }

    /// Enable a specific native procedure.
    pub fn enable_native_procedure(&mut self, name: &str) {
        self.native_procedures.enable(name);
    }

    /// Set a Python override for a procedure.
    ///
    /// When set, the native implementation is never called.
    pub fn set_python_override(&mut self, name: &str) {
        self.native_procedures.set_python_override(name);
    }

    /// Remove a Python override.
    pub fn remove_python_override(&mut self, name: &str) {
        self.native_procedures.remove_python_override(name);
    }

    /// Get list of available native procedures.
    pub fn list_native_procedures(&self) -> Vec<String> {
        self.native_procedures
            .procedure_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Check if a procedure has a native implementation.
    pub fn has_native_procedure(&self, name: &str) -> bool {
        self.native_procedures.has_native(name)
    }

    /// Get native procedure statistics.
    /// See [`stats_api::_native_procedure_stats`] for the body.
    pub fn native_procedure_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self._native_procedure_stats(py)
    }

    /// Register a Python callable as a native procedure.
    ///
    /// The callable receives `list[int]` of concrete arg values and must
    /// return `Optional[int]` for the return value (None = no value).
    /// Symbolic arguments cause an automatic fallback to the regular
    /// Python SimProcedure path; the registered callable is only invoked
    /// when all args are concrete.
    #[pyo3(signature = (name, num_args, no_return, callable))]
    pub fn register_python_procedure(
        &mut self,
        name: String,
        num_args: usize,
        no_return: bool,
        callable: Py<PyAny>,
    ) {
        let proc = std::sync::Arc::new(crate::procedures::python_proc::PythonNativeProcedure::new(
            name, num_args, no_return, callable,
        ));
        self.native_procedures.register(proc);
    }

    // =========================================================================
    // Native Uniqueness Filter
    // =========================================================================

    /// Enable native uniqueness filter with given register names.
    ///
    /// After each step in run(), states with duplicate register tuples
    /// are moved to 'not_unique' stash. This replaces the Python
    /// CheckUniqueness technique with zero FFI overhead.
    pub fn register_uniqueness_filter(&mut self, register_names: Vec<String>) {
        self.constraint_tracker.uniqueness_registers = register_names;
        self.constraint_tracker.uniqueness_set.clear();
        // Ensure not_unique stash exists
        self.sm
            .stashes_mut()
            .entry("not_unique".to_string())
            .or_insert_with(VecDeque::new);
    }

    /// Disable the native uniqueness filter.
    pub fn disable_uniqueness_filter(&mut self) {
        self.constraint_tracker.uniqueness_registers.clear();
        self.constraint_tracker.uniqueness_set.clear();
    }

    /// Check if native uniqueness filter is enabled.
    pub fn uniqueness_filter_enabled(&self) -> bool {
        !self.constraint_tracker.uniqueness_registers.is_empty()
    }

    /// Get the number of unique register tuples seen.
    pub fn uniqueness_set_size(&self) -> usize {
        self.constraint_tracker.uniqueness_set.len()
    }

    // =========================================================================
    // Native Exploration Techniques
    // =========================================================================

    /// Register a native LengthLimiter technique.
    ///
    /// States whose history exceeds `max_length` blocks are moved to "cut"
    /// (or "_DROP" if `drop` is true). Runs entirely in Rust with zero FFI overhead.
    pub fn register_length_limiter(&mut self, max_length: usize, drop: bool) {
        self.native_techniques
            .push(NativeTechnique::LengthLimiter { max_length, drop });
        if !drop {
            self.sm
                .stashes_mut()
                .entry("cut".to_string())
                .or_insert_with(VecDeque::new);
        }
    }

    /// Register a native Timeout technique.
    ///
    /// Exploration stops after `timeout_secs` seconds. All active states are
    /// moved to "timeout" stash. Timer starts on first call to apply_native_techniques().
    pub fn register_timeout(&mut self, timeout_secs: f64) {
        self.native_techniques.push(NativeTechnique::Timeout {
            timeout_secs,
            start_time: None,
        });
        self.sm
            .stashes_mut()
            .entry("timeout".to_string())
            .or_insert_with(VecDeque::new);
    }

    /// Register a native LoopBound technique.
    ///
    /// States where any single address appears more than `bound` times in
    /// their history are moved to `discard_stash`. This is a simplified
    /// version of LoopSeer that doesn't require CFG analysis.
    #[pyo3(signature = (bound, discard_stash="spinning"))]
    pub fn register_loop_bound(&mut self, bound: usize, discard_stash: &str) {
        self.native_techniques.push(NativeTechnique::LoopBound {
            bound,
            discard_stash: discard_stash.to_string(),
        });
        self.sm
            .stashes_mut()
            .entry(discard_stash.to_string())
            .or_insert_with(VecDeque::new);
    }

    /// Get the number of registered native techniques.
    pub fn native_technique_count(&self) -> usize {
        self.native_techniques.len()
    }

    /// Clear all native techniques.
    pub fn clear_native_techniques(&mut self) {
        self.native_techniques.clear();
    }

    // =========================================================================
    // State Export Methods
    // =========================================================================

    /// Export a state by ID as a full snapshot.
    ///
    /// This searches all stashes for the state with the given ID and returns
    /// a complete snapshot that can be used to reconstruct an angr SimState.
    pub fn export_state(&self, state_id: u64) -> PyResult<crate::state::ExplorationStateSnapshot> {
        self._export_state(state_id)
    }

    /// Export a state by ID, flushing pending writes first.
    pub fn export_state_flushed(
        &mut self,
        state_id: u64,
    ) -> PyResult<crate::state::ExplorationStateSnapshot> {
        self._export_state_flushed(state_id)
    }

    /// Export all states in a stash as snapshots.
    pub fn export_stash(&self, stash: &str) -> Vec<crate::state::ExplorationStateSnapshot> {
        self._export_stash(stash)
    }

    /// Export all found states as snapshots (flushing pending writes).
    pub fn export_found_states_flushed(&mut self) -> Vec<crate::state::ExplorationStateSnapshot> {
        self._export_found_states_flushed()
    }

    /// Export all found states as snapshots.
    pub fn export_found_states(&self) -> Vec<crate::state::ExplorationStateSnapshot> {
        self._export_found_states()
    }

    /// Evaluate a symbolic value in a state's solver context.
    ///
    /// This allows Python to get concrete values for symbolic inputs
    /// that were found during exploration.
    pub fn eval_in_state(&self, state_id: u64, addr: u64, size: u32) -> PyResult<Option<Vec<u8>>> {
        self._eval_in_state(state_id, addr, size)
    }

    /// Debug: Get symbolic object info for a state.
    pub fn state_symbolic_info(&self, state_id: u64, addr: u64) -> PyResult<String> {
        self._state_symbolic_info(state_id, addr)
    }

    /// Get Z3 AST pointers for all symbolic objects in a state's memory.
    ///
    /// Returns Vec<(addr, z3_ast_ptr_as_usize, width_bits)> for each symbolic
    /// object. The Z3 ASTs are built in the shared Z3 context, so Python can
    /// directly wrap them as z3.BitVecRef and convert to claripy ASTs.
    ///
    /// This is used to export Rust-computed symbolic expressions (e.g., flag
    /// computations in asisctf) to Python state memory during state export.
    #[cfg(feature = "vex-engine-z3")]
    pub fn get_state_symbolic_z3_asts(&self, state_id: u64) -> PyResult<Vec<(u64, usize, u32)>> {
        self._get_state_symbolic_z3_asts(state_id)
    }

    /// Check if constraints are satisfiable for a state.
    pub fn state_satisfiable(&self, state_id: u64) -> PyResult<bool> {
        self._state_satisfiable(state_id)
    }

    /// Whether strict memory permission enforcement is enabled on a state.
    /// Mirrors angr's STRICT_PAGE_ACCESS option.
    pub fn state_enforce_permissions(&self, state_id: u64) -> PyResult<bool> {
        self._state_enforce_permissions(state_id)
    }

    /// Whether non-executable page enforcement is enabled on a state.
    /// Mirrors angr's ENABLE_NX option.
    pub fn state_enforce_nx(&self, state_id: u64) -> PyResult<bool> {
        self._state_enforce_nx(state_id)
    }

    /// Get a register value from a state.
    pub fn get_state_register(&self, state_id: u64, name: &str) -> PyResult<Option<u128>> {
        self._get_state_register(state_id, name)
    }

    /// Get multiple register values from a state in one FFI call.
    /// Returns a list of Option<u128> in the same order as the input names.
    pub fn get_state_registers_batch(
        &self,
        state_id: u64,
        names: Vec<String>,
    ) -> PyResult<Vec<Option<u128>>> {
        self._get_state_registers_batch(state_id, names)
    }

    /// Get memory from a state.
    pub fn get_state_memory(
        &self,
        state_id: u64,
        addr: u64,
        size: u32,
    ) -> PyResult<Option<Vec<u8>>> {
        self._get_state_memory(state_id, addr, size)
    }

    /// Phase 1.4 (angr-5zw8): route a symbolic-address store through the
    /// Multi-cell lazy path on the given state. Used by
    /// `_cb_memory_store_symbolic_full` when the address AST carries a
    /// `MultiwriteAnnotation` — the SimProcedures in `libc/strchr.py`,
    /// `libc/gets.py`, `libc/fgets.py` tag returned addresses with this
    /// annotation so Range concretization picks up >1 candidate.
    ///
    /// Returns `true` on success. Returns `false` if conversion or store
    /// fails (caller should fall back to the existing Python state path
    /// to keep progress).
    pub fn state_memory_store_symbolic_multi<'py>(
        &mut self,
        py: Python<'py>,
        state_id: u64,
        addr_ast: &Bound<'py, PyAny>,
        data_ast: &Bound<'py, PyAny>,
    ) -> PyResult<bool> {
        self._state_memory_store_symbolic_multi(py, state_id, addr_ast, data_ast)
    }

    /// Check if a state has stdout output (dirty flag check, no allocation).
    pub fn has_state_stdout(&self, state_id: u64) -> bool {
        self._has_state_stdout(state_id)
    }

    /// Get the stdout buffer for a state by ID.
    ///
    /// Returns the accumulated output from native puts/printf calls.
    pub fn get_state_stdout(&self, state_id: u64) -> PyResult<Vec<u8>> {
        self._get_state_stdout(state_id)
    }

    /// Get the output buffer for a specific file descriptor.
    ///
    /// Returns the accumulated output from native write/puts/printf calls.
    pub fn get_state_fd_output(&self, state_id: u64, fd: u32) -> PyResult<Vec<u8>> {
        self._get_state_fd_output(state_id, fd)
    }

    /// Check if a state has recorded stdin symbols from native fgets/fgetc/getchar.
    pub fn has_state_stdin_symbols(&self, state_id: u64) -> bool {
        self._has_state_stdin_symbols(state_id)
    }

    /// Get the stdin symbols for a state by ID.
    ///
    /// Returns list of (name, bit_width) tuples for symbolic variables
    /// created by native fgets/fgetc/getchar. Used to reconstruct stdin
    /// data in Python's posix plugin for posix.dumps(0).
    pub fn get_state_stdin_symbols(&self, state_id: u64) -> PyResult<Vec<(String, u32)>> {
        self._get_state_stdin_symbols(state_id)
    }

    /// Get the call stack for a state by ID.
    ///
    /// Returns list of (call_site_addr, callee_addr, return_addr, stack_ptr) tuples.
    pub fn get_state_call_stack(&self, state_id: u64) -> PyResult<Vec<(u64, u64, u64, u64)>> {
        self._get_state_call_stack(state_id)
    }

    /// Get the call stack depth for a state by ID.
    pub fn get_state_call_stack_depth(&self, state_id: u64) -> PyResult<usize> {
        self._get_state_call_stack_depth(state_id)
    }

    /// Get the detailed execution history for a state by ID.
    ///
    /// Returns list of (addr, jumpkind, jump_target) tuples.
    /// jumpkind: 0=Boring, 1=Call, 2=Ret, 3=Syscall, 4=Other
    pub fn get_state_detailed_history(&self, state_id: u64) -> PyResult<Vec<(u64, u8, u64)>> {
        self._get_state_detailed_history(state_id)
    }

    /// Get heap metadata for a state by ID.
    ///
    /// Returns dict with:
    /// - allocated: list of (addr, size) tuples for active allocations
    /// - freed: list of freed addresses
    /// - alloc_count: number of active allocations
    /// - free_count: number of free calls
    pub fn get_state_heap_metadata(&self, state_id: u64) -> PyResult<(Vec<(u64, u64)>, Vec<u64>)> {
        self._get_state_heap_metadata(state_id)
    }

    /// Get the list of open file descriptors for a state.
    ///
    /// Returns list of (fd, name, position, flags, content_len, is_open) tuples.
    pub fn get_state_open_fds(
        &self,
        state_id: u64,
    ) -> PyResult<Vec<(u32, String, u64, u32, usize, bool)>> {
        self._get_state_open_fds(state_id)
    }

    /// Get the content of a file descriptor for a state.
    pub fn get_state_fd_content(&self, state_id: u64, fd: u32) -> PyResult<Vec<u8>> {
        self._get_state_fd_content(state_id, fd)
    }

    /// Enable inspection for an event type on a state.
    ///
    /// event_type: 0=MemRead, 1=MemWrite, 2=RegRead, 3=RegWrite, 4=Fork, 5=Exit
    pub fn enable_state_inspection(&mut self, state_id: u64, event_type: u8) -> PyResult<()> {
        self._enable_state_inspection(state_id, event_type)
    }

    /// Enable all inspections on a state.
    pub fn enable_all_inspections(&mut self, state_id: u64) -> PyResult<()> {
        self._enable_all_inspections(state_id)
    }

    /// Get inspection event counts for a state.
    ///
    /// Returns list of (event_name, count) tuples for events with count > 0.
    pub fn get_state_inspection_counts(&self, state_id: u64) -> PyResult<Vec<(String, u64)>> {
        self._get_state_inspection_counts(state_id)
    }

    /// Get inspection events for a state.
    ///
    /// Returns list of (event_type, event_name, addr, size, block_addr) tuples.
    pub fn get_state_inspection_events(
        &self,
        state_id: u64,
    ) -> PyResult<Vec<(u8, String, u64, u32, u64)>> {
        self._get_state_inspection_events(state_id)
    }

    /// Evaluate a stdin symbol by name using the state's solver.
    ///
    /// Returns the concrete value as Option<u64>, or None if the symbol
    /// cannot be found or evaluated.
    pub fn eval_stdin_symbol(&self, state_id: u64, name: &str) -> Option<u64> {
        self._eval_stdin_symbol(state_id, name)
    }

    // =========================================================================
    // Run loop (body in run_loop.rs)
    // =========================================================================

    /// Run the exploration loop.
    ///
    /// Returns an ExplorationEvent when:
    /// - Found enough solutions
    /// - Active stash is empty
    /// - Need Python callback (SimProcedure, syscall)
    /// - Max steps reached
    #[pyo3(signature = (n=None))]
    pub fn run(&mut self, py: Python<'_>, n: Option<u32>) -> PyResult<ExplorationEvent> {
        self.run_loop(py, n)
    }

    // =========================================================================
    // Resume methods (from resume.rs)
    // =========================================================================

    /// Resume after a SimProcedure callback. See [`resume::_resume_after_simprocedure`] for the body.
    #[pyo3(signature = (new_pc, register_changes=None, memory_changes=None, new_constraints=None))]
    pub fn resume_after_simprocedure(
        &mut self,
        py: Python<'_>,
        new_pc: u64,
        register_changes: Option<Vec<(u32, u32, Vec<u8>)>>,
        memory_changes: Option<Vec<(u64, Vec<u8>)>>,
        new_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
    ) -> PyResult<()> {
        self._resume_after_simprocedure(
            py,
            new_pc,
            register_changes,
            memory_changes,
            new_constraints,
        )
    }

    /// Resume after a syscall callback.
    #[pyo3(signature = (new_pc, register_changes=None, memory_changes=None, new_constraints=None))]
    pub fn resume_after_syscall(
        &mut self,
        py: Python<'_>,
        new_pc: u64,
        register_changes: Option<Vec<(u32, u32, Vec<u8>)>>,
        memory_changes: Option<Vec<(u64, Vec<u8>)>>,
        new_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
    ) -> PyResult<()> {
        // Same as resume_after_simprocedure - ensures constraint sync (GAP 2)
        self._resume_after_simprocedure(
            py,
            new_pc,
            register_changes,
            memory_changes,
            new_constraints,
        )
    }

    /// Resume after a hook callback. Same semantics as resume_after_simprocedure.
    #[pyo3(signature = (new_pc, register_changes=None, memory_changes=None, new_constraints=None))]
    pub fn resume_after_hook(
        &mut self,
        py: Python<'_>,
        new_pc: u64,
        register_changes: Option<Vec<(u32, u32, Vec<u8>)>>,
        memory_changes: Option<Vec<(u64, Vec<u8>)>>,
        new_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
    ) -> PyResult<()> {
        // Same as resume_after_simprocedure - ensures constraint sync (GAP 2)
        self._resume_after_simprocedure(
            py,
            new_pc,
            register_changes,
            memory_changes,
            new_constraints,
        )
    }

    /// Fast-path: deadend the pending callback state without full apply_changes.
    /// Used for SimProcedure continuations known to just call exit().
    /// See [`resume::_deadend_pending_callback`] for the body.
    pub fn deadend_pending_callback(&mut self) -> PyResult<()> {
        self._deadend_pending_callback()
    }

    /// Resume after an error occurred during callback execution (P17).
    /// See [`resume::_resume_after_error`] for the body.
    pub fn resume_after_error(&mut self, error_msg: &str) -> PyResult<()> {
        self._resume_after_error(error_msg)
    }

    /// Resume after Python handles a symbolic branch.
    /// See [`resume::_resume_after_symbolic_branch`] for the body.
    #[pyo3(signature = (true_pc, false_pc, true_constraints=None, false_constraints=None))]
    pub fn resume_after_symbolic_branch(
        &mut self,
        py: Python<'_>,
        true_pc: u64,
        false_pc: u64,
        true_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
        false_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
    ) -> PyResult<()> {
        self._resume_after_symbolic_branch(
            py,
            true_pc,
            false_pc,
            true_constraints,
            false_constraints,
        )
    }

    /// Resume after Python evaluates a find predicate (P2).
    /// See [`resume::_resume_find_predicate`] for the body.
    pub fn resume_find_predicate(&mut self, matched: bool) -> PyResult<()> {
        self._resume_find_predicate(matched)
    }

    /// Resume after Python evaluates an avoid predicate (P7).
    /// See [`resume::_resume_avoid_predicate`] for the body.
    pub fn resume_avoid_predicate(&mut self, matched: bool) -> PyResult<()> {
        self._resume_avoid_predicate(matched)
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

/// Register the exploration module with Python.
pub fn register_exploration(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustExplorationManager>()?;
    m.add_class::<ExplorationEvent>()?;
    m.add_function(pyo3::wrap_pyfunction!(get_stepping_state_id, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exploration_manager_creation() {
        pyo3::prepare_freethreaded_python();
        Python::attach(|_py| {
            let mgr = RustExplorationManager::new("amd64", None).unwrap();
            assert_eq!(mgr.arch(), "amd64");
            assert_eq!(mgr.active_count(), 0);
            assert_eq!(mgr.found_count(), 0);
        });
    }

    #[test]
    fn test_stash_management() {
        pyo3::prepare_freethreaded_python();
        Python::attach(|_py| {
            let mut mgr = RustExplorationManager::new("amd64", None).unwrap();

            // Create state
            let state_id = mgr.create_state("active").unwrap();
            // state_id is assigned from a global atomic counter starting at 0
            assert!(state_id < u64::MAX);
            assert_eq!(mgr.active_count(), 1);

            // Check state IDs
            let ids = mgr.get_state_ids("active");
            assert_eq!(ids.len(), 1);
            assert_eq!(ids[0], state_id);
        });
    }

    #[test]
    fn test_find_avoid_addresses() {
        pyo3::prepare_freethreaded_python();
        Python::attach(|_py| {
            let mut mgr = RustExplorationManager::new("amd64", None).unwrap();

            mgr.set_find_addrs(vec![0x1000, 0x2000]);
            mgr.set_avoid_addrs(vec![0x3000]);

            assert!(mgr.find_addrs.contains(&0x1000));
            assert!(mgr.find_addrs.contains(&0x2000));
            assert!(mgr.avoid_addrs.contains(&0x3000));
        });
    }

    /// Orchestrator semantics: pyclass setters mutate the sub-struct, not a
    /// shadow field on the manager. This test asserts the delegation pattern
    /// established by the angr-4j5u decomposition — `set_*` and `get_*` round
    /// through `constraint_solver`, `memory_config`, `environment`, and
    /// `profiling` respectively.
    #[test]
    fn test_orchestrator_delegation() {
        pyo3::prepare_freethreaded_python();
        Python::attach(|_py| {
            let mut mgr = RustExplorationManager::new("amd64", None).unwrap();

            // ConstraintSolver: lazy_solves + solver_timeout_ms
            mgr.set_lazy_solves(true);
            assert!(mgr.constraint_solver.lazy_solves);
            mgr.set_solver_timeout(7777);
            assert_eq!(mgr.constraint_solver.solver_timeout_ms, 7777);

            // MemoryConfiguration: zero_fill + vex_opt_level (global + override)
            mgr.set_zero_fill_unconstrained(true);
            assert!(mgr.memory_config.zero_fill_unconstrained);
            mgr.set_vex_opt_level(Some(2));
            assert_eq!(mgr.get_vex_opt_level(), Some(2));
            mgr.set_vex_opt_level_override(0xdead, 0);
            assert_eq!(mgr.resolve_vex_opt_level(0xdead), Some(0));
            assert_eq!(mgr.resolve_vex_opt_level(0xbeef), Some(2));

            // ExecutionEnvironment: max_history applies to existing states
            mgr.set_max_history(42);
            assert_eq!(mgr.environment.max_history, 42);
            assert_eq!(mgr.get_max_history(), 42);

            // ProfilingCollector: enable flag flips
            mgr.set_profiling(true);
            assert!(mgr.profiling.profiling_enabled);

            // ConstraintTracker: uniqueness filter reaches the sub-struct
            mgr.register_uniqueness_filter(vec!["rax".into(), "rbx".into()]);
            assert!(mgr.uniqueness_filter_enabled());
            assert_eq!(mgr.constraint_tracker.uniqueness_registers.len(), 2);
            mgr.disable_uniqueness_filter();
            assert!(!mgr.uniqueness_filter_enabled());
        });
    }

    /// State lifecycle: create + move + reset_for_stage round-trip through the
    /// extracted bodies in `state_lifecycle.rs`. Verifies that the StashManager
    /// remains the canonical state-storage and the index is kept in sync.
    #[test]
    fn test_state_lifecycle_orchestration() {
        pyo3::prepare_freethreaded_python();
        Python::attach(|_py| {
            let mut mgr = RustExplorationManager::new("amd64", None).unwrap();

            let s1 = mgr.create_state("active").unwrap();
            let s2 = mgr.create_state("active").unwrap();
            assert_eq!(mgr.active_count(), 2);
            assert_eq!(mgr.state_stash(s1).as_deref(), Some("active"));

            // Move s1 to a custom stash; index follows.
            assert!(mgr.move_state(s1, "active", "found").unwrap());
            assert_eq!(mgr.state_stash(s1).as_deref(), Some("found"));
            assert_eq!(mgr.active_count(), 1);
            assert_eq!(mgr.found_count(), 1);

            // reset_for_stage: keep s1, drop everything else.
            let kept = mgr.reset_for_stage(s1).unwrap();
            assert_eq!(kept, s1);
            assert_eq!(mgr.active_count(), 1);
            assert_eq!(mgr.found_count(), 0);
            let active_ids = mgr.get_state_ids("active");
            assert_eq!(active_ids, vec![s1]);
        });
    }
}
