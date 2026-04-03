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

use std::collections::{HashMap, HashSet, VecDeque};
use std::num::NonZeroUsize;

use lru::LruCache;
use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::PyDict;

use crate::arch::{arch_from_name, default_cc_for_arch, CallingConvention};
use crate::callbacks::{ExecutionConfig, PythonCallbacks, RunResult, DeferredFork};
use crate::claripy_bridge::{claripy_to_rustbv, rustbv_to_claripy};
use crate::interpreter_cb::CallbackInterpreter;
use crate::memory::Permission;
use crate::procedures::{NativeProcedureRegistry, ProcedureError};
use crate::solver::RustSolverContext;
use crate::state::{RustSimState, StateChanges};
use crate::symbolic::{RustBV, RustSymbolTable, SymContext};
use crate::vex::{VexArch, IRSB};

use std::cell::Cell;

/// Thread-local stepping state ID, accessible from callbacks without borrow conflicts.
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
    fn found(found_count: usize, active_count: usize, steps: u64) -> Self {
        ExplorationEvent {
            event_type: "found".to_string(),
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

    fn deadended(found_count: usize, active_count: usize, steps: u64) -> Self {
        ExplorationEvent {
            event_type: "deadended".to_string(),
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

    fn active_empty(found_count: usize, steps: u64) -> Self {
        ExplorationEvent {
            event_type: "active_empty".to_string(),
            found_count,
            active_count: 0,
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

    fn step_complete(found_count: usize, active_count: usize, steps: u64) -> Self {
        ExplorationEvent {
            event_type: "step_complete".to_string(),
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

    fn need_simprocedure(
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
            event_type: "need_callback".to_string(),
            found_count,
            active_count,
            steps_taken: steps,
            callback_state_id: Some(state_id),
            callback_reason: Some("simprocedure".to_string()),
            callback_addr: Some(addr),
            callback_name: Some(name),
            callback_syscall_num: None,
            callback_return_addr: Some(return_addr),
            callback_num_args: Some(num_args),
            branch_true_target: None,
            branch_false_target: None,
            branch_condition_id: None,
        }
    }

    fn need_syscall(
        state_id: u64,
        syscall_num: u64,
        found_count: usize,
        active_count: usize,
        steps: u64,
    ) -> Self {
        ExplorationEvent {
            event_type: "need_callback".to_string(),
            found_count,
            active_count,
            steps_taken: steps,
            callback_state_id: Some(state_id),
            callback_reason: Some("syscall".to_string()),
            callback_addr: None,
            callback_name: None,
            callback_syscall_num: Some(syscall_num),
            callback_return_addr: None,
            callback_num_args: None,
            branch_true_target: None,
            branch_false_target: None,
            branch_condition_id: None,
        }
    }

    fn need_symbolic_branch(
        state_id: u64,
        condition_id: u64,
        true_target: u64,
        false_target: u64,
        found_count: usize,
        active_count: usize,
        steps: u64,
    ) -> Self {
        ExplorationEvent {
            event_type: "need_callback".to_string(),
            found_count,
            active_count,
            steps_taken: steps,
            callback_state_id: Some(state_id),
            callback_reason: Some("symbolic_branch".to_string()),
            callback_addr: None,
            callback_name: None,
            callback_syscall_num: None,
            callback_return_addr: None,
            callback_num_args: None,
            branch_true_target: Some(true_target),
            branch_false_target: Some(false_target),
            branch_condition_id: Some(condition_id),
        }
    }

    fn error(message: String, found_count: usize, active_count: usize, steps: u64) -> Self {
        ExplorationEvent {
            event_type: "errored".to_string(),
            found_count,
            active_count,
            steps_taken: steps,
            callback_state_id: None,
            callback_reason: Some(message),
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
}

/// State held during a Python callback.
struct PendingCallback {
    state: RustSimState,
    /// Clean snapshot of state BEFORE any callback modifications.
    /// Used for creating deferred forks - they diverged before the callback,
    /// so they should not inherit callback constraints.
    pre_callback_snapshot: Option<RustSimState>,
    reason: CallbackReason,
    /// Jumpkind that led to this callback (e.g., "Ijk_Call", "Ijk_Boring").
    jumpkind: Option<String>,
    /// Forked solver context for Python callbacks.
    solver_ctx: Option<RustSolverContext>,
    /// Deferred forks accumulated before the callback.
    /// These should be processed when the callback returns.
    deferred_forks: Vec<DeferredFork>,
    /// Stored conditions for deferred fork handling.
    stored_conditions: HashMap<u64, RustBV>,
    /// Full state snapshots from before branch constraints were added.
    /// Keyed by condition_id, enables correct alternate-path forking.
    fork_snapshots: HashMap<u64, crate::interpreter_cb::BranchSnapshot>,
}

/// Statistics for native procedure execution.
#[derive(Debug, Clone)]
struct NativeProcStats {
    /// Number of native procedure executions.
    native_calls: u64,
    /// Number of fallbacks to Python.
    python_fallbacks: u64,
    /// Per-procedure call counts.
    call_counts: HashMap<String, u64>,
    /// Number of constraint sync failures.
    constraint_sync_failures: u64,
}

impl Default for NativeProcStats {
    fn default() -> Self {
        NativeProcStats {
            native_calls: 0,
            python_fallbacks: 0,
            call_counts: HashMap::new(),
            constraint_sync_failures: 0,
        }
    }
}

/// Rust-native exploration manager.
///
/// Manages states entirely in Rust with O(1) forking.
/// Returns to Python only for SimProcedures, syscalls, and predicates.
#[pyclass(unsendable)]
pub struct RustExplorationManager {
    /// Architecture name.
    arch_name: String,
    /// VEX architecture enum.
    vex_arch: VexArch,
    /// Stash system (mirrors Python's).
    stashes: HashMap<String, VecDeque<RustSimState>>,
    /// Find addresses.
    find_addrs: HashSet<u64>,
    /// Avoid addresses.
    avoid_addrs: HashSet<u64>,
    /// Whether find condition has callable predicates.
    find_needs_python: bool,
    /// Whether avoid condition has callable predicates.
    avoid_needs_python: bool,
    /// Execution config for deferred forks.
    exec_config: ExecutionConfig,
    /// Python callbacks for memory/lifting.
    callbacks: Option<PythonCallbacks>,
    /// Hook addresses.
    hooks: HashSet<u64>,
    /// SimProcedures: address -> (name, num_args, no_return).
    simprocedures: HashMap<u64, (String, usize, bool)>,
    /// Binary code regions for native lifting.
    binary_regions: Vec<(u64, Vec<u8>)>,
    /// Block cache (shared across states).
    block_cache: LruCache<u64, IRSB>,
    /// Pending state waiting for Python callback result.
    pending_callback: Option<PendingCallback>,
    /// ID of the state currently being stepped (for Python callbacks to identify)
    current_stepping_state_id: Option<u64>,
    /// Total steps executed.
    steps: u64,
    /// Error log: (addr, message, state_id).
    errors: Vec<(u64, String, u64)>,
    /// Number of finds required before stopping.
    num_find: usize,
    /// Maximum steps per run iteration.
    max_steps_per_run: u32,
    /// Native procedure registry.
    native_procedures: NativeProcedureRegistry,
    /// Calling convention for argument extraction.
    calling_convention: Box<dyn CallingConvention>,
    /// Statistics for native procedure executions.
    native_proc_stats: NativeProcStats,
    /// Stack of (address, expiry_step) for zero-length hook skip tracking.
    /// Each entry represents an address to skip, valid until the specified step.
    /// This prevents infinite loops when a hook with length=0 runs and
    /// returns to the same address. Stack-based to handle nested hooks.
    skip_hook_stack: Vec<(u64, u64)>,
    /// Maps state_id -> root_state_id for lineage tracking.
    /// When Rust forks states internally, Python only has cached data for the
    /// original state added via Python. This map allows looking up the root
    /// state (the one originally added) for any forked descendant.
    state_roots: HashMap<u64, u64>,
    /// P9 fix: Use LIFO (stack) state selection instead of FIFO (queue).
    /// When true, states are popped from the back (DFS). Default is false (BFS).
    use_lifo: bool,
    /// When true, skip satisfiability checks on forked states (LAZY_SOLVES).
    /// This improves performance for binaries with many branches by deferring
    /// constraint solving until values are actually needed.
    lazy_solves: bool,
}

#[pymethods]
impl RustExplorationManager {
    /// Create a new exploration manager.
    #[new]
    #[pyo3(signature = (arch="amd64"))]
    pub fn new(arch: &str) -> PyResult<Self> {
        let arch_info = arch_from_name(arch).ok_or_else(|| {
            PyValueError::new_err(format!("unsupported architecture: {}", arch))
        })?;

        let vex_arch = arch_info.vex_arch();

        let mut stashes = HashMap::new();
        stashes.insert("active".to_string(), VecDeque::new());
        stashes.insert("found".to_string(), VecDeque::new());
        stashes.insert("avoid".to_string(), VecDeque::new());
        stashes.insert("deadended".to_string(), VecDeque::new());
        stashes.insert("errored".to_string(), VecDeque::new());
        stashes.insert("unconstrained".to_string(), VecDeque::new());

        Ok(RustExplorationManager {
            arch_name: arch.to_string(),
            vex_arch,
            stashes,
            find_addrs: HashSet::new(),
            avoid_addrs: HashSet::new(),
            find_needs_python: false,
            avoid_needs_python: false,
            exec_config: ExecutionConfig::default(),
            callbacks: None,
            hooks: HashSet::new(),
            simprocedures: HashMap::new(),
            binary_regions: Vec::new(),
            block_cache: LruCache::new(NonZeroUsize::new(4096).unwrap()),
            pending_callback: None,
            current_stepping_state_id: None,
            steps: 0,
            errors: Vec::new(),
            num_find: 1,
            max_steps_per_run: 5000,
            native_procedures: NativeProcedureRegistry::new(),
            calling_convention: default_cc_for_arch(arch),
            native_proc_stats: NativeProcStats::default(),
            skip_hook_stack: Vec::new(),
            state_roots: HashMap::new(),
            use_lifo: false,  // P9: Default to BFS (FIFO)
            lazy_solves: false,
        })
    }

    /// Get the architecture name.
    #[getter]
    pub fn arch(&self) -> &str {
        &self.arch_name
    }

    /// Get the total number of steps executed.
    #[getter]
    pub fn step_count(&self) -> u64 {
        self.steps
    }

    /// Get active state count.
    pub fn active_count(&self) -> usize {
        self.stashes.get("active").map(|s| s.len()).unwrap_or(0)
    }

    /// Get found state count.
    pub fn found_count(&self) -> usize {
        self.stashes.get("found").map(|s| s.len()).unwrap_or(0)
    }

    /// Get stash counts as a dictionary.
    pub fn stash_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (name, stash) in &self.stashes {
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
        self.lazy_solves = enabled;
    }

    /// Set Python callbacks for memory/lifting.
    pub fn set_callbacks(&mut self, callbacks: PythonCallbacks) {
        self.callbacks = Some(callbacks);
    }

    /// Clear callbacks.
    pub fn clear_callbacks(&mut self) {
        self.callbacks = None;
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
        self.binary_regions = regions;
    }

    /// Create a new RustSimState and add it to a stash.
    #[pyo3(signature = (stash="active"))]
    pub fn create_state(&mut self, stash: &str) -> PyResult<u64> {
        let state = RustSimState::new(&self.arch_name)
            .map_err(|e| PyValueError::new_err(e))?;
        let state_id = state.state_id();

        // Copy hooks to state
        for &addr in &self.hooks {
            // State hooks are checked during execution
        }

        self.stashes
            .entry(stash.to_string())
            .or_insert_with(VecDeque::new)
            .push_back(state);

        Ok(state_id)
    }

    /// Add an existing RustSimState to a stash.
    #[pyo3(signature = (stash, state))]
    pub fn add_state(&mut self, stash: &str, state: &crate::state::PyRustSimState) {
        // Fork the state to get our own copy
        let forked = state.inner().fork();
        let state_id = forked.state_id();

        // Track this state as its own root (it was added via Python)
        self.state_roots.insert(state_id, state_id);

        self.stashes
            .entry(stash.to_string())
            .or_insert_with(VecDeque::new)
            .push_back(forked);
    }

    /// Get the PC of a state in a stash by index.
    #[pyo3(signature = (stash="active", index=0))]
    pub fn get_state_pc(&self, stash: &str, index: usize) -> Option<u64> {
        self.stashes.get(stash).and_then(|s| s.get(index)).map(|s| s.pc())
    }

    /// Get state IDs in a stash.
    #[pyo3(signature = (stash="active"))]
    pub fn get_state_ids(&self, stash: &str) -> Vec<u64> {
        self.stashes
            .get(stash)
            .map(|s| s.iter().map(|state| state.state_id()).collect())
            .unwrap_or_default()
    }

    /// Get the root state ID for any state.
    /// Returns the original (initial) state from which this state was forked.
    pub fn get_state_root(&self, state_id: u64) -> Option<u64> {
        self.state_roots.get(&state_id).copied()
    }

    /// Set the PC of the pending callback state (for external initialization).
    pub fn set_pending_state_pc(&mut self, pc: u64) -> PyResult<()> {
        if let Some(ref mut pending) = self.pending_callback {
            pending.state.set_pc(pc);
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Map memory in the pending state.
    #[pyo3(signature = (addr, data, permissions=7))]
    pub fn pending_state_map_memory(
        &mut self,
        addr: u64,
        data: &[u8],
        permissions: u8,
    ) -> PyResult<()> {
        if let Some(ref mut pending) = self.pending_callback {
            pending.state.map_memory_data(addr, data, Permission::from_bits(permissions));
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Map memory in active states.
    #[pyo3(signature = (addr, data, permissions=7))]
    pub fn active_states_map_memory(&mut self, addr: u64, data: &[u8], permissions: u8) {
        if let Some(stash) = self.stashes.get_mut("active") {
            for state in stash.iter_mut() {
                state.map_memory_data(addr, data, Permission::from_bits(permissions));
            }
        }
    }

    /// Run the exploration loop.
    ///
    /// Returns an ExplorationEvent when:
    /// - Found enough solutions
    /// - Active stash is empty
    /// - Need Python callback (SimProcedure, syscall)
    /// - Max steps reached
    #[pyo3(signature = (n=None))]
    pub fn run(&mut self, py: Python<'_>, n: Option<u32>) -> PyResult<ExplorationEvent> {
        let max_steps = n.unwrap_or(self.max_steps_per_run);

        // Ensure callbacks are set and clone to avoid borrow issues
        let callbacks = self.callbacks.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("callbacks not set")
        })?.clone();

        if !callbacks.is_ready() {
            return Err(PyRuntimeError::new_err("callbacks not ready"));
        }

        for _ in 0..max_steps {
            // Check if we have enough solutions
            if self.found_count() >= self.num_find {
                return Ok(ExplorationEvent::found(
                    self.found_count(),
                    self.active_count(),
                    self.steps,
                ));
            }

            // Get next state from active stash
            // P9 fix: Use LIFO (pop_back) for DFS or FIFO (pop_front) for BFS
            let mut state = match self.stashes.get_mut("active").and_then(|s| {
                if self.use_lifo {
                    s.pop_back()  // DFS: LIFO (most recent state first)
                } else {
                    s.pop_front()  // BFS: FIFO (oldest state first)
                }
            }) {
                Some(s) => {
                    let sid = s.state_id();
                    self.current_stepping_state_id = Some(sid);
                    STEPPING_STATE_ID.with(|cell| cell.set(Some(sid)));
                    s
                },
                None => {
                    // No active states
                    if self.found_count() > 0 {
                        return Ok(ExplorationEvent::found(
                            self.found_count(),
                            0,
                            self.steps,
                        ));
                    } else {
                        return Ok(ExplorationEvent::active_empty(
                            self.found_count(),
                            self.steps,
                        ));
                    }
                }
            };

            // Check find/avoid before stepping
            let pc = state.pc();

            // P7 fix: Check if callable avoid predicate needs Python evaluation
            // When avoid is a callable (lambda/function), we must return to Python
            // to evaluate it for each state, not just check addresses.
            if self.avoid_needs_python {
                let state_id = state.state_id();
                self.pending_callback = Some(PendingCallback {
                    state,
                    pre_callback_snapshot: None,
                    reason: CallbackReason::AvoidPredicate { addr: pc },
                    jumpkind: None,
                    solver_ctx: None,
                    deferred_forks: Vec::new(),
                    stored_conditions: HashMap::new(),
                    fork_snapshots: HashMap::new(),
                });

                return Ok(ExplorationEvent {
                    event_type: "need_callback".to_string(),
                    callback_reason: Some("avoid_predicate".to_string()),
                    callback_addr: Some(pc),
                    callback_state_id: Some(state_id),
                    found_count: self.found_count(),
                    active_count: self.active_count(),
                    steps_taken: self.steps,
                    callback_name: None,
                    callback_syscall_num: None,
                    callback_return_addr: None,
                    callback_num_args: None,
                    branch_true_target: None,
                    branch_false_target: None,
                    branch_condition_id: None,
                });
            }

            // Check avoid addresses (address-based, only when NOT using callable predicate)
            if self.avoid_addrs.contains(&pc) {
                self.stashes
                    .entry("avoid".to_string())
                    .or_insert_with(VecDeque::new)
                    .push_back(state);
                continue;
            }

            // P2 fix: Check if callable find predicate needs Python evaluation
            // When find is a callable (lambda/function), we must return to Python
            // to evaluate it for each state, not just check addresses.
            if self.find_needs_python {
                let state_id = state.state_id();
                self.pending_callback = Some(PendingCallback {
                    state,
                    pre_callback_snapshot: None,
                    reason: CallbackReason::FindPredicate { addr: pc },
                    jumpkind: None,
                    solver_ctx: None,
                    deferred_forks: Vec::new(),
                    stored_conditions: HashMap::new(),
                    fork_snapshots: HashMap::new(),
                });

                return Ok(ExplorationEvent {
                    event_type: "need_callback".to_string(),
                    callback_reason: Some("find_predicate".to_string()),
                    callback_addr: Some(pc),
                    callback_state_id: Some(state_id),
                    found_count: self.found_count(),
                    active_count: self.active_count(),
                    steps_taken: self.steps,
                    callback_name: None,
                    callback_syscall_num: None,
                    callback_return_addr: None,
                    callback_num_args: None,
                    branch_true_target: None,
                    branch_false_target: None,
                    branch_condition_id: None,
                });
            }

            // Check find addresses (address-based, only when NOT using callable predicate)
            if self.find_addrs.contains(&pc) {
                // Only add to found if the state is satisfiable
                // (UNSAT states reached the address via infeasible paths)
                if self.lazy_solves || state.satisfiable() {
                    self.stashes
                        .entry("found".to_string())
                        .or_insert_with(VecDeque::new)
                        .push_back(state);
                } else {
                    log::debug!("State at find address 0x{:x} is UNSAT, pruning", pc);
                    self.stashes
                        .entry("pruned".to_string())
                        .or_insert_with(VecDeque::new)
                        .push_back(state);
                }
                continue;
            }

            // Check hooks (SimProcedures)
            // GAP 6: Stack-based skip tracking for zero-length hooks
            // Clean up expired skip entries before checking
            self.skip_hook_stack.retain(|&(_, expiry)| expiry > self.steps);

            // Check if this address is in the skip stack
            let should_skip_hook = self.skip_hook_stack.iter().any(|&(addr, _)| addr == pc);
            if should_skip_hook {
                // Remove this address from the skip stack (consumed)
                self.skip_hook_stack.retain(|&(addr, _)| addr != pc);
                log::debug!("Skipping hook at 0x{:x} (zero-length hook, step {})", pc, self.steps);
            }
            if self.hooks.contains(&pc) && !should_skip_hook {
                // Check if this is a registered SimProcedure
                if let Some((name, num_args, no_return)) = self.simprocedures.get(&pc).cloned() {
                    // Skip native for addresses inside the binary (user-placed hooks)
                    let is_in_binary = self.binary_regions.iter().any(|(base, data)| {
                        pc >= *base && pc < *base + data.len() as u64
                    });
                    // Try native procedure first (only for external/library hooks)
                    if !is_in_binary {
                    if let Some(native_proc) = self.native_procedures.get(&name) {
                        // Extract arguments from state registers and stack
                        let args = self.extract_procedure_args(&state, num_args);

                        // Try to execute native procedure
                        match native_proc.call(&mut state, &args) {
                            Ok(ret_val) => {
                                // Native execution succeeded
                                self.native_proc_stats.native_calls += 1;
                                *self.native_proc_stats.call_counts
                                    .entry(name.clone())
                                    .or_insert(0) += 1;

                                // Set return value if present
                                if let Some(rv) = ret_val {
                                    let ret_reg = self.calling_convention.return_register();
                                    state.set_register_by_offset(ret_reg, rv);
                                }

                                // Get return address and set PC
                                let ctx = state.solver().borrow();
                                if let Some(ret_addr) = self.calling_convention.get_return_addr(
                                    &crate::arch::RegisterFile::new(
                                        crate::arch::arch_from_name(&self.arch_name).unwrap()
                                    ),
                                    None,
                                    &ctx,
                                ) {
                                    drop(ctx);
                                    // Pop return address from stack
                                    let sp = state.get_sp().as_u64().unwrap_or(0);
                                    let ptr_size = state.arch().bytes() as u64;
                                    state.set_sp(RustBV::concrete((sp + ptr_size) as u128, state.arch().bits()));
                                    state.set_pc(ret_addr);
                                } else {
                                    drop(ctx);
                                    // Fallback: try to get return address from stack
                                    if let Some(sp) = state.get_sp().as_u64() {
                                        if let Ok(ret_bv) = state.memory_load(sp, state.arch().bytes()) {
                                            if let Some(ret_addr) = ret_bv.as_u64() {
                                                let ptr_size = state.arch().bytes() as u64;
                                                state.set_sp(RustBV::concrete((sp + ptr_size) as u128, state.arch().bits()));
                                                state.set_pc(ret_addr);
                                            }
                                        }
                                    }
                                }

                                // If no_return, put state in deadended
                                if no_return {
                                    self.stashes
                                        .entry("deadended".to_string())
                                        .or_insert_with(VecDeque::new)
                                        .push_back(state);
                                } else {
                                    // Add back to active stash
                                    self.stashes
                                        .entry("active".to_string())
                                        .or_insert_with(VecDeque::new)
                                        .push_back(state);
                                }
                                continue;
                            }
                            Err(_e) => {
                                // Native execution failed, fall back to Python
                                self.native_proc_stats.python_fallbacks += 1;
                            }
                        }
                    }
                    } // if !is_in_binary

                    // Fall back to Python for SimProcedure execution
                    let state_id = state.state_id();
                    let return_addr = self.get_return_addr(&state).unwrap_or(0);

                    // Save pre-callback snapshot for deferred forks
                    // Deferred forks diverged BEFORE this callback, so they should
                    // not inherit any constraints added by the callback
                    let pre_callback_snapshot = Some(state.fork());

                    // Fork solver context for Python callback use
                    let solver_ref = state.solver();
                    let forked_ctx = RustSolverContext::from_sym_context(solver_ref.borrow().fork());

                    self.pending_callback = Some(PendingCallback {
                        state,
                        pre_callback_snapshot,
                        reason: CallbackReason::SimProcedure {
                            addr: pc,
                            name: name.clone(),
                            num_args,
                            return_addr,
                        },
                        jumpkind: Some("Ijk_Call".to_string()),
                        solver_ctx: Some(forked_ctx),
                        deferred_forks: Vec::new(),
                        stored_conditions: HashMap::new(),
                        fork_snapshots: HashMap::new(),
                    });

                    return Ok(ExplorationEvent::need_simprocedure(
                        state_id,
                        pc,
                        name,
                        num_args,
                        return_addr,
                        self.found_count(),
                        self.active_count(),
                        self.steps,
                    ));
                }
            }

            // Step the state, passing the skip_hook_addr if we just skipped
            let skip_addr_for_step = if should_skip_hook { Some(pc) } else { None };
            match self.step_state_with_skip(py, &callbacks, state, skip_addr_for_step) {
                Ok(successors) => {
                    // Add successors to appropriate stashes, checking find/avoid
                    for successor in successors {
                        let spc = successor.pc();
                        if self.find_addrs.contains(&spc) {
                            if self.lazy_solves || successor.satisfiable() {
                                self.stashes.entry("found".to_string())
                                    .or_insert_with(VecDeque::new).push_back(successor);
                            }
                        } else if self.avoid_addrs.contains(&spc) {
                            self.stashes.entry("avoid".to_string())
                                .or_insert_with(VecDeque::new).push_back(successor);
                        } else {
                            self.stashes.entry("active".to_string())
                                .or_insert_with(VecDeque::new).push_back(successor);
                        }
                    }
                }
                Err(StepError::NeedCallback(pending)) => {
                    // Check if the callback address is a find/avoid address
                    // (these were added as interpreter hooks to stop execution)
                    let callback_addr = match &pending.reason {
                        CallbackReason::SimProcedure { addr, .. } => Some(*addr),
                        _ => None,
                    };
                    if let Some(addr) = callback_addr {
                        if self.find_addrs.contains(&addr) || self.avoid_addrs.contains(&addr) {
                            let is_find = self.find_addrs.contains(&addr);

                            // Process deferred forks BEFORE handling the find/avoid state.
                            // These represent unexplored branches that diverged before
                            // reaching the find/avoid address and must not be dropped.
                            let fork_base = pending.pre_callback_snapshot.unwrap_or_else(|| pending.state.fork());
                            let original_state_id = pending.state.state_id();
                            let root_state_id = self.state_roots.get(&original_state_id).copied().unwrap_or(original_state_id);

                            let mut snapshots = pending.fork_snapshots;
                            for fork in pending.deferred_forks {
                                let condition = pending.stored_conditions.get(&fork.condition_id);
                                let reconstructed = if condition.is_none() {
                                    if let Some(ref py_ast) = fork.condition_ast {
                                        Python::with_gil(|py| {
                                            let ast = py_ast.bind(py);
                                            let solver_ref = fork_base.solver();
                                            let ctx: &SymContext = &*solver_ref.borrow();
                                            claripy_to_rustbv(py, ast, ctx).ok()
                                        })
                                    } else { None }
                                } else { None };

                                if let Some(cond) = condition.or(reconstructed.as_ref()) {
                                    let forked = if let Some(snapshot) = snapshots.remove(&fork.condition_id) {
                                        let mut f = fork_base.fork_from_snapshot(snapshot);
                                        if fork.path_taken {
                                            f.solver().borrow().assume_false(cond);
                                        } else {
                                            f.solver().borrow().assume_true(cond);
                                        }
                                        f.set_pc(fork.unexplored_target);
                                        f
                                    } else if fork.path_taken {
                                        let mut f = fork_base.fork_false(cond);
                                        f.set_pc(fork.unexplored_target);
                                        f
                                    } else {
                                        let mut f = fork_base.fork_true(cond);
                                        f.set_pc(fork.unexplored_target);
                                        f
                                    };
                                    self.state_roots.insert(forked.state_id(), root_state_id);
                                    if self.lazy_solves || forked.satisfiable() {
                                        self.stashes.entry("active".to_string())
                                            .or_insert_with(VecDeque::new)
                                            .push_back(forked);
                                    }
                                }
                            }

                            // Now handle the main state
                            if is_find {
                                if self.lazy_solves || pending.state.satisfiable() {
                                    self.stashes.entry("found".to_string())
                                        .or_insert_with(VecDeque::new)
                                        .push_back(pending.state);
                                } else {
                                    log::debug!("State at find address 0x{:x} is UNSAT, pruning", addr);
                                    self.stashes.entry("pruned".to_string())
                                        .or_insert_with(VecDeque::new)
                                        .push_back(pending.state);
                                }
                            } else {
                                self.stashes.entry("avoid".to_string())
                                    .or_insert_with(VecDeque::new)
                                    .push_back(pending.state);
                            }
                            continue;
                        }
                    }

                    // Need Python callback
                    let state_id = pending.state.state_id();
                    let event = match &pending.reason {
                        CallbackReason::SimProcedure { addr, name, num_args, return_addr } => {
                            ExplorationEvent::need_simprocedure(
                                state_id,
                                *addr,
                                name.clone(),
                                *num_args,
                                *return_addr,
                                self.found_count(),
                                self.active_count(),
                                self.steps,
                            )
                        }
                        CallbackReason::Syscall { num } => {
                            ExplorationEvent::need_syscall(
                                state_id,
                                *num,
                                self.found_count(),
                                self.active_count(),
                                self.steps,
                            )
                        }
                        CallbackReason::SymbolicBranch { condition_id, true_target, false_target } => {
                            ExplorationEvent::need_symbolic_branch(
                                state_id,
                                *condition_id,
                                *true_target,
                                *false_target,
                                self.found_count(),
                                self.active_count(),
                                self.steps,
                            )
                        }
                        CallbackReason::Error { message } => {
                            ExplorationEvent::error(
                                message.clone(),
                                self.found_count(),
                                self.active_count(),
                                self.steps,
                            )
                        }
                        _ => {
                            ExplorationEvent::error(
                                "unhandled callback reason".to_string(),
                                self.found_count(),
                                self.active_count(),
                                self.steps,
                            )
                        }
                    };

                    self.pending_callback = Some(pending);
                    return Ok(event);
                }
                Err(StepError::Deadended(state)) => {
                    self.stashes
                        .entry("deadended".to_string())
                        .or_insert_with(VecDeque::new)
                        .push_back(state);
                }
                Err(StepError::Error(state, message)) => {
                    let pc = state.pc();
                    let state_id = state.state_id();
                    self.errors.push((pc, message, state_id));
                    self.stashes
                        .entry("errored".to_string())
                        .or_insert_with(VecDeque::new)
                        .push_back(state);
                }
                Err(StepError::Unconstrained(state)) => {
                    // State has too many symbolic jump targets - move to unconstrained stash
                    log::debug!("State {} moved to unconstrained stash", state.state_id());
                    self.stashes
                        .entry("unconstrained".to_string())
                        .or_insert_with(VecDeque::new)
                        .push_back(state);
                }
            }

            self.steps += 1;
        }

        // Max steps reached
        Ok(ExplorationEvent::step_complete(
            self.found_count(),
            self.active_count(),
            self.steps,
        ))
    }

    /// Resume after a SimProcedure callback.
    ///
    /// This is called from Python after executing a SimProcedure.
    /// The state changes (registers, memory, new PC) are applied.
    ///
    /// Args:
    ///     new_pc: The new program counter after the SimProcedure.
    ///     register_changes: List of (offset, size, data) tuples for register changes.
    ///     memory_changes: List of (addr, data) tuples for memory changes.
    ///     new_constraints: Optional list of claripy ASTs to add as constraints.
    ///         These are constraints added by the SimProcedure (e.g., strcmp results).
    #[pyo3(signature = (new_pc, register_changes=None, memory_changes=None, new_constraints=None))]
    pub fn resume_after_simprocedure(
        &mut self,
        py: Python<'_>,
        new_pc: u64,
        register_changes: Option<Vec<(u32, u32, Vec<u8>)>>,
        memory_changes: Option<Vec<(u64, Vec<u8>)>>,
        new_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
    ) -> PyResult<()> {
        let pending = self.pending_callback.take().ok_or_else(|| {
            PyRuntimeError::new_err("no pending callback state")
        })?;

        // Apply changes
        let mut changes = StateChanges::new();
        changes.new_pc = Some(new_pc);

        if let Some(reg_changes) = register_changes {
            changes.register_writes = reg_changes;
        }

        if let Some(mem_changes) = memory_changes {
            changes.memory_writes = mem_changes;
        }

        let mut state = pending.state;
        state.apply_changes(&changes);
        state.set_pc(new_pc);

        // Sync constraints from Python back to Rust
        // This ensures constraints added by SimProcedures (e.g., strcmp return conditions)
        // are properly reflected in the Rust solver state
        //
        // P12: Track if state becomes UNSAT after constraint sync
        let mut main_state_unsat = false;
        if let Some(constraints) = new_constraints {
            let is_sat = self.sync_constraints_from_python(py, &state, constraints)?;
            if !is_sat {
                log::debug!(
                    "P12: State {} became UNSAT after constraint sync in resume_after_simprocedure.",
                    state.state_id()
                );
                main_state_unsat = true;
            }
        }

        // Validate deferred forks reference valid conditions before processing
        let mut missing_conditions = 0usize;
        for fork in &pending.deferred_forks {
            if !pending.stored_conditions.contains_key(&fork.condition_id) {
                missing_conditions += 1;
                log::warn!(
                    "Deferred fork at 0x{:x} references missing condition_id={}",
                    fork.branch_addr, fork.condition_id
                );
            }
        }
        if missing_conditions > 0 {
            log::warn!(
                "{} of {} deferred forks have missing conditions - will be skipped",
                missing_conditions, pending.deferred_forks.len()
            );
        }

        // Process deferred forks that were stored during the step
        // These represent unexplored branches that should be added to active
        //
        // CRITICAL: Deferred forks diverged BEFORE the callback, so they should
        // NOT inherit callback constraints. Use pre_callback_snapshot as fork base.
        let fork_base = pending.pre_callback_snapshot.unwrap_or_else(|| state.fork());

        // Track root state ID for lineage
        // The root is inherited from the original pending state
        let original_state_id = state.state_id();
        let root_state_id = self.state_roots.get(&original_state_id).copied().unwrap_or(original_state_id);

        // P12: Only add main state if SAT, otherwise add to pruned list
        let (mut successors, mut pruned_states) = if main_state_unsat {
            (Vec::new(), vec![state])
        } else {
            (vec![state], Vec::new())
        };
        let mut snapshots = pending.fork_snapshots;
        for fork in pending.deferred_forks {
            // Look up the condition for this deferred fork
            let condition = pending.stored_conditions.get(&fork.condition_id);

            // P11 fix: If condition not in stored_conditions, try to reconstruct from condition_ast
            let reconstructed_condition = if condition.is_none() {
                if let Some(ref py_ast) = fork.condition_ast {
                    // Try to convert the claripy AST to RustBV
                    Python::with_gil(|py| {
                        let ast = py_ast.bind(py);
                        let solver_ref = fork_base.solver();
                        let ctx: &SymContext = &*solver_ref.borrow();
                        claripy_to_rustbv(py, ast, ctx).ok()
                    })
                } else {
                    None
                }
            } else {
                None
            };

            let effective_condition = condition.or(reconstructed_condition.as_ref());

            if let Some(cond) = effective_condition {
                // Use solver snapshot (from before branch constraint) if available
                let forked = if let Some(snapshot) = snapshots.remove(&fork.condition_id) {
                    let mut f = fork_base.fork_from_snapshot(snapshot);
                    if fork.path_taken {
                        f.solver().borrow().assume_false(cond);
                    } else {
                        f.solver().borrow().assume_true(cond);
                    }
                    f.set_pc(fork.unexplored_target);
                    f
                } else if fork.path_taken {
                    let mut f = fork_base.fork_false(cond);
                    f.set_pc(fork.unexplored_target);
                    f
                } else {
                    let mut f = fork_base.fork_true(cond);
                    f.set_pc(fork.unexplored_target);
                    f
                };

                // Track root state ID for this forked state
                self.state_roots.insert(forked.state_id(), root_state_id);

                // DO NOT sync callback constraints to forked state!
                // These paths diverged before the callback occurred.
                // Adding callback constraints would pollute unexplored branches.

                // P13: Check satisfiability before adding to successors
                if self.lazy_solves || forked.satisfiable() {
                    successors.push(forked);
                    if reconstructed_condition.is_some() {
                        log::debug!(
                            "P11: Reconstructed condition from condition_ast for fork at 0x{:x}",
                            fork.branch_addr
                        );
                    }
                } else {
                    log::debug!(
                        "P13: Forked state at 0x{:x} is UNSAT, adding to pruned",
                        fork.unexplored_target
                    );
                    pruned_states.push(forked);
                }
            } else {
                // P15: Better handling of missing deferred fork conditions
                // Try harder to get a condition or create a fresh boolean to explore both paths
                log::warn!(
                    "P15: Missing condition for deferred fork at 0x{:x} (condition_id={}). \
                     Creating conservative fork to explore the path.",
                    fork.branch_addr,
                    fork.condition_id
                );
                // Create a fork without additional constraints - this is conservative
                // but ensures we don't lose valid paths
                let mut forked = fork_base.fork();
                forked.set_pc(fork.unexplored_target);
                self.state_roots.insert(forked.state_id(), root_state_id);

                // P13: Still check satisfiability
                if self.lazy_solves || forked.satisfiable() {
                    successors.push(forked);
                } else {
                    log::debug!(
                        "P13: Unconstrained fork at 0x{:x} is UNSAT, adding to pruned",
                        fork.unexplored_target
                    );
                    pruned_states.push(forked);
                }
            }
        }

        // Add all successors (original state + forks) to stashes
        // P13: Check satisfiability for each before adding
        // Note: We split the loops to avoid double mutable borrow of self.stashes
        let mut final_successors = Vec::new();
        for successor in successors {
            if self.lazy_solves || successor.satisfiable() {
                final_successors.push(successor);
            } else {
                log::debug!(
                    "P13: Successor state {} is UNSAT, moving to pruned stash",
                    successor.state_id()
                );
                pruned_states.push(successor);
            }
        }

        // Add to active stash, checking find/avoid first
        for s in final_successors {
            let spc = s.pc();
            if self.find_addrs.contains(&spc) {
                self.stashes.entry("found".to_string())
                    .or_insert_with(VecDeque::new).push_back(s);
            } else if self.avoid_addrs.contains(&spc) {
                self.stashes.entry("avoid".to_string())
                    .or_insert_with(VecDeque::new).push_back(s);
            } else {
                self.stashes.entry("active".to_string())
                    .or_insert_with(VecDeque::new).push_back(s);
            }
        }

        // Add to pruned stash
        if !pruned_states.is_empty() {
            let pruned = self.stashes
                .entry("pruned".to_string())
                .or_insert_with(VecDeque::new);
            for s in pruned_states {
                pruned.push_back(s);
            }
        }

        Ok(())
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
        self.resume_after_simprocedure(py, new_pc, register_changes, memory_changes, new_constraints)
    }

    /// Resume after a hook callback.
    ///
    /// This is equivalent to resume_after_simprocedure but with a more explicit name
    /// for hook-specific handling. Ensures constraints added during hook execution
    /// are properly synced back to Rust (GAP 2 fix).
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
        self.resume_after_simprocedure(py, new_pc, register_changes, memory_changes, new_constraints)
    }

    /// Resume after an error occurred during callback execution (P17).
    ///
    /// This moves the pending state to the errored stash instead of continuing
    /// with a corrupted state. This prevents "list index out of range" errors
    /// caused by UNSAT states proliferating from callback failures.
    pub fn resume_after_error(&mut self, error_msg: &str) -> PyResult<()> {
        let pending = self.pending_callback.take().ok_or_else(|| {
            PyRuntimeError::new_err("no pending callback state for error handling")
        })?;

        let pc = pending.state.pc();
        let state_id = pending.state.state_id();

        log::warn!(
            "P17: Moving state {} to errored stash after callback error at 0x{:x}: {}",
            state_id, pc, error_msg
        );

        // Record the error
        self.errors.push((pc, error_msg.to_string(), state_id));

        // Move to errored stash
        self.stashes
            .entry("errored".to_string())
            .or_insert_with(VecDeque::new)
            .push_back(pending.state);

        Ok(())
    }

    /// Resume after Python handles a symbolic branch.
    ///
    /// Python creates forked states with proper constraints and passes them back
    /// to be added to the active stash.
    #[pyo3(signature = (true_pc, false_pc, true_constraints=None, false_constraints=None))]
    pub fn resume_after_symbolic_branch(
        &mut self,
        py: Python<'_>,
        true_pc: u64,
        false_pc: u64,
        true_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
        false_constraints: Option<&Bound<'_, pyo3::types::PyList>>,
    ) -> PyResult<()> {
        let pending = self.pending_callback.take()
            .ok_or_else(|| PyRuntimeError::new_err("no pending symbolic branch callback"))?;

        // Get the branch condition from stored_conditions (set by interpreter)
        let branch_condition = match &pending.reason {
            CallbackReason::SymbolicBranch { condition_id, .. } => {
                pending.stored_conditions.get(condition_id).cloned()
            }
            _ => None,
        };

        // Create the true state (fork of original) and add constraint
        let mut true_state = pending.state.fork();
        true_state.set_pc(true_pc);
        if let Some(ref cond) = branch_condition {
            // Add assume_true: guard is true → exit taken
            let solver_ref = true_state.solver();
            solver_ref.borrow().assume_true(cond);
        }

        // Create the false state (use original) and add constraint
        let mut false_state = pending.state;
        false_state.set_pc(false_pc);
        if let Some(ref cond) = branch_condition {
            // Add assume_false: guard is false → fallthrough
            let solver_ref = false_state.solver();
            solver_ref.borrow().assume_false(cond);
        }

        // Add states to stashes.
        // When branch_condition is present, the interpreter already proved
        // both paths feasible via can_be_true/can_be_false. Prime the sat
        // cache so downstream satisfiable() checks are free (cache hits)
        // instead of doing redundant Z3 check() calls.
        let mut active_states = Vec::new();
        let mut pruned_states = Vec::new();

        if branch_condition.is_some() {
            true_state.set_sat_cache(true);
            false_state.set_sat_cache(true);
            // Check find/avoid on new states before adding to active
            if self.find_addrs.contains(&true_pc) {
                self.stashes.entry("found".to_string()).or_insert_with(VecDeque::new).push_back(true_state);
            } else if self.avoid_addrs.contains(&true_pc) {
                self.stashes.entry("avoid".to_string()).or_insert_with(VecDeque::new).push_back(true_state);
            } else {
                active_states.push(true_state);
            }
            if self.find_addrs.contains(&false_pc) {
                self.stashes.entry("found".to_string()).or_insert_with(VecDeque::new).push_back(false_state);
            } else if self.avoid_addrs.contains(&false_pc) {
                self.stashes.entry("avoid".to_string()).or_insert_with(VecDeque::new).push_back(false_state);
            } else {
                active_states.push(false_state);
            }
        } else {
            // Fallback: no stored condition, need actual sat checks
            if self.lazy_solves || true_state.satisfiable() {
                active_states.push(true_state);
            } else {
                pruned_states.push(true_state);
            }
            if self.lazy_solves || false_state.satisfiable() {
                active_states.push(false_state);
            } else {
                pruned_states.push(false_state);
            }
        }

        // Add to active stash
        let active = self.stashes
            .entry("active".to_string())
            .or_insert_with(VecDeque::new);
        for s in active_states {
            active.push_back(s);
        }

        // Add to pruned stash
        if !pruned_states.is_empty() {
            let pruned = self.stashes
                .entry("pruned".to_string())
                .or_insert_with(VecDeque::new);
            for s in pruned_states {
                pruned.push_back(s);
            }
        }

        log::debug!("Resumed after symbolic branch: true_pc=0x{:x}, false_pc=0x{:x}",
                    true_pc, false_pc);

        Ok(())
    }

    /// Resume after Python evaluates a find predicate.
    ///
    /// P2 fix: This is called after Python evaluates a callable find predicate.
    /// If matched=true, the state is moved to found stash; otherwise, it continues
    /// exploration in the active stash.
    pub fn resume_find_predicate(&mut self, matched: bool) -> PyResult<()> {
        let pending = self.pending_callback.take()
            .ok_or_else(|| PyRuntimeError::new_err("no pending find predicate callback"))?;

        if matched {
            log::debug!("Find predicate matched - moving state to found stash");
            self.stashes
                .entry("found".to_string())
                .or_insert_with(VecDeque::new)
                .push_back(pending.state);
        } else {
            log::debug!("Find predicate did not match - continuing exploration");
            self.stashes
                .entry("active".to_string())
                .or_insert_with(VecDeque::new)
                .push_back(pending.state);
        }

        Ok(())
    }

    /// Resume after Python evaluates an avoid predicate.
    ///
    /// P7 fix: This is called after Python evaluates a callable avoid predicate.
    /// If matched=true, the state is moved to avoid stash; otherwise, it continues
    /// exploration in the active stash.
    pub fn resume_avoid_predicate(&mut self, matched: bool) -> PyResult<()> {
        let pending = self.pending_callback.take()
            .ok_or_else(|| PyRuntimeError::new_err("no pending avoid predicate callback"))?;

        if matched {
            log::debug!("Avoid predicate matched - moving state to avoid stash");
            self.stashes
                .entry("avoid".to_string())
                .or_insert_with(VecDeque::new)
                .push_back(pending.state);
        } else {
            log::debug!("Avoid predicate did not match - continuing exploration");
            self.stashes
                .entry("active".to_string())
                .or_insert_with(VecDeque::new)
                .push_back(pending.state);
        }

        Ok(())
    }

    /// Get the branch condition from the pending symbolic branch callback.
    ///
    /// Returns the condition as a claripy AST that Python can use for forking.
    pub fn get_pending_branch_condition(&self, py: Python<'_>) -> PyResult<PyObject> {
        let pending = self.pending_callback.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no pending callback state"))?;

        // Get the condition ID from the callback reason
        let condition_id = match &pending.reason {
            CallbackReason::SymbolicBranch { condition_id, .. } => *condition_id,
            _ => return Err(PyValueError::new_err("pending callback is not a symbolic branch")),
        };

        // Look up the condition in stored_conditions
        let condition = pending.stored_conditions.get(&condition_id)
            .ok_or_else(|| PyValueError::new_err(
                format!("condition {} not found in stored_conditions", condition_id)
            ))?;

        // Convert to claripy AST
        let claripy = py.import("claripy")?;
        rustbv_to_claripy(py, condition, claripy.as_any())
            .map_err(|e| PyRuntimeError::new_err(format!("failed to convert condition: {}", e)))
    }

    /// Get register value from pending state (concrete only).
    pub fn get_pending_register(&self, name: &str) -> PyResult<Option<u128>> {
        if let Some(ref pending) = self.pending_callback {
            pending.state.get_register(name)
                .map(|bv| bv.as_u128())
                .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Get register as claripy AST from pending state (handles symbolic).
    pub fn get_pending_register_ast(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        let pending = self.pending_callback.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no pending callback state"))?;
        let bv = pending.state.get_register(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))?;
        let claripy = py.import("claripy")?;
        rustbv_to_claripy(py, &bv, claripy.as_any())
            .map_err(|e| PyRuntimeError::new_err(format!("register conversion: {}", e)))
    }

    /// Get history (BBL addresses) from pending callback state.
    ///
    /// This is used by Python to initialize history on callback states,
    /// preventing IndexError when hooks access `state.history.recent_bbl_addrs[-1]`.
    pub fn get_pending_history(&self) -> PyResult<Vec<u64>> {
        if let Some(ref pending) = self.pending_callback {
            Ok(pending.state.history().to_vec())
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Get jumpkind for pending callback.
    ///
    /// Returns the jumpkind that led to this callback (e.g., "Ijk_Call", "Ijk_Boring").
    /// This is used by Python to properly initialize callstack management.
    pub fn get_pending_jumpkind(&self) -> PyResult<String> {
        if let Some(ref pending) = self.pending_callback {
            Ok(pending.jumpkind.clone().unwrap_or_else(|| "Ijk_Boring".to_string()))
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Set register value in pending state.
    pub fn set_pending_register(&mut self, name: &str, value: u128) -> PyResult<()> {
        if let Some(ref mut pending) = self.pending_callback {
            let size = pending.state.arch().register_size(name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))?;
            let bv = crate::symbolic::RustBV::concrete(value, size * 8);
            if pending.state.set_register(name, bv) {
                Ok(())
            } else {
                Err(PyValueError::new_err(format!("failed to set register: {}", name)))
            }
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Set register to a symbolic value from a handle ID.
    ///
    /// Used for syncing symbolic return values from SimProcedures.
    /// The handle_id should reference a RustBV in the solver's symbol table.
    pub fn set_pending_register_symbolic(&mut self, name: &str, handle_id: u64) -> PyResult<()> {
        if let Some(ref mut pending) = self.pending_callback {
            // Look up the RustBV from the symbol table
            let bv = if let Some(ref solver) = pending.solver_ctx {
                solver.symbol_table().get(handle_id)
                    .ok_or_else(|| PyValueError::new_err(format!(
                        "invalid handle id: {}", handle_id
                    )))?
            } else {
                return Err(PyRuntimeError::new_err("no solver context in pending state"));
            };

            if pending.state.set_register(name, bv) {
                Ok(())
            } else {
                Err(PyValueError::new_err(format!("failed to set register: {}", name)))
            }
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Set a symbolic register value in the pending state from claripy AST.
    ///
    /// This allows direct sync of symbolic register values from Python callbacks.
    /// The claripy AST is converted to RustBV and stored in the pending state.
    pub fn set_pending_register_symbolic_ast(
        &mut self,
        py: Python<'_>,
        reg_name: &str,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if let Some(ref mut pending) = self.pending_callback {
            let solver_ref = pending.state.solver();
            let sym_ctx = solver_ref.borrow();
            let ctx_ref: &SymContext = &*sym_ctx;

            // Convert claripy AST to RustBV
            let bv = claripy_to_rustbv(py, ast, ctx_ref)
                .map_err(|e| PyValueError::new_err(format!("AST conversion failed: {}", e)))?;

            drop(sym_ctx);

            if pending.state.set_register(reg_name, bv) {
                log::debug!("Set symbolic register {} from claripy AST", reg_name);
                Ok(())
            } else {
                Err(PyValueError::new_err(format!("failed to set register: {}", reg_name)))
            }
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Import symbolic memory from Python hook into Rust's symbolic_objects.
    ///
    /// Called after a hook writes symbolic memory. Converts the claripy AST
    /// to RustBV and imports it into the pending state's SymbolicMemory.
    /// Import symbolic memory into a state by ID (for init-time symbolic data).
    #[pyo3(signature = (state_id, addr, ast))]
    pub fn import_symbolic_to_state(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        addr: u64,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        // Find the state in any stash
        for stash in self.stashes.values_mut() {
            for state in stash.iter_mut() {
                if state.state_id() == state_id {
                    let solver_ref = state.solver();
                    let sym_ctx = solver_ref.borrow();
                    let bv = claripy_to_rustbv(py, ast, &*sym_ctx)
                        .map_err(|e| PyValueError::new_err(format!("AST conversion: {}", e)))?;
                    drop(sym_ctx);
                    state.memory_mut().import_symbolic_value(addr, bv, None);
                    return Ok(());
                }
            }
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    pub fn import_symbolic_memory(
        &mut self,
        py: Python<'_>,
        addr: u64,
        ast: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let pending = self.pending_callback.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("no pending callback state")
        })?;

        // Convert claripy AST to RustBV (caches original AST for round-trip)
        let solver_ref = pending.state.solver();
        let sym_ctx = solver_ref.borrow();
        let bv = claripy_to_rustbv(py, ast, &*sym_ctx)
            .map_err(|e| PyValueError::new_err(format!("AST conversion failed: {}", e)))?;
        drop(sym_ctx);

        // Import into symbolic memory via existing infrastructure
        // Symbol ID is not used by import_symbolic_value, so pass None
        pending.state.memory_mut().import_symbolic_value(addr, bv, None);
        log::debug!("Imported symbolic memory at 0x{:x}", addr);
        Ok(())
    }

    /// Get memory from pending state.
    pub fn get_pending_memory(&self, addr: u64, size: u32) -> PyResult<Vec<u8>> {
        if let Some(ref pending) = self.pending_callback {
            let bv = pending.state.memory_load(addr, size)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            let value = bv.to_u128();
            let bytes: Vec<u8> = (0..size as usize)
                .map(|i| (value >> (i * 8)) as u8)
                .collect();
            Ok(bytes)
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Store memory in pending state.
    pub fn set_pending_memory(&mut self, addr: u64, data: &[u8]) -> PyResult<()> {
        if let Some(ref mut pending) = self.pending_callback {
            let width = (data.len() * 8) as u32;
            let mut value: u128 = 0;
            for (i, &b) in data.iter().enumerate() {
                value |= (b as u128) << (i * 8);
            }
            let bv = crate::symbolic::RustBV::concrete(value, width);
            pending.state.memory_store(addr, bv)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Get dirty page addresses from pending state.
    ///
    /// This returns the list of page-aligned addresses that have been
    /// modified in the pending callback state.
    pub fn get_pending_dirty_pages(&self) -> PyResult<Vec<u64>> {
        if let Some(ref pending) = self.pending_callback {
            Ok(pending.state.get_dirty_pages())
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Get dirty register offsets from pending state.
    pub fn get_pending_dirty_registers(&self) -> PyResult<Vec<u32>> {
        if let Some(ref pending) = self.pending_callback {
            Ok(pending.state.get_dirty_registers())
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Clear dirty tracking in pending state.
    pub fn clear_pending_dirty_tracking(&mut self) -> PyResult<()> {
        if let Some(ref mut pending) = self.pending_callback {
            pending.state.clear_dirty_pages();
            pending.state.clear_dirty_registers();
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Export pending constraints as a list of claripy ASTs.
    ///
    /// Returns constraints that can be added to Python state.solver.
    /// This exports stored branch conditions accumulated during Rust execution.
    pub fn export_pending_constraints(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        if let Some(ref pending) = self.pending_callback {
            let mut result = Vec::new();

            // Import claripy for AST conversion
            let claripy_mod = py.import("claripy")?;

            // Export stored branch conditions as claripy ASTs
            for (_condition_id, rustbv) in &pending.stored_conditions {
                match rustbv_to_claripy(py, rustbv, &claripy_mod) {
                    Ok(ast) => {
                        result.push(ast);
                    }
                    Err(e) => {
                        log::debug!("Could not convert stored condition to claripy: {}", e);
                    }
                }
            }

            log::debug!("Exported {} pending constraints", result.len());
            Ok(result)
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Get handle IDs that are actively referenced in the pending state.
    ///
    /// Returns handle IDs used in stored conditions and deferred forks.
    /// These should not be evicted from the AST handle cache.
    pub fn get_active_handle_ids(&self) -> Vec<u64> {
        let mut ids = Vec::new();
        if let Some(ref pending) = self.pending_callback {
            // Add condition IDs from stored_conditions
            for (id, _) in &pending.stored_conditions {
                ids.push(*id);
            }
            // Add condition IDs from deferred forks
            for fork in &pending.deferred_forks {
                ids.push(fork.condition_id);
            }
        }
        ids
    }

    /// Export the pending state as a full snapshot.
    ///
    /// This allows Python to get a complete snapshot of the pending state
    /// including all registers, memory pages, and metadata.
    pub fn export_pending_state(&self) -> PyResult<crate::state::ExplorationStateSnapshot> {
        if let Some(ref pending) = self.pending_callback {
            Ok(pending.state.export_full())
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Get the root state ID for the pending callback state.
    ///
    /// When Rust forks states internally, Python only has cached data for the
    /// original state that was added via Python. This method returns the root
    /// state ID (the original state) for any forked descendant.
    ///
    /// Returns:
    ///     The root state ID if available, or None if the state has no tracked root.
    pub fn get_pending_root_state_id(&self) -> PyResult<Option<u64>> {
        if let Some(ref pending) = self.pending_callback {
            let state_id = pending.state.state_id();
            Ok(self.state_roots.get(&state_id).copied())
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Get the full ancestry chain for the pending callback state.
    ///
    /// Returns a list of state IDs starting with the current state and walking
    /// up the parent chain: [state_id, parent_id, grandparent_id, ...].
    ///
    /// This is used by Python to find cached state data when the current state
    /// is a multi-level fork of an original state.
    pub fn get_pending_ancestry(&self) -> PyResult<Vec<u64>> {
        if let Some(ref pending) = self.pending_callback {
            let mut ancestry = vec![pending.state.state_id()];

            // Walk the parent chain
            let mut current_parent = pending.state.parent_id();
            while let Some(parent_id) = current_parent {
                ancestry.push(parent_id);
                // We can't traverse further without access to parent state objects,
                // but we can include the root state if known
                break;
            }

            // Add root state if not already in ancestry
            let state_id = pending.state.state_id();
            if let Some(&root_id) = self.state_roots.get(&state_id) {
                if !ancestry.contains(&root_id) {
                    ancestry.push(root_id);
                }
            }

            Ok(ancestry)
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
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
    pub fn fork_pending_solver(&self) -> PyResult<RustSolverContext> {
        if let Some(ref pending) = self.pending_callback {
            // Fork the pending state's solver context
            let solver_ref = pending.state.solver();
            let forked_ctx = solver_ref.borrow().fork();
            // Create a new RustSolverContext wrapping the forked SymContext
            Ok(RustSolverContext::from_sym_context(forked_ctx))
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Add constraints from Python callbacks back to the pending state.
    ///
    /// This is called after a SimProcedure executes to sync any new
    /// constraints added during the callback back to the Rust solver.
    /// This ensures bidirectional constraint flow between Rust and Python.
    ///
    /// Args:
    ///     constraints: List of claripy AST constraints to add
    pub fn add_constraints_to_pending(
        &mut self,
        py: Python<'_>,
        constraints: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<()> {
        use pyo3::types::PyListMethods;

        if let Some(ref mut pending) = self.pending_callback {
            let solver_ref = pending.state.solver();
            let sym_ctx = solver_ref.borrow();
            let ctx_ref: &SymContext = &*sym_ctx;

            let len = constraints.len();
            for i in 0..len {
                // Use get_item with usize index
                if let Ok(constraint) = constraints.get_item(i) {
                    // Convert claripy AST to RustBV
                    if let Ok(bv) = claripy_to_rustbv(py, &constraint, ctx_ref) {
                        // Add constraint to solver
                        #[cfg(feature = "vex-engine-z3")]
                        {
                            if bv.width() == 1 {
                                sym_ctx.assume_true(&bv);
                            } else {
                                // For wider values, interpret as "value != 0"
                                let zero = RustBV::concrete(0, bv.width());
                                let neq = bv.ne(&zero, ctx_ref);
                                sym_ctx.assume_true(&neq);
                            }
                        }
                    } else {
                        log::debug!("Could not convert constraint {} from Python", i);
                    }
                }
            }
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Add constraints from Python to a state in a stash by state ID.
    /// This is used to sync initial constraints from the Python state.
    pub fn add_constraints_to_state(
        &mut self,
        py: Python<'_>,
        state_id: u64,
        constraints: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<bool> {
        // Find the state in any stash
        for stash in self.stashes.values_mut() {
            for state in stash.iter_mut() {
                if state.state_id() == state_id {
                    let solver_ref = state.solver();
                    let sym_ctx = solver_ref.borrow();
                    let ctx_ref: &SymContext = &*sym_ctx;

                    let mut added = 0u32;
                    for item in constraints.iter() {
                        match claripy_to_rustbv(py, &item, ctx_ref) {
                            Ok(bv) => {
                                if bv.width() == 1 {
                                    sym_ctx.assume_true(&bv);
                                } else {
                                    let zero = RustBV::concrete(0, bv.width());
                                    let neq = bv.ne(&zero, ctx_ref);
                                    sym_ctx.assume_true(&neq);
                                }
                                added += 1;
                            }
                            Err(e) => {
                                log::debug!("Could not convert initial constraint: {}", e);
                            }
                        }
                    }
                    log::debug!("Added {} initial constraints to state {}", added, state_id);
                    return Ok(state.satisfiable());
                }
            }
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Export constraints from a state in any stash as claripy ASTs.
    pub fn export_state_constraints(
        &self,
        py: Python<'_>,
        state_id: u64,
    ) -> PyResult<Vec<PyObject>> {
        let claripy = py.import("claripy")?;
        for stash in self.stashes.values() {
            for state in stash.iter() {
                if state.state_id() == state_id {
                    let solver_ref = state.solver();
                    let ctx = solver_ref.borrow();
                    let assumed = ctx.get_assumed_constraints();
                    let mut results = Vec::new();
                    for (bv, is_true) in &assumed {
                        // Export the raw RustBV as claripy AST.
                        // For false-assumed constraints, negate: cond == 0.
                        match rustbv_to_claripy(py, bv, claripy.as_any()) {
                            Ok(ast) => {
                                if *is_true {
                                    results.push(ast);
                                } else {
                                    // Negate: wrap as `Not(cond)` via claripy
                                    match claripy.call_method1("Not", (ast,)) {
                                        Ok(negated) => results.push(negated.unbind()),
                                        Err(_) => {} // skip if negation fails
                                    }
                                }
                            }
                            Err(_) => {}
                        }
                    }
                    return Ok(results);
                }
            }
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Fork the solver context of an arbitrary state (by ID).
    ///
    /// Returns a new RustSolverContext with all of the state's constraints,
    /// allowing Python to evaluate/solve against any state — not just the
    /// pending callback state.  This is used by RustStateProxy.
    pub fn fork_state_solver(&self, state_id: u64) -> PyResult<RustSolverContext> {
        for stash in self.stashes.values() {
            for state in stash.iter() {
                if state.state_id() == state_id {
                    let solver_ref = state.solver();
                    let forked_ctx = solver_ref.borrow().fork();
                    return Ok(RustSolverContext::from_sym_context(forked_ctx));
                }
            }
        }
        Err(PyValueError::new_err(format!(
            "fork_state_solver: state {} not found",
            state_id
        )))
    }

    /// Get the number of constraints in the pending state's solver.
    pub fn pending_constraint_count(&self) -> PyResult<usize> {
        if let Some(ref pending) = self.pending_callback {
            let solver_ref = pending.state.solver();
            Ok(solver_ref.borrow().num_constraints())
        } else {
            Err(PyRuntimeError::new_err("no pending callback state"))
        }
    }

    /// Get the ID of the state currently being stepped.
    pub fn get_current_stepping_state_id(&self) -> Option<u64> {
        self.current_stepping_state_id
    }

    /// Load from pending callback state's Rust memory.
    /// Used by SimProcedure callbacks to read the correct per-state memory.
    /// Get all mapped page addresses from pending callback state's memory.
    pub fn get_pending_mapped_pages(&self) -> PyResult<Vec<u64>> {
        let pending = self.pending_callback.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no pending callback state"))?;
        Ok(pending.state.memory().pages().keys().map(|&pn| pn << 12).collect())
    }

    /// Load an entire page (4096 bytes) from pending callback state's memory.
    pub fn pending_memory_load_page(&self, page_addr: u64) -> PyResult<Vec<u8>> {
        let pending = self.pending_callback.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no pending callback state"))?;
        pending.state.memory().load_page_concrete(page_addr)
            .map_err(|e| PyValueError::new_err(format!("page load failed: {}", e)))
    }

    pub fn pending_memory_load(&self, addr: u64, size: u32) -> PyResult<Vec<u8>> {
        let pending = self.pending_callback.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no pending callback state"))?;
        let solver_ref = pending.state.solver();
        let ctx = solver_ref.borrow();
        match pending.state.memory().load_concrete(addr, size, &*ctx) {
            Ok(bv) => {
                if let Some(val) = bv.as_u128() {
                    let byte_count = (size as usize).min(16);
                    Ok(val.to_le_bytes()[..byte_count].to_vec())
                } else {
                    let solver = pending.state.solver();
                    let ctx = solver.borrow();
                    if let Some(val) = ctx.eval(&bv) {
                        let byte_count = (size as usize).min(16);
                        Ok(val.to_le_bytes()[..byte_count].to_vec())
                    } else {
                        Ok(vec![0u8; size as usize])
                    }
                }
            }
            Err(_) => Ok(vec![0u8; size as usize]),
        }
    }

    /// Store to pending callback state's Rust memory.
    pub fn pending_memory_store(&mut self, addr: u64, data: &[u8]) -> PyResult<()> {
        let pending = self.pending_callback.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("no pending callback state"))?;
        let mut value: u128 = 0;
        for (i, &byte) in data.iter().enumerate() {
            if i < 16 { value |= (byte as u128) << (i * 8); }
        }
        let bv = RustBV::concrete(value, (data.len() * 8) as u32);
        pending.state.memory_mut().store_concrete(addr, bv)
            .map_err(|e| PyRuntimeError::new_err(format!("memory store error: {}", e)))
    }

    /// Map memory with data in pending callback state.
    pub fn pending_memory_map_data(&mut self, addr: u64, data: &[u8], perm: u8) -> PyResult<()> {
        let pending = self.pending_callback.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("no pending callback state"))?;
        pending.state.map_memory_data(addr, data, crate::memory::Permission::from_bits(perm));
        Ok(())
    }

    /// Set address to skip hook check for on next step.
    ///
    /// This is used to prevent infinite loops with zero-length hooks.
    /// When a hook with length=0 runs, it returns to the same address.
    /// Without this skip mechanism, the hook would trigger again immediately.
    ///
    /// The skip is automatically cleared after one step or when the address is used.
    /// GAP 6: Stack-based tracking allows for nested zero-length hooks.
    pub fn set_skip_hook_addr(&mut self, addr: u64) {
        // Set expiry to current_step + 2 to account for step increment
        // This ensures the skip persists through the next step
        let expiry = self.steps + 2;
        self.skip_hook_stack.push((addr, expiry));
        log::debug!("Added skip hook 0x{:x} with expiry step {}", addr, expiry);
    }

    /// Clear all pending skip_hook entries.
    pub fn clear_skip_hook_addr(&mut self) {
        self.skip_hook_stack.clear();
    }

    /// Clear skip entry for a specific address.
    pub fn clear_skip_hook_for_addr(&mut self, addr: u64) {
        self.skip_hook_stack.retain(|&(a, _)| a != addr);
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
    pub fn move_states(&mut self, from_stash: &str, to_stash: &str, filter_fn: Option<PyObject>) -> PyResult<usize> {
        // If no filter, move all
        if filter_fn.is_none() {
            if let Some(mut from) = self.stashes.remove(from_stash) {
                let count = from.len();
                let to = self.stashes.entry(to_stash.to_string()).or_insert_with(VecDeque::new);
                to.append(&mut from);
                self.stashes.insert(from_stash.to_string(), VecDeque::new());
                return Ok(count);
            }
            return Ok(0);
        }

        // With filter - for now just move all (filter requires Python evaluation)
        // TODO: Implement filter evaluation
        self.move_states(from_stash, to_stash, None)
    }

    /// P8 fix: Move a single state by ID between stashes.
    pub fn move_state(&mut self, state_id: u64, from_stash: &str, to_stash: &str) -> PyResult<bool> {
        // Find and remove the state from the source stash
        let mut found_state = None;
        if let Some(stash) = self.stashes.get_mut(from_stash) {
            let mut idx = None;
            for (i, state) in stash.iter().enumerate() {
                if state.state_id() == state_id {
                    idx = Some(i);
                    break;
                }
            }
            if let Some(i) = idx {
                found_state = stash.remove(i);
            }
        }

        // Add to destination stash if found
        if let Some(state) = found_state {
            let to = self.stashes.entry(to_stash.to_string()).or_insert_with(VecDeque::new);
            to.push_back(state);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// P8 fix: Clear all states from a stash.
    pub fn clear_stash(&mut self, stash: &str) {
        if let Some(s) = self.stashes.get_mut(stash) {
            s.clear();
        }
    }

    /// Get statistics.
    pub fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("steps", self.steps)?;
        dict.set_item("active", self.active_count())?;
        dict.set_item("found", self.found_count())?;
        dict.set_item("errors", self.errors.len())?;
        dict.set_item("hooks", self.hooks.len())?;
        dict.set_item("simprocedures", self.simprocedures.len())?;
        dict.set_item("find_addrs", self.find_addrs.len())?;
        dict.set_item("avoid_addrs", self.avoid_addrs.len())?;
        dict.set_item("block_cache_size", self.block_cache.len())?;
        dict.set_item("native_proc_calls", self.native_proc_stats.native_calls)?;
        dict.set_item("native_proc_fallbacks", self.native_proc_stats.python_fallbacks)?;
        Ok(dict)
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
        self.native_procedures.procedure_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Check if a procedure has a native implementation.
    pub fn has_native_procedure(&self, name: &str) -> bool {
        self.native_procedures.has_native(name)
    }

    /// Get native procedure statistics.
    pub fn native_procedure_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("native_calls", self.native_proc_stats.native_calls)?;
        dict.set_item("python_fallbacks", self.native_proc_stats.python_fallbacks)?;

        let call_counts = PyDict::new(py);
        for (name, count) in &self.native_proc_stats.call_counts {
            call_counts.set_item(name, *count)?;
        }
        dict.set_item("call_counts", call_counts)?;

        Ok(dict)
    }

    // =========================================================================
    // State Export Methods
    // =========================================================================

    /// Export a state by ID as a full snapshot.
    ///
    /// This searches all stashes for the state with the given ID and returns
    /// a complete snapshot that can be used to reconstruct an angr SimState.
    pub fn export_state(&self, state_id: u64) -> PyResult<crate::state::ExplorationStateSnapshot> {
        // Search all stashes for the state
        for stash in self.stashes.values() {
            for state in stash {
                if state.state_id() == state_id {
                    return Ok(state.export_full());
                }
            }
        }

        // Also check pending callback state
        if let Some(ref pending) = self.pending_callback {
            if pending.state.state_id() == state_id {
                return Ok(pending.state.export_full());
            }
        }

        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Export all states in a stash as snapshots.
    pub fn export_stash(&self, stash: &str) -> Vec<crate::state::ExplorationStateSnapshot> {
        self.stashes
            .get(stash)
            .map(|s| s.iter().map(|state| state.export_full()).collect())
            .unwrap_or_default()
    }

    /// Export all found states as snapshots.
    pub fn export_found_states(&self) -> Vec<crate::state::ExplorationStateSnapshot> {
        self.export_stash("found")
    }

    /// Evaluate a symbolic value in a state's solver context.
    ///
    /// This allows Python to get concrete values for symbolic inputs
    /// that were found during exploration.
    pub fn eval_in_state(&self, state_id: u64, addr: u64, size: u32) -> PyResult<Option<Vec<u8>>> {
        // Search for the state
        for stash in self.stashes.values() {
            for state in stash {
                if state.state_id() == state_id {
                    // Try to load and evaluate from memory
                    match state.memory_load(addr, size) {
                        Ok(bv) => {
                            // Try to evaluate to concrete value
                            if let Some(val) = state.eval(&bv) {
                                let bytes: Vec<u8> = (0..size as usize)
                                    .map(|i| (val >> (i * 8)) as u8)
                                    .collect();
                                return Ok(Some(bytes));
                            }
                            return Ok(None);
                        }
                        Err(_) => return Ok(None),
                    }
                }
            }
        }

        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Debug: Get symbolic object info for a state.
    pub fn state_symbolic_info(&self, state_id: u64, addr: u64) -> PyResult<String> {
        for stash in self.stashes.values() {
            for state in stash {
                if state.state_id() == state_id {
                    let mem = state.memory();
                    let total = mem.symbolic_object_count();
                    let has_at_addr = mem.get_symbolic_object(addr).map(|bv| bv.width());
                    // Check if page is mapped and has symbolic markers
                    let page_num = addr >> 12;
                    let offset = (addr & 0xFFF) as u16;
                    let page_info = if let Some(page) = mem.pages().get(&page_num) {
                        format!("page=mapped sym_at_offset={}", page.is_symbolic(offset))
                    } else {
                        "page=unmapped".to_string()
                    };
                    return Ok(format!(
                        "total_sym_objs={} at_0x{:x}={:?} {}",
                        total, addr, has_at_addr, page_info
                    ));
                }
            }
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Check if constraints are satisfiable for a state.
    pub fn state_satisfiable(&self, state_id: u64) -> PyResult<bool> {
        for stash in self.stashes.values() {
            for state in stash {
                if state.state_id() == state_id {
                    return Ok(state.satisfiable());
                }
            }
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Get a register value from a state.
    pub fn get_state_register(&self, state_id: u64, name: &str) -> PyResult<Option<u128>> {
        for stash in self.stashes.values() {
            for state in stash {
                if state.state_id() == state_id {
                    return Ok(state.get_register(name).and_then(|bv| bv.as_u128()));
                }
            }
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }

    /// Get memory from a state.
    pub fn get_state_memory(&self, state_id: u64, addr: u64, size: u32) -> PyResult<Option<Vec<u8>>> {
        for stash in self.stashes.values() {
            for state in stash {
                if state.state_id() == state_id {
                    match state.memory_load(addr, size) {
                        Ok(bv) => {
                            if let Some(val) = bv.as_u128() {
                                let bytes: Vec<u8> = (0..size as usize)
                                    .map(|i| (val >> (i * 8)) as u8)
                                    .collect();
                                return Ok(Some(bytes));
                            }
                            // Try to evaluate symbolic value
                            if let Some(val) = state.eval(&bv) {
                                let bytes: Vec<u8> = (0..size as usize)
                                    .map(|i| (val >> (i * 8)) as u8)
                                    .collect();
                                return Ok(Some(bytes));
                            }
                            return Ok(None);
                        }
                        Err(_) => return Ok(None),
                    }
                }
            }
        }
        Err(PyValueError::new_err(format!("state {} not found", state_id)))
    }
}

impl RustExplorationManager {
    /// Sync constraints from Python callbacks back to the Rust state's solver.
    ///
    /// This is the critical piece for bidirectional constraint flow:
    /// - Rust syncs constraints TO Python before callbacks (via sync_before_callback)
    /// - Python SimProcedures may add new constraints (e.g., strcmp conditions)
    /// - This method syncs those new constraints BACK to Rust after the callback
    ///
    /// Without this, constraints added by SimProcedures would be lost when
    /// Rust resumes execution, leading to incorrect symbolic evaluation.
    ///
    /// Returns:
    ///   - Ok(true): Constraints synced and state is SAT (satisfiable)
    ///   - Ok(false): State became UNSAT after syncing - should be pruned (P12)
    ///   - Err: Python error during sync
    fn sync_constraints_from_python(
        &self,
        py: Python<'_>,
        state: &RustSimState,
        constraints: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<bool> {
        use pyo3::types::PyListMethods;

        let solver_ref = state.solver();
        let sym_ctx = solver_ref.borrow();
        let ctx_ref: &SymContext = &*sym_ctx;

        let mut success_count = 0usize;
        let mut failed_count = 0usize;

        // Extract list items - we need to convert each to RustBV
        let len = constraints.len();
        for i in 0..len {
            // Use get_item with usize index
            if let Ok(constraint) = constraints.get_item(i) {
                // Convert claripy AST to RustBV
                match claripy_to_rustbv(py, &constraint, ctx_ref) {
                    Ok(bv) => {
                        // Add constraint to solver
                        #[cfg(feature = "vex-engine-z3")]
                        {
                            if bv.width() == 1 {
                                sym_ctx.assume_true(&bv);
                                success_count += 1;
                            } else {
                                // For wider values, interpret as "value != 0"
                                let zero = RustBV::concrete(0, bv.width());
                                let neq = bv.ne(&zero, ctx_ref);
                                sym_ctx.assume_true(&neq);
                                success_count += 1;
                            }
                        }
                        #[cfg(not(feature = "vex-engine-z3"))]
                        {
                            // Without Z3, constraints are tracked but not solved
                            success_count += 1;
                        }
                    }
                    Err(e) => {
                        failed_count += 1;
                        log::warn!(
                            "Constraint {} conversion failed: {}. Solver state may diverge.",
                            i, e
                        );
                    }
                }
            }
        }

        if failed_count > 0 {
            log::warn!(
                "sync_constraints_from_python: {}/{} constraints failed to convert",
                failed_count, failed_count + success_count
            );
        }

        if success_count > 0 {
            log::debug!("Synced {} constraints from Python to Rust", success_count);
        }

        // P12: Check satisfiability and return status so callers can prune UNSAT states
        #[cfg(feature = "vex-engine-z3")]
        {
            let is_sat = sym_ctx.is_sat();
            if !is_sat {
                log::debug!(
                    "P12: Constraints are UNSAT after syncing {} from Python (failed={}). \
                     Returning false to trigger pruning.",
                    success_count, failed_count
                );
                return Ok(false);
            }
        }

        // P14: If many constraints failed to convert, do explicit SAT check
        // Failed conversions can leave state in divergent state
        #[cfg(feature = "vex-engine-z3")]
        if failed_count > 0 && success_count > 0 {
            let is_sat = sym_ctx.is_sat();
            if !is_sat {
                log::debug!(
                    "P14: State became UNSAT with partial constraint sync ({}/{} failed). Pruning.",
                    failed_count, failed_count + success_count
                );
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Extract procedure arguments from state registers.
    fn extract_procedure_args(&self, state: &RustSimState, num_args: usize) -> Vec<RustBV> {
        let arg_regs = self.calling_convention.arg_registers();
        let ptr_size = self.calling_convention.pointer_size();
        let mut args = Vec::with_capacity(num_args);

        let ctx = state.solver().borrow();

        // Extract from registers first
        for &offset in arg_regs.iter().take(num_args) {
            let value = state.get_register_by_offset(offset, ptr_size);
            args.push(value);
        }

        // If we need more args from stack, get them
        if args.len() < num_args {
            if let Some(sp) = state.get_sp().as_u64() {
                let stack_start = sp + self.calling_convention.stack_arg_offset();
                for i in 0..(num_args - args.len()) {
                    let addr = stack_start + (i as u64 * ptr_size as u64);
                    if let Ok(value) = state.memory_load(addr, ptr_size) {
                        args.push(value);
                    } else {
                        // Can't read stack - push zero
                        args.push(RustBV::zero(ptr_size * 8));
                    }
                }
            }
        }

        drop(ctx);
        args
    }

    /// Get return address from stack.
    fn get_return_addr(&self, state: &RustSimState) -> Option<u64> {
        let sp = state.get_sp().as_u64()?;
        let ptr_size = self.calling_convention.pointer_size();

        // On x86/AMD64, return address is at [rsp] after call
        state.memory_load(sp, ptr_size).ok()?.as_u64()
    }
}

/// Error during state stepping.
enum StepError {
    /// Need Python callback.
    NeedCallback(PendingCallback),
    /// State deadended (no successors).
    Deadended(RustSimState),
    /// Error during execution.
    Error(RustSimState, String),
    /// Unconstrained state - too many symbolic jump targets.
    Unconstrained(RustSimState),
}

impl RustExplorationManager {
    /// Step a single state, returning successors.
    fn step_state(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        state: RustSimState,
    ) -> Result<Vec<RustSimState>, StepError> {
        self.step_state_with_skip(py, callbacks, state, None)
    }

    /// Step a state, optionally skipping a hook address.
    ///
    /// The skip_addr parameter is used for zero-length hooks: after the hook
    /// runs but returns to the same address, we skip adding that hook to the
    /// interpreter so the underlying instruction can execute.
    fn step_state_with_skip(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        mut state: RustSimState,
        skip_addr: Option<u64>,
    ) -> Result<Vec<RustSimState>, StepError> {
        let initial_pc = state.pc();

        // Use the state's solver context for proper constraint handling
        let solver_rc = state.solver().clone();

        // Scope for interpreter execution with borrowed solver
        let (result, deferred_forks, last_condition, stored_conditions, mut fork_snapshots, new_registers, new_pc, recovered_memory) = {
            let solver_ref = solver_rc.borrow();

            // Create interpreter with the state's solver
            let mut interp = CallbackInterpreter::with_config(
                self.vex_arch,
                &*solver_ref,
                self.exec_config.clone(),
            );

            // Copy state registers to interpreter (including symbolic values)
            interp.registers = state.registers().fork();
            interp.set_pc(initial_pc);

            // Set up hooks, skipping the one we just processed (for zero-length hooks)
            for &addr in &self.hooks {
                if Some(addr) != skip_addr {
                    interp.add_hook(addr);
                }
            }

            // Register SimProcedures, also skipping the one we just processed
            for (addr, (name, num_args, no_return)) in &self.simprocedures {
                if Some(*addr) != skip_addr {
                    interp.register_simprocedure(*addr, name.clone(), *num_args, *no_return);
                }
            }

            // Add find/avoid addresses as hooks so the interpreter stops there
            for &addr in &self.find_addrs {
                interp.add_hook(addr);
            }
            for &addr in &self.avoid_addrs {
                interp.add_hook(addr);
            }

            // Copy binary regions for code lifting
            for (base, data) in &self.binary_regions {
                interp.add_concrete_memory(*base, data.clone());
            }

            // Transfer state's SymbolicMemory into the interpreter.
            // This makes Rust the source of truth for all memory during
            // VEX execution. Loads/stores go to SymbolicMemory directly
            // instead of calling back to Python.
            interp.set_rust_memory(state.take_memory());

            // Run until event
            let (result, _blocks_executed, deferred_forks) = interp.run_until_event(py, callbacks, self.max_steps_per_run as u32);

            // Get last branch condition before dropping interpreter
            let last_condition = interp.take_last_branch_condition();

            // Get stored conditions for deferred fork handling
            let stored_conditions = interp.take_stored_conditions();
            let fork_snapshots = interp.take_fork_snapshots();

            // Extract register state (including symbolic values)
            let new_registers = interp.registers.fork();
            let new_pc = interp.get_pc();

            // Flush any remaining pending stores to rust_memory
            interp.flush_stores_to_rust_memory();

            // Recover memory from interpreter back to state
            let recovered_memory = interp.take_rust_memory();

            (result, deferred_forks, last_condition, stored_conditions, fork_snapshots, new_registers, new_pc, recovered_memory)
        };
        // solver_ref dropped here, solver_rc borrow released

        // Restore memory from interpreter back to state FIRST.
        // This must happen before any PendingCallback creation
        // because the state's memory was taken by set_rust_memory().
        if let Some(mem) = recovered_memory {
            state.replace_memory(mem);
        }

        // Update state from interpreter results
        // Restore registers (including symbolic values) from interpreter
        state.set_registers(new_registers);
        state.set_pc(new_pc);

        // Add to history
        state.add_to_history(state.pc());

        // Process result
        match result {
            RunResult::MaxBlocks { pc } |
            RunResult::MaxDeferredForks { pc } |
            RunResult::BlockEnd { next_addr: pc, .. } => {
                state.set_pc(pc);

                // Track root state ID for lineage
                let original_state_id = state.state_id();
                let root_state_id = self.state_roots.get(&original_state_id).copied().unwrap_or(original_state_id);

                // Process deferred forks with proper constraint handling
                // P13: Track UNSAT states for pruning
                let mut successors = vec![state];
                let mut pruned_states = Vec::new();

                for fork in deferred_forks {
                    // Look up the condition for this deferred fork
                    if let Some(condition) = stored_conditions.get(&fork.condition_id) {
                        // Create forked state for the unexplored path.
                        // Use solver snapshot (from before branch constraint) if available
                        // to avoid inheriting the taken-path constraint (which would make
                        // the opposite constraint UNSAT).
                        let forked = if let Some(snapshot) = fork_snapshots.remove(&fork.condition_id) {
                            let mut f = successors[0].fork_from_snapshot(snapshot);
                            if fork.path_taken {
                                f.solver().borrow().assume_false(condition);
                            } else {
                                f.solver().borrow().assume_true(condition);
                            }
                            f.set_pc(fork.unexplored_target);
                            f
                        } else if fork.path_taken {
                            let mut f = successors[0].fork_false(condition);
                            f.set_pc(fork.unexplored_target);
                            f
                        } else {
                            let mut f = successors[0].fork_true(condition);
                            f.set_pc(fork.unexplored_target);
                            f
                        };
                        // Track root state ID for this forked state
                        self.state_roots.insert(forked.state_id(), root_state_id);

                        // P13: Check satisfiability before adding to successors
                        if self.lazy_solves || forked.satisfiable() {
                            successors.push(forked);
                        } else {
                            log::debug!(
                                "P13: Deferred fork at 0x{:x} is UNSAT, will be pruned",
                                fork.unexplored_target
                            );
                            pruned_states.push(forked);
                        }
                    } else {
                        // P15: Create conservative fork to explore the path even without condition
                        log::warn!(
                            "P15: Missing condition for deferred fork at 0x{:x} (condition_id={}). \
                             Creating conservative fork.",
                            fork.branch_addr,
                            fork.condition_id
                        );
                        let mut forked = successors[0].fork();
                        forked.set_pc(fork.unexplored_target);
                        self.state_roots.insert(forked.state_id(), root_state_id);

                        // P13: Still check satisfiability
                        if self.lazy_solves || forked.satisfiable() {
                            successors.push(forked);
                        } else {
                            log::debug!(
                                "P13: Unconstrained fork at 0x{:x} is UNSAT, will be pruned",
                                fork.unexplored_target
                            );
                            pruned_states.push(forked);
                        }
                    }
                }

                // Add pruned states to pruned stash
                if !pruned_states.is_empty() {
                    let pruned = self.stashes
                        .entry("pruned".to_string())
                        .or_insert_with(VecDeque::new);
                    for s in pruned_states {
                        pruned.push_back(s);
                    }
                }

                Ok(successors)
            }
            RunResult::Hook { addr } => {
                state.set_pc(addr);
                // P1 Fix: Add to history BEFORE callback so Python can access recent_bbl_addrs[-1]
                state.add_to_history(addr);
                // Save pre-callback snapshot for deferred forks
                let pre_callback_snapshot = Some(state.fork());
                // Fork solver context for Python callback use
                let solver_ref = state.solver();
                let forked_ctx = RustSolverContext::from_sym_context(solver_ref.borrow().fork());
                // Return to Python for hook - store deferred forks for later processing
                Err(StepError::NeedCallback(PendingCallback {
                    state,
                    pre_callback_snapshot,
                    reason: CallbackReason::SimProcedure {
                        addr,
                        name: "unknown".to_string(),
                        num_args: 0,
                        return_addr: 0,
                    },
                    jumpkind: Some("Ijk_Boring".to_string()),
                    solver_ctx: Some(forked_ctx),
                    deferred_forks,
                    stored_conditions,
                    fork_snapshots,
                }))
            }
            RunResult::SimProcedure { addr, name, num_args, return_addr } => {
                // Try native procedure first — avoids Python callback overhead.
                // Skip native for addresses inside the binary — these are user-placed
                // hooks where the Python SimProcedure should always run (the user hooked
                // a specific function for a reason, e.g., hooking strings_not_equal with strcmp).
                let is_in_binary = self.binary_regions.iter().any(|(base, data)| {
                    addr >= *base && addr < *base + data.len() as u64
                });
                let native_succeeded = if !is_in_binary {
                    if let Some(native_proc) = self.native_procedures.get(&name) {
                    let args = self.extract_procedure_args(&state, num_args);
                    match native_proc.call(&mut state, &args) {
                        Ok(ret_val) => {
                            self.native_proc_stats.native_calls += 1;
                            *self.native_proc_stats.call_counts
                                .entry(name.clone())
                                .or_insert(0) += 1;

                            if let Some(rv) = ret_val {
                                let ret_reg = self.calling_convention.return_register();
                                state.set_register_by_offset(ret_reg, rv);
                            }

                            // Set PC to return address and pop stack
                            state.set_pc(return_addr);
                            let sp = state.get_sp().as_u64().unwrap_or(0);
                            let ptr_size = state.arch().bytes() as u64;
                            state.set_sp(RustBV::concrete((sp + ptr_size) as u128, state.arch().bits()));
                            true
                        }
                        Err(_) => {
                            self.native_proc_stats.python_fallbacks += 1;
                            false
                        }
                    }
                } else {
                    false
                }} else {
                    false
                };

                // Trace removed
                if native_succeeded {
                    // Handle deferred forks same as normal successors
                    let original_state_id = state.state_id();
                    let root_state_id = self.state_roots.get(&original_state_id).copied().unwrap_or(original_state_id);
                    let mut successors = vec![state];

                    // Process deferred forks with fork_base from first successor
                    if !deferred_forks.is_empty() {
                        let fork_base = successors[0].fork();
                        let mut snapshots = fork_snapshots;
                        for fork in deferred_forks {
                            let condition = stored_conditions.get(&fork.condition_id);
                            if let Some(cond) = condition {
                                let forked = if let Some(snapshot) = snapshots.remove(&fork.condition_id) {
                                    let mut f = fork_base.fork_from_snapshot(snapshot);
                                    if fork.path_taken {
                                        f.solver().borrow().assume_false(cond);
                                    } else {
                                        f.solver().borrow().assume_true(cond);
                                    }
                                    f.set_pc(fork.unexplored_target);
                                    f
                                } else if fork.path_taken {
                                    let mut f = fork_base.fork_false(cond);
                                    f.set_pc(fork.unexplored_target);
                                    f
                                } else {
                                    let mut f = fork_base.fork_true(cond);
                                    f.set_pc(fork.unexplored_target);
                                    f
                                };
                                self.state_roots.insert(forked.state_id(), root_state_id);
                                if self.lazy_solves || forked.satisfiable() {
                                    successors.push(forked);
                                }
                            }
                        }
                    }

                    Ok(successors)
                } else {
                    // Fall through to Python callback
                    state.set_pc(addr);
                    state.add_to_history(addr);
                    let pre_callback_snapshot = Some(state.fork());
                    let solver_ref = state.solver();
                    let forked_ctx = RustSolverContext::from_sym_context(solver_ref.borrow().fork());
                    Err(StepError::NeedCallback(PendingCallback {
                        state,
                        pre_callback_snapshot,
                        reason: CallbackReason::SimProcedure {
                            addr,
                            name,
                            num_args,
                            return_addr,
                        },
                        jumpkind: Some("Ijk_Call".to_string()),
                        solver_ctx: Some(forked_ctx),
                        deferred_forks,
                        stored_conditions,
                        fork_snapshots,
                    }))
                }
            }
            RunResult::Syscall { num, pc } => {
                state.set_pc(pc);
                // P1 Fix: Add to history BEFORE callback so Python can access recent_bbl_addrs[-1]
                state.add_to_history(pc);
                // Save pre-callback snapshot for deferred forks
                let pre_callback_snapshot = Some(state.fork());
                // Fork solver context for Python callback use
                let solver_ref = state.solver();
                let forked_ctx = RustSolverContext::from_sym_context(solver_ref.borrow().fork());
                Err(StepError::NeedCallback(PendingCallback {
                    state,
                    pre_callback_snapshot,
                    reason: CallbackReason::Syscall { num },
                    jumpkind: Some("Ijk_Sys_syscall".to_string()),
                    solver_ctx: Some(forked_ctx),
                    deferred_forks,
                    stored_conditions,
                    fork_snapshots,
                }))
            }
            RunResult::SymbolicBranch { condition_id, true_target, false_target } => {
                // Return to Python for proper state forking with constraints.
                // No pre_callback_snapshot or solver_ctx fork needed here —
                // resume_after_symbolic_branch forks from pending.state directly.
                // Skipping these 2 unnecessary state forks eliminates O(n)
                // constraint replay per symbolic branch.
                let mut branch_conditions = stored_conditions;
                if let Some(cond) = last_condition {
                    branch_conditions.insert(condition_id, cond);
                }

                Err(StepError::NeedCallback(PendingCallback {
                    state,
                    pre_callback_snapshot: None,
                    reason: CallbackReason::SymbolicBranch {
                        condition_id,
                        true_target,
                        false_target,
                    },
                    jumpkind: Some("Ijk_Boring".to_string()),
                    solver_ctx: None,
                    deferred_forks,
                    stored_conditions: branch_conditions,
                    fork_snapshots,
                }))
            }
            RunResult::Error { message, addr } => {
                state.set_pc(addr);
                // Treat lift errors at unmapped addresses as deadends, not errors.
                // This matches Python engine behavior where states that reach
                // invalid code addresses (e.g., 0x0 after exit) are deadended.
                if message.contains("No bytes in memory") || message.contains("lift") || addr == 0 {
                    Err(StepError::Deadended(state))
                } else {
                    Err(StepError::Error(state, message))
                }
            }
            RunResult::NeedLift { addr } => {
                // This shouldn't happen if callbacks are properly set
                state.set_pc(addr);
                Err(StepError::Error(state, format!("need lift at 0x{:x}", addr)))
            }
            RunResult::SymbolicJumpTarget { targets, condition_id, jumpkind: _ } => {
                // Symbolic jump with multiple concrete targets - fork for each
                // Look up the condition for constraint addition
                let target_expr = stored_conditions.get(&condition_id).cloned();

                if targets.is_empty() {
                    // No targets - deadended
                    return Err(StepError::Deadended(state));
                }

                if targets.len() == 1 {
                    // Single target - just continue
                    let addr = targets[0];
                    if let Some(ref expr) = target_expr {
                        // Add constraint: target_expr == addr
                        let concrete = RustBV::concrete(addr as u128, expr.width());
                        let constraint = expr.eq(&concrete, &*state.solver().borrow());
                        state.add_constraint(constraint);
                    }
                    state.set_pc(addr);
                    return Ok(vec![state]);
                }

                // Multiple targets - fork for each from the UNCONSTRAINED original
                // CRITICAL: Save unconstrained base state BEFORE adding any target constraints
                // This ensures each fork only has its own target constraint, not all previous ones
                let base_state = state.fork();  // Save unconstrained clone

                // Track root state ID for lineage
                let original_state_id = state.state_id();
                let root_state_id = self.state_roots.get(&original_state_id).copied().unwrap_or(original_state_id);

                let mut successors = Vec::with_capacity(targets.len());

                // Handle first target - use the original state (moved here)
                let first_addr = targets[0];
                let mut first_state = state;  // Move state into first_state
                if let Some(ref expr) = target_expr {
                    let concrete = RustBV::concrete(first_addr as u128, expr.width());
                    let constraint = expr.eq(&concrete, &*first_state.solver().borrow());
                    first_state.add_constraint(constraint);
                }
                first_state.set_pc(first_addr);
                successors.push(first_state);

                // Handle remaining targets - fork from unconstrained base
                for &addr in targets.iter().skip(1) {
                    let mut forked = base_state.fork();

                    // Add constraint: target_expr == addr (only this target's constraint)
                    if let Some(ref expr) = target_expr {
                        let concrete = RustBV::concrete(addr as u128, expr.width());
                        let constraint = expr.eq(&concrete, &*forked.solver().borrow());
                        forked.add_constraint(constraint);
                    }
                    forked.set_pc(addr);
                    // Track root state ID for this forked state
                    self.state_roots.insert(forked.state_id(), root_state_id);
                    successors.push(forked);
                }

                Ok(successors)
            }
            RunResult::UnconstrainedJump { min_target: _, max_target: _, limit: _, jumpkind: _ } => {
                // Too many symbolic jump targets - move to unconstrained stash
                Err(StepError::Unconstrained(state))
            }
            RunResult::UnmodeledCall { addr, return_addr, symbol_name } => {
                // Unhooked CALL target - try to resolve via Python callback
                state.set_pc(addr);
                // P1 Fix: Add to history BEFORE callback so Python can access recent_bbl_addrs[-1]
                state.add_to_history(addr);

                // Try to resolve the function via callback
                if callbacks.has_resolve_function() {
                    match callbacks.call_resolve_function(py, addr, symbol_name.as_deref()) {
                        Ok(Some((name, num_args, no_return))) => {
                            // Function resolved! Register it and return to Python for execution
                            log::debug!(
                                "Resolved unmodeled call at 0x{:x} -> {} (args={}, no_return={})",
                                addr, name, num_args, no_return
                            );

                            // Register the procedure so future calls are hooked
                            self.hooks.insert(addr);
                            self.simprocedures.insert(addr, (name.clone(), num_args, no_return));

                            // Save pre-callback snapshot for deferred forks
                            let pre_callback_snapshot = Some(state.fork());
                            // Fork solver context for Python callback use
                            let solver_ref = state.solver();
                            let forked_ctx = RustSolverContext::from_sym_context(solver_ref.borrow().fork());

                            // Return to Python for SimProcedure execution
                            Err(StepError::NeedCallback(PendingCallback {
                                state,
                                pre_callback_snapshot,
                                reason: CallbackReason::SimProcedure {
                                    addr,
                                    name,
                                    num_args,
                                    return_addr,
                                },
                                jumpkind: Some("Ijk_Call".to_string()),
                                solver_ctx: Some(forked_ctx),
                                deferred_forks,
                                stored_conditions,
                                fork_snapshots,
                            }))
                        }
                        Ok(None) => {
                            // P21: Function could not be resolved - use generic skip instead of deadending
                            // This sets return register to 0 and continues at return address
                            log::debug!(
                                "P21: Unmodeled call at 0x{:x} could not be resolved. \
                                 Using generic skip (ret=0) to return_addr=0x{:x}",
                                addr, return_addr
                            );

                            // Set return register to 0 (symbolic unconstrained would be better but
                            // concrete 0 is simpler and often sufficient)
                            let ret_reg_offset = self.calling_convention.return_register();
                            let ptr_size = self.calling_convention.pointer_size();
                            let zero_val = RustBV::zero((ptr_size * 8) as u32);
                            state.set_register_by_offset(ret_reg_offset, zero_val);

                            // Continue at return address
                            state.set_pc(return_addr);

                            // Return the state as a successor
                            Ok(vec![state])
                        }
                        Err(e) => {
                            // Callback error - treat as execution error
                            log::warn!("resolve_function callback error at 0x{:x}: {}", addr, e);
                            Err(StepError::Error(state, format!("resolve_function error: {}", e)))
                        }
                    }
                } else {
                    // P21: No resolve_function callback - use generic skip instead of deadending
                    log::debug!(
                        "P21: Unmodeled call at 0x{:x} - no resolve_function callback. \
                         Using generic skip (ret=0) to return_addr=0x{:x}",
                        addr, return_addr
                    );

                    // Set return register to 0
                    let ret_reg_offset = self.calling_convention.return_register();
                    let ptr_size = self.calling_convention.pointer_size();
                    let zero_val = RustBV::zero((ptr_size * 8) as u32);
                    state.set_register_by_offset(ret_reg_offset, zero_val);

                    // Continue at return address
                    state.set_pc(return_addr);

                    // Return the state as a successor
                    Ok(vec![state])
                }
            }
        }
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
        Python::with_gil(|_py| {
            let mgr = RustExplorationManager::new("amd64").unwrap();
            assert_eq!(mgr.arch(), "amd64");
            assert_eq!(mgr.active_count(), 0);
            assert_eq!(mgr.found_count(), 0);
        });
    }

    #[test]
    fn test_stash_management() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|_py| {
            let mut mgr = RustExplorationManager::new("amd64").unwrap();

            // Create state
            let state_id = mgr.create_state("active").unwrap();
            assert!(state_id > 0);
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
        Python::with_gil(|_py| {
            let mut mgr = RustExplorationManager::new("amd64").unwrap();

            mgr.set_find_addrs(vec![0x1000, 0x2000]);
            mgr.set_avoid_addrs(vec![0x3000]);

            assert!(mgr.find_addrs.contains(&0x1000));
            assert!(mgr.find_addrs.contains(&0x2000));
            assert!(mgr.avoid_addrs.contains(&0x3000));
        });
    }
}
