//! Python callback infrastructure for Rust VEX engine.
//!
//! This module provides the callback holder that allows Rust to call back into Python
//! for operations like memory access, hook execution, and syscall handling.

use pyo3::class::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Arc;

use crate::symbolic::RustBV;

/// A branch that was taken but has an unexplored alternative.
///
/// When the Rust engine encounters a symbolic branch where both paths are
/// feasible, it picks one path to continue executing and records the other
/// as a deferred fork. Python can later create states for these unexplored
/// branches and schedule them for execution.
#[pyclass]
#[derive(Debug, Clone)]
pub struct DeferredFork {
    /// Address where the branch occurred.
    #[pyo3(get)]
    pub branch_addr: u64,
    /// The path we took (true = took true branch, false = took false branch).
    #[pyo3(get)]
    pub path_taken: bool,
    /// Address of the unexplored path.
    #[pyo3(get)]
    pub unexplored_target: u64,
    /// Condition ID for constraint tracking.
    /// Python can use this to reconstruct the branch condition.
    #[pyo3(get)]
    pub condition_id: u64,
    /// Solver push level before this branch constraint was added.
    /// Used for proper constraint handling during fork processing.
    #[pyo3(get)]
    pub push_level: u32,
    /// The branch condition as a claripy AST (if available).
    /// This is the original condition - path_taken indicates which path
    /// was explored. For the fork, we need the opposite constraint.
    #[pyo3(get)]
    pub condition_ast: Option<Py<PyAny>>,
}

#[pymethods]
impl DeferredFork {
    /// Create a new deferred fork.
    #[new]
    #[pyo3(signature = (branch_addr, path_taken, unexplored_target, condition_id, push_level=0, condition_ast=None))]
    pub fn new(
        branch_addr: u64,
        path_taken: bool,
        unexplored_target: u64,
        condition_id: u64,
        push_level: u32,
        condition_ast: Option<Py<PyAny>>,
    ) -> Self {
        DeferredFork {
            branch_addr,
            path_taken,
            unexplored_target,
            condition_id,
            push_level,
            condition_ast,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DeferredFork(branch_addr=0x{:x}, path_taken={}, unexplored=0x{:x}, push_level={})",
            self.branch_addr, self.path_taken, self.unexplored_target, self.push_level
        )
    }
}

/// Policy for choosing which branch to take when both paths are feasible.
#[pyclass]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BranchPolicy {
    /// Always take the true branch (default).
    #[default]
    TakeTrue,
    /// Always take the false branch.
    TakeFalse,
    /// Take the branch that continues to the next instruction (fall-through).
    TakeFallthrough,
    /// Alternate between true and false branches.
    Alternate,
}

#[pymethods]
impl BranchPolicy {
    /// Create the TakeTrue policy.
    #[staticmethod]
    pub fn take_true() -> Self {
        BranchPolicy::TakeTrue
    }

    /// Create the TakeFalse policy.
    #[staticmethod]
    pub fn take_false() -> Self {
        BranchPolicy::TakeFalse
    }

    /// Create the TakeFallthrough policy.
    #[staticmethod]
    pub fn take_fallthrough() -> Self {
        BranchPolicy::TakeFallthrough
    }

    /// Create the Alternate policy.
    #[staticmethod]
    pub fn alternate() -> Self {
        BranchPolicy::Alternate
    }
}

/// Configuration for the execution loop with deferred forks.
#[pyclass]
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Maximum deferred forks before returning to Python.
    /// When this limit is reached, execution returns to Python even if
    /// max_blocks hasn't been hit.
    #[pyo3(get, set)]
    pub max_deferred_forks: u32,
    /// Branch selection policy.
    #[pyo3(get, set)]
    pub branch_policy: BranchPolicy,
    /// Whether to use deferred forks (if false, returns immediately on symbolic branch).
    #[pyo3(get, set)]
    pub use_deferred_forks: bool,
    /// Whether to enable eager region prefetch (batch prefetch entire regions).
    #[pyo3(get, set)]
    pub enable_eager_prefetch: bool,
    /// Maximum pages to prefetch in a single batch (default: 256 = 1MB).
    #[pyo3(get, set)]
    pub max_prefetch_batch: usize,
    /// Maximum concretization range for symbolic addresses (default: 65536).
    #[pyo3(get, set)]
    pub max_concretization_range: u64,
    /// Enable stride detection for array access patterns (default: true).
    #[pyo3(get, set)]
    pub enable_stride_detection: bool,
    /// Maximum symbolic IP targets before marking unconstrained (default: 257).
    /// When a symbolic jump target (e.g., ret from symbolic return address)
    /// concretizes to more than this many targets, the state is marked
    /// as unconstrained rather than forking into many states.
    #[pyo3(get, set)]
    pub max_symbolic_ip_targets: usize,
}

#[pymethods]
impl ExecutionConfig {
    /// Create a new execution config with default values.
    #[new]
    #[pyo3(signature = (max_deferred_forks=500, use_deferred_forks=false))]
    pub fn py_new(max_deferred_forks: u32, use_deferred_forks: bool) -> Self {
        ExecutionConfig {
            max_deferred_forks,
            branch_policy: BranchPolicy::TakeTrue,
            use_deferred_forks,
            enable_eager_prefetch: true,
            max_prefetch_batch: 256,
            max_concretization_range: 65536,
            enable_stride_detection: true,
            max_symbolic_ip_targets: 257,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ExecutionConfig(max_deferred_forks={}, use_deferred_forks={}, policy={:?}, eager_prefetch={}, max_prefetch_batch={})",
            self.max_deferred_forks,
            self.use_deferred_forks,
            self.branch_policy,
            self.enable_eager_prefetch,
            self.max_prefetch_batch
        )
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        ExecutionConfig {
            max_deferred_forks: 500, // Increased from 100 for complex binaries
            branch_policy: BranchPolicy::TakeTrue,
            // Deferred forks enabled with one-per-step limit:
            // interpreter_cb.rs limits to one deferred fork per
            // run_until_event call, then falls back to non-deferred
            // mode. This gives single-exit blocks the performance
            // benefit while multi-exit blocks are handled by Python.
            use_deferred_forks: true,
            // Eager prefetch disabled: fetching entire regions (e.g. 256 stack
            // pages) on first access is extremely slow due to Python callbacks.
            // Individual pages are fetched on demand instead.
            enable_eager_prefetch: false,
            max_prefetch_batch: 256, // 256 pages = 1MB (unused when eager disabled)
            max_concretization_range: 65536,
            enable_stride_detection: true,
            max_symbolic_ip_targets: 257, // Match Python angr default
        }
    }
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
    /// Error during execution.
    Error { message: String, addr: u64 },
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

/// Python callback holder for the Rust VEX engine.
///
/// This struct holds references to Python callback functions that the Rust
/// engine calls during execution for memory access, hooks, syscalls, etc.
#[pyclass]
#[derive(Clone)]
pub struct PythonCallbacks {
    /// Callback for memory loads: fn(addr: u64, size: u32) -> (bytes, is_symbolic, symbolic_ast?)
    pub memory_load: Option<Py<PyAny>>,
    /// Callback for memory stores: fn(addr: u64, data: bytes) -> None
    pub memory_store: Option<Py<PyAny>>,
    /// Callback for batched memory stores: fn(stores: list[tuple[int, bytes]]) -> None
    /// This is more efficient than individual stores when multiple stores can be batched.
    pub memory_store_batch: Option<Py<PyAny>>,
    /// Callback for batched memory loads: fn(loads: list[tuple[int, int]]) -> list[tuple[bytes, bool, object | None]]
    /// Each tuple in input is (address, size). Returns list of (data, is_symbolic, ast_or_none).
    /// This is more efficient than individual loads when multiple loads can be batched.
    pub memory_load_batch: Option<Py<PyAny>>,
    /// Callback for symbolic memory loads: fn(addrs: list[int], size: int, addr_ast) -> RustBV
    pub memory_load_symbolic: Option<Py<PyAny>>,
    /// Callback for symbolic memory stores: fn(addrs: list[int], data: bytes, addr_ast) -> None
    pub memory_store_symbolic: Option<Py<PyAny>>,
    /// Callback for hook execution: fn(addr: u64) -> new_pc
    pub on_hook: Option<Py<PyAny>>,
    /// Callback for syscall handling: fn(num: u64) -> None
    pub on_syscall: Option<Py<PyAny>>,
    /// Callback for lifting a block: fn(addr: u64) -> irsb_json
    pub lift_block: Option<Py<PyAny>>,
    /// Callback for getting register value: fn(offset: u32, size: u32) -> (bytes, is_symbolic, symbolic_ast?)
    pub get_register: Option<Py<PyAny>>,
    /// Callback for setting register value: fn(offset: u32, data: bytes) -> None
    pub put_register: Option<Py<PyAny>>,
    /// Callback for dirty helper calls: fn(name: str, args: list[int], ret_ty_bits: int) -> (bytes, bool, object | None)
    /// This handles VEX dirty calls to helper functions (CPUID, RDTSC, etc.)
    pub dirty_call: Option<Py<PyAny>>,
    /// Callback for fetching a single 4KB page: fn(page_addr: u64) -> (bytes, permissions: u8, is_mapped: bool)
    /// This is used for on-demand page loading when Rust memory encounters an unmapped page.
    pub fetch_page: Option<Py<PyAny>>,
    /// Callback for batched page fetching: fn(page_addrs: list[u64]) -> list[(bytes, u8, bool)]
    /// Returns list of (data, permissions, is_mapped) for each requested page.
    pub batch_fetch_pages: Option<Py<PyAny>>,
    /// Callback for syncing constraints to Python: fn(constraints: list[(str, int, int, int | None)]) -> None
    /// Each constraint is (description, width, concrete_value, handle_id) where:
    /// - description: human-readable description (e.g., "addr_concretize_0x1234")
    /// - width: bit width of the constrained expression
    /// - concrete_value: the value the expression was constrained to
    /// - handle_id: optional handle ID to look up the original claripy AST
    /// Python should add these constraints to its claripy solver.
    pub sync_constraints: Option<Py<PyAny>>,
    /// Callback for storing a symbolic value with full expression tree: fn(addr: int, ast: claripy.AST) -> None
    /// This is called when storing a symbolic value to memory. The AST is reconstructed from
    /// the Rust expression tree, preserving the original symbolic expression structure.
    /// This allows symbolic values to be properly stored without data loss.
    pub memory_store_symbolic_value: Option<Py<PyAny>>,
    /// Callback for storing symbolic data at a symbolic address.
    /// Called when the address cannot be concretized (too many possibilities).
    /// Takes (addr_ast: claripy.AST, data_ast: claripy.AST) -> None
    /// Python should use state.memory.store(addr_ast, data_ast).
    pub memory_store_symbolic_full: Option<Py<PyAny>>,
    /// Callback for loading data at a symbolic address.
    /// Called when the address cannot be concretized (too many possibilities).
    /// Takes (addr_ast: claripy.AST, size: int) -> claripy.AST
    /// Python should use state.memory.load(addr_ast, size) and return the result.
    pub memory_load_symbolic_full: Option<Py<PyAny>>,
    /// Callback for resolving unmodeled function calls.
    /// Called when execution reaches a function that isn't hooked or modeled.
    /// Takes (addr: int, name: str | None) -> (name: str, num_args: int, no_return: bool) | None
    /// If returns None, the function is truly unmodeled and state should be deadended.
    /// If returns info, Rust will register the function as a SimProcedure and retry.
    pub resolve_function: Option<Py<PyAny>>,
    /// Callback for state.inspect mem_read events.
    /// Signature:
    ///   fn(state_id: int, when: str, addr: int, size: int,
    ///      value_ast: object | None, endness: str) -> None
    /// `when` is "before" or "after"; on BEFORE `value_ast` is None,
    /// on AFTER it holds the loaded value (concrete value as int or
    /// claripy AST). `endness` is "Iend_LE" or "Iend_BE".
    /// Only dispatched when bit InspectEvent::MemRead in `inspect_enabled` is set.
    pub inspect_mem_read: Option<Py<PyAny>>,
    /// Callback for state.inspect mem_write events.
    /// Signature mirrors `inspect_mem_read` but `value_ast` is the data
    /// being stored (set on BEFORE and AFTER).
    pub inspect_mem_write: Option<Py<PyAny>>,
    /// Bitmask of enabled inspect events. Bit N = `InspectEvent` variant N.
    /// VEX dispatch sites read this with a single `& != 0` check before
    /// touching any payload — keeps the cost of inspect-disabled
    /// execution at one branch per Load/Store.
    /// Python writes via `set_inspect_enabled`; defaults to 0 (off).
    ///
    /// Wrapped in `Arc<AtomicU8>` because `PythonCallbacks` is `Clone` and
    /// the Rust exploration manager stores a CLONED copy after Python
    /// passes the original in via `set_callbacks`. Bitmask updates from
    /// Python (`mgr._callbacks.set_inspect_enabled(...)`) must be visible
    /// to the Rust side; sharing the atomic makes both copies read/write
    /// the same byte.
    pub inspect_enabled: std::sync::Arc<std::sync::atomic::AtomicU8>,
}

#[pymethods]
impl PythonCallbacks {
    /// Create a new empty callback holder.
    #[new]
    pub fn new() -> Self {
        PythonCallbacks {
            memory_load: None,
            memory_store: None,
            memory_store_batch: None,
            memory_load_batch: None,
            memory_load_symbolic: None,
            memory_store_symbolic: None,
            on_hook: None,
            on_syscall: None,
            lift_block: None,
            get_register: None,
            put_register: None,
            dirty_call: None,
            fetch_page: None,
            batch_fetch_pages: None,
            sync_constraints: None,
            memory_store_symbolic_value: None,
            memory_store_symbolic_full: None,
            memory_load_symbolic_full: None,
            resolve_function: None,
            inspect_mem_read: None,
            inspect_mem_write: None,
            inspect_enabled: std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0)),
        }
    }

    /// Set the memory load callback.
    ///
    /// The callback should have signature:
    /// `fn(addr: int, size: int) -> tuple[bytes, bool, object | None]`
    ///
    /// Returns (concrete_bytes, is_symbolic, symbolic_ast_or_none).
    pub fn set_memory_load(&mut self, cb: Py<PyAny>) {
        self.memory_load = Some(cb);
    }

    /// Set the memory store callback.
    ///
    /// The callback should have signature:
    /// `fn(addr: int, data: bytes) -> None`
    pub fn set_memory_store(&mut self, cb: Py<PyAny>) {
        self.memory_store = Some(cb);
    }

    /// Set the batched memory store callback.
    ///
    /// The callback should have signature:
    /// `fn(stores: list[tuple[int, bytes]]) -> None`
    ///
    /// This is called with a batch of stores for efficiency. Each element is
    /// a (address, data) tuple. If not set, falls back to individual stores.
    pub fn set_memory_store_batch(&mut self, cb: Py<PyAny>) {
        self.memory_store_batch = Some(cb);
    }

    /// Set the batched memory load callback.
    ///
    /// The callback should have signature:
    /// `fn(loads: list[tuple[int, int]]) -> list[tuple[bytes, bool, object | None]]`
    ///
    /// Each tuple in input is (address, size). Returns list of (data, is_symbolic, ast_or_none).
    /// This is called with a batch of loads for efficiency, reducing FFI overhead.
    /// If not set, falls back to individual loads.
    pub fn set_memory_load_batch(&mut self, cb: Py<PyAny>) {
        self.memory_load_batch = Some(cb);
    }

    /// Set the symbolic memory load callback.
    ///
    /// The callback should have signature:
    /// `fn(addrs: list[int], size: int, addr_ast: object) -> RustBV`
    ///
    /// This is called when the address is symbolic and concretizes to multiple values.
    /// The callback should build an ITE chain based on the possible addresses.
    pub fn set_memory_load_symbolic(&mut self, cb: Py<PyAny>) {
        self.memory_load_symbolic = Some(cb);
    }

    /// Set the symbolic memory store callback.
    ///
    /// The callback should have signature:
    /// `fn(addrs: list[int], data: bytes, addr_ast: object) -> None`
    ///
    /// This is called when the address is symbolic and concretizes to multiple values.
    /// The callback should perform conditional stores to each possible address.
    pub fn set_memory_store_symbolic(&mut self, cb: Py<PyAny>) {
        self.memory_store_symbolic = Some(cb);
    }

    /// Set the hook execution callback.
    ///
    /// The callback should have signature:
    /// `fn(addr: int) -> int`
    ///
    /// Returns the new PC after hook execution.
    pub fn set_on_hook(&mut self, cb: Py<PyAny>) {
        self.on_hook = Some(cb);
    }

    /// Set the syscall handling callback.
    ///
    /// The callback should have signature:
    /// `fn(num: int) -> None`
    pub fn set_on_syscall(&mut self, cb: Py<PyAny>) {
        self.on_syscall = Some(cb);
    }

    /// Set the block lifting callback.
    ///
    /// The callback should have signature:
    /// `fn(addr: int) -> str`
    ///
    /// Returns the IRSB as a JSON string.
    pub fn set_lift_block(&mut self, cb: Py<PyAny>) {
        self.lift_block = Some(cb);
    }

    /// Set the register get callback.
    ///
    /// The callback should have signature:
    /// `fn(offset: int, size: int) -> tuple[bytes, bool, object | None]`
    ///
    /// Returns (concrete_bytes, is_symbolic, symbolic_ast_or_none).
    pub fn set_get_register(&mut self, cb: Py<PyAny>) {
        self.get_register = Some(cb);
    }

    /// Set the register put callback.
    ///
    /// The callback should have signature:
    /// `fn(offset: int, data: bytes) -> None`
    pub fn set_put_register(&mut self, cb: Py<PyAny>) {
        self.put_register = Some(cb);
    }

    /// Set the dirty call callback.
    ///
    /// The callback should have signature:
    /// `fn(name: str, args: list[int], ret_ty_bits: int) -> tuple[bytes, bool, object | None]`
    ///
    /// This handles VEX dirty calls to helper functions like CPUID, RDTSC, etc.
    /// Returns (concrete_bytes, is_symbolic, symbolic_ast_or_none).
    pub fn set_dirty_call(&mut self, cb: Py<PyAny>) {
        self.dirty_call = Some(cb);
    }

    /// Set the page fetch callback.
    ///
    /// The callback should have signature:
    /// `fn(page_addr: int) -> tuple[bytes, int, bool]`
    ///
    /// Returns (page_data_4kb, permissions, is_mapped).
    /// If is_mapped is False, the page doesn't exist in Python memory.
    pub fn set_fetch_page(&mut self, cb: Py<PyAny>) {
        self.fetch_page = Some(cb);
    }

    /// Set the batched page fetch callback.
    ///
    /// The callback should have signature:
    /// `fn(page_addrs: list[int]) -> list[tuple[bytes, int, bool]]`
    ///
    /// Each result is (page_data_4kb, permissions, is_mapped).
    pub fn set_batch_fetch_pages(&mut self, cb: Py<PyAny>) {
        self.batch_fetch_pages = Some(cb);
    }

    /// Set the constraint sync callback.
    ///
    /// The callback should have signature:
    /// `fn(constraints: list[tuple[str, int, int, int | None]]) -> None`
    ///
    /// Each tuple is (description, width, concrete_value, handle_id).
    /// Python should add these constraints to its claripy solver.
    /// The handle_id can be used to look up the original claripy AST.
    pub fn set_sync_constraints(&mut self, cb: Py<PyAny>) {
        self.sync_constraints = Some(cb);
    }

    /// Set the symbolic value store callback.
    ///
    /// The callback should have signature:
    /// `fn(addr: int, ast: claripy.AST) -> None`
    ///
    /// This is called when storing a symbolic value to memory. The AST
    /// is the fully reconstructed claripy expression from the Rust engine,
    /// preserving the original symbolic structure (e.g., `x + 10 ^ 0x42`
    /// instead of a fresh symbolic variable).
    pub fn set_memory_store_symbolic_value(&mut self, cb: Py<PyAny>) {
        self.memory_store_symbolic_value = Some(cb);
    }

    /// Set the callback for storing symbolic data at a symbolic address.
    /// Used when the address cannot be concretized (too many possibilities).
    pub fn set_memory_store_symbolic_full(&mut self, cb: Py<PyAny>) {
        self.memory_store_symbolic_full = Some(cb);
    }

    /// Set the callback for loading data at a symbolic address.
    /// Used when the address cannot be concretized (too many possibilities).
    pub fn set_memory_load_symbolic_full(&mut self, cb: Py<PyAny>) {
        self.memory_load_symbolic_full = Some(cb);
    }

    /// Set the callback for resolving unmodeled function calls.
    ///
    /// The callback should have signature:
    /// `fn(addr: int, name: str | None) -> tuple[str, int, bool] | None`
    ///
    /// If the function can be resolved, return (name, num_args, no_return).
    /// If not, return None to deadend the state.
    ///
    /// This is called when execution reaches a function that isn't hooked.
    /// The callback should check project._sim_procedures and angr's procedure registry.
    pub fn set_resolve_function(&mut self, cb: Py<PyAny>) {
        self.resolve_function = Some(cb);
    }

    /// Set the inspect mem_read callback.
    ///
    /// The callback should have signature:
    /// `fn(state_id: int, when: str, addr: int, size: int, value_ast: object | None, endness: str) -> None`
    pub fn set_inspect_mem_read(&mut self, cb: Py<PyAny>) {
        self.inspect_mem_read = Some(cb);
    }

    /// Set the inspect mem_write callback.
    ///
    /// The callback should have signature:
    /// `fn(state_id: int, when: str, addr: int, size: int, value_ast: object | None, endness: str) -> None`
    pub fn set_inspect_mem_write(&mut self, cb: Py<PyAny>) {
        self.inspect_mem_write = Some(cb);
    }

    /// Set the inspect-enabled bitmask. Bit N = `InspectEvent` variant N.
    /// Python aggregates registered breakpoints into this single value;
    /// VEX dispatch sites do a single AND test before any payload work.
    ///
    /// `inspect_enabled` is `Arc<AtomicU8>` so this write is visible to
    /// the cloned PythonCallbacks held by the Rust manager.
    #[pyo3(name = "set_inspect_enabled")]
    pub fn py_set_inspect_enabled(&self, mask: u8) {
        self.inspect_enabled
            .store(mask, std::sync::atomic::Ordering::Relaxed);
    }

    /// Read the inspect-enabled bitmask (Python-side, mostly for tests).
    #[pyo3(name = "get_inspect_enabled")]
    pub fn py_get_inspect_enabled(&self) -> u8 {
        self.inspect_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Test entry point: invoke the registered mem_read callback directly.
    /// Lets the marshalling round-trip be exercised before VEX dispatch
    /// sites are wired (uq4n.3). Returns whatever Python returned.
    #[pyo3(name = "call_inspect_mem_read")]
    #[pyo3(signature = (state_id, when, addr, size, value_ast, endness))]
    pub fn py_call_inspect_mem_read(
        &self,
        py: Python<'_>,
        state_id: i64,
        when: &str,
        addr: u64,
        size: u32,
        value_ast: Option<Py<PyAny>>,
        endness: &str,
    ) -> PyResult<()> {
        self.call_inspect_mem_read(py, state_id, when, addr, size, value_ast.as_ref(), endness)
    }

    /// Test entry point: invoke the registered mem_write callback directly.
    #[pyo3(name = "call_inspect_mem_write")]
    #[pyo3(signature = (state_id, when, addr, size, value_ast, endness))]
    pub fn py_call_inspect_mem_write(
        &self,
        py: Python<'_>,
        state_id: i64,
        when: &str,
        addr: u64,
        size: u32,
        value_ast: Option<Py<PyAny>>,
        endness: &str,
    ) -> PyResult<()> {
        self.call_inspect_mem_write(py, state_id, when, addr, size, value_ast.as_ref(), endness)
    }

    /// Check if all required callbacks are set.
    pub fn is_ready(&self) -> bool {
        self.memory_load.is_some() && self.memory_store.is_some() && self.lift_block.is_some()
    }

    /// GC traversal: visit each held Python callback so the cycle
    /// `mgr -> _callbacks -> bound method -> mgr` is GC-collectible.
    /// Without this, the manager (and its _state_cache, ~4030 angr pages
    /// per call in mma_howtouse) leaks permanently.
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        Self::traverse_fields(self, &visit)
    }

    /// GC clear: drop all Python callback refs. After clear, calls to any
    /// callback method will fail with "callback not set", but by then the
    /// containing object is being collected anyway.
    fn __clear__(&mut self) {
        self.clear_fields();
    }
}

impl PythonCallbacks {
    /// Visit every Py<PyAny> field. Used by __traverse__ on this type and
    /// by RustExplorationManager.__traverse__ which holds a cloned copy of
    /// PythonCallbacks (and so participates in the same cycle).
    pub fn traverse_fields(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for opt in [
            &self.memory_load,
            &self.memory_store,
            &self.memory_store_batch,
            &self.memory_load_batch,
            &self.memory_load_symbolic,
            &self.memory_store_symbolic,
            &self.on_hook,
            &self.on_syscall,
            &self.lift_block,
            &self.get_register,
            &self.put_register,
            &self.dirty_call,
            &self.fetch_page,
            &self.batch_fetch_pages,
            &self.sync_constraints,
            &self.memory_store_symbolic_value,
            &self.memory_store_symbolic_full,
            &self.memory_load_symbolic_full,
            &self.resolve_function,
            &self.inspect_mem_read,
            &self.inspect_mem_write,
        ] {
            if let Some(obj) = opt {
                visit.call(obj)?;
            }
        }
        Ok(())
    }

    /// Drop every Py<PyAny> field. Used by __clear__ on this type and on
    /// RustExplorationManager (which has a cloned copy in its `callbacks` field).
    pub fn clear_fields(&mut self) {
        self.memory_load = None;
        self.memory_store = None;
        self.memory_store_batch = None;
        self.memory_load_batch = None;
        self.memory_load_symbolic = None;
        self.memory_store_symbolic = None;
        self.on_hook = None;
        self.on_syscall = None;
        self.lift_block = None;
        self.get_register = None;
        self.put_register = None;
        self.dirty_call = None;
        self.fetch_page = None;
        self.batch_fetch_pages = None;
        self.sync_constraints = None;
        self.memory_store_symbolic_value = None;
        self.memory_store_symbolic_full = None;
        self.memory_load_symbolic_full = None;
        self.resolve_function = None;
        self.inspect_mem_read = None;
        self.inspect_mem_write = None;
        self.inspect_enabled
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Default for PythonCallbacks {
    fn default() -> Self {
        Self::new()
    }
}

impl PythonCallbacks {
    /// Fast O(1) check for whether an inspect event is enabled.
    /// Bit N = `crate::state::InspectEvent` variant N (MemRead=0, MemWrite=1, …).
    #[inline(always)]
    pub fn inspect_event_enabled(&self, event_bit: u8) -> bool {
        self.inspect_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
            & (1u8 << event_bit)
            != 0
    }

    /// Debug-only: read the raw bitmask. Used by eprintln traces.
    pub fn get_inspect_enabled_for_debug(&self) -> u8 {
        self.inspect_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Invoke the Python inspect mem_read callback.
    ///
    /// Caller is expected to gate this on `inspect_event_enabled(0)` for
    /// the common no-breakpoint case. Errors propagate so the engine can
    /// surface user-action failures rather than swallowing them.
    pub fn call_inspect_mem_read(
        &self,
        py: Python<'_>,
        state_id: i64,
        when: &str,
        addr: u64,
        size: u32,
        value_ast: Option<&Py<PyAny>>,
        endness: &str,
    ) -> PyResult<()> {
        let cb = match self.inspect_mem_read.as_ref() {
            Some(cb) => cb,
            None => return Ok(()),
        };
        let value_obj: Py<PyAny> = match value_ast {
            Some(v) => v.clone_ref(py),
            None => py.None(),
        };
        cb.call1(py, (state_id, when, addr, size, value_obj, endness))?;
        Ok(())
    }

    /// Invoke the Python inspect mem_write callback. See `call_inspect_mem_read`.
    pub fn call_inspect_mem_write(
        &self,
        py: Python<'_>,
        state_id: i64,
        when: &str,
        addr: u64,
        size: u32,
        value_ast: Option<&Py<PyAny>>,
        endness: &str,
    ) -> PyResult<()> {
        let cb = match self.inspect_mem_write.as_ref() {
            Some(cb) => cb,
            None => return Ok(()),
        };
        let value_obj: Py<PyAny> = match value_ast {
            Some(v) => v.clone_ref(py),
            None => py.None(),
        };
        cb.call1(py, (state_id, when, addr, size, value_obj, endness))?;
        Ok(())
    }

    /// Call the memory load callback.
    ///
    /// Returns (data_bytes, is_symbolic, symbolic_ast).
    pub fn call_memory_load(
        &self,
        py: Python<'_>,
        addr: u64,
        size: u32,
    ) -> PyResult<(Vec<u8>, bool, Option<Py<PyAny>>)> {
        let cb = self.memory_load.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("memory_load callback not set")
        })?;

        let result = cb.call1(py, (addr, size))?;
        let tuple = result.cast_bound::<pyo3::types::PyTuple>(py)?;

        // Extract (bytes, is_symbolic, symbolic_ast?)
        let data_obj = tuple.get_item(0)?;
        let data: Vec<u8> = data_obj.extract()?;
        let is_symbolic: bool = tuple.get_item(1)?.extract()?;

        let symbolic_ast = if tuple.len() > 2 {
            let ast_obj = tuple.get_item(2)?;
            if ast_obj.is_none() {
                None
            } else {
                Some(ast_obj.unbind())
            }
        } else {
            None
        };

        Ok((data, is_symbolic, symbolic_ast))
    }

    /// Call the memory store callback.
    pub fn call_memory_store(&self, py: Python<'_>, addr: u64, data: &[u8]) -> PyResult<()> {
        let cb = self.memory_store.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("memory_store callback not set")
        })?;

        let py_bytes = PyBytes::new(py, data);
        cb.call1(py, (addr, py_bytes))?;
        Ok(())
    }

    /// Call the batched memory store callback.
    ///
    /// This sends multiple stores in a single callback for efficiency.
    /// Falls back to individual stores if batch callback is not set.
    pub fn call_memory_store_batch(
        &self,
        py: Python<'_>,
        stores: &[(u64, Vec<u8>)],
    ) -> PyResult<()> {
        if stores.is_empty() {
            return Ok(());
        }

        // Try batch callback first
        if let Some(cb) = &self.memory_store_batch {
            // Convert stores to Python list of tuples
            let py_stores: Vec<(u64, Py<PyBytes>)> = stores
                .iter()
                .map(|(addr, data)| (*addr, PyBytes::new(py, data).unbind()))
                .collect();
            cb.call1(py, (py_stores,))?;
            return Ok(());
        }

        // Fallback: call individual stores
        for (addr, data) in stores {
            self.call_memory_store(py, *addr, data)?;
        }
        Ok(())
    }

    /// Call the batched memory load callback.
    ///
    /// This sends multiple load requests in a single callback for efficiency.
    /// Falls back to individual loads if batch callback is not set.
    ///
    /// Returns a vector of (data_bytes, is_symbolic, symbolic_ast) tuples,
    /// one for each load request.
    pub fn call_memory_load_batch(
        &self,
        py: Python<'_>,
        loads: &[(u64, u32)], // (address, size) pairs
    ) -> PyResult<Vec<(Vec<u8>, bool, Option<Py<PyAny>>)>> {
        if loads.is_empty() {
            return Ok(Vec::new());
        }

        // Try batch callback first
        if let Some(cb) = &self.memory_load_batch {
            // Convert loads to Python list of tuples
            let py_loads: Vec<(u64, u32)> = loads.to_vec();
            let result = cb.call1(py, (py_loads,))?;

            // Parse the result list
            let result_list = result.cast_bound::<pyo3::types::PyList>(py)?;
            let mut results = Vec::with_capacity(loads.len());

            for item in result_list.iter() {
                let tuple = item.cast::<pyo3::types::PyTuple>()?;

                // Extract (bytes, is_symbolic, symbolic_ast?)
                let data_obj = tuple.get_item(0)?;
                let data: Vec<u8> = data_obj.extract()?;
                let is_symbolic: bool = tuple.get_item(1)?.extract()?;

                let symbolic_ast = if tuple.len() > 2 {
                    let ast_obj = tuple.get_item(2)?;
                    if ast_obj.is_none() {
                        None
                    } else {
                        Some(ast_obj.unbind())
                    }
                } else {
                    None
                };

                results.push((data, is_symbolic, symbolic_ast));
            }

            return Ok(results);
        }

        // Fallback: call individual loads
        let mut results = Vec::with_capacity(loads.len());
        for &(addr, size) in loads {
            let (data, is_sym, ast) = self.call_memory_load(py, addr, size)?;
            results.push((data, is_sym, ast));
        }
        Ok(results)
    }

    /// Call the symbolic memory load callback.
    ///
    /// This is called when the address is symbolic and concretizes to multiple values.
    /// Returns the loaded value as a RustBV.
    pub fn call_memory_load_symbolic(
        &self,
        py: Python<'_>,
        addrs: &[u64],
        size: u32,
        addr_ast: &RustBV,
    ) -> PyResult<RustBV> {
        // If symbolic callback is set, use it
        if let Some(cb) = &self.memory_load_symbolic {
            let addrs_list: Vec<u64> = addrs.to_vec();
            // Convert addr_ast to Python representation
            // For now we pass the concrete addresses and let Python handle the ITE chain
            let result = cb.call1(py, (addrs_list, size, addr_ast.width()))?;

            // The callback should return bytes
            let bytes: Vec<u8> = result.extract(py)?;
            let width = (size * 8) as u32;
            let mut value: u128 = 0;
            for (i, &byte) in bytes.iter().enumerate() {
                if (i * 8) as u32 >= width {
                    break;
                }
                value |= (byte as u128) << (i * 8);
            }
            return Ok(RustBV::concrete(value, width));
        }

        // Fallback: load from first address only
        if let Some(first_addr) = addrs.first() {
            let (data, _is_symbolic, _ast) = self.call_memory_load(py, *first_addr, size)?;
            let width = (size * 8) as u32;
            let mut value: u128 = 0;
            for (i, &byte) in data.iter().enumerate() {
                if (i * 8) as u32 >= width {
                    break;
                }
                value |= (byte as u128) << (i * 8);
            }
            Ok(RustBV::concrete(value, width))
        } else {
            // No addresses - return zero
            Ok(RustBV::zero((size * 8) as u32))
        }
    }

    /// Call the symbolic memory store callback.
    ///
    /// This is called when the address is symbolic and concretizes to multiple values.
    /// The callback should perform conditional stores to each possible address.
    pub fn call_memory_store_symbolic(
        &self,
        py: Python<'_>,
        addrs: &[u64],
        data: &RustBV,
        addr_ast: &RustBV,
    ) -> PyResult<()> {
        use crate::claripy_bridge::rustbv_to_claripy;

        // If data is symbolic and we have the full symbolic callback, use it
        if data.is_symbolic() && self.memory_store_symbolic_full.is_some() {
            return self.call_memory_store_symbolic_full(py, addr_ast, data);
        }

        // If data is symbolic, try to use symbolic value callback for each address
        if data.is_symbolic() && self.memory_store_symbolic_value.is_some() {
            let claripy_mod = py.import("claripy")?;
            let data_ast = rustbv_to_claripy(py, data, &claripy_mod)?;
            let _addr_claripy = rustbv_to_claripy(py, addr_ast, &claripy_mod)?;

            // Use the symbolic value callback with claripy AST
            if let Some(cb) = &self.memory_store_symbolic_value {
                // For multiple addresses, we need conditional stores
                // The callback should handle creating ITE chains
                // Store to first address with the full expression
                if let Some(first_addr) = addrs.first() {
                    cb.call1(py, (*first_addr, data_ast))?;
                }
            }
            return Ok(());
        }

        // If symbolic callback is set, use it (concrete data case)
        if let Some(cb) = &self.memory_store_symbolic {
            let addrs_list: Vec<u64> = addrs.to_vec();
            let data_bytes = bv_to_bytes(data);
            let py_bytes = PyBytes::new(py, &data_bytes);
            cb.call1(py, (addrs_list, py_bytes, addr_ast.width()))?;
            return Ok(());
        }

        // Fallback: store to first address only (not ideal but maintains progress)
        if let Some(first_addr) = addrs.first() {
            // Even in fallback, try to preserve symbolic data
            if data.is_symbolic() && self.memory_store_symbolic_value.is_some() {
                self.call_memory_store_symbolic_value(py, *first_addr, data)?;
            } else {
                let data_bytes = bv_to_bytes(data);
                self.call_memory_store(py, *first_addr, &data_bytes)?;
            }
        }
        Ok(())
    }

    /// Call the hook execution callback.
    ///
    /// Returns the new PC after hook execution.
    pub fn call_on_hook(&self, py: Python<'_>, addr: u64) -> PyResult<u64> {
        let cb = self
            .on_hook
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("on_hook callback not set"))?;

        let result = cb.call1(py, (addr,))?;
        result.extract(py)
    }

    /// Call the syscall handling callback.
    pub fn call_on_syscall(&self, py: Python<'_>, num: u64) -> PyResult<()> {
        let cb = self.on_syscall.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("on_syscall callback not set")
        })?;

        cb.call1(py, (num,))?;
        Ok(())
    }

    /// Call the block lifting callback.
    ///
    /// Returns the IRSB as a JSON string.
    /// If `opt_level` is `Some`, passes it as the second positional arg.
    /// If `dirty_bytes` is `Some`, passes it as the third positional arg
    /// (Python callback uses these as `byte_string=` for SMC fresh-bytes lift).
    pub fn call_lift_block(
        &self,
        py: Python<'_>,
        addr: u64,
        opt_level: Option<i32>,
        dirty_bytes: Option<&[u8]>,
    ) -> PyResult<String> {
        let cb = self.lift_block.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("lift_block callback not set")
        })?;

        let result = match (opt_level, dirty_bytes) {
            (None, None) => cb.call1(py, (addr,))?,
            (Some(level), None) => cb.call1(py, (addr, level))?,
            (None, Some(bytes)) => {
                let py_bytes = PyBytes::new(py, bytes);
                cb.call1(py, (addr, py.None(), py_bytes))?
            }
            (Some(level), Some(bytes)) => {
                let py_bytes = PyBytes::new(py, bytes);
                cb.call1(py, (addr, level, py_bytes))?
            }
        };
        result.extract(py)
    }

    /// Call the register get callback.
    ///
    /// Returns (data_bytes, is_symbolic, symbolic_ast).
    pub fn call_get_register(
        &self,
        py: Python<'_>,
        offset: u32,
        size: u32,
    ) -> PyResult<(Vec<u8>, bool, Option<Py<PyAny>>)> {
        let cb = self.get_register.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("get_register callback not set")
        })?;

        let result = cb.call1(py, (offset, size))?;
        let tuple = result.cast_bound::<pyo3::types::PyTuple>(py)?;

        let data: Vec<u8> = tuple.get_item(0)?.extract()?;
        let is_symbolic: bool = tuple.get_item(1)?.extract()?;

        let symbolic_ast = if tuple.len() > 2 {
            let ast_obj = tuple.get_item(2)?;
            if ast_obj.is_none() {
                None
            } else {
                Some(ast_obj.unbind())
            }
        } else {
            None
        };

        Ok((data, is_symbolic, symbolic_ast))
    }

    /// Call the register put callback.
    pub fn call_put_register(&self, py: Python<'_>, offset: u32, data: &[u8]) -> PyResult<()> {
        let cb = self.put_register.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("put_register callback not set")
        })?;

        let py_bytes = PyBytes::new(py, data);
        cb.call1(py, (offset, py_bytes))?;
        Ok(())
    }

    /// Call the dirty call callback for VEX helper functions.
    ///
    /// This handles dirty calls like CPUID, RDTSC, x87 operations, etc.
    /// Returns (data_bytes, is_symbolic, symbolic_ast).
    pub fn call_dirty_call(
        &self,
        py: Python<'_>,
        name: &str,
        args: &[u64],
        ret_ty_bits: u32,
    ) -> PyResult<(Vec<u8>, bool, Option<Py<PyAny>>)> {
        let cb = self.dirty_call.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("dirty_call callback not set")
        })?;

        // Convert args to Python list
        let args_list: Vec<u64> = args.to_vec();

        let result = cb.call1(py, (name, args_list, ret_ty_bits))?;
        let tuple = result.cast_bound::<pyo3::types::PyTuple>(py)?;

        // Extract (bytes, is_symbolic, symbolic_ast?)
        let data_obj = tuple.get_item(0)?;
        let data: Vec<u8> = data_obj.extract()?;
        let is_symbolic: bool = tuple.get_item(1)?.extract()?;

        let symbolic_ast = if tuple.len() > 2 {
            let ast_obj = tuple.get_item(2)?;
            if ast_obj.is_none() {
                None
            } else {
                Some(ast_obj.unbind())
            }
        } else {
            None
        };

        Ok((data, is_symbolic, symbolic_ast))
    }

    /// Check if dirty call callback is available.
    pub fn has_dirty_call(&self) -> bool {
        self.dirty_call.is_some()
    }

    /// Check if fetch_page callback is available.
    pub fn has_fetch_page(&self) -> bool {
        self.fetch_page.is_some()
    }

    /// Call the page fetch callback to load a single 4KB page.
    ///
    /// Returns (page_data, permissions, is_mapped).
    /// - page_data: 4096 bytes of page content
    /// - permissions: permission bits (R=4, W=2, X=1)
    /// - is_mapped: whether the page exists in Python memory
    pub fn call_fetch_page(&self, py: Python<'_>, page_addr: u64) -> PyResult<(Vec<u8>, u8, bool)> {
        let cb = self.fetch_page.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("fetch_page callback not set")
        })?;

        let result = cb.call1(py, (page_addr,))?;
        let tuple = result.cast_bound::<pyo3::types::PyTuple>(py)?;

        let data: Vec<u8> = tuple.get_item(0)?.extract()?;
        let permissions: u8 = tuple.get_item(1)?.extract()?;
        let is_mapped: bool = tuple.get_item(2)?.extract()?;

        Ok((data, permissions, is_mapped))
    }

    /// Call the batched page fetch callback to load multiple 4KB pages.
    ///
    /// Returns a list of (page_data, permissions, is_mapped) for each page.
    pub fn call_batch_fetch_pages(
        &self,
        py: Python<'_>,
        page_addrs: &[u64],
    ) -> PyResult<Vec<(Vec<u8>, u8, bool)>> {
        if page_addrs.is_empty() {
            return Ok(Vec::new());
        }

        // Try batch callback first
        if let Some(cb) = &self.batch_fetch_pages {
            let addrs_list: Vec<u64> = page_addrs.to_vec();
            let result = cb.call1(py, (addrs_list,))?;

            let result_list = result.cast_bound::<pyo3::types::PyList>(py)?;
            let mut results = Vec::with_capacity(page_addrs.len());

            for item in result_list.iter() {
                let tuple = item.cast::<pyo3::types::PyTuple>()?;
                let data: Vec<u8> = tuple.get_item(0)?.extract()?;
                let permissions: u8 = tuple.get_item(1)?.extract()?;
                let is_mapped: bool = tuple.get_item(2)?.extract()?;
                results.push((data, permissions, is_mapped));
            }

            return Ok(results);
        }

        // Fallback: call individual fetches
        let mut results = Vec::with_capacity(page_addrs.len());
        for &page_addr in page_addrs {
            let (data, perms, mapped) = self.call_fetch_page(py, page_addr)?;
            results.push((data, perms, mapped));
        }
        Ok(results)
    }

    /// Sync accumulated constraints to Python's claripy solver.
    ///
    /// This should be called before falling back to Python for operations
    /// that depend on solver state (e.g., symbolic memory operations).
    ///
    /// # Arguments
    /// * `py` - Python GIL token
    /// * `constraints` - List of (description, width, concrete_value, handle_id) tuples
    ///
    /// # Returns
    /// Ok(()) on success, or error if callback fails.
    pub fn call_sync_constraints(
        &self,
        py: Python<'_>,
        constraints: &[(String, u32, u128, Option<u64>)],
    ) -> PyResult<()> {
        if constraints.is_empty() {
            return Ok(());
        }

        if let Some(cb) = &self.sync_constraints {
            // Convert to Python list of tuples
            let py_constraints: Vec<(String, u32, u128, Option<u64>)> = constraints.to_vec();
            cb.call1(py, (py_constraints,))?;
        }
        // If no callback is set, silently succeed - constraints will be lost
        // but this allows gradual adoption of the feature
        Ok(())
    }

    /// Store a symbolic value to memory with full expression tree preservation.
    ///
    /// This method converts the RustBV expression tree to a claripy AST and
    /// calls Python to store it. This preserves symbolic expressions like
    /// `x + 10 ^ 0x42` instead of losing them to zeros.
    ///
    /// # Arguments
    /// * `py` - Python GIL token
    /// * `addr` - The memory address to store to
    /// * `value` - The symbolic RustBV value with expression tree
    ///
    /// # Returns
    /// Ok(()) on success, or falls back to byte-based store if callback unavailable.
    pub fn call_memory_store_symbolic_value(
        &self,
        py: Python<'_>,
        addr: u64,
        value: &RustBV,
    ) -> PyResult<()> {
        use crate::claripy_bridge::rustbv_to_claripy;

        // If the symbolic value callback is set, use it
        if let Some(cb) = &self.memory_store_symbolic_value {
            // Import claripy module
            let claripy_mod = py.import("claripy")?;

            // Convert RustBV expression tree to claripy AST
            let ast = rustbv_to_claripy(py, value, &claripy_mod)?;
            cb.call1(py, (addr, ast))?;
            return Ok(());
        }

        // Fallback: use the standard memory_store with byte representation
        // This will lose symbolic information but maintains backward compatibility
        let data_bytes = bv_to_bytes(value);
        self.call_memory_store(py, addr, &data_bytes)
    }

    /// Check if symbolic value store callback is available.
    pub fn has_memory_store_symbolic_value(&self) -> bool {
        self.memory_store_symbolic_value.is_some()
    }

    /// Call the full symbolic store callback (symbolic address + symbolic value).
    /// Used when the address cannot be concretized to a single value or small set.
    pub fn call_memory_store_symbolic_full(
        &self,
        py: Python<'_>,
        addr_val: &RustBV,
        data_val: &RustBV,
    ) -> PyResult<()> {
        use crate::claripy_bridge::rustbv_to_claripy;

        if let Some(cb) = &self.memory_store_symbolic_full {
            let claripy_mod = py.import("claripy")?;
            let addr_ast = rustbv_to_claripy(py, addr_val, &claripy_mod)?;
            let data_ast = rustbv_to_claripy(py, data_val, &claripy_mod)?;
            cb.call1(py, (addr_ast, data_ast))?;
            return Ok(());
        }

        // Fallback: silently ignore (no proper fallback available)
        Ok(())
    }

    /// Check if full symbolic store callback is available.
    pub fn has_memory_store_symbolic_full(&self) -> bool {
        self.memory_store_symbolic_full.is_some()
    }

    /// Load from memory at a symbolic address (full AST delegation).
    ///
    /// This is called when the address range is too large to concretize.
    /// Python will use angr's memory model to handle the symbolic address,
    /// which may build ITE chains or use address concretization strategies.
    ///
    /// # Arguments
    /// * `py` - Python GIL token
    /// * `addr_val` - The symbolic address as a RustBV
    /// * `size` - Number of bytes to load
    ///
    /// # Returns
    /// The loaded claripy AST from Python's memory model.
    pub fn call_memory_load_symbolic_full(
        &self,
        py: Python<'_>,
        addr_val: &RustBV,
        size: u32,
    ) -> PyResult<Py<PyAny>> {
        use crate::claripy_bridge::rustbv_to_claripy;

        if let Some(cb) = &self.memory_load_symbolic_full {
            let claripy_mod = py.import("claripy")?;
            let addr_ast = rustbv_to_claripy(py, addr_val, &claripy_mod)?;
            return cb.call1(py, (addr_ast, size));
        }
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "memory_load_symbolic_full callback not set",
        ))
    }

    /// Check if full symbolic load callback is available.
    pub fn has_memory_load_symbolic_full(&self) -> bool {
        self.memory_load_symbolic_full.is_some()
    }

    /// Call the resolve_function callback to dynamically resolve unmodeled function calls.
    ///
    /// This is called when Rust encounters a CALL to an address that isn't hooked.
    /// Python can check its procedure registries and return procedure info if available.
    ///
    /// # Arguments
    /// * `py` - Python GIL token
    /// * `addr` - Address of the unmodeled function
    /// * `symbol_name` - Symbol name if known from binary, None otherwise
    ///
    /// # Returns
    /// * `Ok(Some((name, num_args, no_return)))` - Function resolved, register and retry
    /// * `Ok(None)` - Function cannot be resolved, deadend the state
    /// * `Err(...)` - Callback error
    pub fn call_resolve_function(
        &self,
        py: Python<'_>,
        addr: u64,
        symbol_name: Option<&str>,
    ) -> PyResult<Option<(String, usize, bool)>> {
        let cb = self.resolve_function.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("resolve_function callback not set")
        })?;

        let result = cb.call1(py, (addr, symbol_name))?;

        // Check if result is None
        if result.is_none(py) {
            return Ok(None);
        }

        // Extract tuple (name, num_args, no_return)
        let tuple = result.cast_bound::<pyo3::types::PyTuple>(py)?;
        if tuple.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "resolve_function must return (name, num_args, no_return) or None",
            ));
        }

        let name: String = tuple.get_item(0)?.extract()?;
        let num_args: usize = tuple.get_item(1)?.extract()?;
        let no_return: bool = tuple.get_item(2)?.extract()?;

        Ok(Some((name, num_args, no_return)))
    }

    /// Check if resolve_function callback is available.
    pub fn has_resolve_function(&self) -> bool {
        self.resolve_function.is_some()
    }
}

/// Execution event returned to Python from run_loop.
#[pyclass]
#[derive(Debug, Clone)]
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
        match result {
            RunResult::MaxBlocks { pc } => LoopExecutionEvent {
                event_type: "max_blocks".to_string(),
                pc: Some(pc),
                addr: None,
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: None,
                error: None,
                blocks_executed,
                deferred_forks,
                push_level,
                simprocedure_name: None,
                simprocedure_num_args: None,
                simprocedure_return_addr: None,
                jump_targets: None,
                jump_condition_id: None,
                unconstrained_min: None,
                unconstrained_max: None,
                unconstrained_limit: None,
                unmodeled_call_addr: None,
                unmodeled_call_return_addr: None,
                unmodeled_call_symbol: None,
            },
            RunResult::Hook { addr } => LoopExecutionEvent {
                event_type: "hook".to_string(),
                pc: Some(addr),
                addr: Some(addr),
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: None,
                error: None,
                blocks_executed,
                deferred_forks,
                push_level,
                simprocedure_name: None,
                simprocedure_num_args: None,
                simprocedure_return_addr: None,
                jump_targets: None,
                jump_condition_id: None,
                unconstrained_min: None,
                unconstrained_max: None,
                unconstrained_limit: None,
                unmodeled_call_addr: None,
                unmodeled_call_return_addr: None,
                unmodeled_call_symbol: None,
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
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: Some("Ijk_Call".to_string()),
                error: None,
                blocks_executed,
                deferred_forks,
                push_level,
                simprocedure_name: Some(name),
                simprocedure_num_args: Some(num_args),
                simprocedure_return_addr: if return_addr != 0 {
                    Some(return_addr)
                } else {
                    None
                },
                jump_targets: None,
                jump_condition_id: None,
                unconstrained_min: None,
                unconstrained_max: None,
                unconstrained_limit: None,
                unmodeled_call_addr: None,
                unmodeled_call_return_addr: None,
                unmodeled_call_symbol: None,
            },
            RunResult::Syscall { num, pc } => LoopExecutionEvent {
                event_type: "syscall".to_string(),
                pc: Some(pc),
                addr: None,
                syscall_num: num,
                true_target: None,
                false_target: None,
                jumpkind: Some("Ijk_Sys_syscall".to_string()),
                error: None,
                blocks_executed,
                deferred_forks,
                push_level,
                simprocedure_name: None,
                simprocedure_num_args: None,
                simprocedure_return_addr: None,
                jump_targets: None,
                jump_condition_id: None,
                unconstrained_min: None,
                unconstrained_max: None,
                unconstrained_limit: None,
                unmodeled_call_addr: None,
                unmodeled_call_return_addr: None,
                unmodeled_call_symbol: None,
            },
            RunResult::SymbolicBranch {
                true_target,
                false_target,
                ..
            } => LoopExecutionEvent {
                event_type: "symbolic_branch".to_string(),
                pc: None,
                addr: None,
                syscall_num: None,
                true_target: Some(true_target),
                false_target: Some(false_target),
                jumpkind: None,
                error: None,
                blocks_executed,
                deferred_forks,
                push_level,
                simprocedure_name: None,
                simprocedure_num_args: None,
                simprocedure_return_addr: None,
                jump_targets: None,
                jump_condition_id: None,
                unconstrained_min: None,
                unconstrained_max: None,
                unconstrained_limit: None,
                unmodeled_call_addr: None,
                unmodeled_call_return_addr: None,
                unmodeled_call_symbol: None,
            },
            RunResult::BlockEnd {
                next_addr,
                jumpkind,
            } => LoopExecutionEvent {
                event_type: "block_end".to_string(),
                pc: Some(next_addr),
                addr: Some(next_addr),
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: Some(jumpkind),
                error: None,
                blocks_executed,
                deferred_forks,
                push_level,
                simprocedure_name: None,
                simprocedure_num_args: None,
                simprocedure_return_addr: None,
                jump_targets: None,
                jump_condition_id: None,
                unconstrained_min: None,
                unconstrained_max: None,
                unconstrained_limit: None,
                unmodeled_call_addr: None,
                unmodeled_call_return_addr: None,
                unmodeled_call_symbol: None,
            },
            RunResult::Error { message, addr } => LoopExecutionEvent {
                event_type: "error".to_string(),
                pc: Some(addr),
                addr: Some(addr),
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: None,
                error: Some(message),
                blocks_executed,
                deferred_forks,
                push_level,
                simprocedure_name: None,
                simprocedure_num_args: None,
                simprocedure_return_addr: None,
                jump_targets: None,
                jump_condition_id: None,
                unconstrained_min: None,
                unconstrained_max: None,
                unconstrained_limit: None,
                unmodeled_call_addr: None,
                unmodeled_call_return_addr: None,
                unmodeled_call_symbol: None,
            },
            RunResult::NeedLift { addr } => LoopExecutionEvent {
                event_type: "need_lift".to_string(),
                pc: Some(addr),
                addr: Some(addr),
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: None,
                error: None,
                blocks_executed,
                deferred_forks,
                push_level,
                simprocedure_name: None,
                simprocedure_num_args: None,
                simprocedure_return_addr: None,
                jump_targets: None,
                jump_condition_id: None,
                unconstrained_min: None,
                unconstrained_max: None,
                unconstrained_limit: None,
                unmodeled_call_addr: None,
                unmodeled_call_return_addr: None,
                unmodeled_call_symbol: None,
            },
            RunResult::MaxDeferredForks { pc } => LoopExecutionEvent {
                event_type: "max_deferred_forks".to_string(),
                pc: Some(pc),
                addr: None,
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: None,
                error: None,
                blocks_executed,
                deferred_forks,
                push_level,
                simprocedure_name: None,
                simprocedure_num_args: None,
                simprocedure_return_addr: None,
                jump_targets: None,
                jump_condition_id: None,
                unconstrained_min: None,
                unconstrained_max: None,
                unconstrained_limit: None,
                unmodeled_call_addr: None,
                unmodeled_call_return_addr: None,
                unmodeled_call_symbol: None,
            },
            RunResult::SymbolicJumpTarget {
                targets,
                condition_id,
                jumpkind,
            } => LoopExecutionEvent {
                event_type: "symbolic_jump_target".to_string(),
                pc: targets.first().copied(),
                addr: None,
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: Some(jumpkind),
                error: None,
                blocks_executed,
                deferred_forks,
                push_level,
                simprocedure_name: None,
                simprocedure_num_args: None,
                simprocedure_return_addr: None,
                jump_targets: Some(targets),
                jump_condition_id: Some(condition_id),
                unconstrained_min: None,
                unconstrained_max: None,
                unconstrained_limit: None,
                unmodeled_call_addr: None,
                unmodeled_call_return_addr: None,
                unmodeled_call_symbol: None,
            },
            RunResult::UnconstrainedJump {
                min_target,
                max_target,
                limit,
                jumpkind,
            } => LoopExecutionEvent {
                event_type: "unconstrained_jump".to_string(),
                pc: None,
                addr: None,
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: Some(jumpkind),
                error: None,
                blocks_executed,
                deferred_forks,
                push_level,
                simprocedure_name: None,
                simprocedure_num_args: None,
                simprocedure_return_addr: None,
                jump_targets: None,
                jump_condition_id: None,
                unconstrained_min: Some(min_target),
                unconstrained_max: Some(max_target),
                unconstrained_limit: Some(limit),
                unmodeled_call_addr: None,
                unmodeled_call_return_addr: None,
                unmodeled_call_symbol: None,
            },
            RunResult::UnmodeledCall {
                addr,
                return_addr,
                symbol_name,
            } => LoopExecutionEvent {
                event_type: "unmodeled_call".to_string(),
                pc: Some(addr),
                addr: Some(addr),
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: Some("Ijk_Call".to_string()),
                error: None,
                blocks_executed,
                deferred_forks,
                push_level,
                simprocedure_name: None,
                simprocedure_num_args: None,
                simprocedure_return_addr: None,
                jump_targets: None,
                jump_condition_id: None,
                unconstrained_min: None,
                unconstrained_max: None,
                unconstrained_limit: None,
                unmodeled_call_addr: Some(addr),
                unmodeled_call_return_addr: Some(return_addr),
                unmodeled_call_symbol: symbol_name,
            },
            RunResult::NeedPythonVEX { addr, reason } => LoopExecutionEvent {
                event_type: "python_vex_fallback".to_string(),
                pc: Some(addr),
                addr: Some(addr),
                syscall_num: None,
                true_target: None,
                false_target: None,
                jumpkind: None,
                error: Some(reason),
                blocks_executed,
                deferred_forks,
                push_level,
                simprocedure_name: None,
                simprocedure_num_args: None,
                simprocedure_return_addr: None,
                jump_targets: None,
                jump_condition_id: None,
                unconstrained_min: None,
                unconstrained_max: None,
                unconstrained_limit: None,
                unmodeled_call_addr: None,
                unmodeled_call_return_addr: None,
                unmodeled_call_symbol: None,
            },
        }
    }

    /// Create an event from a run result (backward compatibility, no deferred forks).
    pub fn from_run_result(result: RunResult, blocks_executed: u32) -> Self {
        Self::from_run_result_with_forks(result, blocks_executed, Vec::new(), 0)
    }
}

/// Thread-safe wrapper for Python callbacks.
///
/// This allows the callbacks to be shared across interpreter instances
/// during a single execution loop.
pub struct CallbacksRef {
    inner: Arc<PythonCallbacks>,
}

impl CallbacksRef {
    pub fn new(callbacks: PythonCallbacks) -> Self {
        CallbacksRef {
            inner: Arc::new(callbacks),
        }
    }

    pub fn get(&self) -> &PythonCallbacks {
        &self.inner
    }
}

impl Clone for CallbacksRef {
    fn clone(&self) -> Self {
        CallbacksRef {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Convert a RustBV to bytes (little-endian).
fn bv_to_bytes(bv: &RustBV) -> Vec<u8> {
    let width = bv.width();
    let num_bytes = ((width + 7) / 8) as usize;

    if let Some(value) = bv.as_u128() {
        let mut bytes = vec![0u8; num_bytes];
        for i in 0..num_bytes {
            bytes[i] = (value >> (i * 8)) as u8;
        }
        bytes
    } else {
        // For symbolic values, return zeros (the callback will handle it)
        vec![0u8; num_bytes]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_callbacks_creation() {
        pyo3::prepare_freethreaded_python();
        let callbacks = PythonCallbacks::new();
        assert!(!callbacks.is_ready());
    }

    #[test]
    fn test_loop_execution_event() {
        let event = LoopExecutionEvent::from_run_result(
            RunResult::BlockEnd {
                next_addr: 0x1000,
                jumpkind: "Ijk_Boring".to_string(),
            },
            5,
        );
        assert_eq!(event.event_type, "block_end");
        assert_eq!(event.pc, Some(0x1000));
        assert_eq!(event.blocks_executed, 5);
    }
}
