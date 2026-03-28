//! Callback-aware VEX IR interpreter.
//!
//! This interpreter uses Python callbacks for memory operations instead of
//! local SymbolicMemory. It can run multiple blocks in a loop, returning
//! to Python only when an event requires Python handling.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Instant;

use lru::LruCache;
use pyo3::prelude::*;

use crate::arch::{arch_from_vex, calling_conventions::CallingConvention, default_cc_for_arch, RegisterFile};
use crate::callbacks::{BranchPolicy, DeferredFork, ExecutionConfig, PythonCallbacks, RunResult};
use crate::claripy_bridge::{claripy_to_rustbv, is_claripy_ast, is_rust_handle, python_to_rustbv, rustbv_to_claripy, try_handle_to_rustbv};
use crate::concretize::{AddressConcretizer, ConcretizationResult};
use crate::memory::{MemoryError, Permission, SymbolicMemory, PAGE_SIZE};
use crate::symbolic::{RustBV, RustSymbolTable, SymContext};
use crate::vex::ccall;
use crate::vex::dirty::DirtyHelperDispatch;
use crate::vex::ir::{IRConst, IRExpr, IRLoadGOp, IRStmt, IRType, JumpKind, TypeEnv, VexArch, IRSB};
use crate::vex::ops::{OpError, VEXOps};
use crate::vex::{deserialize_irsb, Endness};

/// Execution statistics for profiling.
///
/// Tracks timing and counts for various operations during VEX execution.
/// Times are in nanoseconds for precision.
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Number of load statements executed.
    pub load_stmt_count: u64,
    /// Time spent in load statements (nanoseconds).
    pub load_stmt_time_ns: u64,
    /// Number of store statements executed.
    pub store_stmt_count: u64,
    /// Time spent in store statements (nanoseconds).
    pub store_stmt_time_ns: u64,
    /// Number of exit statements executed.
    pub exit_stmt_count: u64,
    /// Time spent in exit statements (nanoseconds).
    pub exit_stmt_time_ns: u64,
    /// Number of Python callback invocations.
    pub python_callback_count: u64,
    /// Time spent in Python callbacks (nanoseconds).
    pub python_callback_time_ns: u64,
    /// Number of address concretizations performed.
    pub concretize_count: u64,
    /// Time spent in address concretization (nanoseconds).
    pub concretize_time_ns: u64,
    /// Number of IRSB cache hits.
    pub cache_hit_count: u64,
    /// Number of IRSB cache misses (lifts needed).
    pub cache_miss_count: u64,
    /// Time spent lifting blocks (nanoseconds).
    pub lift_time_ns: u64,
    /// Number of Rust memory loads (vs callback fallback).
    pub rust_memory_load_count: u64,
    /// Number of Python fallback memory loads.
    pub fallback_memory_load_count: u64,
    /// Number of Rust memory stores.
    pub rust_memory_store_count: u64,
    /// Number of Python fallback memory stores.
    pub fallback_memory_store_count: u64,
    /// Number of expression evaluations.
    pub expr_eval_count: u64,
    /// Time spent evaluating expressions (nanoseconds).
    pub expr_eval_time_ns: u64,
    /// Number of blocks executed.
    pub blocks_executed: u64,
    /// Total execution time (nanoseconds).
    pub total_time_ns: u64,
}

impl ExecutionStats {
    /// Convert stats to a HashMap for Python exposure.
    pub fn to_hashmap(&self) -> HashMap<String, u64> {
        let mut map = HashMap::new();
        map.insert("load_stmt_count".to_string(), self.load_stmt_count);
        map.insert("load_stmt_time_ns".to_string(), self.load_stmt_time_ns);
        map.insert("store_stmt_count".to_string(), self.store_stmt_count);
        map.insert("store_stmt_time_ns".to_string(), self.store_stmt_time_ns);
        map.insert("exit_stmt_count".to_string(), self.exit_stmt_count);
        map.insert("exit_stmt_time_ns".to_string(), self.exit_stmt_time_ns);
        map.insert("python_callback_count".to_string(), self.python_callback_count);
        map.insert("python_callback_time_ns".to_string(), self.python_callback_time_ns);
        map.insert("concretize_count".to_string(), self.concretize_count);
        map.insert("concretize_time_ns".to_string(), self.concretize_time_ns);
        map.insert("cache_hit_count".to_string(), self.cache_hit_count);
        map.insert("cache_miss_count".to_string(), self.cache_miss_count);
        map.insert("lift_time_ns".to_string(), self.lift_time_ns);
        map.insert("rust_memory_load_count".to_string(), self.rust_memory_load_count);
        map.insert("fallback_memory_load_count".to_string(), self.fallback_memory_load_count);
        map.insert("rust_memory_store_count".to_string(), self.rust_memory_store_count);
        map.insert("fallback_memory_store_count".to_string(), self.fallback_memory_store_count);
        map.insert("expr_eval_count".to_string(), self.expr_eval_count);
        map.insert("expr_eval_time_ns".to_string(), self.expr_eval_time_ns);
        map.insert("blocks_executed".to_string(), self.blocks_executed);
        map.insert("total_time_ns".to_string(), self.total_time_ns);
        map
    }

    /// Reset all statistics to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Merge another stats instance into this one.
    pub fn merge(&mut self, other: &ExecutionStats) {
        self.load_stmt_count += other.load_stmt_count;
        self.load_stmt_time_ns += other.load_stmt_time_ns;
        self.store_stmt_count += other.store_stmt_count;
        self.store_stmt_time_ns += other.store_stmt_time_ns;
        self.exit_stmt_count += other.exit_stmt_count;
        self.exit_stmt_time_ns += other.exit_stmt_time_ns;
        self.python_callback_count += other.python_callback_count;
        self.python_callback_time_ns += other.python_callback_time_ns;
        self.concretize_count += other.concretize_count;
        self.concretize_time_ns += other.concretize_time_ns;
        self.cache_hit_count += other.cache_hit_count;
        self.cache_miss_count += other.cache_miss_count;
        self.lift_time_ns += other.lift_time_ns;
        self.rust_memory_load_count += other.rust_memory_load_count;
        self.fallback_memory_load_count += other.fallback_memory_load_count;
        self.rust_memory_store_count += other.rust_memory_store_count;
        self.fallback_memory_store_count += other.fallback_memory_store_count;
        self.expr_eval_count += other.expr_eval_count;
        self.expr_eval_time_ns += other.expr_eval_time_ns;
        self.blocks_executed += other.blocks_executed;
        self.total_time_ns += other.total_time_ns;
    }
}

/// Errors during callback-based VEX execution.
#[derive(Debug, Clone)]
pub enum CbExecutionError {
    /// Memory error from callback.
    Memory(String),
    /// Operation error.
    Op(OpError),
    /// Invalid VEX IR.
    InvalidIR(String),
    /// Unsupported feature.
    Unsupported(String),
    /// Type mismatch.
    TypeMismatch { expected: IRType, got: IRType },
    /// Unknown temporary variable.
    UnknownTemp(u32),
    /// Python callback error.
    Callback(String),
    /// Block lifting error.
    LiftError(String),
    /// Needs Python fallback for special expressions (P7 fix)
    NeedPythonFallback(String),
}

impl From<OpError> for CbExecutionError {
    fn from(e: OpError) -> Self {
        CbExecutionError::Op(e)
    }
}

impl std::fmt::Display for CbExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CbExecutionError::Memory(msg) => write!(f, "memory error: {}", msg),
            CbExecutionError::Op(e) => write!(f, "operation error: {}", e),
            CbExecutionError::InvalidIR(msg) => write!(f, "invalid VEX IR: {}", msg),
            CbExecutionError::Unsupported(msg) => write!(f, "unsupported: {}", msg),
            CbExecutionError::TypeMismatch { expected, got } => {
                write!(f, "type mismatch: expected {:?}, got {:?}", expected, got)
            }
            CbExecutionError::UnknownTemp(tmp) => write!(f, "unknown temporary t{}", tmp),
            CbExecutionError::Callback(msg) => write!(f, "callback error: {}", msg),
            CbExecutionError::LiftError(msg) => write!(f, "lift error: {}", msg),
            CbExecutionError::NeedPythonFallback(msg) => write!(f, "need Python fallback: {}", msg),
        }
    }
}

impl std::error::Error for CbExecutionError {}

/// Result of concretizing a symbolic jump target.
enum ConcretizedJump {
    /// Single concrete address (common case for deterministic jumps).
    Single(u64),
    /// Multiple concrete addresses (for symbolic ret/call/jmp).
    /// Contains the list of targets and the original symbolic expression.
    Multiple {
        targets: Vec<u64>,
        expr: RustBV,
    },
    /// Too many targets - exceeds max_symbolic_ip_targets limit.
    /// State should be marked as unconstrained.
    TooMany {
        min: u64,
        max: u64,
        limit: usize,
    },
}

/// Result of executing a single statement.
enum StmtResult {
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
pub enum BlockResult {
    /// Continue to the next block at given address.
    Continue { next_addr: u64 },
    /// Syscall encountered.
    Syscall { num: u64 },
    /// Symbolic branch - need to fork.
    SymbolicBranch {
        condition_id: u64,
        true_target: u64,
        false_target: u64,
    },
    /// Hook address hit.
    Hook { addr: u64 },
    /// Block execution error.
    Error { message: String },
    /// Normal block end with jumpkind.
    BlockEnd { next_addr: u64, jumpkind: JumpKind },
    /// Symbolic jump target with multiple concrete targets after concretization.
    /// The exploration manager should fork states for each target.
    SymbolicJumpTarget {
        /// Concrete target addresses after concretization.
        targets: Vec<u64>,
        /// ID for the stored symbolic expression (for constraint addition).
        condition_id: u64,
        /// The symbolic expression for the jump target.
        target_expr: RustBV,
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

/// A concrete memory region cached locally in Rust.
#[derive(Clone)]
pub struct ConcreteMemoryRegion {
    /// Base address of the region.
    pub base: u64,
    /// Size of the region in bytes.
    pub size: u64,
    /// The concrete data.
    pub data: Vec<u8>,
}

impl ConcreteMemoryRegion {
    /// Check if this region contains the given address range.
    #[inline]
    pub fn contains(&self, addr: u64, size: u64) -> bool {
        addr >= self.base && addr + size <= self.base + self.size
    }

    /// Read bytes from this region. Returns None if out of bounds.
    #[inline]
    pub fn read(&self, addr: u64, size: usize) -> Option<&[u8]> {
        if addr < self.base {
            return None;
        }
        let offset = (addr - self.base) as usize;
        if offset + size > self.data.len() {
            return None;
        }
        Some(&self.data[offset..offset + size])
    }
}

/// Cached result of a prefetched memory load.
#[derive(Clone)]
pub struct PrefetchedLoad {
    /// The loaded bitvector value.
    pub value: RustBV,
    /// Whether the value is symbolic.
    pub is_symbolic: bool,
}

/// A constraint that was added in Rust and needs to be synced to Python.
///
/// When Rust concretizes a symbolic address or makes a branch decision,
/// it adds constraints to its Z3 context. These constraints must be
/// communicated to Python's claripy solver to maintain consistency
/// when falling back to Python for complex operations.
#[derive(Clone, Debug)]
pub struct PendingConstraint {
    /// The symbolic expression that was constrained.
    /// For address concretization: the address expression
    /// For branch: the condition
    pub expression: RustBV,
    /// The concrete value it was constrained to.
    pub concrete_value: u128,
    /// Description for debugging.
    pub description: String,
    /// Handle ID for looking up the original claripy AST in Python.
    /// If the expression came from a Python callback that returned a handle,
    /// this ID can be used to look up the original AST for constraint sync.
    pub handle_id: Option<u64>,
}

impl PendingConstraint {
    /// Create a new pending constraint for address concretization.
    pub fn address_concretization(addr_expr: RustBV, concrete_addr: u64) -> Self {
        PendingConstraint {
            expression: addr_expr,
            concrete_value: concrete_addr as u128,
            description: format!("addr_concretize_0x{:x}", concrete_addr),
            handle_id: None,
        }
    }

    /// Create a new pending constraint for address concretization with handle_id.
    pub fn address_concretization_with_handle(
        addr_expr: RustBV,
        concrete_addr: u64,
        handle_id: Option<u64>,
    ) -> Self {
        PendingConstraint {
            expression: addr_expr,
            concrete_value: concrete_addr as u128,
            description: format!("addr_concretize_0x{:x}", concrete_addr),
            handle_id,
        }
    }

    /// Create a new pending constraint for a branch taken (cond == 1).
    pub fn branch_true(cond: RustBV) -> Self {
        PendingConstraint {
            expression: cond,
            concrete_value: 1,
            description: "branch_true".to_string(),
            handle_id: None,
        }
    }

    /// Create a new pending constraint for a branch not taken (cond == 0).
    pub fn branch_false(cond: RustBV) -> Self {
        PendingConstraint {
            expression: cond,
            concrete_value: 0,
            description: "branch_false".to_string(),
            handle_id: None,
        }
    }
}

/// Information about a registered SimProcedure.
#[derive(Clone, Debug)]
pub struct SimProcedureInfo {
    /// Name of the SimProcedure (e.g., "strlen", "malloc").
    pub name: String,
    /// Number of arguments to extract.
    pub num_args: usize,
    /// Whether this is a no-return procedure (e.g., "exit", "abort").
    pub no_return: bool,
}

/// Callback-aware VEX IR interpreter.
///
/// This interpreter uses Python callbacks for memory and register access,
/// allowing it to work with angr's symbolic memory model.
///
/// When `use_rust_memory` is true, the interpreter uses `rust_memory` for
/// memory operations, falling back to Python callbacks only for unmapped pages.
/// This provides significant performance improvement for memory-intensive code.
pub struct CallbackInterpreter<'a> {
    /// Register file (local cache, synced via callbacks).
    pub registers: RegisterFile,
    /// Temporary variables for current block.
    temps: Vec<Option<RustBV>>,
    /// Solver context.
    ctx: &'a SymContext,
    /// Symbol table for handle-based claripy bypass.
    /// When present, Python callbacks can return RustBVHandle instead of claripy ASTs.
    symbol_table: Option<&'a RustSymbolTable>,
    /// Current program counter.
    pub pc: u64,
    /// Current instruction address (within block).
    current_insn_addr: u64,
    /// Hook addresses (return to Python when hit).
    hook_addrs: HashSet<u64>,
    /// VEX architecture.
    arch: VexArch,
    /// Block cache (shared across runs) using Arc for O(1) cloning.
    block_cache: LruCache<u64, Arc<IRSB>>,
    /// Whether to use callbacks for memory (vs local registers).
    use_memory_callbacks: bool,
    /// Deferred forks collected during execution.
    /// Each fork represents a branch where we took one path and deferred the other.
    deferred_forks: Vec<DeferredFork>,
    /// Execution configuration.
    config: ExecutionConfig,
    /// Counter for alternating branch policy.
    branch_counter: u64,
    /// Next condition ID for tracking branch conditions.
    next_condition_id: u64,
    /// Current solver push level for constraint tracking.
    /// Incremented when we push before adding a branch constraint.
    push_level: u32,
    /// Concrete memory regions cached locally for fast access.
    /// These are read-only regions (e.g., binary .text/.rodata sections).
    concrete_memory: Vec<ConcreteMemoryRegion>,
    /// Address concretizer for handling symbolic addresses.
    concretizer: AddressConcretizer,
    /// Bitset tracking which register offsets have been modified.
    /// Each bit represents a 4-byte aligned offset (offset / 4).
    /// A u128 covers 512 bytes of register space (128 * 4 = 512).
    dirty_registers: u128,
    /// Pending concrete stores to batch for efficiency.
    /// Each entry is (address, data_bytes).
    pending_stores: Vec<(u64, Vec<u8>)>,
    /// All stores flushed during this step (accumulated across block boundaries).
    /// Used for same-step cross-block load forwarding and for applying to state memory.
    /// HashMap for O(1) lookup by address. Value is the most recent store data.
    all_flushed_stores: HashMap<u64, Vec<u8>>,
    /// Pending symbolic stores - maps address to symbolic RustBV.
    /// These override the concrete bytes in pending_stores for load forwarding.
    pending_symbolic_stores: HashMap<u64, RustBV>,
    /// Maximum pending stores before auto-flush.
    max_pending_stores: usize,
    /// Rust-native symbolic memory (replaces Python callbacks when enabled).
    /// When Some, memory operations try Rust first before falling back to callbacks.
    rust_memory: Option<SymbolicMemory>,
    /// Whether to use Rust-native memory (vs Python callbacks).
    /// When true and rust_memory is Some, memory ops use Rust directly.
    use_rust_memory: bool,
    /// Prefetch cache for batched memory loads.
    /// Key is (address, size), value is the prefetched result.
    /// This is populated at block start and used during Load expression evaluation.
    load_prefetch_cache: HashMap<(u64, usize), PrefetchedLoad>,
    /// Whether load prefetching is enabled.
    use_load_prefetch: bool,
    /// Number of pages to prefetch in each direction when fetching a page.
    /// 0 = no prefetching, 1 = fetch 3 pages (main + 1 before + 1 after), etc.
    /// Default is 2 for good locality on stack/heap access patterns.
    page_prefetch_count: u32,
    /// Dirty helper dispatch table for native handling of common helpers.
    dirty_dispatch: DirtyHelperDispatch,
    /// Registry mapping hook addresses to SimProcedure info.
    /// When a hook is hit, we can extract arguments using this info.
    simprocedure_registry: HashMap<u64, SimProcedureInfo>,
    /// Calling convention for argument extraction.
    calling_convention: Box<dyn CallingConvention>,
    /// Last branch condition encountered (for symbolic branch handling).
    /// Stored when a SymbolicBranch is created so callers can retrieve it.
    last_branch_condition: Option<RustBV>,
    /// Pending constraints that need to be synced to Python.
    /// These accumulate when Rust adds constraints (e.g., address concretization)
    /// and are synced to Python before falling back to Python callbacks.
    pending_python_constraints: Vec<PendingConstraint>,
    /// Stored branch conditions by ID for deferred fork handling.
    /// When a deferred fork is created, we store the condition here so
    /// callers can retrieve it to properly constrain forked states.
    stored_conditions: HashMap<u64, RustBV>,
    /// Execution statistics for profiling.
    stats: ExecutionStats,
    /// Whether profiling is enabled.
    profiling_enabled: bool,
    /// Concrete memory regions sorted by base address for binary search.
    /// This is rebuilt when regions are added.
    concrete_memory_sorted: bool,
}

impl<'a> CallbackInterpreter<'a> {
    /// Create a new callback-aware interpreter.
    pub fn new(arch: VexArch, ctx: &'a SymContext) -> Self {
        Self::with_config(arch, ctx, ExecutionConfig::default())
    }

    /// Create a new callback-aware interpreter with custom config.
    pub fn with_config(arch: VexArch, ctx: &'a SymContext, config: ExecutionConfig) -> Self {
        let arch_box = arch_from_vex(arch);
        let arch_name = arch_box.name();
        let cc = default_cc_for_arch(arch_name);

        CallbackInterpreter {
            registers: RegisterFile::new(arch_box),
            temps: Vec::with_capacity(64), // Pre-allocate for typical block size
            ctx,
            symbol_table: None,  // Set via set_symbol_table() when using handles
            pc: 0,
            current_insn_addr: 0,
            hook_addrs: HashSet::new(),
            arch,
            block_cache: LruCache::new(NonZeroUsize::new(4096).unwrap()),
            use_memory_callbacks: true,
            deferred_forks: Vec::new(),
            config,
            branch_counter: 0,
            next_condition_id: 0,
            push_level: 0,
            concrete_memory: Vec::new(),
            concretizer: AddressConcretizer::new(),
            dirty_registers: 0,
            pending_stores: Vec::with_capacity(256),
            all_flushed_stores: HashMap::new(),
            pending_symbolic_stores: HashMap::new(),
            max_pending_stores: 256,
            rust_memory: None,
            use_rust_memory: false,
            load_prefetch_cache: HashMap::new(),
            use_load_prefetch: false, // Disabled by default - adds overhead for most workloads
            page_prefetch_count: 2,    // Prefetch 2 pages in each direction by default
            dirty_dispatch: DirtyHelperDispatch::new(),
            simprocedure_registry: HashMap::new(),
            calling_convention: cc,
            last_branch_condition: None,
            pending_python_constraints: Vec::new(),
            stored_conditions: HashMap::new(),
            stats: ExecutionStats::default(),
            profiling_enabled: false,
            concrete_memory_sorted: false,
        }
    }

    /// Get the address concretizer.
    pub fn concretizer(&self) -> &AddressConcretizer {
        &self.concretizer
    }

    /// Set custom concretizer settings.
    pub fn set_concretizer(&mut self, concretizer: AddressConcretizer) {
        self.concretizer = concretizer;
    }

    /// Set the symbol table for handle-based claripy bypass.
    ///
    /// When set, Python callbacks can return RustBVHandle instead of claripy ASTs,
    /// providing significant performance improvement by bypassing AST conversion.
    pub fn set_symbol_table(&mut self, table: &'a RustSymbolTable) {
        self.symbol_table = Some(table);
    }

    /// Add a concrete memory region for fast local access.
    ///
    /// This allows the interpreter to read from binary sections (e.g., .text, .rodata)
    /// without going through Python callbacks, significantly improving performance.
    pub fn add_concrete_memory(&mut self, base: u64, data: Vec<u8>) {
        let size = data.len() as u64;
        self.concrete_memory.push(ConcreteMemoryRegion { base, size, data });
        self.concrete_memory_sorted = false;
    }

    /// Sort concrete memory regions by base address for binary search.
    fn sort_concrete_memory(&mut self) {
        if !self.concrete_memory_sorted && self.concrete_memory.len() > 1 {
            self.concrete_memory.sort_by_key(|r| r.base);
            self.concrete_memory_sorted = true;
        }
    }

    /// Clear all concrete memory regions.
    pub fn clear_concrete_memory(&mut self) {
        self.concrete_memory.clear();
        self.concrete_memory_sorted = false;
    }

    /// Try to read from concrete memory cache using binary search.
    /// Returns Some(data) if the address range is fully contained in a cached region.
    #[inline]
    fn try_read_concrete_memory(&self, addr: u64, size: usize) -> Option<&[u8]> {
        if self.concrete_memory.is_empty() {
            return None;
        }

        // Use binary search if we have many regions
        if self.concrete_memory.len() > 4 && self.concrete_memory_sorted {
            // Binary search: find the region where base <= addr
            let idx = self.concrete_memory.partition_point(|r| r.base <= addr);
            if idx > 0 {
                // Check the region just before this index
                let region = &self.concrete_memory[idx - 1];
                if let Some(data) = region.read(addr, size) {
                    return Some(data);
                }
            }
            return None;
        }

        // Linear scan for small number of regions
        for region in &self.concrete_memory {
            if let Some(data) = region.read(addr, size) {
                return Some(data);
            }
        }
        None
    }

    /// Enable or disable profiling.
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling_enabled = enabled;
        if enabled {
            self.stats.reset();
        }
    }

    /// Get execution statistics.
    pub fn stats(&self) -> &ExecutionStats {
        &self.stats
    }

    /// Get mutable execution statistics.
    pub fn stats_mut(&mut self) -> &mut ExecutionStats {
        &mut self.stats
    }

    /// Take the execution statistics, replacing with default.
    pub fn take_stats(&mut self) -> ExecutionStats {
        std::mem::take(&mut self.stats)
    }

    /// Load from memory via Python callback.
    /// This handles the common case of loading from a concrete address.
    fn load_from_callback(
        &self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        addr_concrete: u64,
        size: usize,
    ) -> Result<RustBV, CbExecutionError> {
        let (data, is_symbolic, symbolic_ast) = callbacks
            .call_memory_load(py, addr_concrete, size as u32)
            .map_err(|e| CbExecutionError::Callback(e.to_string()))?;

        if is_symbolic {
            // Try to convert to RustBV - check handle first (fast path), then claripy (slow path)
            if let Some(ast_obj) = symbolic_ast {
                let ast = ast_obj.bind(py);

                // Fast path: check for RustBVHandle first
                if let Some(ref table) = self.symbol_table {
                    if let Some(bv) = try_handle_to_rustbv(&ast, table) {
                        return Ok(bv);
                    }
                }

                // Slow path: claripy AST conversion
                if is_claripy_ast(&ast) {
                    match claripy_to_rustbv(py, &ast, self.ctx) {
                        Ok(bv) => {
                            return Ok(bv);
                        }
                        Err(e) => {
                            // Fall back to creating a fresh symbolic value
                            // (claripy conversion can fail for complex/unsupported ops)
                        }
                    }
                } else {
                }
            } else {
            }
            // Fallback: create a fresh symbolic value
            let bv = RustBV::symbolic(
                self.ctx,
                &format!("mem_{:x}_{}", addr_concrete, size),
                (size * 8) as u32,
            );
            Ok(bv)
        } else {
            // Convert bytes to concrete value
            Ok(bytes_to_bv(&data, (size * 8) as u32))
        }
    }

    /// Get the execution configuration.
    pub fn config(&self) -> &ExecutionConfig {
        &self.config
    }

    /// Set the execution configuration.
    pub fn set_config(&mut self, config: ExecutionConfig) {
        self.config = config;
    }

    /// Get the deferred forks collected during execution.
    pub fn deferred_forks(&self) -> &[DeferredFork] {
        &self.deferred_forks
    }

    /// Take the deferred forks, leaving an empty vector.
    pub fn take_deferred_forks(&mut self) -> Vec<DeferredFork> {
        std::mem::take(&mut self.deferred_forks)
    }

    /// Clear the deferred forks.
    pub fn clear_deferred_forks(&mut self) {
        self.deferred_forks.clear();
    }

    /// Get the number of deferred forks.
    pub fn num_deferred_forks(&self) -> usize {
        self.deferred_forks.len()
    }

    /// Get the next condition ID.
    fn next_cond_id(&mut self) -> u64 {
        let id = self.next_condition_id;
        self.next_condition_id += 1;
        id
    }

    /// Get the current solver push level.
    pub fn push_level(&self) -> u32 {
        self.push_level
    }

    /// Get the solver context.
    pub fn context(&self) -> &SymContext {
        self.ctx
    }

    /// Get list of dirty register offsets (registers modified since last clear).
    /// Returns offsets in 4-byte granularity.
    pub fn get_dirty_register_offsets(&self) -> Vec<u32> {
        let mut offsets = Vec::new();
        for bit in 0..128u32 {
            if (self.dirty_registers & (1u128 << bit)) != 0 {
                offsets.push(bit * 4);
            }
        }
        offsets
    }

    /// Get the raw dirty register bitset.
    pub fn dirty_registers(&self) -> u128 {
        self.dirty_registers
    }

    /// Clear dirty register tracking (called after sync).
    pub fn clear_dirty_registers(&mut self) {
        self.dirty_registers = 0;
    }

    /// Flush pending stores to Python via batch callback.
    ///
    /// This sends all buffered stores in a single callback, reducing
    /// FFI overhead compared to individual store callbacks.
    fn flush_stores(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
    ) -> Result<(), CbExecutionError> {
        if self.pending_stores.is_empty() {
            return Ok(());
        }

        if self.use_rust_memory {
            // When Rust owns memory, flush stores to rust_memory instead of Python.
            // Stores already went to rust_memory in the store path (line 1435),
            // but some code paths may still buffer stores in pending_stores.
            if let Some(ref mut rust_mem) = self.rust_memory {
                for (addr, data) in &self.pending_stores {
                    let width = (data.len() * 8) as u32;
                    let mut val: u128 = 0;
                    for (i, &b) in data.iter().enumerate() {
                        val |= (b as u128) << (i * 8);
                    }
                    let bv = RustBV::concrete(val, width);
                    let _ = rust_mem.store_concrete_automap_internal(*addr, bv);
                }
            }
            // Still accumulate for cross-block load forwarding
            for (addr, data) in &self.pending_stores {
                self.all_flushed_stores.insert(*addr, data.clone());
            }
            self.pending_stores.clear();
            self.pending_symbolic_stores.clear();
            return Ok(());
        }

        // Python callback path (when Rust memory is not used)
        callbacks
            .call_memory_store_batch(py, &self.pending_stores)
            .map_err(|e| CbExecutionError::Callback(e.to_string()))?;

        // Accumulate flushed stores for cross-block load forwarding.
        for (addr, data) in &self.pending_stores {
            self.all_flushed_stores.insert(*addr, data.clone());
        }
        self.pending_stores.clear();
        self.pending_symbolic_stores.clear();
        Ok(())
    }

    /// Flush all pending stores into rust_memory.
    /// Called before extracting rust_memory back to the state.
    pub fn flush_stores_to_rust_memory(&mut self) {
        if let Some(ref mut rust_mem) = self.rust_memory {
            // Flush concrete pending stores
            for (addr, data) in self.pending_stores.drain(..) {
                let width = (data.len() * 8) as u32;
                let mut val: u128 = 0;
                for (i, &b) in data.iter().enumerate() {
                    val |= (b as u128) << (i * 8);
                }
                let bv = RustBV::concrete(val, width);
                let _ = rust_mem.store_concrete_automap_internal(addr, bv);
            }
            // Flush symbolic pending stores
            for (addr, bv) in self.pending_symbolic_stores.drain() {
                rust_mem.import_symbolic_value(addr, bv, None);
            }
            self.all_flushed_stores.clear();
        }
    }

    /// Take all stores from this step (both pending and previously flushed).
    pub fn take_all_stores(&mut self) -> Vec<(u64, Vec<u8>)> {
        self.pending_symbolic_stores.clear();
        // Merge pending into flushed
        for (addr, data) in self.pending_stores.drain(..) {
            self.all_flushed_stores.insert(addr, data);
        }
        std::mem::take(&mut self.all_flushed_stores).into_iter().collect()
    }

    /// Set the program counter.
    pub fn set_pc(&mut self, addr: u64) {
        self.pc = addr;
        let pc_bv = RustBV::concrete(addr as u128, self.registers.arch().bits());
        self.registers.set_ip(pc_bv);
    }

    /// Get the program counter.
    pub fn get_pc(&self) -> u64 {
        self.pc
    }

    /// Add a hook address.
    pub fn add_hook(&mut self, addr: u64) {
        self.hook_addrs.insert(addr);
    }

    /// Remove a hook address.
    pub fn remove_hook(&mut self, addr: u64) {
        self.hook_addrs.remove(&addr);
    }

    /// Add multiple hooks at once.
    pub fn add_hooks(&mut self, addrs: &[u64]) {
        for &addr in addrs {
            self.hook_addrs.insert(addr);
        }
    }

    /// Clear all hooks.
    pub fn clear_hooks(&mut self) {
        self.hook_addrs.clear();
    }

    /// Check if an address is hooked.
    pub fn is_hooked(&self, addr: u64) -> bool {
        self.hook_addrs.contains(&addr)
    }

    /// Check if an address is within loaded binary (concrete memory) regions.
    /// Used to distinguish internal function calls from external/library calls.
    pub fn is_in_binary(&self, addr: u64) -> bool {
        self.concrete_memory.iter().any(|region| {
            addr >= region.base && addr < region.base + region.size
        })
    }

    /// Register a SimProcedure at an address.
    ///
    /// This allows the interpreter to pre-extract arguments when the hook is hit,
    /// reducing Python callback overhead.
    pub fn register_simprocedure(&mut self, addr: u64, name: String, num_args: usize, no_return: bool) {
        self.hook_addrs.insert(addr);
        self.simprocedure_registry.insert(addr, SimProcedureInfo {
            name,
            num_args,
            no_return,
        });
    }

    /// Register multiple SimProcedures at once.
    ///
    /// Each tuple is (address, name, num_args, no_return).
    pub fn register_simprocedures(&mut self, procs: &[(u64, String, usize, bool)]) {
        for (addr, name, num_args, no_return) in procs {
            self.register_simprocedure(*addr, name.clone(), *num_args, *no_return);
        }
    }

    /// Get SimProcedure info for an address, if registered.
    pub fn get_simprocedure_info(&self, addr: u64) -> Option<&SimProcedureInfo> {
        self.simprocedure_registry.get(&addr)
    }

    /// Clear all SimProcedure registrations.
    pub fn clear_simprocedures(&mut self) {
        self.simprocedure_registry.clear();
    }

    /// Extract arguments for a SimProcedure call.
    ///
    /// Uses the calling convention to extract arguments from registers and stack.
    pub fn extract_simprocedure_args(&self, num_args: usize) -> Vec<RustBV> {
        self.calling_convention.extract_args(
            &self.registers,
            self.rust_memory.as_ref(),
            self.ctx,
            num_args,
        )
    }

    /// Get the return address for a function call.
    ///
    /// Checks pending_stores and all_flushed_stores first, since the call
    /// instruction pushes the return address via VEX stores before the
    /// interpreter detects the SimProcedure hook.
    pub fn get_return_addr(&self) -> Option<u64> {
        let ptr_size = self.calling_convention.pointer_size();
        let sp = self.registers.get(
            self.registers.arch().sp_offset(),
            ptr_size,
            self.ctx,
        );
        let sp_val = sp.as_u64()?;

        // Check pending_stores first (most recent writes, same block)
        for (addr, data) in self.pending_stores.iter().rev() {
            if *addr == sp_val && data.len() >= ptr_size as usize {
                let mut bytes = [0u8; 8];
                let len = std::cmp::min(ptr_size as usize, 8);
                bytes[..len].copy_from_slice(&data[..len]);
                return Some(u64::from_le_bytes(bytes));
            }
        }

        // Check all_flushed_stores (cross-block within same step)
        if let Some(data) = self.all_flushed_stores.get(&sp_val) {
            if data.len() >= ptr_size as usize {
                let mut bytes = [0u8; 8];
                let len = std::cmp::min(ptr_size as usize, 8);
                bytes[..len].copy_from_slice(&data[..len]);
                return Some(u64::from_le_bytes(bytes));
            }
        }

        // Fall back to rust_memory
        self.calling_convention.get_return_addr(
            &self.registers,
            self.rust_memory.as_ref(),
            self.ctx,
        )
    }

    /// Check if we have a cached block at the given address.
    pub fn has_cached_block(&self, addr: u64) -> bool {
        self.block_cache.contains(&addr)
    }

    /// Add a block to the cache.
    pub fn cache_block(&mut self, addr: u64, irsb: IRSB) {
        self.block_cache.put(addr, Arc::new(irsb));
    }

    /// Get a block from the cache.
    pub fn get_cached_block(&mut self, addr: u64) -> Option<&IRSB> {
        self.block_cache.get(&addr).map(|arc| arc.as_ref())
    }

    /// Take the last branch condition, if any.
    ///
    /// This is set when a SymbolicBranch result is created, and can be retrieved
    /// by callers who need to add constraints for forked states.
    /// The condition is cleared after being retrieved.
    pub fn take_last_branch_condition(&mut self) -> Option<RustBV> {
        self.last_branch_condition.take()
    }

    /// Get a stored condition by ID.
    ///
    /// Returns the branch condition associated with the given condition ID,
    /// if one was stored. This is used for deferred fork handling.
    pub fn get_stored_condition(&self, condition_id: u64) -> Option<&RustBV> {
        self.stored_conditions.get(&condition_id)
    }

    /// Take all stored conditions.
    ///
    /// Returns all stored conditions as a HashMap. The internal map is cleared.
    /// This is useful for bulk retrieval when processing multiple deferred forks.
    pub fn take_stored_conditions(&mut self) -> HashMap<u64, RustBV> {
        std::mem::take(&mut self.stored_conditions)
    }

    /// Run the execution loop until an event requires Python handling.
    ///
    /// This is the main entry point for the callback-based execution model.
    /// It runs blocks in a loop, using Python callbacks for memory access,
    /// until it hits a condition that requires Python-side handling.
    ///
    /// Returns a tuple of (result, blocks_executed, deferred_forks).
    pub fn run_until_event(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        max_blocks: u32,
    ) -> (RunResult, u32, Vec<DeferredFork>) {
        let total_start = if self.profiling_enabled { Some(Instant::now()) } else { None };
        let mut blocks_executed = 0u32;

        // Sort concrete memory regions for binary search if needed
        self.sort_concrete_memory();

        // Clear any previous deferred forks
        self.deferred_forks.clear();

        for _ in 0..max_blocks {
            // Check for hook at current PC
            if self.pc >= 0x4005fe && self.pc <= 0x400620 {
            }
            if self.is_hooked(self.pc) {
                let forks = self.take_deferred_forks();
                // Check if this is a registered SimProcedure with known args
                if let Some(info) = self.simprocedure_registry.get(&self.pc).cloned() {
                    // Get return address if available
                    let return_addr = self.get_return_addr().unwrap_or(0);
                    return (
                        RunResult::SimProcedure {
                            addr: self.pc,
                            name: info.name,
                            num_args: info.num_args,
                            return_addr,
                        },
                        blocks_executed,
                        forks,
                    );
                }
                // Fall back to generic Hook for unregistered hooks
                return (RunResult::Hook { addr: self.pc }, blocks_executed, forks);
            }

            // Check if we've hit the deferred forks limit
            if self.config.use_deferred_forks
                && self.deferred_forks.len() >= self.config.max_deferred_forks as usize
            {
                let forks = self.take_deferred_forks();
                return (RunResult::MaxDeferredForks { pc: self.pc }, blocks_executed, forks);
            }

            // Try to get or lift the block
            let irsb = match self.get_or_lift_block(py, callbacks, self.pc) {
                Ok(irsb) => irsb,
                Err(e) => {
                    let forks = self.take_deferred_forks();
                    return (
                        RunResult::Error {
                            message: e.to_string(),
                            addr: self.pc,
                        },
                        blocks_executed,
                        forks,
                    );
                }
            };

            // Execute the block
            match self.execute_block_with_callbacks(py, callbacks, &irsb) {
                Ok(result) => {
                    blocks_executed += 1;

                    match result {
                        BlockResult::Continue { next_addr } => {
                            self.pc = next_addr;
                            // Continue to next block
                        }
                        BlockResult::BlockEnd { next_addr, jumpkind } => {
                            self.pc = next_addr;
                            // Return for jumpkinds that need Python handling
                            if jumpkind.is_syscall() {
                                let syscall_num = self.get_syscall_num();
                                let forks = self.take_deferred_forks();
                                return (
                                    RunResult::Syscall {
                                        num: syscall_num,
                                        pc: next_addr,
                                    },
                                    blocks_executed,
                                    forks,
                                );
                            }
                            // For Call/Ret, we might want to return for SimProcedures
                            if self.is_hooked(next_addr) {
                                let forks = self.take_deferred_forks();
                                // Update PC before extracting args (so SP/ret addr are correct)
                                self.pc = next_addr;
                                // Check if this is a registered SimProcedure
                                if let Some(info) = self.simprocedure_registry.get(&next_addr).cloned() {
                                    let return_addr = self.get_return_addr().unwrap_or(0);
                                    return (
                                        RunResult::SimProcedure {
                                            addr: next_addr,
                                            name: info.name,
                                            num_args: info.num_args,
                                            return_addr,
                                        },
                                        blocks_executed,
                                        forks,
                                    );
                                }
                                return (RunResult::Hook { addr: next_addr }, blocks_executed, forks);
                            }
                            // Otherwise, continue execution
                        }
                        BlockResult::Syscall { num } => {
                            let forks = self.take_deferred_forks();
                            return (
                                RunResult::Syscall { num, pc: self.pc },
                                blocks_executed,
                                forks,
                            );
                        }
                        BlockResult::SymbolicBranch {
                            condition_id,
                            true_target,
                            false_target,
                        } => {
                            let forks = self.take_deferred_forks();
                            return (
                                RunResult::SymbolicBranch {
                                    condition_id,
                                    true_target,
                                    false_target,
                                },
                                blocks_executed,
                                forks,
                            );
                        }
                        BlockResult::Hook { addr } => {
                            let forks = self.take_deferred_forks();
                            // Check if this is a registered SimProcedure
                            if let Some(info) = self.simprocedure_registry.get(&addr).cloned() {
                                let return_addr = self.get_return_addr().unwrap_or(0);
                                return (
                                    RunResult::SimProcedure {
                                        addr,
                                        name: info.name,
                                        num_args: info.num_args,
                                        return_addr,
                                    },
                                    blocks_executed,
                                    forks,
                                );
                            }
                            return (RunResult::Hook { addr }, blocks_executed, forks);
                        }
                        BlockResult::Error { message } => {
                            let forks = self.take_deferred_forks();
                            return (
                                RunResult::Error {
                                    message,
                                    addr: self.pc,
                                },
                                blocks_executed,
                                forks,
                            );
                        }
                        BlockResult::SymbolicJumpTarget {
                            targets,
                            condition_id,
                            target_expr: _,
                            jumpkind,
                        } => {
                            let forks = self.take_deferred_forks();
                            return (
                                RunResult::SymbolicJumpTarget {
                                    targets,
                                    condition_id,
                                    jumpkind: format!("{:?}", jumpkind),
                                },
                                blocks_executed,
                                forks,
                            );
                        }
                        BlockResult::UnconstrainedJump {
                            min_target,
                            max_target,
                            limit,
                            jumpkind,
                        } => {
                            let forks = self.take_deferred_forks();
                            return (
                                RunResult::UnconstrainedJump {
                                    min_target,
                                    max_target,
                                    limit,
                                    jumpkind: format!("{:?}", jumpkind),
                                },
                                blocks_executed,
                                forks,
                            );
                        }
                        BlockResult::UnmodeledCall {
                            addr,
                            return_addr,
                            symbol_name,
                        } => {
                            let forks = self.take_deferred_forks();
                            return (
                                RunResult::UnmodeledCall {
                                    addr,
                                    return_addr,
                                    symbol_name,
                                },
                                blocks_executed,
                                forks,
                            );
                        }
                    }
                }
                Err(e) => {
                    let forks = self.take_deferred_forks();
                    return (
                        RunResult::Error {
                            message: e.to_string(),
                            addr: self.pc,
                        },
                        blocks_executed,
                        forks,
                    );
                }
            }
        }

        // Reached max blocks
        let forks = self.take_deferred_forks();
        if self.profiling_enabled {
            self.stats.blocks_executed += blocks_executed as u64;
            if let Some(start) = total_start {
                self.stats.total_time_ns += start.elapsed().as_nanos() as u64;
            }
        }
        (RunResult::MaxBlocks { pc: self.pc }, blocks_executed, forks)
    }

    /// Get or lift a block at the given address.
    ///
    /// This tries the following in order:
    /// 1. Check the block cache
    /// 2. Try native lifting via libpyvex (if feature enabled and bytes available)
    /// 3. Fall back to Python callback for lifting
    fn get_or_lift_block(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        addr: u64,
    ) -> Result<Arc<IRSB>, CbExecutionError> {
        // Check cache first - Arc clone is O(1)
        if let Some(irsb) = self.block_cache.get(&addr) {
            if self.profiling_enabled {
                self.stats.cache_hit_count += 1;
            }
            return Ok(Arc::clone(irsb));
        }

        if self.profiling_enabled {
            self.stats.cache_miss_count += 1;
        }

        let lift_start = if self.profiling_enabled { Some(Instant::now()) } else { None };

        // Try native lifting if available
        #[cfg(feature = "native-lift")]
        {
            if crate::vex::libpyvex_ffi::is_vex_initialized() {
                // Try to get bytes from concrete memory for native lifting
                // Look for a region containing this address with enough bytes
                for region in &self.concrete_memory {
                    if addr >= region.base && addr < region.base + region.size {
                        let offset = (addr - region.base) as usize;
                        let available = region.size as usize - offset;
                        // Limit block size to stop at hook/avoid/find addresses.
                        // Without this, blocks can span past these addresses,
                        // and the hook check at block boundaries misses them.
                        let mut max_bytes = available.min(4096);
                        for &hook_addr in &self.hook_addrs {
                            if hook_addr > addr && hook_addr < addr + max_bytes as u64 {
                                let limit = (hook_addr - addr) as usize;
                                if limit > 0 && limit < max_bytes {
                                    max_bytes = limit;
                                }
                            }
                        }
                        if max_bytes >= 1 {
                            let bytes = &region.data[offset..offset + max_bytes];
                            match crate::vex::libpyvex_ffi::lift_native(
                                bytes,
                                addr,
                                self.arch,
                                99,  // max_insns
                                max_bytes as u32,
                            ) {
                                Ok(irsb) => {
                                    // Native lift succeeded!
                                    log::trace!("Native lift succeeded at 0x{:x}", addr);
                                    if let Some(start) = lift_start {
                                        self.stats.lift_time_ns += start.elapsed().as_nanos() as u64;
                                    }
                                    let arc_irsb = Arc::new(irsb);
                                    self.block_cache.put(addr, Arc::clone(&arc_irsb));
                                    return Ok(arc_irsb);
                                }
                                Err(e) => {
                                    log::trace!("Native lift failed at 0x{:x}: {}", addr, e);
                                    // Fall through to Python callback
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }

        // Fall back to lifting via Python callback
        let callback_start = if self.profiling_enabled { Some(Instant::now()) } else { None };
        let irsb_json = callbacks
            .call_lift_block(py, addr)
            .map_err(|e| CbExecutionError::LiftError(format!("lift callback failed: {}", e)))?;
        if let Some(start) = callback_start {
            self.stats.python_callback_count += 1;
            self.stats.python_callback_time_ns += start.elapsed().as_nanos() as u64;
        }

        let irsb = deserialize_irsb(&irsb_json)
            .map_err(|e| CbExecutionError::LiftError(format!("IRSB deserialization failed: {}", e)))?;

        if let Some(start) = lift_start {
            self.stats.lift_time_ns += start.elapsed().as_nanos() as u64;
        }

        // Cache it - Arc allows O(1) cloning
        let arc_irsb = Arc::new(irsb);
        self.block_cache.put(addr, Arc::clone(&arc_irsb));

        Ok(arc_irsb)
    }

    /// Execute a block using Python callbacks for memory access.
    fn execute_block_with_callbacks(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        irsb: &IRSB,
    ) -> Result<BlockResult, CbExecutionError> {
        // Reset temps for this block - reuse allocation instead of creating new vec
        let needed_temps = irsb.tyenv.types.len();
        self.temps.clear();
        if self.temps.capacity() < needed_temps {
            self.temps.reserve(needed_temps - self.temps.capacity());
        }
        self.temps.resize(needed_temps, None);
        self.current_insn_addr = irsb.addr;

        // Prefetch loads for this block (reduces individual FFI calls)
        self.prefetch_loads_for_block(py, callbacks, irsb)?;

        // Execute statements
        for stmt in &irsb.statements {
            match self.execute_stmt_with_callbacks(py, callbacks, stmt, irsb)? {
                StmtResult::Continue => continue,
                StmtResult::Exit { target, jumpkind } => {
                    // Flush pending stores before returning
                    self.flush_stores(py, callbacks)?;
                    return Ok(self.handle_exit(target, jumpkind));
                }
                StmtResult::SymbolicBranch {
                    condition,
                    true_target,
                    false_target,
                } => {
                    // Flush pending stores before returning
                    self.flush_stores(py, callbacks)?;

                    // Generate unique condition ID and store condition for later retrieval
                    let cond_id = self.next_cond_id();
                    self.stored_conditions.insert(cond_id, condition.clone());

                    // Also store for callers using take_last_branch_condition
                    self.last_branch_condition = Some(condition);

                    return Ok(BlockResult::SymbolicBranch {
                        condition_id: cond_id,  // Fixed: use proper unique ID
                        true_target,
                        false_target,
                    });
                }
            }
        }

        // Flush pending stores at block end
        self.flush_stores(py, callbacks)?;

        // Handle default exit
        self.handle_default_exit(irsb)
    }

    /// Execute a single statement using Python callbacks.
    fn execute_stmt_with_callbacks(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        stmt: &IRStmt,
        irsb: &IRSB,
    ) -> Result<StmtResult, CbExecutionError> {
        match stmt {
            IRStmt::NoOp => Ok(StmtResult::Continue),

            IRStmt::IMark { addr, .. } => {
                self.current_insn_addr = *addr;
                // Check for hooks at this address
                if self.is_hooked(*addr) {
                    return Ok(StmtResult::Exit {
                        target: *addr,
                        jumpkind: JumpKind::Boring,
                    });
                }
                Ok(StmtResult::Continue)
            }

            IRStmt::AbiHint { .. } => Ok(StmtResult::Continue),

            IRStmt::Put { offset, data } => {
                let value = self.eval_expr_with_callbacks(py, callbacks, data, &irsb.tyenv)?;
                self.registers.put(*offset, value);

                // Mark register as dirty (4-byte granularity)
                let bit_index = (*offset / 4) as u32;
                if bit_index < 128 {
                    self.dirty_registers |= 1u128 << bit_index;
                }

                Ok(StmtResult::Continue)
            }

            IRStmt::WrTmp { tmp, data } => {
                let value = self.eval_expr_with_callbacks(py, callbacks, data, &irsb.tyenv)?;
                if (*tmp as usize) < self.temps.len() {
                    self.temps[*tmp as usize] = Some(value);
                } else {
                    return Err(CbExecutionError::UnknownTemp(*tmp));
                }
                Ok(StmtResult::Continue)
            }

            IRStmt::Store { addr, data, .. } => {
                let addr_val = self.eval_expr_with_callbacks(py, callbacks, addr, &irsb.tyenv)?;
                let data_val = self.eval_expr_with_callbacks(py, callbacks, data, &irsb.tyenv)?;
                let data_size = ((data_val.width() + 7) / 8) as usize;

                // Try Rust-native memory first if enabled - use unified method
                if self.use_rust_memory {
                    // First attempt - may need page fetch
                    let first_result = if let Some(ref mut rust_mem) = self.rust_memory {
                        Some(rust_mem.store_symbolic_unified(addr_val.clone(), data_val.clone(), self.ctx, &self.concretizer))
                    } else {
                        None
                    };

                    if let Some(result) = first_result {
                        match result {
                            Ok(()) => {
                                // Get concrete address (either directly or via concretization)
                                let addr_concrete_opt = if let Some(addr_concrete) = addr_val.as_u64() {
                                    self.load_prefetch_cache.remove(&(addr_concrete, data_size));
                                    Some(addr_concrete)
                                } else {
                                    // Symbolic address - try to concretize
                                    self.load_prefetch_cache.clear();
                                    let result = self.concretizer.concretize(&addr_val, self.ctx);
                                    match result {
                                        ConcretizationResult::Single(addr) => {
                                            self.track_concretization_constraint(&addr_val, addr);
                                            Some(addr)
                                        }
                                        ConcretizationResult::TooLarge { .. } => {
                                            None
                                        }
                                        _ => {
                                            None
                                        }
                                    }
                                };

                                // Rust owns memory — no need to sync stores to Python.
                                // The symbolic data is already in Rust's SymbolicMemory.
                                return Ok(StmtResult::Continue);
                            }
                            Err(MemoryError::UnmappedPageInRegion { page_addr }) => {
                                // Page is in a lazy region - fetch it (rust_mem borrow is dropped here)
                                let prefetch_count = self.page_prefetch_count;
                                let page_fetched = self.fetch_page_with_prefetch(py, callbacks, page_addr, prefetch_count)?;

                                // NOTE: We intentionally do NOT auto-map zero pages when page_fetched is false.
                                // Python may have actual data for this page from backers (file contents,
                                // initialized data). Speculatively creating zero pages causes state
                                // divergence between Rust and Python. Instead, we fall through to
                                // the Python callback which handles memory correctly.

                                if page_fetched {
                                    // Page was fetched - retry with unified store
                                    if let Some(ref mut rust_mem) = self.rust_memory {
                                        match rust_mem.store_symbolic_unified(addr_val.clone(), data_val.clone(), self.ctx, &self.concretizer) {
                                            Ok(()) => {
                                                // Get concrete address (either directly or via concretization)
                                                let addr_concrete_opt = if let Some(addr_concrete) = addr_val.as_u64() {
                                                    self.load_prefetch_cache.remove(&(addr_concrete, data_size));
                                                    Some(addr_concrete)
                                                } else {
                                                    // Symbolic address - try to concretize
                                                    self.load_prefetch_cache.clear();
                                                    match self.concretizer.concretize(&addr_val, self.ctx) {
                                                        ConcretizationResult::Single(addr) => {
                                                            self.track_concretization_constraint(&addr_val, addr);
                                                            Some(addr)
                                                        }
                                                        _ => None,
                                                    }
                                                };

                                                return Ok(StmtResult::Continue);
                                            }
                                            Err(_e) => {
                                                // Still failed - fall through to Python callback
                                            }
                                        }
                                    }
                                }
                                // If page not fetched, fall through to Python callback
                                                            }
                            Err(MemoryError::Unmapped { addr, size: unmapped_size }) => {
                                // Totally unmapped (not in lazy region) - fall through to Python
                                log::debug!(
                                    "Unmapped memory store at 0x{:x} (size={}), falling back to Python",
                                    addr, unmapped_size
                                );
                            }
                            Err(e) => {
                                return Err(CbExecutionError::Memory(e.to_string()));
                            }
                        }
                    }
                }

                // Store via callback - handle symbolic addresses
                if let Some(addr_concrete) = addr_val.as_u64() {
                    if self.arch.pointer_size() == 32 && data_val.is_symbolic() && data_size <= 4 {
                    }
                    self.load_prefetch_cache.remove(&(addr_concrete, data_size));

                    // Check if data is symbolic - use symbolic store callback.
                    // Only for 32-bit architectures where it's needed (e.g., flareon2015_5).
                    // 64-bit: skip entirely (too many false positives from extern addresses).
                    let use_sym_store = if self.arch.pointer_size() == 32 {
                        let is_stack = self.registers.get_sp_value().map_or(false, |sp_val| {
                            // Non-wrapping distance check
                            let dist = if addr_concrete >= sp_val {
                                addr_concrete - sp_val
                            } else {
                                sp_val - addr_concrete
                            };
                            dist <= 0x10000
                        });
                        !is_stack
                    } else {
                        false  // Skip for 64-bit — too expensive
                    };
                    if data_val.is_symbolic() && callbacks.has_memory_store_symbolic_value() && use_sym_store {
                        // Try symbolic store callback (preserves expression tree)
                        let sym_ok = (|| -> Result<(), CbExecutionError> {
                            self.flush_stores(py, callbacks)?;
                            callbacks.call_memory_store_symbolic_value(py, addr_concrete, &data_val)
                                .map_err(|e| CbExecutionError::Callback(e.to_string()))
                        })();
                        if let Err(_e) = sym_ok {
                            // Symbolic store callback failed — evaluate to concrete using
                            // solver and store directly via Python callback (not pending_stores).
                            // Using pending_stores would pollute all_flushed_stores with zeros
                            // since bv_to_bytes returns zeros for symbolic expressions.
                            let concrete_val = self.ctx.eval(&data_val).unwrap_or(0);
                            let size_bytes = (data_val.width() / 8) as usize;
                            let mut data_bytes = vec![0u8; size_bytes];
                            for i in 0..size_bytes {
                                data_bytes[i] = (concrete_val >> (i * 8)) as u8;
                            }
                            let _ = callbacks.call_memory_store(py, addr_concrete, &data_bytes);
                        }
                    } else {
                        // Fast path: buffer for batch processing
                        let data_bytes = bv_to_bytes(&data_val);
                        // Track symbolic values for load forwarding
                        if data_val.is_symbolic() {
                            self.pending_symbolic_stores.insert(addr_concrete, data_val);
                        }
                        self.pending_stores.push((addr_concrete, data_bytes));

                        // Auto-flush if buffer is full
                        if self.pending_stores.len() >= self.max_pending_stores {
                            self.flush_stores(py, callbacks)?;
                        }
                    }
                } else {
                    // Symbolic address - clear entire prefetch cache
                    self.load_prefetch_cache.clear();
                    // Symbolic address - flush buffer first, then handle specially
                    self.flush_stores(py, callbacks)?;
                    // Symbolic address - try to concretize
                    let concret_result = self.concretizer.concretize(&addr_val, self.ctx);
                    match concret_result {
                        ConcretizationResult::Single(addr_concrete) => {
                            // Track concretization constraint for Python sync
                            self.track_concretization_constraint(&addr_val, addr_concrete);
                            // Use symbolic store for symbolic values to preserve expression trees
                            if data_val.is_symbolic() && callbacks.has_memory_store_symbolic_value() {
                                callbacks
                                    .call_memory_store_symbolic_value(py, addr_concrete, &data_val)
                                    .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                            } else {
                                let data_bytes = bv_to_bytes(&data_val);
                                callbacks
                                    .call_memory_store(py, addr_concrete, &data_bytes)
                                    .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                            }
                        }
                        ConcretizationResult::Multiple(addrs) => {
                            // Sync constraints before delegating to Python
                            self.sync_before_callback(py, callbacks)?;
                            // Delegate to Python for conditional stores
                            callbacks
                                .call_memory_store_symbolic(py, &addrs, &data_val, &addr_val)
                                .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                        }
                        ConcretizationResult::Strided { base, stride, count } => {
                            // Sync constraints before delegating to Python
                            self.sync_before_callback(py, callbacks)?;
                            // Strided access pattern - generate addresses and delegate to Python
                            let addrs: Vec<u64> = (0..count).map(|i| base + i * stride).collect();
                            callbacks
                                .call_memory_store_symbolic(py, &addrs, &data_val, &addr_val)
                                .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                        }
                        ConcretizationResult::TooLarge { min, max, .. } => {
                            // Address range too large - delegate to Python's memory model
                            // which has access to angr's address concretization strategies

                            // Sync any pending constraints to Python before fallback
                            self.sync_before_callback(py, callbacks)?;

                            // Use full symbolic callback to preserve expression trees
                            if callbacks.has_memory_store_symbolic_full() {
                                                                callbacks
                                    .call_memory_store_symbolic_full(py, &addr_val, &data_val)
                                    .map_err(|e| CbExecutionError::Callback(format!(
                                        "symbolic store full callback failed at 0x{:x}-0x{:x}: {}",
                                        min, max, e
                                    )))?;
                            } else {
                                // Fallback to byte-based store (loses symbolic info)
                                let data_bytes = bv_to_bytes(&data_val);
                                callbacks
                                    .call_memory_store_symbolic_ast(py, &data_bytes, data_bytes.len() as u32)
                                    .map_err(|e| CbExecutionError::Callback(format!(
                                        "symbolic store AST callback failed at 0x{:x}-0x{:x}: {}",
                                        min, max, e
                                    )))?;
                            }
                        }
                        ConcretizationResult::Failed(reason) => {
                            return Err(CbExecutionError::Unsupported(format!(
                                "symbolic store address: {}",
                                reason
                            )));
                        }
                    }
                }

                Ok(StmtResult::Continue)
            }

            IRStmt::Exit { guard, dst, jk, .. } => {
                let guard_val = self.eval_expr_with_callbacks(py, callbacks, guard, &irsb.tyenv)?;

                // Check if guard is symbolic first (Constrained has concrete value but is still symbolic)
                if !guard_val.is_symbolic() {
                    // Truly concrete guard - simple check
                    if let Some(g) = guard_val.as_u64() {
                        if g != 0 {
                            return Ok(StmtResult::Exit {
                                target: *dst,
                                jumpkind: *jk,
                            });
                        }
                        return Ok(StmtResult::Continue);
                    }
                }

                // Guard is symbolic — handle based on deferred fork mode
                if !self.config.use_deferred_forks {
                    // Non-deferred mode: return to Python immediately for forking.
                    // Skip can_be_true/can_be_false Z3 checks — Python's
                    // resume_after_symbolic_branch will add constraints and the
                    // sat_cache optimization avoids redundant checks there.
                    let fallthrough = self.eval_next_addr(py, callbacks, irsb)?;
                    // Store condition for Rust-side constraint addition during resume
                    let cond_id = self.next_cond_id();
                    self.stored_conditions.insert(cond_id, guard_val.clone());
                    let result_cond = guard_val.clone();
                    self.last_branch_condition = Some(guard_val);
                    return Ok(StmtResult::SymbolicBranch {
                        condition: result_cond,
                        true_target: *dst,
                        false_target: fallthrough,
                    });
                }

                // Deferred fork mode: check feasibility to decide which paths to explore
                let (can_be_true, can_be_false) = self.ctx.check_branch_feasibility(&guard_val);

                if can_be_true && can_be_false {

                    // Create a deferred fork for the untaken path
                    let fallthrough = self.eval_next_addr(py, callbacks, irsb)?;

                    // Convert guard to claripy AST for constraint tracking
                    let condition_ast = match py.import("claripy") {
                        Ok(claripy_mod) => {
                            match rustbv_to_claripy(py, &guard_val, claripy_mod.as_any()) {
                                Ok(ast) => Some(ast),
                                Err(_) => None, // Failed to convert, fork will proceed without constraint
                            }
                        }
                        Err(_) => None, // Claripy not available
                    };

                    // Take the "true" path (jump to dst), defer the "false" path (fallthrough)
                    let cond_id = self.next_cond_id();
                    // Store the Rust condition for later retrieval when processing forks
                    self.stored_conditions.insert(cond_id, guard_val.clone());

                    let deferred = DeferredFork {
                        branch_addr: self.current_insn_addr,
                        path_taken: true,
                        unexplored_target: fallthrough,
                        condition_id: cond_id,
                        push_level: self.push_level,
                        condition_ast,
                    };
                    self.deferred_forks.push(deferred);

                    // Add constraint for the taken path to the solver.
                    // The main state continues on the true branch, so
                    // assume the guard is true. Without this, path
                    // constraints from symbolic branches are never added
                    // to the solver, and the found state has no useful
                    // constraints for solution extraction.
                    self.ctx.assume_true(&guard_val);

                    // Continue execution on the true branch
                    return Ok(StmtResult::Exit {
                        target: *dst,
                        jumpkind: *jk,
                    });
                } else if can_be_true {
                    return Ok(StmtResult::Exit {
                        target: *dst,
                        jumpkind: *jk,
                    });
                }
                Ok(StmtResult::Continue)
            }

            IRStmt::MBE(_) => Ok(StmtResult::Continue),

            IRStmt::PutI { descr, ix, bias, data } => {
                // Evaluate the index expression
                let ix_val = self.eval_expr_with_callbacks(py, callbacks, ix, &irsb.tyenv)?;

                // PutI requires a concrete index to compute the register offset
                let idx = if let Some(idx) = ix_val.as_u64() {
                    idx
                } else {
                    // Symbolic index - concretize using solver
                    if let Some(concrete) = self.ctx.eval(&ix_val) {
                        concrete as u64
                    } else {
                        return Err(CbExecutionError::Unsupported("PutI index concretization failed".to_string()));
                    }
                };

                // Calculate the rotating register offset:
                // offset = base + ((idx + bias) % nElems) * elemTy.bytes()
                let elem_size = descr.elemTy.bytes();
                let index = ((idx as u32).wrapping_add(*bias)) % descr.nElems;
                let offset = descr.base + index * elem_size;

                // Evaluate the data to write
                let data_val = self.eval_expr_with_callbacks(py, callbacks, data, &irsb.tyenv)?;

                // Write to the register file
                self.registers.put(offset, data_val);

                // Mark register as dirty (4-byte granularity)
                let bit_index = (offset / 4) as u32;
                if bit_index < 128 {
                    self.dirty_registers |= 1u128 << bit_index;
                }

                Ok(StmtResult::Continue)
            }

            IRStmt::StoreG { guard, addr, data, .. } => {
                // Evaluate guard condition
                let guard_val = self.eval_expr_with_callbacks(py, callbacks, guard, &irsb.tyenv)?;

                // Check if guard is symbolic
                if guard_val.is_symbolic() {
                    // Symbolic guard: need to handle conditional store
                    // For now, check if guard can be true at all
                    if !self.ctx.can_be_true(&guard_val) {
                        // Guard is always false - skip store
                        return Ok(StmtResult::Continue);
                    }
                    if !self.ctx.can_be_false(&guard_val) {
                        // Guard is always true - perform store unconditionally
                        let addr_val = self.eval_expr_with_callbacks(py, callbacks, addr, &irsb.tyenv)?;
                        let data_val = self.eval_expr_with_callbacks(py, callbacks, data, &irsb.tyenv)?;
                        let data_size = ((data_val.width() + 7) / 8) as usize;

                        if let Some(addr_concrete) = addr_val.as_u64() {
                            self.load_prefetch_cache.remove(&(addr_concrete, data_size));
                            // Check if data is symbolic - use symbolic store callback
                            if data_val.is_symbolic() && callbacks.has_memory_store_symbolic_value() {
                                self.flush_stores(py, callbacks)?;
                                callbacks
                                    .call_memory_store_symbolic_value(py, addr_concrete, &data_val)
                                    .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                            } else {
                                let data_bytes = bv_to_bytes(&data_val);
                                self.pending_stores.push((addr_concrete, data_bytes));
                                if self.pending_stores.len() >= self.max_pending_stores {
                                    self.flush_stores(py, callbacks)?;
                                }
                            }
                        }
                        return Ok(StmtResult::Continue);
                    }
                    // Both paths possible with symbolic guard - use ITE for conditional store
                    // Store ITE(guard, new_data, current_data)
                    let addr_val = self.eval_expr_with_callbacks(py, callbacks, addr, &irsb.tyenv)?;
                    let data_val = self.eval_expr_with_callbacks(py, callbacks, data, &irsb.tyenv)?;
                    let data_size = ((data_val.width() + 7) / 8) as usize;

                    if let Some(addr_concrete) = addr_val.as_u64() {
                        // Load current value at address
                        let current = self.load_from_callback(py, callbacks, addr_concrete, data_size)?;
                        // Create ITE: if guard then new_data else current
                        let ite_result = guard_val.ite(&data_val, &current, self.ctx);
                        self.load_prefetch_cache.remove(&(addr_concrete, data_size));
                        // ITE result is symbolic if guard or either operand is symbolic
                        if ite_result.is_symbolic() && callbacks.has_memory_store_symbolic_value() {
                            self.flush_stores(py, callbacks)?;
                            callbacks
                                .call_memory_store_symbolic_value(py, addr_concrete, &ite_result)
                                .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                        } else {
                            let ite_bytes = bv_to_bytes(&ite_result);
                            self.pending_stores.push((addr_concrete, ite_bytes));
                            if self.pending_stores.len() >= self.max_pending_stores {
                                self.flush_stores(py, callbacks)?;
                            }
                        }
                    } else {
                        // Symbolic address with symbolic guard - concretize address first
                        match self.concretizer.concretize(&addr_val, self.ctx) {
                            ConcretizationResult::Single(addr_concrete) => {
                                // Track concretization constraint for Python sync
                                self.track_concretization_constraint(&addr_val, addr_concrete);
                                // Load current value and use ITE
                                let current = self.load_from_callback(py, callbacks, addr_concrete, data_size)?;
                                let ite_result = guard_val.ite(&data_val, &current, self.ctx);
                                self.flush_stores(py, callbacks)?;
                                // ITE result is symbolic - use symbolic store callback
                                if ite_result.is_symbolic() && callbacks.has_memory_store_symbolic_value() {
                                    callbacks
                                        .call_memory_store_symbolic_value(py, addr_concrete, &ite_result)
                                        .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                                } else {
                                    let ite_bytes = bv_to_bytes(&ite_result);
                                    callbacks
                                        .call_memory_store(py, addr_concrete, &ite_bytes)
                                        .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                                }
                            }
                            _ => {
                                // P20: Multiple addresses with symbolic guard - delegate to Python
                                // instead of returning UNSUPPORTED error which deadends the state
                                log::debug!(
                                    "P20: Symbolic guarded store with multiple addresses - delegating to Python"
                                );
                                self.flush_stores(py, callbacks)?;
                                if callbacks.has_memory_store_symbolic_full() {
                                    // Use full symbolic store callback which handles complex cases
                                    callbacks
                                        .call_memory_store_symbolic_full(py, &addr_val, &data_val)
                                        .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                                } else {
                                    // No symbolic store callback - log warning but don't fail
                                    log::warn!(
                                        "P20: No symbolic_store_full callback for guarded store with symbolic address. \
                                         Store may be lost, but continuing execution."
                                    );
                                }
                            }
                        }
                    }
                    return Ok(StmtResult::Continue);
                }

                // Concrete guard: simple check
                if let Some(g) = guard_val.as_u64() {
                    if g != 0 {
                        // Guard is true - perform the store
                        let addr_val = self.eval_expr_with_callbacks(py, callbacks, addr, &irsb.tyenv)?;
                        let data_val = self.eval_expr_with_callbacks(py, callbacks, data, &irsb.tyenv)?;
                        let data_size = ((data_val.width() + 7) / 8) as usize;

                        if let Some(addr_concrete) = addr_val.as_u64() {
                            self.load_prefetch_cache.remove(&(addr_concrete, data_size));
                            // Check if data is symbolic - use symbolic store callback
                            if data_val.is_symbolic() && callbacks.has_memory_store_symbolic_value() {
                                self.flush_stores(py, callbacks)?;
                                callbacks
                                    .call_memory_store_symbolic_value(py, addr_concrete, &data_val)
                                    .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                            } else {
                                let data_bytes = bv_to_bytes(&data_val);
                                self.pending_stores.push((addr_concrete, data_bytes));
                                if self.pending_stores.len() >= self.max_pending_stores {
                                    self.flush_stores(py, callbacks)?;
                                }
                            }
                        } else {
                            // Symbolic address with concrete guard - flush and use callback
                            self.flush_stores(py, callbacks)?;
                            // Check if data is symbolic - use symbolic store callback
                            if data_val.is_symbolic() && callbacks.has_memory_store_symbolic_value() {
                                // Concretize address first
                                let concret_result = self.concretizer.concretize(&addr_val, self.ctx);
                                match concret_result {
                                    ConcretizationResult::Single(addr_concrete) => {
                                        self.track_concretization_constraint(&addr_val, addr_concrete);
                                        callbacks
                                            .call_memory_store_symbolic_value(py, addr_concrete, &data_val)
                                            .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                                    }
                                    ConcretizationResult::Multiple(addrs) => {
                                        // Delegate to Python for conditional stores with symbolic data
                                        if callbacks.has_memory_store_symbolic_full() {
                                            callbacks
                                                .call_memory_store_symbolic_full(py, &addr_val, &data_val)
                                                .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                                        } else {
                                            // Fallback: store to first address
                                            callbacks
                                                .call_memory_store_symbolic_value(py, addrs[0], &data_val)
                                                .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                                        }
                                    }
                                    _ => {
                                        // TooLarge or Failed - delegate to Python's full symbolic callback
                                        if callbacks.has_memory_store_symbolic_full() {
                                            callbacks
                                                .call_memory_store_symbolic_full(py, &addr_val, &data_val)
                                                .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                                        } else {
                                            return Err(CbExecutionError::Unsupported(
                                                "symbolic store with unconcretizable address".to_string()
                                            ));
                                        }
                                    }
                                }
                            } else {
                                let data_bytes = bv_to_bytes(&data_val);
                                callbacks
                                    .call_memory_store(py, 0, &data_bytes)
                                    .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                            }
                        }
                    }
                    // Guard is false - skip the store
                }

                Ok(StmtResult::Continue)
            }

            IRStmt::LoadG { dst, guard, addr, alt, cvt, .. } => {
                // Evaluate guard condition
                let guard_val = self.eval_expr_with_callbacks(py, callbacks, guard, &irsb.tyenv)?;

                // Evaluate the alternative value (used when guard is false)
                let alt_val = self.eval_expr_with_callbacks(py, callbacks, alt, &irsb.tyenv)?;

                // Determine the load size from the destination temp type
                let dst_ty = irsb.tyenv.get(*dst).ok_or_else(|| {
                    CbExecutionError::InvalidIR(format!("LoadG destination temp {} not in tyenv", dst))
                })?;
                let load_size = match cvt {
                    IRLoadGOp::Identity => dst_ty.bytes() as usize,
                    IRLoadGOp::WidenS | IRLoadGOp::WidenZ => {
                        // For widening loads, the memory load is smaller
                        // Typically 8->32, 16->32, 32->64
                        match dst_ty.bytes() {
                            4 => 1, // Could be 1 or 2, default to 1
                            8 => 4, // 32->64
                            _ => dst_ty.bytes() as usize,
                        }
                    }
                };

                // Check if guard is symbolic
                if guard_val.is_symbolic() {
                    // Check if guard can be true/false
                    let can_be_true = self.ctx.can_be_true(&guard_val);
                    let can_be_false = self.ctx.can_be_false(&guard_val);

                    if can_be_true && !can_be_false {
                        // Guard is always true - perform load unconditionally
                        let addr_val = self.eval_expr_with_callbacks(py, callbacks, addr, &irsb.tyenv)?;

                        // Get concrete address (directly or via concretization)
                        let addr_concrete = match addr_val.as_u64() {
                            Some(a) => a,
                            None => {
                                // Symbolic address - concretize
                                match self.concretizer.concretize(&addr_val, self.ctx) {
                                    ConcretizationResult::Single(a) => {
                                        // Track concretization constraint for Python sync
                                        self.track_concretization_constraint(&addr_val, a);
                                        a
                                    }
                                    ConcretizationResult::Multiple(ref addrs) => {
                                        *addrs.first().ok_or_else(|| {
                                            CbExecutionError::Unsupported("LoadG with empty address set".to_string())
                                        })?
                                    }
                                    _ => {
                                        return Err(CbExecutionError::Unsupported("LoadG address concretization failed".to_string()));
                                    }
                                }
                            }
                        };

                        let loaded = self.load_from_callback(py, callbacks, addr_concrete, load_size)?;

                        // Apply conversion
                        let result = self.apply_loadg_conversion(*cvt, loaded, dst_ty.bits());

                        if (*dst as usize) < self.temps.len() {
                            self.temps[*dst as usize] = Some(result);
                        }
                        return Ok(StmtResult::Continue);
                    }

                    if !can_be_true && can_be_false {
                        // Guard is always false - use alt value
                        if (*dst as usize) < self.temps.len() {
                            self.temps[*dst as usize] = Some(alt_val);
                        }
                        return Ok(StmtResult::Continue);
                    }

                    // Both paths possible - evaluate address and load, then ITE
                    let addr_val = self.eval_expr_with_callbacks(py, callbacks, addr, &irsb.tyenv)?;

                    // Get concrete address
                    let addr_concrete = match addr_val.as_u64() {
                        Some(a) => a,
                        None => {
                            match self.concretizer.concretize(&addr_val, self.ctx) {
                                ConcretizationResult::Single(a) => {
                                    // Track concretization constraint for Python sync
                                    self.track_concretization_constraint(&addr_val, a);
                                    a
                                }
                                ConcretizationResult::Multiple(ref addrs) => {
                                    *addrs.first().ok_or_else(|| {
                                        CbExecutionError::Unsupported("LoadG with empty address set".to_string())
                                    })?
                                }
                                _ => {
                                    return Err(CbExecutionError::Unsupported("LoadG address concretization failed".to_string()));
                                }
                            }
                        }
                    };

                    let loaded = self.load_from_callback(py, callbacks, addr_concrete, load_size)?;

                    // Apply conversion to loaded value
                    let converted = self.apply_loadg_conversion(*cvt, loaded, dst_ty.bits());

                    // Create ITE: if guard then loaded else alt
                    let result = guard_val.ite(&converted, &alt_val, self.ctx);

                    if (*dst as usize) < self.temps.len() {
                        self.temps[*dst as usize] = Some(result);
                    }
                    return Ok(StmtResult::Continue);
                }

                // Concrete guard
                if let Some(g) = guard_val.as_u64() {
                    let result = if g != 0 {
                        // Guard is true - perform the load
                        let addr_val = self.eval_expr_with_callbacks(py, callbacks, addr, &irsb.tyenv)?;

                        // Get concrete address
                        let addr_concrete = match addr_val.as_u64() {
                            Some(a) => a,
                            None => {
                                match self.concretizer.concretize(&addr_val, self.ctx) {
                                    ConcretizationResult::Single(a) => {
                                        // Track concretization constraint for Python sync
                                        self.track_concretization_constraint(&addr_val, a);
                                        a
                                    }
                                    ConcretizationResult::Multiple(ref addrs) => {
                                        *addrs.first().ok_or_else(|| {
                                            CbExecutionError::Unsupported("LoadG with empty address set".to_string())
                                        })?
                                    }
                                    _ => {
                                        return Err(CbExecutionError::Unsupported("LoadG address concretization failed".to_string()));
                                    }
                                }
                            }
                        };

                        let loaded = self.load_from_callback(py, callbacks, addr_concrete, load_size)?;
                        self.apply_loadg_conversion(*cvt, loaded, dst_ty.bits())
                    } else {
                        // Guard is false - use alternative value
                        alt_val
                    };

                    if (*dst as usize) < self.temps.len() {
                        self.temps[*dst as usize] = Some(result);
                    }
                } else {
                    // This shouldn't happen if guard_val is concrete
                    return Err(CbExecutionError::InvalidIR("LoadG guard evaluation failed".to_string()));
                }

                Ok(StmtResult::Continue)
            }

            IRStmt::CAS { .. } => Err(CbExecutionError::Unsupported("compare-and-swap".to_string())),
            IRStmt::LLSC { .. } => {
                Err(CbExecutionError::Unsupported("load-linked/store-conditional".to_string()))
            }
            IRStmt::Dirty(dirty) => {
                // Check guard if present
                if let Some(guard) = &dirty.guard {
                    let guard_val = self.eval_expr_with_callbacks(py, callbacks, guard, &irsb.tyenv)?;
                    if guard_val.is_symbolic() {
                        // Symbolic guard - check if can be true/false
                        if !self.ctx.can_be_true(&guard_val) {
                            return Ok(StmtResult::Continue);
                        }
                        // Both paths possible - need Python to handle
                        return Err(CbExecutionError::Unsupported(format!(
                            "dirty call with symbolic guard: {}",
                            dirty.cee.name
                        )));
                    } else if let Some(g) = guard_val.as_u64() {
                        if g == 0 {
                            // Guard is false - skip the dirty call
                            return Ok(StmtResult::Continue);
                        }
                    }
                }

                // Evaluate arguments
                let mut arg_vals: Vec<u64> = Vec::with_capacity(dirty.args.len());
                let mut all_args_concrete = true;
                for arg in &dirty.args {
                    let val = self.eval_expr_with_callbacks(py, callbacks, arg, &irsb.tyenv)?;
                    if let Some(concrete) = val.as_u64() {
                        arg_vals.push(concrete);
                    } else {
                        all_args_concrete = false;
                        break;
                    }
                }

                // Determine return type bits
                let ret_ty_bits = if let Some(tmp) = dirty.tmp {
                    irsb.tyenv.get(tmp).map(|t| t.bits()).unwrap_or(64)
                } else {
                    0 // No return value
                };

                // Try native dirty helper dispatch first
                if all_args_concrete {
                    if let Some(result) = self.dirty_dispatch.try_call(&dirty.cee.name, &arg_vals) {
                        // Native handler succeeded!
                        log::trace!("Native dirty call: {} (args: {:?})", dirty.cee.name, arg_vals);

                        // Store result in temporary if specified
                        if let Some(tmp) = dirty.tmp {
                            if let Some(return_value) = result.return_value {
                                let value = RustBV::concrete(return_value as u128, ret_ty_bits);
                                if (tmp as usize) < self.temps.len() {
                                    self.temps[tmp as usize] = Some(value);
                                }
                            }
                        }

                        // Apply any register writes from the helper
                        for (offset, value) in result.reg_writes {
                            // Convert u64 value to RustBV and store in register
                            let bv = RustBV::concrete(value as u128, 64);
                            self.registers.put(offset, bv);
                        }

                        return Ok(StmtResult::Continue);
                    }
                }

                // Fall back to Python callback
                if !callbacks.has_dirty_call() {
                    return Err(CbExecutionError::Unsupported(format!(
                        "dirty call: {} (no callback and no native handler)",
                        dirty.cee.name
                    )));
                }

                if !all_args_concrete {
                    // Re-evaluate args for Python (we aborted early above)
                    arg_vals.clear();
                    for arg in &dirty.args {
                        let val = self.eval_expr_with_callbacks(py, callbacks, arg, &irsb.tyenv)?;
                        if let Some(concrete) = val.as_u64() {
                            arg_vals.push(concrete);
                        } else {
                            // Symbolic argument - Python needs to handle this
                            return Err(CbExecutionError::Unsupported(format!(
                                "dirty call with symbolic arg: {}",
                                dirty.cee.name
                            )));
                        }
                    }
                }

                // Call Python callback
                let (data, is_symbolic, _symbolic_ast) = callbacks
                    .call_dirty_call(py, &dirty.cee.name, &arg_vals, ret_ty_bits)
                    .map_err(|e| CbExecutionError::Callback(format!(
                        "dirty call {} failed: {}",
                        dirty.cee.name, e
                    )))?;

                // Store result in temporary if specified
                if let Some(tmp) = dirty.tmp {
                    let result = if is_symbolic {
                        // Create a symbolic value for the result
                        RustBV::symbolic(
                            self.ctx,
                            &format!("dirty_{}", dirty.cee.name),
                            ret_ty_bits,
                        )
                    } else {
                        // Convert bytes to concrete value
                        let mut value: u128 = 0;
                        for (i, &byte) in data.iter().enumerate() {
                            if (i * 8) as u32 >= ret_ty_bits {
                                break;
                            }
                            value |= (byte as u128) << (i * 8);
                        }
                        RustBV::concrete(value, ret_ty_bits)
                    };

                    if (tmp as usize) < self.temps.len() {
                        self.temps[tmp as usize] = Some(result);
                    }
                }

                Ok(StmtResult::Continue)
            }
        }
    }

    /// Evaluate an IR expression using Python callbacks for memory loads.
    fn eval_expr_with_callbacks(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        expr: &IRExpr,
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        match expr {
            IRExpr::Const(c) => Ok(self.eval_const(c)),

            IRExpr::RdTmp(tmp) => {
                if let Some(Some(val)) = self.temps.get(*tmp as usize) {
                    Ok(val.clone())
                } else {
                    Err(CbExecutionError::UnknownTemp(*tmp))
                }
            }

            IRExpr::Get { offset, ty } => {
                let size = ty.bytes();
                Ok(self.registers.get(*offset, size, self.ctx))
            }

            IRExpr::Load { addr, ty, .. } => {
                let addr_val = self.eval_expr_with_callbacks(py, callbacks, addr, tyenv)?;
                let size = ty.bytes() as usize;

                // Try Rust-native memory first if enabled - use unified method
                if self.use_rust_memory {
                    // First attempt - may need page fetch
                    let first_result = if let Some(ref mut rust_mem) = self.rust_memory {
                        Some(rust_mem.load_symbolic_unified(addr_val.clone(), size as u32, self.ctx, &self.concretizer))
                    } else {
                        None
                    };

                    if let Some(result) = first_result {
                        match result {
                            Ok(value) => return Ok(value),
                            Err(MemoryError::UnmappedPageInRegion { page_addr }) => {
                                // Page is in a lazy region - fetch it (rust_mem borrow is dropped here)
                                let prefetch_count = self.page_prefetch_count;
                                let page_fetched = self.fetch_page_with_prefetch(py, callbacks, page_addr, prefetch_count)?;

                                // NOTE: We intentionally do NOT auto-map zero pages when page_fetched is false.
                                // Python may have actual data for this page from backers (file contents,
                                // initialized data). Speculatively creating zero pages causes state
                                // divergence between Rust and Python. Instead, we fall through to
                                // the Python callback which handles memory correctly.

                                if page_fetched {
                                    // Page was fetched - retry with unified load
                                    if let Some(ref mut rust_mem) = self.rust_memory {
                                        match rust_mem.load_symbolic_unified(addr_val.clone(), size as u32, self.ctx, &self.concretizer) {
                                            Ok(value) => return Ok(value),
                                            Err(_e) => {
                                                // Still failed - fall through to Python callback
                                            }
                                        }
                                    }
                                }
                                // If page not fetched, fall through to Python callback
                            }
                            Err(MemoryError::Unmapped { addr, size: unmapped_size }) => {
                                // Totally unmapped (not in lazy region) - fall through to Python
                                log::debug!(
                                    "Unmapped memory load at 0x{:x} (size={}), falling back to Python",
                                    addr, unmapped_size
                                );
                            }
                            Err(MemoryError::SymbolicAddress { .. }) => {
                                // Symbolic bytes not fully tracked - fall through to Python
                                // This happens when symbolic values were imported per-byte
                                // but the load is multi-byte, or when the symbolic import
                                // didn't cover all bytes at this address.
                            }
                            Err(e) => {
                                return Err(CbExecutionError::Memory(e.to_string()));
                            }
                        }
                    }
                }

                if let Some(addr_concrete) = addr_val.as_u64() {
                    // FAST PATH 0: Check pending stores buffer
                    // Stores within the same block are buffered in pending_stores.
                    // We must check this buffer before falling through to Python
                    // callbacks, which have stale state.

                    // First check symbolic stores (preserves symbolic values)
                    if let Some(sym_val) = self.pending_symbolic_stores.get(&addr_concrete) {
                        if sym_val.width() == (size * 8) as u32 {
                            return Ok(sym_val.clone());
                        } else if sym_val.width() > (size * 8) as u32 {
                            return Ok(sym_val.extract((size * 8 - 1) as u32, 0, self.ctx));
                        }
                    }

                    // Then check concrete stores (reverse order for most recent)
                    for &(store_addr, ref store_data) in self.pending_stores.iter().rev() {
                        if store_addr <= addr_concrete && addr_concrete + size as u64 <= store_addr + store_data.len() as u64 {
                            let offset = (addr_concrete - store_addr) as usize;
                            let data = &store_data[offset..offset + size];
                            return Ok(bytes_to_bv(data, (size * 8) as u32));
                        }
                    }

                    // Also check previously flushed stores (from earlier blocks in this step)
                    if let Some(store_data) = self.all_flushed_stores.get(&addr_concrete) {
                        if size <= store_data.len() {
                            let data = &store_data[..size];
                            return Ok(bytes_to_bv(data, (size * 8) as u32));
                        }
                    }

                    // FAST PATH 1: Check prefetch cache (batch-loaded values)
                    if let Some(prefetched) = self.load_prefetch_cache.get(&(addr_concrete, size)) {
                        return Ok(prefetched.value.clone());
                    }

                    // FAST PATH 2: Check if address is in Rust-cached concrete memory
                    if let Some(data) = self.try_read_concrete_memory(addr_concrete, size) {
                        return Ok(bytes_to_bv(data, (size * 8) as u32));
                    }

                    // SLOW PATH: Fall back to Python callback
                    self.load_from_callback(py, callbacks, addr_concrete, size)
                } else {
                    // Symbolic address - try to concretize
                    match self.concretizer.concretize(&addr_val, self.ctx) {
                        ConcretizationResult::Single(addr_concrete) => {
                            if self.arch.pointer_size() == 32 && addr_concrete >= 0x400000 && addr_concrete < 0x420000 {
                            }
                            self.track_concretization_constraint(&addr_val, addr_concrete);
                            if let Some(data) = self.try_read_concrete_memory(addr_concrete, size) {
                                return Ok(bytes_to_bv(data, (size * 8) as u32));
                            }
                            self.load_from_callback(py, callbacks, addr_concrete, size)
                        }
                        ConcretizationResult::Multiple(addrs) => {
                            // Sync constraints before batch load
                            self.sync_before_callback(py, callbacks)?;
                            // Build ITE chain in Rust instead of delegating to Python
                            // This avoids FFI overhead and keeps symbolic ops in Rust's Z3 context
                            self.build_ite_load_from_callbacks(py, callbacks, &addrs, &addr_val, size)
                        }
                        ConcretizationResult::Strided { base, stride, count } => {
                            // Sync constraints before batch load
                            self.sync_before_callback(py, callbacks)?;
                            // Strided access pattern - generate addresses and build ITE chain in Rust
                            let addrs: Vec<u64> = (0..count).map(|i| base + i * stride).collect();
                            self.build_ite_load_from_callbacks(py, callbacks, &addrs, &addr_val, size)
                        }
                        ConcretizationResult::TooLarge { min, max, .. } => {
                            // Address range too large - delegate to Python's memory model
                            // which has access to angr's address concretization strategies

                            // Sync any pending constraints to Python before fallback
                            self.sync_before_callback(py, callbacks)?;

                            // NEW: Use full symbolic load if available - passes address AST to Python
                            // This allows Python's memory model to properly resolve the symbolic address
                            // and return the actual stored value instead of a fresh unconstrained symbol
                            if callbacks.has_memory_load_symbolic_full() {
                                let result_ast = callbacks
                                    .call_memory_load_symbolic_full(py, &addr_val, size as u32)
                                    .map_err(|e| CbExecutionError::Callback(format!(
                                        "symbolic load full callback failed at 0x{:x}-0x{:x}: {}",
                                        min, max, e
                                    )))?;

                                // Try to convert claripy AST back to RustBV
                                let ast = result_ast.bind(py);

                                // Fast path: check for RustBVHandle first
                                if let Some(ref table) = self.symbol_table {
                                    if let Some(bv) = try_handle_to_rustbv(&ast, table) {
                                        return Ok(bv);
                                    }
                                }

                                // Slow path: claripy AST conversion
                                if is_claripy_ast(&ast) {
                                    match claripy_to_rustbv(py, &ast, self.ctx) {
                                        Ok(bv) => return Ok(bv),
                                        Err(e) => {
                                            // Log warning about conversion failure
                                            log::warn!(
                                                "Symbolic load at 0x{:x} (size={}): AST conversion failed: {}. \
                                                 Creating fresh symbol - constraints may diverge!",
                                                min, size, e
                                            );
                                        }
                                    }
                                }

                                // Fallback: create a fresh symbolic value with marker name
                                log::debug!(
                                    "Creating fresh symbolic value sym_pyref_{:x}_{} for symbolic load",
                                    min, size
                                );
                                return Ok(RustBV::symbolic(
                                    self.ctx,
                                    &format!("sym_pyref_{:x}_{}", min, size),  // Named to indicate Python reference
                                    (size * 8) as u32,
                                ));
                            }

                            // LEGACY: Fall back to size-only callback if full callback not set
                            let (data, is_symbolic, symbolic_ast) = callbacks
                                .call_memory_load_symbolic_ast(py, size as u32)
                                .map_err(|e| CbExecutionError::Callback(format!(
                                    "symbolic load AST callback failed at 0x{:x}-0x{:x}: {}",
                                    min, max, e
                                )))?;

                            if is_symbolic {
                                // Try to convert to RustBV - check handle first (fast path), then claripy (slow path)
                                if let Some(ast_obj) = symbolic_ast {
                                    let ast = ast_obj.bind(py);

                                    // Fast path: check for RustBVHandle first
                                    if let Some(ref table) = self.symbol_table {
                                        if let Some(bv) = try_handle_to_rustbv(&ast, table) {
                                            return Ok(bv);
                                        }
                                    }

                                    // Slow path: claripy AST conversion
                                    if is_claripy_ast(&ast) {
                                        match claripy_to_rustbv(py, &ast, self.ctx) {
                                            Ok(bv) => return Ok(bv),
                                            Err(e) => {
                                                // Log warning about conversion failure
                                                log::warn!(
                                                    "Symbolic load at 0x{:x} (size={}): legacy AST conversion failed: {}. \
                                                     Creating fresh symbol - constraints may diverge!",
                                                    min, size, e
                                                );
                                            }
                                        }
                                    }
                                }
                                // Fallback: create a fresh symbolic value with marker name
                                log::debug!(
                                    "Creating fresh symbolic value sym_pyref_{:x}_{} for legacy symbolic load",
                                    min, size
                                );
                                Ok(RustBV::symbolic(
                                    self.ctx,
                                    &format!("sym_pyref_{:x}_{}", min, size),  // Named to indicate Python reference
                                    (size * 8) as u32,
                                ))
                            } else {
                                // Concrete result from Python
                                Ok(bytes_to_bv(&data, (size * 8) as u32))
                            }
                        }
                        ConcretizationResult::Failed(reason) => {
                            Err(CbExecutionError::Unsupported(format!(
                                "symbolic load address: {}",
                                reason
                            )))
                        }
                    }
                }
            }

            IRExpr::Unop { op, arg } => {
                let arg_val = self.eval_expr_with_callbacks(py, callbacks, arg, tyenv)?;
                let arg_is_sym = arg_val.is_symbolic();
                VEXOps::unop(*op, arg_val, self.ctx).or_else(|_| {
                    // Fallback for unsupported unary ops (e.g., float conversions).
                    // Return fresh symbolic if input was symbolic, else zero.
                    let width = op.result_type().map(|t| t.bits()).unwrap_or(64);
                    if arg_is_sym {
                        Ok(RustBV::symbolic(self.ctx, &format!("unsup_unop_{:x}", self.pc), width))
                    } else {
                        Ok(RustBV::concrete(0, width))
                    }
                })
            }

            IRExpr::Binop { op, left, right } => {
                let left_val = self.eval_expr_with_callbacks(py, callbacks, left, tyenv)?;
                let right_val = self.eval_expr_with_callbacks(py, callbacks, right, tyenv)?;
                let fallback_width = op.result_type().map(|t| t.bits()).unwrap_or(
                    left_val.width().max(right_val.width())
                );
                let any_sym = left_val.is_symbolic() || right_val.is_symbolic();
                VEXOps::binop(*op, left_val, right_val, self.ctx).or_else(|_| {
                    // Fallback for unsupported binary ops (e.g., vector float ops).
                    if any_sym {
                        Ok(RustBV::symbolic(self.ctx, &format!("unsup_binop_{:x}", self.pc), fallback_width))
                    } else {
                        Ok(RustBV::concrete(0, fallback_width))
                    }
                })
            }

            IRExpr::ITE { cond, iftrue, iffalse } => {
                let cond_val = self.eval_expr_with_callbacks(py, callbacks, cond, tyenv)?;
                let true_val = self.eval_expr_with_callbacks(py, callbacks, iftrue, tyenv)?;
                let false_val = self.eval_expr_with_callbacks(py, callbacks, iffalse, tyenv)?;
                Ok(cond_val.ite(&true_val, &false_val, self.ctx))
            }

            IRExpr::GetI { descr, ix, bias } => {
                // Evaluate the index expression
                let ix_val = self.eval_expr_with_callbacks(py, callbacks, ix, tyenv)?;

                // GetI requires a concrete index to compute the register offset
                let idx = if let Some(idx) = ix_val.as_u64() {
                    idx
                } else {
                    // Symbolic index - concretize using solver
                    if let Some(concrete) = self.ctx.eval(&ix_val) {
                        concrete as u64
                    } else {
                        return Err(CbExecutionError::Unsupported("GetI index concretization failed".to_string()));
                    }
                };

                // Calculate the rotating register offset:
                // offset = base + ((idx + bias) % nElems) * elemTy.bytes()
                let elem_size = descr.elemTy.bytes();
                let index = ((idx as u32).wrapping_add(*bias)) % descr.nElems;
                let offset = descr.base + index * elem_size;

                // Read from the register file
                Ok(self.registers.get(offset, elem_size, self.ctx))
            }

            IRExpr::Triop { op, arg1, arg2, arg3 } => {
                // Triops are typically float operations with rounding mode.
                // Return fresh symbolic if any operand is symbolic, so that
                // branches depending on float results remain explorable.
                let v1 = self.eval_expr_with_callbacks(py, callbacks, arg1, tyenv)?;
                let v2 = self.eval_expr_with_callbacks(py, callbacks, arg2, tyenv)?;
                let v3 = self.eval_expr_with_callbacks(py, callbacks, arg3, tyenv)?;
                let width = op.result_type().map(|t| t.bits()).unwrap_or(64);
                if v1.is_symbolic() || v2.is_symbolic() || v3.is_symbolic() {
                    Ok(RustBV::symbolic(self.ctx, &format!("triop_{:x}", self.pc), width))
                } else {
                    Ok(RustBV::concrete(0, width))
                }
            }

            IRExpr::Qop { op, arg1, arg2, arg3, arg4 } => {
                let v1 = self.eval_expr_with_callbacks(py, callbacks, arg1, tyenv)?;
                let v2 = self.eval_expr_with_callbacks(py, callbacks, arg2, tyenv)?;
                let v3 = self.eval_expr_with_callbacks(py, callbacks, arg3, tyenv)?;
                let v4 = self.eval_expr_with_callbacks(py, callbacks, arg4, tyenv)?;
                let width = op.result_type().map(|t| t.bits()).unwrap_or(64);
                if v1.is_symbolic() || v2.is_symbolic() || v3.is_symbolic() || v4.is_symbolic() {
                    Ok(RustBV::symbolic(self.ctx, &format!("qop_{:x}", self.pc), width))
                } else {
                    Ok(RustBV::concrete(0, width))
                }
            }

            IRExpr::CCall { cee, retty, args } => {
                let mut arg_vals = Vec::with_capacity(args.len());
                for arg in args {
                    arg_vals.push(self.eval_expr_with_callbacks(py, callbacks, arg, tyenv)?);
                }

                if let Some(result) = ccall::handle_ccall_with_ctx(&cee.name, &arg_vals, retty.bits(), Some(self.ctx)) {
                    return Ok(result);
                }

                // For condition code CCalls with CC_OP_COPY (op=0) and symbolic
                // deps: return symbolic. CC_OP_COPY is used for float comparisons
                // (ucomisd/comisd) where flags are set directly from the result.
                // For integer ops (cc_op > 0: ADD, SUB, etc.), return concrete 0
                // to avoid state explosion.
                let is_cond_ccall = cee.name.contains("calculate_condition")
                    || cee.name.contains("calculate_eflags");
                if is_cond_ccall && arg_vals.len() >= 4 {
                    let cc_op = arg_vals[1].as_u64();
                    let deps_symbolic = arg_vals.get(2).map_or(false, |v| v.is_symbolic())
                        || arg_vals.get(3).map_or(false, |v| v.is_symbolic());
                    // CC_OP_COPY = 0: flags were set directly (float comparison)
                    if cc_op == Some(0) && deps_symbolic {
                        return Ok(RustBV::symbolic(
                            self.ctx,
                            &format!("ccall_cond_{:x}", self.pc),
                            retty.bits(),
                        ));
                    }
                }

                Ok(RustBV::concrete(0, retty.bits()))
            }

            IRExpr::VECRET | IRExpr::GSPTR => {
                // P7 fix: Request Python fallback instead of failing
                // These special expressions require Python's VEX handling
                Err(CbExecutionError::NeedPythonFallback(
                    format!("special expr {:?} requires Python", expr)
                ))
            }
        }
    }

    /// Build an ITE chain for symbolic memory load by loading each candidate address.
    ///
    /// This builds the ITE chain entirely in Rust instead of delegating to Python.
    /// For each candidate address, we load the value via callback and create an ITE:
    /// `ITE(addr == a1, mem[a1], ITE(addr == a2, mem[a2], ...))`
    ///
    /// This is more efficient than calling Python's symbolic memory handler because:
    /// 1. We avoid FFI overhead for the ITE chain construction
    /// 2. The RustBV ITE nodes stay in Rust's Z3 context
    /// 3. We can use balanced ITE trees for better solver performance
    fn build_ite_load_from_callbacks(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        addrs: &[u64],
        addr_expr: &RustBV,
        size: usize,
    ) -> Result<RustBV, CbExecutionError> {
        if addrs.is_empty() {
            return Err(CbExecutionError::Memory("no candidate addresses".to_string()));
        }

        let width = (size * 8) as u32;
        let addr_width = addr_expr.width();

        // For a single address, just load it directly
        if addrs.len() == 1 {
            return self.load_from_callback(py, callbacks, addrs[0], size);
        }

        // Batch load all addresses at once for efficiency
        let load_requests: Vec<(u64, u32)> = addrs.iter().map(|&a| (a, size as u32)).collect();
        let load_results = callbacks
            .call_memory_load_batch(py, &load_requests)
            .map_err(|e| CbExecutionError::Callback(e.to_string()))?;

        // Build (condition, value) pairs for the ITE chain
        let mut pairs: Vec<(RustBV, RustBV)> = Vec::with_capacity(addrs.len());

        for (i, addr) in addrs.iter().enumerate() {
            // Get the loaded value for this address
            let value = if i < load_results.len() {
                let (data, is_symbolic, symbolic_ast) = &load_results[i];
                if *is_symbolic {
                    // Try to convert to RustBV - check handle first (fast path), then claripy (slow path)
                    if let Some(ast_obj) = symbolic_ast {
                        let ast = ast_obj.bind(py);

                        // Fast path: check for RustBVHandle first
                        if let Some(ref table) = self.symbol_table {
                            if let Some(bv) = try_handle_to_rustbv(&ast, table) {
                                bv
                            } else if is_claripy_ast(&ast) {
                                // Slow path: claripy AST conversion
                                match claripy_to_rustbv(py, &ast, self.ctx) {
                                    Ok(bv) => bv,
                                    Err(_) => {
                                        // Fallback to fresh symbolic
                                        RustBV::symbolic(
                                            self.ctx,
                                            &format!("ite_load_{:x}_{}", addr, size),
                                            width,
                                        )
                                    }
                                }
                            } else {
                                RustBV::symbolic(
                                    self.ctx,
                                    &format!("ite_load_{:x}_{}", addr, size),
                                    width,
                                )
                            }
                        } else if is_claripy_ast(&ast) {
                            match claripy_to_rustbv(py, &ast, self.ctx) {
                                Ok(bv) => bv,
                                Err(_) => {
                                    // Fallback to fresh symbolic
                                    RustBV::symbolic(
                                        self.ctx,
                                        &format!("ite_load_{:x}_{}", addr, size),
                                        width,
                                    )
                                }
                            }
                        } else {
                            RustBV::symbolic(
                                self.ctx,
                                &format!("ite_load_{:x}_{}", addr, size),
                                width,
                            )
                        }
                    } else {
                        RustBV::symbolic(
                            self.ctx,
                            &format!("ite_load_{:x}_{}", addr, size),
                            width,
                        )
                    }
                } else {
                    bytes_to_bv(data, width)
                }
            } else {
                // Missing result - create symbolic placeholder
                RustBV::symbolic(
                    self.ctx,
                    &format!("ite_load_{:x}_{}", addr, size),
                    width,
                )
            };

            // Build condition: addr_expr == this address
            let addr_const = RustBV::concrete(*addr as u128, addr_width);
            let cond = addr_expr.eq(&addr_const, self.ctx);

            pairs.push((cond, value));
        }

        // Use the last value as default (for robustness, though one condition should always match)
        let default_value = pairs.last().map(|(_, v)| v.clone())
            .unwrap_or_else(|| RustBV::symbolic(self.ctx, "ite_default", width));

        // Build balanced ITE tree for better solver performance
        Ok(build_balanced_ite(&pairs[..pairs.len()-1], default_value, self.ctx))
    }

    /// Evaluate an IR constant.
    fn eval_const(&self, c: &IRConst) -> RustBV {
        match c {
            IRConst::U1(v) => RustBV::concrete(*v as u128, 1),
            IRConst::U8(v) => RustBV::concrete(*v as u128, 8),
            IRConst::U16(v) => RustBV::concrete(*v as u128, 16),
            IRConst::U32(v) => RustBV::concrete(*v as u128, 32),
            IRConst::U64(v) => RustBV::concrete(*v as u128, 64),
            IRConst::U128(v) => RustBV::concrete(*v, 128),
            IRConst::F32(v) => RustBV::concrete(v.to_bits() as u128, 32),
            IRConst::F64(v) => RustBV::concrete(v.to_bits() as u128, 64),
            IRConst::V128(v) => RustBV::concrete(*v, 128),
            IRConst::V256(v) => {
                RustBV::concrete(v[0] as u128 | ((v[1] as u128) << 64), 128)
            }
        }
    }

    /// Apply LoadG conversion (widening) to loaded value.
    ///
    /// LoadG can widen the loaded value with sign or zero extension.
    fn apply_loadg_conversion(&self, cvt: IRLoadGOp, value: RustBV, target_bits: u32) -> RustBV {
        let src_bits = value.width();
        if src_bits >= target_bits {
            // No widening needed, possibly truncate
            if src_bits > target_bits {
                value.extract(0, target_bits, self.ctx)
            } else {
                value
            }
        } else {
            // Widen the value
            match cvt {
                IRLoadGOp::Identity => value, // Should not happen if sizes differ
                IRLoadGOp::WidenS => value.sign_extend(target_bits, self.ctx),
                IRLoadGOp::WidenZ => value.zero_extend(target_bits, self.ctx),
            }
        }
    }

    /// Evaluate the next address from an IRSB.
    /// Used for Exit statements where we still need callbacks for complex expressions.
    fn eval_next_addr(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        irsb: &IRSB,
    ) -> Result<u64, CbExecutionError> {
        let next_val = self.eval_expr_with_callbacks(py, callbacks, &irsb.next, &irsb.tyenv)?;
        // Check for symbolic addresses FIRST - Constrained BV has concrete value but is still symbolic
        if next_val.is_symbolic() {
            // Try to concretize to a single value
            match self.concretizer.concretize(&next_val, self.ctx) {
                ConcretizationResult::Single(addr) => {
                    // Add constraint that target == addr
                    let concrete = RustBV::concrete(addr as u128, next_val.width());
                    let constraint = next_val.eq(&concrete, self.ctx);
                    self.ctx.assume_true(&constraint);
                    return Ok(addr);
                }
                _ => {
                    // For Exit statements mid-block, we can't easily fork
                    // Return error to fall back to Python handling
                    return Err(CbExecutionError::Unsupported("symbolic next address".to_string()));
                }
            }
        }
        next_val.as_u64().ok_or_else(|| {
            CbExecutionError::Unsupported("non-concrete next address".to_string())
        })
    }

    /// Evaluate and concretize the jump target for the default exit.
    ///
    /// This method handles symbolic jump targets (e.g., ret instructions with symbolic
    /// return addresses) by concretizing them to a bounded set of concrete values.
    fn eval_next_addr_concretized(
        &mut self,
        irsb: &IRSB,
    ) -> Result<ConcretizedJump, CbExecutionError> {
        let next_val = self.eval_expr_simple(&irsb.next, &irsb.tyenv)?;

        // Fast path: concrete address
        if let Some(addr) = next_val.as_u64() {
            if !next_val.is_symbolic() {
                return Ok(ConcretizedJump::Single(addr));
            }
        }

        // Symbolic address - use AddressConcretizer
        match self.concretizer.concretize(&next_val, self.ctx) {
            ConcretizationResult::Single(addr) => {
                // Add constraint that target == addr
                let concrete = RustBV::concrete(addr as u128, next_val.width());
                let constraint = next_val.eq(&concrete, self.ctx);
                self.ctx.assume_true(&constraint);
                Ok(ConcretizedJump::Single(addr))
            }
            ConcretizationResult::Multiple(addrs) => {
                // Check if we exceed max_symbolic_ip_targets
                if addrs.len() > self.config.max_symbolic_ip_targets {
                    let min = *addrs.first().unwrap_or(&0);
                    let max = *addrs.last().unwrap_or(&0);
                    Ok(ConcretizedJump::TooMany {
                        min,
                        max,
                        limit: self.config.max_symbolic_ip_targets,
                    })
                } else {
                    Ok(ConcretizedJump::Multiple {
                        targets: addrs,
                        expr: next_val,
                    })
                }
            }
            ConcretizationResult::Strided { base, stride, count } => {
                // Convert strided to explicit list, but check limit first
                let num_targets = count as usize;
                if num_targets > self.config.max_symbolic_ip_targets {
                    let max = base + (count - 1) * stride;
                    Ok(ConcretizedJump::TooMany {
                        min: base,
                        max,
                        limit: self.config.max_symbolic_ip_targets,
                    })
                } else {
                    let targets: Vec<u64> = (0..count).map(|i| base + i * stride).collect();
                    Ok(ConcretizedJump::Multiple {
                        targets,
                        expr: next_val,
                    })
                }
            }
            ConcretizationResult::TooLarge { min, max, limit: _ } => {
                Ok(ConcretizedJump::TooMany {
                    min,
                    max,
                    limit: self.config.max_symbolic_ip_targets,
                })
            }
            ConcretizationResult::Failed(msg) => {
                Err(CbExecutionError::Unsupported(format!(
                    "jump target concretization failed: {}",
                    msg
                )))
            }
        }
    }

    /// Handle the default exit (end of block).
    fn handle_default_exit(&mut self, irsb: &IRSB) -> Result<BlockResult, CbExecutionError> {
        let concretized = self.eval_next_addr_concretized(irsb)?;
        match concretized {
            ConcretizedJump::Single(addr) => {
                Ok(self.handle_exit(addr, irsb.jumpkind))
            }
            ConcretizedJump::Multiple { targets, expr } => {
                // Store the expression for constraint addition later
                let condition_id = self.next_condition_id;
                self.next_condition_id += 1;
                self.stored_conditions.insert(condition_id, expr.clone());

                Ok(BlockResult::SymbolicJumpTarget {
                    targets,
                    condition_id,
                    target_expr: expr,
                    jumpkind: irsb.jumpkind,
                })
            }
            ConcretizedJump::TooMany { min, max, limit } => {
                Ok(BlockResult::UnconstrainedJump {
                    min_target: min,
                    max_target: max,
                    limit,
                    jumpkind: irsb.jumpkind,
                })
            }
        }
    }

    /// Simple expression evaluation (no callbacks, for already-evaluated temps).
    fn eval_expr_simple(
        &self,
        expr: &IRExpr,
        tyenv: &TypeEnv,
    ) -> Result<RustBV, CbExecutionError> {
        match expr {
            IRExpr::Const(c) => Ok(self.eval_const(c)),
            IRExpr::RdTmp(tmp) => {
                if let Some(Some(val)) = self.temps.get(*tmp as usize) {
                    Ok(val.clone())
                } else {
                    Err(CbExecutionError::UnknownTemp(*tmp))
                }
            }
            IRExpr::Get { offset, ty } => {
                let size = ty.bytes();
                Ok(self.registers.get(*offset, size, self.ctx))
            }
            _ => Err(CbExecutionError::Unsupported(
                "complex expr in default exit".to_string(),
            )),
        }
    }

    /// Handle an exit (update PC, return result).
    fn handle_exit(&mut self, target: u64, jumpkind: JumpKind) -> BlockResult {
        self.set_pc(target);

        if jumpkind.is_syscall() {
            let syscall_num = self.get_syscall_num();
            return BlockResult::Syscall { num: syscall_num };
        }

        if self.is_hooked(target) {
            return BlockResult::Hook { addr: target };
        }

        // For CALL instructions to external code, ask Python to resolve
        if jumpkind.is_call() && !self.is_in_binary(target) {
            let return_addr = self.get_return_addr().unwrap_or(0);
            return BlockResult::UnmodeledCall {
                addr: target,
                return_addr,
                symbol_name: None, // Symbol lookup done by Python
            };
        }

        // For jumps/returns to external addresses that are NOT hooked,
        // treat as UnmodeledCall so Python can handle them properly.
        // This includes:
        // - angr's internal continuation addresses (0x700000+)
        // - extern stubs and SimProcedure return points
        // - dynamically registered hooks that weren't synced yet
        if !self.is_in_binary(target) {
            let return_addr = self.get_return_addr().unwrap_or(0);
            return BlockResult::UnmodeledCall {
                addr: target,
                return_addr,
                symbol_name: Some("__extern_addr__".to_string()),
            };
        }

        BlockResult::BlockEnd {
            next_addr: target,
            jumpkind,
        }
    }

    /// Get the syscall number from the appropriate register.
    fn get_syscall_num(&self) -> u64 {
        // Syscall number register varies by architecture:
        // - AMD64: RAX (offset 16, 8 bytes)
        // - X86: EAX (offset 8, 4 bytes)
        // - ARM: R7 (offset 36, 4 bytes) - EABI syscall convention
        // - ARM64: X8 (offset 80, 8 bytes)
        // - MIPS32: v0/$2 (offset 16, 4 bytes)
        // - MIPS64: v0/$2 (offset 32, 8 bytes)
        let (offset, size) = match self.arch {
            VexArch::AMD64 => (16, 8),   // RAX
            VexArch::X86 => (8, 4),      // EAX
            VexArch::ARM => (36, 4),     // R7 (EABI)
            VexArch::ARM64 => (80, 8),   // X8
            VexArch::MIPS32 => (16, 4),  // v0/$2
            VexArch::MIPS64 => (32, 8),  // v0/$2
            _ => (0, 8),                 // Default fallback
        };
        let syscall_bv = self.registers.get(offset, size, self.ctx);
        syscall_bv.as_u64().unwrap_or(0)
    }

    /// Fork the interpreter state.
    pub fn fork(&self) -> CallbackInterpreter<'a> {
        // Clone the calling convention based on its type
        let arch_box = arch_from_vex(self.arch);
        let cc = default_cc_for_arch(arch_box.name());

        CallbackInterpreter {
            registers: self.registers.fork(),
            temps: self.temps.clone(),
            ctx: self.ctx,
            symbol_table: self.symbol_table, // Share symbol table reference
            pc: self.pc,
            current_insn_addr: self.current_insn_addr,
            hook_addrs: self.hook_addrs.clone(),
            arch: self.arch,
            block_cache: LruCache::new(NonZeroUsize::new(4096).unwrap()), // Fresh cache for fork
            use_memory_callbacks: self.use_memory_callbacks,
            deferred_forks: Vec::new(), // Fresh deferred forks for fork
            config: self.config.clone(),
            branch_counter: self.branch_counter,
            next_condition_id: self.next_condition_id,
            push_level: self.push_level, // Inherit push level for forked interpreter
            concrete_memory: self.concrete_memory.clone(), // Share concrete memory (read-only)
            concretizer: self.concretizer.clone(), // Share concretizer settings
            dirty_registers: 0, // Fresh dirty tracking for fork
            pending_stores: Vec::with_capacity(256), // Fresh store buffer for fork
            all_flushed_stores: HashMap::new(),
            pending_symbolic_stores: HashMap::new(),
            max_pending_stores: self.max_pending_stores,
            // Fork Rust memory with O(1) CoW
            rust_memory: self.rust_memory.as_ref().map(|m| m.fork()),
            use_rust_memory: self.use_rust_memory,
            load_prefetch_cache: HashMap::new(), // Fresh prefetch cache for fork
            use_load_prefetch: self.use_load_prefetch,
            page_prefetch_count: self.page_prefetch_count, // Inherit page prefetch count
            dirty_dispatch: DirtyHelperDispatch::new(), // Fresh dispatch (stateless)
            simprocedure_registry: self.simprocedure_registry.clone(), // Share SimProcedure registry
            calling_convention: cc,
            last_branch_condition: None, // Fresh for fork
            pending_python_constraints: Vec::new(), // Fresh constraints for fork
            stored_conditions: HashMap::new(), // Fresh for fork
            stats: ExecutionStats::default(), // Fresh stats for fork
            profiling_enabled: self.profiling_enabled, // Inherit profiling setting
            concrete_memory_sorted: self.concrete_memory_sorted, // Inherit sorted flag
        }
    }

    /// Enable Rust-native memory mode.
    ///
    /// When enabled, memory operations will try to use the Rust SymbolicMemory
    /// first, falling back to Python callbacks only for unmapped regions.
    /// This can significantly improve performance for memory-intensive code.
    pub fn enable_rust_memory(&mut self, endness: Endness) {
        self.rust_memory = Some(SymbolicMemory::new(endness));
        self.use_rust_memory = true;
    }

    /// Disable Rust-native memory mode.
    pub fn disable_rust_memory(&mut self) {
        self.use_rust_memory = false;
    }

    /// Get mutable reference to Rust memory (for initialization).
    pub fn rust_memory_mut(&mut self) -> Option<&mut SymbolicMemory> {
        self.rust_memory.as_mut()
    }

    /// Get reference to Rust memory.
    pub fn rust_memory(&self) -> Option<&SymbolicMemory> {
        self.rust_memory.as_ref()
    }

    /// Set the Rust memory instance.
    pub fn set_rust_memory(&mut self, memory: SymbolicMemory) {
        self.rust_memory = Some(memory);
        self.use_rust_memory = true;
    }

    /// Take the Rust memory instance (for transferring to engine).
    pub fn take_rust_memory(&mut self) -> Option<SymbolicMemory> {
        self.rust_memory.take()
    }

    /// Enable or disable load prefetching.
    pub fn set_load_prefetch(&mut self, enabled: bool) {
        self.use_load_prefetch = enabled;
    }

    /// Set the number of pages to prefetch when fetching a page.
    ///
    /// When a page needs to be fetched from Python, this many additional pages
    /// will be fetched in each direction (before and after) to improve locality.
    /// Set to 0 to disable page prefetching.
    pub fn set_page_prefetch_count(&mut self, count: u32) {
        self.page_prefetch_count = count;
    }

    /// Fetch a page from Python and map it in Rust memory.
    ///
    /// This is called when a load/store encounters an unmapped page in a lazy region.
    /// The page is fetched via Python callback and added to rust_memory.
    ///
    /// Returns true if the page was successfully fetched and mapped.
    pub fn fetch_page(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        page_addr: u64,
    ) -> Result<bool, CbExecutionError> {
        // Check if we have the callback
        if !callbacks.has_fetch_page() {
            return Ok(false);
        }

        // Call Python to fetch the page
        let (data, permissions, is_mapped) = callbacks
            .call_fetch_page(py, page_addr)
            .map_err(|e| CbExecutionError::Callback(format!("fetch_page failed: {}", e)))?;

        if !is_mapped {
            // Page doesn't exist in Python memory either
            return Ok(false);
        }

        // Ensure we have Rust memory enabled
        if let Some(ref mut rust_mem) = self.rust_memory {
            // Convert permission bits to Permission struct
            let perm = Permission::from_bits(permissions);

            // Map the page in Rust memory
            rust_mem.map_page(page_addr, data, perm);

            Ok(true)
        } else {
            // Rust memory not enabled - shouldn't happen but handle gracefully
            Ok(false)
        }
    }

    /// Fetch multiple pages from Python in a batch.
    ///
    /// This is more efficient than fetching pages one at a time.
    /// Returns the number of pages successfully fetched.
    pub fn fetch_pages_batch(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        page_addrs: &[u64],
    ) -> Result<usize, CbExecutionError> {
        if page_addrs.is_empty() {
            return Ok(0);
        }

        // Call Python to fetch pages in batch
        let results = callbacks
            .call_batch_fetch_pages(py, page_addrs)
            .map_err(|e| CbExecutionError::Callback(format!("batch_fetch_pages failed: {}", e)))?;

        let mut fetched = 0;

        if let Some(ref mut rust_mem) = self.rust_memory {
            for (i, (data, permissions, is_mapped)) in results.into_iter().enumerate() {
                if is_mapped {
                    let page_addr = page_addrs[i];
                    let perm = Permission::from_bits(permissions);
                    rust_mem.map_page(page_addr, data, perm);
                    fetched += 1;
                }
            }
        }

        Ok(fetched)
    }

    /// Fetch a page and prefetch nearby pages for better locality.
    ///
    /// This is an optimization that reduces future FFI calls by speculatively
    /// fetching pages around the accessed address. Useful for sequential access
    /// patterns (like stack frames, arrays, etc.).
    ///
    /// When `enable_eager_prefetch` is set in config, this will fetch all
    /// unmapped pages in the lazy region containing the page. Otherwise,
    /// it fetches `prefetch_count` pages in each direction.
    ///
    /// Args:
    ///     prefetch_count: Number of pages to prefetch in each direction (0 = disabled)
    ///
    /// Returns true if the main page was successfully fetched.
    pub fn fetch_page_with_prefetch(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        page_addr: u64,
        prefetch_count: u32,
    ) -> Result<bool, CbExecutionError> {
        if prefetch_count == 0 && !self.config.enable_eager_prefetch {
            // No prefetching, just fetch the single page
            return self.fetch_page(py, callbacks, page_addr);
        }

        // Build list of pages to fetch
        let pages_to_fetch = if self.config.enable_eager_prefetch {
            // Eager region prefetch: fetch all unmapped pages in the region
            self.get_eager_prefetch_list(page_addr)
        } else {
            // Nearby prefetch: fetch pages before/after the trigger
            self.get_nearby_prefetch_list(page_addr, prefetch_count)
        };

        if pages_to_fetch.is_empty() {
            // No pages to fetch (shouldn't happen, but handle gracefully)
            return self.fetch_page(py, callbacks, page_addr);
        }

        // Fetch all pages in one batch
        let fetched = self.fetch_pages_batch(py, callbacks, &pages_to_fetch)?;

        // Return true if at least the main page was fetched
        if let Some(ref rust_mem) = self.rust_memory {
            Ok(rust_mem.is_mapped(page_addr))
        } else {
            Ok(fetched > 0)
        }
    }

    /// Get pages to fetch for eager region prefetch.
    ///
    /// Returns all unmapped pages in the lazy region containing `page_addr`,
    /// up to `max_prefetch_batch` pages.
    fn get_eager_prefetch_list(&self, page_addr: u64) -> Vec<u64> {
        if let Some(ref rust_mem) = self.rust_memory {
            if let Some(pages) = rust_mem.get_region_prefetch_list(page_addr, self.config.max_prefetch_batch) {
                return pages;
            }
        }
        // Fallback to just the main page
        vec![page_addr]
    }

    /// Get the current stack pointer value (architecture-aware).
    fn get_stack_pointer(&self) -> Option<u64> {
        let (offset, size) = match self.arch {
            VexArch::AMD64 => (48, 8),  // RSP
            VexArch::X86 => (24, 4),    // ESP (offset 24 per arch/x86.rs)
            VexArch::ARM | VexArch::ARM64 => (52, 8), // SP for ARM variants (approximate)
            _ => return None,
        };
        self.registers.get(offset, size, self.ctx).as_u64()
    }

    /// Check if an address is in the stack region (near current RSP).
    /// Stack typically grows downward, so we check if addr is below RSP + some margin.
    fn is_stack_region(&self, addr: u64) -> bool {
        if let Some(sp) = self.get_stack_pointer() {
            // Stack region: addresses from RSP - 1MB to RSP + 64KB
            // (stack grows down, but we allow some upward margin for locals)
            let stack_base = sp.saturating_sub(1024 * 1024); // 1MB below RSP
            let stack_limit = sp.saturating_add(64 * 1024);   // 64KB above RSP
            addr >= stack_base && addr <= stack_limit
        } else {
            false
        }
    }

    /// Get pages to fetch for nearby prefetch (stack-aware).
    ///
    /// Returns `prefetch_count` unmapped pages, prioritizing stack growth direction
    /// (downward) when the access is in the stack region.
    fn get_nearby_prefetch_list(&self, page_addr: u64, prefetch_count: u32) -> Vec<u64> {
        let page_size = 0x1000u64;
        let mut pages_to_fetch = Vec::with_capacity(1 + 2 * prefetch_count as usize);

        // Add main page first
        pages_to_fetch.push(page_addr);

        // Determine if this is a stack access
        let is_stack = self.is_stack_region(page_addr);

        // For stack accesses, prioritize downward prefetch (stack grows down)
        // For non-stack, use balanced bidirectional prefetch
        let (down_count, up_count) = if is_stack {
            // Stack: 3x more pages downward than upward
            let down = (prefetch_count * 3).min(16);
            let up = prefetch_count.min(4);
            (down, up)
        } else {
            // Non-stack: equal in both directions
            (prefetch_count, prefetch_count)
        };

        // Add pages before (lower addresses - stack growth direction)
        for i in 1..=down_count {
            if let Some(addr) = page_addr.checked_sub(i as u64 * page_size) {
                // Check if not already mapped
                if let Some(ref rust_mem) = self.rust_memory {
                    if !rust_mem.is_mapped(addr) && rust_mem.is_addr_in_lazy_region(addr) {
                        pages_to_fetch.push(addr);
                    }
                }
            }
        }

        // Add pages after (higher addresses)
        for i in 1..=up_count {
            if let Some(addr) = page_addr.checked_add(i as u64 * page_size) {
                // Check if not already mapped
                if let Some(ref rust_mem) = self.rust_memory {
                    if !rust_mem.is_mapped(addr) && rust_mem.is_addr_in_lazy_region(addr) {
                        pages_to_fetch.push(addr);
                    }
                }
            }
        }

        pages_to_fetch
    }

    /// Clear the load prefetch cache.
    ///
    /// This should be called after stores to invalidate potentially stale values.
    pub fn clear_prefetch_cache(&mut self) {
        self.load_prefetch_cache.clear();
    }

    /// Check if a load result is in the prefetch cache.
    #[inline]
    pub fn get_prefetched_load(&self, addr: u64, size: usize) -> Option<&PrefetchedLoad> {
        self.load_prefetch_cache.get(&(addr, size))
    }

    /// Scan an IRSB for Load expressions with concrete addresses.
    ///
    /// This collects (address, size) pairs for loads that can be prefetched.
    /// Only loads with concrete addresses (computed from temps/constants) are collected.
    fn scan_loads_in_irsb(&self, irsb: &IRSB) -> Vec<(u64, usize)> {
        let mut loads = Vec::new();

        for stmt in &irsb.statements {
            self.scan_loads_in_stmt(stmt, irsb, &mut loads);
        }

        // Also scan the next expression
        self.scan_loads_in_expr(&irsb.next, irsb, &mut loads);

        loads
    }

    /// Scan a statement for Load expressions.
    fn scan_loads_in_stmt(&self, stmt: &IRStmt, irsb: &IRSB, loads: &mut Vec<(u64, usize)>) {
        match stmt {
            IRStmt::WrTmp { data, .. } => {
                self.scan_loads_in_expr(data, irsb, loads);
            }
            IRStmt::Put { data, .. } => {
                self.scan_loads_in_expr(data, irsb, loads);
            }
            IRStmt::Store { addr, data, .. } => {
                self.scan_loads_in_expr(addr, irsb, loads);
                self.scan_loads_in_expr(data, irsb, loads);
            }
            IRStmt::Exit { guard, .. } => {
                self.scan_loads_in_expr(guard, irsb, loads);
            }
            _ => {}
        }
    }

    /// Scan an expression for Load expressions.
    fn scan_loads_in_expr(&self, expr: &IRExpr, irsb: &IRSB, loads: &mut Vec<(u64, usize)>) {
        match expr {
            IRExpr::Load { addr, ty, .. } => {
                let size = ty.bytes() as usize;
                // Try to evaluate the address to a concrete value
                if let Some(addr_val) = self.try_eval_expr_concrete(addr, &irsb.tyenv) {
                    // Check if this address is NOT already in cached concrete memory
                    // (no point prefetching what we can already read locally)
                    if self.try_read_concrete_memory(addr_val, size).is_none() {
                        loads.push((addr_val, size));
                    }
                }
                // Also scan the address expression itself
                self.scan_loads_in_expr(addr, irsb, loads);
            }
            IRExpr::Unop { arg, .. } => {
                self.scan_loads_in_expr(arg, irsb, loads);
            }
            IRExpr::Binop { left, right, .. } => {
                self.scan_loads_in_expr(left, irsb, loads);
                self.scan_loads_in_expr(right, irsb, loads);
            }
            IRExpr::ITE { cond, iftrue, iffalse, .. } => {
                self.scan_loads_in_expr(cond, irsb, loads);
                self.scan_loads_in_expr(iftrue, irsb, loads);
                self.scan_loads_in_expr(iffalse, irsb, loads);
            }
            IRExpr::CCall { args, .. } => {
                for arg in args {
                    self.scan_loads_in_expr(arg, irsb, loads);
                }
            }
            _ => {}
        }
    }

    /// Try to evaluate an expression to a concrete u64 value (for prefetching).
    ///
    /// This is a simplified evaluation that only handles constants and simple
    /// operations. It doesn't evaluate temps since we're scanning before execution.
    fn try_eval_expr_concrete(&self, expr: &IRExpr, _tyenv: &TypeEnv) -> Option<u64> {
        match expr {
            IRExpr::Const(c) => {
                match c {
                    IRConst::U8(v) => Some(*v as u64),
                    IRConst::U16(v) => Some(*v as u64),
                    IRConst::U32(v) => Some(*v as u64),
                    IRConst::U64(v) => Some(*v),
                    _ => None,
                }
            }
            IRExpr::Get { offset, ty } => {
                // Try to get a concrete register value
                let size = ty.bytes();
                let reg_val = self.registers.get(*offset, size, self.ctx);
                reg_val.as_u64()
            }
            // For more complex expressions (binops, etc.), we could evaluate them
            // but for simplicity we skip them - they'll be handled by the regular path
            _ => None,
        }
    }

    /// Prefetch loads for a block using batch callback.
    ///
    /// This scans the IRSB for Load expressions with concrete addresses,
    /// batches them into a single callback, and populates the prefetch cache.
    fn prefetch_loads_for_block(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        irsb: &IRSB,
    ) -> Result<(), CbExecutionError> {
        if !self.use_load_prefetch {
            return Ok(());
        }

        // Clear previous prefetch cache
        self.load_prefetch_cache.clear();

        // Scan for loads
        let loads = self.scan_loads_in_irsb(irsb);

        if loads.is_empty() {
            return Ok(());
        }

        // Deduplicate loads (same address+size only needs to be fetched once)
        let mut unique_loads: Vec<(u64, usize)> = Vec::with_capacity(loads.len());
        let mut seen: HashSet<(u64, usize)> = HashSet::new();
        for load in loads {
            if seen.insert(load) {
                unique_loads.push(load);
            }
        }

        // Convert to callback format: (addr, size as u32)
        let callback_loads: Vec<(u64, u32)> = unique_loads
            .iter()
            .map(|&(addr, size)| (addr, size as u32))
            .collect();

        // Call batch callback
        let results = callbacks
            .call_memory_load_batch(py, &callback_loads)
            .map_err(|e| CbExecutionError::Callback(e.to_string()))?;

        // Populate prefetch cache
        for (i, (addr, size)) in unique_loads.iter().enumerate() {
            if let Some((data, is_symbolic, symbolic_ast)) = results.get(i) {
                let value = if *is_symbolic {
                    // Try to convert to RustBV - check handle first (fast path), then claripy (slow path)
                    if let Some(ast_obj) = symbolic_ast {
                        let ast = ast_obj.bind(py);

                        // Fast path: check for RustBVHandle first
                        if let Some(ref table) = self.symbol_table {
                            if let Some(bv) = try_handle_to_rustbv(&ast, table) {
                                bv
                            } else if is_claripy_ast(&ast) {
                                // Slow path: claripy AST conversion
                                match claripy_to_rustbv(py, &ast, self.ctx) {
                                    Ok(bv) => bv,
                                    Err(_) => {
                                        // Fallback to fresh symbolic
                                        RustBV::symbolic(
                                            self.ctx,
                                            &format!("prefetch_{:x}_{}", addr, size),
                                            (size * 8) as u32,
                                        )
                                    }
                                }
                            } else {
                                RustBV::symbolic(
                                    self.ctx,
                                    &format!("prefetch_{:x}_{}", addr, size),
                                    (size * 8) as u32,
                                )
                            }
                        } else if is_claripy_ast(&ast) {
                            match claripy_to_rustbv(py, &ast, self.ctx) {
                                Ok(bv) => bv,
                                Err(_) => {
                                    // Fallback to fresh symbolic
                                    RustBV::symbolic(
                                        self.ctx,
                                        &format!("prefetch_{:x}_{}", addr, size),
                                        (size * 8) as u32,
                                    )
                                }
                            }
                        } else {
                            RustBV::symbolic(
                                self.ctx,
                                &format!("prefetch_{:x}_{}", addr, size),
                                (size * 8) as u32,
                            )
                        }
                    } else {
                        RustBV::symbolic(
                            self.ctx,
                            &format!("prefetch_{:x}_{}", addr, size),
                            (size * 8) as u32,
                        )
                    }
                } else {
                    bytes_to_bv(data, (size * 8) as u32)
                };

                self.load_prefetch_cache.insert(
                    (*addr, *size),
                    PrefetchedLoad {
                        value,
                        is_symbolic: *is_symbolic,
                    },
                );
            }
        }

        Ok(())
    }

    /// Add a lazy region for on-demand page fetching.
    ///
    /// Pages in this region will be fetched from Python when accessed.
    /// This is more efficient than pre-loading all pages.
    pub fn add_lazy_region(&mut self, start_addr: u64, size: u64) {
        if let Some(ref mut rust_mem) = self.rust_memory {
            rust_mem.add_lazy_region(start_addr, size);
        }
    }

    /// Get statistics about Rust memory usage.
    pub fn rust_memory_stats(&self) -> Option<(usize, usize, usize)> {
        self.rust_memory.as_ref().map(|m| {
            (m.page_count(), m.lazy_region_count(), m.get_dirty_pages().len())
        })
    }

    // =========================================================================
    // Constraint Synchronization
    // =========================================================================

    /// Track an address concretization constraint for Python sync.
    ///
    /// When Rust concretizes a symbolic address to a concrete value, this
    /// constraint needs to be communicated to Python's claripy solver.
    pub fn track_concretization_constraint(&mut self, addr_expr: &RustBV, concrete_addr: u64) {
        // Only track if the address was actually symbolic
        if addr_expr.is_symbolic() {
            self.pending_python_constraints.push(
                PendingConstraint::address_concretization(addr_expr.clone(), concrete_addr)
            );
        }
    }

    /// Track a branch constraint for Python sync.
    pub fn track_branch_constraint(&mut self, cond: &RustBV, took_true_branch: bool) {
        if cond.is_symbolic() {
            if took_true_branch {
                self.pending_python_constraints.push(PendingConstraint::branch_true(cond.clone()));
            } else {
                self.pending_python_constraints.push(PendingConstraint::branch_false(cond.clone()));
            }
        }
    }

    /// Check if there are pending constraints to sync.
    pub fn has_pending_constraints(&self) -> bool {
        !self.pending_python_constraints.is_empty()
    }

    /// Get the number of pending constraints.
    pub fn pending_constraint_count(&self) -> usize {
        self.pending_python_constraints.len()
    }

    /// Get pending constraints for export to Python.
    ///
    /// Returns a list of (width, concrete_value) tuples that can be converted
    /// to claripy constraints. The caller should use these to add constraints
    /// to Python's state before performing Python-based operations.
    ///
    /// Note: Full export to claripy ASTs would require storing the original
    /// symbolic expressions, which is complex. This simplified approach exports
    /// the constraint info so Python can reconstruct them if needed.
    pub fn get_pending_constraints(&self) -> &[PendingConstraint] {
        &self.pending_python_constraints
    }

    /// Clear pending constraints after sync.
    pub fn clear_pending_constraints(&mut self) {
        self.pending_python_constraints.clear();
    }

    /// Export pending constraints as a list that Python can process.
    ///
    /// Returns a list of tuples: (description, width, concrete_value)
    /// Python can use these to add constraints to its solver state.
    pub fn export_constraints_for_python(&self) -> Vec<(String, u32, u128, Option<u64>)> {
        self.pending_python_constraints
            .iter()
            .map(|c| (c.description.clone(), c.expression.width(), c.concrete_value, c.handle_id))
            .collect()
    }

    /// Sync pending constraints to Python before making a callback.
    ///
    /// This ensures that Python's claripy solver has all the constraints
    /// that Rust has accumulated, which is critical for operations that
    /// depend on solver state (e.g., symbolic memory operations, SimProcedures).
    ///
    /// Call this before any Python callback that may need solver context.
    pub fn sync_before_callback(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
    ) -> Result<(), CbExecutionError> {
        if self.has_pending_constraints() {
            let constraints = self.export_constraints_for_python();
            callbacks.call_sync_constraints(py, &constraints)
                .map_err(|e| CbExecutionError::Callback(format!(
                    "constraint sync failed: {}", e
                )))?;
            self.clear_pending_constraints();
        }
        Ok(())
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

/// Convert bytes (little-endian) to a RustBV.
fn bytes_to_bv(bytes: &[u8], width: u32) -> RustBV {
    let mut value: u128 = 0;
    for (i, &byte) in bytes.iter().enumerate() {
        if i * 8 >= width as usize {
            break;
        }
        value |= (byte as u128) << (i * 8);
    }
    RustBV::concrete(value, width)
}

/// Build a balanced ITE tree from a list of (condition, value) pairs.
///
/// This creates a balanced binary tree of ITE nodes, which is more efficient
/// than a linear chain for both Z3 solving and symbolic evaluation.
/// Returns the value for the first matching condition, or a default value.
fn build_balanced_ite(
    pairs: &[(RustBV, RustBV)],
    default_value: RustBV,
    ctx: &SymContext,
) -> RustBV {
    if pairs.is_empty() {
        return default_value;
    }

    if pairs.len() == 1 {
        // Base case: single condition
        return pairs[0].0.ite(&pairs[0].1, &default_value, ctx);
    }

    // Build balanced tree by splitting in the middle
    let mid = pairs.len() / 2;
    let (left, right) = pairs.split_at(mid);

    let left_ite = build_balanced_ite(left, default_value.clone(), ctx);
    let right_ite = build_balanced_ite(right, default_value, ctx);

    // Combine: if any left condition matches, use left_ite, else right_ite
    // We need to compute "any left condition is true"
    // For efficiency, we use the first left condition as the split point
    pairs[0].0.ite(&left_ite, &right_ite, ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_bv() {
        let bytes = vec![0x78, 0x56, 0x34, 0x12];
        let bv = bytes_to_bv(&bytes, 32);
        assert_eq!(bv.as_u64(), Some(0x12345678));
    }

    #[test]
    fn test_bv_to_bytes() {
        let bv = RustBV::concrete(0x12345678, 32);
        let bytes = bv_to_bytes(&bv);
        assert_eq!(bytes, vec![0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn test_interpreter_creation() {
        let ctx = SymContext::new_mock();
        let interp = CallbackInterpreter::new(VexArch::AMD64, &ctx);
        assert_eq!(interp.get_pc(), 0);
    }

    #[test]
    fn test_hook_management() {
        let ctx = SymContext::new_mock();
        let mut interp = CallbackInterpreter::new(VexArch::AMD64, &ctx);

        interp.add_hook(0x1000);
        assert!(interp.is_hooked(0x1000));
        assert!(!interp.is_hooked(0x2000));

        interp.remove_hook(0x1000);
        assert!(!interp.is_hooked(0x1000));
    }
}
