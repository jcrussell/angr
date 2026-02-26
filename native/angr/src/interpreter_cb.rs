//! Callback-aware VEX IR interpreter.
//!
//! This interpreter uses Python callbacks for memory operations instead of
//! local SymbolicMemory. It can run multiple blocks in a loop, returning
//! to Python only when an event requires Python handling.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;

use lru::LruCache;
use pyo3::prelude::*;

use crate::arch::{arch_from_vex, RegisterFile};
use crate::callbacks::{BranchPolicy, DeferredFork, ExecutionConfig, PythonCallbacks, RunResult};
use crate::claripy_bridge::{claripy_to_rustbv, is_claripy_ast, rustbv_to_claripy};
use crate::concretize::{AddressConcretizer, ConcretizationResult};
use crate::memory::{MemoryError, Permission, SymbolicMemory, PAGE_SIZE};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::ccall;
use crate::vex::dirty::DirtyHelperDispatch;
use crate::vex::ir::{IRConst, IRExpr, IRLoadGOp, IRStmt, IRType, JumpKind, TypeEnv, VexArch, IRSB};
use crate::vex::ops::{OpError, VEXOps};
use crate::vex::{deserialize_irsb, Endness};

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
        }
    }
}

impl std::error::Error for CbExecutionError {}

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
    /// Current program counter.
    pub pc: u64,
    /// Current instruction address (within block).
    current_insn_addr: u64,
    /// Hook addresses (return to Python when hit).
    hook_addrs: HashSet<u64>,
    /// VEX architecture.
    arch: VexArch,
    /// Block cache (shared across runs).
    block_cache: LruCache<u64, IRSB>,
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
}

impl<'a> CallbackInterpreter<'a> {
    /// Create a new callback-aware interpreter.
    pub fn new(arch: VexArch, ctx: &'a SymContext) -> Self {
        Self::with_config(arch, ctx, ExecutionConfig::default())
    }

    /// Create a new callback-aware interpreter with custom config.
    pub fn with_config(arch: VexArch, ctx: &'a SymContext, config: ExecutionConfig) -> Self {
        let arch_box = arch_from_vex(arch);

        CallbackInterpreter {
            registers: RegisterFile::new(arch_box),
            temps: Vec::new(),
            ctx,
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
            max_pending_stores: 256,
            rust_memory: None,
            use_rust_memory: false,
            load_prefetch_cache: HashMap::new(),
            use_load_prefetch: false, // Disabled by default - adds overhead for most workloads
            page_prefetch_count: 2,    // Prefetch 2 pages in each direction by default
            dirty_dispatch: DirtyHelperDispatch::new(),
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

    /// Add a concrete memory region for fast local access.
    ///
    /// This allows the interpreter to read from binary sections (e.g., .text, .rodata)
    /// without going through Python callbacks, significantly improving performance.
    pub fn add_concrete_memory(&mut self, base: u64, data: Vec<u8>) {
        let size = data.len() as u64;
        self.concrete_memory.push(ConcreteMemoryRegion { base, size, data });
    }

    /// Clear all concrete memory regions.
    pub fn clear_concrete_memory(&mut self) {
        self.concrete_memory.clear();
    }

    /// Try to read from concrete memory cache.
    /// Returns Some(data) if the address range is fully contained in a cached region.
    #[inline]
    fn try_read_concrete_memory(&self, addr: u64, size: usize) -> Option<&[u8]> {
        for region in &self.concrete_memory {
            if let Some(data) = region.read(addr, size) {
                return Some(data);
            }
        }
        None
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
            // Try to convert claripy AST to RustBV
            if let Some(ast_obj) = symbolic_ast {
                let ast = ast_obj.bind(py);
                if is_claripy_ast(&ast) {
                    match claripy_to_rustbv(py, &ast, self.ctx) {
                        Ok(bv) => return Ok(bv),
                        Err(_e) => {
                            // Fall back to creating a fresh symbolic value
                            // (claripy conversion can fail for complex/unsupported ops)
                        }
                    }
                }
            }
            // Fallback: create a fresh symbolic value
            Ok(RustBV::symbolic(
                self.ctx,
                &format!("mem_{:x}_{}", addr_concrete, size),
                (size * 8) as u32,
            ))
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

        // Try batch callback first, fall back to individual stores
        callbacks
            .call_memory_store_batch(py, &self.pending_stores)
            .map_err(|e| CbExecutionError::Callback(e.to_string()))?;

        self.pending_stores.clear();
        Ok(())
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

    /// Check if we have a cached block at the given address.
    pub fn has_cached_block(&self, addr: u64) -> bool {
        self.block_cache.contains(&addr)
    }

    /// Add a block to the cache.
    pub fn cache_block(&mut self, addr: u64, irsb: IRSB) {
        self.block_cache.put(addr, irsb);
    }

    /// Get a block from the cache.
    pub fn get_cached_block(&mut self, addr: u64) -> Option<&IRSB> {
        self.block_cache.get(&addr)
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
        let mut blocks_executed = 0u32;

        // Clear any previous deferred forks
        self.deferred_forks.clear();

        for _ in 0..max_blocks {
            // Check for hook at current PC
            if self.is_hooked(self.pc) {
                let forks = self.take_deferred_forks();
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
    ) -> Result<IRSB, CbExecutionError> {
        // Check cache first
        if let Some(irsb) = self.block_cache.get(&addr) {
            return Ok(irsb.clone());
        }

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
                        // Use up to 4096 bytes for lifting (typical max block size)
                        let max_bytes = available.min(4096);
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
                                    self.block_cache.put(addr, irsb.clone());
                                    return Ok(irsb);
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
        let irsb_json = callbacks
            .call_lift_block(py, addr)
            .map_err(|e| CbExecutionError::LiftError(format!("lift callback failed: {}", e)))?;

        let irsb = deserialize_irsb(&irsb_json)
            .map_err(|e| CbExecutionError::LiftError(format!("IRSB deserialization failed: {}", e)))?;

        // Cache it
        self.block_cache.put(addr, irsb.clone());

        Ok(irsb)
    }

    /// Execute a block using Python callbacks for memory access.
    fn execute_block_with_callbacks(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        irsb: &IRSB,
    ) -> Result<BlockResult, CbExecutionError> {
        // Reset temps for this block
        self.temps = vec![None; irsb.tyenv.types.len()];
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
                    return Ok(BlockResult::SymbolicBranch {
                        condition_id: 0, // TODO: proper condition tracking
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

                // Try Rust-native memory first if enabled
                if self.use_rust_memory {
                    if let Some(ref mut rust_mem) = self.rust_memory {
                        match rust_mem.store_symbolic(addr_val.clone(), data_val.clone(), self.ctx, &self.concretizer) {
                            Ok(()) => {
                                // Invalidate prefetch cache for this address
                                if let Some(addr_concrete) = addr_val.as_u64() {
                                    self.load_prefetch_cache.remove(&(addr_concrete, data_size));
                                } else {
                                    // Symbolic address - clear entire cache
                                    self.load_prefetch_cache.clear();
                                }
                                return Ok(StmtResult::Continue);
                            }
                            Err(MemoryError::UnmappedPageInRegion { page_addr }) => {
                                // Page is in a lazy region - try to fetch it from Python with prefetch
                                let prefetch_count = self.page_prefetch_count;
                                let page_fetched = self.fetch_page_with_prefetch(py, callbacks, page_addr, prefetch_count)?;

                                if !page_fetched {
                                    // Python doesn't have the page - auto-map a zero page
                                    // This avoids falling back to Python for uninitialized memory
                                    if let Some(ref mut rust_mem) = self.rust_memory {
                                        rust_mem.auto_map_zero_page(page_addr);
                                    }
                                }

                                // Retry the store (whether from fetch or auto-map)
                                if let Some(ref mut rust_mem) = self.rust_memory {
                                    match rust_mem.store_symbolic(addr_val.clone(), data_val.clone(), self.ctx, &self.concretizer) {
                                        Ok(()) => {
                                            if let Some(addr_concrete) = addr_val.as_u64() {
                                                self.load_prefetch_cache.remove(&(addr_concrete, data_size));
                                            } else {
                                                self.load_prefetch_cache.clear();
                                            }
                                            return Ok(StmtResult::Continue);
                                        }
                                        Err(_e) => {
                                            // Still failed - fall through to Python callback
                                        }
                                    }
                                }
                                // Fall through to Python callback
                            }
                            Err(MemoryError::Unmapped { .. }) => {
                                // Fall through to Python callback for unmapped pages
                            }
                            Err(MemoryError::SymbolicAddress { .. }) => {
                                // Fall through - Python has better symbolic handling
                            }
                            Err(e) => {
                                return Err(CbExecutionError::Memory(e.to_string()));
                            }
                        }
                    }
                }

                // Store via callback - handle symbolic addresses
                if let Some(addr_concrete) = addr_val.as_u64() {
                    // Fast path: concrete address - buffer for batch processing
                    let data_bytes = bv_to_bytes(&data_val);

                    // Invalidate prefetch cache for this address
                    self.load_prefetch_cache.remove(&(addr_concrete, data_size));

                    // Buffer the store instead of immediate callback
                    self.pending_stores.push((addr_concrete, data_bytes));

                    // Auto-flush if buffer is full
                    if self.pending_stores.len() >= self.max_pending_stores {
                        self.flush_stores(py, callbacks)?;
                    }
                } else {
                    // Symbolic address - clear entire prefetch cache
                    self.load_prefetch_cache.clear();
                    // Symbolic address - flush buffer first, then handle specially
                    self.flush_stores(py, callbacks)?;
                    // Symbolic address - try to concretize
                    match self.concretizer.concretize(&addr_val, self.ctx) {
                        ConcretizationResult::Single(addr_concrete) => {
                            let data_bytes = bv_to_bytes(&data_val);
                            callbacks
                                .call_memory_store(py, addr_concrete, &data_bytes)
                                .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                        }
                        ConcretizationResult::Multiple(addrs) => {
                            // Delegate to Python for conditional stores
                            callbacks
                                .call_memory_store_symbolic(py, &addrs, &data_val, &addr_val)
                                .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                        }
                        ConcretizationResult::Strided { base, stride, count } => {
                            // Strided access pattern - generate addresses and delegate to Python
                            let addrs: Vec<u64> = (0..count).map(|i| base + i * stride).collect();
                            callbacks
                                .call_memory_store_symbolic(py, &addrs, &data_val, &addr_val)
                                .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                        }
                        ConcretizationResult::TooLarge { min, max, .. } => {
                            // Address range too large - delegate to Python's memory model
                            // which has access to angr's address concretization strategies
                            let data_bytes = bv_to_bytes(&data_val);
                            callbacks
                                .call_memory_store_symbolic_ast(py, &data_bytes, data_bytes.len() as u32)
                                .map_err(|e| CbExecutionError::Callback(format!(
                                    "symbolic store AST callback failed at 0x{:x}-0x{:x}: {}",
                                    min, max, e
                                )))?;
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

                // Guard is symbolic - check both possibilities
                let can_be_true = self.ctx.can_be_true(&guard_val);
                let can_be_false = self.ctx.can_be_false(&guard_val);

                if can_be_true && can_be_false {
                    // Both paths are feasible
                    if !self.config.use_deferred_forks {
                        // Deferred forks disabled - return to Python for proper state forking
                        let fallthrough = self.eval_next_addr(py, callbacks, irsb)?;
                        return Ok(StmtResult::SymbolicBranch {
                            condition: guard_val,
                            true_target: *dst,
                            false_target: fallthrough,
                        });
                    }

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
                    let deferred = DeferredFork {
                        branch_addr: self.current_insn_addr,
                        path_taken: true,
                        unexplored_target: fallthrough,
                        condition_id: self.next_cond_id(),
                        push_level: self.push_level,
                        condition_ast,
                    };
                    self.deferred_forks.push(deferred);

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
                            let data_bytes = bv_to_bytes(&data_val);
                            self.load_prefetch_cache.remove(&(addr_concrete, data_size));
                            self.pending_stores.push((addr_concrete, data_bytes));
                            if self.pending_stores.len() >= self.max_pending_stores {
                                self.flush_stores(py, callbacks)?;
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
                        let ite_bytes = bv_to_bytes(&ite_result);
                        self.load_prefetch_cache.remove(&(addr_concrete, data_size));
                        self.pending_stores.push((addr_concrete, ite_bytes));
                        if self.pending_stores.len() >= self.max_pending_stores {
                            self.flush_stores(py, callbacks)?;
                        }
                    } else {
                        // Symbolic address with symbolic guard - concretize address first
                        match self.concretizer.concretize(&addr_val, self.ctx) {
                            ConcretizationResult::Single(addr_concrete) => {
                                // Load current value and use ITE
                                let current = self.load_from_callback(py, callbacks, addr_concrete, data_size)?;
                                let ite_result = guard_val.ite(&data_val, &current, self.ctx);
                                let ite_bytes = bv_to_bytes(&ite_result);
                                self.flush_stores(py, callbacks)?;
                                callbacks
                                    .call_memory_store(py, addr_concrete, &ite_bytes)
                                    .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                            }
                            _ => {
                                // Multiple addresses with symbolic guard - unsupported
                                return Err(CbExecutionError::Unsupported(
                                    "symbolic guarded store with multiple possible addresses".to_string()
                                ));
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
                            let data_bytes = bv_to_bytes(&data_val);
                            self.load_prefetch_cache.remove(&(addr_concrete, data_size));
                            self.pending_stores.push((addr_concrete, data_bytes));
                            if self.pending_stores.len() >= self.max_pending_stores {
                                self.flush_stores(py, callbacks)?;
                            }
                        } else {
                            // Symbolic address with concrete guard - flush and use callback
                            self.flush_stores(py, callbacks)?;
                            let data_bytes = bv_to_bytes(&data_val);
                            callbacks
                                .call_memory_store(py, 0, &data_bytes)
                                .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
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
                                    ConcretizationResult::Single(a) => a,
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
                                ConcretizationResult::Single(a) => a,
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
                                    ConcretizationResult::Single(a) => a,
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

                // Try Rust-native memory first if enabled
                if self.use_rust_memory {
                    if let Some(ref rust_mem) = self.rust_memory {
                        match rust_mem.load_symbolic(addr_val.clone(), size as u32, self.ctx, &self.concretizer) {
                            Ok(value) => return Ok(value),
                            Err(MemoryError::UnmappedPageInRegion { page_addr }) => {
                                // Page is in a lazy region - try to fetch it from Python with prefetch
                                let prefetch_count = self.page_prefetch_count;
                                let page_fetched = self.fetch_page_with_prefetch(py, callbacks, page_addr, prefetch_count)?;

                                if !page_fetched {
                                    // Python doesn't have the page - auto-map a zero page
                                    // This avoids falling back to Python for uninitialized memory
                                    if let Some(ref mut rust_mem) = self.rust_memory {
                                        rust_mem.auto_map_zero_page(page_addr);
                                    }
                                }

                                // Retry the load (whether from fetch or auto-map)
                                if let Some(ref rust_mem) = self.rust_memory {
                                    match rust_mem.load_symbolic(addr_val.clone(), size as u32, self.ctx, &self.concretizer) {
                                        Ok(value) => return Ok(value),
                                        Err(_e) => {
                                            // Still failed after fetch/auto-map - fall through to Python
                                        }
                                    }
                                }
                                // Fall through to Python callback
                            }
                            Err(MemoryError::Unmapped { .. }) => {
                                // Fall through to Python callback for unmapped pages
                            }
                            Err(MemoryError::SymbolicAddress { .. }) => {
                                // Fall through - Python has better symbolic handling
                            }
                            Err(e) => {
                                return Err(CbExecutionError::Memory(e.to_string()));
                            }
                        }
                    }
                }

                if let Some(addr_concrete) = addr_val.as_u64() {
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
                            // Check cached concrete memory first
                            if let Some(data) = self.try_read_concrete_memory(addr_concrete, size) {
                                return Ok(bytes_to_bv(data, (size * 8) as u32));
                            }
                            self.load_from_callback(py, callbacks, addr_concrete, size)
                        }
                        ConcretizationResult::Multiple(addrs) => {
                            // Delegate to Python for symbolic load with ITE chain
                            callbacks
                                .call_memory_load_symbolic(py, &addrs, size as u32, &addr_val)
                                .map_err(|e| CbExecutionError::Callback(e.to_string()))
                        }
                        ConcretizationResult::Strided { base, stride, count } => {
                            // Strided access pattern - generate addresses and delegate to Python
                            let addrs: Vec<u64> = (0..count).map(|i| base + i * stride).collect();
                            callbacks
                                .call_memory_load_symbolic(py, &addrs, size as u32, &addr_val)
                                .map_err(|e| CbExecutionError::Callback(e.to_string()))
                        }
                        ConcretizationResult::TooLarge { min, max, .. } => {
                            // Address range too large - delegate to Python's memory model
                            // which has access to angr's address concretization strategies
                            let (data, is_symbolic, symbolic_ast) = callbacks
                                .call_memory_load_symbolic_ast(py, size as u32)
                                .map_err(|e| CbExecutionError::Callback(format!(
                                    "symbolic load AST callback failed at 0x{:x}-0x{:x}: {}",
                                    min, max, e
                                )))?;

                            if is_symbolic {
                                // Try to convert claripy AST to RustBV
                                if let Some(ast_obj) = symbolic_ast {
                                    let ast = ast_obj.bind(py);
                                    if is_claripy_ast(&ast) {
                                        match claripy_to_rustbv(py, &ast, self.ctx) {
                                            Ok(bv) => return Ok(bv),
                                            Err(_e) => {
                                                // Fall back to creating a fresh symbolic value
                                            }
                                        }
                                    }
                                }
                                // Fallback: create a fresh symbolic value
                                Ok(RustBV::symbolic(
                                    self.ctx,
                                    &format!("sym_load_{:x}_{}", min, size),
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
                VEXOps::unop(*op, arg_val, self.ctx).map_err(|e| e.into())
            }

            IRExpr::Binop { op, left, right } => {
                let left_val = self.eval_expr_with_callbacks(py, callbacks, left, tyenv)?;
                let right_val = self.eval_expr_with_callbacks(py, callbacks, right, tyenv)?;
                VEXOps::binop(*op, left_val, right_val, self.ctx).map_err(|e| e.into())
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

            IRExpr::Triop { .. } => Err(CbExecutionError::Unsupported("triop".to_string())),

            IRExpr::Qop { .. } => Err(CbExecutionError::Unsupported("qop".to_string())),

            IRExpr::CCall { cee, retty, args } => {
                let mut arg_vals = Vec::with_capacity(args.len());
                for arg in args {
                    arg_vals.push(self.eval_expr_with_callbacks(py, callbacks, arg, tyenv)?);
                }

                if let Some(result) = ccall::handle_ccall(&cee.name, &arg_vals, retty.bits()) {
                    return Ok(result);
                }

                Ok(RustBV::concrete(0, retty.bits()))
            }

            IRExpr::VECRET | IRExpr::GSPTR => {
                Err(CbExecutionError::Unsupported("special expr".to_string()))
            }
        }
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
    fn eval_next_addr(
        &mut self,
        py: Python<'_>,
        callbacks: &PythonCallbacks,
        irsb: &IRSB,
    ) -> Result<u64, CbExecutionError> {
        let next_val = self.eval_expr_with_callbacks(py, callbacks, &irsb.next, &irsb.tyenv)?;
        // Check for symbolic addresses FIRST - Constrained BV has concrete value but is still symbolic
        if next_val.is_symbolic() {
            return Err(CbExecutionError::Unsupported("symbolic next address".to_string()));
        }
        next_val.as_u64().ok_or_else(|| {
            CbExecutionError::Unsupported("non-concrete next address".to_string())
        })
    }

    /// Handle the default exit (end of block).
    fn handle_default_exit(&mut self, irsb: &IRSB) -> Result<BlockResult, CbExecutionError> {
        // For default exit, we need to evaluate next without callbacks
        // since we already have temps set up
        let next_val = self.eval_expr_simple(&irsb.next, &irsb.tyenv)?;

        // Check for symbolic addresses FIRST, before extracting concrete value.
        // A Constrained BV has a concrete value stored (from solver evaluation),
        // but it's still symbolic and should not be used as a jump target directly.
        if next_val.is_symbolic() {
            return Err(CbExecutionError::Unsupported(
                "symbolic next address".to_string(),
            ));
        }

        if let Some(addr) = next_val.as_u64() {
            Ok(self.handle_exit(addr, irsb.jumpkind))
        } else {
            Err(CbExecutionError::Unsupported(
                "non-concrete next address".to_string(),
            ))
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

        BlockResult::BlockEnd {
            next_addr: target,
            jumpkind,
        }
    }

    /// Get the syscall number from the appropriate register.
    fn get_syscall_num(&self) -> u64 {
        // For AMD64, syscall number is in RAX (offset 16)
        // For x86, syscall number is in EAX (offset 8)
        // TODO: make this architecture-aware
        let offset = match self.arch {
            VexArch::AMD64 => 16,  // RAX
            VexArch::X86 => 8,     // EAX
            _ => 0,  // TODO: other architectures
        };
        let syscall_bv = self.registers.get(offset, 8, self.ctx);
        syscall_bv.as_u64().unwrap_or(0)
    }

    /// Fork the interpreter state.
    pub fn fork(&self) -> CallbackInterpreter<'a> {
        CallbackInterpreter {
            registers: self.registers.fork(),
            temps: self.temps.clone(),
            ctx: self.ctx,
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
            max_pending_stores: self.max_pending_stores,
            // Fork Rust memory with O(1) CoW
            rust_memory: self.rust_memory.as_ref().map(|m| m.fork()),
            use_rust_memory: self.use_rust_memory,
            load_prefetch_cache: HashMap::new(), // Fresh prefetch cache for fork
            use_load_prefetch: self.use_load_prefetch,
            page_prefetch_count: self.page_prefetch_count, // Inherit page prefetch count
            dirty_dispatch: DirtyHelperDispatch::new(), // Fresh dispatch (stateless)
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

    /// Get pages to fetch for nearby prefetch.
    ///
    /// Returns `prefetch_count` unmapped pages in each direction.
    fn get_nearby_prefetch_list(&self, page_addr: u64, prefetch_count: u32) -> Vec<u64> {
        let page_size = 0x1000u64;
        let mut pages_to_fetch = Vec::with_capacity(1 + 2 * prefetch_count as usize);

        // Add main page first
        pages_to_fetch.push(page_addr);

        // Add pages before (lower addresses)
        for i in 1..=prefetch_count {
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
        for i in 1..=prefetch_count {
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
                    // Try to convert claripy AST to RustBV
                    if let Some(ast_obj) = symbolic_ast {
                        let ast = ast_obj.bind(py);
                        if is_claripy_ast(&ast) {
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
