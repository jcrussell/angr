//! Rust-native simulation state for symbolic execution.
//!
//! `RustSimState` provides a Rust-first state representation that:
//! - Owns registers, memory, and solver context
//! - Supports O(1) forking via copy-on-write
//! - Minimizes Python-Rust state transfer overhead
//! - Enables Rust-native exploration loops

use std::collections::HashSet;
use std::rc::Rc;
use std::cell::RefCell;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyDict;

use crate::arch::{arch_from_name, arch_from_vex, Arch, RegisterFile};
use crate::memory::{MemoryError, Permission, SymbolicMemory, PAGE_SIZE};
use crate::symbolic::{RustBV, SymContext};
use crate::vex::{Endness, VexArch};
use crate::concretize::AddressConcretizer;

/// Unique identifier for states.
static NEXT_STATE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn next_state_id() -> u64 {
    NEXT_STATE_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
}

/// Execution event from stepping a state.
#[derive(Debug, Clone)]
pub enum ExecutionEvent {
    /// Reached end of a basic block, continuing to next address.
    BlockEnd { next_addr: u64 },
    /// Encountered a symbolic branch condition.
    SymbolicBranch {
        condition: RustBV,
        true_target: u64,
        false_target: u64,
    },
    /// Hit a hook address (SimProcedure).
    Hook { addr: u64 },
    /// Syscall instruction.
    Syscall { num: u64 },
    /// Unmapped memory access - need Python callback.
    UnmappedMemory { addr: u64, size: u64 },
    /// Error during execution.
    Error { message: String },
}

/// Incremental state changes for efficient sync.
///
/// Instead of syncing entire state, we track only what changed.
#[derive(Debug, Clone, Default)]
pub struct StateChanges {
    /// Register changes: (offset, size, value_bytes).
    pub register_writes: Vec<(u32, u32, Vec<u8>)>,
    /// Memory writes: (addr, value_bytes).
    pub memory_writes: Vec<(u64, Vec<u8>)>,
    /// New constraints added (as Z3 AST indices or serialized form).
    pub new_constraints: Vec<u64>,
    /// Updated PC.
    pub new_pc: Option<u64>,
}

impl StateChanges {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.register_writes.is_empty()
            && self.memory_writes.is_empty()
            && self.new_constraints.is_empty()
            && self.new_pc.is_none()
    }
}

/// Rust-native simulation state.
///
/// This struct owns all state components and provides O(1) forking
/// through copy-on-write semantics. The solver context is shared
/// via Rc<RefCell<>> to allow constraint accumulation across forks.
///
/// # Design
///
/// - **Registers**: Stored in `RegisterFile` with concrete bytes and symbolic overlays
/// - **Memory**: `SymbolicMemory` with O(1) CoW via `im::OrdMap`
/// - **Solver**: Shared `Rc<RefCell<SymContext>>` for constraint accumulation
/// - **History**: Basic block trace for debugging/analysis
///
/// # Fork Semantics
///
/// Forking a state is O(1) because:
/// - Memory uses persistent data structures (im::OrdMap)
/// - Registers are cloned (small, ~700 bytes for AMD64)
/// - Solver context is cloned with constraint state preserved
/// - History is optionally shared or copied based on config
pub struct RustSimState {
    /// Architecture information.
    arch: Box<dyn Arch>,
    /// VEX architecture enum (cached for quick lookup).
    vex_arch: VexArch,
    /// Register file with concrete and symbolic values.
    registers: RegisterFile,
    /// Symbolic memory with O(1) CoW forking.
    memory: SymbolicMemory,
    /// Shared solver context for constraints.
    /// Using Rc<RefCell<>> to allow mutation during stepping
    /// while maintaining shared ownership for forking.
    solver: Rc<RefCell<SymContext>>,
    /// Program counter.
    pc: u64,
    /// Unique state identifier.
    state_id: u64,
    /// Parent state ID (for tracking fork tree).
    parent_id: Option<u64>,
    /// Basic block history (addresses visited).
    history: Vec<u64>,
    /// Maximum history length (0 = unlimited).
    max_history: usize,
    /// Hook addresses.
    hooks: HashSet<u64>,
    /// Address concretization config.
    concretizer: AddressConcretizer,
    /// Dirty register tracking (bitset, each bit = 4 bytes).
    dirty_registers: u128,
    /// Whether to track detailed history.
    track_history: bool,
    /// Stdout buffer — accumulates output from native puts/printf.
    /// Cloned on fork so each path gets its own copy.
    stdout_buffer: Vec<u8>,
    /// Whether stdout_buffer has been written to (dirty flag).
    stdout_dirty: bool,
    /// Heap brk pointer — simple bump allocator for malloc/calloc.
    /// Default: 0xC0000000 (matching angr's DEFAULT_HEAP_LOCATION).
    heap_brk: u64,
}

impl RustSimState {
    /// Create a new state for the given architecture.
    ///
    /// # Arguments
    /// * `arch_name` - Architecture name (e.g., "amd64", "x86", "arm")
    /// * `little_endian` - Override endianness (None = use arch default)
    ///
    /// # Returns
    /// New state with default initialization.
    pub fn new(arch_name: &str) -> Result<Self, String> {
        Self::new_with_endian(arch_name, None)
    }

    /// Create a new state with explicit endianness override.
    pub fn new_with_endian(arch_name: &str, little_endian: Option<bool>) -> Result<Self, String> {
        let arch = arch_from_name(arch_name)
            .ok_or_else(|| format!("unknown architecture: {}", arch_name))?;
        let vex_arch = arch.vex_arch();
        let is_le = little_endian.unwrap_or_else(|| arch.is_little_endian());
        let endness = if is_le {
            Endness::Little
        } else {
            Endness::Big
        };

        Ok(RustSimState {
            vex_arch,
            registers: RegisterFile::new(arch.clone()),
            memory: SymbolicMemory::new(endness),
            solver: Rc::new(RefCell::new(SymContext::new())),
            pc: 0,
            state_id: next_state_id(),
            parent_id: None,
            history: Vec::new(),
            max_history: 1000,
            hooks: HashSet::new(),
            concretizer: AddressConcretizer::default(),
            dirty_registers: 0,
            track_history: true,
            arch,
            stdout_buffer: Vec::new(),
            stdout_dirty: false,
            heap_brk: 0xC000_0000,
        })
    }

    /// Create a state from VexArch.
    pub fn from_vex_arch(vex_arch: VexArch) -> Self {
        let arch = arch_from_vex(vex_arch);
        let endness = if arch.is_little_endian() {
            Endness::Little
        } else {
            Endness::Big
        };

        RustSimState {
            vex_arch,
            registers: RegisterFile::new(arch.clone()),
            memory: SymbolicMemory::new(endness),
            solver: Rc::new(RefCell::new(SymContext::new())),
            pc: 0,
            state_id: next_state_id(),
            parent_id: None,
            history: Vec::new(),
            max_history: 1000,
            hooks: HashSet::new(),
            concretizer: AddressConcretizer::default(),
            dirty_registers: 0,
            track_history: true,
            arch,
            stdout_buffer: Vec::new(),
            stdout_dirty: false,
            heap_brk: 0xC000_0000,
        }
    }

    /// Create a state with a shared solver context.
    ///
    /// This is used when forking to share constraints across states.
    pub fn with_solver(arch_name: &str, solver: Rc<RefCell<SymContext>>) -> Result<Self, String> {
        Self::with_solver_endian(arch_name, solver, None)
    }

    /// Create a state with a shared solver context and explicit endianness.
    pub fn with_solver_endian(arch_name: &str, solver: Rc<RefCell<SymContext>>, little_endian: Option<bool>) -> Result<Self, String> {
        let arch = arch_from_name(arch_name)
            .ok_or_else(|| format!("unknown architecture: {}", arch_name))?;
        let vex_arch = arch.vex_arch();
        let is_le = little_endian.unwrap_or_else(|| arch.is_little_endian());
        let endness = if is_le {
            Endness::Little
        } else {
            Endness::Big
        };

        Ok(RustSimState {
            vex_arch,
            registers: RegisterFile::new(arch.clone()),
            memory: SymbolicMemory::new(endness),
            solver,
            pc: 0,
            state_id: next_state_id(),
            parent_id: None,
            history: Vec::new(),
            max_history: 1000,
            hooks: HashSet::new(),
            concretizer: AddressConcretizer::default(),
            dirty_registers: 0,
            track_history: true,
            arch,
            stdout_buffer: Vec::new(),
            stdout_dirty: false,
            heap_brk: 0xC000_0000,
        })
    }

    // =========================================================================
    // Basic Accessors
    // =========================================================================

    /// Get the state ID.
    pub fn state_id(&self) -> u64 {
        self.state_id
    }

    /// Get the parent state ID.
    pub fn parent_id(&self) -> Option<u64> {
        self.parent_id
    }

    /// Get the program counter.
    pub fn pc(&self) -> u64 {
        self.pc
    }

    /// Set the program counter.
    /// Also updates the IP register in the register file so that
    /// get_register("rip"/"eip") returns the current PC.
    pub fn set_pc(&mut self, pc: u64) {
        self.pc = pc;
        let width = self.arch.bits();
        self.registers.set_ip(RustBV::concrete(pc as u128, width));
    }

    /// Get the VEX architecture.
    pub fn vex_arch(&self) -> VexArch {
        self.vex_arch
    }

    /// Get the architecture.
    pub fn arch(&self) -> &dyn Arch {
        self.arch.as_ref()
    }

    /// Get the stdout buffer.
    pub fn stdout_buffer(&self) -> &[u8] {
        &self.stdout_buffer
    }

    /// Append bytes to the stdout buffer.
    pub fn write_stdout(&mut self, data: &[u8]) {
        self.stdout_buffer.extend_from_slice(data);
        self.stdout_dirty = true;
    }

    /// Check if stdout has been written to.
    pub fn has_stdout(&self) -> bool {
        self.stdout_dirty
    }

    /// Get the current heap brk pointer.
    pub fn heap_brk(&self) -> u64 {
        self.heap_brk
    }

    /// Bump-allocate from the heap. Returns the address of the allocation.
    /// Aligns size up to 16 bytes (matching angr's SimHeapBrk).
    pub fn heap_alloc(&mut self, size: u64) -> u64 {
        let aligned = (size + 15) & !15; // round up to 16
        let addr = self.heap_brk;
        self.heap_brk = addr.wrapping_add(aligned);
        addr
    }

    /// Get the history (basic block addresses visited).
    pub fn history(&self) -> &[u64] {
        &self.history
    }

    /// Add an address to history.
    pub fn add_to_history(&mut self, addr: u64) {
        if self.track_history {
            self.history.push(addr);
            if self.max_history > 0 && self.history.len() > self.max_history {
                self.history.remove(0);
            }
        }
    }

    // =========================================================================
    // Register Access
    // =========================================================================

    /// Get a register by name.
    pub fn get_register(&self, name: &str) -> Option<RustBV> {
        let ctx = self.solver.borrow();
        self.registers.get_reg(name, &ctx)
    }

    /// Set a register by name.
    pub fn set_register(&mut self, name: &str, value: RustBV) -> bool {
        if let Some(offset) = self.arch.register_offset(name) {
            // Mark as dirty
            let bit = offset / 4;
            if bit < 128 {
                self.dirty_registers |= 1u128 << bit;
            }
        }
        self.registers.put_reg(name, value)
    }

    /// Get a register by offset.
    pub fn get_register_by_offset(&self, offset: u32, size: u32) -> RustBV {
        let ctx = self.solver.borrow();
        self.registers.get(offset, size, &ctx)
    }

    /// Set a register by offset.
    pub fn set_register_by_offset(&mut self, offset: u32, value: RustBV) {
        // Mark as dirty
        let bit = offset / 4;
        if bit < 128 {
            self.dirty_registers |= 1u128 << bit;
        }
        self.registers.put(offset, value);
    }

    /// Get the instruction pointer register.
    pub fn get_ip(&self) -> RustBV {
        let ctx = self.solver.borrow();
        self.registers.get_ip(&ctx)
    }

    /// Set the instruction pointer register.
    pub fn set_ip(&mut self, value: RustBV) {
        self.registers.set_ip(value);
        if let Some(v) = self.registers.get_ip(&self.solver.borrow()).as_u64() {
            self.pc = v;
        }
    }

    /// Get the stack pointer register.
    pub fn get_sp(&self) -> RustBV {
        let ctx = self.solver.borrow();
        self.registers.get_sp(&ctx)
    }

    /// Set the stack pointer register.
    pub fn set_sp(&mut self, value: RustBV) {
        self.registers.set_sp(value);
    }

    /// Get all register bytes (for bulk sync to Python).
    pub fn get_registers_raw(&self) -> Vec<u8> {
        let mut bytes = vec![0u8; self.arch.state_size()];
        self.registers.copy_to_bytes(&mut bytes);
        bytes
    }

    /// Set all register bytes (for bulk sync from Python).
    pub fn set_registers_raw(&mut self, bytes: &[u8]) {
        self.registers.copy_from_bytes(bytes);
    }

    /// Get dirty register offsets.
    pub fn get_dirty_registers(&self) -> Vec<u32> {
        let mut offsets = Vec::new();
        for bit in 0..128u32 {
            if (self.dirty_registers & (1u128 << bit)) != 0 {
                offsets.push(bit * 4);
            }
        }
        offsets
    }

    /// Clear dirty register tracking.
    pub fn clear_dirty_registers(&mut self) {
        self.dirty_registers = 0;
    }

    // =========================================================================
    // Memory Access
    // =========================================================================

    /// Get a reference to the memory.
    pub fn memory(&self) -> &SymbolicMemory {
        &self.memory
    }

    /// Get a mutable reference to the memory.
    pub fn memory_mut(&mut self) -> &mut SymbolicMemory {
        &mut self.memory
    }

    /// Take ownership of the memory, replacing it with an empty SymbolicMemory.
    pub fn take_memory(&mut self) -> SymbolicMemory {
        let endness = self.memory.endness();
        std::mem::replace(&mut self.memory, SymbolicMemory::new(endness))
    }

    /// Replace the memory with the given SymbolicMemory.
    pub fn replace_memory(&mut self, memory: SymbolicMemory) {
        self.memory = memory;
    }

    /// Get a reference to the register file.
    pub fn registers(&self) -> &RegisterFile {
        &self.registers
    }

    /// Replace the register file (including symbolic entries).
    pub fn set_registers(&mut self, registers: RegisterFile) {
        self.registers = registers;
    }

    /// Get dirty page numbers (page_num = addr >> 12) from the memory.
    pub fn get_dirty_page_nums(&self) -> Vec<u64> {
        self.memory.get_dirty_pages()
    }

    /// Map a memory region.
    pub fn map_memory(&mut self, addr: u64, size: u64, permissions: Permission) {
        self.memory.map(addr, size, permissions);
    }

    /// Map memory with initial data.
    pub fn map_memory_data(&mut self, addr: u64, data: &[u8], permissions: Permission) {
        self.memory.map_data(addr, data, permissions);
    }

    /// Load from memory.
    pub fn memory_load(&self, addr: u64, size: u32) -> Result<RustBV, MemoryError> {
        let ctx = self.solver.borrow();
        self.memory.load_concrete(addr, size, &ctx)
    }

    /// Store to memory.
    pub fn memory_store(&mut self, addr: u64, value: RustBV) -> Result<(), MemoryError> {
        self.memory.store_concrete(addr, value)
    }

    /// Load from a symbolic address.
    pub fn memory_load_symbolic(&mut self, addr: RustBV, size: u32) -> Result<RustBV, MemoryError> {
        let ctx = self.solver.borrow();
        self.memory.load_symbolic_unified(addr, size, &ctx, &self.concretizer)
    }

    /// Store to a symbolic address.
    pub fn memory_store_symbolic(&mut self, addr: RustBV, value: RustBV) -> Result<(), MemoryError> {
        let ctx = self.solver.borrow();
        self.memory.store_symbolic_unified(addr, value, &ctx, &self.concretizer)
            .map(|_| ())
    }

    /// Add a lazy region for on-demand page fetching.
    pub fn add_lazy_region(&mut self, start_addr: u64, size: u64) {
        self.memory.add_lazy_region(start_addr, size);
    }

    /// Get dirty page addresses.
    pub fn get_dirty_pages(&self) -> Vec<u64> {
        self.memory.get_dirty_page_addrs()
    }

    /// Clear dirty page tracking.
    pub fn clear_dirty_pages(&mut self) {
        self.memory.clear_dirty_pages();
    }

    // =========================================================================
    // Solver/Constraint Access
    // =========================================================================

    /// Get a reference to the solver context.
    pub fn solver(&self) -> &Rc<RefCell<SymContext>> {
        &self.solver
    }

    /// Add a constraint.
    pub fn add_constraint(&self, constraint: RustBV) {
        let ctx = self.solver.borrow();
        ctx.assume_true(&constraint);
    }

    /// Check if current constraints are satisfiable.
    pub fn satisfiable(&self) -> bool {
        let ctx = self.solver.borrow();
        ctx.is_sat()
    }

    /// Prime the SAT cache (avoids redundant Z3 checks after branch forking).
    pub fn set_sat_cache(&self, value: bool) {
        self.solver.borrow().set_sat_cache(value);
    }

    /// Evaluate an expression to a concrete value.
    pub fn eval(&self, expr: &RustBV) -> Option<u128> {
        let ctx = self.solver.borrow();
        ctx.eval(expr)
    }

    /// Get minimum value of an expression.
    pub fn min(&self, expr: &RustBV, signed: bool) -> Option<u128> {
        let ctx = self.solver.borrow();
        ctx.min(expr, signed)
    }

    /// Get maximum value of an expression.
    pub fn max(&self, expr: &RustBV, signed: bool) -> Option<u128> {
        let ctx = self.solver.borrow();
        ctx.max(expr, signed)
    }

    // =========================================================================
    // Hooks
    // =========================================================================

    /// Add a hook address.
    pub fn add_hook(&mut self, addr: u64) {
        self.hooks.insert(addr);
    }

    /// Remove a hook address.
    pub fn remove_hook(&mut self, addr: u64) {
        self.hooks.remove(&addr);
    }

    /// Check if an address is hooked.
    pub fn is_hooked(&self, addr: u64) -> bool {
        self.hooks.contains(&addr)
    }

    /// Clear all hooks.
    pub fn clear_hooks(&mut self) {
        self.hooks.clear();
    }

    // =========================================================================
    // Forking
    // =========================================================================

    /// Fork the state (O(1) copy-on-write).
    ///
    /// Creates a new state that shares memory pages via CoW.
    /// The solver context is forked to preserve constraints.
    ///
    /// # Returns
    /// A new state with the same register/memory/constraint state.
    pub fn fork(&self) -> Self {
        let forked_solver = Rc::new(RefCell::new(self.solver.borrow().fork()));

        RustSimState {
            arch: self.arch.clone(),
            vex_arch: self.vex_arch,
            registers: self.registers.fork(),
            memory: self.memory.fork(),
            solver: forked_solver,
            pc: self.pc,
            state_id: next_state_id(),
            parent_id: Some(self.state_id),
            history: self.history.clone(),
            max_history: self.max_history,
            hooks: self.hooks.clone(),
            concretizer: self.concretizer.clone(),
            dirty_registers: 0, // Fresh dirty tracking for fork
            track_history: self.track_history,
            stdout_buffer: self.stdout_buffer.clone(),
            stdout_dirty: self.stdout_dirty,
            heap_brk: self.heap_brk,
        }
    }

    /// Fork with a constraint on the true branch.
    pub fn fork_true(&self, condition: &RustBV) -> Self {
        let forked_solver = Rc::new(RefCell::new(self.solver.borrow().fork_true(condition)));

        RustSimState {
            arch: self.arch.clone(),
            vex_arch: self.vex_arch,
            registers: self.registers.fork(),
            memory: self.memory.fork(),
            solver: forked_solver,
            pc: self.pc,
            state_id: next_state_id(),
            parent_id: Some(self.state_id),
            history: self.history.clone(),
            max_history: self.max_history,
            hooks: self.hooks.clone(),
            concretizer: self.concretizer.clone(),
            dirty_registers: 0,
            track_history: self.track_history,
            stdout_buffer: self.stdout_buffer.clone(),
            stdout_dirty: self.stdout_dirty,
            heap_brk: self.heap_brk,
        }
    }

    /// Fork with a constraint on the false branch.
    pub fn fork_false(&self, condition: &RustBV) -> Self {
        let forked_solver = Rc::new(RefCell::new(self.solver.borrow().fork_false(condition)));

        RustSimState {
            arch: self.arch.clone(),
            vex_arch: self.vex_arch,
            registers: self.registers.fork(),
            memory: self.memory.fork(),
            solver: forked_solver,
            pc: self.pc,
            state_id: next_state_id(),
            parent_id: Some(self.state_id),
            history: self.history.clone(),
            max_history: self.max_history,
            hooks: self.hooks.clone(),
            concretizer: self.concretizer.clone(),
            dirty_registers: 0,
            track_history: self.track_history,
            stdout_buffer: self.stdout_buffer.clone(),
            stdout_dirty: self.stdout_dirty,
            heap_brk: self.heap_brk,
        }
    }

    /// Replace the solver context with a different one.
    /// Used for deferred fork processing where the alternate path needs a solver
    /// snapshot from before the branch constraint was added.
    pub fn replace_solver(&mut self, ctx: crate::symbolic::SymContext) {
        self.solver = Rc::new(RefCell::new(ctx));
    }

    /// Create a forked state using a full branch snapshot (solver + registers + memory).
    /// The resulting state has the correct state from the branch point, not from
    /// the continuation of the taken path.
    pub fn fork_from_snapshot(&self, snapshot: crate::interpreter_cb::BranchSnapshot) -> Self {
        let forked_solver = Rc::new(RefCell::new(snapshot.solver));
        RustSimState {
            arch: self.arch.clone(),
            vex_arch: self.vex_arch,
            registers: snapshot.registers,
            memory: snapshot.memory.unwrap_or_else(|| self.memory.fork()),
            solver: forked_solver,
            pc: self.pc,
            state_id: next_state_id(),
            parent_id: Some(self.state_id),
            history: self.history.clone(),
            max_history: self.max_history,
            hooks: self.hooks.clone(),
            concretizer: self.concretizer.clone(),
            dirty_registers: 0,
            track_history: self.track_history,
            stdout_buffer: self.stdout_buffer.clone(),
            stdout_dirty: self.stdout_dirty,
            heap_brk: self.heap_brk,
        }
    }

    // =========================================================================
    // Incremental State Changes
    // =========================================================================

    /// Apply incremental changes to the state.
    ///
    /// This is used when syncing from Python after a SimProcedure runs.
    pub fn apply_changes(&mut self, changes: &StateChanges) {
        // Apply register writes
        for (offset, size, bytes) in &changes.register_writes {
            let mut value: u128 = 0;
            for (i, &b) in bytes.iter().enumerate() {
                value |= (b as u128) << (i * 8);
            }
            let bv = RustBV::concrete(value, *size * 8);
            self.set_register_by_offset(*offset, bv);
        }

        // Apply memory writes — split into 16-byte chunks since RustBV
        // uses u128 internally (max 128 bits per concrete value)
        for (addr, bytes) in &changes.memory_writes {
            let mut offset = 0usize;
            while offset < bytes.len() {
                let remaining = bytes.len() - offset;
                let chunk_size = remaining.min(16);
                let chunk = &bytes[offset..offset + chunk_size];
                let width = (chunk_size * 8) as u32;
                let mut value: u128 = 0;
                for (i, &b) in chunk.iter().enumerate() {
                    value |= (b as u128) << (i * 8);
                }
                let bv = RustBV::concrete(value, width);
                let _ = self.memory.store_concrete(*addr + offset as u64, bv);
                offset += chunk_size;
            }
        }

        // Apply PC change
        if let Some(new_pc) = changes.new_pc {
            self.pc = new_pc;
        }
    }

    /// Export changes since last clear for sync to Python.
    ///
    /// Returns the minimal diff needed to update Python state.
    pub fn export_changes(&self) -> StateChanges {
        let mut changes = StateChanges::new();

        // Export dirty registers
        let dirty_offsets = self.get_dirty_registers();
        for offset in dirty_offsets {
            let size = 8u32; // Most registers are 8 bytes
            let bv = self.get_register_by_offset(offset, size);
            if let Some(val) = bv.as_u128() {
                let bytes: Vec<u8> = (0..size as usize)
                    .map(|i| (val >> (i * 8)) as u8)
                    .collect();
                changes.register_writes.push((offset, size, bytes));
            }
        }

        // Export dirty pages
        for page_addr in self.memory.get_dirty_page_addrs() {
            let page_num = page_addr >> 12;
            if let Some((data, _perms)) = self.memory.get_page_data(page_num) {
                changes.memory_writes.push((page_addr, data));
            }
        }

        changes.new_pc = Some(self.pc);
        changes
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    /// Set the address concretization strategy.
    pub fn set_concretizer(&mut self, concretizer: AddressConcretizer) {
        self.concretizer = concretizer;
    }

    /// Configure address concretization.
    pub fn configure_concretization(&mut self, use_approximate: bool, range_limit: Option<u64>) {
        self.concretizer.configure(use_approximate, range_limit);
    }

    /// Set whether to track history.
    pub fn set_track_history(&mut self, track: bool) {
        self.track_history = track;
    }

    /// Set maximum history length.
    pub fn set_max_history(&mut self, max: usize) {
        self.max_history = max;
    }
}

impl Clone for RustSimState {
    fn clone(&self) -> Self {
        self.fork()
    }
}

// =============================================================================
// Python Bindings
// =============================================================================

/// Python-facing wrapper for RustSimState.
///
/// This provides the PyO3 interface for creating and manipulating
/// Rust-native simulation states from Python.
#[pyclass(name = "RustSimState", unsendable)]
pub struct PyRustSimState {
    inner: RustSimState,
}

#[pymethods]
impl PyRustSimState {
    /// Create a new state for the given architecture.
    #[new]
    #[pyo3(signature = (arch="amd64", little_endian=None))]
    pub fn new(arch: &str, little_endian: Option<bool>) -> PyResult<Self> {
        let inner = RustSimState::new_with_endian(arch, little_endian)
            .map_err(|e| PyValueError::new_err(e))?;
        Ok(PyRustSimState { inner })
    }

    /// Get the state ID.
    #[getter]
    pub fn state_id(&self) -> u64 {
        self.inner.state_id()
    }

    /// Get the parent state ID.
    #[getter]
    pub fn parent_id(&self) -> Option<u64> {
        self.inner.parent_id()
    }

    /// Get the program counter.
    #[getter]
    pub fn pc(&self) -> u64 {
        self.inner.pc()
    }

    /// Set the program counter.
    #[setter]
    pub fn set_pc(&mut self, pc: u64) {
        self.inner.set_pc(pc);
    }

    /// Get the architecture name.
    #[getter]
    pub fn arch_name(&self) -> &str {
        self.inner.arch().name()
    }

    /// Get the history (basic block addresses).
    pub fn history(&self) -> Vec<u64> {
        self.inner.history().to_vec()
    }

    /// Get a register value by name.
    pub fn get_register(&self, name: &str) -> PyResult<u128> {
        self.inner.get_register(name)
            .and_then(|bv| bv.as_u128())
            .ok_or_else(|| PyValueError::new_err(format!("cannot read register {}", name)))
    }

    /// Set a register value by name.
    pub fn set_register(&mut self, name: &str, value: u128) -> PyResult<()> {
        let size = self.inner.arch().register_size(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))?;
        let bv = RustBV::concrete(value, size * 8);
        if self.inner.set_register(name, bv) {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!("failed to set register: {}", name)))
        }
    }

    /// Set multiple registers in a single FFI call.
    /// Takes a dict of {name: value} pairs.
    pub fn set_registers_bulk(&mut self, registers: &Bound<'_, PyDict>) -> PyResult<()> {
        for (key, val) in registers.iter() {
            let name: String = key.extract()?;
            let value: u128 = val.extract()?;
            let size = self.inner.arch().register_size(&name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))?;
            let bv = RustBV::concrete(value, size * 8);
            self.inner.set_register(&name, bv);
        }
        Ok(())
    }

    /// Get all register bytes.
    pub fn get_registers_raw(&self) -> Vec<u8> {
        self.inner.get_registers_raw()
    }

    /// Set all register bytes.
    pub fn set_registers_raw(&mut self, bytes: &[u8]) {
        self.inner.set_registers_raw(bytes);
    }

    /// Set a register to a symbolic value from a raw Z3 AST pointer.
    ///
    /// The Z3 AST must be a BitVec in the shared Z3 context.
    /// Used to import symbolic register values (e.g., BVS in rax) from Python.
    /// Set a register to a symbolic value from a raw Z3 AST pointer.
    ///
    /// The Z3 AST must be a BitVec in the shared Z3 context.
    /// Used to import symbolic register values (e.g., BVS in rax) from Python.
    #[cfg(feature = "vex-engine-z3")]
    pub fn set_register_symbolic(&mut self, name: &str, z3_ast_ptr: usize, width: u32) -> PyResult<()> {
        use z3::ast::Ast;
        let size = self.inner.arch().register_size(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown register: {}", name)))?;
        if width != size * 8 {
            return Err(PyValueError::new_err(format!(
                "width mismatch: register {} is {} bits, got {} bits", name, size * 8, width
            )));
        }
        // Reconstruct z3::ast::BV from raw pointer.
        // Safety: caller guarantees pointer is a valid Z3_ast in shared context.
        let z3_bv = unsafe {
            let raw = std::ptr::NonNull::new_unchecked(z3_ast_ptr as *mut _);
            let ctx = z3::Context::thread_local();
            z3::ast::BV::wrap(&ctx, raw)
        };
        let bv = RustBV::Symbolic {
            id: 0,
            ast: z3_bv,
            width,
            name: name.to_string(),
        };
        if self.inner.set_register(name, bv) {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!("failed to set register: {}", name)))
        }
    }

    /// Get dirty register offsets.
    pub fn get_dirty_registers(&self) -> Vec<u32> {
        self.inner.get_dirty_registers()
    }

    /// Clear dirty register tracking.
    pub fn clear_dirty_registers(&mut self) {
        self.inner.clear_dirty_registers();
    }

    /// Map a memory region.
    #[pyo3(signature = (addr, size, permissions=7))]
    pub fn map_memory(&mut self, addr: u64, size: u64, permissions: u8) {
        self.inner.map_memory(addr, size, Permission::from_bits(permissions));
    }

    /// Map memory with initial data.
    #[pyo3(signature = (addr, data, permissions=7))]
    pub fn map_memory_data(&mut self, addr: u64, data: &[u8], permissions: u8) {
        self.inner.map_memory_data(addr, data, Permission::from_bits(permissions));
    }

    /// Map multiple memory pages in a single FFI call.
    /// pages is a list of (addr, data, permissions) tuples.
    pub fn map_memory_batch(&mut self, pages: Vec<(u64, Vec<u8>, u8)>) {
        for (addr, data, permissions) in pages {
            self.inner.map_memory_data(addr, &data, Permission::from_bits(permissions));
        }
    }

    /// Load from memory.
    pub fn memory_load(&self, addr: u64, size: u32) -> PyResult<Vec<u8>> {
        let bv = self.inner.memory_load(addr, size)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let value = bv.to_u128();
        let bytes: Vec<u8> = (0..size as usize)
            .map(|i| (value >> (i * 8)) as u8)
            .collect();
        Ok(bytes)
    }

    /// Store to memory.
    pub fn memory_store(&mut self, addr: u64, data: &[u8]) -> PyResult<()> {
        let width = (data.len() * 8) as u32;
        let mut value: u128 = 0;
        for (i, &b) in data.iter().enumerate() {
            value |= (b as u128) << (i * 8);
        }
        let bv = RustBV::concrete(value, width);
        self.inner.memory_store(addr, bv)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Add a lazy region.
    pub fn add_lazy_region(&mut self, start_addr: u64, size: u64) {
        self.inner.add_lazy_region(start_addr, size);
    }

    /// Get dirty page addresses.
    pub fn get_dirty_pages(&self) -> Vec<u64> {
        self.inner.get_dirty_pages()
    }

    /// Clear dirty page tracking.
    pub fn clear_dirty_pages(&mut self) {
        self.inner.clear_dirty_pages();
    }

    /// Add a hook address.
    pub fn add_hook(&mut self, addr: u64) {
        self.inner.add_hook(addr);
    }

    /// Remove a hook address.
    pub fn remove_hook(&mut self, addr: u64) {
        self.inner.remove_hook(addr);
    }

    /// Check if an address is hooked.
    pub fn is_hooked(&self, addr: u64) -> bool {
        self.inner.is_hooked(addr)
    }

    /// Clear all hooks.
    pub fn clear_hooks(&mut self) {
        self.inner.clear_hooks();
    }

    /// Check if constraints are satisfiable.
    pub fn satisfiable(&self) -> bool {
        self.inner.satisfiable()
    }

    /// Fork the state (O(1) CoW).
    pub fn fork(&self) -> Self {
        PyRustSimState {
            inner: self.inner.fork(),
        }
    }

    /// Configure address concretization.
    #[pyo3(signature = (use_approximate, range_limit=None))]
    pub fn configure_concretization(&mut self, use_approximate: bool, range_limit: Option<u64>) {
        self.inner.configure_concretization(use_approximate, range_limit);
    }

    /// Set whether to track history.
    pub fn set_track_history(&mut self, track: bool) {
        self.inner.set_track_history(track);
    }

    /// Set maximum history length.
    pub fn set_max_history(&mut self, max: usize) {
        self.inner.set_max_history(max);
    }

    /// Export the complete state as a snapshot.
    pub fn export_full(&self) -> ExplorationStateSnapshot {
        self.inner.export_full()
    }
}

impl PyRustSimState {
    /// Get access to the inner state (for Rust-side use).
    pub fn inner(&self) -> &RustSimState {
        &self.inner
    }

    /// Get mutable access to the inner state.
    pub fn inner_mut(&mut self) -> &mut RustSimState {
        &mut self.inner
    }
}

// =============================================================================
// State Snapshot for Exploration Export
// =============================================================================

/// Internal memory page data (addr, data, permissions, symbolic_offsets).
type PageData = (u64, Vec<u8>, u8, Vec<u16>);

/// Complete state snapshot for exploration export.
///
/// This contains all information needed to reconstruct an angr SimState
/// from a Rust execution state.
#[pyclass(name = "ExplorationStateSnapshot")]
pub struct ExplorationStateSnapshot {
    /// Unique state identifier.
    #[pyo3(get)]
    pub state_id: u64,
    /// Parent state ID (for fork tracking).
    #[pyo3(get)]
    pub parent_id: Option<u64>,
    /// Program counter.
    #[pyo3(get)]
    pub pc: u64,
    /// Architecture name.
    #[pyo3(get)]
    pub arch_name: String,
    /// Raw register bytes.
    registers_raw: Vec<u8>,
    /// Memory pages: (addr, data, permissions, symbolic_offsets).
    memory_pages: Vec<PageData>,
    /// Number of constraints in the solver.
    #[pyo3(get)]
    pub constraint_count: usize,
    /// Basic block history.
    history: Vec<u64>,
    /// Named register values: (name, concrete_value, size_bits).
    /// Pre-computed at export time so Python doesn't need offset tables.
    named_registers: Vec<(String, u128, u32)>,
}

#[pymethods]
impl ExplorationStateSnapshot {
    /// Get raw register bytes.
    pub fn get_registers_raw(&self) -> Vec<u8> {
        self.registers_raw.clone()
    }

    /// Get named register values as a dict: {name: (value, size_bits)}.
    ///
    /// Pre-computed at export time using Rust's register tables,
    /// so Python doesn't need architecture-specific offset mapping.
    pub fn get_registers_named(&self) -> std::collections::HashMap<String, (u128, u32)> {
        self.named_registers
            .iter()
            .map(|(name, value, bits)| (name.clone(), (*value, *bits)))
            .collect()
    }

    /// Get history (basic block addresses visited).
    pub fn get_history(&self) -> Vec<u64> {
        self.history.clone()
    }

    /// Get the number of memory pages.
    pub fn page_count(&self) -> usize {
        self.memory_pages.len()
    }

    /// Get a memory page by index.
    /// Returns (addr, data, permissions, symbolic_offsets) or None.
    pub fn get_page(&self, index: usize) -> Option<(u64, Vec<u8>, u8, Vec<u16>)> {
        self.memory_pages.get(index).map(|p| {
            (p.0, p.1.clone(), p.2, p.3.clone())
        })
    }

    /// Get all memory page addresses.
    pub fn page_addresses(&self) -> Vec<u64> {
        self.memory_pages.iter().map(|p| p.0).collect()
    }

    /// Load bytes from memory at a given address.
    /// Returns None if the address is not mapped.
    pub fn memory_load(&self, addr: u64, size: usize) -> Option<Vec<u8>> {
        let page_addr = addr & !0xFFF;
        let offset = (addr & 0xFFF) as usize;

        // Find the page
        for page in &self.memory_pages {
            if page.0 == page_addr {
                if offset + size <= page.1.len() {
                    return Some(page.1[offset..offset + size].to_vec());
                }
            }
        }
        None
    }

    /// Get symbolic byte offsets for a page.
    /// Returns empty vec if page not found.
    pub fn get_symbolic_offsets(&self, page_addr: u64) -> Vec<u16> {
        for page in &self.memory_pages {
            if page.0 == page_addr {
                return page.3.clone();
            }
        }
        Vec::new()
    }
}

impl RustSimState {
    /// Export the complete state as a snapshot.
    ///
    /// This creates a self-contained snapshot that can be used to
    /// reconstruct an angr SimState.
    pub fn export_full(&self) -> ExplorationStateSnapshot {
        // Export registers
        let registers_raw = self.get_registers_raw();

        // Export named registers: read each GP register by name
        let mut named_registers = Vec::new();
        let ctx = self.solver.borrow();
        for &name in self.arch.register_names() {
            if let Some(size) = self.arch.register_size(name) {
                let bv = self.registers.get_reg(name, &ctx);
                if let Some(bv) = bv {
                    if let Some(val) = bv.as_u128() {
                        named_registers.push((name.to_string(), val, size * 8));
                    }
                    // Skip symbolic registers (they'll need AST recovery)
                }
            }
        }

        // Flush pending writes before exporting memory pages.
        // We need a mutable borrow, but export_full takes &self. Use an
        // unsafe interior mutability pattern is not ideal, so we just report
        // unflushed writes via pending_writes_count on the snapshot.
        // Callers should call flush_pending_writes() before export_full()
        // if they need materialized memory.

        // Export memory pages as tuples: (addr, data, permissions, symbolic_offsets)
        let mut memory_pages: Vec<PageData> = Vec::new();
        for (page_num, page) in self.memory.pages().iter() {
            let page_addr = page_num << 12;
            let data = page.load_concrete(0, crate::memory::PAGE_SIZE as u16);
            let permissions = page.permissions().to_bits();
            let symbolic_offsets = page.symbolic_offsets();

            memory_pages.push((page_addr, data, permissions, symbolic_offsets));
        }

        // Get constraint count
        let constraint_count = self.solver.borrow().num_constraints();

        ExplorationStateSnapshot {
            state_id: self.state_id,
            parent_id: self.parent_id,
            pc: self.pc,
            arch_name: self.arch.name().to_string(),
            registers_raw,
            memory_pages,
            constraint_count,
            history: self.history.clone(),
            named_registers,
        }
    }

    /// Flush pending writes and then export.
    /// This materializes any deferred symbolic stores before creating the snapshot.
    pub fn flush_and_export_full(&mut self) -> ExplorationStateSnapshot {
        // Flush pending writes using the current solver context
        {
            let ctx = self.solver.borrow();
            let _ = self.memory.flush_pending_writes(&*ctx, &self.concretizer);
        }
        self.export_full()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_creation() {
        let state = RustSimState::new("amd64").unwrap();
        assert_eq!(state.vex_arch(), VexArch::AMD64);
        assert_eq!(state.pc(), 0);
    }

    #[test]
    fn test_state_fork() {
        let mut state1 = RustSimState::new("amd64").unwrap();
        state1.set_pc(0x1000);
        state1.set_register("rax", RustBV::concrete(42, 64));

        let state2 = state1.fork();

        // Both should have same values
        assert_eq!(state2.pc(), 0x1000);
        assert_eq!(state2.get_register("rax").unwrap().as_u64(), Some(42));

        // Different state IDs
        assert_ne!(state1.state_id(), state2.state_id());

        // state2's parent should be state1
        assert_eq!(state2.parent_id(), Some(state1.state_id()));
    }

    #[test]
    fn test_state_memory() {
        let mut state = RustSimState::new("amd64").unwrap();

        // Map and write
        state.map_memory(0x1000, 0x1000, Permission::RWX);
        state.memory_store(0x1000, RustBV::concrete(0xDEADBEEF, 32)).unwrap();

        // Read back
        let val = state.memory_load(0x1000, 4).unwrap();
        assert_eq!(val.as_u64(), Some(0xDEADBEEF));
    }

    #[test]
    fn test_state_fork_memory_cow() {
        let mut state1 = RustSimState::new("amd64").unwrap();
        state1.map_memory(0x1000, 0x1000, Permission::RWX);
        state1.memory_store(0x1000, RustBV::concrete(0xAAAA, 16)).unwrap();

        let mut state2 = state1.fork();

        // Modify state2
        state2.memory_store(0x1000, RustBV::concrete(0xBBBB, 16)).unwrap();

        // state1 should still have original value
        let val1 = state1.memory_load(0x1000, 2).unwrap();
        assert_eq!(val1.as_u64(), Some(0xAAAA));

        // state2 should have new value
        let val2 = state2.memory_load(0x1000, 2).unwrap();
        assert_eq!(val2.as_u64(), Some(0xBBBB));
    }
}
