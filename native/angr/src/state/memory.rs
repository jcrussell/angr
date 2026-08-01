//! Memory-access family for `RustSimState`.
//!
//! The `SymbolicMemory` reference/replace accessors, region mapping, dirty-page
//! tracking, lazy-region registration, and the concrete/symbolic load/store
//! bridge used by native SimProcedures. Split out of `mod.rs` per the
//! god-object decomposition (angr-0mqkc.5); mirrors the `construction.rs` /
//! `fork.rs` / `registers.rs` extension-impl pattern.
//!
//! The PyO3-facing memory methods (the `#[pymethods]` `map_memory`,
//! `map_memory_data`, `map_memory_batch`, `memory_load`, `memory_store`
//! wrappers) stay in `mod.rs` with the rest of the `#[pymethods]` block.

use super::*;

impl RustSimState {
    /// Get a reference to the memory.
    pub fn memory(&self) -> &SymbolicMemory {
        &self.memory
    }

    /// Get a mutable reference to the memory.
    pub fn memory_mut(&mut self) -> &mut SymbolicMemory {
        &mut self.memory
    }

    /// The state's configured memory endianness.
    ///
    /// Always prefer this over `arch().is_little_endian()` when the answer
    /// feeds observable state behaviour: every `Arch` impl hardcodes
    /// little-endian, so the bi-endian arches (ARM/MIPS) carry their real
    /// byte order only here, set once by `with_solver_endian` in
    /// `state/construction.rs` from the `little_endian` override that
    /// `rust_manager.py` computes off `project.arch.memory_endness`.
    pub fn is_little_endian(&self) -> bool {
        self.memory.endness() == Endness::Little
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
    ///
    /// Uses the auto-mapping store so a write to a not-yet-mapped page inside a
    /// lazy region (e.g. freshly `heap_alloc`-ed memory) maps a zero page and
    /// succeeds, instead of erroring `Unmapped`. Writes to genuinely unmapped
    /// addresses outside any lazy region still error so the caller can fall
    /// back to Python. This is the procedure-facing API; the interpreter has
    /// its own store paths.
    pub fn memory_store(&mut self, addr: u64, value: RustBV) -> Result<(), MemoryError> {
        self.memory.store_concrete_automap_internal(addr, value)
    }

    /// Register `[addr, addr+size)` as a lazy region so a subsequent
    /// `memory_store` auto-maps the covering page(s) instead of erroring
    /// `Unmapped` (angr-5rjbq). Used by the callback-memory-proxy when it
    /// concretizes a symbolic store address to a witness outside any existing
    /// lazy region and must be able to write there.
    pub fn add_memory_lazy_region(&mut self, addr: u64, size: u64) {
        self.memory.add_lazy_region(addr, size);
    }

    /// Load from a symbolic address.
    pub fn memory_load_symbolic(&mut self, addr: RustBV, size: u32) -> Result<RustBV, MemoryError> {
        let ctx = self.solver.borrow();
        self.memory
            .load_symbolic_unified(addr, size, &ctx, &self.concretizer)
    }

    /// Store to a symbolic address.
    pub fn memory_store_symbolic(
        &mut self,
        addr: RustBV,
        value: RustBV,
    ) -> Result<(), MemoryError> {
        let ctx = self.solver.borrow();
        self.memory
            .store_symbolic_unified(addr, value, &ctx, &self.concretizer)
            .map(|_| ())
    }

    /// Store to a symbolic address using the lazy Multi-cell path
    /// (Phase 1.3/1.4 of angr-czph). Mirrors `memory_store_symbolic` but
    /// routes Multiple/Strided concretization results to per-byte Multi
    /// alternatives instead of eager ITE chains. Single addresses still
    /// short-circuit to the eager concrete store; TooLarge / Failed surface
    /// the same errors so callers can fall back identically.
    pub fn memory_store_symbolic_multi(
        &mut self,
        addr: RustBV,
        value: RustBV,
    ) -> Result<(), MemoryError> {
        let ctx = self.solver.borrow();
        self.memory
            .store_symbolic_unified_multi(addr, value, &ctx, &self.concretizer)
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
}
