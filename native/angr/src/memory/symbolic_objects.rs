//! Symbolic object preservation: import, export, and per-address lookup.
//!
//! `SymbolicMemory` keeps a side-table of multi-byte symbolic values (the
//! `symbolic_objects` map) plus a reverse span index so a single-byte load
//! that lands inside a wider symbolic value can find it. This module owns
//! the helpers that get/set/clear those structures and the import/export
//! glue that preserves symbol identity when a state is round-tripped to
//! Python.
//!
//! All entry points are inherent methods on `SymbolicMemory`, so callers in
//! `memory/mod.rs` and external crates keep using `mem.method(...)`.

use super::{MemoryPage, PAGE_MASK, Permission, SymbolicMemory};
use crate::symbolic::RustBV;

impl SymbolicMemory {
    /// Get all symbolic regions for export to Python.
    ///
    /// Returns a list of (address, width, symbol_id) tuples where symbol_id
    /// is the Rust symbol ID that can be used to look up the original Python AST.
    ///
    /// This is critical for preserving symbolic identity when syncing state
    /// back to Python - without it, symbolic values would be recreated as
    /// fresh symbols, losing their relationship to constraints.
    pub fn get_symbolic_regions(&self) -> Vec<(u64, u32, Option<u64>)> {
        let mut regions = Vec::new();

        for (&addr, bv) in &self.symbolic_objects {
            let width = bv.width();
            // Try to get the symbol ID for Symbolic variants
            let sym_id = match bv {
                RustBV::Symbolic { id, .. } => Some(*id),
                RustBV::Expression { .. } => {
                    // For expressions, try to get the hash as an identifier
                    None
                }
                _ => None,
            };
            regions.push((addr, width, sym_id));
        }

        regions
    }

    /// Import a symbolic value with identity preservation.
    ///
    /// This stores a symbolic value at the given address and ensures the
    /// symbol ID is tracked for later export back to Python.
    ///
    /// # Arguments
    /// * `addr` - The address to store the value at
    /// * `value` - The symbolic value to store
    /// * `symbol_id` - Optional symbol ID for identity tracking
    pub fn import_symbolic_value(&mut self, addr: u64, value: RustBV, _symbol_id: Option<u64>) {
        // Track as Python-imported so get_state_symbolic_z3_asts can filter it out
        self.imported_addrs.insert(addr);
        // Store in symbolic_objects for lookup
        self.symbolic_objects.insert(addr, value.clone());
        // Update reverse span index
        let width_bits = value.width();
        let sym_bytes = width_bits / 8;
        for i in 1..sym_bytes {
            self.symbolic_spans
                .insert(addr + i as u64, (addr, width_bits));
        }

        // Mark pages as having symbolic bytes
        // Create pages if they don't exist (critical for stack addresses)
        let size = value.width() / 8;
        for i in 0..size {
            let byte_addr = addr + i as u64;
            let page_num = byte_addr >> 12;
            let offset = (byte_addr & PAGE_MASK) as u16;
            let page_addr = page_num << 12;

            // Get or create page
            let page = self
                .pages
                .entry(page_num)
                .or_insert_with(|| MemoryPage::new(page_addr, Permission::RW));

            // Modify in place (COW handled by bitmap allocation in mark_symbolic)
            page.mark_symbolic(offset, 1);
        }
    }

    /// Get the symbolic object at an address if it exists.
    pub fn get_symbolic_object(&self, addr: u64) -> Option<&RustBV> {
        self.symbolic_objects.get(&addr)
    }

    /// Check if there are any symbolic objects in memory.
    pub fn has_symbolic_objects(&self) -> bool {
        !self.symbolic_objects.is_empty()
    }

    /// Get the count of symbolic objects.
    pub fn symbolic_object_count(&self) -> usize {
        self.symbolic_objects.len()
    }

    /// Check if an address was imported from Python.
    pub fn is_imported_addr(&self, addr: u64) -> bool {
        self.imported_addrs.contains(&addr)
    }

    /// Iterate over all symbolic objects in memory.
    pub fn symbolic_objects_iter(&self) -> impl Iterator<Item = (&u64, &RustBV)> {
        self.symbolic_objects.iter()
    }

    /// Clear all symbolic objects (used when resetting state).
    pub fn clear_symbolic_objects(&mut self) {
        self.symbolic_objects.clear();
        self.symbolic_spans.clear();
    }
}
