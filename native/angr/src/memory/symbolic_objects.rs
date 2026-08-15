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

use super::{Address, MemoryError, MemoryPage, Permission, SymbolicMemory};
use crate::symbolic::RustBV;

impl SymbolicMemory {
    /// Import a symbolic value with identity preservation.
    ///
    /// This stores a symbolic value at the given address and ensures the
    /// symbol ID is tracked for later export back to Python.
    ///
    /// # Arguments
    /// * `addr` - The address to store the value at
    /// * `value` - The symbolic value to store
    /// * `symbol_id` - Optional symbol ID for identity tracking
    ///
    /// # Errors
    /// [`MemoryError::UnalignedWidth`] when `value.width()` is not a positive
    /// multiple of 8. Memory is byte-addressed, so every downstream structure
    /// here is sized by `width_bits / 8`: a 1-bit value truncates to size 0,
    /// the page-marking loop below never runs, and a later byte load — which
    /// finds `symbolic_objects.contains_key(addr)` true but the page bitmap
    /// clear — falls through to the page's concrete placeholder byte and
    /// returns 0 instead of the imported symbol. Rejecting mirrors
    /// `store_concrete`'s [`MemoryError::ZeroSize`] guard (see
    /// `end_page_inclusive`) rather than silently storing an object
    /// `extract_byte_lane` can never reconstruct (angr-sqfj8.73).
    pub fn import_symbolic_value(
        &mut self,
        addr: impl Into<Address>,
        value: RustBV,
        _symbol_id: Option<u64>,
    ) -> Result<(), MemoryError> {
        let addr = addr.into();
        let width_bits = value.width();
        if width_bits == 0 || !width_bits.is_multiple_of(8) {
            return Err(MemoryError::UnalignedWidth {
                addr: addr.raw(),
                width_bits,
            });
        }
        // Track as Python-imported so get_state_symbolic_z3_asts can filter it out
        self.imported_addrs.insert(addr);
        // Store in symbolic_objects for lookup
        self.symbolic_objects.insert(addr, value.clone());
        // Update reverse span index
        let sym_bytes = width_bits / 8;
        for i in 1..sym_bytes {
            // overflow-ok: `Add<u64> for Address` is `wrapping_add`, like the guest.
            self.symbolic_spans
                .insert(addr + i as u64, (addr, width_bits));
        }

        // Mark pages as having symbolic bytes
        // Create pages if they don't exist (critical for stack addresses)
        for i in 0..sym_bytes {
            // overflow-ok: `Address` arithmetic is wrapping (see above).
            let byte_addr = addr + i as u64;
            let page_num = byte_addr.page_num();
            let offset = byte_addr.page_offset();
            let page_addr = byte_addr.page_base();

            // Get or create page
            let page = self
                .pages
                .entry(page_num)
                .or_insert_with(|| MemoryPage::new(page_addr, Permission::RW));

            // Modify in place (COW handled by bitmap allocation in mark_symbolic)
            page.mark_symbolic(offset, 1);
        }
        Ok(())
    }

    /// Get the symbolic object at an address if it exists.
    pub fn get_symbolic_object(&self, addr: impl Into<Address>) -> Option<&RustBV> {
        self.symbolic_objects.get(&addr.into())
    }

    /// Get the count of symbolic objects.
    pub fn symbolic_object_count(&self) -> usize {
        self.symbolic_objects.len()
    }

    /// Check if an address was imported from Python.
    pub fn is_imported_addr(&self, addr: impl Into<Address>) -> bool {
        self.imported_addrs.contains(&addr.into())
    }

    /// Iterate over all symbolic objects in memory.
    ///
    /// Returns `Address` by value (it's `Copy`) so callers can pattern-match
    /// `for (addr, bv) in ...` without borrowing the key.
    pub fn symbolic_objects_iter(&self) -> impl Iterator<Item = (Address, &RustBV)> {
        self.symbolic_objects.iter().map(|(a, b)| (*a, b))
    }
}
