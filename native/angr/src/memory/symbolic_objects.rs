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
        let sym_bytes = width_bits / 8;
        // angr-6cp06.62: this is a Multi->Symbolic transition site, so it owes
        // the same cleanup `store_concrete`'s symbolic branch (angr-9ke6b.95)
        // and `merge`'s per-byte collapse (angr-0jh0j.31) do. A byte may be
        // marked Multi *or* Symbolic but not both (invariant documented on
        // `SymbolicMemory`), and `load_concrete_common` dispatches
        // Multi-marked bytes before `symbolic_objects` is consulted — so a
        // stale Multi cell here would shadow the imported value on every later
        // load, and `flush_multi_cells` would re-derive `symbolic_objects`
        // from the stale payload on export, clobbering it a second time.
        // Must run before the page-marking loop below, which would otherwise
        // flush a stale `multi_bitmap` back over the clear. Gated on the
        // emptiness check like its two siblings so the common Multi-free
        // import keeps its per-byte cost at zero map lookups.
        if !self.multi_objects.is_empty() {
            for i in 0..sym_bytes {
                // overflow-ok: `Add<u64> for Address` is `wrapping_add`, like
                // the guest — an import straddling the top wraps the same way.
                self.clear_multi_at(addr + i as u64);
            }
        }
        // Track as Python-imported so get_state_symbolic_z3_asts can filter it out
        self.imported_addrs.insert(addr);
        // Store in symbolic_objects for lookup
        self.symbolic_objects.insert(addr, value.clone());
        // Update reverse span index
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

    /// Retire the symbolic object covering `addr`, keeping every *other* byte
    /// it covered symbolic as an 8-bit lane.
    ///
    /// Callers that supersede a single byte (today: `set_multi_alternatives`)
    /// used to spell this as `symbolic_objects.remove(&addr)` plus
    /// `symbolic_spans.remove(&addr)` — exact-key removal on both maps. That
    /// leaves two distinct kinds of wreckage (angr-6cp06.63):
    ///
    /// * `addr` is an *interior* byte of a wider object based at `base < addr`.
    ///   Neither exact-key removal touches `symbolic_objects[base]`, which goes
    ///   on claiming the full width — including the byte just superseded. Rust
    ///   loads stay correct (they consult the exact key first), but
    ///   `flush_multi_cells` later inserts the collapsed byte as a *second*,
    ///   disjoint `symbolic_objects[addr]` entry, and
    ///   `_get_state_symbolic_z3_asts` exports both with no containment
    ///   ordering — so whichever `state.memory.store()` lands last wins, and
    ///   the stale wide one silently clobbers the fresh byte in Python.
    /// * `addr` *is* the base of a wider object. The exact-key removal drops
    ///   the object but orphans its `symbolic_spans` entries at
    ///   `addr+1..addr+width/8`, which name a base with no live object.
    ///
    /// Splitting into per-byte lanes preserves the surviving bytes' symbolic
    /// identity, unlike `store_concrete`'s angr-7qon recovery (which can only
    /// reclassify the survivors as concrete, having already lost the value).
    /// Lanes inherit `imported_addrs` membership from `base` so a split of a
    /// Python-imported object does not start exporting `Extract(sym)` stores
    /// back over bytes Python already holds verbatim (the flareon5 rule
    /// documented on `_get_state_symbolic_z3_asts`).
    ///
    /// No-op when no symbolic object covers `addr`.
    pub(super) fn retire_symbolic_object_at(&mut self, addr: Address) {
        // Resolve the base: an exact-key object wins, else the reverse span
        // index names the wider object this byte lies inside.
        let base = if self.symbolic_objects.contains_key(&addr) {
            addr
        } else {
            match self.symbolic_spans.get(&addr) {
                Some(&(base, _)) => base,
                // SILENT(cat-a): no symbolic object covers this byte — the
                // overwhelmingly common case, and nothing to retire.
                None => return,
            }
        };
        let Some(sym) = self.symbolic_objects.remove(&base) else {
            // A reverse-span entry outliving its object (see
            // `store_concrete`'s angr-7qon tail). Drop the dangling entry so
            // the byte stops resolving to a base that no longer exists.
            self.symbolic_spans.remove(&addr);
            return;
        };
        let endness = self.endness;
        let imported = self.imported_addrs.contains(&base);
        // The object's own width is authoritative; a `symbolic_spans` entry
        // can name a stale wider one, and `extract_byte_lane` would refuse
        // those offsets anyway.
        let sym_bytes = u64::from(sym.width() / 8);
        for i in 0..sym_bytes {
            // overflow-ok: `Add<u64> for Address` is `wrapping_add`, like the
            // guest — an object straddling the top wraps the same way.
            let byte_addr = base + i;
            self.symbolic_spans.remove(&byte_addr);
            if byte_addr == addr {
                continue;
            }
            // `i` is bounded by `sym.width() / 8`, so the lane is in range.
            if let Some(lane) = Self::extract_byte_lane(&sym, i as u32, endness) {
                self.symbolic_objects.insert(byte_addr, lane);
                if imported {
                    self.imported_addrs.insert(byte_addr);
                }
            }
        }
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
