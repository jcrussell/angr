//! Store operations for `SymbolicMemory`.
//!
//! Extracted from `memory/mod.rs` (angr-0lre). Holds the store_*/store_strided/
//! install_multi_for_candidates family in a single file. Multiple `impl SymbolicMemory`
//! blocks across files are fine — Rust permits inherent impls to be split.

use crate::concretize::{AddressConcretizer, ConcretizationResult};
use crate::symbolic::{
    RustBV, SymContext, record_concretize_disjunction, record_mem_ite_depth,
    record_mem_lazy_page_fault, record_mem_store, record_mem_store_symbolic_addr,
};
use crate::vex::Endness;

use super::multi::{MultiAlternative, MultiPayload};
use super::page::{MemoryPage, PAGE_SIZE, Permission};
use super::{Address, MemoryError, SymbolicMemory};

impl SymbolicMemory {
    /// Store a value to memory.
    pub fn store(
        &mut self,
        addr: RustBV,
        value: RustBV,
        ctx: &SymContext,
    ) -> Result<(), MemoryError> {
        // Symbolic-addr subcounter only — the base count + bytes live in
        // `store_concrete` so the state.rs hot path bumps them too.
        if addr.as_u64().is_none() {
            record_mem_store_symbolic_addr();
        }
        // For symbolic addresses, we need to concretize
        let concrete_addr = match addr.as_u64() {
            Some(a) => a,
            None => match ctx.eval(&addr) {
                Some(a) => a as u64,
                None => {
                    return Err(MemoryError::SymbolicAddress {
                        description: "could not resolve address for store".to_string(),
                    });
                }
            },
        };

        let result = self.store_concrete(Address(concrete_addr), value);
        if let Err(MemoryError::UnmappedPageInRegion { .. }) = &result {
            record_mem_lazy_page_fault();
        }
        result
    }

    /// Store to a concrete address.
    pub fn store_concrete(
        &mut self,
        addr: impl Into<Address>,
        value: RustBV,
    ) -> Result<(), MemoryError> {
        let addr = addr.into();
        let size = value.width() / 8;
        record_mem_store(size as u64);

        // Check if pages are mapped (fast path for same-page stores)
        let start_page = addr.page_num();
        let end_page = (addr.raw() + size as u64 - 1) >> 12;

        if start_page == end_page {
            if !self.pages.contains_key(&start_page) {
                return Err(MemoryError::Unmapped {
                    addr: start_page << 12,
                    size: PAGE_SIZE,
                });
            }
        } else {
            for page_num in start_page..=end_page {
                if !self.pages.contains_key(&page_num) {
                    return Err(MemoryError::Unmapped {
                        addr: page_num << 12,
                        size: PAGE_SIZE,
                    });
                }
            }
        }

        self.check_perms_range(start_page, end_page, Permission::W)?;

        // If symbolic, store in symbolic_objects
        if value.is_symbolic() {
            self.symbolic_objects.insert(addr, value.clone());
            // Update reverse span index: map each byte offset to (base_addr, width)
            let width_bits = value.width();
            let sym_bytes = width_bits / 8;
            for i in 1..sym_bytes {
                self.symbolic_spans
                    .insert(addr + i as u64, (addr, width_bits));
            }
            // Mark pages as having symbolic bytes — batch per-page
            let mut current_page_num = u64::MAX;
            let mut current_page: Option<MemoryPage> = None;
            for i in 0..size {
                let byte_addr = addr + i as u64;
                let page_num = byte_addr.page_num();
                let offset = byte_addr.page_offset();
                if page_num != current_page_num {
                    // Flush previous page
                    if let Some(p) = current_page.take() {
                        self.pages.insert(current_page_num, p);
                        self.dirty_pages.insert(current_page_num);
                    }
                    current_page_num = page_num;
                    current_page = self.pages.get(&page_num).cloned();
                }
                if let Some(ref mut p) = current_page {
                    p.mark_symbolic(offset, 1);
                }
            }
            if let Some(p) = current_page {
                self.pages.insert(current_page_num, p);
                self.dirty_pages.insert(current_page_num);
            }
            return Ok(());
        }

        // Concrete store
        let concrete_val = value.to_u128();

        // Convert to bytes based on endianness
        let bytes: Vec<u8> = match self.endness {
            Endness::Little => (0..size).map(|i| (concrete_val >> (i * 8)) as u8).collect(),
            Endness::Big => (0..size)
                .rev()
                .map(|i| (concrete_val >> (i * 8)) as u8)
                .collect(),
        };

        // Write to pages
        let mut remaining = &bytes[..];
        let mut current_addr = addr;

        while !remaining.is_empty() {
            let page_num = current_addr.page_num();
            let page_offset = current_addr.page_offset();
            let bytes_in_page = ((PAGE_SIZE - page_offset as u64) as usize).min(remaining.len());

            if let Some(page) = self.pages.get_mut(&page_num) {
                page.store_concrete(page_offset, &remaining[..bytes_in_page]);
                // Mark page as dirty
                self.dirty_pages.insert(page_num);
            }

            remaining = &remaining[bytes_in_page..];
            current_addr = current_addr + bytes_in_page as u64;
        }

        // Clear any symbolic object at this address and its span entries
        if let Some(old_sym) = self.symbolic_objects.remove(&addr) {
            let old_bytes = old_sym.width() / 8;
            for i in 1..old_bytes {
                self.symbolic_spans.remove(&(addr + i as u64));
            }
            // angr-7qon: if the concrete write doesn't cover the full
            // wider sym, the trailing bytes [addr+size, addr+old_bytes)
            // would be orphaned — their `symbolic_spans` entries were
            // removed by the loop above, but the page bitmap still says
            // symbolic (page.store_concrete only cleared bits within
            // [0, size)). Per the bug description's option (b): clear
            // the bitmap bits so those bytes reclassify as concrete.
            // We've already lost the wider sym tracking, so this is the
            // best recovery — the page data bytes for the survivors
            // were never touched by `mark_symbolic` and remain whatever
            // they were prior to the sym store.
            if old_bytes > size {
                let mut tail = addr + size as u64;
                let tail_end = addr + old_bytes as u64;
                while tail < tail_end {
                    let page_num = tail.page_num();
                    let page_offset = tail.page_offset();
                    let bytes_in_page =
                        ((PAGE_SIZE - page_offset as u64) as usize).min((tail_end - tail) as usize);
                    if let Some(page) = self.pages.get_mut(&page_num) {
                        page.clear_symbolic(page_offset, bytes_in_page as u16);
                        self.dirty_pages.insert(page_num);
                    }
                    tail = tail + bytes_in_page as u64;
                }
            }
        }

        // angr-1tes: drop any Multi cells overwritten by this concrete
        // write and bump their per-byte version so the wider-load cache
        // fingerprint mismatches on next read. The page-level
        // `multi_bitmap` is already cleared inside `page.store_concrete`,
        // but the sidecar `multi_objects` map and `multi_versions`
        // counter are owned here and must be kept in sync. Without this,
        // `load_concrete_lazy_inner`'s dispatcher (which checks
        // `multi_objects.contains_key`) would still hand the load to
        // `assemble_load_with_multi`, folding the orphaned alternatives
        // over the new page byte and returning the old Multi value.
        if !self.multi_objects.is_empty() {
            for i in 0..size {
                let byte_addr = addr + i as u64;
                if self.multi_objects.remove(&byte_addr).is_some() {
                    self.bump_multi_version(byte_addr);
                }
            }
        }

        Ok(())
    }

    /// Store to a symbolic address with concretization support.
    ///
    /// This method handles symbolic addresses by:
    /// 1. Trying to concretize the address to a single value (fast path)
    /// 2. Performing conditional stores for strided access patterns
    /// 3. Performing conditional stores for multiple possible addresses
    /// 4. Returning an error if the address range is too large
    ///
    /// For multiple addresses, each candidate gets a conditional store:
    /// `mem[candidate] = If(addr == candidate, new_value, mem[candidate])`
    ///
    /// For unmapped pages in lazy regions, returns `UnmappedPageInRegion` so
    /// the interpreter can fetch the page on-demand.
    ///
    /// # Arguments
    /// * `addr` - The symbolic address to store to
    /// * `value` - The value to store
    /// * `ctx` - The solver context
    /// * `concretizer` - The address concretizer configuration
    ///
    /// # Returns
    /// Ok(()) on success, or a MemoryError if storing fails.
    pub fn store_symbolic(
        &mut self,
        addr: RustBV,
        value: RustBV,
        ctx: &SymContext,
        concretizer: &AddressConcretizer,
    ) -> Result<(), MemoryError> {
        // Fast path: concrete address
        if let Some(concrete_addr) = addr.as_u64() {
            return self.store_concrete_lazy(concrete_addr, value);
        }

        // AVOID_MULTIVALUED_WRITES: silently drop the store. Mirrors the
        // early `return` at `address_concretization_mixin.py:327-329`.
        if concretizer.should_avoid_multivalued_write(&addr) {
            return Ok(());
        }

        // Try to concretize the address (write mode: falls back to Max solution)
        match concretizer.concretize_write(&addr, ctx) {
            ConcretizationResult::Single(concrete_addr) => {
                self.store_concrete_lazy(concrete_addr, value)
            }
            ConcretizationResult::Strided {
                base,
                stride,
                count,
            } => self.store_strided(&addr, &value, base, stride, count, ctx),
            ConcretizationResult::Multiple(addrs) => {
                let size = value.width() / 8;
                for &candidate in &addrs {
                    let addr_const = RustBV::concrete(candidate as u128, addr.width());
                    let cond = addr.eq(&addr_const, ctx);
                    let current = self.load_concrete_lazy(Address(candidate), size, ctx)?;
                    let conditional_value = cond.ite(&value, &current, ctx);
                    self.store_concrete_lazy(candidate, conditional_value)?;
                }
                // Phase 0 instrumentation: eager ITE chain depth = #candidates.
                record_mem_ite_depth(addrs.len() as u32);
                // angr-62li: hoist the addr-domain disjunction.
                Self::assert_address_disjunction(&addr, &addrs, ctx);
                Ok(())
            }
            ConcretizationResult::TooLarge { min, max, .. } => Err(MemoryError::SymbolicAddress {
                description: format!(
                    "address range too large for concretization: 0x{:x} - 0x{:x}",
                    min, max
                ),
            }),
            ConcretizationResult::Failed(reason) => Err(MemoryError::SymbolicAddress {
                description: reason,
            }),
        }
    }

    /// Store to strided addresses with conditional stores.
    ///
    /// For each address in the strided pattern, performs:
    /// `mem[addr] = If(symbolic_addr == addr, new_value, mem[addr])`
    pub(super) fn store_strided(
        &mut self,
        addr_expr: &RustBV,
        value: &RustBV,
        base: u64,
        stride: u64,
        count: u64,
        ctx: &SymContext,
    ) -> Result<(), MemoryError> {
        let size = value.width() / 8;

        for i in 0..count {
            let candidate = base + i * stride;

            // Build condition: addr == candidate
            let addr_const = RustBV::concrete(candidate as u128, addr_expr.width());
            let cond = addr_expr.eq(&addr_const, ctx);

            // Load current value at candidate address
            let current = self.load_concrete_lazy(Address(candidate), size, ctx)?;

            // Build conditional value
            let conditional_value = cond.ite(value, &current, ctx);

            // Store the conditional value
            self.store_concrete_lazy(candidate, conditional_value)?;
        }

        // Phase 0 instrumentation: record the depth of the eager ITE chain
        // produced by this strided store. Phase 1+ Multi cells will record
        // the same metric on Multi insertion for direct comparison.
        record_mem_ite_depth(count as u32);

        Ok(())
    }

    /// Unified symbolic store that handles all concretization results in Rust.
    ///
    /// This method replaces the Python fallback for symbolic memory stores.
    /// It handles all cases by performing conditional stores for each candidate address:
    /// `mem[candidate] = If(addr == candidate, new_value, mem[candidate])`
    ///
    /// # Arguments
    /// * `addr` - The symbolic address to store to
    /// * `value` - The value to store
    /// * `ctx` - The solver context
    /// * `concretizer` - The address concretizer
    ///
    /// # Returns
    /// Ok(()) on success, or an error if storing fails.
    pub fn store_symbolic_unified(
        &mut self,
        addr: RustBV,
        value: RustBV,
        ctx: &SymContext,
        concretizer: &AddressConcretizer,
    ) -> Result<Option<ConcretizationResult>, MemoryError> {
        // Fast path: concrete address
        if let Some(concrete_addr) = addr.as_u64() {
            self.store_concrete_automap(concrete_addr, value)?;
            return Ok(Some(ConcretizationResult::Single(concrete_addr)));
        }

        // AVOID_MULTIVALUED_WRITES: silently drop the store.
        if concretizer.should_avoid_multivalued_write(&addr) {
            return Ok(None);
        }

        // Try to concretize the address (write mode: falls back to Max solution)
        let result = concretizer.concretize_write(&addr, ctx);
        match &result {
            ConcretizationResult::Single(concrete_addr) => {
                self.store_concrete_automap(*concrete_addr, value)?;
                Ok(Some(result))
            }
            ConcretizationResult::Multiple(addrs) => {
                self.install_multi_for_candidates_safe(&addr, &value, addrs, ctx)?;
                // angr-62li: hoist the addr-domain disjunction.
                Self::assert_address_disjunction(&addr, addrs, ctx);
                Ok(Some(result))
            }
            ConcretizationResult::Strided {
                base,
                stride,
                count,
            } => {
                let (base, stride, count) = (*base, *stride, *count);
                let addrs: Vec<u64> = (0..count).map(|i| base + i * stride).collect();
                self.install_multi_for_candidates_safe(&addr, &value, &addrs, ctx)?;
                Ok(Some(result))
            }
            ConcretizationResult::TooLarge { min, max, .. } => {
                // Return error so caller can fall back to Python's memory model,
                // which handles large symbolic address ranges natively.
                Err(MemoryError::SymbolicAddress {
                    description: format!(
                        "address range too large for concretization: 0x{:x} - 0x{:x}",
                        min, max
                    ),
                })
            }
            ConcretizationResult::Failed(reason) => Err(MemoryError::SymbolicAddress {
                description: reason.clone(),
            }),
        }
    }

    /// Store using a pre-computed concretization result.
    ///
    /// `Multiple` / `Strided` results install Multi cells via
    /// `install_multi_for_candidates` rather than folding eager ITE
    /// chains at store time. Load-time collapse handles the cost only for
    /// bytes that are actually re-read (see `assemble_load_with_multi`).
    /// `Single` and `TooLarge` / `Failed` paths are unchanged.
    pub fn store_with_concretization(
        &mut self,
        addr: &RustBV,
        value: RustBV,
        conc_result: &ConcretizationResult,
        ctx: &SymContext,
    ) -> Result<(), MemoryError> {
        match conc_result {
            ConcretizationResult::Single(concrete_addr) => {
                self.store_concrete_automap(*concrete_addr, value)
            }
            ConcretizationResult::Multiple(addrs) => {
                let addrs_v: Vec<u64> = addrs.clone();
                self.install_multi_for_candidates_safe(addr, &value, &addrs_v, ctx)?;
                // angr-62li: hoist the addr-domain disjunction.
                Self::assert_address_disjunction(addr, &addrs_v, ctx);
                Ok(())
            }
            ConcretizationResult::Strided {
                base,
                stride,
                count,
            } => {
                let addrs: Vec<u64> = (0..*count).map(|i| base + i * stride).collect();
                self.install_multi_for_candidates_safe(addr, &value, &addrs, ctx)
            }
            ConcretizationResult::TooLarge { min, max, .. } => {
                // Return error so caller can fall back to Python's memory model,
                // which handles large symbolic address ranges natively.
                Err(MemoryError::SymbolicAddress {
                    description: format!(
                        "address range too large for concretization: 0x{:x} - 0x{:x}",
                        min, max
                    ),
                })
            }
            ConcretizationResult::Failed(reason) => Err(MemoryError::SymbolicAddress {
                description: reason.clone(),
            }),
        }
    }

    /// Lazy alternative to `store_conditional_multiple`: instead of folding
    /// one ITE chain per candidate cell at store time, append per-byte
    /// alternatives to each candidate's `MultiPayload`. The load-time
    /// collapse (Phase 1.2 `assemble_load_with_multi`) materializes the
    /// ITE only for the bytes a subsequent load actually touches.
    ///
    /// Splits `value` into bytes (endianness-aware), then for each
    /// candidate `c` and each byte offset `b`, appends
    /// `(addr_expr == c, byte_b)` to `multi_objects[c + b]`. Existing
    /// alternatives at the same byte address are preserved — new ones
    /// are appended after. By the Multi-payload invariant (disjoint
    /// conds), order does not affect the load result.
    ///
    /// Per memory `invariant-mem-ite-depth-counter`, each per-byte
    /// `set_multi_alternatives` call records the payload length.
    pub(super) fn install_multi_for_candidates(
        &mut self,
        addr_expr: &RustBV,
        value: &RustBV,
        addrs: &[u64],
        ctx: &SymContext,
    ) -> Result<(), MemoryError> {
        let size = value.width() / 8;
        let endness = self.endness;

        // Auto-map all candidate byte addresses before installing — matches
        // what set_multi_alternatives does per-byte, but we batch it so the
        // permission check happens up-front for every page.
        for &cand in addrs {
            let start_page = cand >> 12;
            let end_page = (cand + size as u64 - 1) >> 12;
            for page_num in start_page..=end_page {
                self.pages
                    .entry(page_num)
                    .or_insert_with(|| MemoryPage::new(page_num << 12, Permission::RW));
            }
        }

        // For each candidate, build cond once and per-byte split.
        // Accumulate by byte address so payloads merge cleanly with
        // existing alternatives.
        for &cand in addrs {
            let addr_const = RustBV::concrete(cand as u128, addr_expr.width());
            let cond = addr_expr.eq(&addr_const, ctx);
            for b in 0..size {
                let byte_addr = cand + b as u64;
                let byte_value = Self::extract_byte_lane(value, b, endness, ctx).ok_or(
                    MemoryError::SymbolicAddress {
                        description: "value byte offset out of range".to_string(),
                    },
                )?;
                let alt = MultiAlternative::new(cond.clone(), byte_value);

                // Merge with any existing payload at this byte. New alt
                // goes after existing ones; conds are disjoint so order
                // is semantically irrelevant.
                let merged: Vec<MultiAlternative> =
                    match self.multi_objects.get(&Address(byte_addr)) {
                        Some(p) => {
                            let mut v = p.alternatives().to_vec();
                            v.push(alt);
                            v
                        }
                        None => vec![alt],
                    };
                self.set_multi_alternatives(byte_addr, MultiPayload::from_alternatives(merged));
            }
        }
        Ok(())
    }

    /// Production-safe Multi-cell install.
    ///
    /// Phase 2 (angr-qh5u) entry point used by `store_symbolic_unified` and
    /// `store_with_concretization`. Mirrors the eager safety semantics of
    /// the pre-Phase-2 `store_strided` so Multi installation can't lose
    /// Python backer data:
    ///
    /// * If any candidate page is in a lazy region, return
    ///   `UnmappedPageInRegion { page_addr: first_missing }` so the
    ///   interpreter can fetch the page and retry. Auto-mapping zero
    ///   pages here would diverge state from Python (which has the
    ///   backer data) — same invariant that
    ///   `prepare_addresses_for_ite` and `store_concrete_lazy` enforce.
    /// * Otherwise filter `addrs` down to candidates whose pages are
    ///   already mapped (mirrors `prepare_addresses_for_ite`'s skip
    ///   behavior for non-lazy unmapped pages — those are treated as
    ///   unreachable since the symbolic address could not validly
    ///   resolve to them) and pass to `install_multi_for_candidates`.
    pub(super) fn install_multi_for_candidates_safe(
        &mut self,
        addr_expr: &RustBV,
        value: &RustBV,
        addrs: &[u64],
        ctx: &SymContext,
    ) -> Result<(), MemoryError> {
        let size = value.width() / 8;

        // Probe every candidate's pages first. Lazy-region misses must
        // surface so the interpreter fetches the page; non-lazy misses
        // are silently filtered (matches eager `prepare_addresses_for_ite`).
        let mut ready: Vec<u64> = Vec::with_capacity(addrs.len());
        for &cand in addrs {
            let start_page = cand >> 12;
            let end_page = (cand + size as u64 - 1) >> 12;
            let mut all_mapped = true;
            for page_num in start_page..=end_page {
                if !self.pages.contains_key(&page_num) {
                    if self.is_in_lazy_region(page_num) {
                        return Err(MemoryError::UnmappedPageInRegion {
                            page_addr: page_num << 12,
                        });
                    }
                    all_mapped = false;
                    break;
                }
            }
            if all_mapped {
                ready.push(cand);
            }
        }

        if ready.is_empty() {
            // No candidate page is currently mapped and none are lazy.
            // Match eager semantics: silently no-op rather than error.
            return Ok(());
        }

        self.install_multi_for_candidates(addr_expr, value, &ready, ctx)
    }

    /// angr-62li: hoist `Or(addr == a0, ..., addr == aK)` to the top-level
    /// solver when the concretizer returned `Multiple`. Python's
    /// `address_concretization_mixin` (angr/storage/memory_mixins/
    /// address_concretization_mixin.py:292-294, 342-344) does the same for
    /// every concretized read/write; the Rust-native path was missing it,
    /// so Z3's `propagate_values` tactic never saw the address domain
    /// restriction.
    ///
    /// Scope:
    /// * Only the `Multiple` arm. `Strided` is skipped — the strided
    ///   abstraction is concretizer policy, not a tight constraint.
    /// * Gated by `MAX_DISJUNCTION_TERMS = 8`. Empirically (flareon2015_5,
    ///   sym-write) hoisting a 64-way Or regresses Z3 because the cost of
    ///   processing the new constraint on every subsequent `check()` swamps
    ///   the propagate-values benefit. Small K (<=8) lets cheap cases
    ///   benefit without the long-Or tax.
    ///
    /// Returns early without touching the solver when `addrs.len() <= 1`
    /// (no domain restriction to communicate), when `addr` is concrete,
    /// or when `addrs.len() > MAX_DISJUNCTION_TERMS`.
    pub(super) fn assert_address_disjunction(addr: &RustBV, addrs: &[u64], ctx: &SymContext) {
        /// Maximum disjunction width worth hoisting. Above this the
        /// per-solver-check overhead of the long Or chain dominates.
        const MAX_DISJUNCTION_TERMS: usize = 8;

        if addrs.len() <= 1 || addrs.len() > MAX_DISJUNCTION_TERMS || addr.as_u64().is_some() {
            return;
        }
        let width = addr.width();
        let mut disjunction: Option<RustBV> = None;
        for &cand in addrs {
            let addr_const = RustBV::concrete(cand as u128, width);
            let eq = addr.eq(&addr_const, ctx);
            disjunction = Some(match disjunction {
                Some(prev) => prev.or(&eq, ctx),
                None => eq,
            });
        }
        if let Some(or_bv) = disjunction {
            // or() simplifies `eq | 1 -> 1` etc., so a concrete-true result
            // means the disjunction is tautological — nothing to assert.
            if or_bv.as_u128() == Some(1) {
                return;
            }
            ctx.assume_true(&or_bv);
            record_concretize_disjunction(addrs.len() as u32);
        }
    }

    /// Test-rig helper: install Multi alternatives for a multi-byte value
    /// across a known list of candidate addresses without going through
    /// the address concretizer. Pure wrapper around
    /// `install_multi_for_candidates`. Useful for round-trip tests that
    /// drive the Phase 1.2 load path.
    pub fn store_concrete_multi(
        &mut self,
        addr_expr: &RustBV,
        value: &RustBV,
        candidates: &[u64],
        ctx: &SymContext,
    ) -> Result<(), MemoryError> {
        self.install_multi_for_candidates(addr_expr, value, candidates, ctx)
    }

    /// Lazy variant of `store_symbolic_unified`. After Phase 2 (angr-qh5u)
    /// the default `store_symbolic_unified` also installs Multi cells for
    /// Multiple / Strided, so this is now an alias kept for the Python
    /// `_try_multi_cell_store` wiring (Phase 1.4, angr-5zw8) and for
    /// existing tests that monkey-patch routing. Behavior is identical to
    /// the default unified path.
    ///
    /// Unlike the default path, this variant uses the bare
    /// `install_multi_for_candidates` helper (which auto-maps unmapped
    /// pages as zero RW), matching the original Phase 1.3 contract used
    /// by SimProcedure-originated stores. The default path uses the
    /// `_safe` variant which preserves Python backer data.
    pub fn store_symbolic_unified_multi(
        &mut self,
        addr: RustBV,
        value: RustBV,
        ctx: &SymContext,
        concretizer: &AddressConcretizer,
    ) -> Result<Option<ConcretizationResult>, MemoryError> {
        // Fast path: concrete address — eager store, no Multi cells needed.
        if let Some(concrete_addr) = addr.as_u64() {
            self.store_concrete_automap(concrete_addr, value)?;
            return Ok(Some(ConcretizationResult::Single(concrete_addr)));
        }

        // AVOID_MULTIVALUED_WRITES: silently drop the store.
        if concretizer.should_avoid_multivalued_write(&addr) {
            return Ok(None);
        }

        let result = concretizer.concretize_write(&addr, ctx);
        match &result {
            ConcretizationResult::Single(concrete_addr) => {
                self.store_concrete_automap(*concrete_addr, value)?;
                Ok(Some(result))
            }
            ConcretizationResult::Multiple(addrs) => {
                let addrs_v: Vec<u64> = addrs.clone();
                self.install_multi_for_candidates(&addr, &value, &addrs_v, ctx)?;
                // angr-62li: hoist the addr-domain disjunction.
                Self::assert_address_disjunction(&addr, &addrs_v, ctx);
                Ok(Some(result))
            }
            ConcretizationResult::Strided {
                base,
                stride,
                count,
            } => {
                let (base, stride, count) = (*base, *stride, *count);
                let addrs_v: Vec<u64> = (0..count).map(|i| base + i * stride).collect();
                self.install_multi_for_candidates(&addr, &value, &addrs_v, ctx)?;
                Ok(Some(result))
            }
            ConcretizationResult::TooLarge { min, max, .. } => Err(MemoryError::SymbolicAddress {
                description: format!(
                    "address range too large for concretization: 0x{:x} - 0x{:x}",
                    min, max
                ),
            }),
            ConcretizationResult::Failed(reason) => Err(MemoryError::SymbolicAddress {
                description: reason.clone(),
            }),
        }
    }

    /// Store to a concrete address, returning UnmappedPageInRegion for lazy regions.
    pub fn store_concrete_lazy(
        &mut self,
        addr: impl Into<Address>,
        value: RustBV,
    ) -> Result<(), MemoryError> {
        let addr = addr.into();
        let size = value.width() / 8;

        // Check if pages are mapped
        let start_page = addr.page_num();
        let end_page = (addr.raw() + size as u64 + PAGE_SIZE - 1) >> 12;

        for page_num in start_page..end_page {
            if !self.pages.contains_key(&page_num) {
                // Page not mapped - check if it's in a lazy region
                if self.is_in_lazy_region(page_num) {
                    return Err(MemoryError::UnmappedPageInRegion {
                        page_addr: page_num << 12,
                    });
                } else {
                    return Err(MemoryError::Unmapped {
                        addr: page_num << 12,
                        size: PAGE_SIZE,
                    });
                }
            }
        }

        // Permission checks live in store_concrete; this wrapper only adds
        // lazy-region detection for unmapped pages.
        self.store_concrete(addr, value)
    }

    /// Store to a concrete address with lazy region support.
    ///
    /// # Deprecation Warning
    ///
    /// This function previously auto-mapped zero pages for unmapped regions,
    /// but that behavior caused state divergence with Python's actual backer
    /// data. Now it returns UnmappedPageInRegion error so callers can fall
    /// back to Python callbacks to handle the store correctly.
    ///
    /// If you need auto-mapping behavior for internal Rust operations that
    /// don't involve Python state, use `store_concrete_automap_internal`.
    pub fn store_concrete_automap(
        &mut self,
        addr: impl Into<Address>,
        value: RustBV,
    ) -> Result<(), MemoryError> {
        let addr = addr.into();
        let size = value.width() / 8;
        let start_page = addr.page_num();
        let end_page = (addr.raw() + size as u64 + PAGE_SIZE - 1) >> 12;

        // Check all pages are mapped - do NOT auto-map
        for page_num in start_page..end_page {
            if !self.pages.contains_key(&page_num) {
                let page_addr = page_num << 12;
                if self.is_in_lazy_region(page_num) {
                    // Return error so caller can fall back to Python
                    return Err(MemoryError::UnmappedPageInRegion { page_addr });
                } else {
                    return Err(MemoryError::Unmapped {
                        addr: page_addr,
                        size: PAGE_SIZE,
                    });
                }
            }
        }

        // All pages mapped, proceed with store
        self.store_concrete(addr, value)
    }

    /// Store to a concrete address with internal auto-mapping.
    ///
    /// This is for internal Rust operations that don't involve Python state.
    /// For interpreter callbacks, use `store_concrete_automap` which propagates
    /// errors so Python can handle the store correctly.
    pub fn store_concrete_automap_internal(
        &mut self,
        addr: impl Into<Address>,
        value: RustBV,
    ) -> Result<(), MemoryError> {
        let addr = addr.into();
        let size = value.width() / 8;
        let start_page = addr.page_num();
        let end_page = (addr.raw() + size as u64 + PAGE_SIZE - 1) >> 12;

        // Auto-map any missing pages in lazy regions
        for page_num in start_page..end_page {
            if !self.pages.contains_key(&page_num) {
                let page_addr = page_num << 12;
                if self.is_in_lazy_region(page_num) {
                    self.auto_map_zero_page(page_addr);
                } else {
                    return Err(MemoryError::Unmapped {
                        addr: page_addr,
                        size: PAGE_SIZE,
                    });
                }
            }
        }

        // All pages now mapped, proceed with store
        self.store_concrete(addr, value)
    }

    /// Store an arbitrarily-wide concrete value given its little-endian
    /// value bytes (as produced by `bv_to_bytes`).
    ///
    /// `store_concrete` funnels the value through a single `u128`, so a store
    /// wider than 16 bytes (e.g. a 32-byte V256/AVX register) would overflow
    /// the shift and silently corrupt the low bytes. This helper chunks the
    /// store into `<=16`-byte `RustBV` writes so wide values round-trip
    /// byte-for-byte. Each chunk auto-maps lazy pages like
    /// `store_concrete_automap_internal` and the per-chunk address is computed
    /// so the final memory layout matches a single full-width store under
    /// either endianness.
    pub fn store_concrete_le_bytes_automap_internal(
        &mut self,
        addr: impl Into<Address>,
        le_bytes: &[u8],
    ) -> Result<(), MemoryError> {
        let addr = addr.into();
        let total = le_bytes.len();
        if total == 0 {
            return Ok(());
        }
        // Fast path: fits in a u128 — preserve the exact prior behavior.
        if total <= 16 {
            return self.store_concrete_automap_internal(addr, le_bytes_to_bv(le_bytes));
        }
        // Wide path: split into <=16-byte chunks. `store_concrete` reverses the
        // chunk bytes under big-endian, so for big-endian we place chunks from
        // the high value bytes (lowest data offset) at the highest addresses.
        let mut off = 0usize;
        while off < total {
            let cs = (total - off).min(16);
            let chunk_addr = match self.endness {
                Endness::Little => addr + off as u64,
                Endness::Big => addr + (total - off - cs) as u64,
            };
            self.store_concrete_automap_internal(
                chunk_addr,
                le_bytes_to_bv(&le_bytes[off..off + cs]),
            )?;
            off += cs;
        }
        Ok(())
    }
}

/// Pack up to 16 little-endian value bytes into a concrete `RustBV` of width
/// `bytes.len() * 8`.
fn le_bytes_to_bv(bytes: &[u8]) -> RustBV {
    debug_assert!(bytes.len() <= 16);
    let mut val: u128 = 0;
    for (i, &b) in bytes.iter().enumerate() {
        val |= (b as u128) << (i * 8);
    }
    RustBV::concrete(val, (bytes.len() * 8) as u32)
}
