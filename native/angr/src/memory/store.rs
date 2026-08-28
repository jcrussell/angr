//! Store operations for `SymbolicMemory`.
//!
//! Extracted from `memory/mod.rs` (angr-0lre). Holds the store_*/store_strided/
//! install_multi_for_candidates family in a single file. Multiple `impl SymbolicMemory`
//! blocks across files are fine — Rust permits inherent impls to be split.
//!
//! # Variant matrix (angr-9ke6b.102)
//!
//! Same four axes as the `memory::load` matrix, with `Multi` reading
//! differently on the write side — a store either **clears** the Multi cells it
//! overwrites or **installs** new ones:
//!
//! * **Perm** — `check_perms_range(.., Permission::W)`. Every eager path
//!   inherits it from `SymbolicMemory::store_concrete`; the Multi-installing
//!   paths run their own copy in `install_multi_for_candidates` (angr-9ke6b.94)
//!   *before* their auto-map loop, so an already-mapped read-only candidate is
//!   still rejected.
//! * **Unmapped** — `Unmapped` hard error vs. `UnmappedPageInRegion` (fetch +
//!   retry) vs. auto-mapped away.
//! * **Auto-map** — unlike loads, several store paths *do* map missing pages as
//!   zero-filled RW. That is safe only where Python holds no backer data for
//!   the page; the `_safe` Multi installer exists precisely to refuse it.
//! * **Multi** — `clear` = drops any `Multi` cell on the overwritten bytes
//!   (required by `invariant-multi-vs-symbolic-cell-states`: a byte is Multi
//!   *or* Symbolic, never both). `install` = appends `(cond, byte)`
//!   alternatives for load-time collapse instead of folding eager ITEs.
//!
//! | Entry point | Address | Perm | Unmapped | Auto-map | Multi |
//! |---|---|---|---|---|---|
//! | [`SymbolicMemory::store`] | symbolic (`ctx.eval` + pin fallback) | W | `Unmapped` | no | clear |
//! | [`SymbolicMemory::store_concrete`] | concrete | W | `Unmapped` | no | clear |
//! | [`SymbolicMemory::store_concrete_lazy`] | concrete | W | `UnmappedPageInRegion` | no | clear |
//! | [`SymbolicMemory::store_concrete_automap_internal`] | concrete | W | `Unmapped` (non-lazy only) | **yes**, zero RW in lazy regions | clear |
//! | [`SymbolicMemory::store_concrete_le_bytes_automap_internal`] | concrete | W | `Unmapped` (non-lazy only) | **yes**, per chunk | clear |
//! | [`SymbolicMemory::store_symbolic`] | symbolic (concretizer) | W | `UnmappedPageInRegion` | no | clear (eager ITE per candidate) |
//! | [`SymbolicMemory::store_symbolic_unified`] | symbolic (concretizer) | W | `UnmappedPageInRegion` | mapped candidates only | **install** (`_safe`) |
//! | [`SymbolicMemory::store_symbolic_unified_multi`] | symbolic (multiwrite concretizer) | W | `UnmappedPageInRegion` (concrete-address path only) | **yes**, zero RW for every candidate | **install** (bare) |
//! | [`SymbolicMemory::store_with_concretization`] | pre-computed result | W | `UnmappedPageInRegion` | mapped candidates only | **install** (`_safe`) |
//! | `SymbolicMemory::store_concrete_multi` (`#[cfg(test)]`) | explicit candidate list | W | never — auto-mapped | **yes**, zero RW | **install** (bare, test rig) |
//!
//! Notes that the axes alone don't carry:
//!
//! * Only `store_concrete` bumps the `mem_store` volume counter (mirror of the
//!   load-side `invariant-mem-counter-two-paths` rule).
//! * There used to be a `store_concrete_automap` that notably did *not*
//!   auto-map — the name was historical; angr-5mnx3.27 removed it as a
//!   body-identical duplicate of `store_concrete_lazy`, mirroring what
//!   angr-sqfj8.75 did to the load-side `load_concrete_automap`.
//! * `store_concrete` uses an **inclusive** end-page range
//!   (`end_page_inclusive`); `store_concrete_lazy` (via
//!   `check_pages_mapped_lazy`) and `store_concrete_automap_internal` (via its
//!   own auto-map loop) use an exclusive ceil-div range
//!   (`end_page_exclusive`). Both cover the accessed bytes.
//! * The bare vs. `_safe` Multi installer is the whole auto-map distinction:
//!   `install_multi_for_candidates` maps every candidate page RW, while
//!   `install_multi_for_candidates_safe` returns `UnmappedPageInRegion` for a
//!   lazy-region miss (so Python's backer data is fetched first) and silently
//!   filters non-lazy misses as unreachable — when that filter empties the
//!   candidate list the store is dropped entirely (tagged `SILENT(cat-b)`).
//! * Every symbolic entry point drops the store entirely under
//!   AVOID_MULTIVALUED_WRITES, before any page is touched (tagged
//!   `SILENT(cat-b)` at the two `_unified` sites).
//!
//! **Adding a variant:** route it through `store_concrete` (which owns the
//! permission check, the Multi clear, and the counter) or, for a Multi-cell
//! path, through `install_multi_for_candidates{,_safe}` — then add a row here.
//! The load-side matrix lives in the `memory::load` module docs.
//!
//! **Panic policy (angr-9ke6b.212):** stores run on guest-supplied addresses
//! and values, so nothing here may panic on their shape. An unresolvable
//! address becomes `MemoryError::SymbolicAddress` (via
//! [`ConcretizationResult::as_symbolic_address_error`]), which the caller turns
//! into a Python-memory-model fallback.
//!
//! **Enforcement (angr-qwyti.11):** this module carries
//! `#![deny(clippy::unwrap_used, clippy::expect_used)]`, so a new panic on an
//! untrusted address needs a reviewed, reasoned `#[allow]`.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use crate::concretize::{AddressConcretizer, ConcretizationResult, strided_addrs};
use crate::symbolic::{
    RustBV, SymContext, record_concretize_disjunction, record_mem_ite_depth,
    record_mem_lazy_page_fault, record_mem_store, record_mem_store_symbolic_addr,
};
use crate::vex::Endness;

use super::multi::MultiAlternative;
use super::page::{MemoryPage, PAGE_SIZE, Permission};
use super::{Address, MemoryError, SymbolicMemory, end_page_exclusive, end_page_inclusive};

impl SymbolicMemory {
    /// Store a value to memory.
    pub fn store(
        &mut self,
        addr: &RustBV,
        value: RustBV,
        ctx: &SymContext,
    ) -> Result<(), MemoryError> {
        // angr-9ke6b.229: the symbolic-addr subcounter bump used to live here,
        // where it was dead — this wrapper has no production caller (`.228`).
        // It now lives on the four concretizer entry points production uses:
        // `store_symbolic`, `store_symbolic_unified`,
        // `store_symbolic_unified_multi` and `store_with_concretization`.
        //
        // For symbolic addresses, we need to concretize
        let concrete_addr = match addr.as_u64() {
            Some(a) => a,
            None => match ctx.eval(addr) {
                Some(a) => {
                    // Pin the arbitrarily-chosen store address on the path so the
                    // store is not invisible to a later read that concretizes
                    // elsewhere (angr-mv08h, parity with Python's store pin).
                    crate::concretize::pin_fallback_addr(ctx, addr, a as u64);
                    a as u64
                }
                None => {
                    return Err(MemoryError::SymbolicAddress {
                        description: "could not resolve address for store".to_string(),
                    });
                }
            },
        };

        // angr-9ke6b.228: no lazy-page-fault bump here. `store_concrete` has no
        // lazy classification at all (it only ever yields `Unmapped`), so the
        // branch that used to live here was dead. The counter is bumped by the
        // producers instead — `check_pages_mapped_lazy` and
        // `install_multi_for_candidates_safe` on this side.
        self.store_concrete(Address(concrete_addr), value)
    }

    /// Store to a concrete address.
    ///
    /// angr-9ke6b.99: a sub-byte-width `value` makes `size` zero, which
    /// `end_page_inclusive` rejects as [`MemoryError::ZeroSize`] before any
    /// page is touched — so a rejected zero-size store installs nothing, the
    /// same all-or-nothing property the permission check below has.
    ///
    /// angr-0jh0j.35: the store side needs no upper-bound sibling of that
    /// guard (the load side's `check_access_size`). `size` is not caller-
    /// supplied here — it is `value.width() / 8`, so `size * 8 <= u32::MAX` by
    /// construction and none of the width arithmetic behind this entry point
    /// can wrap. Keep it derived that way; a `store_*` variant that took a
    /// separate `size` argument would need the explicit check.
    ///
    /// # Errors
    /// [`MemoryError::ZeroSize`] for any `value` narrower than one byte (both
    /// branches), and — for a *symbolic* `value` — [`MemoryError::UnalignedWidth`]
    /// when the width is a non-multiple of 8 wide enough to survive that first
    /// guard (angr-03vl4.40); see the guard in the symbolic branch below for
    /// why partial trailing bytes cannot be tracked. Also
    /// [`MemoryError::Unmapped`] / [`MemoryError::Permission`] per the range
    /// checks, all before any page is written.
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
        let end_page = end_page_inclusive(addr.raw(), size as u64)?;

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
            // angr-03vl4.40: same guard `import_symbolic_value` carries, for
            // the same reason. Every structure below is sized by `size`
            // (= `width() / 8`, truncating), so a width like 12 bits tracks
            // only 1 of its 1.5 logical bytes in `symbolic_spans` and the page
            // bitmap; the pre-existing `ZeroSize` guard in `end_page_inclusive`
            // only accidentally catches widths 1-7. Reject rather than store an
            // object `extract_byte_lane` can never reconstruct.
            let width_bits = value.width();
            if width_bits == 0 || !width_bits.is_multiple_of(8) {
                return Err(MemoryError::UnalignedWidth {
                    addr: addr.raw(),
                    width_bits,
                });
            }
            // angr-9ke6b.95: mirror the concrete branch's angr-1tes cleanup
            // below. A byte may be marked Multi *or* Symbolic but not both
            // (invariant documented on `SymbolicMemory`), and
            // `load_concrete_lazy_inner` dispatches to Multi cells *before*
            // consulting `symbolic_objects` — so leaving a pre-existing Multi
            // cell in place would make every later load return the stale
            // alternatives and silently drop the value being stored here.
            // Must run before the page-cloning loop below, which would
            // otherwise flush a stale `multi_bitmap` back over the clear.
            if !self.multi_objects.is_empty() {
                for i in 0..size {
                    // overflow-ok: `Add<u64> for Address` is `wrapping_add` — a
                    // store straddling the top wraps like the guest's own math.
                    self.clear_multi_at(addr + i as u64);
                }
            }
            let old_sym_bytes = self
                .symbolic_objects
                .insert(addr, value.clone())
                .map_or(0, |old| old.width() / 8);
            // Update reverse span index: map each byte offset to (base_addr, width)
            let sym_bytes = width_bits / 8;
            for i in 1..sym_bytes {
                // overflow-ok: `Address` arithmetic is wrapping (see above).
                self.symbolic_spans
                    .insert(addr + i as u64, (addr, width_bits));
            }
            // angr-9ke6b.98: a narrower symbolic value overwriting a wider one
            // at the same base abandons the tail `[addr+size, addr+old_bytes)`.
            // The loop above only refreshed spans inside the new width, so the
            // tail's `symbolic_spans` entries still name `(addr, old_width)` —
            // a base whose live object no longer reaches them. `load_concrete`
            // would follow such a span into an out-of-range extract and report
            // `SymbolicAddress { "symbolic bytes not fully tracked" }`, turning
            // a readable byte into a hard error. Retire the tail the same way
            // the concrete branch's angr-7qon cleanup does: drop the spans and
            // clear the bitmap so those bytes reclassify as concrete. Must run
            // before the page-cloning loop below, which would otherwise flush a
            // stale `symbolic_bitmap` back over the clear.
            if old_sym_bytes > size {
                for i in size..old_sym_bytes {
                    // overflow-ok: `Address` arithmetic is wrapping (see above).
                    self.symbolic_spans.remove(&(addr + i as u64));
                }
                // overflow-ok: `Address` arithmetic is wrapping (see above); the
                // `old_sym_bytes > size` guard keeps the range non-empty.
                self.clear_symbolic_bitmap_range(addr + size as u64, addr + old_sym_bytes as u64);
            }
            // Mark pages as having symbolic bytes — batch per-page
            let mut current_page_num = u64::MAX;
            let mut current_page: Option<MemoryPage> = None;
            for i in 0..size {
                // overflow-ok: `Address` arithmetic is wrapping (see above).
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

        // Concrete store. Reached only on the concrete path (symbolic values
        // early-return above); as_u128 + error keeps a bypassed guard from
        // aborting the process.
        let concrete_val = value
            .as_u128()
            .ok_or(MemoryError::UnexpectedSymbolic { addr: addr.0 })?;

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
            // overflow-ok: `Address::page_offset` masks with `PAGE_MASK`, so
            // `page_offset < PAGE_SIZE` and the difference is in `1..=PAGE_SIZE`.
            let bytes_in_page = ((PAGE_SIZE - page_offset as u64) as usize).min(remaining.len());

            if let Some(page) = self.pages.get_mut(&page_num) {
                page.store_concrete(page_offset, &remaining[..bytes_in_page]);
                // Mark page as dirty
                self.dirty_pages.insert(page_num);
            }

            remaining = &remaining[bytes_in_page..];
            // overflow-ok: `Address` arithmetic is wrapping; a store straddling
            // the top of the address space walks into page 0 like the guest.
            current_addr = current_addr + bytes_in_page as u64;
        }

        // Clear any symbolic object at this address and its span entries
        if let Some(old_sym) = self.symbolic_objects.remove(&addr) {
            let old_bytes = old_sym.width() / 8;
            for i in 1..old_bytes {
                // overflow-ok: `Address` arithmetic is wrapping (see above).
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
                // overflow-ok: `Address` arithmetic is wrapping (see above); the
                // `old_bytes > size` guard keeps the range non-empty.
                self.clear_symbolic_bitmap_range(addr + size as u64, addr + old_bytes as u64);
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
                // overflow-ok: `Address` arithmetic is wrapping (see above).
                let byte_addr = addr + i as u64;
                if self.multi_objects.remove(&byte_addr).is_some() {
                    self.bump_multi_version(byte_addr);
                }
            }
        }

        Ok(())
    }

    /// Clear the page `symbolic_bitmap` bits covering `[start, end)`.
    ///
    /// Shared by both `store_concrete` branches to retire the tail bytes a
    /// narrower store abandons when it overwrites a wider symbolic object at
    /// the same base (angr-7qon for the concrete branch, angr-9ke6b.98 for the
    /// symbolic one). Caller is responsible for dropping the matching
    /// `symbolic_spans` entries; this only touches the per-page bitmap and
    /// walks page boundaries so a range spanning pages is fully cleared.
    /// Unmapped pages in the range are skipped — there is no bitmap to clear.
    fn clear_symbolic_bitmap_range(&mut self, start: Address, end: Address) {
        let mut cur = start;
        while cur < end {
            let page_num = cur.page_num();
            let page_offset = cur.page_offset();
            // overflow-ok: `page_offset()` masks, so it is `< PAGE_SIZE`.
            let to_page_end = (PAGE_SIZE - page_offset as u64) as usize;
            // overflow-ok: the loop condition holds `cur < end`.
            let bytes_in_page = to_page_end.min((end - cur) as usize);
            if let Some(page) = self.pages.get_mut(&page_num) {
                page.clear_symbolic(page_offset, bytes_in_page as u16);
                self.dirty_pages.insert(page_num);
            }
            cur = cur + bytes_in_page as u64;
        }
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
        // Past the fast path the address is genuinely symbolic (angr-9ke6b.229).
        record_mem_store_symbolic_addr();

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
                self.store_eager_ite_candidates(&addr, &value, &addrs, ctx)?;
                // angr-62li: hoist the addr-domain disjunction.
                Self::assert_address_disjunction(&addr, &addrs, ctx);
                Ok(())
            }
            other => Err(other.as_symbolic_address_error()),
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
        let addrs = strided_addrs(base, stride, count);
        self.store_eager_ite_candidates(addr_expr, value, &addrs, ctx)
    }

    /// Shared eager-ITE conditional-store loop (angr-24pv4.3) used by the
    /// `Multiple` arm of `store_symbolic` and by `store_strided`. For each
    /// `candidate` builds `mem[candidate] = If(addr_expr == candidate, value,
    /// mem[candidate])` and records the eager ITE chain depth (Phase 0
    /// instrumentation, per memory `invariant-mem-ite-depth-counter`). The
    /// caller is responsible for any `assert_address_disjunction` (only the
    /// `Multiple` path hoists it; strided does not).
    pub(super) fn store_eager_ite_candidates(
        &mut self,
        addr_expr: &RustBV,
        value: &RustBV,
        addrs: &[u64],
        ctx: &SymContext,
    ) -> Result<(), MemoryError> {
        let size = value.width() / 8;
        for &candidate in addrs {
            let addr_const = RustBV::concrete(candidate as u128, addr_expr.width());
            let cond = addr_expr.eq(&addr_const, ctx);
            let current = self.load_concrete_lazy(Address(candidate), size, ctx)?;
            let conditional_value = cond.ite(value, &current, ctx);
            self.store_concrete_lazy(candidate, conditional_value)?;
        }
        record_mem_ite_depth(addrs.len() as u32);
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
            self.store_concrete_lazy(concrete_addr, value)?;
            return Ok(Some(ConcretizationResult::Single(concrete_addr)));
        }
        // Past the fast path the address is genuinely symbolic (angr-9ke6b.229).
        record_mem_store_symbolic_addr();

        // AVOID_MULTIVALUED_WRITES: silently drop the store.
        // SILENT(cat-b): under the AVOID_MULTIVALUED_WRITES option the caller
        // deliberately degrades a multi-valued symbolic-address write to a
        // no-op (matches the Python engine's opt); the store is lost but this
        // is the requested, bounded behavior, not a wrong-answer surprise.
        if concretizer.should_avoid_multivalued_write(&addr) {
            return Ok(None);
        }

        // Try to concretize the address (write mode: falls back to Max solution)
        let result = concretizer.concretize_write(&addr, ctx);
        match &result {
            ConcretizationResult::Single(concrete_addr) => {
                self.store_concrete_lazy(*concrete_addr, value)?;
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
                let addrs = strided_addrs(base, stride, count);
                self.install_multi_for_candidates_safe(&addr, &value, &addrs, ctx)?;
                Ok(Some(result))
            }
            // Return error so caller can fall back to Python's memory model,
            // which handles large symbolic address ranges natively.
            other => other
                .to_symbolic_address_error()
                .map_or(Ok(Some(result)), Err),
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
        // angr-9ke6b.229: this is the interpreter's store path
        // (`try_rust_memory_store`), which concretizes one step earlier and
        // hands the result in — so the symbolic-vs-concrete split has to be
        // read off `addr` rather than off an early-return fast path.
        if addr.as_u64().is_none() {
            record_mem_store_symbolic_addr();
        }
        match conc_result {
            ConcretizationResult::Single(concrete_addr) => {
                self.store_concrete_lazy(*concrete_addr, value)
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
                let addrs = strided_addrs(*base, *stride, *count);
                self.install_multi_for_candidates_safe(addr, &value, &addrs, ctx)
            }
            // Return error so caller can fall back to Python's memory model,
            // which handles large symbolic address ranges natively.
            other => other.to_symbolic_address_error().map_or(Ok(()), Err),
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
    ///
    /// Returns `MemoryError::Permission` (angr-9ke6b.94) when
    /// `enforce_permissions` is on and any already-mapped candidate page is
    /// not writable, matching `store_concrete`. The check precedes every
    /// mutation, so a rejected store installs nothing at all.
    pub(super) fn install_multi_for_candidates(
        &mut self,
        addr_expr: &RustBV,
        value: &RustBV,
        addrs: &[u64],
        ctx: &SymContext,
    ) -> Result<(), MemoryError> {
        let size = value.width() / 8;
        let endness = self.endness;

        // angr-9ke6b.94: enforce W permission before mutating anything, so a
        // Multiple/Strided-concretized store is rejected on a read-only page
        // exactly like the Single-concretized one `store_concrete` handles.
        // Must run before the auto-map loop below: that loop maps missing
        // pages as RW, which would make the check vacuously pass for them.
        // `check_perms_range` skips pages that aren't mapped yet, so only
        // pre-existing pages are consulted — matching `store_concrete`, which
        // only ever sees already-mapped pages.
        // The page range is loop-invariant across the two passes, so compute it
        // once here and reuse it below (angr-0h6hm). Building the Vec inside
        // this loop rather than in a separate pre-pass keeps the original
        // interleaving of `end_page_inclusive` and permission errors.
        let mut page_ranges: Vec<(u64, u64)> = Vec::with_capacity(addrs.len());
        for &cand in addrs {
            let start_page = cand >> 12;
            let end_page = end_page_inclusive(cand, size as u64)?;
            self.check_perms_range(start_page, end_page, Permission::W)?;
            page_ranges.push((start_page, end_page));
        }

        // Auto-map all candidate byte addresses before installing — matches
        // what set_multi_alternatives does per-byte, but we batch it so the
        // permission check happens up-front for every page.
        for (start_page, end_page) in page_ranges {
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
                // angr-03vl4.37: `cand` is a Z3 solution for a guest-supplied
                // store pointer, so this addition is on untrusted shape. A
                // bare `+` panics under debug-assertions and wraps in release,
                // which would plant the Multi cell on a low page — the exact
                // silent redirection `end_page_inclusive` exists to stop.
                // Unreachable in practice (`b < size`, and the page-range pass
                // above already rejected every candidate whose last byte
                // overflows), so this is defense-in-depth of the module's panic
                // policy in the same shape `MemoryError::UnexpectedSymbolic`
                // takes — it must not silently self-heal if that pass is ever
                // changed to saturate instead of erroring.
                let byte_addr = cand.checked_add(b as u64).ok_or(MemoryError::OutOfBounds {
                    addr: cand,
                    size: size as u64,
                })?;
                let byte_value = Self::extract_byte_lane(value, b, endness, ctx).ok_or(
                    MemoryError::SymbolicAddress {
                        description: "value byte offset out of range".to_string(),
                    },
                )?;
                let alt = MultiAlternative::new(cond.clone(), byte_value);

                // Merge with any existing payload at this byte. New alt
                // goes after existing ones; conds are disjoint so order
                // is semantically irrelevant.
                //
                // angr-c7xno.54: take the payload *out* of the map and
                // `MultiPayload::push` onto it rather than cloning its
                // alternatives into a fresh Vec — `push` is the append
                // primitive its doc comment advertises, and moving avoids
                // an allocation per byte. `set_multi_alternatives` puts the
                // payload straight back (and keeps the ITE-depth counter
                // contract); `push` clears the collapse cache the moved
                // payload carried.
                let mut payload = self
                    .multi_objects
                    .remove(&Address(byte_addr))
                    .unwrap_or_default();
                payload.push(alt);
                self.set_multi_alternatives(byte_addr, payload);
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
            let end_page = end_page_inclusive(cand, size as u64)?;
            let mut all_mapped = true;
            for page_num in start_page..=end_page {
                if !self.pages.contains_key(&page_num) {
                    if self.is_in_lazy_region(page_num) {
                        // Second store-side producer of the counter
                        // (angr-9ke6b.228); see `check_pages_mapped_lazy`.
                        record_mem_lazy_page_fault();
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
            // SILENT(cat-b): the whole symbolic-address store is dropped, but
            // every candidate was already filtered as unreachable by the loop
            // above (non-lazy unmapped page => the address could not validly
            // resolve there), so there is no reachable cell left to write.
            // This mirrors eager `prepare_addresses_for_ite`, which skips the
            // same candidates and likewise ends up storing nothing; erroring
            // instead would diverge from the eager path.
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
    ///
    /// `#[cfg(test)] pub(crate)` (angr-c7xno.55): every call site lives in
    /// `memory/tests/multi.rs`, and no pyo3/manager binding exposes it, so a
    /// crate-external `pub` would overstate the supported API surface. The
    /// `cfg(test)` gate is what `pub(crate)` alone cannot express — the test
    /// submodules are themselves `cfg(test)` (see `test_submod!` in `lib.rs`),
    /// so a non-test build sees no caller at all and `dead_code` fires.
    #[cfg(test)]
    pub(crate) fn store_concrete_multi(
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
            self.store_concrete_lazy(concrete_addr, value)?;
            return Ok(Some(ConcretizationResult::Single(concrete_addr)));
        }
        // Past the fast path the address is genuinely symbolic (angr-9ke6b.229).
        record_mem_store_symbolic_addr();

        // AVOID_MULTIVALUED_WRITES: silently drop the store.
        // SILENT(cat-b): under the AVOID_MULTIVALUED_WRITES option the caller
        // deliberately degrades a multi-valued symbolic-address write to a
        // no-op (matches the Python engine's opt); the store is lost but this
        // is the requested, bounded behavior, not a wrong-answer surprise.
        if concretizer.should_avoid_multivalued_write(&addr) {
            return Ok(None);
        }

        // This entry point is only reached for addresses that carry a
        // `MultiwriteAnnotation` on the Python side, so it passes Python's
        // `_multiwrite_filter` and keeps the Range strategy even when
        // SYMBOLIC_WRITE_ADDRESSES is off (angr-9ke6b.194).
        let result = concretizer.concretize_write_multiwrite(&addr, ctx);
        match &result {
            ConcretizationResult::Single(concrete_addr) => {
                self.store_concrete_lazy(*concrete_addr, value)?;
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
                let addrs_v = strided_addrs(base, stride, count);
                self.install_multi_for_candidates(&addr, &value, &addrs_v, ctx)?;
                Ok(Some(result))
            }
            other => other
                .to_symbolic_address_error()
                .map_or(Ok(Some(result)), Err),
        }
    }

    /// Store to a concrete address, returning `UnmappedPageInRegion` for lazy
    /// regions.
    ///
    /// This path deliberately does **not** auto-map missing pages: a
    /// speculative zero page diverges from Python's actual backer data, so the
    /// error is propagated instead and the caller falls back to a Python
    /// callback that stores correctly. For internal Rust operations that touch
    /// no Python state, use `store_concrete_automap_internal`.
    pub fn store_concrete_lazy(
        &mut self,
        addr: impl Into<Address>,
        value: RustBV,
    ) -> Result<(), MemoryError> {
        let addr = addr.into();
        let size = value.width() / 8;

        // Check if pages are mapped
        let start_page = addr.page_num();
        let end_page = end_page_exclusive(addr.raw(), size as u64)?;
        self.check_pages_mapped_lazy(start_page, end_page)?;

        // Permission checks live in store_concrete; this wrapper only adds
        // lazy-region detection for unmapped pages.
        self.store_concrete(addr, value)
    }

    /// Verify every page in `[start_page, end_page)` is mapped, classifying
    /// the first unmapped page as `UnmappedPageInRegion` (in a lazy region,
    /// fetchable on-demand) or `Unmapped` (angr-24pv4.3). Shared by the
    /// lazy-aware store wrapper `store_concrete_lazy`. NOT used by
    /// `store_concrete` (no lazy
    /// detection, inclusive range) or `store_concrete_automap_internal`
    /// (auto-maps lazy pages instead of erroring) — their semantics differ.
    ///
    /// Bumps `record_mem_lazy_page_fault` on the lazy classification
    /// (angr-9ke6b.228) — this is one of the counter's two store-side
    /// producers, the other being `install_multi_for_candidates_safe`.
    fn check_pages_mapped_lazy(&self, start_page: u64, end_page: u64) -> Result<(), MemoryError> {
        for page_num in start_page..end_page {
            if !self.pages.contains_key(&page_num) {
                let page_addr = page_num << 12;
                return Err(if self.is_in_lazy_region(page_num) {
                    record_mem_lazy_page_fault();
                    MemoryError::UnmappedPageInRegion { page_addr }
                } else {
                    MemoryError::Unmapped {
                        addr: page_addr,
                        size: PAGE_SIZE,
                    }
                });
            }
        }
        Ok(())
    }

    /// Store to a concrete address with internal auto-mapping.
    ///
    /// This is for internal Rust operations that don't involve Python state.
    /// For interpreter callbacks, use `store_concrete_lazy` which propagates
    /// errors so Python can handle the store correctly.
    pub fn store_concrete_automap_internal(
        &mut self,
        addr: impl Into<Address>,
        value: RustBV,
    ) -> Result<(), MemoryError> {
        let addr = addr.into();
        let size = value.width() / 8;
        let start_page = addr.page_num();
        let end_page = end_page_exclusive(addr.raw(), size as u64)?;

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
                // overflow-ok: `Address` arithmetic is wrapping; `off < total`
                // and `cs = (total - off).min(16)`, so `total - off - cs >= 0`.
                Endness::Little => addr + off as u64,
                // overflow-ok: same bound as the Little arm above — the
                // scan reads only the two lines above the flagged one.
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
