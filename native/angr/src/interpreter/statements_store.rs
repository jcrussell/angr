//! `IRStmt::Store` execution: the store path split out of `statements.rs`.
//!
//! `store_value` picks the lane by how concrete the address and value are —
//! `handle_concrete_store` (with `buffer_store_for_rust_memory` deferring the
//! commit to the block-end pending-store flush), `handle_symbolic_store`, or
//! `dispatch_multi_store` for a concretized address set — with
//! `try_rust_memory_store` taking the native path and `fallback_to_python_store`
//! the callback one. The rest is the bookkeeping a store must not skip:
//! invalidating cached IRSBs when the written bytes are code
//! (`invalidate_code_on_store` and friends, see `code_invalidation.rs`) and
//! evicting concrete/symbolic cache entries the store now contradicts.

use super::bv_utils::{bv_to_bytes, reject_symbolic_byte_store};
use super::concretize_cache::GuardedWrite;
use super::pending_store::overlap_offsets;
use super::*;
use crate::symbolic::{MAX_CONCRETE_CHUNK, u128_to_le_bytes};

/// How a guarded store's concrete target address was obtained. Selects the
/// flush/write lane in `VEXInterpreter::store_guarded_ite` — the only thing
/// that still differs between `handle_storeg`'s two symbolic-guard branches.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum GuardedStoreAddr {
    /// The `addr` expression evaluated to a literal, so a concrete ITE result
    /// can join the buffered `pending_stores` fast path.
    Literal,
    /// The `addr` expression was symbolic and `concretize_cached_write`
    /// returned a single solution. The buffer is drained unconditionally and a
    /// concrete ITE result goes straight to the Python store callback rather
    /// than being buffered behind it.
    Concretized,
}

impl GuardedStoreAddr {
    /// Context label for `reject_symbolic_byte_store`'s error message.
    fn reject_label(self) -> &'static str {
        match self {
            Self::Literal => "StoreG",
            Self::Concretized => "StoreG (concretized addr)",
        }
    }
}

impl<'a> VEXInterpreter<'a> {
    /// Attempt to store via Rust-native memory. Returns Ok(true) if the store
    /// was handled, Ok(false) if the caller should fall back to the Python path.
    pub(super) fn try_rust_memory_store(
        &mut self,
        callbacks: &dyn EngineHost,
        addr_val: &RustBV,
        data_val: &RustBV,
        data_size: usize,
        store_start: Option<Instant>,
    ) -> Result<bool, CbExecutionError> {
        // AVOID_MULTIVALUED_WRITES: silently drop symbolic-addr stores.
        // Mirrors `address_concretization_mixin.py:327-329`.
        if self.concretizer.should_avoid_multivalued_write(addr_val) {
            profile_add!(store_start, self.stats.store_stmt_time_ns);
            return Ok(true);
        }
        // angr-vfst: address_concretization BP_BEFORE for the store path. Only
        // dispatches when addr is symbolic (concrete-addr stores have nothing
        // to concretize). Gated on bit 17 inside the helper.
        if !addr_val.is_concrete() {
            self.dispatch_address_concretization_inspect(
                callbacks, addr_val, "store", "before", None,
            );
        }
        // Concretize for write with per-block cache (avoids redundant Z3 calls)
        let conc_result = self.concretize_cached_write(addr_val);
        if !addr_val.is_concrete() {
            let result_addrs = conc_result.addresses();
            self.dispatch_address_concretization_inspect(
                callbacks,
                addr_val,
                "store",
                "after",
                result_addrs,
            );
        }

        // Attempt store using pre-computed concretization
        let first_result = match self.rust_memory.as_mut() {
            Some(rust_mem) => rust_mem.store_with_concretization(
                addr_val,
                data_val.clone(),
                &conc_result,
                self.ctx,
            ),
            None => return Ok(false),
        };

        match first_result {
            Ok(()) => {
                self.invalidate_code_on_store(&conc_result, data_size);
                // Rust owns memory — no need to sync stores to Python.
                profile_add!(store_start, self.stats.store_stmt_time_ns);
                Ok(true)
            }
            Err(MemoryError::UnmappedPageInRegion { page_addr }) => {
                // Page is in a lazy region - fetch it (rust_mem borrow is dropped here)
                let prefetch_count = self.page_prefetch_count;
                let page_fetched =
                    self.fetch_page_with_prefetch(callbacks, page_addr, prefetch_count)?;

                if page_fetched {
                    // Page was fetched - retry store using cached concretization
                    if let Some(ref mut rust_mem) = self.rust_memory {
                        match rust_mem.store_with_concretization(
                            addr_val,
                            data_val.clone(),
                            &conc_result,
                            self.ctx,
                        ) {
                            Ok(()) => {
                                self.invalidate_code_on_store(&conc_result, data_size);
                                return Ok(true);
                            }
                            Err(_e) => {
                                // Still failed - fall through to Python callback
                            }
                        }
                    }
                } else if let ConcretizationResult::Single(store_addr) = *conc_result
                    && self.map_fresh_pages_for_store(callbacks, store_addr, data_size)
                    && let Some(ref mut rust_mem) = self.rust_memory
                    && rust_mem
                        .store_with_concretization(
                            addr_val,
                            data_val.clone(),
                            &conc_result,
                            self.ctx,
                        )
                        .is_ok()
                {
                    // The host has no page here either: map a fresh one, as
                    // the `Unmapped` arm below does, rather than park the
                    // store in `pending_stores`, where a load of a neighbour
                    // byte mints a filler the flush's zero page never sees
                    // (angr-vmi2l.325).
                    self.invalidate_code_on_store(&conc_result, data_size);
                    profile_add!(store_start, self.stats.store_stmt_time_ns);
                    return Ok(true);
                }
                Ok(false)
            }
            Err(MemoryError::Unmapped {
                addr,
                size: unmapped_size,
            }) => {
                if let ConcretizationResult::Single(store_addr) = *conc_result
                    && self.fetch_host_pages_for_store(callbacks, store_addr, data_size)?
                    && self.map_fresh_pages_for_store(callbacks, store_addr, data_size)
                    && let Some(ref mut rust_mem) = self.rust_memory
                    && rust_mem
                        .store_with_concretization(
                            addr_val,
                            data_val.clone(),
                            &conc_result,
                            self.ctx,
                        )
                        .is_ok()
                {
                    self.invalidate_code_on_store(&conc_result, data_size);
                    profile_add!(store_start, self.stats.store_stmt_time_ns);
                    return Ok(true);
                }
                log::debug!(
                    "Unmapped memory store at 0x{addr:x} (size={unmapped_size}), falling back to Python"
                );
                Ok(false)
            }
            Err(MemoryError::SymbolicAddress { description }) => {
                // Address range too large or symbolic — Python's memory model handles natively
                log::debug!("Symbolic address store: {description}, falling back to Python");
                Ok(false)
            }
            Err(e) => Err(CbExecutionError::Memory(e)),
        }
    }

    /// Map zero pages under a concrete store that hit pages neither side holds,
    /// so the write lands in `rust_memory` (angr-sy3xj). Without this the
    /// store is buffered in `pending_stores`, which a multi-block run's later
    /// loads read back, but the flush drops it: the next step and every export
    /// see the page as never written.
    ///
    /// angr's `DefaultMemory` maps a page on any write when STRICT_PAGE_ACCESS
    /// is off; the proxy-write twin is
    /// `state_api::_set_state_memory_concrete_automap` (angr-ijwp0). A page the
    /// host may hold (`has_page`, which fails open without a snapshot) is left
    /// alone, since a zero page would shadow its real bytes. Returns whether
    /// every page under the store is now mapped.
    pub(super) fn map_fresh_pages_for_store(
        &mut self,
        callbacks: &dyn EngineHost,
        addr: u64,
        size: usize,
    ) -> bool {
        let Some(rust_mem) = self.rust_memory.as_mut() else {
            return false;
        };
        if rust_mem.enforce_permissions() {
            return false;
        }
        let pages = crate::memory::PageIndex::range_covering(addr, size as u64);
        if pages.clone().any(|page| {
            !rust_mem.is_mapped(page.base_addr()) && callbacks.has_page(page.base_addr())
        }) {
            return false;
        }
        rust_mem.add_lazy_region(addr, size as u64);
        for page in pages {
            rust_mem.auto_map_zero_page(page.base_addr());
        }
        true
    }

    /// Fetch each page under a concrete store outside every lazy region that
    /// Rust lacks and the host may hold, so `map_fresh_pages_for_store` maps
    /// zero pages only where the host has none (angr-0mo14). Without a
    /// page-universe snapshot (a ZERO_FILL_UNCONSTRAINED_MEMORY seed)
    /// `has_page` fails open on every page, so the store used to go to the
    /// callback ladder and the flush dropped it as `Unmapped`. Under zero-fill
    /// the host serves a never-touched page as zeros, and a page it holds with
    /// its own bytes. Returns `false` when the host declines a page, leaving
    /// the store on the ladder as before.
    fn fetch_host_pages_for_store(
        &mut self,
        callbacks: &dyn EngineHost,
        addr: u64,
        size: usize,
    ) -> Result<bool, CbExecutionError> {
        for page in crate::memory::PageIndex::range_covering(addr, size as u64) {
            let base = page.base_addr();
            let Some(rust_mem) = self.rust_memory.as_ref() else {
                return Ok(false);
            };
            if rust_mem.is_mapped(base) || !callbacks.has_page(base) {
                continue;
            }
            if !self.fetch_page(callbacks, base)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Invalidate cached IRSBs after a successful Rust-native store when the
    /// store hits a loaded binary region (self-modifying code support).
    /// Single-address writes invalidate the overlapping block; other
    /// concretization shapes defer to `invalidate_code_for_concretization`.
    pub(super) fn invalidate_code_on_store(
        &mut self,
        conc_result: &ConcretizationResult,
        data_size: usize,
    ) {
        match conc_result {
            ConcretizationResult::Single(addr_concrete) => {
                if self.is_in_binary(*addr_concrete) {
                    self.invalidate_code_at(*addr_concrete, data_size);
                }
            }
            _ => {
                self.invalidate_code_for_concretization(conc_result, data_size);
            }
        }
    }

    /// Invalidate cached IRSBs for a multi-address concretization result whose
    /// solutions hit loaded binary regions. Conservative: any in-binary
    /// solution triggers a per-address invalidation; ranges/Any clear the
    /// cache outright since the affected bytes are unbounded.
    pub(super) fn invalidate_code_for_concretization(
        &mut self,
        conc_result: &ConcretizationResult,
        data_size: usize,
    ) {
        match conc_result {
            ConcretizationResult::Single(_) => {} // handled by caller
            ConcretizationResult::Multiple(addrs) => {
                for &addr in addrs.iter() {
                    if self.is_in_binary(addr) {
                        self.invalidate_code_at(addr, data_size);
                    }
                }
            }
            ConcretizationResult::Strided {
                base,
                stride,
                count,
            } => {
                // `strided_addrs` (wrapping), not the saturating arithmetic this
                // used to open-code: the addresses invalidated here must be the
                // same ones `handle_symbolic_store`'s `Strided` arm goes on to
                // write, or a wrapped store escapes code-cache invalidation.
                for addr in strided_addrs(*base, *stride, *count) {
                    if self.is_in_binary(addr) {
                        self.invalidate_code_at(addr, data_size);
                    }
                }
            }
            ConcretizationResult::TooLarge { min, max, .. } => {
                // Range too large to enumerate. If the [min, max] range
                // intersects any loaded binary region, drop the entire
                // block cache to be safe.
                let intersects_binary = self.concrete_memory.iter().any(|region| {
                    // overflow-ok: `saturating_add` — a region abutting the top
                    // of the address space clamps to `u64::MAX`, which only
                    // widens this intersection test (more cache invalidation,
                    // never less).
                    let region_end = region.base.saturating_add(region.size);
                    *min < region_end && *max >= region.base
                });
                if intersects_binary {
                    self.block_cache.clear();
                    // Mark all binary pages as dirtied so native lift skips
                    // them until they're re-lifted via Python.
                    let pages: Vec<crate::memory::PageIndex> = self
                        .concrete_memory
                        .iter()
                        .flat_map(|region| {
                            crate::memory::PageIndex::range_covering(region.base, region.size)
                        })
                        .collect();
                    self.dirtied_code_pages.extend(pages);
                }
            }
            ConcretizationResult::Failed(_) => {
                // Concretization failed; addresses are unknown. Be safe.
                self.block_cache.clear();
            }
        }
    }

    /// Fall back to the Python callback path for a store. Splits on
    /// concrete-vs-symbolic address; the concrete branch invalidates cached
    /// code then dispatches via `handle_concrete_store`, the symbolic
    /// branch goes through `handle_symbolic_store`.
    pub(super) fn fallback_to_python_store(
        &mut self,
        callbacks: &dyn EngineHost,
        addr_val: &RustBV,
        data_val: RustBV,
        data_size: usize,
    ) -> Result<(), CbExecutionError> {
        if let Some(addr_concrete) = addr_val.as_u64() {
            self.invalidate_code_at_store(addr_concrete, data_size);
            self.handle_concrete_store(callbacks, addr_concrete, data_val, data_size)
        } else {
            self.handle_symbolic_store(callbacks, addr_val, &data_val, data_size)
        }
    }

    /// Drop any cached IRSB whose bytes overlap the store (self-modifying
    /// code support). Concrete-address stores only.
    pub(super) fn invalidate_code_at_store(&mut self, addr: u64, data_size: usize) {
        if self.is_in_binary(addr) {
            self.invalidate_code_at(addr, data_size);
        }
    }

    /// Concrete-address soundness prelude shared by manual store dispatchers
    /// (e.g. `handle_storeg`): invalidate any cached IRSB the write overlaps
    /// (self-modifying code) and evict stale overlapping symbolic shadows so a
    /// later overlap load can't return them. Combines the two steps the plain
    /// Store path gets for free via `fallback_to_python_store` +
    /// `handle_concrete_store` (angr-myzjx.26).
    pub(super) fn invalidate_and_evict_concrete_store(&mut self, addr: u64, data_size: usize) {
        self.invalidate_code_at_store(addr, data_size);
        self.evict_overlapping_symbolic_stores(addr, data_size);
    }

    /// Canonical store dispatch for the *unconditional* arms of
    /// `handle_storeg` (guard always-true / concrete-true), where a guarded
    /// store is semantically identical to a plain `IRStmt::Store` of
    /// `data_val` at `addr_val`. Mirrors the `IRStmt::Store` handler: try
    /// Rust-native memory first, else the Python fallback. Both paths perform
    /// code-cache invalidation and symbolic-shadow eviction, which the old
    /// inline `handle_storeg` dispatch skipped entirely (angr-myzjx.26).
    pub(super) fn store_value(
        &mut self,
        callbacks: &dyn EngineHost,
        addr_val: &RustBV,
        data_val: RustBV,
        data_size: usize,
    ) -> Result<(), CbExecutionError> {
        if self.use_rust_memory
            && self.try_rust_memory_store(callbacks, addr_val, &data_val, data_size, None)?
        {
            return Ok(());
        }
        record_mem_store(data_size as u64);
        self.fallback_to_python_store(callbacks, addr_val, data_val, data_size)
    }

    /// Conditional-store body shared by both of `handle_storeg`'s
    /// symbolic-guard branches (angr-5mnx3.24): load the current bytes at
    /// `addr_concrete`, build `ITE(guard, data, current)`, and store it. Under
    /// `use_rust_memory` that is `store_value`; otherwise invalidate the code
    /// cache and evict overlapping symbolic shadows the write now
    /// contradicts, then dispatch the ITE result. `kind` carries the only
    /// remaining divergence — see [`GuardedStoreAddr`]. Both branches used to
    /// spell this sequence out independently, so a fix to one silently missed
    /// the other.
    pub(super) fn store_guarded_ite(
        &mut self,
        callbacks: &dyn EngineHost,
        addr_concrete: u64,
        guard_val: &RustBV,
        data_val: &RustBV,
        data_size: usize,
        kind: GuardedStoreAddr,
    ) -> Result<(), CbExecutionError> {
        // `current` comes through the same ladder a plain `Load` reads, so it
        // sees this run's native and buffered stores. Under `use_rust_memory`
        // the ITE then commits through `store_value`, never parked behind a
        // later direct store (angr-vmi2l.168).
        let addr_bv = RustBV::concrete(addr_concrete as u128, self.arch.pointer_size());
        let current = self.load_layered(callbacks, &addr_bv, data_size, None)?;
        if self.use_rust_memory {
            let ite_result = guard_val.ite(data_val, &current);
            return self.store_value(callbacks, &addr_bv, ite_result, data_size);
        }
        let ite_result = guard_val.ite(data_val, &current);
        // The ITE captured `current`; now that we're about to overwrite this
        // range, invalidate stale cached code and evict overlapping symbolic
        // shadows (angr-myzjx.26).
        self.invalidate_and_evict_concrete_store(addr_concrete, data_size);

        // A symbolic dispatch needs the pending buffer drained so Python cannot
        // observe a write ordered after this one; the concretized lane drains it
        // unconditionally because its concrete write also bypasses the buffer.
        let use_sym_cb = ite_result.is_symbolic() && callbacks.has_memory_store_symbolic_value();
        if use_sym_cb || kind == GuardedStoreAddr::Concretized {
            self.flush_stores(callbacks)?;
        }
        if use_sym_cb {
            return self.store_symbolic_value_buffered(callbacks, addr_concrete, &ite_result);
        }
        reject_symbolic_byte_store(&ite_result, addr_concrete, kind.reject_label())?;
        let ite_bytes = bv_to_bytes(&ite_result);
        match kind {
            GuardedStoreAddr::Literal => {
                self.pending_stores.push(addr_concrete, ite_bytes);
                if self.pending_stores.len() >= self.max_pending_stores {
                    self.flush_stores(callbacks)?;
                }
                Ok(())
            }
            GuardedStoreAddr::Concretized => callbacks
                .memory_store(addr_concrete, &ite_bytes)
                .map_err(|e| CbExecutionError::Callback(e.to_string())),
        }
    }

    /// Store `data_val` at the symbolic `addr_val` only where the symbolic
    /// `guard_val` holds: the shared tail of `handle_storeg`'s symbolic-guard
    /// arm and of a CAS whose comparison is undecided.
    ///
    /// The address is concretized under the guard
    /// (`concretize_guarded_write`), so the guard-false path keeps every
    /// address value. A single target gets `ITE(guard, data, current)` through
    /// [`store_guarded_ite`](Self::store_guarded_ite). A `Multiple` or
    /// `Strided` set within [`MAX_ITE_ADDRS`] gets `ITE(guard && addr == a,
    /// data, mem[a])` at each candidate. Any other shape goes to the host's
    /// full symbolic store with `host_data`, the value the caller would write
    /// there unguarded; `what` names the statement in the no-callback refusal.
    pub(super) fn store_guarded_symbolic_addr(
        &mut self,
        callbacks: &dyn EngineHost,
        addr_val: &RustBV,
        guard_val: &RustBV,
        data_val: &RustBV,
        host_data: &RustBV,
        what: &str,
    ) -> Result<(), CbExecutionError> {
        let data_size = data_val.width().div_ceil(8) as usize;
        match self.concretize_guarded_write(addr_val, guard_val) {
            GuardedWrite::Infeasible => Ok(()),
            // Code-cache invalidation for this arm happens inside
            // `store_guarded_ite` (via `invalidate_and_evict_concrete_store`).
            GuardedWrite::Concretized(ConcretizationResult::Single(addr_concrete)) => self
                .store_guarded_ite(
                    callbacks,
                    addr_concrete,
                    guard_val,
                    data_val,
                    data_size,
                    GuardedStoreAddr::Concretized,
                ),
            GuardedWrite::Concretized(other) => {
                // Invalidate cached code at the concretized target(s) before
                // dispatching (self-modifying-code support).
                self.invalidate_code_for_concretization(&other, data_size);
                self.flush_stores(callbacks)?;
                // A complete candidate set within the ITE cap stores
                // `ITE(guard && addr == a, data, mem[a])` at each candidate,
                // as Python's conditional store does. The host's full store
                // takes no guard (and a proxy-backed host drops it), so only
                // the shapes Rust cannot enumerate go there (angr-vmi2l.323).
                let addrs = match &other {
                    ConcretizationResult::Multiple(addrs) => Some(addrs.clone()),
                    ConcretizationResult::Strided {
                        base,
                        stride,
                        count,
                    } => Some(strided_addrs(*base, *stride, *count)),
                    _ => None,
                };
                if let Some(addrs) = addrs
                    && addrs.len() <= MAX_ITE_ADDRS
                    && callbacks.has_memory_store_symbolic_value()
                {
                    return self.build_ite_store_from_callbacks(
                        callbacks,
                        &addrs,
                        addr_val,
                        data_val,
                        Some(guard_val),
                    );
                }
                if !callbacks.has_memory_store_symbolic_full() {
                    return Err(CbExecutionError::Unsupported(format!(
                        "{what} with symbolic address: no memory_store_symbolic_full callback"
                    )));
                }
                callbacks
                    .memory_store_symbolic_full(addr_val, host_data)
                    .map_err(|e| CbExecutionError::Callback(e.to_string()))
            }
        }
    }

    /// Concrete-address store path: chooses between the
    /// `memory_store_symbolic_value` callback (32-bit non-stack only) and
    /// the buffered `pending_stores` fast path. The 32-bit heuristic exists
    /// to keep flareon2015_5 working while avoiding the false-positive cost
    /// of routing every 64-bit symbolic store through Python.
    pub(super) fn handle_concrete_store(
        &mut self,
        callbacks: &dyn EngineHost,
        addr_concrete: u64,
        data_val: RustBV,
        _data_size: usize,
    ) -> Result<(), CbExecutionError> {
        let use_sym_store = if self.arch.pointer_size() == 32 {
            let is_stack = self.registers.get_sp_value().is_some_and(|sp_val| {
                // Non-wrapping distance check
                let dist = addr_concrete.abs_diff(sp_val);
                dist <= 0x10000
            });
            !is_stack
        } else {
            false // Skip for 64-bit — too expensive
        };

        if data_val.is_symbolic() && callbacks.has_memory_store_symbolic_value() && use_sym_store {
            // Try symbolic store callback (preserves expression tree)
            let sym_ok = (|| -> Result<(), CbExecutionError> {
                self.flush_stores(callbacks)?;
                self.store_symbolic_value_buffered(callbacks, addr_concrete, &data_val)
            })();
            if let Err(sym_err) = sym_ok {
                // Symbolic store callback failed — evaluate to concrete and
                // store directly via Python callback (not pending_stores).
                // pending_stores would pollute all_flushed_stores with zeros
                // since bv_to_bytes returns zeros for symbolic expressions.
                //
                // angr-sqfj8.61: when the *concretization* also fails (unsat /
                // unknown context, or a value wider than the `u128` the eval
                // path and `u128_to_le_bytes` can carry), this used to
                // `unwrap_or(0)` and write concrete 0 — the same zero-fill
                // wrong answer `reject_symbolic_byte_store` exists to prevent,
                // except self-inflicted. Propagate the original callback error
                // instead so the store fails loudly. Ditto the byte-level
                // callback's own error, which used to be dropped on the floor.
                let size_bytes = (data_val.width() / 8) as usize;
                let concrete_val = self
                    .ctx
                    .eval(&data_val)
                    .filter(|_| size_bytes <= MAX_CONCRETE_CHUNK);
                let Some(concrete_val) = concrete_val else {
                    log::warn!(
                        "symbolic store at {addr_concrete:#x} failed ({sym_err}) and the \
                         {}-bit value could not be concretized; failing the store rather \
                         than writing zeros",
                        data_val.width(),
                    );
                    return Err(sym_err);
                };
                let data_bytes = u128_to_le_bytes(concrete_val, size_bytes);
                callbacks
                    .memory_store(addr_concrete, &data_bytes)
                    .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
            }
            Ok(())
        } else if data_val.is_symbolic() {
            // Symbolic fast path: record the value for exact + overlap load
            // forwarding. Do NOT push zero placeholder bytes into
            // pending_stores — bv_to_bytes yields all-zeros for symbolic data
            // (see the comment in store_concrete), which would make an offset
            // load inside the store return concrete 0 (angr-ofyh). The symbolic
            // maps (incl. overlap) are consulted before the concrete buffer on
            // load.
            //
            // Also evict any stale symbolic shadow this store overlaps
            // (mirrors the concrete branch below) — otherwise two
            // overlapping-but-different-address symbolic stores can coexist
            // in the map and a later overlap load would return whichever one
            // hash-iteration visits first instead of the most recent
            // (angr-vvzf5).
            let size_bytes = (data_val.width() / 8) as usize;
            self.evict_overlapping_symbolic_stores(addr_concrete, size_bytes);
            self.pending_symbolic_stores.insert(addr_concrete, data_val);
            if self.pending_symbolic_stores.len() >= self.max_pending_stores {
                self.flush_stores(callbacks)?;
            }
            Ok(())
        } else {
            // Concrete fast path: evict any stale symbolic shadow this store
            // overwrites (angr-ofyh) so a later load can't return the old
            // symbolic value, then buffer the bytes for batch processing.
            let size_bytes = (data_val.width() / 8) as usize;
            self.evict_overlapping_symbolic_stores(addr_concrete, size_bytes);
            let data_bytes = bv_to_bytes(&data_val);
            self.pending_stores.push(addr_concrete, data_bytes);

            if self.pending_stores.len() >= self.max_pending_stores {
                self.flush_stores(callbacks)?;
            }
            Ok(())
        }
    }

    /// Buffer a store that is *also* being dispatched to a Python memory
    /// callback, for the case where that callback cannot absorb it.
    ///
    /// The callback state's `state.memory` is a `RustMemoryProxy`, so the
    /// `memory_store` / `memory_store_symbolic_value` callbacks are deliberate
    /// no-ops (`state.memory` *is* Rust memory, and re-entering `run()` would
    /// double-borrow), and a callback-only store would be dropped outright:
    /// flareon2015_5's base64 encoder wrote its output through the
    /// symbolic-address path and the found state read back concrete zeros
    /// (angr-5rjbq). Buffering it here gives the value a home in
    /// `rust_memory`.
    pub(super) fn buffer_store_for_rust_memory(&mut self, addr: u64, data_val: &RustBV) {
        if !self.use_rust_memory {
            return;
        }
        if data_val.is_symbolic() {
            // Evict overlapping symbolic shadows first — same reasoning as
            // handle_concrete_store's symbolic branch (angr-vvzf5): without
            // this, two overlapping-but-different-address symbolic stores
            // can coexist and a later overlap load returns hash-order-
            // arbitrary data instead of the most recent store.
            let size_bytes = (data_val.width() / 8) as usize;
            self.evict_overlapping_symbolic_stores(addr, size_bytes);
            self.pending_symbolic_stores.insert(addr, data_val.clone());
        } else {
            let size_bytes = (data_val.width() / 8) as usize;
            self.evict_overlapping_symbolic_stores(addr, size_bytes);
            self.pending_stores.push(addr, bv_to_bytes(data_val));
        }
    }

    /// Buffer-then-dispatch a symbolic value to
    /// `PythonCallbacks::call_memory_store_symbolic_value`.
    ///
    /// **Every** call site of that callback must go through this helper.
    /// The callback is a documented no-op (the callback state's memory is a
    /// `RustMemoryProxy`), so a raw dispatch that skips `buffer_store_for_rust_memory`
    /// lands in neither Rust's pending-store buffers nor Python's shadow
    /// memory and is lost outright. The raw form recurred at the two
    /// `statements_store.rs` paths and four siblings — `handle_storeg`'s two
    /// symbolic-guard branches, `cas_store_symbolic_data`, and
    /// `build_ite_store_from_callbacks` (angr-5rjbq, angr-5mnx3.23). Bundling
    /// the pair here is what stops another site from repeating it.
    ///
    /// Callers that need the pending buffer drained first (so Python cannot
    /// observe a write ordered after this one) still call `flush_stores`
    /// themselves *before* this helper — buffering after the flush keeps the
    /// value live in `pending_symbolic_stores` for load forwarding until the
    /// next flush.
    pub(super) fn store_symbolic_value_buffered(
        &mut self,
        callbacks: &dyn EngineHost,
        addr: u64,
        data_val: &RustBV,
    ) -> Result<(), CbExecutionError> {
        self.buffer_store_for_rust_memory(addr, data_val);
        callbacks
            .memory_store_symbolic_value(addr, data_val)
            .map_err(|e| CbExecutionError::Callback(e.to_string()))
    }

    /// Remove symbolic store entries (pending and flushed) whose byte range
    /// overlaps `[addr, addr + size)`. A concrete store overwriting (part of) a
    /// prior symbolic store must drop the symbolic shadow, otherwise a later
    /// load at the same address would return the stale symbolic value instead
    /// of the concrete bytes (angr-ofyh). Short-circuits when no symbolic
    /// stores are buffered, so the all-concrete hot path pays nothing.
    ///
    /// Overlap is decided by `pending_store::overlap_offsets` — the same
    /// offset-space test `PendingStoreBuffer::push` uses — so a symbolic store
    /// (or a concrete store) that wraps past the top of the guest address
    /// space still evicts its wrapped tail (angr-fs8kb.64). Comparing end
    /// addresses instead, as this did with `saturating_add`, clamped a wrapped
    /// range onto `u64::MAX` and left the stale symbolic shadow in place.
    pub(super) fn evict_overlapping_symbolic_stores(&mut self, addr: u64, size: usize) {
        if self.pending_symbolic_stores.is_empty() && self.all_flushed_symbolic_stores.is_empty() {
            return;
        }
        let overlaps = |s_addr: u64, bv: &RustBV| {
            overlap_offsets(addr, size, s_addr, (bv.width() / 8) as usize).is_some()
        };
        self.pending_symbolic_stores
            .retain(|&s_addr, bv| !overlaps(s_addr, bv));
        self.all_flushed_symbolic_stores
            .retain(|&s_addr, bv| !overlaps(s_addr, bv));
    }

    /// Symbolic-address store path. Clears the prefetch cache, flushes the
    /// pending-store buffer, then dispatches on the 5 ConcretizationResult
    /// shapes returned by `concretize_cached_write`. Single → direct callback;
    /// Multiple/Strided → `dispatch_multi_store`; TooLarge → full symbolic
    /// callback or Unsupported; Failed → `fallback_store_symbolic_full`.
    ///
    /// Invalidates any stale cached IRSB at the concretized target(s) before
    /// dispatching the store (self-modifying-code support) via
    /// `invalidate_code_on_store` — the same dispatcher `try_rust_memory_store`
    /// and `cas_store_symbolic_data` use, so Multiple/Strided/TooLarge/Failed
    /// get the same per-address (or defensive block-cache-clear) treatment as
    /// the Single case, instead of only Single being covered (angr-srk4b).
    pub(super) fn handle_symbolic_store(
        &mut self,
        callbacks: &dyn EngineHost,
        addr_val: &RustBV,
        data_val: &RustBV,
        data_size: usize,
    ) -> Result<(), CbExecutionError> {
        // AVOID_MULTIVALUED_WRITES: silently drop. Do not touch pending
        // stores — the address was never resolved, so no mutation propagates.
        if self.concretizer.should_avoid_multivalued_write(addr_val) {
            return Ok(());
        }
        // Touched addresses are unknown — flush pending stores before Python
        // sees the symbolic write.
        self.flush_stores(callbacks)?;

        let concret_result = self.concretize_cached_write(addr_val);
        // Invalidate before dispatching the store so Python (or the in-Rust
        // ITE path) never observes a write that landed without evicting the
        // stale lifted block first — mirrors the Single-arm ordering below.
        self.invalidate_code_on_store(&concret_result, data_size);
        match &*concret_result {
            ConcretizationResult::Single(addr_concrete) => {
                let addr_concrete = *addr_concrete;
                if data_val.is_symbolic() && callbacks.has_memory_store_symbolic_value() {
                    self.store_symbolic_value_buffered(callbacks, addr_concrete, data_val)?;
                } else {
                    self.buffer_store_for_rust_memory(addr_concrete, data_val);
                    reject_symbolic_byte_store(
                        data_val,
                        addr_concrete,
                        "symbolic store (single concretization)",
                    )?;
                    let data_bytes = bv_to_bytes(data_val);
                    callbacks
                        .memory_store(addr_concrete, &data_bytes)
                        .map_err(|e| CbExecutionError::Callback(e.to_string()))?;
                }
                Ok(())
            }
            ConcretizationResult::Multiple(addrs) => {
                self.dispatch_multi_store(callbacks, addrs, addr_val, data_val)
            }
            ConcretizationResult::Strided {
                base,
                stride,
                count,
            } => {
                let addrs = strided_addrs(*base, *stride, *count);
                self.dispatch_multi_store(callbacks, &addrs, addr_val, data_val)
            }
            ConcretizationResult::TooLarge { min, max, .. } => {
                let (min, max) = (*min, *max);
                // Range too large to enumerate — delegate to Python's memory
                // model (which has access to angr's address concretization
                // strategies) via the full symbolic callback.
                if callbacks.has_memory_store_symbolic_full() {
                    callbacks
                        .memory_store_symbolic_full(addr_val, data_val)
                        .map_err(|e| {
                            CbExecutionError::Callback(format!(
                                "symbolic store full callback failed at 0x{min:x}-0x{max:x}: {e}"
                            ))
                        })?;
                    Ok(())
                } else {
                    Err(CbExecutionError::Unsupported(format!(
                        "symbolic store with too-large address range 0x{min:x}-0x{max:x}: \
                         no memory_store_symbolic_full callback"
                    )))
                }
            }
            ConcretizationResult::Failed(reason) => {
                // Concretization failed entirely. Try the full symbolic
                // store callback so Python's memory model can still resolve
                // the address; only error out if the callback isn't wired up.
                let descr = format!("concretize failed: {reason}");
                self.fallback_store_symbolic_full(callbacks, addr_val, data_val, "store", &descr)
            }
        }
    }

    /// Shared dispatch for Multiple/Strided concretization results: build an
    /// in-Rust ITE chain when the address count is within [`MAX_ITE_ADDRS`] and
    /// the symbolic-value callback is available, otherwise hand the full address
    /// list to Python. The load-side counterpart is `dispatch_multi_load`.
    pub(super) fn dispatch_multi_store(
        &mut self,
        callbacks: &dyn EngineHost,
        addrs: &[u64],
        addr_val: &RustBV,
        data_val: &RustBV,
    ) -> Result<(), CbExecutionError> {
        if addrs.len() <= MAX_ITE_ADDRS && callbacks.has_memory_store_symbolic_value() {
            self.build_ite_store_from_callbacks(callbacks, addrs, addr_val, data_val, None)
        } else {
            callbacks
                .memory_store_symbolic(addrs, data_val, addr_val)
                .map_err(|e| CbExecutionError::Callback(e.to_string()))
        }
    }
}
